"""Stage 3 - baselines + four model families + rolling-origin validation.

Protocol
--------
Outer:  train = yr <= 2022, test = yr >= 2023 (never touched before the end).
Inner:  6 expanding rolling-origin folds, validation years 2017..2022.
         fold k: train = yr <= vy-1, val = yr == vy.
Everything (category vocabularies, medians, one-hot encoder, imputers,
calibrators) is fitted inside the fold's training window.
Text scores come pre-computed out-of-fold by year from ``text_model.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, log_loss, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src import features as F

MODELS = ["B0_majority", "B1_goal_only", "B2_cell_rate",
          "LR", "TREE", "RF", "HGB", "HGB_text", "BLEND"]


def metrics(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "acc": float((pred == y).mean()),
        "f1": f1_score(y, pred, zero_division=0),
    }


# ---------------------------------------------------------------- designs
class LRDesign:
    """One-hot categoricals + median-imputed, standardised numerics."""

    def __init__(self, spec):
        self.spec = spec

    def fit(self, df: pd.DataFrame) -> "LRDesign":
        X, mask, names = F.transform(df, self.spec)
        F.check_no_leakage(names)
        self.cat = [n for n, m in zip(names, mask) if m]
        self.num = [n for n, m in zip(names, mask) if not m]
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                 dtype=np.float32).fit(X[self.cat])
        numv = X[self.num].to_numpy(dtype=np.float64)
        self.med = np.nanmedian(numv, axis=0)
        numv = np.where(np.isnan(numv), self.med, numv)
        self.scaler = StandardScaler().fit(numv)
        return self

    def transform(self, df: pd.DataFrame):
        X, _, _ = F.transform(df, self.spec)
        A = self.enc.transform(X[self.cat])
        numv = X[self.num].to_numpy(dtype=np.float64)
        numv = np.where(np.isnan(numv), self.med, numv)
        B = sp.csr_matrix(self.scaler.transform(numv).astype(np.float32))
        return sp.hstack([A, B]).tocsr()


def dense_design(df, spec, text=None):
    X, mask, names = F.transform(df, spec, text_score=text)
    F.check_no_leakage(names)
    return X.to_numpy(np.float32), mask, names


# ---------------------------------------------------------------- cell-rate baseline
def _keys(df: pd.DataFrame, level: int) -> pd.Series:
    keys = df["category_key"].astype(str)
    if level >= 2:
        keys = keys + "|" + df["goal_bin"].astype(str)
    if level >= 3:
        keys = keys + "|" + df["dur4_bin"].astype(str)
    return keys


def fit_cell_rate(train_df: pd.DataFrame) -> dict:
    df = C.add_bins(train_df)
    maps = {}
    for level in (3, 2, 1):
        k = _keys(df, level)
        g = df.groupby(k, observed=True)["succ"].agg(["mean", "size"])
        maps[level] = g
    maps["prior"] = float(train_df["succ"].mean())
    return maps


def apply_cell_rate(df: pd.DataFrame, maps: dict, min_n: int = 200) -> np.ndarray:
    df = C.add_bins(df)
    n = len(df)
    out = np.full(n, maps["prior"], dtype=float)
    done = np.zeros(n, dtype=bool)
    for level in (3, 2, 1):
        k = _keys(df, level)
        g = maps[level]
        rate = k.map(g["mean"]).to_numpy(dtype=float)
        cnt = k.map(g["size"]).fillna(0).to_numpy(dtype=float)
        ok = (~done) & (cnt >= min_n) & np.isfinite(rate)
        out[ok] = rate[ok]
        done |= ok
    return out


# ---------------------------------------------------------------- fold routine
def fit_predict_fold(df, tr_mask, va_mask, spec, text_all) -> dict:
    tr, va = df[tr_mask], df[va_mask]
    y_tr = tr["succ"].to_numpy().astype(int)
    y_va = va["succ"].to_numpy().astype(int)
    preds = {}

    preds["B0_majority"] = np.full(len(va), y_tr.mean())

    # B1: goal only
    sc1 = StandardScaler().fit(tr[["log_goal"]])
    lr1 = LogisticRegression(max_iter=1000).fit(sc1.transform(tr[["log_goal"]]), y_tr)
    preds["B1_goal_only"] = lr1.predict_proba(sc1.transform(va[["log_goal"]]))[:, 1]

    # B2: historical cell rate
    maps = fit_cell_rate(tr)
    preds["B2_cell_rate"] = apply_cell_rate(va, maps)

    # LR on one-hot
    lr_design = LRDesign(spec).fit(tr)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    lr.fit(lr_design.transform(tr), y_tr)
    preds["LR"] = lr.predict_proba(lr_design.transform(va))[:, 1]

    # tree / RF / HGB on dense code matrix
    Xtr, mask, names = dense_design(tr, spec)
    Xva, _, _ = dense_design(va, spec)
    tree = DecisionTreeClassifier(**C.TREE_CLS).fit(Xtr, y_tr)
    preds["TREE"] = tree.predict_proba(Xva)[:, 1]

    rf = RandomForestClassifier(**C.RF_CLS).fit(Xtr, y_tr)
    preds["RF"] = rf.predict_proba(Xva)[:, 1]

    hgb = HistGradientBoostingClassifier(categorical_features=mask,
                                         **C.HGB_CLS).fit(Xtr, y_tr)
    p_hgb = hgb.predict_proba(Xva)[:, 1]
    preds["HGB"] = p_hgb

    # HGB + text score (out-of-fold text, NaN where no text history)
    ts_tr = text_all[tr_mask]
    ts_va = text_all[va_mask]
    Xtr2, mask2, _ = dense_design(tr, spec, text=ts_tr)
    Xva2, _, _ = dense_design(va, spec, text=ts_va)
    hgb_t = HistGradientBoostingClassifier(categorical_features=mask2,
                                           **C.HGB_CLS).fit(Xtr2, y_tr)
    p_t = hgb_t.predict_proba(Xva2)[:, 1]
    preds["HGB_text"] = p_t

    # simple late-fusion control
    blend = np.where(np.isfinite(ts_va), 0.5 * p_hgb + 0.5 * p_t, p_hgb)
    preds["BLEND"] = blend
    return preds, y_va


def main() -> None:
    df = C.load_v2().reset_index(drop=True)
    text_scores = pd.read_csv(C.DATA_DIR / "text_scores.csv.gz")
    assert (text_scores["id"].to_numpy() == df["id"].to_numpy()).all(), "text/id misaligned"
    text_all = text_scores["text_score"].to_numpy(dtype=float)
    n = len(df)
    C.log(f"models: n={n:,} text_scored={np.isfinite(text_all).sum():,}")

    oof = pd.DataFrame({m: np.full(n, np.nan) for m in MODELS})
    rows = []
    for vy in C.VAL_YEARS:
        tr_mask = (df["yr"] <= vy - 1).to_numpy()
        va_mask = (df["yr"] == vy).to_numpy()
        spec = F.build_spec(df[tr_mask])
        C.log(f"fold val={vy}: train={tr_mask.sum():,} val={va_mask.sum():,}")
        preds, y_va = fit_predict_fold(df, tr_mask, va_mask, spec, text_all)
        for m, p in preds.items():
            row = {"model": m, "fold": vy, **metrics(y_va, p)}
            rows.append(row)
            C.log(f"  {m:<14} auc={row['auc']:.4f} brier={row['brier']:.4f}")
            oof.loc[va_mask, m] = p

    rolling = pd.DataFrame(rows)
    rolling.to_csv(C.TABLE_DIR / "metrics_rolling.csv", index=False)
    summ = (rolling.groupby("model")["auc"]
            .agg(["mean", "std", "min", "max"]).reset_index()
            .sort_values("mean", ascending=False))
    summ.columns = ["model", "auc_mean", "auc_sd", "auc_min", "auc_max"]
    summ = summ.merge(rolling.groupby("model")["brier"].mean().rename("brier_mean").reset_index(),
                      on="model")
    summ.to_csv(C.TABLE_DIR / "metrics_rolling_summary.csv", index=False)
    C.log("\nrolling summary:\n" + summ.to_string(index=False))

    # ---------------- refit on the full training window and score the test window
    train_mask = (df["yr"] <= C.TRAIN_MAX_YEAR).to_numpy()
    test_mask = (df["yr"] >= C.TEST_MIN_YEAR).to_numpy()
    tr, te = df[train_mask], df[test_mask]
    y_tr = tr["succ"].to_numpy().astype(int)
    y_te = te["succ"].to_numpy().astype(int)
    spec = F.build_spec(tr)

    test_pred = {}
    test_pred["B0_majority"] = np.full(len(te), y_tr.mean())
    lr1 = LogisticRegression(max_iter=1000)
    sc1 = StandardScaler().fit(tr[["log_goal"]])
    lr1.fit(sc1.transform(tr[["log_goal"]]), y_tr)
    test_pred["B1_goal_only"] = lr1.predict_proba(sc1.transform(te[["log_goal"]]))[:, 1]
    test_pred["B2_cell_rate"] = apply_cell_rate(te, fit_cell_rate(tr))

    lr_design = LRDesign(spec).fit(tr)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(
        lr_design.transform(tr), y_tr)
    test_pred["LR"] = lr.predict_proba(lr_design.transform(te))[:, 1]

    Xtr, mask, names = dense_design(tr, spec)
    Xte, _, _ = dense_design(te, spec)
    tree = DecisionTreeClassifier(**C.TREE_CLS).fit(Xtr, y_tr)
    test_pred["TREE"] = tree.predict_proba(Xte)[:, 1]
    rf = RandomForestClassifier(**C.RF_CLS).fit(Xtr, y_tr)
    test_pred["RF"] = rf.predict_proba(Xte)[:, 1]
    hgb = HistGradientBoostingClassifier(categorical_features=mask, **C.HGB_CLS).fit(Xtr, y_tr)
    test_pred["HGB"] = hgb.predict_proba(Xte)[:, 1]

    ts_tr = text_all[train_mask]
    ts_te = text_all[test_mask]
    Xtr2, mask2, _ = dense_design(tr, spec, text=ts_tr)
    Xte2, _, _ = dense_design(te, spec, text=ts_te)
    hgb_t = HistGradientBoostingClassifier(categorical_features=mask2,
                                           **C.HGB_CLS).fit(Xtr2, y_tr)
    test_pred["HGB_text"] = hgb_t.predict_proba(Xte2)[:, 1]
    test_pred["BLEND"] = np.where(np.isfinite(ts_te),
                                  0.5 * test_pred["HGB"] + 0.5 * test_pred["HGB_text"],
                                  test_pred["HGB"])

    test_metrics = []
    for m, p in test_pred.items():
        test_metrics.append({"model": m, "calibrated": False, **metrics(y_te, p)})
    test_tb = pd.DataFrame(test_metrics)

    # ---------------- select by rolling AUC, then isotonic-calibrate on OOF preds
    best = summ.iloc[0]["model"]
    C.log(f"selected model by rolling mean AUC: {best}")
    oof_best = oof.loc[df["yr"].between(C.VAL_YEARS[0], C.VAL_YEARS[-1]), best].to_numpy()
    y_oof = df.loc[df["yr"].between(C.VAL_YEARS[0], C.VAL_YEARS[-1]), "succ"].to_numpy()
    ok = np.isfinite(oof_best)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_best[ok], y_oof[ok])
    cal = iso.predict(test_pred[best])
    test_tb = pd.concat([test_tb, pd.DataFrame([{
        "model": f"{best}+isotonic", "calibrated": True, **metrics(y_te, cal)}])],
        ignore_index=True)
    test_tb.to_csv(C.TABLE_DIR / "metrics_test.csv", index=False)
    C.log("\ntest metrics:\n" + test_tb.to_string(index=False))

    # ---------------- persist artefacts
    oof_out = oof.copy()
    oof_out.insert(0, "succ", df["succ"])
    oof_out.insert(0, "yr", df["yr"])
    oof_out.insert(0, "id", df["id"])
    oof_out.to_csv(C.OUT / "models" / "oof_preds.csv.gz", index=False, compression="gzip")
    te_out = pd.DataFrame(test_pred)
    te_out.insert(0, "succ", y_te)
    te_out.insert(0, "yr", te["yr"].to_numpy())
    te_out.insert(0, "id", te["id"].to_numpy())
    te_out["calibrated"] = cal
    te_out.to_csv(C.OUT / "models" / "test_preds.csv.gz", index=False, compression="gzip")
    joblib.dump({"iso": iso, "selected": best}, C.MODEL_DIR / "calibration.joblib")
    C.dump_json(C.TABLE_DIR / "model_selection.json",
                {"selected": best, "rolling_summary": summ.to_dict("records")})
    C.log("stage models done")


if __name__ == "__main__":
    main()
