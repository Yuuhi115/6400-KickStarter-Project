# -*- coding: utf-8 -*-
"""S0: replicate the group pilot study on OUR full model-ready data, then compare.

Replaces the pilot's 8 口径 choices with ours:
  goal / log1p(goal)        -> m_log_goal_usd
  campaign_duration_days    -> m_dur_days_exact
  project_name_length etc.  -> m_name_len / m_blurb_len / m_blurb_words
  creator_previous_*        -> creator_rank-1 / creator_prev_success_rate (full-sample, point-in-time)
  category_name / country   -> subcategory_key / country_final
  year cutoffs              -> --split ours => m_split (train<=2022 / test>=2023)
                               --split pilot => train<=2019 / val 2020-2022 / test >2022

usage:
  python replicate_pilot.py                     # full data, pilot split (replication)
  python replicate_pilot.py --split ours        # full data, our protocol
  python replicate_pilot.py --data usd --split pilot
  python replicate_pilot.py --no-prep-days      # ablation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = r"E:\Workspace\IS6400\group\IS6400_Kickstarter_交付包_完整"
DATA = {"full": os.path.join(ROOT, "08_建模就绪数据", "ks_model_ready_v1.parquet"),
        "usd": os.path.join(ROOT, "08_建模就绪数据", "ks_model_ready_usdonly.parquet")}
OUTDIR = os.path.join(ROOT, "11_结果与报告")
RANDOM_STATE = 42

NEEDED = ["succ", "m_is_modelable", "launched_at", "created_at",
          "m_log_goal_usd", "m_goal_usd", "m_dur_days_exact", "yr", "mon", "dow",
          "m_name_len", "m_blurb_len", "m_blurb_words", "name",
          "category_parent_name", "subcategory_key", "country_final",
          "location_type", "currency", "creator_rank", "creator_prev_success_rate"]


def parse_kickstarter_datetime(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").str.strip()
    numeric = pd.to_numeric(as_text, errors="coerce")
    mask = numeric.notna()
    parsed_text = pd.to_datetime(as_text.where(~mask), errors="coerce", utc=True)
    parsed_num = pd.to_datetime(numeric.where(mask), unit="s", errors="coerce", utc=True)
    out = parsed_text.fillna(parsed_num)
    return out.where(out.dt.year.between(2005, 2030))


def build_frame(path: str, use_prep_days: bool) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=NEEDED)
    df = df[df["m_is_modelable"].fillna(False)].copy()
    df["succ"] = pd.to_numeric(df["succ"], errors="coerce")
    df = df[df["succ"].notna()].copy()

    if pd.api.types.is_numeric_dtype(df["launched_at"]):
        launch = pd.to_datetime(pd.to_numeric(df["launched_at"], errors="coerce"),
                                unit="s", utc=True, errors="coerce")
    else:
        launch = parse_kickstarter_datetime(df["launched_at"])
    created = parse_kickstarter_datetime(df["created_at"])
    df["_launch_dt"] = launch

    df["prep_days"] = (launch - created).dt.total_seconds() / 86400.0
    df["launch_quarter"] = ((pd.to_numeric(df["mon"], errors="coerce") - 1) // 3 + 1)
    df["name_word_count"] = df["name"].astype("string").fillna("").str.split().str.len()

    # creator history: our creator_rank = prior+1  ->  previous_count = rank-1
    df["creator_previous_count"] = (pd.to_numeric(df["creator_rank"], errors="coerce") - 1).clip(lower=0)

    for c in ["category_parent_name", "subcategory_key", "country_final", "location_type", "currency"]:
        df[c] = df[c].astype("string").fillna("Unknown").astype(str)

    num = ["m_log_goal_usd", "m_dur_days_exact", "launch_quarter", "dow",
           "m_name_len", "m_blurb_len", "m_blurb_words", "name_word_count",
           "creator_previous_count", "creator_prev_success_rate"]
    if use_prep_days:
        num.append("prep_days")
    df["yr"] = pd.to_numeric(df["yr"], errors="coerce")
    num.append("yr")
    df["mon"] = pd.to_numeric(df["mon"], errors="coerce")
    num.append("mon")
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df.attrs["numeric"] = num
    df.attrs["categorical"] = ["category_parent_name", "subcategory_key", "country_final",
                               "location_type", "currency"]
    return df


def make_splits(df: pd.DataFrame, mode: str):
    yr = df["yr"]
    if mode == "pilot":
        train = df[yr <= 2019]
        val = df[(yr >= 2020) & (yr <= 2022)]
        test = df[yr > 2022]
    else:  # ours: train <=2022, test >=2023; validation = 2021-2022 slice of train-era
        train = df[yr <= 2022]
        val = df[(yr >= 2021) & (yr <= 2022)]
        train = df[yr <= 2020]
        test = df[yr >= 2023]
    return train, val, test


def preprocessor(numeric, categorical) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                  ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                       min_frequency=20, sparse_output=False))]),
         categorical),
    ], remainder="drop")


def models(numeric, categorical, rf_trees: int = 250):
    out = {
        "Logistic Regression": Pipeline([("pre", preprocessor(numeric, categorical)),
                                         ("model", LogisticRegression(max_iter=3000,
                                                                      solver="liblinear",
                                                                      random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("pre", preprocessor(numeric, categorical)),
                                       ("model", HistGradientBoostingClassifier(learning_rate=0.08,
                                                                                max_iter=250, max_leaf_nodes=31,
                                                                                l2_regularization=1.0,
                                                                                early_stopping=True,
                                                                                validation_fraction=0.10,
                                                                                n_iter_no_change=20,
                                                                                random_state=RANDOM_STATE))]),
    }
    if rf_trees and rf_trees > 0:
        out["Random Forest"] = Pipeline([("pre", preprocessor(numeric, categorical)),
                                         ("model", RandomForestClassifier(n_estimators=rf_trees,
                                                                          min_samples_leaf=5, n_jobs=1,
                                                                          random_state=RANDOM_STATE))])
    return out


def pick_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "Brier_Score": brier_score_loss(y, p), "Threshold": thr, "n": int(len(y)),
            "base_rate": float(np.mean(y))}


def top_decile(y, p):
    f = pd.DataFrame({"y": np.asarray(y), "p": p}).sort_values("p", ascending=False)
    k = max(1, int(math.ceil(len(f) * 0.10)))
    overall, top = float(f["y"].mean()), float(f.iloc[:k]["y"].mean())
    return {"top_n": int(k), "overall": overall, "top_rate": top,
            "lift": top / overall if overall else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["full", "usd"], default="full")
    ap.add_argument("--split", choices=["pilot", "ours"], default="pilot")
    ap.add_argument("--no-prep-days", action="store_true")
    ap.add_argument("--rf-trees", type=int, default=250, help="0 = skip Random Forest (fast)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or f"{args.data}_{args.split}" + ("_noprep" if args.no_prep_days else "")
    print(f"[{tag}] loading {args.data} ...", flush=True)
    df = build_frame(DATA[args.data], use_prep_days=not args.no_prep_days)
    numeric, categorical = df.attrs["numeric"], df.attrs["categorical"]
    train, val, test = make_splits(df, args.split)
    print(f"[{tag}] rows modelable={len(df):,} | train={len(train):,} val={len(val):,} test={len(test):,}",
          flush=True)
    print(f"[{tag}] base rates: train={train['succ'].mean():.4f} test={test['succ'].mean():.4f}", flush=True)

    feats = numeric + categorical
    rows, oof = [], {}
    # baseline
    base = DummyClassifier(strategy="prior", random_state=RANDOM_STATE).fit(train[feats], train["succ"])
    pval = base.predict_proba(val[feats])[:, 1]
    thr = pick_threshold(val["succ"].values, pval)
    ptest = base.predict_proba(test[feats])[:, 1]
    m = evaluate(test["succ"].values, ptest, 0.5)
    m.update({"Model": "Baseline", "Dataset": "Test", "tag": tag})
    rows.append(m)

    for name, pipe in models(numeric, categorical, args.rf_trees).items():
        print(f"[{tag}] fitting {name} ...", flush=True)
        pipe.fit(train[feats], train["succ"])
        pv = pipe.predict_proba(val[feats])[:, 1]
        t = pick_threshold(val["succ"].values, pv)
        pt = pipe.predict_proba(test[feats])[:, 1]
        m = evaluate(test["succ"].values, pt, t)
        m.update({"Model": name, "Dataset": "Test", "tag": tag,
                  "val_threshold": t, "val_positive_rate": float((pv >= t).mean())})
        rows.append(m)
        oof[name] = pt
        print(f"[{tag}]   {name}: AUC={m['ROC_AUC']:.4f} PR={m['PR_AUC']:.4f} "
              f"F1={m['F1']:.4f} Brier={m['Brier_Score']:.4f} thr={t:.2f}", flush=True)

    metrics = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, f"s0_metrics_{tag}.csv")
    metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # lift + calibration for the best model
    best = metrics[metrics["Model"] != "Baseline"].sort_values("ROC_AUC", ascending=False).iloc[0]
    lift = top_decile(test["succ"].values, oof[best["Model"]])
    frac, mean_pred = calibration_curve(test["succ"].values, oof[best["Model"]],
                                        n_bins=10, strategy="quantile")
    cal = pd.DataFrame({"mean_predicted": mean_pred, "observed": frac})
    cal.to_csv(os.path.join(OUTDIR, f"s0_calibration_{tag}.csv"), index=False, encoding="utf-8-sig")

    # 3-fold expanding window stability (best model only, threshold 0.5)
    ordered = df.sort_values("_launch_dt").reset_index(drop=True)
    stab = []
    for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=3).split(ordered), start=1):
        a, b = ordered.iloc[tr], ordered.iloc[te]
        pipe = models(numeric, categorical, args.rf_trees)[best["Model"]]
        pipe.fit(a[feats], a["succ"])
        p = pipe.predict_proba(b[feats])[:, 1]
        r = evaluate(b["succ"].values, p, 0.5)
        r.update({"fold": i, "train_rows": len(a), "test_rows": len(b),
                  "test_start": str(b["_launch_dt"].min()), "test_end": str(b["_launch_dt"].max())})
        stab.append(r)
    stab_df = pd.DataFrame(stab)
    stab_df.to_csv(os.path.join(OUTDIR, f"s0_stability_{tag}.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "tag": tag, "data": args.data, "split": args.split,
        "use_prep_days": not args.no_prep_days,
        "rows": {"modelable": int(len(df)), "train": int(len(train)),
                 "val": int(len(val)), "test": int(len(test))},
        "base_rates": {"train": float(train["succ"].mean()), "test": float(test["succ"].mean())},
        "metrics": metrics.to_dict(orient="records"),
        "best_model": str(best["Model"]),
        "best_auc": float(best["ROC_AUC"]),
        "top_decile": lift,
        "stability_auc": {f"fold{k}": float(v) for k, v in
                          zip(stab_df["fold"], stab_df["ROC_AUC"])},
    }
    with open(os.path.join(OUTDIR, f"s0_summary_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n=== metrics (Test) ===")
    print(metrics[["Model", "ROC_AUC", "PR_AUC", "F1", "Brier_Score", "Accuracy", "Threshold"]]
          .to_string(index=False))
    print(f"\nbest={best['Model']} AUC={best['ROC_AUC']:.4f} | top-decile lift="
          f"{lift['lift']:.4f} (top={lift['top_rate']:.4f} vs overall={lift['overall']:.4f})")
    print("stability AUC:", summary["stability_auc"])
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
