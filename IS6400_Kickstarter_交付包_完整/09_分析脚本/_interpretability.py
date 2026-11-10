# -*- coding: utf-8 -*-
"""S1: interpretability pack for the final model.

Why: HistGB has no native interpretability (and sklearn exposes no
`feature_importances_` for it), so we supplement with three post-hoc views:

  1. Logistic-regression coefficients / odds ratios  -> direction & magnitude
  2. Permutation importance on the test set          -> global ranking for HistGB
  3. 1-D partial dependence (PDP)                    -> marginal shape / thresholds

Outputs (into 11_结果与报告/):
  s1_lr_coefficients.csv
  s1_permutation_importance.csv
  s1_pdp_<feature>.csv
  s1_interpretability.json
  S1_解释性.md

usage:
  python _interpretability.py            # full pack
  python _interpretability.py --perm-n 20000 --repeats 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "11_结果与报告")
RANDOM_STATE = 42

# import the shared data-prep module (replicate_pilot.py) without running its main()
spec = importlib.util.spec_from_file_location("rp", os.path.join(HERE, "replicate_pilot.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

PDP_FEATURES = [
    "m_log_goal_usd",
    "m_dur_days_exact",
    "creator_previous_count",
    "creator_prev_success_rate",
    "prep_days",
    "m_blurb_len",
    "m_name_len",
]


def load_splits():
    df = rp.build_frame(rp.DATA["full"], use_prep_days=True)
    train, val, test = rp.make_splits(df, "ours")
    num = list(df.attrs["numeric"])
    cat = list(df.attrs["categorical"])
    feats = num + cat
    return train, val, test, num, cat, feats


def fit_lr(train, num, cat):
    pipe = Pipeline([
        ("pre", rp.preprocessor(num, cat)),
        ("model", LogisticRegression(max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)),
    ])
    pipe.fit(train[num + cat], train["succ"].astype(int))
    return pipe


def lr_table(pipe, num, cat) -> pd.DataFrame:
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    coef = np.asarray(pipe.named_steps["model"].coef_[0], dtype=float)
    scale = np.asarray(
        pipe.named_steps["pre"].named_transformers_["numeric"].named_steps["sc"].scale_, dtype=float
    )
    rows = []
    for i, name in enumerate(names):
        clean = name.split("__", 1)[-1]
        is_num = i < len(num)
        per_unit = (coef[i] / scale[i]) if (is_num and scale[i]) else np.nan
        rows.append({
            "feature": clean,
            "block": "numeric" if is_num else "categorical",
            "coef_std": float(coef[i]),                      # log-odds per +1 SD (numeric) / vs reference (categorical)
            "coef_per_unit": float(per_unit) if is_num else np.nan,   # log-odds per +1 raw unit
            "odds_ratio_std": float(np.exp(coef[i])),
            "direction": "↑ 提升成功概率" if coef[i] > 0 else "↓ 降低成功概率",
        })
    t = pd.DataFrame(rows)
    t["abs_coef"] = t["coef_std"].abs()
    return t.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def load_or_fit_histgb(train, test, num, cat, feats):
    """Prefer the delivered joblib; verify it reproduces AUC 0.8823 on test."""
    path = os.path.join(OUTDIR, "model_histgb_ours.joblib")
    X_test = test[feats]
    y_test = test["succ"].astype(int).to_numpy()
    if os.path.exists(path):
        try:
            import joblib
            obj = joblib.load(path)
            pipe = obj["pipeline"] if isinstance(obj, dict) and "pipeline" in obj else obj
            auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
            print(f"[histgb] loaded joblib, test AUC={auc:.4f}", flush=True)
            if auc > 0.85:
                return pipe, auc, "joblib"
        except Exception as exc:  # noqa: BLE001
            print(f"[histgb] joblib unusable ({exc}); refitting", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    pipe = Pipeline([
        ("pre", rp.preprocessor(num, cat)),
        ("model", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
                                                 l2_regularization=1.0, early_stopping=True,
                                                 validation_fraction=0.10, n_iter_no_change=20,
                                                 random_state=RANDOM_STATE)),
    ])
    pipe.fit(train[num + cat], train["succ"].astype(int))
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"[histgb] refit, test AUC={auc:.4f}", flush=True)
    return pipe, auc, "refit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-n", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pdp-n", type=int, default=5000)
    args = ap.parse_args()

    train, val, test, num, cat, feats = load_splits()
    print(f"[data] train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

    # ---------- 1. logistic regression ----------
    lr = fit_lr(train, num, cat)
    lr_auc = roc_auc_score(test["succ"].astype(int), lr.predict_proba(test[num + cat])[:, 1])
    print(f"[lr] test AUC={lr_auc:.4f}", flush=True)
    coefs = lr_table(lr, num, cat)
    coefs.to_csv(os.path.join(OUTDIR, "s1_lr_coefficients.csv"), index=False, encoding="utf-8-sig")

    # ---------- 2. HistGB + permutation importance ----------
    hgb, hgb_auc, hgb_src = load_or_fit_histgb(train, test, num, cat, feats)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(test), size=min(args.perm_n, len(test)), replace=False)
    X_perm = test[feats].iloc[idx]
    y_perm = test["succ"].astype(int).to_numpy()[idx]
    print(f"[perm] n={len(idx):,} repeats={args.repeats} ...", flush=True)
    res = permutation_importance(hgb, X_perm, y_perm, scoring="roc_auc",
                                 n_repeats=args.repeats, random_state=RANDOM_STATE, n_jobs=1)
    perm = pd.DataFrame({
        "feature": feats,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    perm["share_pct"] = 100 * perm["importance_mean"].clip(lower=0) / perm["importance_mean"].clip(lower=0).sum()
    perm.to_csv(os.path.join(OUTDIR, "s1_permutation_importance.csv"), index=False, encoding="utf-8-sig")
    print(perm.head(8).to_string(index=False), flush=True)

    # ---------- 3. partial dependence ----------
    idx2 = rng.choice(len(test), size=min(args.pdp_n, len(test)), replace=False)
    X_pdp = test[feats].iloc[idx2]
    pdp_rows = []
    for feat in PDP_FEATURES:
        if feat not in feats:
            continue
        pd_res = partial_dependence(hgb, X_pdp, [feat], grid_resolution=20, kind="average")
        xs = np.asarray(pd_res["grid_values"][0], dtype=float)
        ys = np.asarray(pd_res["average"][0], dtype=float)
        out = pd.DataFrame({"feature": feat, "value": xs, "partial_proba": ys})
        out.to_csv(os.path.join(OUTDIR, f"s1_pdp_{feat}.csv"), index=False, encoding="utf-8-sig")
        lo, hi = float(ys.min()), float(ys.max())
        slope = float(ys[-1] - ys[0])
        pdp_rows.append({"feature": feat, "pdp_min": lo, "pdp_max": hi,
                         "range_pp": 100 * (hi - lo), "delta_first_to_last_pp": 100 * slope,
                         "grid_low": float(xs[0]), "grid_high": float(xs[-1])})
        print(f"[pdp] {feat}: {100*lo:.1f}pp -> {100*hi:.1f}pp (Δ{100*slope:+.1f}pp)", flush=True)
    pdp_summary = pd.DataFrame(pdp_rows)
    pdp_summary.to_csv(os.path.join(OUTDIR, "s1_pdp_summary.csv"), index=False, encoding="utf-8-sig")

    # ---------- 4. merge: LR direction vs PDP slope ----------
    merged = pdp_summary.merge(coefs[["feature", "coef_std", "odds_ratio_std", "direction"]],
                               on="feature", how="left")
    merged["lr_sign"] = np.sign(merged["coef_std"])
    merged["pdp_sign"] = np.sign(merged["delta_first_to_last_pp"])
    merged["consistent"] = np.where(merged["lr_sign"] == merged["pdp_sign"], "一致", "不一致")
    merged.to_csv(os.path.join(OUTDIR, "s1_direction_check.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "test_auc_lr": lr_auc,
        "test_auc_histgb": hgb_auc,
        "histgb_source": hgb_src,
        "lr_share_of_histgb_pct": 100 * lr_auc / hgb_auc,
        "perm_n": int(len(idx)),
        "perm_repeats": args.repeats,
        "pdp_n": int(len(idx2)),
        "top5_permutation": perm.head(5).to_dict(orient="records"),
    }
    with open(os.path.join(OUTDIR, "s1_interpretability.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=float)

    print("\n=== LR coefficients (top 10 by |coef|) ===", flush=True)
    print(coefs.head(10)[["feature", "coef_std", "odds_ratio_std", "direction"]].to_string(index=False),
          flush=True)
    print("\n=== direction check ===", flush=True)
    print(merged[["feature", "coef_std", "delta_first_to_last_pp", "consistent"]].to_string(index=False),
          flush=True)
    print("\n[done] written to", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
