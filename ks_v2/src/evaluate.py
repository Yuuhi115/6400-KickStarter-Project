"""Stage 5 - evaluation: discrimination, calibration, stability, lift, deltas.

All numbers in this file are computed on the out-of-time window (yr >= 2023)
unless explicitly marked as OOF (rolling-origin validation predictions).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
RNG = np.random.default_rng(C.SEED)


def ece_score(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return tot / len(y)


def boot_auc(y, p, n_boot: int = 1000) -> tuple[float, float, float]:
    auc = roc_auc_score(y, p)
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            vals[i] = np.nan
            continue
        vals[i] = roc_auc_score(yy, p[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return auc, lo, hi


def boot_delta_auc(y, p_a, p_b, n_boot: int = 500) -> tuple[float, float, float]:
    """AUC(p_a) - AUC(p_b) with a paired bootstrap interval."""
    d0 = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            vals[i] = np.nan
            continue
        vals[i] = roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return d0, lo, hi


def main() -> None:
    df = C.load_v2().reset_index(drop=True)
    te = pd.read_csv(C.OUT / "models" / "test_preds.csv.gz")
    oof = pd.read_csv(C.OUT / "models" / "oof_preds.csv.gz")
    sel = json.loads((C.TABLE_DIR / "model_selection.json").read_text(encoding="utf-8"))["selected"]
    C.log(f"evaluate: selected model = {sel}, test n={len(te):,}")

    y = te["succ"].to_numpy().astype(int)
    p = te[sel].to_numpy(dtype=float)
    p_cal = te["calibrated"].to_numpy(dtype=float)

    # ---------------- 1. calibration
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    for preds, label, color in [(p, "raw", "#1f4e79"), (p_cal, "isotonic-calibrated", "#c00000")]:
        frac_pos, mean_pred = calibration_curve(y, preds, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", ms=3.5, lw=1.2, color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed success rate")
    ax.set_title(f"Reliability diagram, out-of-time (n={len(te):,})")
    ax.legend(fontsize=7)
    C.savefig(fig, "fig_calibration.png")

    cal_tb = pd.DataFrame([
        {"model": sel, "brier": brier_score_loss(y, p), "ece": ece_score(y, p)},
        {"model": f"{sel}+isotonic", "brier": brier_score_loss(y, p_cal),
         "ece": ece_score(y, p_cal)},
    ])
    cal_tb.to_csv(C.TABLE_DIR / "calibration_summary.csv", index=False)
    C.log("\n" + cal_tb.to_string(index=False))

    # prior-correction sensitivity (oracle prior shift, reported only as a check)
    pi_tr = float((df.loc[df.yr <= C.TRAIN_MAX_YEAR, "succ"]).mean())
    pi_te = float(y.mean())
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    shift = np.log(pi_te / (1 - pi_te)) - np.log(pi_tr / (1 - pi_tr))
    p_prior = 1 / (1 + np.exp(-(logit + shift)))
    prior_tb = pd.DataFrame([{
        "train_rate": pi_tr, "test_rate": pi_te, "logit_shift": shift,
        "brier_raw": brier_score_loss(y, p),
        "brier_prior_shift": brier_score_loss(y, p_prior),
        "auc_raw": roc_auc_score(y, p), "auc_prior_shift": roc_auc_score(y, p_prior),
    }])
    prior_tb.to_csv(C.TABLE_DIR / "prior_shift.csv", index=False)
    C.log("\n" + prior_tb.T.to_string())

    # ---------------- 2. per-year AUC (OOF 2017-2022, test 2023-2026)
    yr_rows = []
    for yy, sub in oof.groupby("yr"):
        if yy < C.VAL_YEARS[0] or yy > C.VAL_YEARS[-1]:
            continue
        m = np.isfinite(sub[sel])
        yr_rows.append({"year": yy, "split": "rolling-OOF",
                        "n": int(m.sum()),
                        "auc": roc_auc_score(sub.loc[m, "succ"], sub.loc[m, sel])})
    for yy, sub in te.groupby("yr"):
        yr_rows.append({"year": int(yy), "split": "out-of-time",
                        "n": len(sub),
                        "auc": roc_auc_score(sub["succ"], sub[sel])})
    year_tb = pd.DataFrame(yr_rows).sort_values("year")
    year_tb.to_csv(C.TABLE_DIR / "auc_by_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    a = year_tb[year_tb.split == "rolling-OOF"]
    b = year_tb[year_tb.split == "out-of-time"]
    ax.plot(a.year, a.auc, "o-", color="#1f4e79", label="rolling-origin OOF")
    ax.plot(b.year, b.auc, "s-", color="#c00000", label="out-of-time (>=2023)")
    ax.axhline(0.5, ls=":", color="grey")
    ax.set_ylim(0.6, 1.0); ax.set_xlabel("year"); ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Year-by-year AUC, {sel}")
    ax.legend(fontsize=7)
    C.savefig(fig, "fig_year_auc.png")
    C.log("\n" + year_tb.to_string(index=False))

    # ---------------- 3. decile lift
    order = np.argsort(-p)
    dec = np.array_split(order, 10)
    base = y.mean()
    lift_rows = []
    for i, idx in enumerate(dec):
        lift_rows.append({
            "decile": i + 1, "n": len(idx),
            "pred_mean": float(p[idx].mean()),
            "observed_rate": float(y[idx].mean()),
            "lift_vs_base": float(y[idx].mean() / base),
            "share_of_successes": float(y[idx].sum() / y.sum()),
            "precision_hits": int(y[idx].sum()),
        })
    lift_tb = pd.DataFrame(lift_rows)
    lift_tb.to_csv(C.TABLE_DIR / "decile_lift.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(lift_tb.decile, lift_tb.lift_vs_base, color="#1f4e79")
    ax.axhline(1.0, ls="--", color="black", lw=1)
    ax.set_xlabel("decile (1 = highest predicted probability)")
    ax.set_ylabel("lift vs base rate")
    ax.set_title("Decile lift, out-of-time")
    C.savefig(fig, "fig_decile_lift.png")
    C.log("\n" + lift_tb.to_string(index=False))

    # ---------------- 4. subgroup AUC + bootstrap CIs
    full = C.add_bins(df)
    te_full = full.loc[full.yr >= C.TEST_MIN_YEAR].reset_index(drop=True)
    assert len(te_full) == len(te)
    sub_rows = []
    for key in ["category_key", "goal_bin", "dur_bin", "rank_bin"]:
        for val, g in te_full.groupby(key, observed=True):
            if len(g) < 300 or g["succ"].nunique() < 2:
                continue
            idx = g.index.to_numpy()
            sub_rows.append({"group_by": key, "group": str(val), "n": len(g),
                             "rate": float(g["succ"].mean()),
                             "auc": roc_auc_score(y[idx], p[idx])})
    sub_tb = pd.DataFrame(sub_rows)
    sub_tb.to_csv(C.TABLE_DIR / "auc_by_subgroup.csv", index=False)

    cat = sub_tb[sub_tb.group_by == "category_key"].sort_values("auc")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(cat.group, cat.auc, color="#1f4e79")
    ax.set_xlim(0.6, 0.95); ax.set_xlabel("ROC-AUC")
    ax.set_title(f"AUC by category, out-of-time, {sel}")
    C.savefig(fig, "fig_subgroup_auc.png")
    C.log("\n" + sub_tb.to_string(index=False))

    # ---------------- 5. model deltas (paired bootstrap)
    deltas = []
    pairs = [(sel, "HGB"), (sel, "HGB_text"), ("HGB_text", "HGB"),
             (sel, "B2_cell_rate"), (sel, "B1_goal_only"),
             (sel, "LR"), (sel, "RF"), (sel, "B0_majority")]
    for a_, b_ in pairs:
        if a_ not in te.columns or b_ not in te.columns:
            continue
        d, lo, hi = boot_delta_auc(y, te[a_].to_numpy(float), te[b_].to_numpy(float))
        deltas.append({"model_a": a_, "model_b": b_, "delta_auc": d,
                       "ci_lo": lo, "ci_hi": hi,
                       "significant": bool(lo > 0 or hi < 0)})
    delta_tb = pd.DataFrame(deltas)
    delta_tb.to_csv(C.TABLE_DIR / "auc_deltas.csv", index=False)
    C.log("\n" + delta_tb.to_string(index=False))

    # ---------------- 6. text increment, overall and by category
    inc_rows = []
    auc_hgb = roc_auc_score(y, te["HGB"])
    auc_txt = roc_auc_score(y, te["HGB_text"])
    d, lo, hi = boot_delta_auc(y, te["HGB_text"].to_numpy(float), te["HGB"].to_numpy(float))
    inc_rows.append({"scope": "all", "n": len(y), "auc_structured": auc_hgb,
                     "auc_plus_text": auc_txt, "delta_auc": d,
                     "ci_lo": lo, "ci_hi": hi})
    for val, g in te_full.groupby("category_key", observed=True):
        if len(g) < 3000 or g["succ"].nunique() < 2:
            continue
        idx = g.index.to_numpy()
        d, lo, hi = boot_delta_auc(y[idx], te["HGB_text"].to_numpy(float)[idx],
                                   te["HGB"].to_numpy(float)[idx], n_boot=300)
        inc_rows.append({"scope": str(val), "n": len(g),
                         "auc_structured": roc_auc_score(y[idx], te["HGB"].to_numpy(float)[idx]),
                         "auc_plus_text": roc_auc_score(y[idx], te["HGB_text"].to_numpy(float)[idx]),
                         "delta_auc": d, "ci_lo": lo, "ci_hi": hi})
    inc_tb = pd.DataFrame(inc_rows)
    inc_tb.to_csv(C.TABLE_DIR / "text_increment.csv", index=False)
    C.log("\n" + inc_tb.to_string(index=False))

    # ---------------- 7. headline bootstrap CI for the selected model
    auc, lo, hi = boot_auc(y, p)
    head = pd.DataFrame([{"model": sel, "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
                          "test_n": len(y), "test_rate": float(y.mean())}])
    head.to_csv(C.TABLE_DIR / "headline_ci.csv", index=False)
    C.log("\n" + head.to_string(index=False))
    C.log("stage evaluate done")


if __name__ == "__main__":
    main()
