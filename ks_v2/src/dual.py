"""Stage 6 - dual objective: conditional pledge distribution and the trade-off.

Only successful projects define ``usd_pledged``; regressing on the full sample
would mix in structural zeros, so the quantile models are fitted on successes
inside the training window and evaluated on successes inside the test window.
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
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src import features as F

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

QUANTILES = [0.1, 0.5, 0.9]


def pinball(y, q_hat, q):
    d = y - q_hat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def main() -> None:
    df = C.load_v2().reset_index(drop=True)
    df = C.add_bins(df)
    train_mask = (df["yr"] <= C.TRAIN_MAX_YEAR).to_numpy()
    test_mask = (df["yr"] >= C.TEST_MIN_YEAR).to_numpy()
    tr_s = df[train_mask & (df["succ"] == 1)].reset_index(drop=True)
    te_s = df[test_mask & (df["succ"] == 1)].reset_index(drop=True)
    C.log(f"dual: train successes={len(tr_s):,} test successes={len(te_s):,}")

    spec = F.build_spec(df[train_mask])
    Xtr, mask, names = F.transform(tr_s, spec)
    Xte, _, _ = F.transform(te_s, spec)
    Xtr_np, mask = Xtr.to_numpy(np.float32), mask
    Xte_np = Xte.to_numpy(np.float32)
    ytr = np.log1p(tr_s["usd_pledged"].to_numpy(dtype=float))
    yte = np.log1p(te_s["usd_pledged"].to_numpy(dtype=float))

    preds = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                          categorical_features=mask, **C.HGB_REG)
        m.fit(Xtr_np, ytr)
        preds[q] = m.predict(Xte_np)

    # enforce monotonicity across quantiles (rearrangement)
    Q = np.column_stack([preds[q] for q in QUANTILES])
    Q = np.sort(Q, axis=1)
    crossed = 0
    for i in range(1, len(QUANTILES)):
        crossed += int((Q[:, i] < Q[:, i - 1]).sum())

    eval_rows = []
    for i, q in enumerate(QUANTILES):
        eval_rows.append({"quantile": q, "pinball": pinball(yte, Q[:, i], q)})
    q_tb = pd.DataFrame(eval_rows)
    q_tb.to_csv(C.TABLE_DIR / "quantile_pinball.csv", index=False)

    cover = pd.DataFrame([{
        "p10_p90_coverage": float(((yte >= Q[:, 0]) & (yte <= Q[:, 2])).mean()),
        "p50_below": float((yte <= Q[:, 1]).mean()),
        "n": len(yte),
    }])
    cover.to_csv(C.TABLE_DIR / "quantile_coverage.csv", index=False)
    C.log("\n" + q_tb.to_string(index=False))
    C.log("\n" + cover.to_string(index=False))

    # ------------------------ trade-off: predicted P(success) vs conditional pledge
    te = pd.read_csv(C.OUT / "models" / "test_preds.csv.gz")
    sel = json.loads((C.TABLE_DIR / "model_selection.json").read_text(encoding="utf-8"))["selected"]
    p_succ = te[sel].to_numpy(float)
    # align test successes to the quantile predictions by row position
    test_pos = np.flatnonzero(test_mask)
    succ_pos = np.flatnonzero((df["succ"] == 1).to_numpy() & test_mask)
    pos_in_test = pd.Series(np.arange(len(test_pos)), index=test_pos)
    row_map = pos_in_test.loc[succ_pos].to_numpy()
    assert len(row_map) == len(te_s)

    p_s = p_succ[row_map]
    med = np.expm1(Q[:, 1])
    hi = np.expm1(Q[:, 2])
    observed = te_s["usd_pledged"].to_numpy(float)

    trade = pd.DataFrame({
        "goal_bin": te_s["goal_bin"].astype(str).values,
        "p_success": p_s, "pred_median": med, "pred_p90": hi,
        "observed": observed,
    })
    agg = trade.groupby("goal_bin", observed=True).agg(
        n=("observed", "size"), p_success_mean=("p_success", "mean"),
        pred_median_median=("pred_median", "median"),
        observed_median=("observed", "median"),
        pred_p90_median=("pred_p90", "median"),
    ).reindex(C.GOAL_LABELS).reset_index()
    agg.to_csv(C.TABLE_DIR / "tradeoff_by_goal.csv", index=False)
    C.log("\n" + agg.to_string(index=False))

    # drop bins too thin to plot (>500k has n=4); they stay in the CSV table
    plot = agg[agg.n >= 20].reset_index(drop=True)
    labels = [f"{r.goal_bin}\n(n={int(r.n):,})" for r in plot.itertuples()]
    fig, ax1 = plt.subplots(figsize=(7.4, 4.0))
    ax1.plot(range(len(plot)), plot.p_success_mean, "o-", color="#1f4e79",
             lw=1.6, ms=5, label="mean predicted P(success)")
    ax1.set_ylabel("mean predicted P(success)", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.set_ylim(0.55, 0.92)
    ax2 = ax1.twinx()
    ax2.plot(range(len(plot)), plot.observed_median, "s--", color="#c00000",
             lw=1.4, ms=4, label="observed median pledge (successes)")
    ax2.plot(range(len(plot)), plot.pred_median_median, "^:", color="#7f7f7f",
             lw=1.4, ms=4, label="model-predicted P50 pledge")
    ax2.set_ylabel("median pledge (USD, log scale)", color="#c00000")
    ax2.set_yscale("log")
    ax1.set_xticks(range(len(plot)))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_xlabel("goal bin")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle("Trade-off: success probability vs conditional pledge (out-of-time; n<20 bars omitted)",
                 y=0.99, fontsize=9)
    fig.tight_layout()
    C.savefig(fig, "fig_tradeoff.png")

    C.log("dual: trade-off table and figure written")
    C.log("stage dual done")


if __name__ == "__main__":
    main()
