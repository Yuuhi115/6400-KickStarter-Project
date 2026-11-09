"""Stage 1 - descriptive numbers with Wilson confidence intervals.

Everything is computed separately on the training window (<=2022) and the
out-of-time window (>=2023) so that the drift is visible instead of smoothed
away. Figures use English labels (no CJK font dependency).
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src import config as C

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def rate_block(df: pd.DataFrame, key: str) -> pd.DataFrame:
    g = df.groupby(key, observed=True).agg(n=("succ", "size"), k=("succ", "sum"))
    g["rate"] = g["k"] / g["n"]
    g["lo"], g["hi"] = C.wilson_ci(g["k"], g["n"])
    return g.reset_index()


def train_test_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    tr = rate_block(df[df.yr <= C.TRAIN_MAX_YEAR], key)[["n", "rate"]]
    te = rate_block(df[df.yr >= C.TEST_MIN_YEAR], key)[["n", "rate"]]
    tr.columns = ["n_train", "rate_train"]
    te.columns = ["n_test", "rate_test"]
    out = tr.join(te, how="outer")
    out["delta"] = out["rate_test"] - out["rate_train"]
    return out.reset_index()


def main() -> None:
    df = C.load_v2()
    df = C.add_bins(df)
    tr = df[df.yr <= C.TRAIN_MAX_YEAR]
    te = df[df.yr >= C.TEST_MIN_YEAR]
    C.log(f"eda: v2={len(df):,} train={len(tr):,} test={len(te):,}")

    # ------------------------------- 1. headline panel (single-row table)
    headline = pd.DataFrame([{
        "v2_rows": len(df), "v2_rate": df.succ.mean(),
        "train_rows": len(tr), "train_rate": tr.succ.mean(),
        "test_rows": len(te), "test_rate": te.succ.mean(),
        "majority_baseline_test": max(te.succ.mean(), 1 - te.succ.mean()),
        "train_median_goal": tr.goal.median(),
        "test_median_goal": te.goal.median(),
        "train_median_duration": tr.dur_days.median(),
        "test_median_duration": te.dur_days.median(),
    }])
    headline.to_csv(C.TABLE_DIR / "eda_headline.csv", index=False)
    C.log(headline.T.to_string())

    # ------------------------------- 2. marginal tables
    for key, fname in [("yr", "eda_year.csv"), ("category_key", "eda_category.csv"),
                       ("goal_bin", "eda_goal_bin.csv"), ("dur_bin", "eda_duration.csv"),
                       ("rank_bin", "eda_creator_rank.csv"),
                       ("location_type", "eda_location_type.csv")]:
        tb = train_test_table(df, key)
        tb.to_csv(C.TABLE_DIR / fname, index=False)

    # ------------------------------- 3. money side: median pledged per goal bin
    succ_tr = tr[tr.succ == 1]
    money = succ_tr.groupby("goal_bin", observed=True)["usd_pledged"].agg(
        ["size", "median", "mean"]).reset_index()
    money.columns = ["goal_bin", "n_success", "median_usd_pledged", "mean_usd_pledged"]
    money.to_csv(C.TABLE_DIR / "eda_money_by_goal.csv", index=False)
    C.log("\n" + money.to_string())

    # ------------------------------- 4. Simpson check: category x goal bin
    ct = (tr.groupby(["category_key", "goal_bin"], observed=True)
          .agg(n=("succ", "size"), rate=("succ", "mean")).reset_index())
    ct = ct[ct.n >= 100]
    ct.to_csv(C.TABLE_DIR / "eda_simpson_category_goal.csv", index=False)

    # ------------------------------- figures
    gaps = df.groupby("yr", observed=True)["succ"].agg(["size", "mean"]).reset_index()
    gaps.columns = ["yr", "n", "rate"]
    gaps["lo"], gaps["hi"] = C.wilson_ci(gaps["rate"] * gaps["n"], gaps["n"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    tr_mask = gaps.yr <= C.TRAIN_MAX_YEAR
    ax.plot(gaps.loc[tr_mask, "yr"], gaps.loc[tr_mask, "rate"], "o-", color="#1f4e79",
            label="train window (<=2022)")
    ax.plot(gaps.loc[~tr_mask, "yr"], gaps.loc[~tr_mask, "rate"], "s-", color="#c00000",
            label="out-of-time (>=2023)")
    ax.fill_between(gaps.yr, gaps.lo, gaps.hi, color="grey", alpha=0.18, label="95% Wilson CI")
    ax.axhline(df.succ.mean(), ls="--", lw=1, color="black", label="overall rate")
    ax.set_xlabel("launch year"); ax.set_ylabel("success rate")
    ax.set_title("Success rate by launch year (USD-only, n=408,347)")
    ax.legend(fontsize=7, loc="lower right")
    C.savefig(fig, "fig_eda_year_rate.png")

    for key, fname, title, order in [
        ("goal_bin", "fig_eda_goal_bin.png", "Success rate by goal (USD)", C.GOAL_LABELS),
        ("dur_bin", "fig_eda_duration.png", "Success rate by campaign duration (days)", C.DUR_LABELS),
    ]:
        tb = rate_block(tr, key).set_index(key).reindex(order)
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        ax.bar(range(len(tb)), tb["rate"], color="#1f4e79")
        ax.errorbar(range(len(tb)), tb["rate"],
                    yerr=[tb["rate"] - tb["lo"], tb["hi"] - tb["rate"]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_xticks(range(len(tb))); ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_xlabel("bin"); ax.set_ylabel("success rate")
        ax.set_title(f"{title} - train window (<=2022), n={len(tr):,}")
        C.savefig(fig, fname)

    cat = rate_block(tr, "category_key").sort_values("rate")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(cat.category_key, cat.rate, color="#1f4e79")
    ax.errorbar(cat.rate, range(len(cat)),
                xerr=[cat.rate - cat.lo, cat.hi - cat.rate],
                fmt="none", ecolor="black", capsize=2, lw=1)
    ax.set_xlabel("success rate"); ax.set_title("Success rate by category (train window)")
    C.savefig(fig, "fig_eda_category.png")

    C.log("stage eda done")


if __name__ == "__main__":
    main()
