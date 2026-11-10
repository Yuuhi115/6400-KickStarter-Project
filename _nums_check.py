import pandas as pd

d = pd.read_parquet(r"IS6400_Kickstarter_交付包_完整/11_结果与报告/cells.parquet")
u = d[d["usable_n300"]]
tier = u["intervention_tier"].astype(str)
hi = u[tier.str.startswith("high")]
hr = d[(d["rate"] < 0.10) & (d["median_payout_usd_success"] > 50000)]
print("cells", len(d), "usable", len(u))
print("high tier 格数 %d / %d = %.1f%%" % (len(hi), len(u), 100 * len(hi) / len(u)))
print("high tier 覆盖样本 %.1f%%（分母=可用格样本）" % (100 * hi["n"].sum() / u["n"].sum()))
print("high tier 占全部可建模 604,629 的 %.1f%%" % (100 * hi["n"].sum() / 604629))
print("高危高赔格数 =", len(hr))
imax = d["p_weighted_q50_usd"].idxmax()
print("p_weighted_q50 最大 = $%.0f" % d.loc[imax, "p_weighted_q50_usd"])
print("  ->", d.loc[imax, ["category", "goal_band", "dur_band", "n",
                          "p_cell_mean", "median_payout_usd_success"]].to_dict())
print("Q50 最大 = $%.0f" % d["median_payout_usd_success"].max())
imin = d["rate"].idxmin()
print("最低历史达标率 %.2f%% @ %s" % (100 * d.loc[imin, "rate"],
      d.loc[imin, ["category", "goal_band", "dur_band", "n"]].to_dict()))
