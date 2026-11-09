# -*- coding: utf-8 -*-
"""S2.3/S2.4 三个决策轴：business value、intervention potential、risk。

口径：
  business_value = P(达标) × 同格「成功时中位募资」(usd_pledged)   —— 募资额代理，非真实商业价值
  intervention   = max over {目标金额档, 时长档} [P(改后) − P(当前)]
                   ΔP≥15pp = 高 / 5–15pp = 中 / <5pp = 低
  risk           = (1 − P) × 同格「成功时中位募资」
                   另标「高危高赔格」：达标率<10% 且 成功时中位募资 > $50,000
"""
import math
import os

import joblib
import numpy as np
import pandas as pd

import replicate_pilot as R

OUT = os.path.join(R.ROOT, "11_结果与报告")
MODEL = os.path.join(OUT, "model_histgb_ours.joblib")
MIN_N = 300

GOAL_BINS = [0, 1000, 5000, 10000, 20000, 50000, np.inf]
GOAL_LABELS = ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"]
GOAL_MID = {"<=1k": 800.0, "1-5k": 2500.0, "5-10k": 7000.0,
            "10-20k": 14000.0, "20-50k": 30000.0, ">50k": 100000.0}
DUR_BINS = [0, 14, 30, 45, 60, np.inf]
DUR_LABELS = ["1-14d", "15-30d", "31-45d", "46-60d", "61d+"]
DUR_MID = {"1-14d": 10.0, "15-30d": 25.0, "31-45d": 38.0, "46-60d": 53.0, "61d+": 75.0}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


print("features ...", flush=True)
df = R.build_frame(R.DATA["full"], use_prep_days=True)
NUM, CAT = df.attrs["numeric"], df.attrs["categorical"]
feats = NUM + CAT
extra = pd.read_parquet(R.DATA["full"], columns=["usd_pledged", "category_key"])
df["usd_pledged"] = pd.to_numeric(extra["usd_pledged"].reindex(df.index), errors="coerce")
df["category_key"] = extra["category_key"].reindex(df.index).astype("string").fillna("unknown")
df = df[df["m_goal_usd"].notna() & df["m_dur_days_exact"].notna()].copy()
df["goal_band"] = pd.cut(df["m_goal_usd"], GOAL_BINS, labels=GOAL_LABELS)
df["dur_band"] = pd.cut(df["m_dur_days_exact"], DUR_BINS, labels=DUR_LABELS)
print(f"rows: {len(df):,}", flush=True)

pipe = joblib.load(MODEL)["pipeline"]
df["p"] = pipe.predict_proba(df[feats])[:, 1]

print("cell stats ...", flush=True)
recs, profiles = [], []
for (cat, gb, db), g in df.groupby(["category_key", "goal_band", "dur_band"], observed=True):
    n, k = len(g), int(g["succ"].sum())
    lo, hi = wilson(k, n)
    win = g.loc[g["succ"] == 1, "usd_pledged"]
    prof = {c: float(g[c].median()) for c in NUM}
    prof.update({c: str(g[c].mode().iloc[0]) for c in CAT})
    profiles.append(prof)
    recs.append({"category": cat, "goal_band": str(gb), "dur_band": str(db), "n": n,
                 "success_rate": k / n, "ci_low": lo, "ci_high": hi,
                 "median_payout_usd_success": float(win.median()) if len(win) else np.nan,
                 "mean_p_observed": float(g["p"].mean())})
cells = pd.DataFrame(recs).reset_index(drop=True)
print(f"cells: {len(cells)}", flush=True)

prof_df = pd.DataFrame(profiles)[feats]
cells["p_base"] = pipe.predict_proba(prof_df)[:, 1]

print("counterfactual scan (cell level, 450 x 30) ...", flush=True)
rows, idx = [], []
for i in range(len(cells)):
    cur_g, cur_d = cells.at[i, "goal_band"], cells.at[i, "dur_band"]
    for gb2 in GOAL_LABELS:
        for db2 in DUR_LABELS:
            if gb2 == cur_g and db2 == cur_d:
                continue
            prof = dict(profiles[i])
            prof["m_log_goal_usd"] = float(np.log10(GOAL_MID[gb2]))
            prof["m_dur_days_exact"] = float(DUR_MID[db2])
            rows.append(prof)
            idx.append((i, gb2, db2))
cf = pd.DataFrame(rows)[feats]
cf_p = pipe.predict_proba(cf)[:, 1]
meta = pd.DataFrame(idx, columns=["cell_idx", "new_goal_band", "new_dur_band"])
meta["p_new"] = cf_p
meta["delta"] = meta["p_new"].to_numpy() - cells["p_base"].to_numpy()[meta["cell_idx"].to_numpy()]
best = meta.loc[meta.groupby("cell_idx")["delta"].idxmax()].set_index("cell_idx")

cells["dp_max"] = best["delta"].reindex(cells.index).to_numpy()
cells["best_goal_band"] = best["new_goal_band"].reindex(cells.index).to_numpy()
cells["best_dur_band"] = best["new_dur_band"].reindex(cells.index).to_numpy()
cells["intervention_tier"] = pd.cut(cells["dp_max"], [-1.0, 0.05, 0.15, 1.0],
                                    labels=["low(<5pp)", "mid(5-15pp)", "high(>=15pp)"])
cells["usable_n300"] = cells["n"] >= MIN_N
cells["expected_payout_usd"] = cells["p_base"] * cells["median_payout_usd_success"]
cells["risk_score_usd"] = (1 - cells["p_base"]) * cells["median_payout_usd_success"]
cells["high_risk_high_stake"] = (cells["success_rate"] < 0.10) & \
                                (cells["median_payout_usd_success"] > 50000)
cells = cells.sort_values(["category", "goal_band", "dur_band"])
cells.to_csv(os.path.join(OUT, "decision_axes_cells.csv"), index=False, encoding="utf-8-sig")

# 项目级（抽样输出）
lookup = cells.set_index(["category", "goal_band", "dur_band"])
key = pd.MultiIndex.from_frame(df[["category_key", "goal_band", "dur_band"]].astype(str))
proj = df[["p", "succ", "m_goal_usd", "m_dur_days_exact"]].copy()
proj["median_payout"] = lookup["median_payout_usd_success"].reindex(key).to_numpy()
proj["dp_max"] = lookup["dp_max"].reindex(key).to_numpy()
proj["intervention_tier"] = lookup["intervention_tier"].reindex(key).to_numpy()
proj["expected_payout"] = proj["p"] * proj["median_payout"]
proj["risk_score"] = (1 - proj["p"]) * proj["median_payout"]
proj.sample(min(5000, len(proj)), random_state=42).to_csv(
    os.path.join(OUT, "decision_axes_sample.csv"), index=False, encoding="utf-8-sig")

usable = cells[cells["usable_n300"]]
N = len(df)
lines = [
    "# S2.3/S2.4 三个决策轴：business value / intervention potential / risk", "",
    "## 口径（照抄进报告）", "",
    "- **business value（代理）**：`E[募资额] = P(达标) × 同格「成功时中位募资」`；数据无收入/成本，属代理指标。",
    "- **intervention potential**：`ΔP_max = max over {目标金额档, 时长档} [P(改后) − P(当前)]`；"
    "≥15pp 高 / 5–15pp 中 / <5pp 低（只扫这两个可行动杠杆，文案层待 TF-IDF）。",
    "- **risk**：`(1 − P) × 同格「成功时中位募资」`；另标「高危高赔格」= 达标率<10% 且 中位募资>$50,000。",
    "", f"参与行：**{N:,}**；格子：**{len(cells)}**（可用格 n≥{MIN_N}：**{len(usable)}**）。", "",
    "## intervention potential 分档（按覆盖行数）", "",
    "| 档位 | 格数 | 覆盖行数 | 占比 |", "| --- | ---: | ---: | ---: |",
]
for t in ["low(<5pp)", "mid(5-15pp)", "high(>=15pp)"]:
    g = usable[usable["intervention_tier"] == t]
    lines.append(f"| {t} | {len(g)} | {int(g['n'].sum()):,} | {g['n'].sum()/N*100:.1f}% |")
lines += ["", "## ΔP 最大的可用格（最值得干预）", "",
          "| 品类 | 当前配置 | n | 当前 P | ΔP_max | 建议动作 |", "| --- | --- | ---: | ---: | ---: | --- |"]
for _, r in usable.nlargest(8, "dp_max").iterrows():
    lines.append(f"| {r['category']} | {r['goal_band']} × {r['dur_band']} | {r['n']:,} | "
                 f"{r['p_base']*100:.1f}% | **+{r['dp_max']*100:.1f}pp** | "
                 f"目标→{r['best_goal_band']}，时长→{r['best_dur_band']} |")
lines += ["", "## 高危高赔格（risk 最高）", "",
          "| 品类 | 配置 | n | 达标率 | 成功时中位募资 | 风险分 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
for _, r in cells[cells["high_risk_high_stake"]].nlargest(8, "risk_score_usd").iterrows():
    lines.append(f"| {r['category']} | {r['goal_band']} × {r['dur_band']} | {r['n']:,} | "
                 f"{r['success_rate']*100:.1f}% | ${r['median_payout_usd_success']:,.0f} | "
                 f"${r['risk_score_usd']:,.0f} |")
lines += ["", "## business value：期望募资额最高的可用格", "",
          "| 品类 | 配置 | n | P | 成功时中位募资 | E[募资额] |", "| --- | --- | ---: | ---: | ---: | ---: |"]
for _, r in usable.nlargest(8, "expected_payout_usd").iterrows():
    lines.append(f"| {r['category']} | {r['goal_band']} × {r['dur_band']} | {r['n']:,} | "
                 f"{r['p_base']*100:.1f}% | ${r['median_payout_usd_success']:,.0f} | "
                 f"**${r['expected_payout_usd']:,.0f}** |")
lines += ["", "## 局限（必须写进报告）", "",
          "1. business value 用募资额代理，不含平台佣金、获客成本、创作者生命周期价值；",
          "2. intervention 只扫「目标金额档」「时长档」；文案层须等 TF-IDF（当前长度代理会误导）；",
          "3. 概率与金额均为**关联**，不是因果。"]
with open(os.path.join(OUT, "decision_axes.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    s = usable.dropna(subset=["median_payout_usd_success"])
    ax.scatter(s["median_payout_usd_success"], s["success_rate"] * 100,
               s=np.sqrt(s["n"]) * 2, alpha=0.55, c="#2E6FBA",
               edgecolors="white", linewidths=0.4)
    ax.set_xscale("log")
    ax.axhline(50, color="#999", ls="--", lw=1)
    ax.axvline(50000, color="#C0392B", ls="--", lw=1)
    ax.set_xlabel("median payout among successes (USD, log)")
    ax.set_ylabel("historical success rate (%)")
    ax.set_title("Cells (n>=300): success rate vs payout")
    ax.grid(alpha=0.25)
    ax = axes[1]
    counts = [usable[usable["intervention_tier"] == t]["n"].sum() for t in
              ["low(<5pp)", "mid(5-15pp)", "high(>=15pp)"]]
    ax.bar(["low(<5pp)", "mid(5-15pp)", "high(>=15pp)"], np.array(counts) / 1000,
           color=["#9AA7B4", "#E0A33A", "#2E8B57"])
    ax.set_ylabel("projects (thousand)")
    ax.set_title("Intervention potential (max dP from goal/duration moves)")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(np.array(counts) / 1000):
        ax.text(i, v, f"{v:.0f}k", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "decision_axes.png"), dpi=170)
    plt.close(fig)
    print("figure saved", flush=True)
except Exception as exc:
    print("figure skipped:", exc, flush=True)

print("\n".join(lines[:24]))
print("\nusable:", len(usable), "| high-risk-high-stake:", int(cells["high_risk_high_stake"].sum()))
