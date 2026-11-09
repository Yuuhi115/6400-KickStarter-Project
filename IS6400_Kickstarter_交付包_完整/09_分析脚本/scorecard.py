# -*- coding: utf-8 -*-
"""S2.1 配置评分卡：品类 × 金额档 × 时长档 的历史达标率 + Wilson CI + 覆盖率。

输出：
  11_结果与报告/config_scorecard.csv        全部 450 格（含 n、达标率、CI、成功时中位募资、模型均值）
  11_结果与报告/config_scorecard_usables.csv 仅 n>=300 的可用格
  11_结果与报告/config_scorecard.md         摘要（含覆盖率与代表性格子）
  11_结果与报告/config_scorecard_heatmap.png 热力图（n>=300 的格）
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
DUR_BINS = [0, 14, 30, 45, 60, np.inf]
DUR_LABELS = ["1-14d", "15-30d", "31-45d", "46-60d", "61d+"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


print("loading features ...", flush=True)
df = R.build_frame(R.DATA["full"], use_prep_days=True)
feats = df.attrs["numeric"] + df.attrs["categorical"]
# 额外取两列（不在特征集内）：usd_pledged 仅用于募资额描述；category_key 是标准化后的一级品类（15 个）
extra = pd.read_parquet(R.DATA["full"], columns=["usd_pledged", "category_key"])
df["usd_pledged"] = pd.to_numeric(extra["usd_pledged"].reindex(df.index), errors="coerce")
df["category_key"] = extra["category_key"].reindex(df.index).astype("string").fillna("unknown")
n_goal_na = int(df["m_goal_usd"].isna().sum())
n_dur_na = int(df["m_dur_days_exact"].isna().sum())
df = df[df["m_goal_usd"].notna() & df["m_dur_days_exact"].notna()].copy()
print(f"dropped: goal_usd NA={n_goal_na}, dur NA={n_dur_na}", flush=True)
print(f"rows usable for grid: {len(df):,}", flush=True)

df["goal_band"] = pd.cut(df["m_goal_usd"], GOAL_BINS, labels=GOAL_LABELS)
df["dur_band"] = pd.cut(df["m_dur_days_exact"], DUR_BINS, labels=DUR_LABELS)

print("scoring with the saved model ...", flush=True)
bundle = joblib.load(MODEL)
df["pred"] = bundle["pipeline"].predict_proba(df[feats])[:, 1]

print("aggregating cells ...", flush=True)
rows = []
for (cat, gb, db), g in df.groupby(["category_key", "goal_band", "dur_band"],
                                   observed=True):
    n = len(g)
    k = int(g["succ"].sum())
    lo, hi = wilson(k, n)
    win = g[g["succ"] == 1]["usd_pledged"]
    rows.append({
        "category": cat, "goal_band": str(gb), "dur_band": str(db), "n": n,
        "success_rate": k / n, "ci_low": lo, "ci_high": hi,
        "median_payout_usd_success": float(win.median()) if len(win) else np.nan,
        "mean_predicted": float(g["pred"].mean()),
        "usable_n300": bool(n >= MIN_N),
    })
card = pd.DataFrame(rows).sort_values(["category", "goal_band", "dur_band"])
card.to_csv(os.path.join(OUT, "config_scorecard.csv"), index=False, encoding="utf-8-sig")

usable = card[card["usable_n300"]].copy()
usable.to_csv(os.path.join(OUT, "config_scorecard_usables.csv"), index=False, encoding="utf-8-sig")

total = len(df)
cov = usable["n"].sum() / total
lines = [
    "# S2.1 配置评分卡（品类 × 金额档 × 时长档）", "",
    f"- 参与网格的可建模行：**{total:,}**（可建模 604,629 中排除 goal_usd 缺失 {n_goal_na} 行、时长缺失 {n_dur_na} 行）",
    f"- 格子总数：**{len(card)}**；其中 **n≥{MIN_N}** 的可用格：**{len(usable)}**"
    f"（占 {len(usable)/len(card)*100:.1f}%）",
    f"- **建议覆盖率**：可用格覆盖 **{usable['n'].sum():,} 行 = {cov*100:.1f}%** 的项目",
    f"- 中位格子样本量：{card['n'].median():.0f}；最大：{card['n'].max():,}",
    "", "## 达标率最高 / 最低的可用格（n≥300）", "",
    "| 品类 | 金额档 | 时长档 | n | 达标率 | 95% CI | 成功时中位募资 |",
    "| --- | --- | --- | ---: | ---: | --- | ---: |",
]
for _, r in usable.nlargest(8, "success_rate").iterrows():
    lines.append(f"| {r['category']} | {r['goal_band']} | {r['dur_band']} | {r['n']:,} | "
                 f"{r['success_rate']*100:.1f}% | [{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}] | "
                 f"${r['median_payout_usd_success']:,.0f} |")
lines.append("")
for _, r in usable.nsmallest(8, "success_rate").iterrows():
    lines.append(f"| {r['category']} | {r['goal_band']} | {r['dur_band']} | {r['n']:,} | "
                 f"{r['success_rate']*100:.1f}% | [{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}] | "
                 f"${r['median_payout_usd_success']:,.0f} |")
lines += ["", "## 品类级汇总（n≥300 的格加权）", "",
          "| 品类 | 覆盖行数 | 达标率 | 可用格数 |", "| --- | ---: | ---: | ---: |"]
for cat, g in usable.groupby("category"):
    lines.append(f"| {cat} | {g['n'].sum():,} | "
                 f"{(g['success_rate']*g['n']).sum()/g['n'].sum()*100:.1f}% | {len(g)} |")
lines += ["", "> 注：CI 为 Wilson 95%；`median_payout_usd_success` 用平台换算的 `usd_pledged`（仅描述用）。"]
with open(os.path.join(OUT, "config_scorecard.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

# heatmap: category x (goal,dur) for usable cells
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = usable.pivot_table(index="category", columns=["goal_band", "dur_band"],
                            values="success_rate")
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(piv.values * 100, aspect="auto", cmap="RdYlGn", vmin=0, vmax=90)
    ax.set_xticks(range(len(piv.columns)),
                  [f"{a}\n{b}" for a, b in piv.columns], fontsize=6, rotation=90)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, label="historical success rate (%)")
    ax.set_title(f"Config scorecard (cells with n>={MIN_N})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "config_scorecard_heatmap.png"), dpi=170)
    plt.close(fig)
    print("heatmap saved", flush=True)
except Exception as exc:  # pragma: no cover
    print("heatmap skipped:", exc, flush=True)

print("\n".join(lines[:12]))
print("\nusable cells:", len(usable), "| coverage:", f"{cov*100:.1f}%")
