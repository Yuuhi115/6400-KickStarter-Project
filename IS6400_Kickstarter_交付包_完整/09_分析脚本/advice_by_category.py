# -*- coding: utf-8 -*-
"""按品类生成配置建议：最稳妥 / 最高期望价值 / 应避免，并给出相对"该品类最常见配置"的变化。

评级（四道闸门的前三道）：
  推荐级   = 推荐格与基准格 n≥300，且两者 Wilson CI 不重叠
  参考级   = n≥300，但 CI 重叠（差异不显著）
  数据不足 = 该品类可用格 < 2

输出：11_结果与报告/advice_by_category.md / .csv
"""
import os

from pathlib import Path

import numpy as np
import pandas as pd

OUT = os.path.join(str(Path(__file__).resolve().parents[1]), "11_结果与报告")
cells = pd.read_csv(os.path.join(OUT, "decision_axes_cells.csv"), encoding="utf-8-sig")
cells = cells[cells["n"] >= 300].copy()   # 只对 n>=300 的格给建议
print("usable cells:", len(cells))

rows = []
for cat, g in cells.groupby("category"):
    if len(g) < 2:
        rows.append({"category": cat, "grade": "数据不足", "n_cells": len(g)})
        continue
    base = g.loc[g["n"].idxmax()]                       # 该品类最常见配置
    safest = g.loc[g["success_rate"].idxmax()]          # 最稳妥
    best_e = g.loc[g["expected_payout_usd"].idxmax()]   # 最高期望价值
    worst = g.loc[g["success_rate"].idxmin()]           # 最差

    def overlap(a, b):
        return not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])

    def grade_of(rec):
        return "推荐级" if not overlap(rec, base) else "参考级"

    rows.append({
        "category": cat, "n_cells": len(g),
        "base_cfg": f"{base['goal_band']} × {base['dur_band']}",
        "base_n": int(base["n"]), "base_rate": round(base["success_rate"] * 100, 1),
        "base_ci": f"[{base['ci_low']*100:.1f}, {base['ci_high']*100:.1f}]",
        "base_E": round(base["expected_payout_usd"]),
        "safest_cfg": f"{safest['goal_band']} × {safest['dur_band']}",
        "safest_rate": round(safest["success_rate"] * 100, 1),
        "safest_dp": round((safest["p_base"] - base["p_base"]) * 100, 1),
        "safest_grade": grade_of(safest),
        "bestE_cfg": f"{best_e['goal_band']} × {best_e['dur_band']}",
        "bestE_P": round(best_e["p_base"] * 100, 1),
        "bestE_E": round(best_e["expected_payout_usd"]),
        "bestE_dE": round(best_e["expected_payout_usd"] - base["expected_payout_usd"]),
        "bestE_grade": grade_of(best_e),
        "avoid_cfg": f"{worst['goal_band']} × {worst['dur_band']}",
        "avoid_rate": round(worst["success_rate"] * 100, 1),
        "avoid_risk": round(worst["risk_score_usd"]),
    })

adv = pd.DataFrame(rows).sort_values("category")
adv.to_csv(os.path.join(OUT, "advice_by_category.csv"), index=False, encoding="utf-8-sig")

L = ["# 按品类的配置建议（基于 450 格评分卡，仅 n≥300 的格）", "",
     "> **口径**：建议来自**历史关联**，不是因果；只改「目标金额档」与「时长档」两个乐杆。",
     "> **评级**：推荐级 = 与“该品类最常见配置”的 95% CI 不重叠；参考级 = n≥300 但 CI 重叠。", "",
     "| 品类 | 最常见配置(n) | 达标率[95%CI] | 最稳妥配置 | ΔP | 最高期望价值配置 | P | E[募资额] | 应避免 | 级别 |",
     "| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |"]
for _, r in adv.iterrows():
    if r.get("grade", "") == "数据不足" or "base_cfg" not in r:
        L.append(f"| {r['category']} | — | — | — | — | — | — | — | — | 数据不足（{r['n_cells']} 格） |")
        continue
    L.append(f"| {r['category']} | {r['base_cfg']} ({r['base_n']:,}) | {r['base_rate']}% {r['base_ci']} | "
             f"{r['safest_cfg']} | {r['safest_dp']:+.1f}pp | {r['bestE_cfg']} | {r['bestE_P']}% | "
             f"${r['bestE_E']:,} | {r['avoid_cfg']} ({r['avoid_rate']}%) | "
             f"稳妥:{r['safest_grade']} / 价值:{r['bestE_grade']} |")

L += ["", "## 三个具体品类的完整建议", ""]
for cat in ["games", "technology", "comics"]:
    g = cells[cells["category"] == cat].sort_values("expected_payout_usd", ascending=False)
    if g.empty:
        continue
    L += [f"### {cat}", "",
          "| 目标金额档 | 时长档 | n | 达标率 | 95% CI | 成功时中位募资 | E[募资额] | 干预后 ΔP | 建议动作 |",
          "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for _, r in g.head(8).iterrows():
        L.append(f"| {r['goal_band']} | {r['dur_band']} | {r['n']:,} | {r['success_rate']*100:.1f}% | "
                 f"[{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}] | ${r['median_payout_usd_success']:,.0f} | "
                 f"${r['expected_payout_usd']:,.0f} | +{r['dp_max']*100:.1f}pp | "
                 f"{r['best_goal_band']} × {r['best_dur_band']} |")
    L.append("")

L += ["## 使用限制（写报告必须带）", "",
      "1. 只能给「目标金额档」与「时长档」两类建议；**文案建议需要 TF-IDF 面板（未完成）**；",
      "2. 未拆分到子品类（147 个）与国家（25 个）——若需要更细粒度，需重跑评分卡并做多重比较校正；",
      "3. 建议是**同品类内**的历史关联，跨品类不可直接套用；",
      "4. 「最高期望价值」格往往目标较大、达标率较低，**必须与风险同时展示**（见 avoid 列）。"]
with open(os.path.join(OUT, "advice_by_category.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print("\n".join(L[:20]))
print("\nsaved advice_by_category.md / .csv")
