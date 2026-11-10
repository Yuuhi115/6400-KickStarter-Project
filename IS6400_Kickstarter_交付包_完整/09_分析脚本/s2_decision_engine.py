# -*- coding: utf-8 -*-
"""S2: single source of truth (cells) + constrained recommendation engine.

修复《检查报告_20261109.md》的 P1-1 / P1-2 / P1-3 / P1-4 根因：

  1. `cells.parquet` —— 同一格子的**四种口径并排**，下游只能读它
       n / k / rate / ci_low / ci_high / median_payout_usd_success
       p_profile   （中位数+众数画像的模型预测）
       p_cell_mean （格内逐项目模型预测的均值）
  2. 建议引擎加约束（顺序即报告给的顺序）：
       候选生成 → 平台合法性(时长 1–60 天) → 目标格 n>=300 → 允许"保持原配置"
       → 输出拒绝理由
  3. 指标改名（去掉伪期望）：
       p_weighted_q50_usd  = p_cell_mean * median_payout   「概率加权典型募资额」
       exposure_usd        = (1-p_cell_mean) * median_payout 「未实现金额暴露」

输出：11_结果与报告/cells.parquet、decision_axes_v2.csv、S2_建议引擎.md、S2_案例.md
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
DATA_FULL = PKG / "08_建模就绪数据" / "ks_model_ready_v1.parquet"
MODEL = OUT / "model_histgb_ours.joblib"

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

# ---- 档位定义（唯一定义处，下游不得再定义） --------------------------------
GOAL_BINS = [0, 1_000, 5_000, 10_000, 20_000, 50_000, np.inf]
GOAL_LABELS = ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"]
GOAL_MID = {"<=1k": 800.0, "1-5k": 2500.0, "5-10k": 7000.0,
            "10-20k": 14000.0, "20-50k": 30000.0, ">50k": 100000.0}
DUR_BINS = [0, 14, 30, 45, 60, np.inf]
DUR_LABELS = ["1-14d", "15-30d", "31-45d", "46-60d", "61d+"]
DUR_MID = {"1-14d": 10.0, "15-30d": 25.0, "31-45d": 38.0, "46-60d": 53.0}
# 平台合法性：Kickstarter 项目周期 1–60 天 ⇒ 61d+ 只能作历史描述，不得作为建议
CAND_DUR = ["1-14d", "15-30d", "31-45d", "46-60d"]
MIN_CELL_N = 300


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    print("[1/5] 载入数据与模型 ...", flush=True)
    df = rp.build_frame(rp.DATA["full"], use_prep_days=True)
    num, cat = df.attrs["numeric"], df.attrs["categorical"]
    feats = num + cat
    train, val, test = rp.make_splits(df, "ours")

    obj = joblib.load(MODEL)
    pipe = obj["pipeline"]
    print(f"      模型特征数={len(obj['features'])}，阈值={obj['threshold']:.2f}", flush=True)

    # 静态列（成交额），用于格级典型募资额
    static = pd.read_parquet(DATA_FULL, columns=[
        "m_is_modelable", "succ", "usd_pledged", "category_key"])
    static = static[static["m_is_modelable"].fillna(False)]
    static = static[pd.to_numeric(static["succ"], errors="coerce").notna()]
    static = static.reindex(df.index)
    df["usd_pledged"] = static["usd_pledged"].to_numpy()
    df["category_key"] = static["category_key"].to_numpy()

    df["goal_band"] = pd.cut(df["m_goal_usd"], GOAL_BINS, labels=GOAL_LABELS, right=True)
    df["dur_band"] = pd.cut(df["m_dur_days_exact"], DUR_BINS, labels=DUR_LABELS, right=True)
    df["succ"] = df["succ"].astype(int)

    print("[2/5] 逐项目打分（用于 p_cell_mean）...", flush=True)
    df["p"] = pipe.predict_proba(df[feats])[:, 1]

    print("[3/5] 建 cells（四种口径并排）...", flush=True)
    recs, profs = [], []
    for (c, gb, db), g in df.groupby(["category_key", "goal_band", "dur_band"], observed=True):
        n, k = len(g), int(g["succ"].sum())
        lo, hi = wilson(k, n)
        win = g.loc[g["succ"] == 1, "usd_pledged"]
        prof = {x: float(g[x].median()) for x in num}
        prof.update({x: str(g[x].mode().iloc[0]) for x in cat})
        profs.append(prof)
        recs.append({"category": c, "goal_band": str(gb), "dur_band": str(db),
                     "n": n, "k": k, "rate": k / n, "ci_low": lo, "ci_high": hi,
                     "median_payout_usd_success": float(win.median()) if len(win) else np.nan,
                     "p_cell_mean": float(g["p"].mean())})
    cells = pd.DataFrame(recs)
    prof_df = pd.DataFrame(profs)[feats]
    cells["p_profile"] = pipe.predict_proba(prof_df)[:, 1]
    cells["usable_n300"] = cells["n"] >= MIN_CELL_N
    cells = cells.sort_values(["category", "goal_band", "dur_band"]).reset_index(drop=True)
    print(f"      cells={len(cells)}，可用(n>=300)={int(cells['usable_n300'].sum())}", flush=True)

    print("[4/5] 受约束的反事实扫描 ...", flush=True)
    idx = cells.set_index(["category", "goal_band", "dur_band"])["n"]
    rows = []
    for i, r in cells.iterrows():
        best = {"dp": 0.0, "goal_band": r["goal_band"], "dur_band": r["dur_band"],
                "label": "保持原配置", "p_new": r["p_profile"], "reason": "原配置始终是候选"}
        rejected = []
        if not r["usable_n300"]:
            rows.append({"cell": i, "dp_max": 0.0, "action": "样本不足(出发格n<300)",
                         "best_goal_band": r["goal_band"], "best_dur_band": r["dur_band"],
                         "p_new": r["p_profile"], "n_target": r["n"], "rejected": ""})
            continue
        for gb2 in GOAL_LABELS:
            for db2 in CAND_DUR:
                if gb2 == r["goal_band"] and db2 == r["dur_band"]:
                    continue
                n_target = idx.get((r["category"], gb2, db2))
                if n_target is None or pd.isna(n_target):
                    rejected.append(f"{gb2}x{db2}:目标格不存在")
                    continue
                if n_target < MIN_CELL_N:
                    rejected.append(f"{gb2}x{db2}:目标格n={int(n_target)}<{MIN_CELL_N}")
                    continue
                prof = dict(profs[i])
                prof["m_log_goal_usd"] = float(np.log10(GOAL_MID[gb2]))
                prof["m_dur_days_exact"] = float(DUR_MID[db2])
                p_new = float(pipe.predict_proba(pd.DataFrame([prof])[feats])[:, 1][0])
                dp = p_new - float(r["p_profile"])
                if dp > best["dp"]:
                    best = {"dp": dp, "goal_band": gb2, "dur_band": db2,
                            "label": f"{gb2}x{db2}", "p_new": p_new,
                            "reason": f"目标格n={int(n_target)}"}
        rows.append({"cell": i, "dp_max": max(0.0, best["dp"]), "action": best["label"],
                     "best_goal_band": best["goal_band"], "best_dur_band": best["dur_band"],
                     "p_new": best["p_new"], "n_target": idx.get(
                         (r["category"], best["goal_band"], best["dur_band"]), r["n"]),
                     "rejected": "; ".join(rejected[:4])})
    adv = pd.DataFrame(rows).set_index("cell")
    cells = cells.join(adv)

    # 指标改名：不用"期望"
    cells["p_weighted_q50_usd"] = cells["p_cell_mean"] * cells["median_payout_usd_success"]
    cells["exposure_usd"] = (1 - cells["p_cell_mean"]) * cells["median_payout_usd_success"]
    cells["intervention_tier"] = pd.cut(cells["dp_max"], [-0.001, 0.05, 0.15, 1.0],
                                        labels=["low(<5pp)", "mid(5-15pp)", "high(>=15pp)"])
    cells.to_parquet(OUT / "cells.parquet", index=False)
    cells.to_csv(OUT / "decision_axes_v2.csv", index=False, encoding="utf-8-sig")

    print("[5/5] 校验与案例 ...", flush=True)
    use = cells[cells["usable_n300"]]
    neg = int((use["dp_max"] < 0).sum())
    n61 = int((use["best_dur_band"].astype(str) == "61d+").sum())
    under = int(sum(1 for _, r in use.iterrows()
                    if pd.notna(r["n_target"]) and r["n_target"] < MIN_CELL_N
                    and r["action"] != "保持原配置"))
    lowest = int((use["best_goal_band"].astype(str) == "<=1k").sum())
    print(f"  G3 自检: 负dp_max={neg}（应为0）, 61d+建议={n61}（应为0）, 目标格n<300={under}（应为0）")
    print(f"  可用格 {len(use)}；最优动作降到<=1k 的格数={lowest} "
          f"({lowest/len(use)*100:.1f}%)；保持原配置的格数="
          f"{int((use['action']=='保持原配置').sum())}")

    # ---- 案例：真实项目（不是合成画像） -------------------------------------
    case = df[(df["category_key"] == "games") & (df["goal_band"].astype(str) == "5-10k")
              & (df["dur_band"].astype(str) == "15-30d") & (df.index.isin(test.index))]
    # 取格内“中位水平”的项目作为代表（不取最高/最低，避免选择性）
    med = case["p"].median()
    proj = case.iloc[int((case["p"] - med).abs().to_numpy().argmin())]
    cell = cells[(cells["category"] == "games") & (cells["goal_band"] == "5-10k")
                 & (cells["dur_band"] == "15-30d")].iloc[0]
    alt_goal = cell["best_goal_band"] if cell["action"] != "保持原配置" else "<=1k"
    prof = dict(profs[int(cell.name)])
    prof["m_log_goal_usd"] = float(np.log10(GOAL_MID[alt_goal]))
    p_alt = float(pipe.predict_proba(pd.DataFrame([prof])[feats])[:, 1][0])

    lines = [
        "# S2 案例（真实项目，同一口径）", "",
        "> 修复《检查报告》P1-4：不再混用四种概率。下面每个数字都标注了**取数对象**。", "",
        "## 一、这个项目", "",
        f"- 项目：一个 games 项目（标题匿名化，原长 {len(str(proj['name'])) if 'name' in proj else 0} 字符；目标 {proj['m_goal_usd']:.0f} USD / {proj['m_dur_days_exact']:.1f} 天；测试期项目）",
        f"- 归格：**{cell['goal_band']} × {cell['dur_band']}**（右闭：10,000 属 5–10k，不属 10–20k）", "",
        "## 二、四个数字，四个明确的对象", "",
        "| 数字 | 值 | 取数对象（不可混用） |", "| --- | ---: | --- |",
        f"| 该项目的模型概率 P | **{proj['p']*100:.2f}%** | 个体：`pipe.predict_proba(该项目特征)` |",
        f"| 同格模型平均预测 p_cell_mean | {cell['p_cell_mean']*100:.2f}% | 格级：格内全部项目逐条打分取均值 |",
        f"| 同格画像预测 p_profile | {cell['p_profile']*100:.2f}% | 格级：中位数+众数造的**合成画像**（不等于个体，也不等于均值） |",
        f"| 同格历史达标率 rate | **{cell['rate']*100:.2f}%**（95% CI {cell['ci_low']*100:.1f}–{cell['ci_high']*100:.1f}%，n={int(cell['n']):,}） | 观测：该格真实 successful/failed 计数 |", "",
        "## 三、改一个配置（同一画像，只动目标/时长）", "",
        f"- 原配置 {cell['goal_band']}×{cell['dur_band']} 画像预测：**{cell['p_profile']*100:.2f}%**",
        f"- 候选 {alt_goal}×{cell['dur_band']} 画像预测：**{p_alt*100:.2f}%**",
        f"- **ΔP（同画像口径）= {(p_alt-cell['p_profile'])*100:+.1f}pp** —— 这是模型情景分差，非因果收益", "",
        "## 四、钱的两面（不叫期望值）", "",
        f"- 成功时典型募资额 Q50：**${cell['median_payout_usd_success']:,.0f}**（该格成功项目 `usd_pledged` 中位数）",
        f"- 概率加权典型募资额 p_cell_mean × Q50：**${cell['p_weighted_q50_usd']:,.0f}**（**不是** E[Y]）",
        f"- 未实现金额暴露 (1−p_cell_mean) × Q50：**${cell['exposure_usd']:,.0f}**（无成本字段，非期望损失）", "",
        "## 五、一句话", "",
        f"该项目模型概率 {proj['p']*100:.1f}%，同格历史 {cell['rate']*100:.1f}%，"
        f"同格模型均值 {cell['p_cell_mean']*100:.1f}%，画像 {cell['p_profile']*100:.1f}% —— 四者本就不同，"
        "任何引用必须说明取的是哪一个。",
    ]
    (OUT / "S2_案例.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = [
        "# S2 建议引擎（受约束版）", "",
        "> 修复 P1-2 / P1-3。候选顺序：平台合法性 → 目标格支持 → 允许保持原配置。", "",
        f"- cells 总数 **{len(cells)}**，可用（n≥{MIN_CELL_N}）**{len(use)}**",
        f"- 候选时长仅 1–60 天（`61d+` 仅作历史描述）",
        f"- 最优动作为“保持原配置”的格：**{int((use['action']=='保持原配置').sum())}**",
        f"- 最优动作仍为“降到 ≤1k”的格：**{lowest}**（{lowest/len(use)*100:.1f}%）",
        f"- 被拒绝的候选都记在 `rejected` 列（前 4 条/格）", "",
        "## 新指标命名（去掉伪期望）", "",
        "| 列 | 含义 | 不是 |", "| --- | --- | --- |",
        "| `p_cell_mean` | 格内模型预测均值 | 不是该格任何单个项目的概率 |",
        "| `median_payout_usd_success` | 成功时 `usd_pledged` 中位数（Q50） | 不是条件均值 |",
        "| `p_weighted_q50_usd` | p×Q50 的概率加权典型值 | **不是 E[Y]** |",
        "| `exposure_usd` | (1−p)×Q50 的未实现金额 | **不是期望损失**（无成本字段） |",
    ]
    (OUT / "S2_建议引擎.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    json.dump({"cells": len(cells), "usable": int(len(use)), "neg_dp": neg,
               "dur61": n61, "target_under": under, "lowest_goal": lowest,
               "keep_current": int((use["action"] == "保持原配置").sum())},
              open(OUT / "s2_engine_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[done] cells.parquet / decision_axes_v2.csv / S2_建议引擎.md / S2_案例.md", flush=True)


if __name__ == "__main__":
    main()
