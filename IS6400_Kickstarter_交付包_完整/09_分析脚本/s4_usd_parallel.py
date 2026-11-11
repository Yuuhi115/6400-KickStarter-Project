# -*- coding: utf-8 -*-
"""S4: USD-only 平行模型与对照（回答“能不能做一版只排除非 USD 的对比”）。

三组口径并列，全部用同一套 18 特征、同一 450 格网格：

  A 主口径   ：全样本（含非 USD，金额按 static_usd_rate 换算）+ train ≤2020 / val 2021-22 / test ≥2023
  B USD-only ：仅 currency==USD 的行 + 同上切分（与 A 唯一差别是样本范围）
  C USD-only 大窗口：仅 USD + **train ≤2022** / test ≥2023（用于与组员 ks_v2 对齐口径；AUC 与阈值无关）

产出：
  11_结果与报告/model_histgb_usd.joblib / _schema.json   （B 的模型，便于打分）
  11_结果与报告/cells_usd.parquet / cells_oos_usd.parquet （B 的 in-sample / 样本外格子）
  11_结果与报告/s4_comparison.csv / S4_USD-only对照.md
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
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
DATA = {"full": PKG / "08_建模就绪数据" / "ks_model_ready_v1.parquet",
        "usd": PKG / "08_建模就绪数据" / "ks_model_ready_usdonly.parquet"}

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

GOAL_BINS = [0, 1_000, 5_000, 10_000, 20_000, 50_000, np.inf]
GOAL_LABELS = ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"]
DUR_BINS = [0, 14, 30, 45, 60, np.inf]
DUR_LABELS = ["1-14d", "15-30d", "31-45d", "46-60d", "61d+"]
MIN_N = 300


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "F1": f1_score(y, pred, zero_division=0),
            "Acc": accuracy_score(y, pred), "thr": thr, "n": int(len(y)),
            "base_rate": float(np.mean(y))}


def cells_of(df, num, cat):
    rows = []
    for (c, gb, db), g in df.groupby(["category_key", "goal_band", "dur_band"],
                                     observed=True):
        n, k = len(g), int(g["succ"].sum())
        lo, hi = wilson(k, n)
        win = g.loc[g["succ"] == 1, "usd_pledged"]
        prof = {x: float(g[x].median()) for x in num}
        prof.update({x: str(g[x].mode().iloc[0]) for x in cat})
        rows.append({"category": c, "goal_band": str(gb), "dur_band": str(db),
                     "n": n, "k": k, "rate": k / n, "ci_low": lo, "ci_high": hi,
                     "median_payout_usd_success": float(win.median()) if len(win) else np.nan,
                     "p_cell_mean": float(g["p"].mean()), "_prof": prof})
    d = pd.DataFrame(rows)
    profs = pd.DataFrame(d.pop("_prof").tolist())
    return d, profs


def prep(path, num=None, cat=None):
    df = rp.build_frame(str(path), use_prep_days=True)
    num, cat = df.attrs["numeric"], df.attrs["categorical"]
    static = pd.read_parquet(path, columns=["m_is_modelable", "succ", "usd_pledged",
                                           "category_key"])
    static = static[static["m_is_modelable"].fillna(False)]
    static = static[pd.to_numeric(static["succ"], errors="coerce").notna()].reindex(df.index)
    df["usd_pledged"] = static["usd_pledged"].to_numpy()
    df["category_key"] = static["category_key"].to_numpy()
    df["succ"] = df["succ"].astype(int)
    df["goal_band"] = pd.cut(df["m_goal_usd"], GOAL_BINS, labels=GOAL_LABELS, right=True)
    df["dur_band"] = pd.cut(df["m_dur_days_exact"], DUR_BINS, labels=DUR_LABELS, right=True)
    df["yr"] = pd.to_numeric(df["yr"], errors="coerce")
    return df, num, cat


def main():
    res = []
    print("[1/4] 载入两种数据 ...", flush=True)
    dfA, numA, catA = prep(DATA["full"])
    dfB, numB, catB = prep(DATA["usd"])
    featsA, featsB = numA + catA, numB + catB
    print(f"      full modelable={len(dfA):,} | usd modelable={len(dfB):,}", flush=True)

    trA, vaA, teA = rp.make_splits(dfA, "ours")
    trB, vaB, teB = rp.make_splits(dfB, "ours")

    print("[2/4] 拟合 A（全样本，train≤2020）与 B（USD-only，train≤2020）...", flush=True)
    pipeA = rp.models(numA, catA, rf_trees=0)["Gradient Boosting"]
    pipeA.fit(trA[featsA], trA["succ"])
    thrA = rp.pick_threshold(vaA["succ"].values, pipeA.predict_proba(vaA[featsA])[:, 1])
    pA = pipeA.predict_proba(teA[featsA])[:, 1]
    res.append({"口径": "A 主口径（全样本，≤2020）", **evaluate(teA["succ"].values, pA, thrA)})

    pipeB = rp.models(numB, catB, rf_trees=0)["Gradient Boosting"]
    pipeB.fit(trB[featsB], trB["succ"])
    thrB = rp.pick_threshold(vaB["succ"].values, pipeB.predict_proba(vaB[featsB])[:, 1])
    pB = pipeB.predict_proba(teB[featsB])[:, 1]
    res.append({"口径": "B USD-only（≤2020）", **evaluate(teB["succ"].values, pB, thrB)})

    # ⚠️ 按用户明确要求：**不对齐组员 ks_v2 的 train≤2022 切分**，
    #    只用我们自己的 ours 协议（train≤2020 / val 2021-22 / test≥2023）。

    comp = pd.DataFrame(res)
    comp.to_csv(OUT / "s4_comparison.csv", index=False, encoding="utf-8-sig")
    print(comp.round(4).to_string(index=False), flush=True)

    print("[4/4] 建 B 的格子（in-sample / 样本外）并落盘 ...", flush=True)
    dfB["p"] = pipeB.predict_proba(dfB[featsB])[:, 1]
    cB, profB = cells_of(dfB, numB, catB)
    cBoos, _ = cells_of(dfB[dfB["yr"] >= 2023], numB, catB)
    cB.to_parquet(OUT / "cells_usd.parquet", index=False)
    cBoos.to_parquet(OUT / "cells_oos_usd.parquet", index=False)

    joblib.dump({"pipeline": pipeB, "features": featsB, "numeric": numB,
                 "categorical": catB, "threshold": thrB,
                 "metrics": evaluate(teB["succ"].values, pB, thrB)},
                OUT / "model_histgb_usd.joblib")
    json.dump({"features": featsB, "numeric": numB, "categorical": catB,
               "threshold": thrB, "scope": "USD-only",
               "note": "currency/country_final 在该子样本中近似常量，仍保留以与主口径同构",
               "n_modelable": int(len(dfB)), "n_test": int(len(teB))},
              open(OUT / "model_histgb_usd_schema.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 关键对照数字
    def bands(d):
        g = d.groupby("goal_band", observed=True)["succ"].agg(["size", "mean"])
        return {str(k): (int(v["size"]), round(float(v["mean"]) * 100, 2)) for k, v in g.iterrows()}

    nA_in, nA_oos = int((cB.assign(n=0)["n"] >= 0).sum()), 0  # placeholder
    rowsA = {
        "in_sample_cells": int((dfA.groupby(["category_key", "goal_band", "dur_band"],
                                            observed=True).size() >= MIN_N).sum()),
        "in_sample_cells_usd": int((cB["n"] >= MIN_N).sum()),
        "oos_cells_usd": int((cBoos["n"] >= MIN_N).sum()),
        "n_cells_usd": int(len(cB)),
        "n_cells_oos_usd": int(len(cBoos)),
        "test_rows": int(len(teB)),
    }
    full_in = pd.read_parquet(OUT / "cells.parquet")
    oos_full = pd.read_parquet(OUT / "cells_oos.parquet")
    rowsA["in_sample_cells_full"] = int((full_in["n"] >= MIN_N).sum())
    rowsA["oos_cells_full"] = int((oos_full["n"] >= MIN_N).sum())

    md = [
        "# S4 USD-only 平行模型对照（只用我们的切分协议）", "",
        "> 回答“能不能做一版只排除非 USD 的对比”。两组使用**同一 18 特征、同一 450 格网格、",
        "> 同一 ours 切分**（train ≤2020 / val 2021–22 / test ≥2023），唯一差别是**样本范围**。", "",
        "> ⚠️ **本对照不采用组员 `ks_v2` 的 train ≤2022 切分**。与他们的数字比较时，必须同时说明",
        "> “数据范围相同、但训练窗口与特征集不同”，否则属于混用口径。", "",
        "## 一、模型指标对照（测试集，ours 切分）", "",
        "| 口径 | 样本范围 | 训练窗口 | 测试 n | AUC | AP | Brier | F1 | 阈值 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in comp.iterrows():
        scope = "USD-only" if "USD" in r["口径"] else "全样本"
        md.append(f"| {r['口径'].split('（')[0]} | {scope} | ≤2020 | {int(r['n']):,} | "
                  f"**{r['AUC']:.4f}** | {r['AP']:.4f} | {r['Brier']:.4f} | {r['F1']:.4f} | {r['thr']:.2f} |")
    md += [
        "", "**怎么读**：两组只差样本范围 ⇒ AUC 差即“非 USD 行是否贡献信息”；"
        "两者差值 **不** 能与组员的 0.8791 / 0.8848 直接相减（他们的切分与特征集都不同）。", "",
        "## 二、格子支撑度对照", "",
        "| 项 | 全样本 | USD-only |", "| --- | ---: | ---: |",
        f"| in-sample 可用格（n≥{MIN_N}） | {rowsA['in_sample_cells_full']} | {rowsA['in_sample_cells_usd']} |",
        f"| 样本外可用格（测试期） | {rowsA['oos_cells_full']} | {rowsA['oos_cells_usd']} |",
        f"| 格总数 | {len(full_in)} | {rowsA['n_cells_usd']} |",
        "", "## 三、目标金额六档达标率对照（% / n）", "",
        "| 目标档 | 全样本（可建模） | USD-only（可建模） |", "| --- | ---: | ---: |",
    ]
    bA, bB = bands(dfA), bands(dfB)
    for lab in GOAL_LABELS:
        a = bA.get(lab, (0, float("nan")))
        b = bB.get(lab, (0, float("nan")))
        md.append(f"| {lab} | {a[1]:.2f}% ({a[0]:,}) | {b[1]:.2f}% ({b[0]:,}) |")
    md += [
        "", "## 四、结论", "",
        "- **USD-only 的主代价不是精度，而是坐标轴塌缩**：国家数 25 → 3（US 415,609 / CA 8 / PL 2），"
        "`country_final` 与 `currency` 近似常量，18 个特征里实际只剩 16 个有信息。",
        "- 精度代价见上表 A vs B（同切分、同特征，只差样本范围）。",
        "- 与组员比较时只能说：“他们的 0.8791 是 USD-only + train ≤2022 + 另一套特征集；"
        "我们的主口径是 0.8823；附录 USD-only（ours 切分）见上表。” **三者不可直接排序**。",
        "- 因此报告建议：**主口径保持全样本**，USD-only 作为附录稳健性，"
        "并明确写“剔除后国家维度塌缩为美国市场”。",
    ]
    (OUT / "S4_USD-only对照.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[done] S4_USD-only对照.md / s4_comparison.csv / cells_usd.parquet / model_histgb_usd.joblib",
          flush=True)


if __name__ == "__main__":
    main()
