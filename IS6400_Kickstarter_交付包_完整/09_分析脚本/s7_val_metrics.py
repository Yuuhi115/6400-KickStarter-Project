# -*- coding: utf-8 -*-
"""S7: 验证集指标 + 用验证集做模型选择（修 P1-6）。

背景：此前 val(2021–22, 60,467) 只用于选判定阈值，所有指标都报 test；
      P1-6 指出“应当用验证集选模型、test 只评一次”。

本脚本：读取已落盘的六个模型，在 **val** 上算 AUC/AP/Brier/Acc/F1（含阈值），
并在 val 上做模型排序，与 test 上的排序对照。

产出：11_结果与报告/s7_val_metrics.csv、S7_验证集指标与选型.md
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
MDIR = OUT / "models"

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

MODELS = [("model_lr_full.joblib", "lr", "full"), ("model_lr_usd.joblib", "lr", "usd"),
          ("model_rf_full.joblib", "rf", "full"), ("model_rf_usd.joblib", "rf", "usd"),
          ("model_hgb_full.joblib", "hgb", "full"), ("model_hgb_usd.joblib", "hgb", "usd")]
LAB = {"lr": "LR", "rf": "RF(150)", "hgb": "HistGB"}


def prep(path):
    df = rp.build_frame(str(path), use_prep_days=True)
    static = pd.read_parquet(path, columns=["m_is_modelable", "succ", "currency"])
    static = static[static["m_is_modelable"].fillna(False)]
    static = static[pd.to_numeric(static["succ"], errors="coerce").notna()].reindex(df.index)
    df["is_usd"] = (static["currency"].astype("string").str.strip().str.upper() == "USD").to_numpy()
    df["succ"] = df["succ"].astype(int)
    return df, df.attrs["numeric"] + df.attrs["categorical"]


def main():
    dfF, featsF = prep(rp.DATA["full"])
    trF, vaF, teF = rp.make_splits(dfF, "ours")
    val_all = vaF
    val_usd = vaF[vaF["is_usd"]]
    print(f"val_all={len(val_all):,} val_usd={len(val_usd):,}", flush=True)

    rows = []
    for fname, short, scope in MODELS:
        p = MDIR / fname
        if not p.exists():
            print(f"[skip] {fname} 不存在", flush=True)
            continue
        obj = joblib.load(p)
        pipe, feats, thr = obj["pipeline"], obj["features"], obj["threshold"]
        for rname, sub in (("val_all", val_all), ("val_usd", val_usd)):
            y = sub["succ"].to_numpy()
            prob = pipe.predict_proba(sub[feats])[:, 1]
            pred = (prob >= thr).astype(int)
            rows.append({"scope": scope, "algo": short, "val_rows": rname, "n": len(y),
                         "base_rate": y.mean(), "thr": thr,
                         "AUC": roc_auc_score(y, prob),
                         "AP": average_precision_score(y, prob),
                         "Brier": brier_score_loss(y, prob),
                         "Acc": accuracy_score(y, pred),
                         "F1": f1_score(y, pred, zero_division=0),
                         "Precision": precision_score(y, pred, zero_division=0),
                         "Recall": recall_score(y, pred, zero_division=0)})

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s7_val_metrics.csv", index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False), flush=True)

    # 选型：只用 val_all 的 AUC（不看 test）
    sel = t[t["val_rows"] == "val_all"].sort_values("AUC", ascending=False)
    best = sel.iloc[0]
    # 与 test 排序对照
    mat = OUT / "s5_model_scope_matrix.csv"
    test_best = None
    if mat.exists():
        m = pd.read_csv(mat, encoding="utf-8-sig")
        m = m[m["test_rows"] == "all_test"].sort_values("AUC", ascending=False)
        test_best = m.iloc[0]

    L = ["# S7 验证集指标与选型（修 P1-6）", "",
         "> 此前 val(2021–22, 60,467) 只用于选阈值；本表补上验证集全指标，**并只用 val 做模型排序**。",
         "> 判定阈值取自各模型训练时的 val 选阈（存于 schema），此处直接复用。", "",
         "## 一、验证集指标", "",
         "| 模型 | 口径 | 评估行 | n | AUC | AP | Brier | Acc | F1 | 阈值 |",
         "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in t.iterrows():
        L.append(f"| {LAB[r['algo']]} | {r['scope']} | {r['val_rows']} | {r['n']:,} | "
                 f"{r['AUC']:.4f} | {r['AP']:.4f} | {r['Brier']:.4f} | {r['Acc']:.4f} | "
                 f"{r['F1']:.4f} | {r['thr']:.3f} |")

    L += ["", "## 二、只用验证集选型（P1-6 的正确做法）", "",
          f"- 按 **val_all AUC** 排序第一名：**{LAB[best['algo']]} / {best['scope']}**，"
          f"AUC **{best['AUC']:.4f}**、Acc {best['Acc']:.4f}、F1 {best['F1']:.4f}"]
    if test_best is not None:
        L.append(f"- 此前按 **test** 排序的第一名：{LAB[test_best['algo']]} / {test_best['scope']} "
                 f"（AUC {test_best['AUC']:.4f}）")
        same = (best["algo"], best["scope"]) == (test_best["algo"], test_best["scope"])
        L.append(f"- 两者{'**一致**' if same else '**不一致**'} ⇒ "
                 + ("结论稳健：用 val 选出的模型与用 test 选出的相同，"
                    "因此可以把选型口径正式改为 val，test 只评一次。"
                    if same else
                    "存在分歧，必须在报告中采用 val 的结论，并说明 test 选型属违规。"))
    L += ["", "## 三、验证集 vs 测试集对照（同一模型）", "",
          "| 模型 | 口径 | val AUC | test AUC | 差 |", "| --- | --- | ---: | ---: | ---: |"]
    if mat.exists():
        for _, r in t[t["val_rows"] == "val_all"].iterrows():
            hit = m[(m["algo"] == r["algo"]) & (m["scope"] == r["scope"])]
            if len(hit):
                tv = hit.iloc[0]["AUC"]
                L.append(f"| {LAB[r['algo']]} | {r['scope']} | {r['AUC']:.4f} | {tv:.4f} | "
                         f"{(tv - r['AUC'])*100:+.2f}pp |")
    L += ["", "> 注意：验证集（2021–22）比测试集（≥2023）离训练期更近，且基准率不同"
              "（val 约 61% vs test 约 64%），因此 val 指标不能代替 test 作为最终评估。"]
    (OUT / "S7_验证集指标与选型.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n[done] s7_val_metrics.csv / S7_验证集指标与选型.md", flush=True)


if __name__ == "__main__":
    main()
