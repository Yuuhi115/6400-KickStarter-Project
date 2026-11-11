# -*- coding: utf-8 -*-
"""S4b: 干净消融 —— 两个模型 × 两批测试行（2×2）。

A = 全样本训练（model_histgb_ours.joblib）
B = USD-only 训练（model_histgb_usd.joblib）
两批行：全样本测试行(100,469) / USD-only 测试行(54,435)

只在同一批行上比较 AUC/AP/Brier，才能回答“非 USD 训练数据是否真的有用”。
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

objA = joblib.load(OUT / "model_histgb_ours.joblib")
objB = joblib.load(OUT / "model_histgb_usd.joblib")
feats = objA["features"]

df = rp.build_frame(rp.DATA["full"], use_prep_days=True)

# USD-only 的测试行 = 全样本测试行里 currency==USD 的那些
static = pd.read_parquet(rp.DATA["full"], columns=["m_is_modelable", "succ", "currency"])
static = static[static["m_is_modelable"].fillna(False)]
static = static[pd.to_numeric(static["succ"], errors="coerce").notna()].reindex(df.index)
df["is_usd"] = (static["currency"].astype("string").str.strip().str.upper() == "USD").to_numpy()

train, val, test = rp.make_splits(df, "ours")

rows_scope = {
    "全样本测试行 (n≈100,469)": test,
    "仅 USD 测试行 (n≈54,435)": test[test["is_usd"]],
}

recs = []
for row_name, sub in rows_scope.items():
    y = sub["succ"].astype(int).to_numpy()
    for m_name, obj in (("A 全样本训练", objA), ("B USD-only 训练", objB)):
        p = obj["pipeline"].predict_proba(sub[feats])[:, 1]
        recs.append({"测试行": row_name, "模型": m_name, "n": len(y),
                     "基准率": 100 * y.mean(),
                     "AUC": roc_auc_score(y, p),
                     "AP": average_precision_score(y, p),
                     "Brier": brier_score_loss(y, p)})

t = pd.DataFrame(recs)
t.to_csv(OUT / "s4b_ablation_same_rows.csv", index=False, encoding="utf-8-sig")
print(t.round(4).to_string(index=False), flush=True)

# 配对差
for row_name in rows_scope:
    s = t[t["测试行"] == row_name].set_index("模型")
    d_auc = s.loc["A 全样本训练", "AUC"] - s.loc["B USD-only 训练", "AUC"]
    d_ap = s.loc["A 全样本训练", "AP"] - s.loc["B USD-only 训练", "AP"]
    d_br = s.loc["B USD-only 训练", "Brier"] - s.loc["A 全样本训练", "Brier"]
    print(f"\n[{row_name}] A − B: ΔAUC {d_auc*100:+.2f}pp | ΔAP {d_ap*100:+.2f}pp | "
          f"ΔBrier(B−A) {d_br*100:+.2f}pp", flush=True)

# 判读
s_usd = t[t["测试行"] == "仅 USD 测试行 (n≈54,435)"].set_index("模型")
verdict = "A 在 USD 行上更优 ⇒ 非 USD 训练数据有正迁移" \
    if s_usd.loc["A 全样本训练", "AUC"] > s_usd.loc["B USD-only 训练", "AUC"] \
    else "B 在 USD 行上更优 ⇒ 非 USD 训练数据无正迁移"
print("\n判读:", verdict, flush=True)
