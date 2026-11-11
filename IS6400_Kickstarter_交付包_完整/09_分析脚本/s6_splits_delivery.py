# -*- coding: utf-8 -*-
"""S6: 生成「数据切分」交付物 —— 切分规则清单 + 行级归属表。

产出：
  11_结果与报告/splits_manifest.json        切分规则的唯一声明（含行数核对）
  11_结果与报告/split_assignment.parquet    行级归属：id / yr / 币种 / 各口径 split
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
FULL = PKG / "08_建模就绪数据" / "ks_model_ready_v1.parquet"

COLS = ["id", "succ", "m_is_modelable", "currency", "yr", "m_split"]

d = pd.read_parquet(FULL, columns=COLS)
d["succ_n"] = pd.to_numeric(d["succ"], errors="coerce")
modelable = d[d["m_is_modelable"].fillna(False) & d["succ_n"].notna()].copy()
modelable["is_usd"] = (modelable["currency"].astype("string").str.strip().str.upper() == "USD")
yr = pd.to_numeric(modelable["yr"], errors="coerce")
modelable["split_ours"] = "train"
modelable.loc[(yr >= 2021) & (yr <= 2022), "split_ours"] = "val"
modelable.loc[yr >= 2023, "split_ours"] = "test"

man = {
    "source_parquet": str(FULL.relative_to(PKG)),
    "source_columns_used": COLS,
    "rows_total_in_parquet": int(len(d)),
    "rows_modelable": int(len(modelable)),
    "filter": "m_is_modelable == True 且 succ 非空（succ = successful/failed 二分类）",
    "rule_ours": {
        "train": "yr <= 2020",
        "val": "2021 <= yr <= 2022",
        "test": "yr >= 2023",
        "note": "val 仅用于选判定阈值（0.39）；模型族与超参为预先设定，未在 val 上搜索",
    },
    "rule_pilot": {
        "train": "yr <= 2019",
        "val": "2020 <= yr <= 2022",
        "test": "yr > 2022",
        "note": "仅用于复刻组员 pilot，测试行与 ours 相同",
    },
    "parquet_m_split": "内置列 m_split 为两段口径：train = yr<=2022 / test = yr>=2023（与上方三段不一致，引用需说明）",
    "counts_ours_full_scope": {
        s: int((modelable["split_ours"] == s).sum()) for s in ("train", "val", "test")
    },
    "counts_ours_usd_scope": {
        s: int(((modelable["split_ours"] == s) & modelable["is_usd"]).sum())
        for s in ("train", "val", "test")
    },
    "counts_currency": {
        "usd_rows_modelable": int(modelable["is_usd"].sum()),
        "non_usd_rows_modelable": int((~modelable["is_usd"]).sum()),
    },
    "parquet_m_split_counts_modelable": {
        str(k): int(v) for k, v in modelable["m_split"].value_counts().items()
    },
}
(OUT / "splits_manifest.json").write_text(
    json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

assign = modelable[["id", "yr", "is_usd", "split_ours", "m_split"]].copy()
assign.to_parquet(OUT / "split_assignment.parquet", index=False)

print(json.dumps(man, ensure_ascii=False, indent=2), flush=True)
print(f"\n[ok] split_assignment.parquet: {len(assign):,} 行, {(OUT / 'split_assignment.parquet').stat().st_size/1e6:.2f} MB", flush=True)
