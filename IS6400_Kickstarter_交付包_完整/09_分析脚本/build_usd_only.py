# -*- coding: utf-8 -*-
"""Build the USD-only companion file (appendix robustness), from the model-ready parquet.

Keeps every USD-priced row (including non-modelable ones) so that the same downstream
code works unchanged: filter with m_is_modelable / m_split as usual.
"""
import json
import os

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = (r"E:\Workspace\IS6400\group\IS6400_Kickstarter_交付包_完整"
        r"\08_建模就绪数据")
SRC = os.path.join(BASE, "ks_model_ready_v1.parquet")
OUT = os.path.join(BASE, "ks_model_ready_usdonly.parquet")
REPORT = os.path.join(BASE, "build_report_usdonly.json")

print("reading (filtered pushdown)...", flush=True)
try:
    table = pq.read_table(SRC, filters=[("m_currency_usd", "==", True)])
except Exception as exc:                                    # pragma: no cover
    print("filter pushdown failed, falling back to pandas:", exc, flush=True)
    df = pd.read_parquet(SRC)
    table = df[df["m_currency_usd"].fillna(False)].pipe(
        lambda x: __import__("pyarrow").Table.from_pandas(x, preserve_index=False))

print("rows kept:", table.num_rows, "cols:", table.num_columns, flush=True)
pq.write_table(table, OUT, compression="zstd")
print("wrote", OUT, os.path.getsize(OUT), "B", flush=True)

# ---- verification -----------------------------------------------------------
d = pd.read_parquet(OUT, columns=["currency", "m_currency_usd", "goal", "m_goal_usd",
                                  "m_is_modelable", "m_split", "succ"])
non_usd = int((~d["m_currency_usd"].fillna(False)).sum())
mod = d[d["m_is_modelable"].fillna(False)]
stats = {
    "rows": int(len(d)),
    "cols": int(table.num_columns),
    "non_usd_rows": non_usd,
    "modelable_rows": int(len(mod)),
    "modelable_split": {k: int(v) for k, v in mod["m_split"].value_counts().items()},
    "success_rate_modelable": round(float(pd.to_numeric(mod["succ"], errors="coerce").mean()), 4),
    "goal_usd_equals_goal": bool((d["m_goal_usd"] - d["goal"]).abs().max() < 1e-9),
    "goal_usd_na": int(d["m_goal_usd"].isna().sum()),
    "bytes": os.path.getsize(OUT),
}
print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

bands = [0, 1000, 5000, 10000, 20000, 50000, float("inf")]
labels = ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"]
g = mod[mod["m_goal_usd"].notna()].copy()
g["band"] = pd.cut(g["m_goal_usd"], bands, labels=labels)
tab = g.groupby("band", observed=False)["succ"].agg(["size", "mean"])
tab["mean"] = (tab["mean"] * 100).round(2)
print("\ngoal_usd 六档（USD-only，可建模）:")
print(tab.rename(columns={"size": "n", "mean": "rate_%"}).to_string(), flush=True)
stats["goal_bands"] = {
    str(k): {"n": int(v["size"]), "rate_pct": float(v["mean"])} for k, v in tab.iterrows()
}

with open(REPORT, "w", encoding="utf-8") as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=2)
print("\nreport:", REPORT, flush=True)
assert non_usd == 0, "non-USD rows leaked into the USD-only file"
assert stats["goal_usd_equals_goal"], "USD rows should have goal_usd == goal"
print("ALL CHECKS PASSED", flush=True)
