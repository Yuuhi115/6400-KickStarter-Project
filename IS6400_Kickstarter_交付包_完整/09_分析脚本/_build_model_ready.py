# -*- coding: utf-8 -*-
"""Build a model-ready dataset from the 81-column cleaned CSV.

Adds (never overwrites) a `m_` block:
  m_goal_usd / m_goal_usd_ok / m_goal_usd_filled / m_log_goal_usd   currency-normalised goal
  m_currency_usd                                1 if the campaign is priced in USD
  m_dur_days_exact                              duration in days with fraction (no .dt.days truncation)
  m_name_len / m_blurb_len / m_blurb_words      NA-aware lengths from the *cleaned* text
  m_split                                       'train' (yr<=2022) / 'test' (yr>=2023)
  m_is_modelable                                successful+failed pair (== is_terminal_pair)

Keeps every original column (incl. the 10 post-outcome ones) so nothing is lost;
the leakage columns must be excluded by the modelling script, not here.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = (r"E:\Workspace\IS6400\group\IS6400_Kickstarter_交付包_完整"
       r"\01_清洗数据\ks_clean.csv")
OUTDIR = (r"E:\Workspace\IS6400\group\IS6400_Kickstarter_交付包_完整"
          r"\08_建模就绪数据")
OUT = os.path.join(OUTDIR, "ks_model_ready_v1.parquet")
REPORT = os.path.join(OUTDIR, "build_report.json")
CHUNK = 100_000


def transform(chunk):
    out = chunk.copy()
    goal = pd.to_numeric(out["goal"], errors="coerce")
    rate = pd.to_numeric(out["static_usd_rate"], errors="coerce")
    usd = out["currency"].astype("string").str.strip().str.upper().eq("USD").fillna(False)
    ok = rate.gt(0) & rate.notna() & np.isfinite(rate) & goal.notna()
    out["m_goal_usd"] = (goal * rate).where(ok)
    # A USD campaign is USD by definition: where only the rate is missing, goal_usd == goal (exact, not an estimate).
    fill = usd & out["m_goal_usd"].isna() & goal.notna()
    out["m_goal_usd"] = out["m_goal_usd"].where(~fill, goal)
    out["m_goal_usd_ok"] = (ok | fill).astype("boolean")
    out["m_goal_usd_filled"] = fill.astype("boolean")
    out["m_currency_usd"] = usd.astype("boolean")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["m_log_goal_usd"] = np.log10(out["m_goal_usd"].where(out["m_goal_usd"] > 0))

    ldt = pd.to_datetime(out["launch_dt"], errors="coerce")
    ddt = pd.to_datetime(out["deadline_dt"], errors="coerce")
    out["m_dur_days_exact"] = (ddt - ldt).dt.total_seconds() / 86400.0

    name_s = out["name_clean"].astype("string")
    blurb_s = out["blurb_clean"].astype("string")
    out["m_name_len"] = name_s.str.len().astype("Int64")
    out["m_blurb_len"] = blurb_s.str.len().astype("Int64")
    out["m_blurb_words"] = blurb_s.str.split().str.len().astype("Int64")

    yr = pd.to_numeric(out["yr"], errors="coerce")
    out["m_split"] = np.where(yr <= 2022, "train", "test")
    out["m_is_modelable"] = out["is_terminal_pair"].astype("boolean")
    return out.convert_dtypes()


os.makedirs(OUTDIR, exist_ok=True)
writer = None
schema = None
rows = 0
acc = {"rows": 0, "m_goal_usd_ok": 0, "m_goal_usd_na": 0, "m_goal_usd_filled": 0,
       "m_currency_usd": 0, "m_name_len_na": 0, "m_blurb_len_na": 0,
       "train": 0, "test": 0, "modelable": 0, "goal_usd_sum": 0.0,
       "goal_usd_min": None, "goal_usd_max": None}

for i, chunk in enumerate(pd.read_csv(SRC, chunksize=CHUNK, low_memory=False)):
    out = transform(chunk)
    table = pa.Table.from_pandas(out, preserve_index=False)
    if writer is None:
        schema = table.schema
        writer = pq.ParquetWriter(OUT, schema, compression="zstd")
    else:
        table = table.cast(schema)
    writer.write_table(table)

    rows += len(out)
    acc["rows"] += len(out)
    ok = out["m_goal_usd_ok"].fillna(False).to_numpy()
    gu = out["m_goal_usd"]
    acc["m_goal_usd_ok"] += int(ok.sum())
    acc["m_goal_usd_na"] += int(out["m_goal_usd"].isna().sum())
    acc["m_goal_usd_filled"] += int(out["m_goal_usd_filled"].fillna(False).sum())
    acc["m_currency_usd"] += int(out["m_currency_usd"].fillna(False).sum())
    acc["m_name_len_na"] += int(out["m_name_len"].isna().sum())
    acc["m_blurb_len_na"] += int(out["m_blurb_len"].isna().sum())
    acc["train"] += int((out["m_split"] == "train").sum())
    acc["test"] += int((out["m_split"] == "test").sum())
    acc["modelable"] += int(out["m_is_modelable"].fillna(False).sum())
    acc["goal_usd_sum"] += float(gu.sum(skipna=True))
    mn, mx = gu.min(skipna=True), gu.max(skipna=True)
    if acc["goal_usd_min"] is None or (pd.notna(mn) and mn < acc["goal_usd_min"]):
        acc["goal_usd_min"] = float(mn) if pd.notna(mn) else acc["goal_usd_min"]
    if acc["goal_usd_max"] is None or (pd.notna(mx) and mx > acc["goal_usd_max"]):
        acc["goal_usd_max"] = float(mx) if pd.notna(mx) else acc["goal_usd_max"]
    print(f"chunk {i}: rows={len(out):,} total={rows:,} cols={out.shape[1]}", flush=True)

writer.close()

acc["mean_goal_usd"] = round(acc.pop("goal_usd_sum") / max(acc["m_goal_usd_ok"], 1), 2)
acc["output"] = OUT
acc["output_bytes"] = os.path.getsize(OUT)
acc["columns"] = len(schema.names)
with open(REPORT, "w", encoding="utf-8") as fh:
    json.dump(acc, fh, ensure_ascii=False, indent=2)
print(json.dumps(acc, ensure_ascii=False, indent=2))
