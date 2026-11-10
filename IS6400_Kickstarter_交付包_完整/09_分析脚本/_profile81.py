# -*- coding: utf-8 -*-
"""Fresh profile of the CURRENT 81-column cleaned data (post-revision)."""
import numpy as np
import pandas as pd
import os
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
P = os.path.join(_ROOT, "01_清洗数据", "ks_clean.csv")
COLS = ["state", "succ", "is_terminal_pair", "goal", "log_goal", "dur_days", "yr",
        "category_key", "subcategory_key", "country_final", "currency",
        "name", "blurb", "name_len", "blurb_len", "blurb_words",
        "creator_rank", "creator_prev_success_rate", "is_relaunch",
        "percent_funded", "pledged", "backers_count"]

d = pd.read_csv(P, usecols=COLS, low_memory=False)
print("shape:", d.shape)
print("state:", d["state"].value_counts().to_dict())

term = d[d["state"].isin(["successful", "failed"])].copy()
term["succ"] = pd.to_numeric(term["succ"], errors="coerce")
print("\nmodelable:", len(term), "success rate: %.4f" % term["succ"].mean())

print("\n--- missing rate (modelable subset) ---")
for c in ["name", "blurb", "subcategory_key", "creator_prev_success_rate", "percent_funded"]:
    print(f"  {c:28s} {term[c].isna().mean()*100:6.2f}%  (n_missing={int(term[c].isna().sum()):,})")

print("\n--- success rate by duration bin ---")
bins = [0, 7, 14, 21, 30, 45, 60, 400]
labs = ["1-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61+"]
term["db"] = pd.cut(term["dur_days"], bins=bins, labels=labs)
g = term.groupby("db", observed=False)["succ"].agg(["size", "mean"])
g["mean"] = (g["mean"] * 100).round(2)
print(g.rename(columns={"size": "n", "mean": "rate_%"}).to_string())

print("\n--- success rate by category (15) ---")
c = term.groupby("category_key")["succ"].agg(["size", "mean"]).sort_values("mean", ascending=False)
c["mean"] = (c["mean"] * 100).round(2)
print(c.rename(columns={"size": "n", "mean": "rate_%"}).to_string())

print("\n--- success rate by launch year ---")
y = term.groupby("yr")["succ"].agg(["size", "mean"])
y["mean"] = (y["mean"] * 100).round(2)
print(y.rename(columns={"size": "n", "mean": "rate_%"}).to_string())

print("\n--- creator experience ladder (creator_rank) ---")
term["exp"] = np.where(term["creator_rank"] == 1, "1st",
                       np.where(term["creator_rank"] <= 3, "2-3rd",
                                np.where(term["creator_rank"] <= 10, "4-10th", "11th+")))
e = term.groupby("exp", observed=False)["succ"].agg(["size", "mean"])
e["mean"] = (e["mean"] * 100).round(2)
print(e.rename(columns={"size": "n", "mean": "rate_%"}).to_string())

print("\n--- goal band (local currency) vs USD ---")
BANDS, LAB = [0, 1000, 5000, 10000, 20000, 50000, np.inf], ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"]
d2 = d[d["state"].isin(["successful", "failed"])].copy()
d2["succ"] = pd.to_numeric(d2["succ"], errors="coerce")
print(pd.crosstab(pd.cut(d2["goal"], bins=BANDS, labels=LAB), d2["succ"], normalize="index").round(4).to_string())
print("\npercent_funded missing:", int(term["percent_funded"].isna().sum()))
