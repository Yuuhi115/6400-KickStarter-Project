# -*- coding: utf-8 -*-
"""What does dropping non-USD rows actually cost? Compare: full / converted / USD-only."""
import io
import zipfile

import numpy as np
import pandas as pd

ZIP = (r"D:\dsh-data\attachments\v1\files\98"
       r"\98d570d5e702e9b6babb682db423c6d11f6d3171e8884b4b52e657bb004bb2d1"
       r"\IS6400_Kickstarter_交付包_完整v1.zip")
MEMBER = "IS6400_Kickstarter_交付包_完整/01_清洗数据/ks_clean.csv"

z = zipfile.ZipFile(ZIP)
with z.open(MEMBER) as fh:
    d = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"),
                    usecols=["succ", "is_terminal_pair", "currency", "goal",
                             "static_usd_rate", "category_key", "country_final"],
                    low_memory=False)
d["succ"] = pd.to_numeric(d["succ"], errors="coerce")
d["goal"] = pd.to_numeric(d["goal"], errors="coerce")
d["rate"] = pd.to_numeric(d["static_usd_rate"], errors="coerce")
d["cur"] = d["currency"].astype("string").str.strip().str.upper()

m = d[d["succ"].notna()].copy()
m["goal_usd"] = m["goal"] * m["rate"]
usd = m[m["cur"] == "USD"]

BANDS, LAB = [0, 1000, 5000, 10000, 20000, 50000, np.inf], ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"]


def curve(frame, col):
    x = frame[frame[col].notna()].copy()
    x["b"] = pd.cut(x[col], bins=BANDS, labels=LAB)
    g = x.groupby("b", observed=False)["succ"].agg(["size", "mean"])
    return g


print(f"modelable total          : {len(m):,}  success {m['succ'].mean()*100:.2f}%")
print(f"non-USD rows             : {(m['cur']!='USD').sum():,} ({(m['cur']!='USD').mean()*100:.2f}%)")
print(f"USD-only subset          : {len(usd):,}  success {usd['succ'].mean()*100:.2f}%")
print(f"convertible (rate>0)     : {m['rate'].gt(0).sum():,} ({m['rate'].gt(0).mean()*100:.2f}%)")

print("\n=== count of distinct countries ===")
print("all modelable :", m["country_final"].nunique())
print("USD subset    :", usd["country_final"].nunique())
print("USD subset top countries:",
      usd["country_final"].value_counts().head(6).to_dict())

print("\n=== category coverage ===")
print("all modelable :", m["category_key"].nunique(), "categories")
print("USD subset    :", usd["category_key"].nunique(), "categories")
share_all = (m["category_key"].value_counts(normalize=True) * 100).round(2)
share_usd = (usd["category_key"].value_counts(normalize=True) * 100).round(2)
cmp = pd.DataFrame({"all_%": share_all, "usd_only_%": share_usd}).fillna(0)
cmp["shift_pp"] = (cmp["usd_only_%"] - cmp["all_%"]).round(2)
print(cmp.sort_values("shift_pp").to_string())

print("\n=== goal curve: USD-only (local == USD) ===")
c1 = curve(usd, "goal")
c1["rate_%"] = (c1["mean"] * 100).round(2)
print(c1.rename(columns={"size": "n"})[["n", "rate_%"]].to_string())

print("\n=== goal curve: converted goal_usd, all convertible rows ===")
c2 = curve(m, "goal_usd")
c2["rate_%"] = (c2["mean"] * 100).round(2)
print(c2.rename(columns={"size": "n"})[["n", "rate_%"]].to_string())
