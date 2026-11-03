# -*- coding: utf-8 -*-
"""Is 49% base rate 'no better than guessing'? Quantify how much signal exists
without any ML: base rate vs majority guess vs a 3-variable lookup table."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P = (r"E:\Workspace\IS6400\group\IS6400_Kickstarter_交付包_完整"
     r"\01_清洗数据\ks_clean.csv")
d = pd.read_csv(P, usecols=["succ", "goal", "dur_days", "category_key",
                            "country_final", "creator_rank"], low_memory=False)
d["succ"] = pd.to_numeric(d["succ"], errors="coerce")
m = d[d["succ"].notna()].copy()
print(f"modelable rows: {len(m):,}   base success rate (this IS the 49%): {m['succ'].mean()*100:.2f}%")

# 1) blind guessing
maj = 1.0 - m["succ"].mean()
print(f"\n[blind] always predict 'failed'  -> accuracy {maj*100:.2f}%   (AUC of a constant = 0.500)")
print("[why it is not 50%] the sample is 49.21/50.79 -> majority guess = 50.79%")

# 2) a 3-variable lookup table (category x goal band x duration bin)
GB = [0, 1000, 5000, 10000, 20000, 50000, np.inf]
DB = [0, 7, 14, 21, 30, 45, 60, 400]
m["gb"] = pd.cut(m["goal"], GB, labels=["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"])
m["db"] = pd.cut(m["dur_days"], DB, labels=["1-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61+"])
cell = m.groupby(["category_key", "gb", "db"], observed=True)["succ"].agg(["size", "mean"])
print(f"\n[cells] category x goal x duration: {len(cell)} cells; "
      f"n>=300: {(cell['size']>=300).sum()} cells covering "
      f"{cell.loc[cell['size']>=300,'size'].sum():,} rows "
      f"({cell.loc[cell['size']>=300,'size'].sum()/len(m)*100:.1f}%)")

ok = cell[cell["size"] >= 300]["mean"]
print("[spread of conditional success rate, n>=300 cells]")
for q in (0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0):
    print(f"   p{int(q*100):>3} = {ok.quantile(q)*100:6.2f}%")

# score each row with its cell's rate (fallback = global rate)
m = m.merge(cell["mean"].rename("cell_rate"), left_on=["category_key", "gb", "db"],
            right_index=True, how="left")
m["cell_rate"] = m["cell_rate"].fillna(m["succ"].mean())

auc = roc_auc_score(m["succ"], m["cell_rate"])
pred = (m["cell_rate"] > 0.5).astype(int)
acc = (pred == m["succ"]).mean()
print(f"\n[3-variable lookup table, no ML at all] AUC = {auc:.4f} | accuracy = {acc*100:.2f}%")
print(f"   vs blind majority guess: accuracy {maj*100:.2f}%, AUC 0.5000")

# 3) what ranking buys: deciles by cell_rate
m["dec"] = pd.qcut(m["cell_rate"], 10, labels=False, duplicates="drop")
lift = m.groupby("dec")["succ"].agg(["size", "mean"])
lift["mean"] = (lift["mean"] * 100).round(2)
print("\n[decile lift by predicted risk] bottom decile -> top decile")
print(lift.rename(columns={"size": "n", "mean": "rate_%"}).to_string())
print(f"\n   top decile success {lift['mean'].iloc[-1]:.1f}% vs bottom decile {lift['mean'].iloc[0]:.1f}%"
      f"  -> {lift['mean'].iloc[-1] - lift['mean'].iloc[0]:.1f} pp spread")

# 4) creator experience adds more signal on top
ladder = m.groupby(m["creator_rank"].clip(upper=11))["succ"].agg(["size", "mean"])
ladder["mean"] = (ladder["mean"] * 100).round(2)
print("\n[creator_rank ladder, modelable]")
print(ladder.rename(columns={"size": "n", "mean": "rate_%"}).to_string())
