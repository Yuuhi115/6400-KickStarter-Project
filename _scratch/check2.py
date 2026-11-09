import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
df = pd.read_csv(P, low_memory=False, usecols=["succ","goal","yr","goal_capped","goal_under_1","static_usd_rate","creator_history_scope","creator_prev_success_rate","creator_rank","category_key"])
m = df[df.succ.notna()].copy()
print("goal_capped dtype/uniq:", m.goal_capped.dtype, m.goal_capped.nunique())
print("goal_under_1 value counts:", m.goal_under_1.value_counts(dropna=False).to_dict())
print("goal_under_1 by succ:", m.groupby("goal_under_1")["succ"].agg(["count","mean"]).round(4).to_string())
print()
print("creator_history_scope:", m.creator_history_scope.value_counts(dropna=False).to_dict())
print("prev_success_rate desc:", m.creator_prev_success_rate.describe().round(3).to_dict())
print("rows with prev_rate notna:", int(m.creator_prev_success_rate.notna().sum()))
print()
print("goal quantiles of tiny goals:", m.goal[m.goal<5].value_counts().head(8).to_dict())
print()
# maybe 84.4% is Games + goal<=1k in 2023+
g = m[(m.category_key=="games")&(m.goal<=1000)]
print("games <=1k overall:", round(float(g.succ.mean()),4), len(g))
print("games <=1k 2023+:", round(float(g[g.yr>=2023].succ.mean()),4), len(g[g.yr>=2023]))
print()
# maybe docs used only rows with goal within 1-1000000 after cleaning in package?
print("state succ mean by goal<=1000 year>=2020:", round(float(m[(m.yr>=2020)&(m.goal<=1000)].succ.mean()),4))
