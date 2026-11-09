import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
cols=["succ","currency","usd_type","goal","log_goal","yr","category_key","country_final","dur_days","creator_prev_success_rate","creator_rank","is_terminal_pair","state","static_usd_rate","goal_capped"]
df = pd.read_csv(P, low_memory=False, usecols=cols)
print("usd_type:", df.usd_type.value_counts(dropna=False).to_dict())
print()
print("currency counts & succ (all rows):")
t = df.groupby("currency").agg(n=("succ","size"), succ_n=("succ","count"), rate=("succ","mean")).sort_values("n", ascending=False)
print(t.round(4).to_string())
print()
m = df[df.succ.notna()]
usd = m[m.currency=="USD"]
print("modelable all:", len(m), "rate", round(float(m.succ.mean()),4))
print("modelable USD:", len(usd), "rate", round(float(usd.succ.mean()),4))
print()
print("USD per-year:")
print(usd.groupby("yr")["succ"].agg(n="size", rate="mean").round(4).to_string())
print()
tr = usd[usd.yr<=2022]; te = usd[usd.yr>=2023]
print("train:", len(tr), round(float(tr.succ.mean()),4), " test:", len(te), round(float(te.succ.mean()),4))
print()
print("USD category:")
print(usd.groupby("category_key")["succ"].agg(n="size", rate="mean").round(4).sort_values("rate").to_string())
print()
print("USD country final top12:")
print(usd.country_final.value_counts().head(12).to_dict())
print()
print("USD goal desc:", usd.goal.describe().round(1).to_dict())
print("USD prev_success_rate missing:", round(float(usd.creator_prev_success_rate.isna().mean()),4))
print("USD dur median:", usd.dur_days.median(), "max", usd.dur_days.max())
print()
# does dropping non-USD change time drift?
print("full-sample years 2015/2020/2024:", [round(float(m[m.yr==y].succ.mean()),4) for y in (2015,2020,2024)])
print("usd-sample years 2015/2020/2024:", [round(float(usd[usd.yr==y].succ.mean()),4) for y in (2015,2020,2024)])
print()
print("non-USD share by year (modelable):")
tmp = m.assign(isusd=(m.currency=="USD"))
print(tmp.groupby("yr")["isusd"].mean().round(3).to_string())
