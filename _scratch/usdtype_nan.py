import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
cols=["succ","state","currency","usd_type","yr","goal","static_usd_rate","fx_rate_invalid","static_usd_rate_invalid","static_usd_rate_zero_usd_fixed","creator_id","collected_at","country_final"]
df = pd.read_csv(P, low_memory=False, usecols=cols)
usd = df[(df.currency=="USD") & df.succ.notna()].copy()
nan_t = usd[usd.usd_type.isna()]
print("NaN usd_type rows:", len(nan_t), "succ rate", round(float(nan_t.succ.mean()),4))
print("by year:", nan_t.yr.value_counts().sort_index().to_dict())
print("goal desc:", nan_t.goal.describe().round(1).to_dict())
print("static_usd_rate values:", nan_t.static_usd_rate.value_counts(dropna=False).head(5).to_dict())
print("fx_invalid:", int(nan_t.fx_rate_invalid.sum()), "static_invalid:", int(nan_t.static_usd_rate_invalid.sum()), "zero_fixed:", int(nan_t.static_usd_rate_zero_usd_fixed.sum()))
print("collected_at batches:", nan_t.collected_at.nunique())
print()
print("ALL USD modelable: static_usd_rate!=1 share:", round(float((usd.static_usd_rate!=1).mean()),4))
print("static_usd_rate NaN share:", round(float(usd.static_usd_rate.isna().mean()),4))
print("fx_rate_invalid count:", int(usd.fx_rate_invalid.sum()), "static_invalid:", int(usd.static_usd_rate_invalid.sum()), "zero_fixed:", int(usd.static_usd_rate_zero_usd_fixed.sum()))
print()
# where do NaN usd_type rows sit vs year totals?
tot = usd.groupby("yr").size()
print("share of NaN by year:", (nan_t.groupby("yr").size()/tot).round(3).to_dict())
