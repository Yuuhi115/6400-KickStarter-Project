import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
cols=["succ","state","currency","usd_type","country","location_country","country_final","location_state","location_type","yr"]
df = pd.read_csv(P, low_memory=False, usecols=cols)
usd = df[df.currency=="USD"]
print("USD rows:", len(usd))
print("state in USD:", usd.state.value_counts().to_dict())
print()
print("usd_type in USD:", usd.usd_type.value_counts(dropna=False).to_dict())
print()
print("original country top15 in USD:", usd.country.value_counts().head(15).to_dict())
print("location_country top10 in USD:", usd.location_country.value_counts().head(10).to_dict())
print("country_final top5:", usd.country_final.value_counts().head(5).to_dict())
print()
u = usd[usd.succ.notna()]
print("modelable USD state:", u.state.value_counts().to_dict())
print("location_state nunique (USD modelable):", u.location_state.nunique())
print("location_state top10:", u.location_state.value_counts().head(10).to_dict())
print("location_type:", u.location_type.value_counts(dropna=False).to_dict())
print()
print("succ by usd_type (USD modelable):")
print(u.groupby("usd_type", dropna=False)["succ"].agg(n="size", rate="mean").round(4).to_string())
print()
print("succ by original country (USD modelable, top8):")
g = u.groupby("country")["succ"].agg(n="size", rate="mean").sort_values("n", ascending=False).head(8)
print(g.round(4).to_string())
