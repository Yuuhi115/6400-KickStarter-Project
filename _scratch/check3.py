import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
cols=["succ","goal","yr","pledged","usd_pledged","category_key","state"]
df = pd.read_csv(P, low_memory=False, usecols=cols)
m = df[df.succ.notna()].copy()
bins=[-1,1000,5000,10000,20000,50000,1e12]; labels=["<=1k","1-5k","5-10k","10-20k","20-50k",">50k"]
for cat in ["games"]:
    g=m[m.category_key==cat].copy()
    g["b"]=pd.cut(g.goal,bins=bins,labels=labels)
    t=g.groupby("b",observed=True)["succ"].agg(["count","mean"]).round(4)
    print(cat); print(t.to_string())
    print("median usd_pledged among successes by bin:")
    print(g[g.succ==1].groupby("b",observed=True)["usd_pledged"].median().round(1).to_string())
print()
m["b"]=pd.cut(m.goal,bins=bins,labels=labels)
t=m.groupby("b",observed=True)["succ"].agg(["count","mean"]).round(4)
print("ALL:"); print(t.to_string())
print("ALL median usd_pledged (all projects) by bin:")
print(m.groupby("b",observed=True)["usd_pledged"].median().round(1).to_string())
print("ALL median usd_pledged (successes) by bin:")
print(m[m.succ==1].groupby("b",observed=True)["usd_pledged"].median().round(1).to_string())
