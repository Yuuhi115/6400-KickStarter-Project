import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
df = pd.read_csv(P, low_memory=False, usecols=["succ","goal","yr","static_usd_rate","state","goal_capped","goal_under_1"])
m = df[df.succ.notna()].copy()
def tab(g, label, bins, labels):
    x = pd.cut(g, bins=bins, labels=labels)
    t = m.groupby(x, observed=True)["succ"].agg(["count","mean"])
    print(label); print(t.round(4).to_string()); print()
bins=[-1,1000,5000,10000,20000,50000,500000,1e12]; labels=["<=1k","1-5k","5-10k","10-20k","20-50k","50-500k",">500k"]
tab(m.goal,"A full sample goal",bins,labels)
m15=m[m.yr>=2015]
tab(m15.goal,"B yr>=2015 goal",bins,labels)
m15=m[m.yr.between(2015,2022)]
tab(m15.goal,"C yr 2015-2022 goal",bins,labels)
u = m.goal*m.static_usd_rate
tab(u,"D usd goal full",bins,labels)
tab(m.goal_under_1*1.0,"E goal_under_1",[ -0.5,0.5],["no","yes"])
print("goal_under_1 rate:", m.groupby("goal_under_1")["succ"].agg(["count","mean"]).round(4).to_string())
print("<=1k by year:"); print(m[m.goal<=1000].groupby("yr")["succ"].agg(["count","mean"]).round(4).tail(12).to_string())
