import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
cols=["succ","currency","goal","usd_pledged","dur_days","creator_rank","yr","category_key"]
df = pd.read_csv(P, low_memory=False, usecols=cols)
u = df[(df.currency=="USD") & df.succ.notna()].copy()
gb=[-1,1000,5000,10000,20000,50000,100000,500000,1e12]
lab=["<=1k","1-5k","5-10k","10-20k","20-50k","50-100k","100-500k",">500k"]
u["g"]=pd.cut(u.goal,gb,labels=lab)
print("goal bins (USD-only, all categories):")
print(u.groupby("g",observed=True)["succ"].agg(n="size", rate="mean").round(4).to_string())
print("success median usd_pledged by bin:")
print(u[u.succ==1].groupby("g",observed=True)["usd_pledged"].median().round(0).to_string())
print("all-project median usd_pledged by bin:")
print(u.groupby("g",observed=True)["usd_pledged"].median().round(0).to_string())
print()
db=[0,7,14,21,30,45,60,90,1000]; dl=["1-7","8-14","15-21","22-30","31-45","46-60","61-90",">90"]
u["d"]=pd.cut(u.dur_days,db,labels=dl)
print("duration bins:"); print(u.groupby("d",observed=True)["succ"].agg(n="size", rate="mean").round(4).to_string())
print()
rc=[0,1,2,3,4,5,8,1e9]; rl=["1","2","3","4","5","6-8",">8"]
u["r"]=pd.cut(u.creator_rank,rc,labels=rl)
print("creator_rank bins:"); print(u.groupby("r",observed=True)["succ"].agg(n="size", rate="mean").round(4).to_string())
print()
print("time split:", len(u[u.yr<=2022]), round(float(u[u.yr<=2022].succ.mean()),4), len(u[u.yr>=2023]), round(float(u[u.yr>=2023].succ.mean()),4))
