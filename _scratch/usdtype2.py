import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
cols=["succ","state","currency","usd_type","yr","goal","pledged","usd_pledged","backers_count","percent_funded","static_usd_rate","creator_id","id"]
df = pd.read_csv(P, low_memory=False, usecols=cols)
usd = df[(df.currency=="USD") & df.succ.notna()].copy()
a = usd[usd.usd_type.isna()]; b = usd[usd.usd_type.notna()]
for name,x in [("NaN_usd_type",a),("normal",b)]:
    print(name, "n=",len(x))
    print("  pledged median/mean:", round(float(x.pledged.median()),1), round(float(x.pledged.mean()),1))
    print("  usd_pledged median/mean:", round(float(x.usd_pledged.median()),1), round(float(x.usd_pledged.mean()),1))
    print("  backers median:", float(x.backers_count.median()), " share backers==0:", round(float((x.backers_count==0).mean()),4))
    print("  share pledged==0:", round(float((x.pledged==0).mean()),4))
    print("  goal median:", round(float(x.goal.median()),1))
    print("  pct_funded median:", round(float(x.percent_funded.median()),4))
    print()
r = a[(a.pledged>0)]
print("ratio usd_pledged/pledged (NaN grp):", (r.usd_pledged/r.pledged).describe().round(3).to_dict())
print()
print("dup creator share in NaN grp:", round(float(a.duplicated('creator_id').mean()),4), " normal:", round(float(b.duplicated('creator_id').mean()),4))
print("id range NaN grp:", int(a.id.min()), int(a.id.max()), "normal:", int(b.id.min()), int(b.id.max()))
