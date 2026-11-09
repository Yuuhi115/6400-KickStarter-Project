import pandas as pd, numpy as np
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
df = pd.read_csv(P, low_memory=False, usecols=["succ","blurb_clean","name_clean","blurb_len","blurb_words","name_len","subcategory_key","country_final","location_type","currency","is_relaunch"])
m=df[df.succ.notna()]
print("modelable",len(m))
for c in ["blurb_clean","name_clean"]:
    s=m[c].astype(str)
    print(c,"empty",int((s.str.strip()=="").sum()),"nan",int(m[c].isna().sum()),"mean chars",round(float(s.str.len().mean()),1))
print("blurb_words:", m.blurb_words.describe().round(1).to_dict())
print("subcategory n:", m.subcategory_key.nunique(), "country n:", m.country_final.nunique(), "location_type n:", m.location_type.nunique(), "currency n:", m.currency.nunique())
print("is_relaunch:", m.is_relaunch.value_counts(dropna=False).to_dict())
print("subcategory missing share:", round(float((m.subcategory_key=="unknown").mean()),4) if (m.subcategory_key=="unknown").any() else "no unknown label")
print(m.subcategory_key.value_counts().head(5).to_dict())
