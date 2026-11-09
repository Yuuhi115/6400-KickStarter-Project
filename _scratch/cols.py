import pandas as pd
P = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
df = pd.read_csv(P, nrows=3, low_memory=False)
for i,c in enumerate(df.columns):
    print(i, c, "|", df[c].dtype)
