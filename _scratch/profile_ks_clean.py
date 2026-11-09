import pandas as pd, numpy as np, json

PATH = r"E:\个人\港城\courses\IS6400\group project\ks_clean.csv"
df = pd.read_csv(PATH, low_memory=False)
out = {}
out["shape"] = list(df.shape)
out["columns"] = list(df.columns)
out["state_counts"] = df["state"].value_counts().to_dict()
out["succ_counts"] = df["succ"].value_counts(dropna=False).to_dict() if "succ" in df else None
mod = df[df["succ"].notna()]
out["modelable_rows"] = int(len(mod))

# year distribution
g = mod.groupby("yr")["succ"].agg(["count", "mean"])
out["year"] = {int(k): [int(v["count"]), round(float(v["mean"]), 4)] for k, v in g.iterrows()}

# category
g = mod.groupby("category_key")["succ"].agg(["count", "mean"]).sort_values("count", ascending=False)
out["category"] = {str(k): [int(v["count"]), round(float(v["mean"]), 4)] for k, v in g.iterrows()}

# goal bins
bins = [-1, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1e12]
labels = ["<=1k", "1k-5k", "5k-10k", "10k-20k", "20k-50k", "50k-100k", "100k-500k", ">500k"]
mod = mod.copy()
mod["goal_bin"] = pd.cut(mod["goal"], bins=bins, labels=labels)
g = mod.groupby("goal_bin", observed=True)["succ"].agg(["count", "mean"])
out["goal_bin"] = {str(k): [int(v["count"]), round(float(v["mean"]), 4)] for k, v in g.iterrows()}

# duration bins
bins = [0, 7, 14, 21, 30, 45, 60, 90, 400]
labels = ["1-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", ">90"]
mod["dur_bin"] = pd.cut(mod["dur_days"], bins=bins, labels=labels)
g = mod.groupby("dur_bin", observed=True)["succ"].agg(["count", "mean"])
out["dur_bin"] = {str(k): [int(v["count"]), round(float(v["mean"]), 4)] for k, v in g.iterrows()}

# creator rank
g = mod.groupby("creator_rank")["succ"].agg(["count", "mean"])
out["creator_rank_head"] = {int(k): [int(v["count"]), round(float(v["mean"]), 4)] for k, v in g.head(8).iterrows()}

# missingness of key cols in modelable
keys = ["category_key", "subcategory_key", "country_final", "location_type", "currency",
        "log_goal", "dur_days", "blurb_clean", "name_clean", "creator_rank",
        "creator_n_projects", "creator_prev_success_rate", "is_relaunch", "yr", "mon", "dow", "hour"]
out["missing_in_modelable"] = {k: round(float(mod[k].isna().mean()), 4) for k in keys if k in mod}

# country
g = mod.groupby("country_final")["succ"].agg(["count", "mean"]).sort_values("count", ascending=False)
out["country_top"] = {str(k): [int(v["count"]), round(float(v["mean"]), 4)] for k, v in g.head(12).iterrows()}

# overall rate
out["overall_rate"] = round(float(mod["succ"].mean()), 4)
out["train_test_split_2022"] = {
    "train_rows": int((mod["yr"] <= 2022).sum()),
    "train_rate": round(float(mod.loc[mod["yr"] <= 2022, "succ"].mean()), 4),
    "test_rows": int((mod["yr"] >= 2023).sum()),
    "test_rate": round(float(mod.loc[mod["yr"] >= 2023, "succ"].mean()), 4),
}

# usable text
out["text_empty_share"] = {
    "blurb_clean": round(float(mod["blurb_clean"].isna().mean() if "blurb_clean" in mod else np.nan), 4),
}

# durations validity
out["dur_days_desc"] = {k: round(float(v), 2) for k, v in mod["dur_days"].describe().items()}
out["goal_desc"] = {k: round(float(v), 2) for k, v in mod["goal"].describe().items()}
out["goal_le_0"] = int((mod["goal"] <= 0).sum())

print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
