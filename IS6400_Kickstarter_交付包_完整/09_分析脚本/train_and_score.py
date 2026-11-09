# -*- coding: utf-8 -*-
"""Train the final HistGB model on our protocol, persist it, and score example configurations.

Scoring needs only the 19 feature columns, so a "new configuration" is just a dict of those.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

import replicate_pilot as R

ROOT = R.ROOT
OUTDIR = os.path.join(ROOT, "11_结果与报告")
MODEL_PATH = os.path.join(OUTDIR, "model_histgb_ours.joblib")
SCHEMA_PATH = os.path.join(OUTDIR, "model_histgb_ours_schema.json")

print("building features ...", flush=True)
df = R.build_frame(R.DATA["full"], use_prep_days=True)
numeric, categorical = df.attrs["numeric"], df.attrs["categorical"]
feats = numeric + categorical
train, val, test = R.make_splits(df, "ours")
print(f"rows: train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

pipe = R.models(numeric, categorical, rf_trees=0)["Gradient Boosting"]
print("fitting HistGB ...", flush=True)
pipe.fit(train[feats], train["succ"])

pv = pipe.predict_proba(val[feats])[:, 1]
thr = R.pick_threshold(val["succ"].values, pv)
test_metrics = R.evaluate(test["succ"].values, pipe.predict_proba(test[feats])[:, 1], thr)
lift = R.top_decile(test["succ"].values, pipe.predict_proba(test[feats])[:, 1])
print("TEST:", json.dumps({k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, (int, float))},
                          ensure_ascii=False))
print("top-decile:", {k: round(v, 4) for k, v in lift.items()}, flush=True)

joblib.dump({"pipeline": pipe, "features": feats, "numeric": numeric,
             "categorical": categorical, "threshold": thr, "metrics": test_metrics},
            MODEL_PATH)
with open(SCHEMA_PATH, "w", encoding="utf-8") as fh:
    json.dump({"features": feats, "numeric": numeric, "categorical": categorical,
               "threshold": thr, "test_metrics": test_metrics, "top_decile": lift},
              fh, ensure_ascii=False, indent=2)
print("saved:", MODEL_PATH, flush=True)

# ---------------- scoring demo -------------------------------------------------
BASE = {
    "m_log_goal_usd": np.log10(5000), "m_dur_days_exact": 30.0, "prep_days": 14.0,
    "yr": 2026, "mon": 6, "launch_quarter": 2, "dow": 2,
    "m_name_len": 35.0, "m_blurb_len": 120.0, "m_blurb_words": 20.0, "name_word_count": 6.0,
    "creator_previous_count": 0.0, "creator_prev_success_rate": np.nan,
    "category_parent_name": "games", "subcategory_key": "tabletop games",
    "country_final": "US", "location_type": "1", "currency": "USD",
}

configs = {
    "基线：Games / $5k / 30天 / 首次发起": {},
    "目标翻倍：$10k": {"m_log_goal_usd": np.log10(10000)},
    "目标大幅提高：$50k": {"m_log_goal_usd": np.log10(50000)},
    "$10k + 时长压到 14 天": {"m_log_goal_usd": np.log10(10000), "m_dur_days_exact": 14.0},
    "$10k + 发起人第 4 次、历史成功率 67%": {"m_log_goal_usd": np.log10(10000),
                                    "creator_previous_count": 3.0,
                                    "creator_prev_success_rate": 2 / 3},
    "$10k + 文案加长（简介 40 词 / 标题 45 字）": {"m_log_goal_usd": np.log10(10000),
                                       "m_blurb_words": 40.0, "m_blurb_len": 240.0,
                                       "m_name_len": 45.0, "name_word_count": 8.0},
    "$10k + 品类换成 Technology": {"m_log_goal_usd": np.log10(10000),
                                "category_parent_name": "technology",
                                "subcategory_key": "apps"},
}

rows = []
for name, override in configs.items():
    row = dict(BASE)
    row.update(override)
    rows.append({"配置": name, **{k: row[k] for k in feats}})

X = pd.DataFrame(rows)[feats]
probs = pipe.predict_proba(X)[:, 1]
out = pd.DataFrame({"配置": list(configs.keys()), "预测达标概率": (probs * 100).round(1)})
print("\n=== 打分演示（HistGB）===\n", out.to_string(index=False), flush=True)
out.to_csv(os.path.join(OUTDIR, "score_demo.csv"), index=False, encoding="utf-8-sig")
print("\nthreshold used:", round(thr, 3))
