# -*- coding: utf-8 -*-
"""Reproduce the delivered HistGB (18 features) from the raw cleaned CSV.

Purpose: the package ships no modelling data and its joblib cannot be loaded
under sklearn 1.5.1, so this script rebuilds the 18 features from
..\\ks_clean.csv (read-only) and reruns the package's split / hyperparameters.

Result (this machine, 2026-11-09, printed by the run below):
    rows 604629 / train 443693 / val 60467 / test 100469
    test AUC 0.882180 / PR-AUC 0.929543 / Brier 0.131651 / F1 0.857232
    val-tuned threshold 0.4150 / top-10% success 0.992635 (lift 1.5411)
vs the delivered 0.8823 / 0.9295 / 0.1316 / 0.8564 / 0.39 / 0.9924 -- matches
within noise. An earlier ad-hoc run of the same protocol printed 0.8825 /
0.9297 / 0.1315 / 0.8577 / 0.43; the difference is run-to-run variation in
the HistGB early-stopping split, not a protocol change.

Writes nothing except stdout. Run:  python repro_18feat.py
"""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (brier_score_loss, f1_score, roc_auc_score,
                             average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path

# package / 12_审查与文本层 / _scratch -> repo root above "KS project"
RAW = str(Path(__file__).resolve().parents[4] / "ks_clean.csv")
COLS = [
    "id", "name", "name_clean", "blurb_clean", "category_parent_name",
    "subcategory_key", "country_final", "location_type", "currency",
    "goal", "static_usd_rate", "launched_at", "deadline_dt", "created_at",
    "yr", "mon", "dow", "creator_rank", "creator_prev_success_rate",
    "is_terminal_pair", "succ",
]
NUMERIC = [
    "m_log_goal_usd", "m_dur_days_exact", "launch_quarter", "dow",
    "m_name_len", "m_blurb_len", "m_blurb_words", "name_word_count",
    "creator_previous_count", "creator_prev_success_rate", "prep_days",
    "yr", "mon",
]
CATEGORICAL = ["category_parent_name", "subcategory_key", "country_final",
               "location_type", "currency"]


def build_frame(path=RAW):
    df = pd.read_csv(path, usecols=COLS, low_memory=False)
    df = df[df["is_terminal_pair"].fillna(False) == True].copy()  # noqa: E712
    df["succ"] = df["succ"].astype(int)

    launched = pd.to_datetime(df["launched_at"], unit="s", errors="coerce")
    created = pd.to_datetime(pd.to_numeric(df["created_at"], errors="coerce"),
                             unit="s", errors="coerce")
    deadline = pd.to_datetime(df["deadline_dt"], errors="coerce")
    df["prep_days"] = (launched - created).dt.total_seconds() / 86400.0
    df["m_dur_days_exact"] = (deadline - launched).dt.total_seconds() / 86400.0

    rate = pd.to_numeric(df["static_usd_rate"], errors="coerce")
    goal = pd.to_numeric(df["goal"], errors="coerce")
    usd = np.where((rate > 0) & np.isfinite(rate), goal * rate, goal)
    df["m_log_goal_usd"] = np.log10(np.clip(usd, 1, None))

    df["launch_quarter"] = (df["mon"].astype(int) - 1) // 3 + 1
    name = df["name_clean"].astype("string")
    blurb = df["blurb_clean"].astype("string")
    df["m_name_len"] = pd.to_numeric(name.str.len())
    df["m_blurb_len"] = pd.to_numeric(blurb.str.len())
    df["m_blurb_words"] = pd.to_numeric(blurb.str.split().str.len())
    # replicate_pilot.py:87 uses the raw `name` for word count (kept for parity)
    df["name_word_count"] = pd.to_numeric(
        df["name"].astype("string").str.split().str.len())
    df["creator_previous_count"] = (pd.to_numeric(df["creator_rank"],
                                                  errors="coerce") - 1).clip(lower=0)
    df["creator_prev_success_rate"] = pd.to_numeric(
        df["creator_prev_success_rate"], errors="coerce")
    df["yr"] = df["yr"].astype(int)
    return df


def make_splits(df):
    train = df[df["yr"] <= 2020]
    val = df[df["yr"].between(2021, 2022)]
    test = df[df["yr"] >= 2023]
    return train, val, test


def preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant",
                                          fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                          min_frequency=20,
                                          sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC),
                              ("cat", cat, CATEGORICAL)])


def model():
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.10,
        n_iter_no_change=20, random_state=42)


def pick_threshold(y, p):
    best, thr = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y, (p >= t).astype(int))
        if f > best:
            best, thr = f, float(t)
    return thr


def main():
    t0 = time.time()
    df = build_frame()
    feats = NUMERIC + CATEGORICAL
    train, val, test = make_splits(df)
    print(f"rows modelable={len(df)} train={len(train)} val={len(val)} "
          f"test={len(test)} prep={time.time()-t0:.1f}s", flush=True)
    pipe = Pipeline([("prep", preprocessor()), ("clf", model())])
    pipe.fit(train[feats], train["succ"])
    p_val = pipe.predict_proba(val[feats])[:, 1]
    thr = pick_threshold(val["succ"].values, p_val)
    p_te = pipe.predict_proba(test[feats])[:, 1]
    y = test["succ"].values
    auc = roc_auc_score(y, p_te)
    ap = average_precision_score(y, p_te)
    br = brier_score_loss(y, p_te)
    f1 = f1_score(y, (p_te >= thr).astype(int))
    k = int(np.ceil(0.10 * len(y)))
    order = np.argsort(-p_te)
    top = y[order[:k]].mean()
    print(f"threshold(val,F1)={thr:.4f}  pred_pos_rate={float((p_te>=thr).mean()):.4f}")
    print(f"TEST  AUC={auc:.6f}  PR-AUC={ap:.6f}  Brier={br:.6f}  F1={f1:.6f}")
    print(f"top10% success={top:.6f}  lift={top/y.mean():.4f}  n={len(y)}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
