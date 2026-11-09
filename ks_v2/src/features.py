"""Feature engineering for the v2 pipeline.

All specifications (category vocabularies, medians) are fitted on the training
window only and then applied to validation/test - never the other way round.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as C

OTHER = 0  # reserved code for unseen categories (never -1: HistGB wants >= 0)


def build_spec(train: pd.DataFrame, top_states: int = 20) -> Dict:
    cat_vocab: Dict[str, Dict[str, int]] = {}
    for col in C.CAT_COLS:
        vals = train[col].astype(str)
        if col == "location_state":
            keep = list(vals.value_counts().head(top_states).index)
        else:
            keep = list(vals.value_counts().index)
        cat_vocab[col] = {v: i + 1 for i, v in enumerate(keep)}  # 0 = OTHER
    cat_median = train.groupby("category_key")["log_goal"].median()
    return {
        "cat_vocab": cat_vocab,
        "cat_median": cat_median,
        "global_median": float(train["log_goal"].median()),
        "top_states": top_states,
    }


def transform(df: pd.DataFrame, spec: Dict, text_score: np.ndarray | None = None,
              strict: bool = True) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """Return (X, cat_mask, feature_names).

    X is dense float32 (HistGB does not accept sparse input in sklearn 1.5.1);
    categorical columns hold integer codes and are flagged by cat_mask.
    """
    out = pd.DataFrame(index=df.index)
    for col in C.CAT_COLS:
        mapping = spec["cat_vocab"][col]
        codes = df[col].astype(str).map(mapping).fillna(OTHER).astype(np.int32)
        out[col] = codes

    out["log_goal"] = df["log_goal"].astype(float)
    out["dur_days"] = df["dur_days"].astype(float)
    out["creator_rank"] = df["creator_rank"].astype(float)
    out["creator_n_projects"] = df["creator_n_projects"].astype(float)
    out["creator_prev_success_rate"] = df["creator_prev_success_rate"].astype(float)
    out["prev_miss"] = df["creator_prev_success_rate"].isna().astype(float)
    out["is_relaunch"] = df["is_relaunch"].astype(bool).astype(float)
    out["mon"] = df["mon"].astype(float)
    out["dow"] = df["dow"].astype(float)
    out["hour"] = df["hour"].astype(float)
    out["blurb_len"] = df["blurb_len"].astype(float)
    out["blurb_words"] = df["blurb_words"].astype(float)
    out["name_len"] = df["name_len"].astype(float)

    med = df["category_key"].astype(str).map(spec["cat_median"])
    med = med.fillna(spec["global_median"])
    out["goal_vs_cat_median"] = (df["log_goal"].astype(float) - med.astype(float))

    if text_score is not None:
        ts = pd.Series(np.asarray(text_score, dtype=float), index=df.index)
        out["text_score"] = ts

    cat_mask = np.array([c in C.CAT_COLS for c in out.columns], dtype=bool)
    X = out.astype(np.float32)
    if strict:
        assert not X.isna().all(axis=0).any(), "a feature column is entirely NaN"
    return X, cat_mask, list(X.columns)


def check_no_leakage(feature_names: list) -> None:
    bad = set(feature_names) & set(C.LEAK_COLS)
    assert not bad, f"leakage features: {bad}"
