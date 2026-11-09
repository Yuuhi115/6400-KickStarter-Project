"""Global config: paths, v2 (USD-only) scope constants, seeds, helpers.

Every stage imports from here so there is a single source of truth for the
split, the leakage list and the binning used in tables/figures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]          # ...\KS project\ks_v2
WORKSPACE = ROOT.parent                             # ...\KS project
DATA_RAW = WORKSPACE.parent / "ks_clean.csv"        # ...\group project\ks_clean.csv
OUT = ROOT / "outputs"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
LOG_DIR = OUT / "logs"
MODEL_DIR = OUT / "models"

for _d in (DATA_DIR, TABLE_DIR, FIG_DIR, LOG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

V2_PATH = DATA_DIR / "ks_v2_usd.csv.gz"
DROPPED_PATH = DATA_DIR / "dropped_rows_currency.csv.gz"

# ---------------------------------------------------------------- scope
SEED = 42
CURRENCY_KEEP = "USD"
TRAIN_MAX_YEAR = 2022
TEST_MIN_YEAR = 2023
VAL_YEARS = list(range(2017, 2023))      # rolling-origin validation years

SUCC = "succ"

# columns that must never enter the feature matrix (post-outcome leakage)
LEAK_COLS = [
    "pledged", "usd_pledged", "backers_count", "percent_funded",
    "converted_pledged_amount", "spotlight", "profile_state", "staff_pick",
    "state_changed_at", "profile_state_changed_at",
]

CAT_COLS = ["category_key", "subcategory_key", "location_type", "location_state"]

NUM_COLS = [
    "log_goal", "dur_days", "creator_rank", "creator_n_projects",
    "creator_prev_success_rate", "prev_miss", "is_relaunch",
    "mon", "dow", "hour", "blurb_len", "blurb_words", "name_len",
    "goal_vs_cat_median",
]

# goal bins in USD (right-closed), labels used in every table
GOAL_BINS = [0, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000, np.inf]
GOAL_LABELS = ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", "50-100k",
               "100-500k", ">500k"]

DUR_BINS = [0, 7, 14, 21, 30, 45, 60, 90, np.inf]
DUR_LABELS = ["1-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", ">90"]

# shorter duration split for the scorecard (keeps cells populated)
DUR4_BINS = [0, 14, 30, 45, np.inf]
DUR4_LABELS = ["1-14", "15-30", "31-45", "46+"]

RANK_BINS = [0, 1, 2, 3, 4, 5, 8, np.inf]
RANK_LABELS = ["1st", "2nd", "3rd", "4th", "5th", "6-8th", ">8th"]

MIN_CELL_N = 300

# ---------------------------------------------------------------- model params
HGB_CLS = dict(
    learning_rate=0.06, max_iter=400, max_depth=6, max_leaf_nodes=40,
    min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=30, random_state=SEED,
)
HGB_REG = dict(
    learning_rate=0.06, max_iter=500, max_depth=6, max_leaf_nodes=40,
    min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=30, random_state=SEED,
)
RF_CLS = dict(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
              n_jobs=-1, random_state=SEED)
TREE_CLS = dict(max_depth=5, min_samples_leaf=500, random_state=SEED)
TFIDF = dict(ngram_range=(1, 2), min_df=10, max_features=15_000,
             sublinear_tf=True, stop_words="english")

# ---------------------------------------------------------------- helpers
def ensure_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): ensure_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(ensure_jsonable(obj), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_v2() -> pd.DataFrame:
    df = pd.read_csv(V2_PATH, low_memory=False)
    # 守卫：全流程约定 log_goal = log10(goal)；反事实/示例卡重算时必须用同一底数
    np.testing.assert_allclose(np.log10(df["goal"].to_numpy(float)),
                               df["log_goal"].to_numpy(float), atol=1e-9,
                               err_msg="log_goal 不是 log10(goal)，反事实重算会失真")
    return df


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["goal_bin"] = pd.cut(df["goal"], GOAL_BINS, labels=GOAL_LABELS, right=True)
    df["dur_bin"] = pd.cut(df["dur_days"], DUR_BINS, labels=DUR_LABELS, right=True)
    df["dur4_bin"] = pd.cut(df["dur_days"], DUR4_BINS, labels=DUR4_LABELS, right=True)
    df["rank_bin"] = pd.cut(df["creator_rank"], RANK_BINS, labels=RANK_LABELS, right=True)
    return df


def wilson_ci(k, n, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion (vectorised)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def savefig(fig, name: str, dpi: int = 150) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def log(msg: str) -> None:
    line = f"[{pd.Timestamp.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
