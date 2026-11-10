"""Stage 0 - build the v2 (USD-only) modelling table + deletion audit.

Scope decisions (frozen):
  * keep currency == 'USD'
  * keep rows with a binary outcome (succ not null, i.e. successful/failed)
  * drop the 7,272-row cluster where usd_type is null: pledged == usd_pledged
    there (ratio exactly 1.0) so it is not a currency error; it is an early
    (2011-2014) low-funding cluster with a 6.6% success rate that would bias
    the early-year rates and is not explainable to a user.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src import config as C

USECOLS = [
    "id", "state", "succ", "is_terminal_pair", "currency", "usd_type",
    "goal", "log_goal", "usd_pledged", "pledged",
    "blurb_clean", "name_clean", "blurb_len", "blurb_words", "name_len",
    "dur_days", "yr", "mon", "dow", "hour",
    "category_key", "subcategory_key", "country_final", "location_type",
    "location_state", "creator_id", "creator_rank", "creator_n_projects",
    "creator_prev_success_rate", "is_relaunch", "launch_dt", "deadline_dt",
]


def main() -> None:
    C.log("stage data: reading raw csv ...")
    raw = pd.read_csv(C.DATA_RAW, usecols=USECOLS, low_memory=False)
    C.log(f"raw rows={len(raw):,} cols={raw.shape[1]}")

    # ---------- classification of every row, in priority order
    reason = pd.Series(pd.NA, index=raw.index, dtype="object")
    is_usd = raw["currency"].eq(C.CURRENCY_KEEP)
    has_succ = raw["succ"].notna()
    usd_known = raw["usd_type"].notna()

    reason[~is_usd] = "non_usd_currency"
    reason[is_usd & ~has_succ] = "no_binary_outcome"          # canceled
    reason[is_usd & has_succ & ~usd_known] = "usd_type_missing"

    v2 = raw[reason.isna()].copy()
    C.log(f"v2 rows={len(v2):,}  success rate={v2['succ'].mean():.4f}")

    # ---------- sanity assertions
    assert v2["succ"].notna().all()
    assert set(v2["currency"].unique()) == {"USD"}
    assert v2["usd_type"].notna().all()
    assert v2["dur_days"].between(1, 120).all()
    assert (v2["goal"] > 0).all()
    train = v2[v2["yr"] <= C.TRAIN_MAX_YEAR]
    test = v2[v2["yr"] >= C.TEST_MIN_YEAR]
    assert len(v2) == len(train) + len(test), "year split must be exhaustive"
    C.log(f"train(<= {C.TRAIN_MAX_YEAR}) n={len(train):,} rate={train['succ'].mean():.4f}")
    C.log(f"test(>= {C.TEST_MIN_YEAR})  n={len(test):,} rate={test['succ'].mean():.4f}")

    # leakage columns may be carried for audit/target use only (pledged is the
    # audit trail, usd_pledged is the dual-objective target) - they must never
    # appear in the feature lists.
    feature_guard = set(C.CAT_COLS) | set(C.NUM_COLS)
    leaks = [c for c in C.LEAK_COLS if c in feature_guard]
    assert not leaks, f"leakage columns in feature lists: {leaks}"
    assert "usd_pledged" in v2.columns and "pledged" in v2.columns

    # ---------- persist
    v2.to_csv(C.V2_PATH, index=False, compression="gzip")
    C.log(f"wrote {C.V2_PATH}  ({C.V2_PATH.stat().st_size / 1e6:.1f} MB)")

    dropped = raw[reason.notna()].copy()
    dropped["reason"] = reason[reason.notna()]
    dropped["goal"] = dropped["goal"].astype(float)
    dropped.to_csv(C.DROPPED_PATH, index=False, compression="gzip")
    C.log(f"dropped {len(dropped):,} rows -> {C.DROPPED_PATH}")
    C.log(dropped.groupby("reason", observed=True).size().to_string())

    C.dump_json(C.DATA_DIR / "v2_stats.json", {
        "raw_rows": len(raw),
        "v2_rows": len(v2),
        "v2_rate": v2["succ"].mean(),
        "train_rows": len(train),
        "train_rate": train["succ"].mean(),
        "test_rows": len(test),
        "test_rate": test["succ"].mean(),
        "dropped": dropped.groupby("reason", observed=True).size().to_dict(),
    })


if __name__ == "__main__":
    main()
