"""Text branch: TF-IDF + logistic regression, scored out-of-fold by year.

Design note
-----------
The TF-IDF vocabulary is fitted once on the *training window only*. Text is
available before launch, so using the whole training vocabulary is not
outcome leakage; what must stay out-of-fold is the classifier. For every
training row we therefore train the classifier on strictly earlier years and
score that year (expanding, one model per year). Test rows are scored by a
classifier trained on the full training window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

MIN_HISTORY = 20_000


def _clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def build_text_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with id, yr, split, text_score (NaN where no history)."""
    df = df.reset_index(drop=True)
    train_idx = np.flatnonzero((df["yr"] <= C.TRAIN_MAX_YEAR).to_numpy())
    test_idx = np.flatnonzero((df["yr"] >= C.TEST_MIN_YEAR).to_numpy())
    texts = _clean_text(df["blurb_clean"])

    C.log("text: fitting TF-IDF on training window ...")
    vec = TfidfVectorizer(**C.TFIDF)
    X_tr = vec.fit_transform(texts.iloc[train_idx])
    C.log(f"text: train matrix {X_tr.shape} nnz={X_tr.nnz:,}")

    yr_tr = df["yr"].to_numpy()[train_idx]
    y_tr = df["succ"].to_numpy()[train_idx].astype(int)
    scores = np.full(len(df), np.nan, dtype=float)

    for y in sorted(np.unique(yr_tr)):
        prior = yr_tr < y
        tag = int(prior.sum())
        if tag < MIN_HISTORY:
            C.log(f"text: year {y} skipped (history={tag:,} < {MIN_HISTORY:,})")
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
        clf.fit(X_tr[prior], y_tr[prior])
        cur = yr_tr == y
        pos = train_idx[cur]
        scores[pos] = clf.predict_proba(X_tr[cur])[:, 1]
        C.log(f"text: year {y} n={cur.sum():,} history={tag:,} "
              f"auc_in_sample_na")

    C.log("text: fitting final classifier on all train rows for the test window")
    X_te = vec.transform(texts.iloc[test_idx])
    clf_full = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    clf_full.fit(X_tr, y_tr)
    scores[test_idx] = clf_full.predict_proba(X_te)[:, 1]

    from sklearn.metrics import roc_auc_score
    auc_test = roc_auc_score(df["succ"].to_numpy()[test_idx],
                             scores[test_idx])
    C.log(f"text: text-only out-of-time AUC on test = {auc_test:.4f}")

    out = pd.DataFrame({
        "id": df["id"], "yr": df["yr"], "text_score": scores,
        "split": np.where(df["yr"] <= C.TRAIN_MAX_YEAR, "train", "test"),
    })
    joblib.dump({"vec": vec, "clf": clf_full}, C.MODEL_DIR / "text_model.joblib")
    C.dump_json(C.TABLE_DIR / "text_model_summary.json",
                {"test_auc_text_only": auc_test,
                 "n_features": int(X_tr.shape[1]),
                 "nnz_train": int(X_tr.nnz),
                 "n_scored_train": int(np.isfinite(scores[train_idx]).sum())})
    return out


def main() -> None:
    df = C.load_v2()
    out = build_text_scores(df)
    out.to_csv(C.DATA_DIR / "text_scores.csv.gz", index=False, compression="gzip")
    C.log(f"text: wrote scores for {int(out.text_score.notna().sum()):,} rows")


if __name__ == "__main__":
    main()
