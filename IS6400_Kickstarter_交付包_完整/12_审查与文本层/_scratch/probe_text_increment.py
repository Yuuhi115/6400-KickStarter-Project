# -*- coding: utf-8 -*-
"""Probe: how much does a PIT-safe TF-IDF text score add to the 18-feature model?

Reuses repro_18feat.build_frame / make_splits / preprocessor / model so the
structural side is exactly the delivered baseline.

Protocol
  * vocabulary : fit on yr <= 2022 only (train + val window)
  * text score : per-year expanding-window out-of-fold LogisticRegression
                 (C=1, liblinear, MIN_HISTORY=20_000) -> PIT-safe for <=2022
  * test score : classifier fit on all yr <= 2022 rows
  * model      : HistGB 18 features vs 19 features (+text_score), same splits
  * CI         : paired bootstrap on the test rows (300 reps)

Writes nothing except stdout. Runtime ~3-4 min.
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, ".")
import repro_18feat as R  # noqa: E402

TFIDF = dict(ngram_range=(1, 2), min_df=5, max_features=30_000,
             sublinear_tf=True, stop_words="english", lowercase=True)
MIN_HISTORY = 20_000


def prep(numeric):
    """Same preprocessor as the package, but with an explicit numeric list
    (ColumnTransformer selects by name, so extra columns are dropped unless
    they are listed here)."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant",
                                          fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                          min_frequency=20,
                                          sparse_output=False))])
    return ColumnTransformer([("num", num, numeric),
                              ("cat", cat, R.CATEGORICAL)])


def text_of(df):
    return (df["name_clean"].astype("string").fillna("") + " "
            + df["blurb_clean"].astype("string").fillna("")).str.lower()


def main():
    t0 = time.time()
    df = R.build_frame()
    train, val, test = R.make_splits(df)
    ndf = len(df)
    text = text_of(df).reset_index(drop=True)
    yr = df["yr"].to_numpy()
    y = df["succ"].to_numpy()
    vocab_mask = (yr <= 2022)
    print(f"rows={ndf} train={len(train)} val={len(val)} test={len(test)} "
          f"vocab_rows={int(vocab_mask.sum())} prep={time.time()-t0:.1f}s",
          flush=True)

    vec = TfidfVectorizer(**TFIDF)
    vec.fit(text[vocab_mask])
    X = vec.transform(text)
    print(f"tfidf {X.shape} nnz={X.nnz} fit_years<=2022 "
          f"({time.time()-t0:.1f}s)", flush=True)

    score = np.full(ndf, np.nan)
    first = None
    for cur in range(2012, 2023):
        prior = (yr < cur) & vocab_mask
        if prior.sum() < MIN_HISTORY:
            continue
        clf = LogisticRegression(C=1, max_iter=1000, solver="liblinear")
        clf.fit(X[prior], y[prior])
        first = first or clf
        cur_mask = (yr == cur)
        score[cur_mask] = clf.predict_proba(X[cur_mask])[:, 1]
        print(f"  OOF yr={cur} prior={int(prior.sum())}", flush=True)
    early = (yr < 2012) & vocab_mask
    if first is not None and early.sum():
        score[early] = first.predict_proba(X[early])[:, 1]
        print(f"  early rows scored in-sample by first clf: {int(early.sum())}",
              flush=True)
    clf_all = LogisticRegression(C=1, max_iter=1000, solver="liblinear")
    clf_all.fit(X[vocab_mask], y[vocab_mask])
    score[yr >= 2023] = clf_all.predict_proba(X[yr >= 2023])[:, 1]
    print(f"text-only AUC train<=2022 OOF="
          f"{roc_auc_score(y[vocab_mask], score[vocab_mask]):.4f} "
          f"test={roc_auc_score(y[yr >= 2023], score[yr >= 2023]):.4f}",
          flush=True)

    df = df.reset_index(drop=True)
    df["text_score"] = score
    print(f"text_score nan={int(df['text_score'].isna().sum())} "
          f"nunique={int(df['text_score'].nunique())}", flush=True)
    tr, va, te = R.make_splits(df)
    out = {}
    for tag, numeric in (("struct18", R.NUMERIC),
                         ("+text", R.NUMERIC + ["text_score"])):
        feats = numeric + R.CATEGORICAL
        t = time.time()
        pipe = Pipeline([("prep", prep(numeric)), ("clf", R.model())])
        pipe.fit(tr[feats], tr["succ"])
        thr = R.pick_threshold(va["succ"].to_numpy(),
                               pipe.predict_proba(va[feats])[:, 1])
        p = pipe.predict_proba(te[feats])[:, 1]
        out[tag] = p
        print(f"{tag:9s} AUC={roc_auc_score(te['succ'], p):.6f} "
              f"Brier={brier_score_loss(te['succ'], p):.6f} thr={thr:.4f} "
              f"({time.time()-t:.1f}s)", flush=True)

    yte = te["succ"].to_numpy()
    rng = np.random.default_rng(0)
    d = []
    for _ in range(300):
        idx = rng.integers(0, len(yte), len(yte))
        if yte[idx].min() == yte[idx].max():
            continue
        d.append(roc_auc_score(yte[idx], out["+text"][idx])
                 - roc_auc_score(yte[idx], out["struct18"][idx]))
    d = np.array(d)
    print(f"delta AUC={d.mean():.5f} CI95=[{np.percentile(d,2.5):.5f}, "
          f"{np.percentile(d,97.5):.5f}] n_boot={len(d)}", flush=True)
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
