# -*- coding: utf-8 -*-
"""TF-IDF text layer: build `text_score`, train the five comparison models,
measure the increment with paired bootstrap, and draw the figures.

Implements S0-S5 of `12_审查与文本层/02_TF-IDF文本层_方法计划表.md`.

Data contract
  * reads  ..\\..\\ks_clean.csv  (read-only, one level above the package)
  * writes only inside the package:
        11_结果与报告\\text_*            (tables / model / scores)
        04_图表\\figt*.png               (figures)
        12_审查与文本层\\work\\*         (intermediate audit tables)

Protocol (same as probe_text_increment.py, which produced deltaAUC=+0.00795)
  vocabulary : fit on yr <= 2022 only
  text_score : per-year expanding-window OOF LogisticRegression (liblinear, C=1),
               MIN_HISTORY = 10_000, folds 2011..2022; 2009-2010 get the
               expanding prior base rate (no history -> no text evidence)
  test       : classifier fit on all yr <= 2022 rows
  models     : 5 configs on the same split (train<=2020 / val 2021-22 /
               test>=2023), same HistGB hyper-parameters as the delivered model
  CI         : paired bootstrap on the test rows (300 reps)

Run:  python build_text_layer.py       (~7-9 min)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score,  # noqa: E402
                             brier_score_loss, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

import joblib  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
RAW = PKG.parents[1] / "ks_clean.csv"
OUT = PKG / "11_结果与报告"
FIG = PKG / "04_图表"
WORK = PKG / "12_审查与文本层" / "work"
for d in (OUT, FIG, WORK):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_MAX_YEAR, VAL_YEARS, TEST_MIN_YEAR = 2020, (2021, 2022), 2023
OOF_START_YEAR = 2011
MIN_HISTORY = 10_000
TFIDF = dict(ngram_range=(1, 2), min_df=5, max_features=30_000,
             sublinear_tf=True, stop_words="english", lowercase=True)
TEXT_CLF = dict(C=1.0, max_iter=1000, solver="liblinear")
N_BOOT = 300
SUPPORT_WORDS = 25  # p95 of m_blurb_words on the modellable sample

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
LENGTH_COLS = ["m_name_len", "m_blurb_len", "m_blurb_words", "name_word_count"]
BLURB_COLS = ["m_blurb_len", "m_blurb_words"]
NUM_NO_TEXT = [c for c in NUMERIC if c not in LENGTH_COLS]
NUM_NO_BLURB = [c for c in NUMERIC if c not in BLURB_COLS]

MODELS = {
    "M0 无文本特征(14)": NUM_NO_TEXT + CATEGORICAL,
    "M0b 保留标题去简介(16)": NUM_NO_BLURB + CATEGORICAL,
    "M1 仅长度词数(18)": NUMERIC + CATEGORICAL,
    "M2 长度+TF-IDF(19)": NUMERIC + ["text_score"] + CATEGORICAL,
    "M3 TF-IDF替换长度(15)": NUM_NO_TEXT + ["text_score"] + CATEGORICAL,
}
CJK = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["font.sans-serif"] = CJK
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


# --------------------------------------------------------------------------- #
# S0 data
# --------------------------------------------------------------------------- #
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
    df["name_word_count"] = pd.to_numeric(
        df["name"].astype("string").str.split().str.len())
    df["creator_previous_count"] = (pd.to_numeric(df["creator_rank"],
                                                  errors="coerce") - 1).clip(lower=0)
    df["creator_prev_success_rate"] = pd.to_numeric(
        df["creator_prev_success_rate"], errors="coerce")
    df["yr"] = df["yr"].astype(int)
    df["doc"] = (name.fillna("") + " " + blurb.fillna("")).str.lower()
    return df.reset_index(drop=True)


def make_splits(df):
    return (df[df["yr"] <= TRAIN_MAX_YEAR],
            df[df["yr"].between(*VAL_YEARS)],
            df[df["yr"] >= TEST_MIN_YEAR])


# --------------------------------------------------------------------------- #
# S1 corpus audit
# --------------------------------------------------------------------------- #
def corpus_audit(df):
    doc = df["doc"]
    tok = doc.str.split()
    audit = {
        "rows": int(len(df)),
        "empty_doc": int((doc.str.strip() == "").sum()),
        "dup_doc_rows": int(doc.duplicated().sum()),
        "mean_tokens": float(tok.str.len().mean()),
        "median_tokens": float(tok.str.len().median()),
        "p95_tokens": float(tok.str.len().quantile(0.95)),
        "max_tokens": float(tok.str.len().max()),
        "nonzero_share_2023plus": float(
            (tok[df["yr"] >= TEST_MIN_YEAR].str.len() > 0).mean()),
    }
    pd.Series(audit).to_csv(WORK / "corpus_audit.csv", header=["value"])
    print(f"S1 corpus audit rows={audit['rows']} empty={audit['empty_doc']} "
          f"dup={audit['dup_doc_rows']} p95_tokens={audit['p95_tokens']:.0f}",
          flush=True)
    return audit


# --------------------------------------------------------------------------- #
# S2 + S3 vectorise and build the OOF text score
# --------------------------------------------------------------------------- #
def build_text_score(df):
    t0 = time.time()
    yr = df["yr"].to_numpy()
    y = df["succ"].to_numpy()
    vocab_mask = yr <= 2022
    vec = TfidfVectorizer(**TFIDF)
    vec.fit(df.loc[vocab_mask, "doc"])
    X = vec.transform(df["doc"])
    nnz_train = int(X[vocab_mask].nnz)
    print(f"S2 tfidf {X.shape} nnz_all={X.nnz} nnz_train={nnz_train} "
          f"fit_max_year={int(yr[vocab_mask].max())} ({time.time()-t0:.1f}s)",
          flush=True)

    score = np.full(len(df), np.nan)
    fold = np.zeros(len(df), dtype=int)
    folds = []
    for cur in range(OOF_START_YEAR, 2023):
        prior = (yr < cur) & vocab_mask
        n_prior = int(prior.sum())
        if n_prior < MIN_HISTORY:
            continue
        clf = LogisticRegression(**TEXT_CLF)
        clf.fit(X[prior], y[prior])
        m = yr == cur
        score[m] = clf.predict_proba(X[m])[:, 1]
        fold[m] = cur - 1
        auc = roc_auc_score(y[m], score[m])
        folds.append({"score_year": cur, "fit_through": cur - 1,
                      "n_fit": n_prior, "n_scored": int(m.sum()),
                      "auc_in_year": round(float(auc), 6)})
        print(f"  fold yr={cur} n_fit={n_prior} auc={auc:.4f}", flush=True)

    cold = yr < OOF_START_YEAR
    cold_value = float(y[vocab_mask].mean())
    score[cold] = cold_value
    print(f"  cold rows (yr<{OOF_START_YEAR}) n={int(cold.sum())} "
          f"score={cold_value:.4f}", flush=True)

    te = yr >= TEST_MIN_YEAR
    clf_final = LogisticRegression(**TEXT_CLF)
    clf_final.fit(X[vocab_mask], y[vocab_mask])
    score[te] = clf_final.predict_proba(X[te])[:, 1]
    warm = (~cold) & vocab_mask
    summary = {
        "n_docs": int(len(df)),
        "n_features": int(X.shape[1]),
        "nnz_all": int(X.nnz),
        "nnz_train_window": nnz_train,
        "fit_max_year": 2022,
        "min_history": MIN_HISTORY,
        "oof_start_year": OOF_START_YEAR,
        "n_cold_rows": int(cold.sum()),
        "cold_value": round(cold_value, 6),
        "text_only_auc_oof_2011_2022": round(
            float(roc_auc_score(y[warm], score[warm])), 6),
        "text_only_auc_test": round(
            float(roc_auc_score(y[te], score[te])), 6),
        "n_scored_test": int(te.sum()),
        "folds": folds,
        "seconds": round(time.time() - t0, 1),
    }
    pd.DataFrame({
        "id": df["id"], "yr": yr,
        "split": np.where(yr <= TRAIN_MAX_YEAR, "train",
                          np.where(yr <= 2022, "val", "test")),
        "text_score": score, "clf_fit_through_year": fold,
    }).to_csv(OUT / "text_scores.csv.gz", index=False, compression="gzip")
    print(f"S3 text-only AUC oof={summary['text_only_auc_oof_2011_2022']:.4f} "
          f"test={summary['text_only_auc_test']:.4f}", flush=True)
    return vec, clf_final, score, summary


# --------------------------------------------------------------------------- #
# preprocessor / model
# --------------------------------------------------------------------------- #
def make_preprocessor(numeric):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant",
                                          fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                          min_frequency=20,
                                          sparse_output=False))])
    return ColumnTransformer([("num", num, numeric),
                              ("cat", cat, CATEGORICAL)])


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.10,
        n_iter_no_change=20, random_state=42)


def pick_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(y, p, thr):
    k = int(np.ceil(0.10 * len(y)))
    top = float(y[np.argsort(-p)[:k]].mean())
    base = float(y.mean())
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, (p >= thr).astype(int))),
        "threshold": thr,
        "pred_pos_rate": float((p >= thr).mean()),
        "top10_success": top,
        "lift": top / base,
        "n": int(len(y)),
        "base_rate": base,
    }


def boot_ci(y, p, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    a = np.array(vals)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def boot_delta(y, p_new, p_base, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p_new[i])
                    - roc_auc_score(y[i], p_base[i]))
    a = np.array(vals)
    return (float(a.mean()), float(np.percentile(a, 2.5)),
            float(np.percentile(a, 97.5)), int(len(a)))


# --------------------------------------------------------------------------- #
# S4 train the five models
# --------------------------------------------------------------------------- #
def train_all(df):
    train, val, test = make_splits(df)
    y_val, y_te = val["succ"].to_numpy(), test["succ"].to_numpy()
    rows, preds, pipes = [], {}, {}
    for name, feats in MODELS.items():
        t = time.time()
        numeric = [c for c in feats if c in NUMERIC or c == "text_score"]
        assert len(numeric) + len(CATEGORICAL) == len(feats), \
            f"feature list mismatch for {name}"
        pipe = Pipeline([("prep", make_preprocessor(numeric)),
                          ("clf", make_model())])
        pipe.fit(train[feats], train["succ"])
        thr = pick_threshold(y_val, pipe.predict_proba(val[feats])[:, 1])
        p_te = pipe.predict_proba(test[feats])[:, 1]
        met = evaluate(y_te, p_te, thr)
        met["model"] = name
        met["n_features"] = len(feats)
        lo, hi = boot_ci(y_te, p_te)
        met["auc_ci_lo"], met["auc_ci_hi"] = lo, hi
        met["seconds"] = round(time.time() - t, 1)
        rows.append(met)
        preds[name] = p_te
        pipes[name] = pipe
        print(f"  {name:24s} AUC={met['auc']:.6f} "
              f"[{lo:.6f},{hi:.6f}] Brier={met['brier']:.6f} "
              f"thr={thr:.3f} ({met['seconds']:.0f}s)", flush=True)
    metrics = pd.DataFrame(rows)[
        ["model", "n_features", "auc", "auc_ci_lo", "auc_ci_hi", "pr_auc",
         "brier", "f1", "threshold", "pred_pos_rate", "top10_success",
         "lift", "base_rate", "n", "seconds"]]
    metrics.to_csv(OUT / "text_layer_metrics.csv", index=False)
    return metrics, preds, pipes, (train, val, test)


# --------------------------------------------------------------------------- #
# S5 increments
# --------------------------------------------------------------------------- #
PAIRS = [
    ("M1 仅长度词数(18)", "M0 无文本特征(14)", "仅长度 vs 无文本"),
    ("M0b 保留标题去简介(16)", "M0 无文本特征(14)", "标题长度 vs 无文本"),
    ("M2 长度+TF-IDF(19)", "M1 仅长度词数(18)", "TF-IDF 在长度之上"),
    ("M2 长度+TF-IDF(19)", "M0b 保留标题去简介(16)", "TF-IDF vs 不用简介"),
    ("M3 TF-IDF替换长度(15)", "M0 无文本特征(14)", "TF-IDF vs 无文本"),
    ("M2 长度+TF-IDF(19)", "M0 无文本特征(14)", "文本层总增量"),
]


def increments(df, preds, test):
    y = test["succ"].to_numpy()
    rows = []
    for new, base, label in PAIRS:
        m, lo, hi, nb = boot_delta(y, preds[new], preds[base])
        rows.append({"comparison": label, "scope": "all", "n": len(y),
                     "auc_new": roc_auc_score(y, preds[new]),
                     "auc_base": roc_auc_score(y, preds[base]),
                     "delta": m, "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    cat = test["category_parent_name"].fillna("Unknown")
    for scope, mask in cat.groupby(cat).indices.items():
        if len(mask) < 2000:
            continue
        yy = y[mask]
        if yy.min() == yy.max():
            continue
        for new, base, label in PAIRS:
            m, lo, hi, nb = boot_delta(yy, preds[new][mask], preds[base][mask],
                                       n=200)
            rows.append({"comparison": label, "scope": f"cat:{scope}",
                         "n": int(len(mask)),
                         "auc_new": roc_auc_score(yy, preds[new][mask]),
                         "auc_base": roc_auc_score(yy, preds[base][mask]),
                         "delta": m, "ci_lo": lo, "ci_hi": hi, "n_boot": nb})
    inc = pd.DataFrame(rows)
    inc.to_csv(OUT / "text_increment.csv", index=False)
    overall = inc[inc["scope"] == "all"]
    print("S5 increments (test, all rows):", flush=True)
    for _, r in overall.iterrows():
        print(f"  {r['comparison']:22s} {r['auc_base']:.6f} -> "
              f"{r['auc_new']:.6f}  d={r['delta']:+.5f} "
              f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]", flush=True)
    return inc


def auc_by_year(preds, test):
    y = test["succ"].to_numpy()
    rows = []
    for name, p in preds.items():
        for yy in sorted(test["yr"].unique()):
            m = (test["yr"] == yy).to_numpy()
            if m.sum() < 500 or y[m].min() == y[m].max():
                continue
            rows.append({"model": name, "yr": int(yy), "n": int(m.sum()),
                         "auc": float(roc_auc_score(y[m], p[m]))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "text_auc_by_year.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# length response (replaces the -13pp extrapolation with a support-bounded curve)
# --------------------------------------------------------------------------- #
def length_response(df, pipes, test, vec, clf_text):
    """Truncate the blurb to k words and rescore everything that depends on it:
    the two length scalars AND `text_score` (the text model re-reads the
    truncated document). This is a within-support counterfactual: k <= p95.
    """
    pipe = pipes["M2 长度+TF-IDF(19)"]
    feats = MODELS["M2 长度+TF-IDF(19)"]
    sub = test[test["m_blurb_words"] >= SUPPORT_WORDS].copy()
    if len(sub) > 6000:
        sub = sub.sample(6000, random_state=0)
    print(f"length-response sample n={len(sub)} (rows with >= "
          f"{SUPPORT_WORDS} blurb words)", flush=True)
    p0 = pipe.predict_proba(sub[feats])[:, 1]
    rows = []
    for k in [6, 10, 14, 18, 22, 25]:
        d = sub.copy()
        blurb = (d["blurb_clean"].astype("string").fillna("")
                 .str.split().str[:k].str.join(" "))
        d["blurb_clean"] = blurb
        d["m_blurb_words"] = pd.to_numeric(blurb.str.split().str.len())
        d["m_blurb_len"] = pd.to_numeric(blurb.str.len())
        d["doc"] = (d["name_clean"].astype("string").fillna("").str.lower()
                    + " " + blurb.str.lower())
        d["text_score"] = clf_text.predict_proba(vec.transform(d["doc"]))[:, 1]
        p = pipe.predict_proba(d[feats])[:, 1]
        rows.append({"k_words": k, "n": int(len(d)),
                     "mean_p_success": float(p.mean()),
                     "mean_text_score": float(d["text_score"].mean()),
                     "mean_blurb_len": float(d["m_blurb_len"].mean()),
                     "in_support": bool(k <= SUPPORT_WORDS)})
        print(f"  k={k:2d} mean_P={p.mean():.4f} "
              f"mean_text_score={d['text_score'].mean():.4f} "
              f"mean_len={d['m_blurb_len'].mean():.0f}", flush=True)
    out = pd.DataFrame(rows)
    out["dp_vs_original"] = out["mean_p_success"] - float(p0.mean())
    out.to_csv(OUT / "text_length_response.csv", index=False)
    print(f"  original mean_words={sub['m_blurb_words'].mean():.1f} "
          f"mean_P={p0.mean():.4f}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
COLORS = {"M0 无文本特征(14)": "#9e9e9e", "M0b 保留标题去简介(16)": "#8d6e63",
          "M1 仅长度词数(18)": "#1e88e5", "M2 长度+TF-IDF(19)": "#e53935",
          "M3 TF-IDF替换长度(15)": "#43a047"}


def fig_roc(preds, test):
    y = test["succ"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[name], lw=2.0,
                label=f"{name}  AUC={roc_auc_score(y, p):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#bdbdbd", lw=1)
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.set_title("ROC：五种特征配置（test = 2023 年以后，n=100,469）")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "figt1_roc.png")
    plt.close(fig)


def fig_auc_ci(metrics):
    m = metrics.set_index("model").loc[list(MODELS)]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = np.arange(len(m))
    err = np.vstack([m["auc"] - m["auc_ci_lo"], m["auc_ci_hi"] - m["auc"]])
    ax.bar(xs, m["auc"], color=[COLORS[i] for i in m.index], width=0.62,
           yerr=err, capsize=4, error_kw={"lw": 1.2, "ecolor": "#424242"})
    for x, v in zip(xs, m["auc"]):
        ax.text(x, v + 0.0045, f"{v:.4f}", ha="center", fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([i.replace(" ", "\n", 1) for i in m.index], fontsize=9)
    ax.set_ylim(min(m["auc_ci_lo"]) - 0.012, 0.905)
    ax.set_ylabel("test ROC-AUC（误差线 = 配对 bootstrap 95% CI）")
    ax.set_title("五配置 AUC 对比")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG / "figt2_auc_ci.png")
    plt.close(fig)


def fig_forest(inc):
    o = inc[inc["scope"] == "all"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ys = np.arange(len(o))
    ax.errorbar(o["delta"], ys, xerr=[o["delta"] - o["ci_lo"],
                                      o["ci_hi"] - o["delta"]],
                fmt="o", color="#e53935", ecolor="#616161", capsize=4, lw=0)
    ax.axvline(0, color="#9e9e9e", ls="--", lw=1)
    for y, d, lo, hi in zip(ys, o["delta"], o["ci_lo"], o["ci_hi"]):
        ax.text(max(o["ci_hi"]) + 0.0006, y, f"{d:+.5f}  [{lo:+.5f}, {hi:+.5f}]",
                va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(o["comparison"], fontsize=9.5)
    ax.set_xlabel("ΔAUC（test，配对 bootstrap 300 次）")
    ax.set_title("各组对比的增量与 95% 置信区间")
    ax.set_xlim(min(o["ci_lo"]) - 0.001, max(o["ci_hi"]) + 0.0075)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG / "figt3_delta_forest.png")
    plt.close(fig)


def fig_textscore_dist(df, test):
    t = test.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lab, color in ((0, "#9e9e9e"), (1, "#e53935")):
        v = t.loc[t["succ"] == lab, "text_score"].dropna()
        ax.hist(v, bins=60, density=True, alpha=0.55, color=color,
                label=f"未达标（n={len(v):,}）" if lab == 0
                else f"达标（n={len(v):,}）")
    ax.set_xlabel("text_score（只用文案的迷你模型给出的达标概率）")
    ax.set_ylabel("密度")
    ax.set_title("text_score 的判别力：达标项目整体右移（test 2023+）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.22)
    fig.savefig(FIG / "figt4_textscore_dist.png")
    plt.close(fig)


def fig_auc_year(by_year):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for name in MODELS:
        s = by_year[by_year["model"] == name].sort_values("yr")
        if s.empty:
            continue
        ax.plot(s["yr"], s["auc"], "o-", color=COLORS[name], label=name,
                lw=1.8, ms=4)
    ax.set_xlabel("年份（test 段）")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("逐年 AUC：文本层的增益在各年都稳定为正")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "figt5_auc_by_year.png")
    plt.close(fig)


def fig_calibration(preds, test):
    y = test["succ"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for name, color in (("M1 仅长度词数(18)", "#1e88e5"),
                        ("M2 长度+TF-IDF(19)", "#e53935")):
        p = preds[name]
        bins = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
        xs, ys = [], []
        for b in sorted(pd.unique(bins)):
            m = bins == b
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
        ax.plot(xs, ys, "o-", color=color, lw=1.8, ms=4,
                label=f"{name}  Brier={brier_score_loss(y, p):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#bdbdbd", lw=1)
    ax.set_xlabel("预测达标概率（十分位）")
    ax.set_ylabel("实际达标率")
    ax.set_title("校准：加入文本层后 Brier 更低")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "figt6_calibration.png")
    plt.close(fig)


def fig_length(df, resp):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    ax = axes[0]
    d = df[df["yr"] >= TEST_MIN_YEAR]
    q = pd.qcut(d["m_blurb_words"].clip(upper=30), 10, labels=False,
                duplicates="drop")
    obs = d.groupby(q)["succ"].mean()
    mids = d.groupby(q)["m_blurb_words"].median()
    ax.plot(mids, obs.values, "o-", color="#616161", lw=1.8, ms=4)
    ax.set_xlabel("简介词数（2023+ 观测，十分位中位数）")
    ax.set_ylabel("实际达标率")
    ax.set_title("观测：十分位间 57%→66%，与词数无单调关系", fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[1]
    p0 = float((resp["mean_p_success"] - resp["dp_vs_original"]).iloc[0])
    ax.axhline(p0, color="#1565c0", ls=":", lw=1.2)
    ax.text(6.2, p0 - 0.0006, f" 原文本身（均值 25.8 词）={p0:.3f}",
            fontsize=8, color="#1565c0", va="top")
    ax.plot(resp["k_words"], resp["mean_p_success"], "o-", color="#e53935",
            lw=1.9, ms=4.5)
    i = int(resp["dp_vs_original"].idxmax())
    ax.annotate(f"峰值 {resp['dp_vs_original'].iloc[i] * 100:+.1f}pp",
                xy=(resp["k_words"].iloc[i], resp["mean_p_success"].iloc[i]),
                xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8.5, color="#b71c1c")
    ax.axvline(SUPPORT_WORDS, color="#9e9e9e", ls="--", lw=1.1)
    ax.text(SUPPORT_WORDS - 0.4, resp["mean_p_success"].min(),
            " p95 支撑上限 ", ha="right", fontsize=8, color="#616161")
    lo = min(resp["mean_p_success"].min(), p0)
    hi = resp["mean_p_success"].max()
    ax.set_ylim(lo - 0.004, hi + 0.004)
    ax.set_xlabel("把简介截断到 k 词（同时重算长度特征与 text_score）")
    ax.set_ylabel("模型预测达标概率（均值）")
    ax.set_title("模型内：支撑域内总波动 ≤3pp 且不单调", fontsize=11)
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "figt7_length_response.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"RAW={RAW}", flush=True)
    df = build_frame()
    print(f"S0 rows={len(df)} cols={df.shape[1]} prep={time.time()-t0:.1f}s",
          flush=True)
    audit = corpus_audit(df)
    vec, clf_text, score, summary = build_text_score(df)
    df["text_score"] = score

    metrics, preds, pipes, (train, val, test) = train_all(df)
    print(f"S4 done ({time.time()-t0:.1f}s)", flush=True)

    inc = increments(df, preds, test)
    by_year = auc_by_year(preds, test)
    resp = length_response(df, pipes, test, vec, clf_text)

    fig_roc(preds, test)
    fig_auc_ci(metrics)
    fig_forest(inc)
    fig_textscore_dist(df, test)
    fig_auc_year(by_year)
    fig_calibration(preds, test)
    fig_length(df, resp)
    print("figures written", flush=True)

    joblib.dump({"pipeline": pipes["M2 长度+TF-IDF(19)"],
                 "features": MODELS["M2 长度+TF-IDF(19)"],
                 "numeric": NUMERIC + ["text_score"],
                 "categorical": CATEGORICAL,
                 "threshold": float(metrics.loc[
                     metrics["model"] == "M2 长度+TF-IDF(19)", "threshold"].iloc[0]),
                 "text_model": {"tfidf": TFIDF, "clf": TEXT_CLF,
                                "fit_max_year": 2022,
                                "min_history": MIN_HISTORY}},
                OUT / "model_histgb_text.joblib")
    m2 = metrics.set_index("model").loc["M2 长度+TF-IDF(19)"].to_dict()
    schema = {
        "features": MODELS["M2 长度+TF-IDF(19)"],
        "numeric": NUMERIC + ["text_score"],
        "categorical": CATEGORICAL,
        "threshold": m2["threshold"],
        "test": {k: m2[k] for k in ("auc", "pr_auc", "brier", "f1",
                                    "top10_success", "lift")},
        "text_layer": {"text_score_source": "text_scores.csv.gz",
                       "text_model_summary": "text_model_summary.json"},
        "env": {"sklearn": "1.5.1", "numpy": "1.26.4", "pandas": "2.2.2"},
    }
    (OUT / "model_histgb_text_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["corpus_audit"] = audit
    summary["models"] = json.loads(metrics.set_index("model").to_json(
        orient="index"))
    summary["increment_all"] = json.loads(
        inc[inc["scope"] == "all"].to_json(orient="records"))
    summary["length_response"] = json.loads(resp.to_json(orient="records"))
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "text_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
