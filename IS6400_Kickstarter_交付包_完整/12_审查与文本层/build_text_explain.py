"""S6 - keyword panel + per-document attribution for the TF-IDF text layer.

Outputs (all new files, nothing in the package is overwritten):

  11_结果与报告/keyword_panel.csv        per-category token panel
                                         (smoothed log-odds, BH-FDR, era sign
                                         stability, model coefficient,
                                         trust grade, direction)
  11_结果与报告/text_contrib_top.csv     per-document top-5 positive /
                                         negative token contributions
                                         (test window, 2023+)
  11_结果与报告/text_explain_summary.json
  04_图表/figt8_keyword_panel.png
  12_审查与文本层/work/text_model_fitted.joblib   fitted vectoriser + LR

Method
------
Panel : for every category, count documents containing each token, separately
        among success and failure docs inside the training window (yr<=2022).
        Standard 2x2 -> adjusted log-odds ratio (0.5 continuity correction):
            a = #success with token      b = #failure with token
            c = #success without token   d = #failure without token
            lo = log(a/b) - log(c/d),   se = sqrt(1/a+1/b+1/c+1/d)
        two-sided normal p; Benjamini-Hochberg FDR within the category;
        sign stability against the independent test era (yr>=2023);
        direction agreement with the text-model coefficient w_j.
Attrib : contribution of token j to document i is x_ij * w_j (x = tf-idf,
         w = logistic coefficient on that vocabulary), so
         sum_j contrib_ij + b = logit(text_score_i) exactly.

Run: python -X utf8 build_text_explain.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_text_layer as B                                  # noqa: E402

PANEL_MIN_CAT = 2_000        # category needs >= this many training docs
VOCAB_MAX_YEAR = 2022        # vocabulary / final classifier window
TOKEN_MIN_DOCS = 20          # token must appear in >=20 success AND >=20 fail
TOKEN_MIN_SHARE = 0.005      # ... and in >=0.5% of the category documents
STABLE_MIN_ABS_LO = 0.10     # |log-odds| needed in BOTH eras to call it stable
TOP_K = 5                    # tokens per side in text_contrib_top.csv
TRUST_POS = 10               # panel rows flagged "top" per category and side
# Tokens that survive cleaning but carry no editorial meaning: URL / HTML
# fragments, bare years, replacement characters. Kept in the CSV (flagged) but
# never promoted into the curated top lists or the figure.
ARTIFACT_RE = re.compile(
    r"^(com|ll|http|https|www|amp|nbsp|quot|apos|rsquo|lsquo|ldquo|rdquo"
    r"|19\d\d|20\d\d)$")
OUT = B.OUT
FIG = B.FIG
WORK = B.PKG / "12_审查与文本层" / "work"


# --------------------------------------------------------------------------- #
def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return out


def fit_text_model(df):
    """Refit exactly what build_text_layer.build_text_score fits."""
    import joblib
    t = time.time()
    # build_text_layer.build_text_score() fits the vocabulary and the final
    # classifier on the whole <=2022 window (not the 2020 model-train window),
    # so the ported model must use the same rows or the text scores differ.
    tr = (df["yr"] <= VOCAB_MAX_YEAR).to_numpy()
    vec = B.TfidfVectorizer(**B.TFIDF)
    X = vec.fit_transform(df["doc"].to_numpy()[tr])
    Xall = vec.transform(df["doc"].to_numpy())
    y = df["succ"].to_numpy()[tr].astype(int)
    clf = LogisticRegression(**B.TEXT_CLF).fit(X, y)
    print(f"  vectoriser {Xall.shape} nnz={Xall.nnz} fit_docs={int(tr.sum())} "
          f"({time.time()-t:.1f}s)", flush=True)
    te = (df["yr"] >= B.TEST_MIN_YEAR).to_numpy()
    auc = roc_auc_score(df["succ"].to_numpy()[te],
                        clf.predict_proba(Xall[te])[:, 1])
    print(f"  text-only test AUC={auc:.4f}  (pipeline reference 0.7799)",
          flush=True)
    joblib.dump({"vec": vec, "clf": clf}, WORK / "text_model_fitted.joblib")
    return vec, clf, Xall, auc


def era_totals(df, cats, masks):
    """(n_docs, n_success) per category for each era."""
    out = {}
    for name, m in masks.items():
        g = (df.loc[m].groupby("category_parent_name")["succ"]
             .agg(["size", "sum"]).reindex(cats).fillna(0))
        out[name] = (g["size"].to_numpy(float), g["sum"].to_numpy(float))
    return out


def category_counts(Xb, A, mask):
    return (A[mask].T @ Xb[mask]).toarray().astype(np.float32)


def build_panel_counts(df, Xall, cats):
    t = time.time()
    Xb = Xall.copy()
    Xb.data = np.ones_like(Xb.data, dtype=np.float32)
    cat_code = pd.Categorical(df["category_parent_name"],
                              categories=cats).codes
    A = sp.csr_matrix((np.ones(len(df), np.float32),
                       (np.arange(len(df)), cat_code)),
                      shape=(len(df), len(cats)))
    yr = df["yr"].to_numpy()
    succ = df["succ"].to_numpy().astype(int)
    # train must be the SAME window as the vocabulary / classifier (<=2022),
    # otherwise log-odds and coef live in different populations; <=2020 is the
    # separate "early" era used only as an extra archive column.
    masks = {"train": yr <= VOCAB_MAX_YEAR,
             "early": yr <= 2020,
             "test": yr >= B.TEST_MIN_YEAR}
    counts = {}
    for name, m in masks.items():
        counts[name] = {
            "all": category_counts(Xb, A, m),
            "succ": category_counts(Xb, A, m & (succ == 1)),
        }
        print(f"  counts[{name}] ({time.time()-t:.1f}s)", flush=True)
    tot = era_totals(df, cats, masks)
    return counts, tot


def panel_table(df, counts, tot, cats, vocab, w):
    out = []
    for ci, cat in enumerate(cats):
        n_docs, n_succ = tot["train"][0][ci], tot["train"][1][ci]
        if n_docs < PANEL_MIN_CAT:
            continue
        n_s = counts["train"]["succ"][ci].astype(np.float64)
        n_f = counts["train"]["all"][ci].astype(np.float64) - n_s
        if (n_s + n_f).max() == 0:
            continue
        a = n_s + 0.5
        b = n_f + 0.5
        c = np.maximum(n_succ - n_s, 0) + 0.5
        d = np.maximum((n_docs - n_succ) - n_f, 0) + 0.5
        lo = np.log(a / b) - np.log(c / d)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        p = 2 * stats.norm.sf(np.abs(lo / se))

        def era_lo(era):
            en_docs, en_succ = tot[era][0][ci], tot[era][1][ci]
            es = counts[era]["succ"][ci].astype(np.float64)
            ef = counts[era]["all"][ci].astype(np.float64) - es
            aa, bb = es + .5, ef + .5
            cc = np.maximum(en_succ - es, 0) + .5
            dd = np.maximum((en_docs - en_succ) - ef, 0) + .5
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log(aa / bb) - np.log(cc / dd)
            v[~np.isfinite(v)] = np.nan
            return v

        e_lo, t_lo = era_lo("early"), era_lo("test")
        keep = ((n_s >= TOKEN_MIN_DOCS) & (n_f >= TOKEN_MIN_DOCS)
                & ((n_s + n_f) >= max(TOKEN_MIN_DOCS,
                                      TOKEN_MIN_SHARE * n_docs)))
        if not keep.any():
            continue
        q = np.full(len(lo), np.nan)
        q[keep] = bh_fdr(p[keep])
        rate_tok = n_s / np.maximum(n_s + n_f, 1)
        cat_rate = float(n_succ / max(n_docs, 1))
        out.append(pd.DataFrame({
            "category": cat,
            "token": vocab,
            "n_docs_category": n_docs,
            "doc_share": (n_s + n_f) / n_docs,
            "n_success_docs": n_s,
            "n_failure_docs": n_f,
            "success_rate_with_token": rate_tok,
            "success_rate_category": cat_rate,
            "lift": rate_tok / cat_rate,
            "log_odds": lo,
            "se": se,
            "p_value": p,
            "p_fdr": q,
            "coef": w,
            "log_odds_early_2020": e_lo,
            "log_odds_test_2023": t_lo,
        })[keep])
    panel = pd.concat(out, ignore_index=True)
    panel["significant_fdr"] = panel["p_fdr"] < 0.05
    both = panel["log_odds_test_2023"].notna()
    panel["sign_stable"] = (
        both
        & (np.sign(panel["log_odds"]) == np.sign(panel["log_odds_test_2023"]))
        & (panel["log_odds"].abs() >= STABLE_MIN_ABS_LO)
        & (panel["log_odds_test_2023"].abs() >= STABLE_MIN_ABS_LO))
    panel["coef_agrees"] = np.sign(panel["coef"]) == np.sign(panel["log_odds"])
    panel["trust"] = np.select(
        [panel["significant_fdr"] & panel["sign_stable"] & panel["coef_agrees"],
         panel["significant_fdr"] & (panel["sign_stable"] | panel["coef_agrees"])],
        ["高", "中"], default="低")
    panel["direction"] = np.where(panel["log_odds"] > 0, "多用", "少用")
    panel["is_artifact"] = (
        panel["token"].str.match(ARTIFACT_RE)
        | panel["token"].str.contains("\ufffd", na=False))
    panel["is_top_pos"] = False
    panel["is_top_neg"] = False
    for cat, g in panel.groupby("category"):
        cand = g[(g["trust"] != "低") & g["significant_fdr"]
                 & ~g["is_artifact"]]
        if cand.empty:
            continue
        panel.loc[cand.sort_values("log_odds", ascending=False)
                  .head(TRUST_POS).index, "is_top_pos"] = True
        panel.loc[cand.sort_values("log_odds").head(TRUST_POS).index,
                  "is_top_neg"] = True
    return panel.sort_values(["category", "log_odds"],
                             ascending=[True, False]).reset_index(drop=True)


def attribute(df, Xall, clf, test_mask, vocab):
    t = time.time()
    w = clf.coef_.ravel().astype(np.float64)
    b = float(clf.intercept_[0])
    Xt = Xall[test_mask].tocsr()
    sub = df.loc[test_mask].reset_index(drop=True)
    p_text = clf.predict_proba(Xt)[:, 1]
    names = sub["name_clean"].astype("string").fillna("").str.slice(0, 70)
    indptr, indices, data = Xt.indptr, Xt.indices, Xt.data
    n = len(sub)
    res = {"id": sub["id"].to_numpy(), "yr": sub["yr"].to_numpy(),
           "category": sub["category_parent_name"].to_numpy(),
           "name": names.to_numpy(), "p_text": p_text,
           "n_tokens": np.diff(indptr),
           "sum_pos_contrib": np.zeros(n), "sum_neg_contrib": np.zeros(n)}
    pos_cols = [[] for _ in range(TOP_K)]
    neg_cols = [[] for _ in range(TOP_K)]
    for i in range(n):
        lo, hi = indptr[i], indptr[i + 1]
        if lo == hi:
            for k in range(TOP_K):
                pos_cols[k].append("")
                neg_cols[k].append("")
            continue
        idx = indices[lo:hi]
        c = data[lo:hi] * w[idx]
        order = np.argsort(c)
        pos = order[::-1][:TOP_K]
        neg = order[:TOP_K]
        res["sum_pos_contrib"][i] = c[c > 0].sum()
        res["sum_neg_contrib"][i] = c[c < 0].sum()
        for k in range(TOP_K):
            pos_cols[k].append(f"{vocab[idx[pos[k]]]}:{c[pos[k]]:+.4f}"
                               if k < len(pos) else "")
            neg_cols[k].append(f"{vocab[idx[neg[k]]]}:{c[neg[k]]:+.4f}"
                               if k < len(neg) else "")
    for k in range(TOP_K):
        res[f"top_pos_{k+1}"] = pos_cols[k]
        res[f"top_neg_{k+1}"] = neg_cols[k]
    res["logit_p_text"] = np.log(p_text / (1 - p_text))
    res["intercept"] = b
    contrib = pd.DataFrame(res)
    contrib.to_csv(OUT / "text_contrib_top.csv", index=False)
    print(f"  attribution rows={len(contrib)} mean_tokens="
          f"{contrib['n_tokens'].mean():.2f} ({time.time()-t:.1f}s)", flush=True)
    return contrib, b


def fig_panel(panel):
    cats = ["Music", "Food", "Fashion", "Publishing"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
    for ax, cat in zip(axes.ravel(), cats):
        g = panel[(panel["category"] == cat) & panel["significant_fdr"]
                  & (panel["trust"] == "高")]
        sub = pd.concat([g.sort_values("log_odds").head(7),
                         g.sort_values("log_odds", ascending=False).head(7)]
                        ).drop_duplicates("token").sort_values("log_odds")
        colors = ["#c62828" if v < 0 else "#1565c0" for v in sub["log_odds"]]
        ax.barh(sub["token"], sub["log_odds"], color=colors)
        ax.axvline(0, color="#616161", lw=0.9)
        ax.set_title(f"{cat}：FDR<0.05 且 2023+ 同号", fontsize=10.5)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("分品类简介关键词面板　蓝=多用（更常出现在达标项目）　红=少用",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "figt8_keyword_panel.png", dpi=150)
    plt.close(fig)
    print("  figure written", flush=True)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    df = B.build_frame()
    print(f"S0 rows={len(df)} prep={time.time()-t0:.1f}s", flush=True)

    vec, clf, Xall, auc = fit_text_model(df)
    vocab = np.asarray(vec.get_feature_names_out())
    w = clf.coef_.ravel().astype(np.float64)
    print(f"S1 vocab={len(vocab)}", flush=True)

    cats = (df.loc[df["yr"] <= VOCAB_MAX_YEAR, "category_parent_name"]
            .value_counts().index.tolist())
    counts, tot = build_panel_counts(df, Xall, cats)
    panel = panel_table(df, counts, tot, cats, vocab, w)
    # guard: the "early" era must be a genuinely different population from the
    # training window; if it is not, the masks have collapsed again.
    diff_share = ((panel["log_odds_early_2020"] - panel["log_odds"]).abs() > 1e-6).mean()
    assert diff_share > 0.90, f"early era collapsed into train era ({diff_share:.2f})"
    panel.to_csv(OUT / "keyword_panel.csv", index=False)
    print(f"S2 panel rows={len(panel)} cats={panel['category'].nunique()} "
          f"trust={panel['trust'].value_counts().to_dict()} "
          f"({time.time()-t0:.1f}s)", flush=True)
    demo = panel[panel["is_top_pos"]].groupby("category")["token"].apply(
        lambda s: ", ".join(s.head(5)))
    print(demo.to_string(), flush=True)

    test_mask = (df["yr"] >= B.TEST_MIN_YEAR).to_numpy()
    contrib, b = attribute(df, Xall, clf, test_mask, vocab)
    fig_panel(panel)

    summary = {
        "text_only_auc_test": float(auc),
        "vocab": int(len(vocab)),
        "intercept": b,
        "panel_rows": int(len(panel)),
        "panel_categories": sorted(panel["category"].unique().tolist()),
        "trust_counts": {k: int(v) for k, v in
                         panel["trust"].value_counts().items()},
        "panel_significant": int(panel["significant_fdr"].sum()),
        "panel_sign_stable_share": float(
            panel.loc[panel["significant_fdr"], "sign_stable"].mean()),
        "panel_artifact_rows": int(panel["is_artifact"].sum()),
        "panel_trust_high_by_category": {
            cat: int(((g["trust"] == "高") & ~g["is_artifact"]).sum())
            for cat, g in panel.groupby("category")},
        "top_pos_lift_median": float(
            panel.loc[panel["is_top_pos"], "lift"].median()),
        "top_pos_doc_share_median": float(
            panel.loc[panel["is_top_pos"], "doc_share"].median()),
        "top_pos_per_category": {
            cat: g.loc[g["is_top_pos"], "token"].tolist()
            for cat, g in panel.groupby("category")},
        "top_neg_per_category": {
            cat: g.loc[g["is_top_neg"], "token"].tolist()
            for cat, g in panel.groupby("category")},
        "contrib_rows": int(len(contrib)),
        "contrib_mean_n_tokens": float(contrib["n_tokens"].mean()),
        "contrib_pct_no_positive_token": float(
            (contrib["sum_pos_contrib"] == 0).mean()),
        "logit_identity_max_abs_err": float(np.abs(
            contrib["sum_pos_contrib"] + contrib["sum_neg_contrib"] + b
            - contrib["logit_p_text"]).max()),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "text_explain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE {summary['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
