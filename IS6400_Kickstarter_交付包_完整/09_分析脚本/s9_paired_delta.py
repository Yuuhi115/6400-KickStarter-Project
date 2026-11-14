# -*- coding: utf-8 -*-
"""S9: LR 的文本增益与 val→test gap 变化，是随机偏差还是真实？

方法：**按 creator_id 整簇配对 bootstrap**（1,000 次），全部在同一条数据上配对比较。

回答三件事：
  A. 文本增益 ΔAUC = AUC(text) − AUC(base)：在 val 与 test 上分别是多少、CI 是否排除 0
  B. gap = AUC_test − AUC_val：base 与 text 各自的 gap，以及 **DiD = gap_text − gap_base**
     （DiD 的 CI 由 val 与 test 独立重抽后相减合成）
  C. 稀疏表示 vs SVD 表示：谁的增益更稳

对象：LR（base / +sparse3000 / +SVD128）——用户指定看 LR。
产出：11_结果与报告/s9_lr_paired_bootstrap.csv、S9_LR增益是否真实.md
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
FULL = PKG / "08_建模就绪数据" / "ks_model_ready_v1.parquet"
REPS = 1000
RNG = np.random.RandomState(42)

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

TFIDF_KW = dict(max_features=3000, min_df=20, stop_words="english", ngram_range=(1, 2),
                sublinear_tf=True, strip_accents="unicode", lowercase=True)


def auc_fast(y, p):
    """Mann-Whitney AUC，忽略并列（预测为连续值）。"""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot_delta(y, p_base, p_text, cidx, cluster_sizes, reps=REPS):
    """配对 bootstrap：每次重抽同一批簇，同时算两个模型的 AUC 再相减。"""
    n = len(y)
    nc = len(cluster_sizes)
    deltas = np.empty(reps)
    for r in range(reps):
        w = RNG.multinomial(nc, np.full(nc, 1.0 / nc))
        idx = np.repeat(np.arange(n), w[cidx])
        yy = y[idx]
        deltas[r] = auc_fast(yy, p_text[idx]) - auc_fast(yy, p_base[idx])
    return deltas


def main():
    t0 = time.time()
    print("[1/5] 载入 ...", flush=True)
    df = rp.build_frame(rp.DATA["full"], use_prep_days=True)
    num, cat = df.attrs["numeric"], df.attrs["categorical"]
    feats = num + cat
    extra = pd.read_parquet(FULL, columns=["m_is_modelable", "succ", "creator_id",
                                           "name_clean", "blurb_clean"])
    extra = extra[extra["m_is_modelable"].fillna(False)]
    extra = extra[pd.to_numeric(extra["succ"], errors="coerce").notna()].reindex(df.index)
    df["creator_id"] = extra["creator_id"].to_numpy()
    df["_doc"] = (extra["name_clean"].astype("string").fillna("") + " "
                  + extra["blurb_clean"].astype("string").fillna("")).str.strip()
    train, val, test = rp.make_splits(df, "ours")

    pre = rp.preprocessor(num, cat)
    pre.fit(train[feats], train["succ"])
    Xtr, Xva, Xte = (pre.transform(train[feats]), pre.transform(val[feats]),
                     pre.transform(test[feats]))
    ytr, yva, yte = (train["succ"].to_numpy(), val["succ"].to_numpy(), test["succ"].to_numpy())

    print("[2/5] TF-IDF + SVD ...", flush=True)
    vec = TfidfVectorizer(**TFIDF_KW)
    Ttr = vec.fit_transform(train["_doc"])
    Tva, Tte = vec.transform(val["_doc"]), vec.transform(test["_doc"])
    svd = TruncatedSVD(n_components=128, random_state=42)
    Str = svd.fit_transform(Ttr)
    Sva, Ste = svd.transform(Tva), svd.transform(Tte)

    print("[3/5] 拟合三个 LR ...", flush=True)
    def mk():
        return LogisticRegression(max_iter=3000, solver="liblinear", random_state=42)

    specs = {
        "base_struct": (Xtr, Xva, Xte),
        "lr_sparse3000": (sparse.hstack([sparse.csr_matrix(Xtr), Ttr]).tocsr(),
                          sparse.hstack([sparse.csr_matrix(Xva), Tva]).tocsr(),
                          sparse.hstack([sparse.csr_matrix(Xte), Tte]).tocsr()),
        "lr_svd128": (np.hstack([Xtr, Str]), np.hstack([Xva, Sva]), np.hstack([Xte, Ste])),
    }
    preds = {}
    for name, (A, B, C) in specs.items():
        m = mk()
        t1 = time.time()
        m.fit(A, ytr)
        preds[name] = {"val": m.predict_proba(B)[:, 1], "test": m.predict_proba(C)[:, 1]}
        print(f"      {name} fitted ({time.time()-t1:.0f}s)", flush=True)

    # 簇结构
    def clusters(sub):
        cid = sub["creator_id"].to_numpy()
        _, inv = np.unique(cid, return_inverse=True)
        return inv, np.bincount(inv)

    cva, sva_ = clusters(val)
    cte, ste_ = clusters(test)
    print(f"[4/5] bootstrap {REPS} 次（val 簇 {len(sva_)}, test 簇 {len(ste_)}）...", flush=True)

    rows, store = [], {}
    for name in ("lr_sparse3000", "lr_svd128"):
        for rname, y, cidx, sizes in (("val", yva, cva, sva_), ("test", yte, cte, ste_)):
            d = boot_delta(y, preds["base_struct"][rname], preds[name][rname],
                           cidx, sizes)
            store[(name, rname)] = d
            rows.append({"model": name, "rowset": rname,
                         "auc_base": auc_fast(y, preds["base_struct"][rname]),
                         "auc_text": auc_fast(y, preds[name][rname]),
                         "delta_pp": d.mean() * 100,
                         "ci_lo_pp": np.percentile(d, 2.5) * 100,
                         "ci_hi_pp": np.percentile(d, 97.5) * 100,
                         "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))})
    t = pd.DataFrame(rows)

    # DiD：gap_text − gap_base
    print("[5/5] gap 变化（DiD）与结论 ...", flush=True)
    did_rows = []
    for name in ("lr_sparse3000", "lr_svd128"):
        base_gap = auc_fast(yte, preds["base_struct"]["test"]) - auc_fast(yva, preds["base_struct"]["val"])
        text_gap = auc_fast(yte, preds[name]["test"]) - auc_fast(yva, preds[name]["val"])
        did = (store[(name, "test")] - store[(name, "val")])  # 每次重抽的 gap 变化
        did_rows.append({"model": name,
                         "gap_base_pp": base_gap * 100,
                         "gap_text_pp": text_gap * 100,
                         "DiD_pp": (text_gap - base_gap) * 100,
                         "DiD_ci_lo_pp": np.percentile(did, 2.5) * 100,
                         "DiD_ci_hi_pp": np.percentile(did, 97.5) * 100})
    dd = pd.DataFrame(did_rows)

    t.to_csv(OUT / "s9_lr_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    dd.to_csv(OUT / "s9_lr_gap_did.csv", index=False, encoding="utf-8-sig")

    L = ["# S9 LR 的文本增益：是随机偏差还是真实提升？", "",
         f"> 方法：**按 `creator_id` 整簇配对 bootstrap {REPS} 次**，每次在同一批重抽行上同时算两个模型的 AUC 再相减（配对，消除行集差异）。",
         "> 评估集：val（2021–22, 60,467）与 test（≥2023, 100,469）。", "",
         "## 一、文本增益 ΔAUC（配对 CI）", "",
         "| 模型 | 评估集 | AUC 基线 | AUC 加文本 | **ΔAUC** | 95% CI | p |",
         "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for _, r in t.iterrows():
        L.append(f"| {r['model']} | {r['rowset']} | {r['auc_base']:.4f} | {r['auc_text']:.4f} | "
                 f"**{r['delta_pp']:+.3f}pp** | [{r['ci_lo_pp']:+.3f}, {r['ci_hi_pp']:+.3f}]pp | "
                 f"{r['p_two_sided']:.4f} |")
    L += ["", "## 二、val→test gap 及其变化（DiD）", "",
          "| 模型 | gap 基线 | gap 加文本 | **DiD（gap 变化）** | 95% CI |",
          "| --- | ---: | ---: | ---: | --- |"]
    for _, r in dd.iterrows():
        L.append(f"| {r['model']} | {r['gap_base_pp']:+.2f}pp | {r['gap_text_pp']:+.2f}pp | "
                 f"**{r['DiD_pp']:+.2f}pp** | [{r['DiD_ci_lo_pp']:+.2f}, {r['DiD_ci_hi_pp']:+.2f}]pp |")
    L += ["", "## 三、判读规则", "",
          "- **增益是否真实**：CI 不含 0（即 p<0.05）⇒ 真实；CI 含 0 ⇒ 不能排除随机。",
          "- **gap 缩小是否真实**：DiD 的 CI 完全低于 0 ⇒ gap 确实缩小；含 0 ⇒ gap 变化在噪声范围内。",
          "- ⚠️ gap 本身（test − val）**不是“提升”**，它是两个不同年份样本的难度差；"
          "即使 gap 缩小，也不代表模型变强，只代表**泛化衰减变小**。"]
    (OUT / "S9_LR增益是否真实.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(t.round(4).to_string(index=False), flush=True)
    print(dd.round(4).to_string(index=False), flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
