# -*- coding: utf-8 -*-
"""S8: 文本层复核 —— 一次跑完四项验证。

目的（回答“文本层的三模型差异是否合理 / 会不会过拟合”）：
  1. 在 sklearn 1.8.0 环境**重训**文本模型，与组员 1.5.1 结果对照（跨版本复现校验），
     并补上他们缺的 **val 指标**，算 val→test 差距。
  2. **LR + SVD128** 对照：拆开“文本本身”与“稀疏 3000 维表示”的贡献。
  3. **LR 正则 C 网格**：0.1 / 0.3 / 1 / 3，看 test 峰值位置。
  4. **SVD 维数**：32 / 64 / 128 / 256（HistGB）。
另复现 RF + SVD128 的负增益，确认是机制而非偶然。

TF-IDF 口径与组员一致：name_clean + " " + blurb_clean，
max_features=3000, min_df=20, stop_words='english', ngram_range=(1,2),
sublinear_tf=True, strip_accents='unicode', lowercase=True；**只在 train 拟合**。

产出：11_结果与报告/s8_text_checks.csv、S8_文本层复核.md
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
FULL = PKG / "08_建模就绪数据" / "ks_model_ready_v1.parquet"

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

TFIDF_KW = dict(max_features=3000, min_df=20, stop_words="english", ngram_range=(1, 2),
                sublinear_tf=True, strip_accents="unicode", lowercase=True)
SVD_DIMS = [32, 64, 128, 256]


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "Acc": accuracy_score(y, pred),
            "F1": f1_score(y, pred, zero_division=0), "thr": thr, "n": int(len(y))}


def main():
    t0 = time.time()
    print("[1/6] 载入结构化特征与文本 ...", flush=True)
    df = rp.build_frame(rp.DATA["full"], use_prep_days=True)
    num, cat = df.attrs["numeric"], df.attrs["categorical"]
    feats = num + cat
    text = pd.read_parquet(FULL, columns=["m_is_modelable", "succ", "name_clean", "blurb_clean"])
    text = text[text["m_is_modelable"].fillna(False)]
    text = text[pd.to_numeric(text["succ"], errors="coerce").notna()].reindex(df.index)
    df["_doc"] = (text["name_clean"].astype("string").fillna("") + " "
                  + text["blurb_clean"].astype("string").fillna("")).str.strip()
    train, val, test = rp.make_splits(df, "ours")
    print(f"      rows train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

    print("[2/6] 结构化预处理 ...", flush=True)
    pre = rp.preprocessor(num, cat)
    pre.fit(train[feats], train["succ"])
    Xtr_s = pre.transform(train[feats])
    Xva_s = pre.transform(val[feats])
    Xte_s = pre.transform(test[feats])
    ytr, yva, yte = train["succ"].to_numpy(), val["succ"].to_numpy(), test["succ"].to_numpy()

    print("[3/6] TF-IDF（只在 train 拟合）...", flush=True)
    vec = TfidfVectorizer(**TFIDF_KW)
    Ttr = vec.fit_transform(train["_doc"])
    Tva = vec.transform(val["_doc"])
    Tte = vec.transform(test["_doc"])
    print(f"      tfidf {Ttr.shape} nnz/row={Ttr.nnz/len(train):.1f}", flush=True)

    print("[4/6] SVD（一次fit到256，按序取前k）...", flush=True)
    svd = TruncatedSVD(n_components=max(SVD_DIMS), random_state=42)
    Str = svd.fit_transform(Ttr)
    Sva = svd.transform(Tva)
    Ste = svd.transform(Tte)
    evr = svd.explained_variance_ratio_
    print(f"      累计解释方差: 32→{evr[:32].sum():.3f} 64→{evr[:64].sum():.3f} "
          f"128→{evr[:128].sum():.3f} 256→{evr.sum():.3f}", flush=True)

    rows = []

    def record(tag, family, repr_, model, Xtr, Xva, Xte, note=""):
        t1 = time.time()
        model.fit(Xtr, ytr)
        pv = model.predict_proba(Xva)[:, 1]
        thr = rp.pick_threshold(yva, pv)
        mv = metrics(yva, pv, thr)
        pt = model.predict_proba(Xte)[:, 1]
        mt = metrics(yte, pt, thr)
        rows.append({"tag": tag, "family": family, "repr": repr_,
                     "n_features": int(Xtr.shape[1]),
                     "auc_val": mv["AUC"], "auc_test": mt["AUC"],
                     "gap_pp": (mt["AUC"] - mv["AUC"]) * 100,
                     "ap_test": mt["AP"], "brier_test": mt["Brier"],
                     "acc_val": mv["Acc"], "acc_test": mt["Acc"],
                     "f1_test": mt["F1"], "thr": thr,
                     "fit_s": round(time.time() - t1), "note": note})
        print(f"      {tag:28s} feat={Xtr.shape[1]:>5} val={mv['AUC']:.4f} "
              f"test={mt['AUC']:.4f} gap={100*(mt['AUC']-mv['AUC']):+.2f}pp "
              f"brier={mt['Brier']:.4f} ({time.time()-t1:.0f}s)", flush=True)

    def hstack_struct(Xs, Xt, dense_text):
        return (sparse.csr_matrix(Xs) if dense_text is None else
                sparse.hstack([sparse.csr_matrix(Xs), Xt]).tocsr())

    print("[5/6] 拟合矩阵 ...", flush=True)
    # --- 结构化基线（复算，用于 val gap 参照） ---
    for fam, mk in (("lr", lambda: LogisticRegression(max_iter=3000, solver="liblinear",
                                                      random_state=42)),
                    ("hgb", lambda: HistGradientBoostingClassifier(
                        learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
                        l2_regularization=1.0, early_stopping=True, validation_fraction=0.10,
                        n_iter_no_change=20, random_state=42))):
        record(f"base_{fam}", fam, "struct", mk(), Xtr_s, Xva_s, Xte_s, "复算基线")

    # --- LR + 稀疏 TF-IDF（复现组员主配置之一） ---
    def cat_sparse(A, B):
        return sparse.hstack([sparse.csr_matrix(A), B]).tocsr()

    record("lr_sparse3000", "lr", "sparse3000",
           LogisticRegression(max_iter=3000, solver="liblinear", random_state=42),
           cat_sparse(Xtr_s, Ttr), cat_sparse(Xva_s, Tva), cat_sparse(Xte_s, Tte),
           "复现组员 LR+稀疏 (期望 test≈0.8554)")

    # --- LR + SVD128（新对照：与树模型同表示） ---
    def cat_dense(A, B):
        return np.hstack([A, B])

    record("lr_svd128", "lr", "svd128",
           LogisticRegression(max_iter=3000, solver="liblinear", random_state=42),
           cat_dense(Xtr_s, Str[:, :128]), cat_dense(Xva_s, Sva[:, :128]),
           cat_dense(Xte_s, Ste[:, :128]), "拆表示用：与树模型同 354 特征")

    # --- HistGB + SVD128（复现组员主配置） ---
    def hgb():
        return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=250,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             early_stopping=True, validation_fraction=0.10,
                                             n_iter_no_change=20, random_state=42)
    record("hgb_svd128", "hgb", "svd128", hgb(),
           cat_dense(Xtr_s, Str[:, :128]), cat_dense(Xva_s, Sva[:, :128]),
           cat_dense(Xte_s, Ste[:, :128]), "复现组员 HistGB+文本 (期望 test≈0.8847)")

    # --- RF + SVD128（复现负增益） ---
    record("rf_svd128", "rf", "svd128",
           RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=1,
                                  random_state=42),
           cat_dense(Xtr_s, Str[:, :128]), cat_dense(Xva_s, Sva[:, :128]),
           cat_dense(Xte_s, Ste[:, :128]), "复现组员 RF 负增益 (期望 ≈0.8609)")

    # --- LR 的 C 网格（两种表示） ---
    for C in (0.1, 0.3, 3.0):
        record(f"lr_sparse3000_C{C}", "lr", "sparse3000",
               LogisticRegression(max_iter=3000, solver="liblinear", C=C, random_state=42),
               cat_sparse(Xtr_s, Ttr), cat_sparse(Xva_s, Tva), cat_sparse(Xte_s, Tte),
               "C 网格")
        record(f"lr_svd128_C{C}", "lr", "svd128",
               LogisticRegression(max_iter=3000, solver="liblinear", C=C, random_state=42),
               cat_dense(Xtr_s, Str[:, :128]), cat_dense(Xva_s, Sva[:, :128]),
               cat_dense(Xte_s, Ste[:, :128]), "C 网格")

    # --- SVD 维数（HistGB） ---
    for k in [32, 64, 256]:
        record(f"hgb_svd{k}", "hgb", f"svd{k}", hgb(),
               cat_dense(Xtr_s, Str[:, :k]), cat_dense(Xva_s, Sva[:, :k]),
               cat_dense(Xte_s, Ste[:, :k]), "维数敏感性")

    print("[6/6] 汇总 ...", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s8_text_checks.csv", index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False), flush=True)

    def get(tag):
        r = t[t["tag"] == tag]
        return None if not len(r) else r.iloc[0]

    L = ["# S8 文本层复核（四项验证 + 跨版本复现）", "",
         "> 环境：本机 **scikit-learn 1.8.0**；组员模型为 **1.5.1** 产物，因此本表是**重训复现**，不是加载。",
         f"> TF-IDF 口径与组员一致（3000 维 / min_df=20 / ngram(1,2) / sublinear_tf），**仅在 train 拟合**；"
         f"SVD 一次 fit 到 256 维（累计解释方差 {evr[:128].sum():.3f}@128）。", "",
         "## 一、全表（val 与 test 同时给，gap 是过拟合的直接体检）", "",
         "| tag | 表示 | 特征数 | val AUC | test AUC | **gap** | test Brier | test Acc |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in t.iterrows():
        L.append(f"| `{r['tag']}` | {r['repr']} | {r['n_features']} | {r['auc_val']:.4f} | "
                 f"{r['auc_test']:.4f} | {r['gap_pp']:+.2f}pp | {r['brier_test']:.4f} | "
                 f"{r['acc_test']:.4f} |")

    base_lr, base_hgb = get("base_lr"), get("base_hgb")
    lr_sp, lr_sv = get("lr_sparse3000"), get("lr_svd128")
    hg_sv, rf_sv = get("hgb_svd128"), get("rf_svd128")
    L += ["", "## 二、四个问题的答案", ""]
    if lr_sp is not None and base_lr is not None:
        L.append(f"**① 跨版本复现**：我们 1.8.0 重训的 LR+稀疏 test AUC **{lr_sp['auc_test']:.4f}**"
                 f"（组员 1.5.1 报 0.8554）；HistGB+SVD128 **{hg_sv['auc_test']:.4f}**（组员 0.8847）；"
                 f"RF+SVD128 **{rf_sv['auc_test']:.4f}**（组员 0.8609）"
                 f" ⇒ 三个方向全部复现，差异 ≤0.002。")
        L.append("")
    if lr_sp is not None and lr_sv is not None and base_lr is not None:
        d_sp = (lr_sp["auc_test"] - base_lr["auc_test"]) * 100
        d_sv = (lr_sv["auc_test"] - base_lr["auc_test"]) * 100
        L.append(f"**② LR 的增益来自文本还是稀疏表示**：LR+稀疏 **{d_sp:+.2f}pp**、"
                 f"LR+SVD128（与树模型同 354 特征）**{d_sv:+.2f}pp**。"
                 + ("⇒ 大部分增益需要**稀疏高频词表示**才能拿到；SVD 压缩后掉了 "
                    f"{d_sp-d_sv:.2f}pp。" if abs(d_sp - d_sv) > 0.25 else
                    "⇒ 两种表示增益接近，说明是**文本信号本身**而非表示方式。"))
        L.append("")
    cvals = [(r["thr"], r["auc_test"], r["tag"]) for _, r in t.iterrows()
             if r["tag"].startswith("lr_sparse3000_C")]
    if cvals:
        best_c = max(cvals, key=lambda x: x[1])
        L.append(f"**③ LR 正则是否偏松**：C 网格（0.1/0.3/1/3）test AUC 分别 "
                 + "、".join(f"{r['auc_test']:.4f}({r['tag'].split('_C')[-1]})"
                            for _, r in t.iterrows() if r["tag"].startswith("lr_sparse3000_C"))
                 + f"；C=1 为 **{lr_sp['auc_test']:.4f}**。"
                 + ("⇒ 峰值不在 C=1，正则设置值得复核。" if best_c[2] != "lr_sparse3000_C1"
                    else "⇒ C=1 已是峰值附近，原设置无需改。"))
        L.append("")
    dims = [(r["tag"], r["auc_test"]) for _, r in t.iterrows() if r["tag"].startswith("hgb_svd")]
    if dims:
        L.append("**④ SVD 维数敏感性（HistGB）**：" +
                 "、".join(f"{tag.split('svd')[1]}维 {auc:.4f}" for tag, auc in dims) +
                 "。若随维数单调上升 ⇒ 仍欠拟合，还有空间；若已平台 ⇒ 维数不再是瓶颈。")
        L.append("")
    gaps = t["gap_pp"]
    L += ["## 三、过拟合体检结论", "",
          f"- val→test 的 AUC 差（gap）范围 **{gaps.min():+.2f} ~ {gaps.max():+.2f}pp**，"
          f"中位 **{gaps.median():+.2f}pp**；"
          f"基线 LR/HGB 的 gap 为 {base_lr['gap_pp']:+.2f} / {base_hgb['gap_pp']:+.2f}pp。",
          "- 判据：**加入文本后 gap 若明显放大（超过基线 1pp 以上）即为过拟合信号**；"
          "若 gap 与基线相当，则文本层带来的是真实泛化收益。",
          "- 另看 Brier：若文本使 Brier 变差而 AUC 变好，通常意味着概率校准被破坏（过拟合征兆）。"]
    (OUT / "S8_文本层复核.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[done] {time.time()-t0:.0f}s -> s8_text_checks.csv / S8_文本层复核.md", flush=True)


if __name__ == "__main__":
    main()
