# -*- coding: utf-8 -*-
"""run_matrix.py — #7–#12：文本层入模（both 文本源）。

  #7/#8   hgb + [base 226 | SVD128]         full / usd
  #9/#10  lr  + [base 226 | TF-IDF 3000]    full / usd
  #11/#12 rf  + [base 226 | SVD128]         full / usd

纪律：TF-IDF 与 SVD 每个口径只 fit 一次、只用 train；val 只选阈值；test 只评估。
"""
from __future__ import annotations

import gc
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import cfg
import features


def main() -> None:
    t0 = time.time()
    rp = cfg.load_rp()
    df = features.load_prepared()
    rows, blocks = [], {}

    for scope in ("full", "usd"):
        print(f"\n================ scope={scope} ================", flush=True)
        tr, va, te = features.frames(df, scope)
        ytr, yva, yte = features.y_of(tr), features.y_of(va), features.y_of(te)
        design = features.BaseDesign().fit(tr)
        Xb = {"train": design.transform(tr), "val": design.transform(va), "test": design.transform(te)}
        print(f"[{scope}] base design {Xb['train'].shape}", flush=True)

        t1 = time.time()
        other_ids = np.concatenate([va["id"].values.astype(str), te["id"].values.astype(str)])
        block = features.TextBlock().fit(features.text_series(tr),
                                         tr["id"].values.astype(str), other_ids)
        print(f"[{scope}] TF-IDF vocab={block.n_vocab:,} nnz/doc={block.nnz_per_doc:.1f} "
              f"SVD{cfg.SVD_DIM} evr={block.evr:.4f} ({time.time() - t1:.1f}s)", flush=True)
        blocks[scope] = {"n_vocab": block.n_vocab, "nnz_per_doc": block.nnz_per_doc,
                         "svd_evr": block.evr, "svd_dim": cfg.SVD_DIM,
                         "tfidf_params": cfg.TFIDF_PARAMS}

        Xt_val = block.sparse(features.text_series(va))
        Xt_test = block.sparse(features.text_series(te))
        Z_val = block.dense(features.text_series(va))
        Z_test = block.dense(features.text_series(te))
        mask_usd = te["is_usd"].values.astype(bool) if scope == "full" else None

        def record(key: str, algo: str, est, Xtr, Xva, Xte, secs: float, extra: dict):
            pv = est.predict_proba(Xva)[:, 1]
            thr = rp.pick_threshold(yva, pv)
            pt = est.predict_proba(Xte)[:, 1]
            m = rp.evaluate(yte, pt, thr)
            td = rp.top_decile(yte, pt)
            row = {"key": key, "algo": algo, "scope": scope, "kind": "text",
                   "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                   "n_features": int(Xtr.shape[1]),
                   "fit_seconds": round(secs, 1), "threshold": float(thr),
                   "AUC": float(m["ROC_AUC"]), "AP": float(m["PR_AUC"]),
                   "Brier": float(m["Brier_Score"]), "F1": float(m["F1"]),
                   "Accuracy": float(m["Accuracy"]), "Precision": float(m["Precision"]),
                   "Recall": float(m["Recall"]), "base_rate": float(m["base_rate"]),
                   "top_rate": float(td["top_rate"]), "lift": float(td["lift"])}
            row.update(extra)
            rows.append(row)
            print(f"[{key}] AUC={m['ROC_AUC']:.10f} AP={m['PR_AUC']:.6f} "
                  f"Brier={m['Brier_Score']:.6f} thr={thr:.3f} {secs:.1f}s", flush=True)
            if scope == "full":
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__all.npy"), pt.astype(np.float32))
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__usd.npy"), pt[mask_usd].astype(np.float32))
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__nonusd.npy"), pt[~mask_usd].astype(np.float32))
            else:
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__usd.npy"), pt.astype(np.float32))
            return thr, pt, row

        # ---------- #7/#8 hgb + SVD ----------
        key = f"hgb_text_{scope}"
        print(f"[{key}] fitting ...", flush=True)
        t2 = time.time()
        Xh = {"train": features.hstack_dense(Xb["train"], block.Ztr),
              "val": features.hstack_dense(Xb["val"], Z_val),
              "test": features.hstack_dense(Xb["test"], Z_test)}
        est = HistGradientBoostingClassifier(**cfg.HGB_KW)
        est.fit(Xh["train"], ytr)
        thr, pt, row = record(key, "hgb", est, Xh["train"], Xh["val"], Xh["test"],
                              time.time() - t2, {"text_source": cfg.TEXT_SOURCE, "rep": "svd"})
        if scope == "full":
            _save_wrapper("model_histgb_text", design, block, est, thr, row, te, pt,
                          rep="svd", family="hgb")
        del Xh, est
        gc.collect()

        # ---------- #11/#12 rf + SVD ----------
        key = f"rf_text_{scope}"
        print(f"[{key}] fitting ...", flush=True)
        t2 = time.time()
        Xh = {"train": features.hstack_dense(Xb["train"], block.Ztr),
              "val": features.hstack_dense(Xb["val"], Z_val),
              "test": features.hstack_dense(Xb["test"], Z_test)}
        est = RandomForestClassifier(**cfg.RF_KW)
        est.fit(Xh["train"], ytr)
        record(key, "rf", est, Xh["train"], Xh["val"], Xh["test"],
               time.time() - t2, {"text_source": cfg.TEXT_SOURCE, "rep": "svd"})
        del Xh, est
        gc.collect()

        # ---------- #9/#10 lr + 稀疏 TF-IDF ----------
        key = f"lr_text_{scope}"
        print(f"[{key}] fitting ...", flush=True)
        t2 = time.time()
        Xl = {"train": features.hstack_sparse(Xb["train"], block.Xtr),
              "val": features.hstack_sparse(Xb["val"], Xt_val),
              "test": features.hstack_sparse(Xb["test"], Xt_test)}
        est = LogisticRegression(**cfg.LR_KW)
        est.fit(Xl["train"], ytr)
        thr, pt, row = record(key, "lr", est, Xl["train"], Xl["val"], Xl["test"],
                              time.time() - t2, {"text_source": cfg.TEXT_SOURCE, "rep": "sparse"})
        if scope == "full":
            _save_wrapper("model_lr_text_sparse", design, block, est, thr, row, te, pt,
                          rep="sparse", family="lr")
        del Xl, est, Xt_val, Xt_test, Z_val, Z_test
        gc.collect()

        del Xb, design, block
        gc.collect()

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(cfg.RES, "matrix_raw.csv"), index=False, encoding="utf-8-sig")
    cfg.save_json({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "blocks": blocks,
                   "timings": {r["key"]: r["fit_seconds"] for r in rows}},
                  os.path.join(cfg.RES, "text_blocks.json"))
    print("\n=== text matrix ===", flush=True)
    print(res[["key", "n_features", "AUC", "AP", "Brier", "threshold", "fit_seconds"]]
          .to_string(index=False), flush=True)
    print(f"[done] {time.time() - t0:.1f}s", flush=True)


def _save_wrapper(stem: str, design, block, est, thr: float, row: dict,
                  te: pd.DataFrame, pt: np.ndarray, rep: str, family: str) -> None:
    """保存可部署模型（18 特征 + 文本块）+ schema，并验证包装器与直接预测一致。"""
    svd = block.svd if rep == "svd" else None
    model = features.TextModel(features.BASE_COLS, design.pre, block.vec, svd, est, thr)
    path = os.path.join(cfg.MODELS, f"{stem}.joblib")
    joblib.dump(model, path, compress=3)
    pt_w = model.predict_proba(te)[:, 1]
    max_dev = float(np.max(np.abs(pt_w - pt)))
    schema = {
        "package": "IS6400_Kickstarter_交付包_完整",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": stem, "family": family, "rep": rep, "text_source": cfg.TEXT_SOURCE,
        "svd_dim": cfg.SVD_DIM if rep == "svd" else None,
        "tfidf": cfg.TFIDF_PARAMS,
        "base_features": features.BASE_COLS,
        "numeric": cfg.NUMERIC, "categorical": cfg.CATEGORICAL,
        "required_text_columns": ["name_clean", "blurb_clean"],
        "protocol": {"fit": "yr<=2020", "threshold": "val 2021-2022", "eval": "test yr>=2023"},
        "threshold": float(thr),
        "metrics_test": {"AUC": row["AUC"], "AP": row["AP"], "Brier": row["Brier"],
                         "F1": row["F1"], "top_rate": row["top_rate"], "lift": row["lift"]},
        "n_train": int(row["n_train"]), "n_test_own": int(row["n_test"]),
        "n_features": int(row["n_features"]), "fit_seconds": row["fit_seconds"],
        "artifact": path, "wrapper_max_abs_prediction_deviation": max_dev,
        "sklearn_version": __import__("sklearn").__version__,
    }
    cfg.save_json(schema, os.path.join(cfg.MODELS, f"{stem}_schema.json"))
    print(f"[save] {path} ({os.path.getsize(path) / 1e6:.2f} MB) wrapper_dev={max_dev:.2e}", flush=True)


if __name__ == "__main__":
    main()
