# -*- coding: utf-8 -*-
"""run_baseline.py — #1–#6：18 特征基线（hgb/lr/rf × full/usd）。

职责：
  1. 用交付包同款超参重训 6 个基线；
  2. G1 对账：与 03_模型文件/*_schema.json 的 published 指标比，容差 1e-4；
  3. 保存 test 预测（含 full 模型在 usd / 非 usd 行上的切片），供配对比较。
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import cfg
import features


def make_est(algo: str):
    if algo == "lr":
        return LogisticRegression(**cfg.LR_KW)
    if algo == "hgb":
        return HistGradientBoostingClassifier(**cfg.HGB_KW)
    if algo == "rf":
        return RandomForestClassifier(**cfg.RF_KW)
    raise ValueError(algo)


def main() -> None:
    t0 = time.time()
    rp = cfg.load_rp()
    df = features.load_prepared()
    rows, timings = [], {}

    mask_usd = None
    for scope in ("full", "usd"):
        tr, va, te = features.frames(df, scope)
        print(f"[{scope}] train={len(tr):,} val={len(va):,} test={len(te):,}", flush=True)
        design = features.BaseDesign().fit(tr)
        Xtr, Xva, Xte = design.transform(tr), design.transform(va), design.transform(te)
        ytr, yva, yte = features.y_of(tr), features.y_of(va), features.y_of(te)
        print(f"[{scope}] base design {Xtr.shape} (train) / {Xte.shape} (test)", flush=True)

        if scope == "full":
            mask_usd = te["is_usd"].values.astype(bool)
            np.save(os.path.join(cfg.PRED_DIR, "_y_all.npy"), yte.astype(np.int8))
            np.save(os.path.join(cfg.PRED_DIR, "_y_usd.npy"), yte[mask_usd].astype(np.int8))
            np.save(os.path.join(cfg.PRED_DIR, "_y_nonusd.npy"), yte[~mask_usd].astype(np.int8))
            np.save(os.path.join(cfg.PRED_DIR, "_mask_usd.npy"), mask_usd)
            np.save(os.path.join(cfg.PRED_DIR, "_test_ids.npy"), te["id"].values.astype(str))
            meta = te[["id", "is_usd", "yr", "succ", "category_parent_name"]].reset_index(drop=True)
            cmap = pd.read_parquet(cfg.CREATOR_MAP)
            meta = meta.merge(cmap, on="id", how="left")
            assert meta["creator_id"].notna().all(), "creator_id 映射缺失"
            meta.to_parquet(os.path.join(cfg.RES, "test_meta.parquet"), index=False)
        else:
            assert np.array_equal(te["is_usd"].values.astype(bool), np.ones(len(te), bool))

        for algo in ("hgb", "lr", "rf"):
            key = f"base_{algo}_{scope}"
            print(f"[{key}] fitting ...", flush=True)
            t1 = time.time()
            est = make_est(algo)
            est.fit(Xtr, ytr)
            pv = est.predict_proba(Xva)[:, 1]
            thr = rp.pick_threshold(yva, pv)
            pt = est.predict_proba(Xte)[:, 1]
            secs = time.time() - t1
            timings[key] = round(secs, 1)

            m = rp.evaluate(yte, pt, thr)
            td = rp.top_decile(yte, pt)
            pub = cfg.PUBLISHED[(algo, scope)]
            rows.append({
                "key": key, "algo": algo, "scope": scope, "kind": "base",
                "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                "n_features": int(Xtr.shape[1]), "fit_seconds": round(secs, 1),
                "threshold": float(thr),
                "AUC": float(m["ROC_AUC"]), "AP": float(m["PR_AUC"]),
                "Brier": float(m["Brier_Score"]), "F1": float(m["F1"]),
                "Accuracy": float(m["Accuracy"]), "Precision": float(m["Precision"]),
                "Recall": float(m["Recall"]), "base_rate": float(m["base_rate"]),
                "top_rate": float(td["top_rate"]), "lift": float(td["lift"]),
                "published_AUC": pub["AUC"], "published_AP": pub["AP"], "published_Brier": pub["Brier"],
                "published_thr": pub["thr"],
                "dAUC": float(m["ROC_AUC"]) - pub["AUC"],
                "dAP": float(m["PR_AUC"]) - pub["AP"],
                "dBrier": float(m["Brier_Score"]) - pub["Brier"],
            })
            print(f"[{key}] AUC={m['ROC_AUC']:.10f} (pub {pub['AUC']:.10f}, "
                  f"d={m['ROC_AUC'] - pub['AUC']:+.2e}) thr={thr:.3f} n_feat={Xtr.shape[1]} "
                  f"{secs:.1f}s", flush=True)

            if scope == "full":
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__all.npy"), pt.astype(np.float32))
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__usd.npy"), pt[mask_usd].astype(np.float32))
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__nonusd.npy"), pt[~mask_usd].astype(np.float32))
            else:
                np.save(os.path.join(cfg.PRED_DIR, f"{key}__usd.npy"), pt.astype(np.float32))
            del est, pv, pt
            gc.collect()

        del Xtr, Xva, Xte, design
        gc.collect()

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(cfg.RES, "baseline_metrics.csv"), index=False, encoding="utf-8-sig")

    # ---------------- G1 对账门 ----------------
    ok_all = True
    check_rows = []
    for _, r in res.iterrows():
        ok = (abs(r["dAUC"]) <= cfg.TOL and abs(r["dAP"]) <= cfg.TOL and abs(r["dBrier"]) <= cfg.TOL)
        ok_all &= bool(ok)
        check_rows.append({"key": r["key"], "dAUC": r["dAUC"], "dAP": r["dAP"],
                           "dBrier": r["dBrier"], "thr": r["threshold"],
                           "published_thr": r["published_thr"], "ok": bool(ok)})
    chk = pd.DataFrame(check_rows)
    chk.to_csv(os.path.join(cfg.RES, "g1_baseline_check.csv"), index=False, encoding="utf-8-sig")
    cfg.save_json({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "tol": cfg.TOL,
                   "all_ok": bool(ok_all), "rows": check_rows, "timings": timings},
                  os.path.join(cfg.RES, "g1_baseline_check.json"))
    print("\n=== G1 对账 ===", flush=True)
    print(chk.to_string(index=False), flush=True)
    print(f"G1 all_ok={ok_all} elapsed={time.time() - t0:.1f}s", flush=True)
    if not ok_all:
        raise SystemExit("G1 baseline reconciliation FAILED")


if __name__ == "__main__":
    main()
