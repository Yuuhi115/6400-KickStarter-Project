# -*- coding: utf-8 -*-
"""resave_models.py — 把 05_模型 里的 TextModel joblib 重存为「可独立加载」的 dict artifact。

背景：原 artifact 是 `features.TextModel` 的 pickle，加载时需要模块 `features`（进而 `cfg`），
在交付包里会 `ModuleNotFoundError` / `SystemExit`。改为 dict（只含 sklearn 对象）+ 独立的
`text_model.py`，即可在任意目录 `load_model(path)`。

校验：重存前后对 test 的预测逐位比对（应 ≤1e-6），并与 `02_结果/preds/*.npy` 对账。
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import cfg
import features
import text_model

STEMS = ["model_histgb_text", "model_lr_text_sparse"]


def _test_frame() -> pd.DataFrame:
    prep = pd.read_parquet(cfg.PREPARED)
    ids = np.load(os.path.join(cfg.PRED_DIR, "_test_ids.npy")).astype(str)
    df = prep[prep["id"].astype(str).isin(set(ids))].copy()
    assert len(df) == len(ids), (len(df), len(ids))
    return df.set_index(df["id"].astype(str)).loc[ids].reset_index(drop=True)


def main() -> None:
    te = _test_frame()
    y = pd.to_numeric(te["succ"], errors="coerce").fillna(0).astype(int).values
    for stem in STEMS:
        path = os.path.join(cfg.MODELS, f"{stem}.joblib")
        old = joblib.load(path)
        if isinstance(old, dict):            # 已重存过：从 artifact 还原，保证幂等
            old = text_model.TextModel.from_artifact(old)
        elif isinstance(old, features.TextModel):   # 原始 pickle：用旧接口算基准预测
            old = text_model.TextModel(
                base_cols=old.base_cols, pre=old.pre, vec=old.vec, svd=old.svd, clf=old.clf,
                threshold=old.threshold, text_mode="sparse" if old.svd is None else "svd")
        assert isinstance(old, text_model.TextModel), type(old)
        p_old = old.predict_proba(te)[:, 1]

        art = text_model.TextModel(
            base_cols=old.base_cols, pre=old.pre, vec=old.vec, svd=old.svd, clf=old.clf,
            threshold=old.threshold,
            text_mode="sparse" if old.svd is None else "svd",
            algorithm="hgb" if "histgb" in stem else "lr", scope="full",
            meta={"text_source": cfg.TEXT_SOURCE, "rep": "sparse" if old.svd is None else "svd",
                  "trained_on": "yr<=2020, val 2021-2022 (threshold only), test yr>=2023",
                  "tfidf": cfg.TFIDF_PARAMS, "svd_dim": cfg.SVD_DIM if old.svd is not None else None})
        joblib.dump(art.to_artifact(), path, compress=3)

        new = text_model.load_model(path)          # 走独立加载路径，验证可用
        p_new = new.predict_proba(te)[:, 1]
        dev = float(np.max(np.abs(p_new - p_old)))
        flag_same = bool((new.predict(te) == (p_old >= old.threshold).astype(int)).all())
        art_again = joblib.load(path)
        print(f"[{stem}] size={os.path.getsize(path) / 1e6:.2f} MB "
              f"max|Δp|={dev:.2e} flag_same={flag_same} format={art_again['_format']}", flush=True)
        assert dev <= 1e-6 and flag_same, f"{stem}: dev={dev} flag_same={flag_same}"

        # 与落盘的 test 预测对账（容忍 1e-4：训练时 float32 基座矩阵与本模块 float64→float32 路径
        # 在极少数行上会有 1e-5 量级差异；AUC 层面为 1e-10 级，见下）
        npz = os.path.join(cfg.PRED_DIR, f"{'hgb' if 'histgb' in stem else 'lr'}_text_full__all.npy")
        p_ref = np.load(npz).astype(np.float64)
        dev_ref = float(np.max(np.abs(p_new - p_ref)))
        from sklearn.metrics import roc_auc_score
        auc_ref = float(roc_auc_score(y, p_ref))
        auc_new = float(roc_auc_score(y, p_new))
        print(f"[{stem}] vs preds npy: max|Δp|={dev_ref:.2e} AUC {auc_ref:.10f} vs {auc_new:.10f}", flush=True)
        assert dev_ref <= 1e-4 and abs(auc_new - auc_ref) <= 1e-7

        # 更新 schema：记录 artifact 格式与加载方式
        sc_path = os.path.join(cfg.MODELS, f"{stem}_schema.json")
        sc = cfg.load_json(sc_path) if hasattr(cfg, "load_json") else None
        if sc is None:
            import json
            sc = json.load(open(sc_path, encoding="utf-8"))
        import json
        sc.update({
            "artifact_format": text_model.FORMAT,
            "artifact_is_pure_sklearn_dict": True,
            "load_with": "from text_model import load_model; m = load_model('<path>.joblib')",
            "requires": ["numpy", "pandas", "scipy", "scikit-learn(1.5.1)"],
            "wrapper_max_abs_prediction_deviation": dev,
            "revised": time.strftime("%Y-%m-%d %H:%M：由 features.TextModel pickle 改为 dict artifact（可独立加载）"),
        })
        with open(sc_path, "w", encoding="utf-8", newline="") as fh:
            json.dump(sc, fh, ensure_ascii=False, indent=2)
        print(f"[{stem}] schema updated", flush=True)

    # 把独立模块放进 05_模型，使包内 05_模型 自包含
    import shutil
    dst = os.path.join(cfg.MODELS, "text_model.py")
    shutil.copy2(os.path.join(cfg.SCRIPTS, "text_model.py"), dst)
    print(f"[copy] text_model.py → {dst}", flush=True)


if __name__ == "__main__":
    main()
