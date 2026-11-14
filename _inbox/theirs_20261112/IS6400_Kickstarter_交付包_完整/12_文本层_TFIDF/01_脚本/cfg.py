# -*- coding: utf-8 -*-
"""cfg.py — v2 文本层全局配置：路径、口径常量、已发布指标、超参、工具函数。

纪律：
  * 只读交付包与归档目录，绝不写入；
  * 所有路径基于本文件位置推导，换机器只需整包搬移。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)                      # 13_TFIDF文本层_v2
PROJ = os.path.dirname(WS)                      # IS6400_期末进度包_20261102(1)
PKG = os.path.join(PROJ, "IS6400_Kickstarter_交付包_完整")
if not os.path.isdir(PKG):   # 说明这是在交付包内的审计副本（12_文本层_TFIDF/01_脚本）里运行
    raise SystemExit(f"找不到交付包目录 {PKG}；本脚本按工作区布局解析路径，"
                     f"请在项目工作区 13_TFIDF文本层_v2/01_脚本 下运行（包内副本仅供审计阅读）。")
ARCH = os.path.join(PROJ, "_archive_12_文本层_20261112")
MODELED = os.path.join(PROJ, "建模交付")
if not os.path.isdir(MODELED):                  # 2026-11-12 用户把该目录改名为「TF-IDF前的建模交付」
    _alt = os.path.join(PROJ, "TF-IDF前的建模交付")
    if os.path.isdir(_alt):
        MODELED = _alt

BASE_PARQUET = os.path.join(ARCH, "work", "base_model_ready.parquet")
KS_CLEAN = os.path.join(PKG, "ks_clean.csv")
SPLIT_ASSIGN = os.path.join(MODELED, "02_数据切分", "split_assignment.parquet")
SCHEMA_DIR = os.path.join(MODELED, "03_模型文件")
RP_PATH = os.path.join(PKG, "09_分析脚本", "replicate_pilot.py")

RES = os.path.join(WS, "02_结果")
PRED_DIR = os.path.join(RES, "preds")
FIG = os.path.join(WS, "03_图表")
REPORT = os.path.join(WS, "04_报告")
MODELS = os.path.join(WS, "05_模型")
SCRIPTS = HERE

PREPARED = os.path.join(RES, "prepared_modelable.parquet")
CREATOR_MAP = os.path.join(RES, "creator_id.parquet")

for _d in (RES, PRED_DIR, FIG, REPORT, MODELS):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- protocol
# 三段 ours 协议：train yr<=2020 / val 2021-2022 / test yr>=2023
EXPECT = {
    "rows_modelable": 604629,
    "rows_usd_modelable": 415619,
    "rows_nonusd_modelable": 189010,
    "train_full": 443693, "val_full": 60467, "test_full": 100469,
    "train_usd": 327642, "val_usd": 33542, "test_usd": 54435,
    "nonusd_test": 46034,
    "base_rate_train": 0.4397162001654297,
    "base_rate_test": 0.6441190815077288,
    "ks_clean_sha256": "aaf7a89e08e91cdefb4b7b56a079e74f358ae3c0b0548471ced8565396b70b3f",
    "ks_clean_bytes": 710301125,
    "name_na": 5,
    "blurb_na": 13,
}
SCOPES = ("full", "usd")
ROWSETS = ("all", "usd", "nonusd")
TOL = 1e-4                      # G1 基线对账容差

NUMERIC = ["m_log_goal_usd", "m_dur_days_exact", "launch_quarter", "dow",
           "m_name_len", "m_blurb_len", "m_blurb_words", "name_word_count",
           "creator_previous_count", "creator_prev_success_rate",
           "prep_days", "yr", "mon"]
CATEGORICAL = ["category_parent_name", "subcategory_key", "country_final",
               "location_type", "currency"]

TFIDF_PARAMS = dict(max_features=3000, min_df=20, stop_words="english",
                    ngram_range=(1, 2), sublinear_tf=True,
                    strip_accents="unicode", lowercase=True)
SVD_DIM = 128
TEXT_SOURCE = "both"

HGB_KW = dict(learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.10,
              n_iter_no_change=20, random_state=42)
RF_KW = dict(n_estimators=150, min_samples_leaf=5, n_jobs=1, random_state=42)
LR_KW = dict(max_iter=3000, solver="liblinear", random_state=42)

# 12 次拟合清单：(key, algo, scope, kind)
CONFIGS = [
    ("base_hgb_full", "hgb", "full", "base"),
    ("base_lr_full", "lr", "full", "base"),
    ("base_rf_full", "rf", "full", "base"),
    ("base_hgb_usd", "hgb", "usd", "base"),
    ("base_lr_usd", "lr", "usd", "base"),
    ("base_rf_usd", "rf", "usd", "base"),
    ("hgb_text_full", "hgb", "full", "text"),
    ("hgb_text_usd", "hgb", "usd", "text"),
    ("lr_text_full", "lr", "full", "text"),
    ("lr_text_usd", "lr", "usd", "text"),
    ("rf_text_full", "rf", "full", "text"),
    ("rf_text_usd", "rf", "usd", "text"),
]
TEXT_CANDIDATES = {("hgb", "full"): "hgb_text_full", ("hgb", "usd"): "hgb_text_usd",
                   ("lr", "full"): "lr_text_full", ("lr", "usd"): "lr_text_usd",
                   ("rf", "full"): "rf_text_full", ("rf", "usd"): "rf_text_usd"}

MAIN = "hgb_text_full"          # 预注册判读主配置


# ---------------------------------------------------------------- helpers
def load_rp():
    """动态载入交付包的 replicate_pilot.py（权威口径实现）。"""
    if "replicate_pilot" in sys.modules:
        return sys.modules["replicate_pilot"]
    spec = importlib.util.spec_from_file_location("replicate_pilot", RP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["replicate_pilot"] = mod
    return mod


def published() -> dict:
    """6 个已发布模型的 own-test 指标（来自 03_模型文件/*_schema.json）。"""
    out = {}
    for algo in ("hgb", "lr", "rf"):
        for scope in SCOPES:
            p = os.path.join(SCHEMA_DIR, f"model_{algo}_{scope}_schema.json")
            with open(p, encoding="utf-8") as fh:
                s = json.load(fh)
            m = s["metrics_own_test"]
            out[(algo, scope)] = {"AUC": float(m["AUC"]), "AP": float(m["AP"]),
                                  "Brier": float(m["Brier"]), "thr": float(s["threshold"]),
                                  "n_train": int(s["n_train"]), "n_test": int(s["n_test_own"]),
                                  "features": list(s["features"])}
    return out


PUBLISHED = published()


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)


def manifest(root: str) -> "list[dict]":
    """目录下所有文件的 (relpath, bytes, sha256)，按 relpath 排序。"""
    rows = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            rows.append({"relpath": rel, "bytes": os.path.getsize(full),
                         "sha256": sha256_file(full)})
    rows.sort(key=lambda r: r["relpath"])
    return rows
