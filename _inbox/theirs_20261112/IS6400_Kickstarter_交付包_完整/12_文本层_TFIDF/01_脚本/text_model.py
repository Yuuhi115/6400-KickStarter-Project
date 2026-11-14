# -*- coding: utf-8 -*-
"""text_model.py — 文本层模型的**可独立加载**包装（不依赖 cfg、不依赖工作区布局）。

用途：交付包 `12_文本层_TFIDF/05_模型/*.joblib` 是纯 sklearn 对象的 dict artifact，
      用本模块的 `load_model(path)` 即可重建预测器，不必把 `01_脚本/` 加进 `sys.path`：

    import sys; sys.path.insert(0, r"<...>/12_文本层_TFIDF/05_模型")
    from text_model import load_model
    m = load_model(r"<...>/model_histgb_text.joblib")
    p = m.predict_proba(df)[:, 1]        # df 需含 18 个入模特征 + name_clean/blurb_clean
    flag = m.predict(df)                 # 按 artifact 内 threshold 判定「达标」

artifact 结构（dict）：
    {"_format": "isf6400.textlayer/1", "algorithm": ..., "scope": ...,
     "base_cols": [...18], "pre": <sklearn Pipeline>, "vec": <TfidfVectorizer>,
     "svd": <TruncatedSVD or None>, "clf": <classifier>, "threshold": float,
     "text_mode": "sparse" | "svd", "meta": {...}}

CSV/parquet 读入后列名若只有 `name`/`blurb`，本模块会退化为 `name + " " + blurb`。
空文本（None/NaN/空串）记为空字符串，与训练时一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

FORMAT = "isf6400.textlayer/1"

BASE_COLS_18 = [
    "m_log_goal_usd", "m_dur_days_exact", "prep_days", "yr", "mon", "launch_quarter", "dow",
    "m_name_len", "m_blurb_len", "m_blurb_words", "name_word_count",
    "creator_previous_count", "creator_prev_success_rate",
    "category_parent_name", "subcategory_key", "country_final", "location_type", "currency",
]


def text_series(frame: pd.DataFrame) -> pd.Series:
    """训练时的文本拼接口径：name_clean + ' ' + blurb_clean（缺失则回退 name/blurb）。"""
    if "name_clean" in frame.columns and "blurb_clean" in frame.columns:
        name, blurb = frame["name_clean"], frame["blurb_clean"]
    else:
        name, blurb = frame.get("name", ""), frame.get("blurb", "")
        if not isinstance(name, pd.Series):
            name = pd.Series([""] * len(frame), index=frame.index)
        if not isinstance(blurb, pd.Series):
            blurb = pd.Series([""] * len(frame), index=frame.index)
    return (name.fillna("").astype(str) + " " + blurb.fillna("").astype(str)).astype(str)


class TextModel:
    """18 特征基座 + TF-IDF 文本块；svd=None 表示稀疏 TF-IDF 直接进线性模型。"""

    def __init__(self, base_cols, pre, vec, svd, clf, threshold=0.5,
                 text_mode="svd", algorithm="", scope="", meta=None):
        self.base_cols = list(base_cols)
        self.pre = pre
        self.vec = vec
        self.svd = svd
        self.clf = clf
        self.threshold = float(threshold)
        self.text_mode = text_mode
        self.algorithm = algorithm
        self.scope = scope
        self.meta = dict(meta or {})

    # ---------------------------------------------------------------- IO
    @classmethod
    def from_artifact(cls, art: dict) -> "TextModel":
        fmt = art.get("_format")
        if fmt != FORMAT:
            raise ValueError(f"unsupported artifact format: {fmt!r}（期望 {FORMAT!r}）")
        return cls(base_cols=art["base_cols"], pre=art["pre"], vec=art["vec"], svd=art["svd"],
                   clf=art["clf"], threshold=art["threshold"], text_mode=art["text_mode"],
                   algorithm=art.get("algorithm", ""), scope=art.get("scope", ""),
                   meta=art.get("meta"))

    def to_artifact(self) -> dict:
        return {"_format": FORMAT, "algorithm": self.algorithm, "scope": self.scope,
                "base_cols": self.base_cols, "pre": self.pre, "vec": self.vec,
                "svd": self.svd, "clf": self.clf, "threshold": self.threshold,
                "text_mode": self.text_mode, "meta": self.meta}

    # ---------------------------------------------------------------- predict
    def _design(self, frame: pd.DataFrame):
        missing = [c for c in self.base_cols if c not in frame.columns]
        if missing:
            raise KeyError(f"输入缺列：{missing}（需要 18 个入模特征）")
        Xb = np.asarray(self.pre.transform(frame[self.base_cols]), dtype=np.float32)
        Xt = self.vec.transform(text_series(frame)).astype(np.float32)
        if self.svd is not None:
            return np.hstack([Xb, np.asarray(self.svd.transform(Xt), dtype=np.float32)])
        return sp.hstack([sp.csr_matrix(Xb), Xt], format="csr")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._design(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= self.threshold).astype(int)

    # ---------------------------------------------------------------- 解释用
    def text_terms(self, top_k: int = 20):
        """按 LR 系数给出「更易达标 / 更不易达标」的词（仅稀疏模式、且 clf 有 coef_）。"""
        if self.svd is not None or not hasattr(self.clf, "coef_"):
            return None
        inv = {i: t for t, i in self.vec.vocabulary_.items()}
        coef = np.asarray(self.clf.coef_).ravel()
        n_base = coef.shape[0] - len(inv)
        body = coef[n_base:]
        order = np.argsort(body)
        return {"up": [(inv[i], float(body[i])) for i in order[::-1][:top_k]],
                "down": [(inv[i], float(body[i])) for i in order[:top_k]]}


def load_model(path: str) -> TextModel:
    """读取 joblib dict artifact → TextModel。"""
    import joblib
    return TextModel.from_artifact(joblib.load(path))
