# -*- coding: utf-8 -*-
"""features.py — 数据装载、三段切分、基座设计矩阵、文本块（TF-IDF + SVD）。

防漏纪律：
  * 基座 preprocessor 只在 train 上 fit；
  * TfidfVectorizer / TruncatedSVD 只在 train 上 fit，且显式断言行数与 id 不相交；
  * val 只用于选阈值，test 只用于评估。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import cfg

BASE_COLS = cfg.NUMERIC + cfg.CATEGORICAL


def load_prepared() -> pd.DataFrame:
    df = pd.read_parquet(cfg.PREPARED)
    df.attrs["numeric"] = list(cfg.NUMERIC)
    df.attrs["categorical"] = list(cfg.CATEGORICAL)
    return df


def frames(df: pd.DataFrame, scope: str):
    """按 ours 三段协议切分；scope='usd' 时先过滤 m_currency_usd。"""
    rp = cfg.load_rp()
    if scope == "full":
        sub = df
    elif scope == "usd":
        sub = df[df["is_usd"]].reset_index(drop=True)
    else:
        raise ValueError(scope)
    tr, va, te = rp.make_splits(sub, "ours")
    n = (len(tr), len(va), len(te))
    want = (cfg.EXPECT[f"train_{scope}"], cfg.EXPECT[f"val_{scope}"], cfg.EXPECT[f"test_{scope}"])
    assert n == want, f"split mismatch {scope}: {n} != {want}"
    return tr, va, te


def y_of(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["succ"], errors="coerce").fillna(0).astype(int).values


class BaseDesign:
    """18 特征 → 226 列稠密矩阵（median 填补 + 标准化；常量 'Unknown' + OneHot min_freq=20）。"""

    def __init__(self) -> None:
        rp = cfg.load_rp()
        self.pre = rp.preprocessor(cfg.NUMERIC, cfg.CATEGORICAL)
        self.fitted = False
        self.width = None

    def fit(self, train: pd.DataFrame) -> "BaseDesign":
        self.pre.fit(train[BASE_COLS])
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.fitted
        X = np.asarray(self.pre.transform(frame[BASE_COLS]), dtype=np.float32)
        if self.width is None:
            self.width = int(X.shape[1])
        return X


def text_series(frame: pd.DataFrame) -> pd.Series:
    return (frame["name_clean"].astype(str) + " " + frame["blurb_clean"].astype(str)).astype(str)


class TextBlock:
    """TF-IDF(3000) + 可选 SVD(128)，只在 train 上拟合。"""

    def __init__(self) -> None:
        self.vec = TfidfVectorizer(**cfg.TFIDF_PARAMS)
        self.svd = TruncatedSVD(n_components=cfg.SVD_DIM, random_state=42,
                                algorithm="randomized", n_iter=5)
        self.n_vocab = None
        self.evr = None
        self.nnz_per_doc = None
        self.Xtr = None      # train 的 TF-IDF（稀疏，float32）
        self.Ztr = None      # train 的 SVD 嵌入（稠密，float32）

    def fit(self, series_train: pd.Series, train_ids, other_ids) -> "TextBlock":
        n = len(series_train)
        assert n == len(train_ids), f"text/train rows differ: {n} vs {len(train_ids)}"
        inter = set(train_ids) & set(other_ids)
        assert not inter, f"train ids leak into val/test: {len(inter)}"
        X = self.vec.fit_transform(series_train).astype(np.float32)
        self.Xtr = X
        self.n_vocab = int(len(self.vec.vocabulary_))
        self.nnz_per_doc = float(X.nnz / X.shape[0])
        self.svd.fit(X)
        self.evr = float(self.svd.explained_variance_ratio_.sum())
        self.Ztr = np.asarray(self.svd.transform(X), dtype=np.float32)
        return self

    def sparse(self, series: pd.Series):
        return self.vec.transform(series).astype(np.float32)

    def dense(self, series: pd.Series) -> np.ndarray:
        return np.asarray(self.svd.transform(self.vec.transform(series)), dtype=np.float32)


class TextModel:
    """可部署包装：18 特征基座 + TF-IDF 文本块（sparse→LR / svd→树模型）。"""

    def __init__(self, base_cols, pre, vec, svd, clf, threshold, meta=None):
        self.base_cols = list(base_cols)
        self.pre = pre
        self.vec = vec
        self.svd = svd
        self.clf = clf
        self.threshold = float(threshold)
        self.meta = dict(meta or {})

    def _design(self, frame: pd.DataFrame):
        Xb = np.asarray(self.pre.transform(frame[self.base_cols]), dtype=np.float32)
        Xt = self.vec.transform(text_series(frame)).astype(np.float32)
        if self.svd is not None:
            return np.hstack([Xb, np.asarray(self.svd.transform(Xt), dtype=np.float32)])
        return sp.hstack([sp.csr_matrix(Xb), Xt], format="csr")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._design(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= self.threshold).astype(int)


def hstack_dense(base: np.ndarray, extra: np.ndarray) -> np.ndarray:
    return np.hstack([base, extra]).astype(np.float32)


def hstack_sparse(base: np.ndarray, extra) -> sp.csr_matrix:
    return sp.hstack([sp.csr_matrix(base), extra], format="csr")
