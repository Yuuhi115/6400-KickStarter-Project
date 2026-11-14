# -*- coding: utf-8 -*-
"""evaluate.py — 指标、配对增量、按 creator 整簇 bootstrap、跨行集与决策影响。

输出：
  02_结果/matrix_metrics.csv      12 个配置的 test 指标（与已发布基线并排）
  02_结果/deltas_bootstrap.csv    每个配对 × 行集 × 抽样方案 的增量与 95%CI
  02_结果/cross_rowset.csv        跨行集（all / usd / 非usd）增量
  02_结果/decision_impact.csv     主配置的阈值穿越、top-10% 重合度、|Δp| 分位
  02_结果/decision.json           预注册判读
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import cfg

N_BOOT = 1000
SEED = 42
PRED = cfg.PRED_DIR


def load_arrays():
    y = {k: np.load(os.path.join(PRED, f"_y_{k}.npy")).astype(int)
         for k in ("all", "usd", "nonusd")}
    mask = np.load(os.path.join(PRED, "_mask_usd.npy"))
    meta = pd.read_parquet(os.path.join(cfg.RES, "test_meta.parquet"))
    prep_len = pd.read_parquet(cfg.PREPARED, columns=["id", "m_blurb_len"])
    meta = meta.merge(prep_len, on="id", how="left", validate="one_to_one")
    assert len(meta) == len(y["all"])
    return y, mask, meta


def load_pred(key: str, rowset: str) -> np.ndarray:
    return np.load(os.path.join(PRED, f"{key}__{rowset}.npy")).astype(np.float64)


def point_metrics(y, p) -> dict:
    return {"AUC": float(roc_auc_score(y, p)),
            "AP": float(average_precision_score(y, p)),
            "Brier": float(brier_score_loss(y, p)),
            "n": int(len(y)), "base_rate": float(np.mean(y))}


class BootCurve:
    """固定 p 的排序与并列分组，之后每次重抽只更新权重 → O(n) 计算加权 AUC。"""

    def __init__(self, p: np.ndarray):
        order = np.argsort(-p, kind="mergesort")
        ps = p[order]
        _, gid = np.unique(ps, return_inverse=True)
        self.order = order
        self.gid = gid.astype(np.int64)
        self.G = int(gid.max()) + 1

    def auc(self, y_sorted, w_sorted) -> float:
        pos = w_sorted * y_sorted
        neg = w_sorted * (1 - y_sorted)
        pg = np.bincount(self.gid, weights=pos, minlength=self.G)
        ng = np.bincount(self.gid, weights=neg, minlength=self.G)
        cn = np.cumsum(ng) - ng
        num = float(np.sum(pg * cn) + 0.5 * np.sum(pg * ng))
        den = float(pg.sum() * ng.sum())
        return num / den if den > 0 else float("nan")


def boot_delta(y, p_new, p_base, clusters, n_boot=N_BOOT, seed=SEED, scheme="cluster"):
    """返回 ΔAUC 的 bootstrap 分布统计。cluster: 按 creator 整簇抽；iid: 行级抽。"""
    rng = np.random.RandomState(seed)
    n = len(y)
    curve_new, curve_base = BootCurve(p_new), BootCurve(p_base)
    yn = y[curve_new.order].astype(np.float64)
    yb = y[curve_base.order].astype(np.float64)
    pn = p_new[curve_new.order].astype(np.float64)
    pb = p_base[curve_base.order].astype(np.float64)

    if scheme == "cluster":
        _, codes = np.unique(clusters, return_inverse=True)
        K = int(codes.max()) + 1
        codes_n, codes_b = codes[curve_new.order], codes[curve_base.order]
        probs = np.full(K, 1.0 / K)
    else:
        K = n
        codes_n = codes_b = None
        probs = np.full(n, 1.0 / n)

    deltas = np.empty(n_boot)
    briers = np.empty(n_boot)
    for i in range(n_boot):
        cnt = rng.multinomial(K if scheme == "cluster" else n, probs)
        if scheme == "cluster":
            w_n = cnt[codes_n].astype(np.float64)
            w_b = cnt[codes_b].astype(np.float64)
        else:
            # 行级抽样：cnt 按原始行序，必须各自映射回「按分数重排后」的顺序
            w_n = cnt[curve_new.order].astype(np.float64)
            w_b = cnt[curve_base.order].astype(np.float64)
        a_new = curve_new.auc(yn, w_n)
        a_base = curve_base.auc(yb, w_b)
        deltas[i] = a_new - a_base
        # 主配置 Brier 增量（加权）
        m_new = float(np.sum(w_n * (pn - yn) ** 2) / np.sum(w_n))
        m_base = float(np.sum(w_b * (pb - yb) ** 2) / np.sum(w_b))
        briers[i] = m_new - m_base

    deltas = deltas[~np.isnan(deltas)]
    briers = briers[~np.isnan(briers)]
    return {
        "n_boot": int(len(deltas)),
        "n_clusters": int(K) if scheme == "cluster" else None,
        "dAUC": float(point_metrics(y, p_new)["AUC"] - point_metrics(y, p_base)["AUC"]),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
        "p_delta_gt0": float((deltas > 0).mean()),
        "dAUC_boot_mean": float(deltas.mean()),
        "dBrier": float(point_metrics(y, p_new)["Brier"] - point_metrics(y, p_base)["Brier"]),
        "dBrier_ci_low": float(np.percentile(briers, 2.5)),
        "dBrier_ci_high": float(np.percentile(briers, 97.5)),
    }


def main() -> None:
    t0 = time.time()
    rp = cfg.load_rp()
    y, mask, meta = load_arrays()
    base_metrics = pd.read_csv(os.path.join(cfg.RES, "baseline_metrics.csv"))
    text_metrics = pd.read_csv(os.path.join(cfg.RES, "matrix_raw.csv"))
    all_metrics = pd.concat([base_metrics, text_metrics], ignore_index=True)
    thr = dict(zip(all_metrics["key"], all_metrics["threshold"]))
    y_of = {"all": y["all"], "usd": y["usd"], "nonusd": y["nonusd"]}

    # ---------- 1. matrix_metrics ----------
    rows = []
    for _, r in all_metrics.iterrows():
        own = "all" if r["scope"] == "full" else "usd"
        p = load_pred(r["key"], own)
        m = point_metrics(y_of[own], p)
        td = rp.top_decile(y_of[own], p)
        row = {"key": r["key"], "algo": r["algo"], "scope": r["scope"], "kind": r["kind"],
               "n_test": m["n"], "base_rate": m["base_rate"], "threshold": r["threshold"],
               "n_features": r["n_features"], "fit_seconds": r["fit_seconds"],
               "AUC": m["AUC"], "AP": m["AP"], "Brier": m["Brier"],
               "F1": rp.evaluate(y_of[own], p, r["threshold"])["F1"],
               "top_rate": td["top_rate"], "lift": td["lift"]}
        if r["kind"] == "base":
            pub = cfg.PUBLISHED[(r["algo"], r["scope"])]
            row.update({"published_AUC": pub["AUC"], "published_AP": pub["AP"],
                        "published_Brier": pub["Brier"]})
        rows.append(row)
    mm = pd.DataFrame(rows)
    mm.to_csv(os.path.join(cfg.RES, "matrix_metrics.csv"), index=False, encoding="utf-8-sig")
    print(mm[["key", "n_features", "AUC", "AP", "Brier", "lift"]].to_string(index=False), flush=True)

    # ---------- 2. 配对增量 + bootstrap ----------
    pairs = []
    for algo in ("hgb", "lr", "rf"):
        for rs in ("all", "usd", "nonusd"):
            pairs.append((f"base_{algo}_full", f"{algo}_text_full", rs))
    for algo in ("hgb", "lr", "rf"):
        pairs.append((f"base_{algo}_usd", f"{algo}_text_usd", "usd"))

    deltas = []
    for k_base, k_new, rs in pairs:
        yy = y_of[rs]
        p_base, p_new = load_pred(k_base, rs), load_pred(k_new, rs)
        assert len(p_base) == len(yy) == len(p_new), (k_base, k_new, rs)
        cl = _clusters_for(meta, mask, rs)
        for scheme in ("cluster", "iid"):
            st = boot_delta(yy, p_new, p_base, cl, scheme=scheme)
            row = {"base_key": k_base, "new_key": k_new, "rowset": rs, "scheme": scheme,
                   "n": len(yy),
                   "dAP": float(point_metrics(yy, p_new)["AP"] - point_metrics(yy, p_base)["AP"]),
                   **{k: v for k, v in st.items()}}
            deltas.append(row)
            print(f"[boot] {k_new:16s} {rs:7s} {scheme:7s} dAUC={st['dAUC']:+.6f} "
                  f"CI[{st['ci95_low']:+.6f},{st['ci95_high']:+.6f}] p={st['p_two_sided']:.4f} "
                  f"dBrier={st['dBrier']:+.6f}", flush=True)
    db = pd.DataFrame(deltas)
    db.to_csv(os.path.join(cfg.RES, "deltas_bootstrap.csv"), index=False, encoding="utf-8-sig")

    # ---------- 3. 跨行集 ----------
    cr = db[db["scheme"] == "cluster"][
        ["base_key", "new_key", "rowset", "n", "dAUC", "ci95_low", "ci95_high", "p_two_sided",
         "dAP", "dBrier", "dBrier_ci_low", "dBrier_ci_high"]].copy()
    cr.to_csv(os.path.join(cfg.RES, "cross_rowset.csv"), index=False, encoding="utf-8-sig")

    # ---------- 4. 决策影响（主配置） ----------
    main_new, main_base = cfg.MAIN, "base_hgb_full"
    p_new, p_base = load_pred(main_new, "all"), load_pred(main_base, "all")
    thr_new, thr_base = thr[main_new], thr[main_base]
    k = max(1, int(np.ceil(len(y["all"]) * 0.10)))
    idx_new = np.argsort(-p_new, kind="mergesort")[:k]
    idx_base = np.argsort(-p_base, kind="mergesort")[:k]
    jac = len(set(idx_new.tolist()) & set(idx_base.tolist())) / len(set(idx_new.tolist()) | set(idx_base.tolist()))
    dp = p_new - p_base
    bl = meta["m_blurb_len"].astype(float).values
    ok_len = ~np.isnan(bl)

    def _flips(tn: float, tb: float) -> dict:
        up = int(((p_new >= tn) & (p_base < tb)).sum())
        dn = int(((p_new < tn) & (p_base >= tb)).sum())
        return {"threshold_new": float(tn), "threshold_base": float(tb),
                "up": up, "down": dn, "total": up + dn,
                "rate_total": (up + dn) / len(y["all"]),
                "rate_up": up / len(y["all"]), "rate_down": dn / len(y["all"]),
                "both": int(((p_new >= tn) & (p_base >= tb)).sum()),
                "none": int(((p_new < tn) & (p_base < tb)).sum())}

    conv = {"own_val_thresholds": _flips(thr_new, thr_base),
            "common_0.38": _flips(0.38, 0.38),
            "common_0.39": _flips(0.39, 0.39),
            "common_0.395": _flips(0.395, 0.395)}
    pd.DataFrame([{"convention": k2, **v} for k2, v in conv.items()]) \
        .to_csv(os.path.join(cfg.RES, "decision_flips.csv"), index=False, encoding="utf-8-sig")

    di = {
        "main_config": main_new, "base_config": main_base,
        "thresholds": {"new": float(thr_new), "base": float(thr_base)},
        "n_test": int(len(y["all"])),
        "flagged_new": int((p_new >= thr_new).sum()), "flagged_base": int((p_base >= thr_base).sum()),
        "flagged_both": int(((p_new >= thr_new) & (p_base >= thr_base)).sum()),
        "cross_up": conv["own_val_thresholds"]["up"],
        "cross_down": conv["own_val_thresholds"]["down"],
        "cross_total": conv["own_val_thresholds"]["total"],
        "cross_rate": conv["own_val_thresholds"]["rate_total"],
        "cross_up_common039": conv["common_0.39"]["up"],
        "cross_down_common039": conv["common_0.39"]["down"],
        "cross_total_common039": conv["common_0.39"]["total"],
        "cross_rate_common039": conv["common_0.39"]["rate_total"],
        "flips": conv,
        "top10_k": int(k), "top10_overlap": int(len(set(idx_new.tolist()) & set(idx_base.tolist()))),
        "top10_jaccard": float(jac),
        "abs_dp": {f"p{q}": float(np.percentile(np.abs(dp), q)) for q in (50, 90, 99)},
        "abs_dp_max": float(np.abs(dp).max()),
        "corr_absdp_with_blurb_len": float(
            np.corrcoef(np.abs(dp)[ok_len], bl[ok_len])[0, 1]),
    }
    pd.DataFrame([{k2: (json.dumps(v) if isinstance(v, dict) else v) for k2, v in di.items()}]) \
        .to_csv(os.path.join(cfg.RES, "decision_impact.csv"), index=False, encoding="utf-8-sig")

    # ---------- 5. 预注册判读 ----------
    row_main = db[(db["new_key"] == main_new) & (db["scheme"] == "cluster") & (db["rowset"] == "all")].iloc[0]
    d_auc, lo, d_brier = float(row_main["dAUC"]), float(row_main["ci95_low"]), float(row_main["dBrier"])
    if d_auc > 0 and lo > 0 and d_brier <= 0.002:
        verdict = "adopt_significant_small" if d_auc < 0.005 else "adopt"
    else:
        verdict = "appendix"
    decision = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_config": main_new, "base_config": main_base,
        "rule": "dAUC>0 且 CI下界>0 且 dBrier<=+0.002 → 可并入；0<dAUC<0.005 → '统计显著、实用有限'；"
                "CI 含 0 或 dAUC<=0 → 附录",
        "observed": {"dAUC": d_auc, "ci95_low": lo, "ci95_high": float(row_main["ci95_high"]),
                     "p_two_sided": float(row_main["p_two_sided"]), "dBrier": d_brier,
                     "dBrier_ci_low": float(row_main["dBrier_ci_low"]),
                     "dBrier_ci_high": float(row_main["dBrier_ci_high"])},
        "verdict": verdict,
        "decision_impact": di,
        "published_baseline": {k3: v for k3, v in cfg.PUBLISHED[("hgb", "full")].items()},
    }
    cfg.save_json(decision, os.path.join(cfg.RES, "decision.json"))
    print(f"\n[decision] {verdict} dAUC={d_auc:+.6f} CI[{lo:+.6f},...] dBrier={d_brier:+.6f}", flush=True)
    print(f"[done] {time.time() - t0:.1f}s", flush=True)


def _clusters_for(meta: pd.DataFrame, mask: np.ndarray, rowset: str) -> np.ndarray:
    if rowset == "all":
        return meta["creator_id"].values.astype(str)
    if rowset == "usd":
        return meta["creator_id"].values.astype(str)[mask]
    return meta["creator_id"].values.astype(str)[~mask]


if __name__ == "__main__":
    main()
