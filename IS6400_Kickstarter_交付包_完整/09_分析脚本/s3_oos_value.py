# -*- coding: utf-8 -*-
"""S3: out-of-sample cells + value-dimension tests (fix P2-2 / P2-3 / P2-1).

为什么要这个脚本（回答“实测效果应该在测试集里取”）：
  - 描述性基准可以用全量，但必须标注 in-sample；
  - **声称"效果/验证"只能来自测试期（yr>=2023, n=100,469）**；
  - ΔP 是模型反事实，不是实测效果，不能叫"提升"。

产出：
  11_结果与报告/cells_oos.parquet        仅测试期行重建的格子（真实样本外）
  11_结果与报告/S3_样本外与价值检验.md
  11_结果与报告/s3_oos_vs_insample.csv   同一格 in-sample vs OOS 达标率差
  11_结果与报告/s3_value_tests.csv       价值维度 bootstrap（按 creator 聚类）
  11_结果与报告/s3_cancel_sensitivity.csv 取消项目计入失败的敏感性
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
DATA_FULL = PKG / "08_建模就绪数据" / "ks_model_ready_v1.parquet"

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

GOAL_BINS = [0, 1_000, 5_000, 10_000, 20_000, 50_000, np.inf]
GOAL_LABELS = ["<=1k", "1-5k", "5-10k", "10-20k", "20-50k", ">50k"]
DUR_BINS = [0, 14, 30, 45, 60, np.inf]
DUR_LABELS = ["1-14d", "15-30d", "31-45d", "46-60d", "61d+"]
MIN_N = 300
RNG = np.random.RandomState(42)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _agg(g):
    """按 creator 聚合成 (n, 成功数 k, 成功项目募资额之和 pl)。

    注意 value = rate × 成功项目平均募资额 = pl / n，因此只需 n/k/pl 三个和。
    """
    t = g.assign(pl=g["succ"] * g["usd_pledged"].fillna(0.0))
    return t.groupby("creator_id").agg(n=("succ", "size"), k=("succ", "sum"),
                                       pl=("pl", "sum"))[["n", "k", "pl"]].to_numpy(float)


def boot_delta(a_rows, b_rows, reps=200):
    """Δrate 与 Δvalue 的 95% CI：两组独立按 creator 整簇重抽后相减。

    【修正记录】旧实现把两组拼成一份 (d_rate=+succ / −succ) 再除以合并总数，
    当两组 n 相差大时会被大组主导，给出 −48.8pp 之类的假值。
    """
    A, B = _agg(a_rows), _agg(b_rows)

    def stat(arr):
        n, k, pl = arr.sum(axis=0)
        return (k / n if n else np.nan), (pl / n if n else np.nan)

    r_a, v_a = stat(A)
    r_b, v_b = stat(B)
    idx_a, idx_b = np.arange(len(A)), np.arange(len(B))
    dr, dv = [], []
    for _ in range(reps):
        sa = A[RNG.choice(idx_a, len(idx_a), replace=True)].sum(axis=0)
        sb = B[RNG.choice(idx_b, len(idx_b), replace=True)].sum(axis=0)
        if sa[0] == 0 or sb[0] == 0:
            continue
        dr.append(sa[1] / sa[0] - sb[1] / sb[0])
        dv.append(sa[2] / sa[0] - sb[2] / sb[0])
    dr, dv = np.array(dr), np.array(dv)
    return (r_a - r_b, np.percentile(dr, 2.5), np.percentile(dr, 97.5),
            v_a - v_b, np.percentile(dv, 2.5), np.percentile(dv, 97.5))


def main():
    print("[1/5] 载入 ...", flush=True)
    df = rp.build_frame(rp.DATA["full"], use_prep_days=True)
    num, cat = df.attrs["numeric"], df.attrs["categorical"]
    feats = num + cat
    train, val, test = rp.make_splits(df, "ours")

    obj = joblib.load(OUT / "model_histgb_ours.joblib")
    pipe = obj["pipeline"]

    static = pd.read_parquet(DATA_FULL, columns=[
        "m_is_modelable", "succ", "usd_pledged", "creator_id", "category_key"])
    static = static[static["m_is_modelable"].fillna(False)]
    static = static[pd.to_numeric(static["succ"], errors="coerce").notna()]
    static = static.reindex(df.index)
    for c in ("usd_pledged", "creator_id", "category_key"):
        df[c] = static[c].to_numpy()

    df["goal_band"] = pd.cut(df["m_goal_usd"], GOAL_BINS, labels=GOAL_LABELS, right=True)
    df["dur_band"] = pd.cut(df["m_dur_days_exact"], DUR_BINS, labels=DUR_LABELS, right=True)
    df["succ"] = df["succ"].astype(int)
    df["p"] = pipe.predict_proba(df[feats])[:, 1]
    df["is_test"] = df.index.isin(test.index)

    def build(sub):
        rows = []
        for (c, gb, db), g in sub.groupby(["category_key", "goal_band", "dur_band"],
                                          observed=True):
            n, k = len(g), int(g["succ"].sum())
            lo, hi = wilson(k, n)
            win = g.loc[g["succ"] == 1, "usd_pledged"]
            rows.append({"category": c, "goal_band": str(gb), "dur_band": str(db),
                         "n": n, "k": k, "rate": k / n, "ci_low": lo, "ci_high": hi,
                         "median_payout_usd_success": float(win.median()) if len(win) else np.nan,
                         "mean_payout_usd_success": float(win.mean()) if len(win) else np.nan,
                         "p_cell_mean": float(g["p"].mean())})
        d = pd.DataFrame(rows)
        d["value_obs"] = d["rate"] * d["mean_payout_usd_success"]
        return d

    print("[2/5] in-sample / out-of-sample 格子 ...", flush=True)
    cells_in = build(df)
    cells_oos = build(df[df["is_test"]])
    cells_in["is_oos"] = False
    cells_oos["is_oos"] = True
    cells_oos.to_parquet(OUT / "cells_oos.parquet", index=False)
    n_in = int((cells_in["n"] >= MIN_N).sum())
    n_oos = int((cells_oos["n"] >= MIN_N).sum())
    print(f"      in-sample 可用格 {n_in}/{len(cells_in)}；OOS 可用格 {n_oos}/{len(cells_oos)}",
          flush=True)

    print("[3/5] 同一格 in-sample vs OOS ...", flush=True)
    m = cells_in.merge(cells_oos, on=["category", "goal_band", "dur_band"],
                       suffixes=("_in", "_oos"))
    m = m[(m["n_in"] >= MIN_N) & (m["n_oos"] >= MIN_N)].copy()
    m["gap_pp"] = (m["rate_oos"] - m["rate_in"]) * 100
    m.to_csv(OUT / "s3_oos_vs_insample.csv", index=False, encoding="utf-8-sig")
    if len(m):
        print(f"      可比格 {len(m)}；OOS−in 达标率差 中位 {m['gap_pp'].median():+.2f}pp，"
              f"范围 {m['gap_pp'].min():+.2f} ~ {m['gap_pp'].max():+.2f}pp，"
              f"|差|>10pp 占 {(m['gap_pp'].abs() > 10).mean()*100:.1f}%", flush=True)

    print("[4/5] 价值维度 bootstrap（按 creator 聚类）...", flush=True)
    rows = []
    for c, g in df.groupby("category_key", observed=True):
        cg = cells_in[cells_in["category"] == c]
        usable = cg[cg["n"] >= MIN_N]
        if len(usable) < 3:
            continue
        base = usable.loc[usable["n"].idxmax()]
        cands = {
            "最稳妥": usable.loc[usable["rate"].idxmax()],
            "最高价值": usable.loc[usable["value_obs"].idxmax()],
            "应避免": usable.loc[usable["rate"].idxmin()],
        }
        for label, cand in cands.items():
            if cand[["goal_band", "dur_band"]].equals(base[["goal_band", "dur_band"]]):
                rows.append({"category": c, "role": label, "candidate": "=基准",
                             "d_rate_pp": 0.0, "rate_ci": "—", "d_value_usd": 0.0,
                             "value_ci": "—", "grade": "无差异"})
                continue
            sel = ((g["goal_band"].astype(str) == cand["goal_band"])
                   & (g["dur_band"].astype(str) == cand["dur_band"]))
            sel_base = ((g["goal_band"].astype(str) == base["goal_band"])
                        & (g["dur_band"].astype(str) == base["dur_band"]))
            a, b = g[sel], g[sel_base]
            r, rlo, rhi, v, vlo, vhi = boot_delta(a, b)
            need = {"最稳妥": "rate", "最高价值": "value", "应避免": "rate"}[label]
            lo, hi = (rlo, rhi) if need == "rate" else (vlo, vhi)
            if label == "应避免":
                ok = hi < 0
            else:
                ok = lo > 0
            rows.append({"category": c, "role": label,
                         "candidate": f"{cand['goal_band']}×{cand['dur_band']}",
                         "d_rate_pp": r * 100,
                         "rate_ci": f"[{rlo*100:+.1f}, {rhi*100:+.1f}]pp",
                         "d_value_usd": v,
                         "value_ci": f"[${vlo:+,.0f}, ${vhi:+,.0f}]",
                         "grade": "推荐级" if ok else "参考级"})
    vt = pd.DataFrame(rows)
    vt.to_csv(OUT / "s3_value_tests.csv", index=False, encoding="utf-8-sig")
    print(f"      检验条数 {len(vt)}；推荐级 {int((vt['grade']=='推荐级').sum())}，"
          f"参考级 {int((vt['grade']=='参考级').sum())}", flush=True)

    print("[5/5] 取消项目敏感性 ...", flush=True)
    raw = pd.read_parquet(DATA_FULL, columns=["state", "category_key"])
    raw_state = raw["state"].astype(str).str.lower()
    n_cancel = int((raw_state == "canceled").sum())
    base_rate = float(df["succ"].mean())
    # 取消计入失败后的整体成功率（分母加入 canceled）
    n_used = len(df)
    sens_all = n_used * base_rate / (n_used + n_cancel)
    sens = pd.DataFrame([
        {"scope": "整体", "modelable_n": n_used, "success": int(df["succ"].sum()),
         "rate_excl_canceled": base_rate, "canceled_n": n_cancel,
         "rate_incl_canceled_as_fail": sens_all},
    ])
    sens.to_csv(OUT / "s3_cancel_sensitivity.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# S3 样本外实测与价值维度检验", "",
        "> 回答“实测效果应该在测试集里取”：**效果/验证只能来自测试期**（本次 = `yr≥2023`，n=100,469）；",
        "> 全量表只能作 **in-sample 描述性基准**。ΔP 是模型反事实，**不可称为实测效果**。", "",
        "## 一、样本外格子支撑度", "",
        f"- in-sample 可用格（n≥{MIN_N}）：**{n_in}/{len(cells_in)}**",
        f"- **out-of-sample 可用格：{n_oos}/{len(cells_oos)}** ← 450 格在测试期只有这么多格有足够样本",
        "", "## 二、同一格 in-sample vs OOS 的达标率差", "",
    ]
    if len(m):
        lines += [
            f"- 可比格数：**{len(m)}**（两边 n 都 ≥{MIN_N}）",
            f"- OOS − in-sample：中位 **{m['gap_pp'].median():+.2f}pp**，"
            f"范围 {m['gap_pp'].min():+.2f} ~ {m['gap_pp'].max():+.2f}pp",
            f"- 绝对差 >10pp 的格占 **{(m['gap_pp'].abs()>10).mean()*100:.1f}%**",
            "",
            "⇒ 含义：训练期观测率**不能**直接当作未来预期；引用时必须标 in-sample。",
        ]
    lines += ["", "## 三、价值维度检验（P2-2）", "",
              f"- 检验条数 **{len(vt)}**，推荐级 **{int((vt['grade']=='推荐级').sum())}**，"
              f"参考级 **{int((vt['grade']=='参考级').sum())}**",
              "- 判级规则：**成功率类角色只认 Δrate 的 95% CI；「最高价值」只认 Δvalue 的 95% CI**"
              "（value = rate × 成功项目平均募资额），按 `creator_id` 整簇重抽 200 次。",
              "", "## 四、取消项目敏感性（P2-1）", "",
              f"- 二分类口径（排除取消）：**{base_rate*100:.2f}%**（n={n_used:,}）",
              f"- 取消计入失败：**{sens_all*100:.2f}%**（分母 +{n_cancel:,}）",
              f"- 差 **{(base_rate-sens_all)*100:.2f}pp** ⇒ 报告必须写明任务条件是“最终进入成功/失败的项目”。"]
    (OUT / "S3_样本外与价值检验.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[done] cells_oos.parquet / s3_*.csv / S3_样本外与价值检验.md", flush=True)


if __name__ == "__main__":
    main()
