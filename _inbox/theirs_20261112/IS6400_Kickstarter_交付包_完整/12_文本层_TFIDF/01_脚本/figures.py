# -*- coding: utf-8 -*-
"""figures.py — 6 张图（只依赖 #1–#12 的产物 + v1 train-only 关键词表）。

  fig1_delta_auc_forest.png    12 个配对的 ΔAUC + 95%CI（creator 整簇 bootstrap）
  fig2_base_vs_text.png         3 算法 × {基线, +文本} 的 AUC / AP / Brier
  fig3_usd_vs_nonusd.png        跨行集 ΔAUC：all / usd / 非usd
  fig4_decision_impact.png      阈值穿越 + |Δp| 分布 + top-10% 重合
  fig5_pdp_blurb_words.png      m_blurb_words 的 PDP（21 点，主模型）
  fig6_keyword_panel.png        关键词 lift 高/低各 15（train-only）
"""
from __future__ import annotations

import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cfg
import features

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})
FIG = cfg.FIG
os.makedirs(FIG, exist_ok=True)

ALGO_CN = {"hgb": "HistGB", "lr": "LR", "rf": "RF"}
SCOPE_CN = {"full": "全样本", "usd": "USD 口径"}
RS_CN = {"all": "全测试行", "usd": "USD 测试行", "nonusd": "非 USD 测试行"}
COL = {"all": "#4C72B0", "usd": "#DD8452", "nonusd": "#55A868"}


def save(fig, name: str) -> None:
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}", flush=True)


def fig1_forest() -> None:
    db = pd.read_csv(os.path.join(cfg.RES, "deltas_bootstrap.csv"))
    d = db[db["scheme"] == "cluster"].copy()
    d = d[d["rowset"].isin(["all", "usd"])]
    d["label"] = d.apply(lambda r: f"{ALGO_CN[r['new_key'].split('_')[0]]} + 文本"
                                    f"（{'USD 口径' if r['new_key'].endswith('_usd') else '全样本'}）"
                                    f" | {RS_CN[r['rowset']]}", axis=1)
    d = d.sort_values(["rowset", "dAUC"]).reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, (_, r) in enumerate(d.iterrows()):
        c = COL[r["rowset"]]
        ax.errorbar(r["dAUC"] * 100, i,
                    xerr=[[(r["dAUC"] - r["ci95_low"]) * 100], [(r["ci95_high"] - r["dAUC"]) * 100]],
                    fmt="o", color=c, ms=5, lw=1.4, capsize=2.5)
        ax.annotate(f"{r['dAUC'] * 100:+.2f}pp", (r["ci95_high"] * 100, i),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=c)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("ΔAUC（百分点，正 = 加文本更好；creator 整簇 bootstrap 95%CI）")
    ax.set_title("TF-IDF 文本层增量：全样本口径 6 个配对 + USD 口径 3 个配对", fontsize=10)
    save(fig, "fig1_delta_auc_forest.png")


def fig2_base_vs_text() -> None:
    mm = pd.read_csv(os.path.join(cfg.RES, "matrix_metrics.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    x = np.arange(3)
    w = 0.36
    for ax, metric in zip(axes, ["AUC", "AP", "Brier"]):
        base = [mm[(mm["algo"] == a) & (mm["kind"] == "base") & (mm["scope"] == "full")][metric].iloc[0]
                for a in ("hgb", "lr", "rf")]
        text = [mm[(mm["algo"] == a) & (mm["kind"] == "text") & (mm["scope"] == "full")][metric].iloc[0]
                for a in ("hgb", "lr", "rf")]
        b1 = ax.bar(x - w / 2, base, w, label="18 特征基线", color="#9aa5b1")
        b2 = ax.bar(x + w / 2, text, w, label="+ TF-IDF", color="#4C72B0")
        for bars, vals in ((b1, base), (b2, text)):
            for bb, v in zip(bars, vals):
                ax.annotate(f"{v:.4f}", (bb.get_x() + bb.get_width() / 2, v),
                            xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.6)
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_CN[a] for a in ("hgb", "lr", "rf")])
        ax.set_title(metric + ("（越低越好）" if metric == "Brier" else "（越高越好）"), fontsize=9)
        lo = min(base + text)
        ax.set_ylim(lo - (0.02 if metric != "AUC" else 0.02), max(base + text) + 0.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("全测试行（n=100,469）：加文本后 HistGB/LR 上升，RF 下降", fontsize=10)
    save(fig, "fig2_base_vs_text.png")


def fig3_usd_vs_nonusd() -> None:
    db = pd.read_csv(os.path.join(cfg.RES, "deltas_bootstrap.csv"))
    d = db[(db["scheme"] == "cluster") & db["new_key"].str.endswith("_full")].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(3)
    w = 0.26
    for j, rs in enumerate(("all", "usd", "nonusd")):
        sub = [d[(d["new_key"] == f"{a}_text_full") & (d["rowset"] == rs)].iloc[0] for a in ("hgb", "lr", "rf")]
        vals = [s["dAUC"] * 100 for s in sub]
        err = [[(s["dAUC"] - s["ci95_low"]) * 100 for s in sub], [(s["ci95_high"] - s["dAUC"]) * 100 for s in sub]]
        ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=2.5, color=COL[rs], label=RS_CN[rs])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_CN[a] for a in ("hgb", "lr", "rf")])
    ax.set_ylabel("ΔAUC（百分点）")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("文本增量在 USD 与非 USD 测试行上方向一致（n=54,435 / 46,034）", fontsize=10)
    save(fig, "fig3_usd_vs_nonusd.png")


def fig4_decision_impact() -> None:
    dec = json.load(open(os.path.join(cfg.RES, "decision.json"), encoding="utf-8"))
    di = dec["decision_impact"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    ax = axes[0]
    vals = [di["flagged_both"], di["cross_up"], di["cross_down"]]
    labs = ["两者都判达标", "新增达标", "撤销达标"]
    cols = ["#55A868", "#4C72B0", "#C44E52"]
    ax.bar(labs, vals, color=cols)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:,}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title(f"阈值判定变化（thr={di['thresholds']['new']:.3f} vs {di['thresholds']['base']:.3f}）", fontsize=9)
    ax.set_ylabel("测试行数")

    ax = axes[1]
    dp = np.load(os.path.join(cfg.PRED_DIR, f"{cfg.MAIN}__all.npy")) - \
        np.load(os.path.join(cfg.PRED_DIR, "base_hgb_full__all.npy"))
    ax.hist(dp * 100, bins=120, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    q = di["abs_dp"]
    ax.set_xlabel("Δp（百分点）")
    ax.set_title(f"|Δp| 中位 {q['p50'] * 100:.2f}pp / p99 {q['p99'] * 100:.2f}pp；"
                 f"top-10% 重合 {di['top10_jaccard']:.3f}", fontsize=9)
    ax.set_ylabel("测试行数")
    save(fig, "fig4_decision_impact.png")


def fig5_pdp_blurb_words() -> None:
    import text_model
    model = text_model.load_model(os.path.join(cfg.MODELS, "model_histgb_text.joblib"))
    df = pd.read_parquet(cfg.PREPARED)
    test = df[df["id"].isin(pd.read_parquet(os.path.join(cfg.RES, "test_meta.parquet"),
                                            columns=["id"])["id"])].reset_index(drop=True)
    sample = test.sample(n=min(4000, len(test)), random_state=0).copy()
    grid = list(range(3, 44, 2))
    means = []
    for v in grid:
        sample["m_blurb_words"] = v
        means.append(float(model.predict_proba(sample)[:, 1].mean()))
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.plot(grid, means, marker="o", ms=3.5, color="#4C72B0")
    i_best = int(np.argmax(means))
    ax.annotate(f"峰值 {grid[i_best]} 词 ≈ {means[i_best]:.3f}", (grid[i_best], means[i_best]),
                xytext=(10, -14), textcoords="offset points", fontsize=7.5)
    ax.axvline(26, color="gray", ls=":", lw=0.9)
    ax.text(26.4, ax.get_ylim()[0] + 0.004, "右侧样本稀疏\n（外推区）", fontsize=6.5, color="gray")
    ax.set_xlabel("m_blurb_words（仅改词数列；blurb 原文、TF-IDF 与其他特征保持观测值）")
    ax.set_ylabel("平均预测达标概率")
    ax.set_title("历史关联，描述性：blurb 词数与预测达标率（n=4,000，非因果）", fontsize=9)
    save(fig, "fig5_pdp_blurb_words.png")
    pd.DataFrame({"feature": "m_blurb_words", "value": grid, "mean_pred": means}) \
        .to_csv(os.path.join(cfg.RES, "pdp_blurb_words.csv"), index=False, encoding="utf-8-sig")


def fig6_keyword_panel() -> None:
    src = os.path.join(cfg.ARCH, "work", "keyword_panel.csv")
    dst = os.path.join(cfg.RES, "keyword_panel.csv")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
    k = pd.read_csv(dst)
    g = k[(k["scope"] == "global") & (k["source"] == "blurb")]
    top = g.nlargest(15, "lift").iloc[::-1]
    bot = g.nsmallest(15, "lift").iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, d, color, title in ((axes[0], top, "#55A868", "高信号（lift Top 15）"),
                                (axes[1], bot, "#C44E52", "低信号（lift Bottom 15）")):
        ax.barh(np.arange(len(d)), d["lift"], color=color)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels([f"{w}（n={n:,}）" for w, n in zip(d["keyword"], d["n_present"])], fontsize=7)
        ax.axvline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("blurb 内出现该词的达标率 / 训练集基准率（lift）")
        ax.set_title(title, fontsize=9)
    fig.suptitle("blurb 关键词面板（train n=443,693，只用于建议层解释，不入模）", fontsize=10)
    save(fig, "fig6_keyword_panel.png")


def main() -> None:
    fig1_forest()
    fig2_base_vs_text()
    fig3_usd_vs_nonusd()
    fig4_decision_impact()
    fig5_pdp_blurb_words()
    fig6_keyword_panel()


if __name__ == "__main__":
    main()
