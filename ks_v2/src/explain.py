"""Stage 8 - interpretation and the three landing artefacts.

  * configuration scorecard (category x goal bin x duration bin, n>=300)
  * per-category blurb keyword panel (smoothed log-odds + BH-FDR)
  * logistic-regression odds ratios (with 95% CI)
  * permutation importance of the gradient-boosting model
  * three worked example cards (goal / duration / blurb counterfactuals)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src import features as F

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

TOKEN_RE = re.compile(r"[a-z][a-z']{1,}")


def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    tr = C.add_bins(df[df.yr <= C.TRAIN_MAX_YEAR])
    te = C.add_bins(df[df.yr >= C.TEST_MIN_YEAR])
    keys = ["category_key", "goal_bin", "dur4_bin"]
    g = (tr.groupby(keys, observed=True)["succ"]
         .agg(n_train="size", k_train="sum").reset_index())
    g["rate_train"] = g["k_train"] / g["n_train"]
    g["lo"], g["hi"] = C.wilson_ci(g["k_train"], g["n_train"])
    g = g[g["n_train"] >= C.MIN_CELL_N]

    gt = (te.groupby(keys, observed=True)["succ"]
          .agg(n_test="size", k_test="sum").reset_index())
    gt["rate_test"] = gt["k_test"] / gt["n_test"]
    out = g.merge(gt, on=keys, how="left")
    out["delta_test_minus_train"] = out["rate_test"] - out["rate_train"]
    return out.sort_values(keys)


def keyword_panel(df: pd.DataFrame, min_n: int = 3000, top_k: int = 15) -> pd.DataFrame:
    tr = df[df.yr <= C.TRAIN_MAX_YEAR]
    rows = []
    for cat, g in tr.groupby("category_key", observed=True):
        if len(g) < min_n:
            continue
        tok = g["blurb_clean"].fillna("").astype(str).map(
            lambda s: set(TOKEN_RE.findall(s.lower())))
        succ_tok = tok[g["succ"] == 1]
        fail_tok = tok[g["succ"] == 0]
        n_s, n_f = len(succ_tok), len(fail_tok)
        cnt_s = pd.Series([t for s in succ_tok for t in s]).value_counts()
        cnt_f = pd.Series([t for s in fail_tok for t in s]).value_counts()
        tokens = cnt_s.index.union(cnt_f.index)
        cnt_s = cnt_s.reindex(tokens, fill_value=0)
        cnt_f = cnt_f.reindex(tokens, fill_value=0)
        keep = (cnt_s >= max(30, 0.01 * n_s)) | (cnt_f >= max(30, 0.01 * n_f))
        cnt_s, cnt_f = cnt_s[keep], cnt_f[keep]
        if len(cnt_s) == 0:
            continue
        tokens = cnt_s.index
        a = cnt_s + 0.5
        b = n_s - cnt_s + 0.5
        c = cnt_f + 0.5
        d = n_f - cnt_f + 0.5
        log_odds = np.log(a / b) - np.log(c / d)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = log_odds / se
        p = 2 * stats.norm.sf(np.abs(z))
        rej = multipletests(p, alpha=0.05, method="fdr_bh")[0] if len(p) else np.array([], bool)
        tb = pd.DataFrame({
            "category": cat, "token": tokens, "n_success_docs": cnt_s.values,
            "n_failure_docs": cnt_f.values, "log_odds": log_odds.values,
            "p_value": p, "significant_fdr": rej,
        })
        tb = tb[tb.significant_fdr]
        top_pos = tb.sort_values("log_odds", ascending=False).head(top_k)
        top_neg = tb.sort_values("log_odds").head(top_k)
        rows.append(pd.concat([top_pos, top_neg]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def odds_ratios(df: pd.DataFrame) -> pd.DataFrame:
    tr = C.add_bins(df[df.yr <= C.TRAIN_MAX_YEAR]).copy()
    tr["prev_miss"] = tr["creator_prev_success_rate"].isna().astype(int)
    tr["creator_prev_success_rate"] = tr["creator_prev_success_rate"].fillna(
        tr["creator_prev_success_rate"].median())
    cols = ["category_key", "goal_bin", "dur4_bin", "rank_bin"]
    X = pd.get_dummies(tr[cols].astype(str), drop_first=True, dtype=float)
    X["prev_miss"] = tr["prev_miss"].values
    X["is_relaunch"] = tr["is_relaunch"].astype(float).values
    X["creator_prev_success_rate"] = tr["creator_prev_success_rate"].values
    X["log_goal"] = tr["log_goal"].values
    X = sm.add_constant(X)
    y = tr["succ"].astype(int).values
    C.log(f"odds ratios: design {X.shape}, fitting logit ...")
    res = sm.Logit(y, X).fit(method="lbfgs", maxiter=300, disp=0)
    ci = res.conf_int()
    out = pd.DataFrame({
        "term": X.columns, "coef": res.params.values,
        "odds_ratio": np.exp(res.params.values),
        "ci_lo": np.exp(ci[0].values), "ci_hi": np.exp(ci[1].values),
        "p_value": res.pvalues.values,
    })
    return out.sort_values("coef")


def main() -> None:
    df = C.load_v2().reset_index(drop=True)
    train_mask = (df["yr"] <= C.TRAIN_MAX_YEAR).to_numpy()
    test_mask = (df["yr"] >= C.TEST_MIN_YEAR).to_numpy()

    # ---------------- 1. scorecard
    sc = scorecard(df)
    sc.to_csv(C.TABLE_DIR / "scorecard.csv", index=False)
    C.log(f"scorecard: {len(sc)} cells with n_train>={C.MIN_CELL_N}")
    C.log(sc.sort_values("rate_train").head(8).to_string(index=False))

    # ---------------- 2. keyword panel
    kw = keyword_panel(df)
    kw.to_csv(C.TABLE_DIR / "keyword_panel.csv", index=False)
    C.log(f"keyword panel: {len(kw)} rows, categories={kw.category.nunique() if len(kw) else 0}")
    # top 4 categories by training volume (largest, most interpretable panels)
    big_cats = (df.loc[train_mask, "category_key"].value_counts().head(4).index.tolist()
                if len(kw) else [])
    for cat in big_cats:
        sub = kw[kw.category == cat]
        pos = sub.sort_values("log_odds", ascending=False).head(6)
        neg = sub.sort_values("log_odds").head(6)
        panel = pd.concat([pos, neg]).drop_duplicates("token").sort_values("log_odds")
        colors = ["#c00000" if v < 0 else "#1f4e79" for v in panel.log_odds]
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.barh(panel.token, panel.log_odds, color=colors)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("smoothed log-odds (success vs failure)")
        ax.set_title(f"{cat}: blurb tokens, FDR<0.05")
        fig.tight_layout()
        C.savefig(fig, f"fig_keywords_{re.sub(r'[^a-z]+', '_', cat)}.png")

    # ---------------- 3. odds ratios
    ors = odds_ratios(df)
    ors.to_csv(C.TABLE_DIR / "lr_odds_ratios.csv", index=False)
    C.log("odds ratios done; extreme terms:\n" +
          pd.concat([ors.head(6), ors.tail(6)]).to_string(index=False))

    # ---------------- 4. permutation importance (structured HGB)
    spec = F.build_spec(df[train_mask])
    Xtr, mask, names = F.transform(df[train_mask], spec)
    Xte, _, _ = F.transform(df[test_mask], spec)
    ytr = df.loc[train_mask, "succ"].to_numpy(int)
    yte = df.loc[test_mask, "succ"].to_numpy(int)
    hgb = HistGradientBoostingClassifier(categorical_features=mask, **C.HGB_CLS)
    hgb.fit(Xtr.to_numpy(np.float32), ytr)
    rng = np.random.default_rng(C.SEED)
    sub = rng.choice(len(yte), size=min(20_000, len(yte)), replace=False)
    pi = permutation_importance(hgb, Xte.to_numpy(np.float32)[sub], yte[sub],
                                scoring="roc_auc", n_repeats=5,
                                random_state=C.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": names, "auc_drop_mean": pi.importances_mean,
                        "auc_drop_sd": pi.importances_std}).sort_values(
        "auc_drop_mean", ascending=False)
    imp.to_csv(C.TABLE_DIR / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    top = imp.head(14).iloc[::-1]
    ax.barh(top.feature, top.auc_drop_mean, xerr=top.auc_drop_sd,
            color="#1f4e79", error_kw={"lw": 1, "capsize": 2})
    ax.set_xlabel("AUC drop when permuted (test sample)")
    ax.set_title("Permutation importance, structured HistGB")
    C.savefig(fig, "fig_importance.png")
    C.log("\n" + imp.head(14).to_string(index=False))

    # ---------------- 5. three example cards
    te_full = C.add_bins(df[test_mask]).reset_index(drop=True)
    base_pred = hgb.predict_proba(Xte.to_numpy(np.float32))[:, 1]

    cards = []
    picks = []
    m1 = (te_full.succ == 0) & (te_full.goal >= 100_000) & (te_full.goal < 500_000)
    if m1.any():
        picks.append(("goal_too_high", te_full[m1].index[len(te_full[m1]) // 2]))
    m2 = (te_full.succ == 0) & te_full.dur_days.between(46, 60)
    if m2.any():
        picks.append(("duration_too_long", te_full[m2].index[len(te_full[m2]) // 2]))
    m3 = (te_full.succ == 0) & (te_full.blurb_words <= 6)
    if m3.any():
        picks.append(("weak_blurb", te_full[m3].index[len(te_full[m3]) // 2]))

    goal_mult = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    dur_grid = [7, 14, 30, 45, 60]
    for tag, idx in picks:
        row = te_full.loc[[idx]].copy()
        base = float(base_pred[idx])
        best = ((1.0, int(row["dur_days"].iloc[0])), base)
        for gm in goal_mult:
            for dg in dur_grid:
                r = row.copy()
                r["goal"] = row["goal"] * gm
                r["log_goal"] = np.log10(r["goal"])  # 与数据口径一致: log_goal = log10(goal)
                r["dur_days"] = dg
                Xr, _, _ = F.transform(r, spec, strict=False)
                p = float(hgb.predict_proba(Xr.to_numpy(np.float32))[0, 1])
                if p > best[1]:
                    best = ((gm, dg), p)
                cards.append({
                    "card": tag, "project_id": int(row["id"].iloc[0]),
                    "category": row["category_key"].iloc[0],
                    "goal_actual": float(row["goal"].iloc[0]),
                    "dur_actual": int(row["dur_days"].iloc[0]),
                    "blurb_words": int(row["blurb_words"].iloc[0]),
                    "goal_multiplier": gm, "duration": dg,
                    "p_success": p, "observed_success": int(row["succ"].iloc[0]),
                })
        C.log(f"card {tag}: id={int(row['id'].iloc[0])} base={base:.3f} "
              f"best={best[1]:.3f} at goal x{best[0][0]} dur={best[0][1]}")
    cards_tb = pd.DataFrame(cards)
    cards_tb.to_csv(C.TABLE_DIR / "example_cards.csv", index=False)

    # goal-sensitivity curve for each card (duration held at actual)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for tag, idx in picks:
        row = te_full.loc[[idx]].copy()
        ys = []
        for gm in goal_mult:
            r = row.copy()
            r["goal"] = row["goal"] * gm
            r["log_goal"] = np.log10(r["goal"])  # 与数据口径一致: log_goal = log10(goal)
            Xr, _, _ = F.transform(r, spec, strict=False)
            ys.append(float(hgb.predict_proba(Xr.to_numpy(np.float32))[0, 1]))
        ax.plot(goal_mult, ys, "o-", lw=1.2, ms=3, label=tag)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel("goal multiplier vs actual")
    ax.set_ylabel("predicted P(success)")
    ax.set_title("Goal counterfactual for three failed projects")
    ax.legend(fontsize=7)
    C.savefig(fig, "fig_example_cards_goal.png")
    C.log("stage explain done")


if __name__ == "__main__":
    main()
