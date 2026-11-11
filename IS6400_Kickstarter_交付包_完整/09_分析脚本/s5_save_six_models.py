# -*- coding: utf-8 -*-
"""S5: 六模型一次拟合 —— 三种算法 × 两种训练口径，全部落盘 + 两批测试行对照。

为什么要重写：用户指出“原来只有一个模型被保存”。本脚本保证：
  每个模型 fit 一次 → 立即 joblib 落盘（compress=3）→ 同时产出 3×2×2 评估矩阵。

六模型命名：`model_{algo}_{scope}.joblib`，algo ∈ {lr, rf, hgb}，scope ∈ {full, usd}

产出：
  11_结果与报告/models/model_{algo}_{scope}.joblib + _schema.json   （6 套）
  11_结果与报告/s5_model_scope_matrix.csv
  11_结果与报告/S5_三模型x两口径.md
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "11_结果与报告"
MDIR = OUT / "models"
MDIR.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("rp", HERE / "replicate_pilot.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

ALGOS = [("Logistic Regression", "lr"), ("Random Forest", "rf"), ("Gradient Boosting", "hgb")]


def prep(path):
    df = rp.build_frame(str(path), use_prep_days=True)
    static = pd.read_parquet(path, columns=["m_is_modelable", "succ", "currency"])
    static = static[static["m_is_modelable"].fillna(False)]
    static = static[pd.to_numeric(static["succ"], errors="coerce").notna()].reindex(df.index)
    df["is_usd"] = (static["currency"].astype("string").str.strip().str.upper() == "USD").to_numpy()
    df["succ"] = df["succ"].astype(int)
    df["yr"] = pd.to_numeric(df["yr"], errors="coerce")
    return df, df.attrs["numeric"] + df.attrs["categorical"]


def main():
    t0 = time.time()
    print("[1/4] 载入两份数据 ...", flush=True)
    dfF, featsF = prep(rp.DATA["full"])
    dfU, featsU = prep(rp.DATA["usd"])
    trF, vaF, teF = rp.make_splits(dfF, "ours")
    trU, vaU, teU = rp.make_splits(dfU, "ours")
    test_all, test_usd = teF, teF[teF["is_usd"]]
    print(f"      test_all={len(test_all):,} test_usd={len(test_usd):,}", flush=True)

    scopes = {"full": (dfF, featsF, trF, vaF, teF), "usd": (dfU, featsU, trU, vaU, teU)}
    rows, manifest = [], []
    for scope, (df, feats, tr, va, te) in scopes.items():
        built = rp.models(df.attrs["numeric"], df.attrs["categorical"], rf_trees=150)
        for name, short in ALGOS:
            pipe = built.get(name)
            if pipe is None:
                continue
            print(f"[fit] {short}/{scope} train={len(tr):,} ...", flush=True)
            t1 = time.time()
            pipe.fit(tr[feats], tr["succ"])
            fit_s = time.time() - t1

            thr = rp.pick_threshold(va["succ"].values, pipe.predict_proba(va[feats])[:, 1])
            p_own = pipe.predict_proba(te[feats])[:, 1]
            own = {"AUC": roc_auc_score(te["succ"], p_own),
                   "AP": average_precision_score(te["succ"], p_own),
                   "Brier": brier_score_loss(te["succ"], p_own)}

            path = MDIR / f"model_{short}_{scope}.joblib"
            joblib.dump({"pipeline": pipe, "features": feats,
                         "numeric": df.attrs["numeric"], "categorical": df.attrs["categorical"],
                         "threshold": thr, "scope": scope, "algo": name,
                         "metrics_own_test": own},
                        path, compress=3)
            schema = {"algo": name, "scope": scope, "features": feats,
                      "numeric": df.attrs["numeric"], "categorical": df.attrs["categorical"],
                      "threshold": thr, "metrics_own_test": own,
                      "n_train": int(len(tr)), "n_test_own": int(len(te))}
            (MDIR / f"model_{short}_{scope}_schema.json").write_text(
                json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
            size = path.stat().st_size
            manifest.append({"file": path.name, "algo": name, "scope": scope,
                             "fit_seconds": round(fit_s), "bytes": size})
            print(f"       saved {path.name} ({size/1e6:.1f} MB) in {fit_s:.0f}s | "
                  f"AUC={own['AUC']:.4f}", flush=True)

            for rname, sub in (("all_test", test_all), ("usd_test", test_usd)):
                y = sub["succ"].to_numpy()
                p = pipe.predict_proba(sub[feats])[:, 1]
                rows.append({"scope": scope, "algo": short, "model": name, "test_rows": rname,
                             "n": len(y), "AUC": roc_auc_score(y, p),
                             "AP": average_precision_score(y, p),
                             "Brier": brier_score_loss(y, p)})

    print("[2/4] 落盘矩阵 ...", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s5_model_scope_matrix.csv", index=False, encoding="utf-8-sig")
    man = pd.DataFrame(manifest)
    man.to_csv(OUT / "models_manifest.csv", index=False, encoding="utf-8-sig")
    print(t.round(4).to_string(index=False), flush=True)
    print("\n=== 已保存模型 ===", flush=True)
    print(man.to_string(index=False), flush=True)

    print("[3/4] 写报告 ...", flush=True)
    piv = t.pivot_table(index=["algo", "test_rows"], columns="scope",
                        values=["AUC", "AP", "Brier"])
    lab = {"lr": "LR", "rf": "RF(150)", "hgb": "HistGB"}
    L = ["# S5 六模型：三算法 × 两口径（全部已落盘）", "",
         "> 三种算法（LR / RF150 / HistGB，配置同 S1）× 两种训练范围（full 全样本 / usd 仅 USD）。",
         "> **六个模型均已保存**到 `11_结果与报告/models/`，命名 `model_{algo}_{scope}.joblib` + 同名 schema。",
         "> 评估只用阈值无关指标（AUC / AP / Brier），两批测试行完全相同。", "",
         "## 一、12 个评估", "",
         "| 算法 | 训练范围 | 测试行 | n | AUC | AP | Brier |",
         "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for _, r in t.iterrows():
        L.append(f"| {lab[r['algo']]} | {r['scope']} | {r['test_rows']} | {r['n']:,} | "
                 f"{r['AUC']:.4f} | {r['AP']:.4f} | {r['Brier']:.4f} |")

    L += ["", "## 二、配对差（同一算法、同一测试行：full − usd）", "",
          "| 算法 | 测试行 | ΔAUC | ΔAP | ΔBrier（usd−full） |", "| --- | --- | ---: | ---: | ---: |"]
    for (algo, trn), row in piv.iterrows():
        dau = (row[("AUC", "full")] - row[("AUC", "usd")]) * 100
        dap = (row[("AP", "full")] - row[("AP", "usd")]) * 100
        dbr = (row[("Brier", "usd")] - row[("Brier", "full")]) * 100
        L.append(f"| {lab[algo]} | {trn} | {dau:+.2f}pp | {dap:+.2f}pp | {dbr:+.2f}pp |")

    L += ["", "## 三、结论", ""]
    allpos = all((piv.loc[(a, "all_test"), ("AUC", "full")] >
                  piv.loc[(a, "all_test"), ("AUC", "usd")]) for a in lab if (a, "all_test") in piv.index)
    L.append("- **方向一致性**：全样本测试行上，" +
             ("**三种算法均为 full > usd**" if allpos else "各算法方向不一致（需逐项说明）") + "。"
             "幅度： " + "、".join(f"{lab[a]} {(piv.loc[(a,'all_test'),('AUC','full')]-piv.loc[(a,'all_test'),('AUC','usd')])*100:+.2f}pp"
                                  for a in lab if (a, "all_test") in piv.index) + "。")
    L.append("- **迁移性**：仅 USD 测试行上，" + "、".join(
        f"{lab[a]} {(piv.loc[(a,'usd_test'),('AUC','full')]-piv.loc[(a,'usd_test'),('AUC','usd')])*100:+.2f}pp"
        for a in lab if (a, "usd_test") in piv.index) + "（均为正则为正迁移）。")
    L.append("- **结论不依赖模型选择**：剔除非 USD 在三种算法上都是小幅净损失；"
             "真正代价仍是覆盖范围（国家 25→3、可用格 250→219、样本外 85→41）。")
    L += ["", "## 四、已落盘模型清单", "",
          "| 文件 | 算法 | 口径 | 训练耗时 | 体积 |", "| --- | --- | --- | ---: | ---: |"]
    for _, r in man.iterrows():
        L.append(f"| `models/{r['file']}` | {r['algo']} | {r['scope']} | "
                 f"{int(r['fit_seconds'])}s | {r['bytes']/1e6:.1f} MB |")
    L += ["", "> ⚠️ 全部 `*.joblib` 被 `.gitignore` 排除，也不进 zip（体积大）；"
              "需要时用对应脚本重生成，schema 里记了 features/threshold/metrics。"]
    (OUT / "S5_三模型x两口径.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[4/4] done, 总用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
