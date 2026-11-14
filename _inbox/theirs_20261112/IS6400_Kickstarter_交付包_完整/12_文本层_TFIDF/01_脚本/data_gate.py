# -*- coding: utf-8 -*-
"""data_gate.py — G0 数据门。

校验：数据源 sha256、行/列/计数、三段切分与 split_assignment 行级一致、
USD 口径一致、文本诊断、creator_id 映射（整簇 bootstrap 用）。
产出：02_结果/prepared_modelable.parquet、creator_id.parquet、
      g0_data_gate.json、text_diagnostics.json、pkg_manifest_before.sha256.csv
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import cfg

L: "list[str]" = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    L.append(line)


def main() -> None:
    t0 = time.time()
    rp = cfg.load_rp()
    checks: "dict[str, dict]" = {}

    def chk(name: str, ok: bool, detail=None) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        log(f"{'OK  ' if ok else 'FAIL'} {name} :: {detail}")

    # ---------------- A. 数据源 sha256 ----------------
    log("hashing ks_clean.csv ...")
    sha = cfg.sha256_file(cfg.KS_CLEAN)
    size = os.path.getsize(cfg.KS_CLEAN)
    chk("A1_ks_clean_sha256", sha == cfg.EXPECT["ks_clean_sha256"], {"sha256": sha, "bytes": size})
    chk("A2_ks_clean_bytes", size == cfg.EXPECT["ks_clean_bytes"], size)

    # ---------------- B. base parquet 结构 ----------------
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(cfg.BASE_PARQUET)
    n_rows, n_cols = meta.metadata.num_rows, meta.metadata.num_columns
    chk("B1_base_rows", n_rows == 673839, n_rows)
    chk("B2_base_cols", n_cols == 33, n_cols)

    ids = pd.read_parquet(cfg.BASE_PARQUET, columns=["id", "m_is_modelable",
                                                     "m_currency_usd", "m_goal_usd",
                                                     "m_name_len", "m_blurb_len",
                                                     "succ"])
    ids["id"] = _id_str(ids["id"])
    chk("B3_id_unique", ids["id"].is_unique, int(ids["id"].nunique()))
    modelable = ids[ids["m_is_modelable"].fillna(False) & pd.to_numeric(ids["succ"], errors="coerce").notna()]
    chk("B4_modelable", len(modelable) == cfg.EXPECT["rows_modelable"], len(modelable))
    chk("B5_usd_modelable", int(modelable["m_currency_usd"].fillna(False).sum()) == cfg.EXPECT["rows_usd_modelable"],
        int(modelable["m_currency_usd"].fillna(False).sum()))
    # 9 项对账是「全量 673,839 行」口径（prepare_report.json），先校验全量，再记录可建模子集
    chk("B6_goal_na_all", int(ids["m_goal_usd"].isna().sum()) == 1487, int(ids["m_goal_usd"].isna().sum()))
    chk("B7_name_len_na_all", int(ids["m_name_len"].isna().sum()) == 5, int(ids["m_name_len"].isna().sum()))
    chk("B8_blurb_len_na_all", int(ids["m_blurb_len"].isna().sum()) == 47, int(ids["m_blurb_len"].isna().sum()))
    chk("B9_goal_na_modelable", int(modelable["m_goal_usd"].isna().sum()) == 850,
        int(modelable["m_goal_usd"].isna().sum()))
    chk("B10_blurb_na_modelable", int(modelable["m_blurb_len"].isna().sum()) == 13,
        int(modelable["m_blurb_len"].isna().sum()))

    # ---------------- C. 建模帧 + 切分 ----------------
    log("building modelable frame via replicate_pilot.build_frame ...")
    df = rp.build_frame(cfg.BASE_PARQUET, use_prep_days=True)
    chk("C1_frame_rows", len(df) == cfg.EXPECT["rows_modelable"], len(df))
    numeric, categorical = df.attrs["numeric"], df.attrs["categorical"]
    chk("C2_numeric_order", numeric == cfg.NUMERIC, numeric)
    chk("C3_categorical_order", categorical == cfg.CATEGORICAL, categorical)

    extra = pd.read_parquet(cfg.BASE_PARQUET,
                            columns=["name_clean", "blurb_clean", "m_currency_usd"]).loc[df.index]
    df["name_clean"] = extra["name_clean"].astype("string").fillna("").astype(str).values
    df["blurb_clean"] = extra["blurb_clean"].astype("string").fillna("").astype(str).values
    df["is_usd"] = extra["m_currency_usd"].fillna(False).astype(bool).values
    df["id"] = _id_str(ids.loc[df.index, "id"]).values
    chk("C1b_id_present", "id" in df.columns, int(df["id"].nunique()))

    train, val, test = rp.make_splits(df, "ours")
    got = {"train_full": len(train), "val_full": len(val), "test_full": len(test)}
    chk("C4_split_counts", got == {k: cfg.EXPECT[k] for k in got}, got)
    br = {"train": float(train["succ"].mean()), "test": float(test["succ"].mean())}
    chk("C5_base_rates", abs(br["train"] - cfg.EXPECT["base_rate_train"]) < 1e-12
        and abs(br["test"] - cfg.EXPECT["base_rate_test"]) < 1e-12, br)

    s_tr, s_va, s_te = set(train["id"]), set(val["id"]), set(test["id"])
    chk("C6_split_disjoint", not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te),
        {"tr&va": len(s_tr & s_va), "tr&te": len(s_tr & s_te), "va&te": len(s_va & s_te)})
    chk("C7_split_union", len(s_tr | s_va | s_te) == len(df), len(s_tr | s_va | s_te))

    usd = df[df["is_usd"]]
    utr, uva, ute = rp.make_splits(usd, "ours")
    ugot = {"train_usd": len(utr), "val_usd": len(uva), "test_usd": len(ute)}
    chk("C8_usd_split_counts", ugot == {k: cfg.EXPECT[k] for k in ugot}, ugot)
    chk("C9_nonusd_test", len(test) - len(ute) == cfg.EXPECT["nonusd_test"], len(test) - len(ute))
    chk("C10_usd_total", len(usd) == cfg.EXPECT["rows_usd_modelable"], len(usd))

    # ---------------- D. 与 split_assignment 行级对账 ----------------
    sa = pd.read_parquet(cfg.SPLIT_ASSIGN)
    sa["id"] = _id_str(sa["id"])
    chk("D1_assign_rows", len(sa) == cfg.EXPECT["rows_modelable"], len(sa))
    chk("D2_assign_id_unique", sa["id"].is_unique, int(sa["id"].nunique()))
    m = df[["id", "yr", "is_usd"]].merge(sa[["id", "split_ours", "is_usd"]].rename(
        columns={"is_usd": "is_usd_assign"}), on="id", how="left", validate="one_to_one")
    chk("D3_assign_coverage", int(m["split_ours"].isna().sum()) == 0, int(m["split_ours"].isna().sum()))
    computed = np.where(m["yr"] <= 2020, "train", np.where(m["yr"] <= 2022, "val", "test"))
    agree_split = float((computed == m["split_ours"].astype(str)).mean())
    agree_usd = float((m["is_usd"].values == m["is_usd_assign"].fillna(False).astype(bool).values).mean())
    chk("D4_agree_split_ours", agree_split == 1.0, agree_split)
    chk("D5_agree_is_usd", agree_usd == 1.0, agree_usd)

    # ---------------- E. creator_id 映射 ----------------
    log("reading id/creator_id from ks_clean.csv (pyarrow streaming) ...")
    cmap = _read_creator_map()
    cmap["id"] = _id_str(cmap["id"])
    chk("E1_creator_rows", len(cmap) == 673839, len(cmap))
    chk("E2_creator_id_unique", cmap["id"].is_unique, int(cmap["id"].nunique()))
    merged = df[["id", "creator_rank", "is_usd"]].merge(cmap, on="id", how="left")
    chk("E3_creator_coverage", int(merged["creator_id"].isna().sum()) == 0,
        int(merged["creator_id"].isna().sum()))
    tmap = merged[merged["id"].isin(s_te)]
    vc = tmap["creator_id"].value_counts()
    test_rows = int(tmap["creator_rank"].eq(1).sum())
    detail_c = {"test_rows": len(tmap), "n_clusters": int(vc.size),
                "mean_rows_per_cluster": float(len(tmap) / vc.size),
                "max_rows_per_cluster": int(vc.max()), "singletons": int((vc == 1).sum()),
                "share_singleton": float((vc == 1).mean())}
    chk("E4_test_clusters", len(tmap) == cfg.EXPECT["test_full"], detail_c)
    log(f"     test 首次发起人 creator_rank==1: {test_rows:,} ({test_rows/len(tmap):.4f})")

    # ---------------- F. 文本诊断 ----------------
    diag = _text_diagnostics(df, pd.Series(np.where(df["id"].isin(s_te), "test",
                                                   np.where(df["id"].isin(s_va), "val", "train"))))
    diag["creator"] = detail_c
    cfg.save_json(diag, os.path.join(cfg.RES, "text_diagnostics.json"))
    log(f"text: empty name={diag['empty_name']} blurb={diag['empty_blurb']} "
        f"blurb chars median={diag['blurb_chars']['median']} max={diag['blurb_chars']['max']}")

    # ---------------- 落盘 ----------------
    out = df.reset_index(drop=True)
    keep = [c for c in out.columns if c != "_launch_dt"] + ["_launch_dt"]
    out = out[[c for c in keep if c in out.columns]]
    out.to_parquet(cfg.PREPARED, index=False)
    cmap[["id", "creator_id"]].loc[cmap["id"].isin(set(df["id"]))].to_parquet(cfg.CREATOR_MAP, index=False)
    log(f"prepared -> {cfg.PREPARED} ({len(out):,} x {out.shape[1]})")

    log("hashing 交付包 (manifest before) ...")
    man = cfg.manifest(cfg.PKG)
    pd.DataFrame(man).to_csv(os.path.join(cfg.RES, "pkg_manifest_before.sha256.csv"),
                             index=False, encoding="utf-8-sig")
    chk("G1_pkg_files", len(man) == 140, len(man))

    all_ok = all(v["ok"] for v in checks.values())
    cfg.save_json({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "all_ok": all_ok,
                   "checks": checks, "log": L}, os.path.join(cfg.RES, "g0_data_gate.json"))
    log(f"G0 all_ok={all_ok} elapsed={time.time() - t0:.1f}s")
    if not all_ok:
        failed = [k for k, v in checks.items() if not v["ok"]]
        raise SystemExit(f"G0 failed: {failed}")


def _id_str(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s):
        return s.astype("Int64").astype(str)
    return s.astype(str).str.strip()


def _read_creator_map() -> pd.DataFrame:
    try:
        import pyarrow.csv as pacsv
        tbl = pacsv.read_csv(cfg.KS_CLEAN, read_options=pacsv.ReadOptions(use_threads=True),
                             convert_options=pacsv.ConvertOptions(
                                 include_columns=["id", "creator_id"]))
        return tbl.to_pandas()
    except Exception as exc:  # pragma: no cover
        log(f"pyarrow.csv failed ({exc}); falling back to pandas chunks")
        parts = []
        for chunk in pd.read_csv(cfg.KS_CLEAN, usecols=["id", "creator_id"], dtype=str, chunksize=200_000):
            parts.append(chunk)
        return pd.concat(parts, ignore_index=True)


def _text_diagnostics(df: pd.DataFrame, split: pd.Series) -> dict:
    name = df["name_clean"].astype(str)
    blurb = df["blurb_clean"].astype(str)
    nchar = blurb.str.len()
    nword = blurb.str.split().str.len()
    nword_name = name.str.split().str.len()
    out = {
        "n_rows": int(len(df)),
        "empty_name": int((name.str.strip() == "").sum()),
        "empty_blurb": int((blurb.str.strip() == "").sum()),
        "both_empty": int(((name.str.strip() == "") & (blurb.str.strip() == "")).sum()),
        "blurb_chars": {"median": float(nchar.median()), "p90": float(nchar.quantile(.90)),
                        "p99": float(nchar.quantile(.99)), "p999": float(nchar.quantile(.999)),
                        "max": int(nchar.max())},
        "blurb_words": {"median": float(nword.median()), "p90": float(nword.quantile(.90)),
                        "p99": float(nword.quantile(.99)), "max": int(nword.max())},
        "name_words": {"median": float(nword_name.median()), "p90": float(nword_name.quantile(.90)),
                       "max": int(nword_name.max())},
        "blurb_ellipsis_share": float(blurb.str.endswith("...").mean()),
        "empty_by_split": {s: {"name": int(((split == s) & (name.str.strip() == "")).sum()),
                               "blurb": int(((split == s) & (blurb.str.strip() == "")).sum())}
                           for s in ("train", "val", "test")},
    }
    return out


if __name__ == "__main__":
    main()
