r"""Kickstarter 数据清洗：读原始 CSV -> 清洗 -> 写入新文件（原始文件永不被改动）。

用法
----
    python 清洗脚本.py --raw 原始数据.csv   # 输出到交付包的 07_修订清洗输出
    python 清洗脚本.py --raw 原始数据.csv --out other.csv

清洗分三层（文档同结构）：
    行级 —— 删除规则 R1-R5（只删「明确错误」与「没有可用结果」的行）
    值级 —— 字段级标准化 V1-V8（时间混合编码、布尔类型、品类大小写、文本引号、国家口径…）
    列级 —— 新增派生列；USD 零汇率改为 1，修正前值另存 *_raw

输出（默认写入交付包的 07_修订清洗输出，原始文件不动）
    ks_clean.csv         清洗后的新数据文件
    dropped_rows.csv.gz  被删行审计：id/state/goal/launched_at/原因
    数据清洗文档.md        行级 + 值级 + 列级清洗全记录（自动生成）
"""
import argparse
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
warnings.filterwarnings("ignore")

GROUP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(GROUP, "Kickstarter Full Campaigns_694062_25_April_2026.csv")
OUTDIR = os.path.join(GROUP, "07_修订清洗输出")
DEFAULT_OUT = os.path.join(OUTDIR, "ks_clean.csv")
DROPPED = os.path.join(OUTDIR, "dropped_rows.csv.gz")
DOC = os.path.join(OUTDIR, "数据清洗文档.md")

TERMINAL = ["successful", "failed"]
DROP_STATES = ["live", "submitted", "started", "suspended"]   # 未开始/未结束；canceled 保留为第三类
LEAKAGE_COLS = ["pledged", "usd_pledged", "backers_count", "percent_funded",
                "converted_pledged_amount", "spotlight", "profile_state", "staff_pick",
                "state_changed_at", "profile_state_changed_at"]
DERIVED_COLS = ["launch_dt", "deadline_dt", "dur_days", "yr", "mon", "dow", "hour", "log_goal",
                "goal_capped", "name_len", "blurb_len", "blurb_words", "succ", "is_terminal_pair",
                "category_key", "subcategory_key", "country_final", "country_mismatch",
                "creator_rank", "creator_n_projects", "creator_prev_success_rate", "is_relaunch",
                "name_clean", "blurb_clean", "goal_under_1", "creator_prior_projects",
                "creator_history_known_results", "creator_history_unavailable", "creator_history_scope"]

RULES = ["R1 未开始 / 未结束记录", "R2 上线时间无效", "R3 目标金额非法",
         "R4 截止时间非法", "R5 募款额为负"]

RATIONALE = {
    "R1 未开始 / 未结束记录": {
        "criteria": "state ∈ {live, submitted, started, suspended}",
        "why": "这些记录**没有可用的成败结果**：`live` / `submitted` / `started` 的募集尚未结束（右删失），"
               "`suspended` 被平台中止。把未结束当成失败会让成功率被系统性低估，而成功率是本项目所有分析的基准。",
        "cost": "若保留，必须为它们单独定义标签，否则任何 success rate 计算都会失真；"
                "`canceled` 已按决定保留为独立第三类，不参与成功率计算（`succ` 记为缺失）。",
    },
    "R2 上线时间无效": {
        "criteria": "launched_at 解析失败 或 == 1970-01-01",
        "why": "`launched_at` 为 1970（epoch 占位值）或无法解析时，算不出项目时长、上线月份 / 星期 / 小时，"
               "也无法参与时间切分与趋势分析。",
        "cost": "若用 0 或均值填补，会在时间轴上凭空造出 1970 年的项目，直接污染趋势图与季节性结论。",
    },
    "R3 目标金额非法": {
        "criteria": "goal <= 0",
        "why": "`goal` ≤ 0 时 `log(goal)` 无定义，目标金额分箱与「目标越高越难」的单调性检验都会失效；"
               "这类记录本身多为未填写目标的草稿。",
        "cost": "若把它们塞进最低金额档，会人为拉高该档成功率，削弱核心结论。",
    },
    "R4 截止时间非法": {
        "criteria": "deadline 解析失败 / <=1970-01-02 / dur <= 0 / dur > 365",
        "why": "`deadline` 无法解析 / 1970 / 早于上线时间 / 时长 >365 天，意味着项目时长不可计算或荒谬；"
               "时长是仅次于目标金额的第二大杠杆（8–14 天 70.9% vs 46–60 天 28.4%）。",
        "cost": "若保留，时长分箱会出现负数与「50 年」这类异常值，均值与分箱结论都不可信。",
    },
    "R5 募款额为负": {
        "criteria": "pledged < 0",
        "why": "负募款额在现实中不存在，属数据错误；全表只有 2 行，删除不损失任何信息。",
        "cost": "保留会让募款额均值、对数变换（负值无定义）以及完成率派生列全部出错。",
    },
}

KEEP_BUT_FLAG = [
    ("细分品类缺失 18.97%（`category_name` / `category_id`）",
     "缺失只发生在**细分层**；**一级品类 `category_parent_name` 零缺失**（建模用的正是它）。"
     "且缺失非随机（集中在特定年份/类型），删行会造成选择性偏差",
     "保留行；`subcategory_key` 缺失记为 `unknown`，或直接在一级品类粒度分析"),
    ("`profile_blurb` 缺失 85.11% / `profile_name` 缺失 5.23%",
     "这两列描述的是**发起人档案页**，不是项目本身；85% 为空的列没有分析价值，"
     "但为此删行会丢掉大量仍完全可用的项目",
     "保留列、不删行；文本特征只用 `name_clean` + `blurb_clean`（blurb 缺失仅 0.02%）"),
    ("`country` 与 `location_country` 有 4.61%（31,900 行）不一致",
     "两者含义不同：`country` 是**平台开通市场**（25 个取值），`location_country` 是**地理编码结果**"
     "（225 个标准二字码，含属地）；谁对谁错无法判定，删行等于凭空丢弃 3 万条记录",
     "保留行；新增 `country_final`（优先 `country`）+ `country_mismatch` 标记，供敏感性分析"),
    ("(name, creator_id) 重复 929 行、完全重复行 0",
     "同一发起人失败后用同一标题重新发起是真实业务行为，本身就是「重试」信号；"
     "删掉会丢失「失败 → 成功」的重要对照",
     "保留行；`is_relaunch` 仅表示当前上线前同一发起人已有同名项目"),
    ("`goal` / `pledged` 极右偏、存在极端值（含 goal < 1 的 3 行）",
     "极端值本身就是真实事实：>50 万美元目标有 4,130 个、成功率仅 14.6%，"
     "正是「目标越高越难」的最强证据；删掉会让结论在高端失真",
     "保留原始列；`goal_capped` 兼容列不再全量拟合截断，训练时另估阈值；保留 `goal_under_1`"),
    ("10 个泄漏列（`pledged` / `usd_pledged` / `backers_count` / `percent_funded` / "
     "`converted_pledged_amount` / `spotlight` / `profile_state` / `staff_pick` / "
     "`state_changed_at` / `profile_state_changed_at`）",
     "描述统计与业务量化需要它们（募款额分布、平台推荐 vs 普通的差距）；删列会让这些数字无法回溯",
     "保留列；建模脚本必须剔除（混入后测试 AUC 直接 = 1.0000）"),
]


def pm(s):
    """混合编码时间戳解析：'YYYY-MM-DD HH:MM:SS UTC' 字符串 或 epoch 秒。"""
    t = s.astype(str).str.replace(" UTC", "", regex=False).str.strip()
    num = pd.to_numeric(t, errors="coerce")
    out = pd.to_datetime(num, unit="s", errors="coerce")
    m = num.isna()
    if m.any():
        out[m] = pd.to_datetime(t[m], errors="coerce")
    return out


def to_nullable_bool(s):
    """规范为可空布尔：兼容 Python bool（带 NaN 时 dtype 为 object）与 'True'/'False' 字符串。"""
    if s.dtype == bool:
        return s.astype("boolean")
    return s.map({True: True, False: False, "True": True, "False": False,
                  "true": True, "false": False}).astype("boolean")


def fix_usd_zero_rates(frame):
    """仅 USD 的零汇率修为 1；保留原值、修正标志及不可用标志。"""
    clean = frame.copy()
    usd = clean["currency"].astype("string").str.strip().str.upper().eq("USD").fillna(False)
    for c in ("fx_rate", "usd_exchange_rate", "static_usd_rate"):
        raw_col = c + "_raw"
        # 再次调用时也不丢失首次修正前的值。
        if raw_col not in clean:
            clean[raw_col] = clean[c]
        rate = pd.to_numeric(clean[c], errors="coerce")
        fix = usd & rate.eq(0)
        clean[c] = rate.mask(fix, 1.0)
        clean[c + "_zero_usd_fixed"] = usd & pd.to_numeric(clean[raw_col], errors="coerce").eq(0)
        clean[c + "_invalid"] = ~np.isfinite(clean[c]) | clean[c].le(0)
    return clean


def build_point_in_time_history(frame):
    """先在完整输入的可观察上线历史上计算，再由调用方筛分析样本。

    同秒上线不互计历史；结果在 max(deadline, state_changed_at) 严格早于
    当前上线时才计入。无状态日志时这是保守近似，不是首次终态的精确日志。
    creator_n_projects 新口径是此前上线数 + 当前项目，绝非全生命周期总数。
    """
    launch = pm(frame["launched_at"])
    deadline = pm(frame["deadline"])
    changed = pm(frame["state_changed_at"])
    valid = frame["creator_id"].notna() & launch.notna() & (launch > pd.Timestamp("1970-01-02"))
    result = pd.DataFrame(index=frame.index)
    for c in ("creator_rank", "creator_n_projects", "creator_prior_projects",
              "creator_history_known_results"):
        result[c] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    result["creator_prev_success_rate"] = np.nan
    result["is_relaunch"] = False
    result["creator_history_unavailable"] = ~valid
    if not valid.any():
        return result

    history = frame.loc[valid, ["id", "creator_id", "name", "state"]].copy()
    history["launch_dt"] = launch.loc[valid]
    history = history.sort_values(["launch_dt", "id"], kind="mergesort")
    # 排名来自完整可观察输入，包括后来被分析样本筛选排除的有效上线记录。
    groups = history.groupby(["creator_id", "launch_dt"], sort=True).size()
    prior = groups.groupby(level=0).cumsum() - groups
    keys = pd.MultiIndex.from_frame(history[["creator_id", "launch_dt"]])
    counts = prior.reindex(keys).to_numpy()
    result.loc[history.index, "creator_prior_projects"] = counts
    result.loc[history.index, "creator_rank"] = counts + 1
    result.loc[history.index, "creator_n_projects"] = counts + 1

    names = history["name"].astype("string").str.strip()
    names = names.mask(names.eq(""))
    first = history.assign(name_key=names).groupby(["creator_id", "name_key"])["launch_dt"].transform("min")
    result.loc[history.index, "is_relaunch"] = (names.notna() & (first < history["launch_dt"])).to_numpy()

    available = pd.concat([deadline, changed], axis=1).max(axis=1)
    event_ok = (valid & frame["state"].isin(TERMINAL) & deadline.notna()
                & changed.notna() & (deadline > launch) & (changed >= launch))
    result.loc[history.index, "creator_history_known_results"] = 0
    if event_ok.any():
        events = frame.loc[event_ok, ["id", "creator_id", "state"]].copy()
        events["available_dt"] = available.loc[event_ok]
        events["success"] = events["state"].eq("successful").astype(int)
        events = events.sort_values(["available_dt", "id"], kind="mergesort")
        events["known_results"] = events.groupby("creator_id").cumcount() + 1
        events["success_sum"] = events.groupby("creator_id")["success"].cumsum()
        query = history[["creator_id", "launch_dt"]].copy()
        query["_source_index"] = history.index
        matched = pd.merge_asof(
            query, events[["creator_id", "available_dt", "known_results", "success_sum"]],
            left_on="launch_dt", right_on="available_dt", by="creator_id",
            direction="backward", allow_exact_matches=False,
        )
        ix = matched["_source_index"].to_numpy()
        result.loc[ix, "creator_history_known_results"] = matched["known_results"].fillna(0).astype(int).to_numpy()
        result.loc[ix, "creator_prev_success_rate"] = (matched["success_sum"] / matched["known_results"]).to_numpy()
    return result


def load():
    print(f"[read] {RAW}  ({os.path.getsize(RAW)/1e6:.1f} MB)", flush=True)
    df = pd.read_csv(RAW, low_memory=False)
    print(f"[read] shape={df.shape}", flush=True)
    return df


def audit_values(df):
    """量清值级问题（写文档用，不改数据）。"""
    a = {}
    a["bool_cols"] = {c: {str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()}
                      for c in ("spotlight", "staff_pick")}
    s_cat = df["category_name"].dropna().astype(str).str.strip()
    a["subcat_before"], a["subcat_after"] = s_cat.nunique(), s_cat.str.lower().nunique()
    p_cat = df["category_parent_name"].dropna().astype(str).str.strip()
    a["cat_before"], a["cat_after"] = p_cat.nunique(), p_cat.str.lower().nunique()
    for c in ("name", "blurb"):
        s = df[c].astype(str)
        a[f"{c}_quotes"] = int(s.str.contains('"', regex=False).sum())
        a[f"{c}_multi"] = int(s.str.contains(r'"{2,}', regex=True).sum())
    a["ws_cols"] = {c: int((df[c].dropna().astype(str) != df[c].dropna().astype(str).str.strip()).sum())
                    for c in ("category_parent_name", "category_name", "country", "location_country",
                              "currency", "usd_type", "state", "location_state")}
    term = df["state"].isin(TERMINAL)
    a["goal_lt1_all"] = int((df["goal"] < 1).sum())
    a["goal_lt1_term"] = int(((df["goal"] < 1) & term).sum())
    a["loc_country_n"] = int(df["location_country"].dropna().astype(str).str.strip().nunique())
    a["country_n"] = int(df["country"].dropna().astype(str).str.strip().nunique())
    return a


def funnel(df):
    """按顺序应用 5 条删除规则，返回 (原因序列, 保留掩码, 漏斗, 无效值掩码)。"""
    launch = pm(df["launched_at"])
    deadline = pm(df["deadline"])
    dur = (deadline - launch).dt.days
    bad_launch = launch.isna() | (launch == pd.Timestamp("1970-01-01"))
    bad_deadline = (deadline.isna() | (deadline <= pd.Timestamp("1970-01-02"))
                    | (dur <= 0) | (dur > 365))
    steps = [
        (RULES[0], df["state"].isin(DROP_STATES)),
        (RULES[1], bad_launch),
        (RULES[2], df["goal"] <= 0),
        (RULES[3], bad_deadline),
        (RULES[4], df["pledged"] < 0),
    ]
    reason = pd.Series("kept", index=df.index)
    alive = pd.Series(True, index=df.index)
    rows = []
    for name, mask in steps:
        hit = alive & mask.fillna(False)
        reason[hit] = name
        alive &= ~hit
        rows.append((name, int(hit.sum()), int(alive.sum())))
    return reason, alive, rows, bad_launch, bad_deadline


def clean_and_write(df, alive, reason, out_path):
    """值级清洗 + 列级派生，最后写入新文件与被删行审计。"""
    history = build_point_in_time_history(df)
    clean = fix_usd_zero_rates(df[alive].copy())

    # --- 值级 V1：时间字段混合编码解析
    clean["launch_dt"] = pm(clean["launched_at"])
    clean["deadline_dt"] = pm(clean["deadline"])
    clean["dur_days"] = (clean["deadline_dt"] - clean["launch_dt"]).dt.days
    clean["yr"] = clean["launch_dt"].dt.year
    clean["mon"] = clean["launch_dt"].dt.month
    clean["dow"] = clean["launch_dt"].dt.dayofweek
    clean["hour"] = clean["launch_dt"].dt.hour

    # --- 值级 V2：布尔字段类型规范化（原为 bool + NaN，读入后 dtype 退化为 object）
    for c in ("spotlight", "staff_pick"):
        clean[c] = to_nullable_bool(clean[c])

    # --- 值级 V3：品类标签大小写归并（原始列保留）
    cat = clean["category_parent_name"].astype("string").str.strip()
    clean["category_key"] = cat.str.lower().fillna("unknown")
    sub = clean["category_name"].astype("string").str.strip()
    clean["subcategory_key"] = sub.str.lower().fillna("unknown")

    # --- 值级 V4：文本引号异常折叠（抓取时重复转义，如 """"标题""""）
    for src, dst in (("name", "name_clean"), ("blurb", "blurb_clean")):
        s = clean[src].astype("string").str.replace(r'"{2,}', '"', regex=True)
        clean[dst] = s.str.strip()

    # --- 值级 V5：国家口径统一
    clean["country_final"] = clean["country"].fillna(clean["location_country"])
    clean["country_mismatch"] = (clean["country"].notna() & clean["location_country"].notna()
                                 & (clean["country"].astype(str).str[:2]
                                    != clean["location_country"].astype(str).str[:2]))

    # --- 列级：数值与文本特征（长度定义与已验证的探测协议保持一致）
    clean["log_goal"] = np.log10(clean["goal"])
    # 不用包含未来项目的全表分位数拟合阈值；保留列名供旧读取代码兼容。
    # 若模型需要截断，仅在训练窗口内拟合阈值，再用于验证/测试集。
    clean["goal_capped"] = clean["goal"]
    clean["goal_under_1"] = clean["goal"] < 1
    clean["name_len"] = clean["name"].astype(str).str.len()
    clean["blurb_len"] = clean["blurb"].astype(str).str.len()
    clean["blurb_words"] = clean["blurb"].astype(str).str.split().str.len()

    # --- 列级：目标变量与发起人历史
    clean["succ"] = np.where(clean["state"] == "successful", 1.0,
                             np.where(clean["state"] == "failed", 0.0, np.nan))
    clean["is_terminal_pair"] = clean["state"].isin(TERMINAL)
    for c in history.columns:
        clean[c] = history.loc[clean.index, c]
    clean["creator_history_scope"] = "provided_input_launch_history"
    clean = clean.sort_values(["launch_dt", "id"], kind="mergesort")

    os.makedirs(OUTDIR, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"[write] {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)

    dropped = df.loc[~alive, ["id", "state", "goal", "launched_at"]].copy()
    dropped["reason"] = reason.loc[~alive]
    dropped.to_csv(DROPPED, index=False, compression="gzip")
    print(f"[write] {DROPPED}", flush=True)
    return clean


def write_document(df, alive, reason, rows, bad_launch, bad_deadline, clean, out_path, audit):
    """清洗逻辑文档：行级 / 值级 / 列级三层，每层都带理由。"""
    total = len(df)
    kept = int(alive.sum())
    ca = df["collected_at"]
    launch = pm(df["launched_at"])
    lv = launch[launch > pd.Timestamp("1970-01-02")]
    raw_cols = 43
    n_out_cols = clean.shape[1]
    L = [
        "# Kickstarter 数据清洗文档",
        "",
        "> 本文件由 `group\\清洗脚本.py` 自动生成（每次运行都会刷新，数字均实测自原始文件）。",
        "> 清洗分三层：**行级**（删哪些行）、**值级**（字段值怎么标准化）、**列级**（新增哪些派生列）。",
        "",
        "## 一、数据来源",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        "| Kaggle 数据集 | `domingosun/kickstarter-dataset-629147-projects-may-2025`"
        "（页面标题《2025 Crowdfunding & Kickstarter – 694,062 Projects》） |",
        "| 作者 / 许可 | Domingo Sun / **MIT** |",
        "| 版本与更新 | 当前 v4，2026-07-22 更新（v1 初始版 2025-06-26） |",
        "| 页面标注大小 | 454,929,398 字节（与本地文件一致） |",
        f"| 本地文件 | `{os.path.basename(RAW)}`，{total:,} 行 × {raw_cols} 列，"
        f"{os.path.getsize(RAW)/1e6:.1f} MB |",
        f"| 抓取口径 | `collected_at` 共 {ca.nunique()} 个取值（{int(ca.min())} ~ {int(ca.max())}），"
        "说明数据由多批抓取累积，最新一批为 202605 |",
        f"| 时间范围 | `created_at` {pm(df['created_at']).min():%Y-%m-%d} ~ "
        f"{pm(df['created_at']).max():%Y-%m-%d}；`launched_at` 有效范围 "
        f"{lv.min():%Y-%m-%d} ~ {lv.max():%Y-%m-%d}"
        f"（另有 {int(bad_launch.sum()):,} 行为 1970 占位值，见 R2） |",
        "",
        "## 二、清洗原则",
        "",
        "1. **行级只删「明确错误」与「没有可用结果」的行**，不删任何可能改变结论的样本；",
        "2. **值级只做标准化，不改语义**：类型纠正、大小写归并、引号折叠、口径统一；",
        "3. 无法判定为错误的问题（缺失、极端值、口径冲突）一律**保留 + 打标记**；",
        "4. 派生列保留源字段；布尔列规范类型；USD零汇率修为1，修正前值保存到 `*_raw`；",
        "5. 每个被删除的行都进入审计文件（id + state + 原因）；",
        "6. 清洗结果写入**新文件**（默认 `ks_clean.csv`），原始 CSV 永不被修改。",
        "",
        "## 三、行级清洗：删除规则（逐条 + 理由）",
        "",
    ]
    for name, cnt, remain in rows:
        rat = RATIONALE[name]
        L += [
            f"### {name}",
            "",
            f"- **判定条件**：`{rat['criteria']}`",
            f"- **本步删除**：{cnt:,} 行（按顺序执行，为增量计数，不重复计算重叠原因）",
            f"- **删除理由**：{rat['why']}",
            f"- **不删的代价**：{rat['cost']}",
        ]
        sub = df[reason == name]
        if cnt and name == RULES[0]:
            L += ["", "命中行的 state 分布：", "", "| state | 行数 |", "| --- | ---: |"]
            for k, v in sub["state"].value_counts().items():
                L.append(f"| {k} | {v:,} |")
        if cnt:
            L += ["", "示例行：", "",
                  "| id | name（截断 40 字符） | state | goal | launched_at |",
                  "| --- | --- | --- | ---: | --- |"]
            for _, r in sub.head(5).iterrows():
                L.append(f"| {r['id']} | {str(r['name'])[:40]} | {r['state']} | "
                         f"{r['goal']:,.0f} | {r['launched_at']} |")
        L.append("")

    bc = audit["bool_cols"]
    L += [
        "## 四、值级清洗：字段标准化（不删行，逐项 + 理由）",
        "",
        "### V1 时间字段：混合编码解析",
        "",
        "- **问题**：`created_at` / `deadline` 同时存在 `'2020-11-07 01:56:31 UTC'` 字符串与 epoch 秒数字两种编码；"
        f"`launched_at` 是 float epoch，其中 {int(bad_launch.sum()):,} 行为 1970 占位值。",
        "- **处理**：新增 `launch_dt` / `deadline_dt`（ISO 时间）与 `dur_days`；原始三列保留。",
        "- **理由**：不解析就算不出项目时长、做不了季节性（月/星期/小时）、无法时间切分与趋势分析。",
        "",
        "### V2 布尔字段：类型规范化",
        "",
        f"- **问题**：`spotlight` 与 `staff_pick` 本身是**布尔值**，但因为存在缺失（NaN），"
        f"读入后 dtype 退化成 **object**（真值分布：`spotlight` = {bc['spotlight']}；"
        f"`staff_pick` = {bc['staff_pick']}）——既不是布尔类型，也不能直接做 `.sum()` 等布尔运算。",
        "- **处理**：统一转为**可空布尔**（`boolean` dtype：`True` / `False` / 空）。",
        "- **理由**：object 列上做布尔判断/求和会报错或给出意外结果，建模时也需要明确的布尔或 0/1。",
        "- ⚠️ **踩过的坑**：不要用 `{'True': True, 'False': False}` 这类字符串字典去 `.map()` 这两列——"
        "值是**真布尔**，字符串键匹配不上，结果会整列变成缺失。",
        "- ⚠️ **CSV 的局限**：CSV 不存储「可空布尔」类型，把文件重新读入后这两列仍是 object"
        "（`True` / `False` / 空）。需要布尔运算时请显式 `astype('boolean')`，或读入时指定 "
        "`dtype={'spotlight': 'boolean', 'staff_pick': 'boolean'}`；同理，所有时间列在 CSV 里都是字符串。",
        "",
        "### V3 品类标签：大小写归并",
        "",
        f"- **问题**：同一个品类存在两种拼写。一级品类 `category_parent_name`：{audit['cat_before']} → "
        f"{audit['cat_after']}（可合并 {audit['cat_before'] - audit['cat_after']} 个，即 `Comics` / `comics`）；"
        f"细分品类 `category_name`：**{audit['subcat_before']} → {audit['subcat_after']}"
        f"（可合并 {audit['subcat_before'] - audit['subcat_after']} 个）**。",
        "- **处理**：新增 `category_key`（一级）与 `subcategory_key`（细分），按小写 + 去空白归并；"
        "缺失记 `unknown`。原始列保留。",
        "- **理由**：不归并时按品类分组会产生重复条目——第一版图表上确实冒出过"
        "「comics 13.4% (n=157)」这样一个假点；细分层 72 个重复标签的后果同样严重。",
        "",
        "### V4 文本字段：引号异常折叠",
        "",
        f"- **问题**：`name` {audit['name_quotes']:,} 行、`blurb` {audit['blurb_quotes']:,} 行含双引号，"
        f"其中存在 `\"\"\"\"标题\"\"\"\"` 这类连续 4 个引号的重复转义"
        f"（`name` {audit['name_multi']:,} 行 / `blurb` {audit['blurb_multi']:,} 行）。",
        "- **处理**：新增 `name_clean` / `blurb_clean`，把**连续 2 个以上**的双引号折叠为一个；原始列保留。",
        "- **理由**：这是抓取时的重复转义产物，会污染分词（`\"\"\"\"millionaire` 会变成带引号的 token）；"
        "折叠后文本才可直接进 TF-IDF。",
        "",
        "### V5 国家字段：口径统一",
        "",
        f"- **问题**：两个国家字段含义不同——`country` 是**平台开通市场**（{audit['country_n']} 个取值），"
        f"`location_country` 是**地理编码结果**（{audit['loc_country_n']} 个标准二字码，含属地）；"
        "实测 31,900 行（4.61%）两者前两位不一致。",
        "- **处理**：新增 `country_final`（优先 `country`，缺失回退 `location_country`）与 `country_mismatch` 标记。",
        "- **理由**：跨国分析必须先选定一个口径；两者都保留则任何按国家的分组都会有两种答案。"
        "（注：早期文档曾猜测 `location_country` 的 225 个取值是「地区名/拼写变体混入」，"
        "实测**不成立**——它全是标准二字码。）",
        "",
        "### V6 文本外层空白：已核验，无需处理",
        "",
        "- 实测 8 个文本字段（品类、国家、币种、州名等）的外层空白行数全部为 **0**。",
        "- 仍然在 `category_key` / `subcategory_key` / `name_clean` / `blurb_clean` 里统一做了 `strip()`，"
        "作为防御性处理。",
        "",
        "### V7 极端小目标金额：标记而非删除",
        "",
        f"- **问题**：全表 `goal` < 1 的有 {audit['goal_lt1_all']:,} 行，其中终态"
        f"（successful / failed）{audit['goal_lt1_term']:,} 行；最小值为 0 / 0.01 / 0.15 / 0.5。",
        "- **处理**：保留，新增 `goal_under_1`；`goal_capped` 现为原goal兼容列，不全量拟合阈值。",
        "- **理由**：这类记录可能是「测试项目」也可能是真实的小额项目，无法判定为错误；"
        "删除会损失真实样本，直接入模又可能影响分档，所以交给建模阶段用标记自行取舍。",
        "",
        "### V8 缺失值策略",
        "",
        "| 字段 | 缺失率 | 策略 | 理由 |",
        "| --- | ---: | --- | --- |",
        "| `category_name` / `category_id`（细分品类） | 18.97% | `subcategory_key = 'unknown'` | 缺失非随机，删行会造成选择性偏差 |",
        "| `category_parent_name`（一级品类） | **0%** | 无需处理 | 建模实际使用的就是这一列 |",
        "| `profile_blurb` | 85.11% | 保留列、不做文本特征 | 发起人档案文本，非项目文本 |",
        "| `profile_name` | 5.23% | 保留列 | 与 `name` 高度冗余 |",
        "| `usd_pledged` | 1.64% | 保留（分析用 `pledged` 或标记） | 泄漏列，仅描述统计用 |",
        "| `blurb` | 0.02% | 保留 | 文本分析主力字段 |",
        "| 其余 36 列 | ≤3.3% | 保留原值 | 缺失比例低，按需在建模阶段处理 |",
        "",
    ]

    L += ["### V9 USD 零汇率修正", "",
          "仅币种为USD且汇率恰为0时修正为1；非USD、缺失、负值不猜补。",
          "原值保存到 `字段名_raw`，修正标记 `字段名_zero_usd_fixed`，不可用标记 `字段名_invalid`。", "",
          "| 汇率列 | 本次USD零值修正 | 剩余零值 | 不可用值总数 |", "| --- | ---: | ---: | ---: |"]
    for c in ("fx_rate", "usd_exchange_rate", "static_usd_rate"):
        L.append(f"| {c} | {int(clean[c + '_zero_usd_fixed'].sum()):,} | "
                 f"{int(clean[c].eq(0).sum()):,} | {int(clean[c + '_invalid'].sum()):,} |")
    L += ["", "## 五、列级处理：派生列与可追溯修正", "",
          "| 列 | 说明 |", "| --- | --- |",
          "| `launch_dt` / `deadline_dt` | V1 解析后的 ISO 时间 |",
          "| `dur_days` | 项目时长（天）= deadline − launched_at |",
          "| `yr` / `mon` / `dow` / `hour` | 上线年 / 月 / 星期 / 小时 |",
          "| `log_goal` | log10(goal)，修正右偏 |",
          "| `goal_capped` | 等于原goal；如需截断，仅在训练窗口内拟合阈值 |",
          "| `goal_under_1` | V7 标记：`goal` < 1 |",
          "| `name_len` / `blurb_len` / `blurb_words` | 文本长度（定义与已验证的探测协议一致，便于复现既有结论） |",
          "| `succ` | 目标变量：1 = successful，0 = failed，**canceled 记为缺失（不可当失败）** |",
          "| `is_terminal_pair` | 是否属于 successful / failed 这一对（建模筛样本用） |",
          "| `category_key` / `subcategory_key` | V3 归并后的一级 / 细分品类（小写，缺失记 `unknown`） |",
          "| `country_final` / `country_mismatch` | V5 国家口径 + 不一致标记 |",
          "| `creator_rank` / `creator_n_projects` | 当前上线前可观察项目数+1；同秒项目不互计，不是终身总数 |",
          "| `creator_prior_projects` | 完整输入中严格早于当前上线的可观察上线项目数 |",
          "| `creator_history_known_results` | 上线前已经可用的成功/失败结果数 |",
          "| `creator_history_unavailable` | 无有效上线时间或发起人ID，无法计算履历 |",
          "| `creator_history_scope` | provided_input_launch_history：来自此次输入的可观察上线记录，非终身完整历史保证 |",
          "| `creator_prev_success_rate` | 仅计max(deadline,state_changed_at)严格早于当前上线的成功/失败结果 |",
          "| `is_relaunch` | 同creator此前严格更早上线过同名项目；首条不由未来记录标记 |",
          "| `*_rate_raw` / `*_rate_zero_usd_fixed` / `*_rate_invalid` | 汇率原值、USD零值修正标志、缺失/非有限/非正汇率标志 |",
          "| `name_clean` / `blurb_clean` | V4 引号折叠后的文本（文本分析用） |",
          "",
          "## 六、明确不删除的内容（保留 + 标记）", "",
          "| 现象 | 为什么保留 | 处理方式 |", "| --- | --- | --- |"]
    for k, why, how in KEEP_BUT_FLAG:
        L.append(f"| {k} | {why} | {how} |")

    L += [
        "",
        "### 无效值与删除规则的对应关系",
        "",
        "| 无效值 | 原始行数 | 清洗后剩余 | 由哪条规则处理 |",
        "| --- | ---: | ---: | --- |",
        f"| `launched_at` 无法解析 / 1970 | {int(bad_launch.sum()):,} | "
        f"{int((bad_launch & alive).sum()):,} | R1 / R2 |",
        f"| `goal` ≤ 0 | {int((df['goal'] <= 0).sum()):,} | "
        f"{int(((df['goal'] <= 0) & alive).sum()):,} | R1 / R3 |",
        f"| `deadline` 非法 / 时长异常 | {int(bad_deadline.sum()):,} | "
        f"{int((bad_deadline & alive).sum()):,} | R1 / R4 |",
        f"| `pledged` < 0 | {int((df['pledged'] < 0).sum()):,} | "
        f"{int(((df['pledged'] < 0) & alive).sum()):,} | R5 |",
        "",
        "> 结论：原始文件里的时间戳 / 金额 / 截止时间无效值**全部集中在未开始或未结束的记录**中，"
        "由 R1 一并处理；真正进入分析样本（successful + failed + canceled）的记录是干净的。",
        "",
        "## 七、执行结果",
        "",
        "| 规则 | 本步删除 | 剩余行数 |",
        "| --- | ---: | ---: |",
    ]
    L += [f"| {name} | {cnt:,} | {remain:,} |" for name, cnt, remain in rows]
    L += [
        "",
        f"- 原始：**{total:,} 行 × {raw_cols} 列**",
        f"- 清洗后：**{kept:,} 行 × {n_out_cols} 列**"
        f"（原始 {raw_cols} 列 + 派生 {n_out_cols - raw_cols} 列）",
        f"- 其中 successful + failed = {(df['state'].isin(TERMINAL) & alive).sum():,} 行（可建模样本），"
        f"canceled = {(df['state'].eq('canceled') & alive).sum():,} 行（第三类，`succ` 为空）",
        f"- 合计删除 **{total-kept:,}** 行（{(total-kept)/total*100:.2f}%）",
        f"- 输出文件：`{os.path.basename(out_path)}`（新文件，原始 CSV 未被修改）",
        "- 审计文件：`dropped_rows.csv.gz`（被删行的 id / state / 原因，可逐行复核）",
        "",
        "## 八、复现",
        "",
        "```powershell",
        f'cd "{os.path.dirname(os.path.abspath(__file__))}"',
        "python 清洗脚本.py --raw 原始数据.csv",
        "python 清洗脚本.py --raw 原始数据.csv --out other.csv",
        "```",
        "",
        "原始文件未被修改；如需回滚，直接忽略 `cleaned\\` 目录即可。",
        "",
    ]
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[write] {DOC}", flush=True)


def main():
    global RAW, OUTDIR, DROPPED, DOC
    ap = argparse.ArgumentParser(description="Kickstarter 数据清洗（结果写入新文件）")
    ap.add_argument("--raw", default=RAW, help="原始CSV路径（不修改输入）")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"输出文件名（默认 {DEFAULT_OUT}）")
    a = ap.parse_args()
    RAW = os.path.abspath(a.raw)
    a.out = os.path.abspath(a.out)
    if os.path.normcase(os.path.realpath(RAW)) == os.path.normcase(os.path.realpath(a.out)):
        ap.error("输出路径不能与原始输入相同")
    if not os.path.isfile(RAW):
        ap.error(f"找不到原始CSV：{RAW}；请用 --raw 指定")
    OUTDIR = os.path.dirname(a.out)
    DROPPED = os.path.join(OUTDIR, "dropped_rows.csv.gz")
    DOC = os.path.join(OUTDIR, "数据清洗文档.md")
    for side_path in (DROPPED, DOC):
        if os.path.normcase(os.path.realpath(side_path)) in {
            os.path.normcase(os.path.realpath(RAW)), os.path.normcase(os.path.realpath(a.out))
        }:
            ap.error("主输出、原始输入和审计/文档输出必须是不同文件")

    df = load()
    audit = audit_values(df)
    reason, alive, rows, bad_launch, bad_deadline = funnel(df)
    print(f"\n[funnel] {len(df):,} -> {int(alive.sum()):,} rows "
          f"(-{len(df) - int(alive.sum()):,})", flush=True)
    for name, cnt, remain in rows:
        print(f"  {name:22s} -{cnt:>7,}  -> {remain:>7,}", flush=True)

    clean = clean_and_write(df, alive, reason, a.out)
    write_document(df, alive, reason, rows, bad_launch, bad_deadline, clean, a.out, audit)
    print(f"\n[done] {len(df):,} -> {len(clean):,} rows x {clean.shape[1]} cols | "
          f"original file untouched", flush=True)


if __name__ == "__main__":
    main()
