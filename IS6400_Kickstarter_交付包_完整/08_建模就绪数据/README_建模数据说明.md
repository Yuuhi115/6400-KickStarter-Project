# 建模就绪数据说明（v1）

> 生成时间：2026-11-02　｜　生成脚本：`09_分析脚本/_build_model_ready.py`
> 数据来源：`01_清洗数据/ks_clean.csv`（81 列版本，sha256 `aaf7a89e…`，710,301,125 B）
> 本文件**不覆盖任何原始/清洗文件**，只新增一列块 `m_*`；原始 81 列逐列保留。

## 一、文件清单

| 文件 | 规模 | sha256 |
| --- | --- | --- |
| `ks_model_ready_v1.parquet` | **673,839 行 × 92 列**，231,240,096 B（zstd 压缩） | `71dea1f8bf94fbd4…` |
| `ks_model_ready_usdonly.parquet` | **460,596 行 × 92 列**（仅 USD 计价项目，附录稳健性用），143,756,030 B | 见 `build_report_usdonly.json` |
| `build_report.json` | 构建统计（行数、各新列缺失、金额分布） | — |
| `build_report_usdonly.json` | USD-only 版的构建与校验统计（含六档表） | — |

CSV 版可按需生成（约 730 MB），默认只给 parquet（保留可空类型，避免 CSV 的布尔/时间退化）。

## 二、新增的 11 列（只增不改）

| 列 | 类型 | 定义 | 缺失 |
| --- | --- | --- | ---: |
| `m_goal_usd` | Float64 | `goal × static_usd_rate`（方向已用币种中位汇率验证：USD=1.0000、GBP=1.3376、JPY=0.0074） | 1,487 |
| `m_goal_usd_ok` | boolean | 该行 `m_goal_usd` 是否可用（含下方填充规则） | 0 |
| `m_goal_usd_filled` | boolean | 仅汇率缺失、但项目本身是 USD 而被精确填充的行 | 0 |
| `m_currency_usd` | boolean | 项目是否以 USD 计价 | 0 |
| `m_log_goal_usd` | Float64 | `log10(m_goal_usd)`，用于右偏修正 | 1,487 |
| `m_dur_days_exact` | Float64 | 精确时长（天，带小数）= deadline − launch；**不再用 `.dt.days` 截断** | 0 |
| `m_name_len` | Int64 | `name_clean` 的字符数，**原文缺失则为 NA（不再计成 3）** | 5 |
| `m_blurb_len` | Int64 | `blurb_clean` 的字符数，缺失为 NA | 47 |
| `m_blurb_words` | Int64 | `blurb_clean` 的词数，缺失为 NA | 47 |
| `m_split` | string | `train` = `yr ≤ 2022`；`test` = `yr ≥ 2023` | 0 |
| `m_is_modelable` | boolean | 与 `is_terminal_pair` 相同（successful + failed） | 0 |

### 关键处理规则（3 条）

1. **USD 项目的精确填充**：`currency = USD` 的项目，若 `static_usd_rate` 缺失，则 `m_goal_usd = goal`（USD→USD 恒为 1，属**精确值而非估算**），共 **7,836 行**，以 `m_goal_usd_filled = True` 标记。
2. **非 USD 且汇率缺失的 1,487 行不填**，`m_goal_usd = NA`，由建模脚本自行排除或单列标记。
3. **文本长度改为缺失感知**：原 `name_len` / `blurb_len` 把缺失文本算成 3 字符（审核指出的缺陷），新列用 `NA`。

## 三、口径（写报告必须一致）

- **可建模样本 = 604,629 行**（`m_is_modelable`）；`canceled` 69,210 行保留在文件里，其 `succ` 为空，**不算失败**。
- **成功率分母 = 604,629**，不是 673,839。
- **时间切分**：`m_split` 已预置 —— 可建模集内 **train 504,160 / test 100,469**。
  ⚠️ 旧文档写的 502,975 是此前另一口径（特征完整筛选后的行数），**以 504,160 为准**。
- **金额档（换算后，可建模集）**：

| ≤1k | 1–5k | 5–10k | 10–20k | 20–50k | >50k |
| --- | --- | --- | --- | --- | --- |
| 68.17% (n=125,899) | 54.16% (n=201,197) | 46.33% (n=104,673) | 40.64% (n=72,879) | 30.41% (n=63,636) | 14.84% (n=35,468) |

（相比换算前，>50k 从 21.57% 降到 14.84%，因为 JPY/MXN 等高名义额项目被正确归入低档。）

## 四、必须剔除的 10 个事后/泄漏列

`pledged`、`usd_pledged`、`backers_count`、`percent_funded`、`converted_pledged_amount`、`spotlight`、`profile_state`、`staff_pick`、`state_changed_at`、`profile_state_changed_at`

> 它们**保留在文件里**供描述统计；混入模型会让测试 AUC 变成 1.0000 的假象。

## 五、推荐的特征白名单（上线前可得）

```python
FEATURES = [
    "m_log_goal_usd", "m_dur_days_exact",           # 可行动：金额、时长
    "m_name_len", "m_blurb_len", "m_blurb_words",   # 文案长度（文本本身用 TF-IDF）
    "category_key", "subcategory_key",              # 市场背景
    "country_final", "location_type", "m_currency_usd",
    "creator_rank", "creator_prev_success_rate", "creator_history_known_results",  # 上线前履历
    "yr", "mon", "dow", "hour",                     # 时间背景（只作控制，不当建议）
]
TARGET   = "succ"            # 1/0，canceled 为 NA
MASK     = "m_is_modelable"
```

## 六、已知但**未在本文件修复**的问题（写进 limitation）

| 问题 | 规模 | 说明 |
| --- | --- | --- |
| 汇率是项目级静态值 | 全表 | 非逐日牌价，换算为近似 |
| `percent_funded` 与 pledged/goal 不一致 | 341 行 | 仅影响描述性指标 |
| 状态与金额冲突 | 22 行 | 未改标签，需逐条核对 |
| 履历只覆盖"保留样本" | 全表 | `creator_history_scope = retained_clean_sample`；用 `--raw` 重跑可改善 |
| 时间/布尔列在 CSV 中是字符串 | — | 本 parquet 已保留正确类型，CSV 版需显式转换 |
| `goal_capped ≡ goal`、`creator_n_projects ≡ creator_rank` | 全表 | 冗余列，勿重复入模 |

## 七、用法

```python
import pandas as pd

df = pd.read_parquet(r"08_建模就绪数据\ks_model_ready_v1.parquet")
model = df[df["m_is_modelable"]]                 # 604,629 行
train = model[model["m_split"] == "train"]       # 504,160
test  = model[model["m_split"] == "test"]        # 100,469

X = train[FEATURES]        # 见第五节
y = train["succ"].astype(float)
```

## 八、复现

```powershell
cd "E:\Workspace\IS6400\group\IS6400_Kickstarter_交付包_完整\09_分析脚本"
python _build_model_ready.py     # 约 3–5 分钟，重新生成 parquet 与 build_report.json
python build_usd_only.py         # 约 1 分钟，生成 USD-only 附录文件
```

## 八·五、USD-only 附录文件（`ks_model_ready_usdonly.parquet`）

用途：**只作稳健性检验**（写进报告附录），证明"汇率换算没有改变结论"，不作为主分析口径。

| 项 | 值 |
| --- | --- |
| 行/列 | **460,596 × 92**（仅 `m_currency_usd = True`，含 canceled 等不可建模行，保持与主文件同构） |
| 可建模 | **415,619**（train 361,184 / test 54,435） |
| 成功率 | 48.75% |
| `m_goal_usd` | 与 `goal` **完全相等**（USD→USD），无缺失 |
| 校验 | 非 USD 行 = **0**；`abs(m_goal_usd − goal).max() < 1e-9` |
| 体积 | 143,756,030 B |

**六档达标率（USD-only，可建模）**：≤1k 65.66%｜1–5k 53.66%｜5–10k 46.76%｜10–20k 41.61%｜20–50k 31.40%｜>50k 15.72%

与全量换算口径（68.17 / 54.16 / 46.33 / 40.64 / 30.41 / 14.84）对比：**最大差 2.9pp（最低档）**，结论一致。

⚠️ 使用限制：USD-only 只剩 **3 个国家**（US 415,609 / CA 8 / PL 2），且 games −2.83pp、technology −1.22pp、design −0.94pp 被系统性低估 → **不可用它替代主分析**，也不要在它上面做跨国比较。

## 九、版本记录

- **v1（2026-11-02）**：首次生成。92 列 = 81 原始/派生 + 11 个 `m_*`；`m_goal_usd` 覆盖 672,352/673,839 行（99.78%），其中 7,836 行为 USD 精确填充。
- **v1 + USD-only 附录（2026-11-02）**：新增 `ks_model_ready_usdonly.parquet`（460,596 × 92，可建模 415,619），用于附录稳健性检验；生成脚本 `09_分析脚本/build_usd_only.py`。
