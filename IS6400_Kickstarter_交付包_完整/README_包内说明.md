# IS6400 众筹项目 · 交付包说明

> **2026-10-03 修订提示：** `01_清洗数据/ks_clean.csv` 已按授权覆盖，现为 **673,839行 × 81列**。修复上线前履历的未来信息及USD零汇率，备份和最新规则见 `07_脚本修订验证/脚本与数据修订说明.md`。下方原交付统计、原清洗文档和一致性报告属于修订前68列版本，不应直接当作新版本验收结果。

> 数据集：Kaggle《2025 Crowdfunding & Kickstarter – 694,062 Projects》（作者 Domingo Sun，MIT 许可，v4 2026-07-22）
> 本包内容：清洗后的数据、清洗规则与理由、方法与预期模型说明、图表、简报清单。

---

## 一、包内有什么

```
IS6400_Kickstarter_交付包/
├─ README_包内说明.md                ← 本文件
├─ 01_清洗数据/
│   ├─ ks_clean.csv                  清洗后的数据（673,839 行 × 68 列）
│   └─ dropped_rows.csv.gz           被删的 20,223 行审计（id / state / goal / launched_at / reason）
├─ 02_清洗规则与理由/
│   ├─ 数据清洗文档.md                八节：来源 / 原则 / 行级 R1–R5 / 值级 V1–V8 / 列级派生列 /
│   │                                 保留不删项 / 执行结果 / 复现 —— 每条规则都带「理由」与「不删的代价」
│   ├─ 清洗脚本.py                    清洗规则的实现脚本（跑一次即可复现 ks_clean.csv）
│   ├─ 一致性审计报告.md              一致性审计：28 项全部通过
│   └─ 字段字典.md                    43 字段字典（dtype / 缺失率 / 唯一值 / 能否入模）
├─ 03_方法与预期模型/
│   ├─ 方法与预期模型说明_中文.md      预计采用什么方法、为什么选它、想产出什么模型用于什么
│   └─ 选题说明_中文.md                选题定位与冻结的关键变量、方法（含 5 节结构）
├─ 04_图表/
│   ├─ figd1_size_funnel.png … figd6_goal_dist.png   数据集介绍图（6 张）
│   ├─ fig1_goal.png … fig4_trend.png                分析发现图（4 张）
│   └─ tab1_dictionary.png                           43 字段分 9 类与「能否入模」
└─ 05_简报清单/
    └─ 简报已完成清单_中文.md          中期简报所需内容逐项打勾
```

> 若你拿到的是**轻量包**（文件名带 `_轻量`），则 `01_清洗数据/ks_clean.csv` 未包含——请从完整包或项目目录 `cleaned\` 取。

---

## 二、怎么用这份数据

```python
import pandas as pd
df = pd.read_csv("01_清洗数据/ks_clean.csv", low_memory=False)

# 1) 建模样本：只取有明确成败的项目（604,629 行）
model = df[df["is_terminal_pair"]]          # succ: 1 = successful, 0 = failed

# 2) 若要做描述统计（含第三方类别 canceled），用全量 673,839 行
df["state"].value_counts()

# 3) 真正可当特征的列（可行动 + 条件）
print(df[["log_goal", "dur_days", "category_key", "subcategory_key",
          "country_final", "creator_rank", "creator_prev_success_rate"]].head())
```

**几个必须注意的口径**

1. **成功率的分母是 604,629**（successful 297,538 + failed 307,091），不是 673,839；`canceled` 69,210 行保留为独立第三类，其 `succ` 为空——**不能当失败**。
2. 时间列（`launch_dt` / `deadline_dt`）与可空布尔列（`spotlight` / `staff_pick`）在 CSV 里都是字符串/object，需要时请显式转换：
   ```python
   df["launch_dt"] = pd.to_datetime(df["launch_dt"])
   for c in ("spotlight", "staff_pick"):
       df[c] = df[c].astype("boolean")
   ```
3. **10 个泄漏列必须剔除**才能建模：`pledged`、`usd_pledged`、`backers_count`、`percent_funded`、`converted_pledged_amount`、`spotlight`、`profile_state`、`staff_pick`、`state_changed_at`、`profile_state_changed_at`（保留在数据里供描述统计用；混入模型会让测试 AUC 变成 1.0000 的假象）。
4. 目标金额中位数 $5,000 是**全量口径**（含 canceled 等所有状态），不是 604,629 可建模口径。
5. 缺失最多的是 `profile_blurb`（85.11%，发起人档案简介）与细分品类 `category_name`（18.97%）；**一级品类零缺失**，建模用 `category_key`。

---

## 三、怎么复现

```powershell
# 需要原始文件：Kickstarter Full Campaigns_694062_25_April_2026.csv（454.9 MB）
python 清洗脚本.py                           # 重新清洗，输出 ks_clean.csv + 审计文件 + 清洗文档
```

原始 CSV 全程只读，脚本不会修改它；删除的每一行都有审计记录，可回滚。

---

## 四、数据一览（速查）

| 项 | 值 |
| --- | --- |
| 原始规模 | 694,062 行 × 43 列（454.9 MB，2026 年 4 月抓取，`collected_at` 共 124 个批次） |
| 清洗后 | 673,839 行 × 68 列（原始 43 列完整保留 + 25 个派生列） |
| 可建模 | 604,629 行（successful + failed） |
| 删除 | 20,223 行（未开始/未结束 20,215；上线时间无效 6；募款额为负 2） |
| 时间跨度 | 2009-04 ~ 2026-04，17 个上线年；年度峰值 65,127（2015） |
| 品类 | 15 个一级品类（Games 108,750 最多） |
| 国家 | 25 个平台市场（US 68.0%） |
| 目标金额 | Q1 $1,550 / 中位 $5,000 / Q3 $15,000（重尾） |

---

## 五、数字出处

包内所有数字都可在 `02_清洗规则与理由/一致性审计报告.md`（一致性审计）与
`03_方法与预期模型/选题说明_中文.md` 中核对；项目目录里另有 `numbers.md`（逐条带文件行号）与
`dataset_profile.md`（数据集概况）可供追溯。
