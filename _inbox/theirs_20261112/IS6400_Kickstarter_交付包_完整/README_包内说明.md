# IS6400 众筹项目 · 交付包说明

> **2026-11-12 补充（最新，先读这段）：** **TF-IDF 文本层已完成**并并入新目录 **`12_文本层_TFIDF/`**（47 个文件）。
> 包内现为 **13 个目录 / 186 个文件**（含数据本体时为 187；`ks_clean.csv` 710 MB 数据本体已按下方「数据本体不进分发版」的既有约定移出，需要时从原始来源或 `_backup_交付包_20261111_pre_textlayer` 取）。并入后逐文件 sha256 复核：既有 140 个文件内容零改动（只对本 README 与 `交付说明_期末进度.md` 做了进度补充，两次改动的 sha256 前后值记录在工作区 `13_TFIDF文本层_v2/02_结果/pkg_docs_updated.csv`）。
> 文本层结论：HistGB **0.8823 → 0.8847（+0.24pp AUC）**、LR **0.8464 → 0.8554（+0.90pp）**、RF **0.8749 → 0.8609（−1.40pp）**；
> 按发起人整簇 bootstrap 的正向增量只在 HistGB / LR 上显著（95%CI 排除 0）；行级 i.i.d. 对照修正后与整簇一致（见 `results_text_layer.md` §4.3 修订记录）。决策影响：top-10% 名单换掉约 17%，约 4.1% 测试行判定翻转（上调 2.1% / 下调 2.1%）。
> ⚠️ 组内 `11_工作计划/期末计划_v2.md` 的 S1 验收线「TF-IDF 增量 ≥ +0.02 AUC」：**本次 12 个配置均未达到**，按 S1 约定入附录；是否修改验收口径属组内决策，三条数值证据与建议见 `12_文本层_TFIDF/04_报告/results_text_layer.md` §6；
> 报告第 6 章可直接粘贴 `12_文本层_TFIDF/04_报告/report_section_中文.md`。

> **2026-11-08 补充：** 本 README 描述的是 **10 月初的原始交付包**（当时 68 列、目录只到 `05_简报清单`）。现在包内已扩到 **13 个目录 / 186 个文件**（2026-11-12 实测；并入文本层前为 140 个文件；`ks_clean.csv` 已移出分发版，含它时为 187），新增了 `08_建模就绪数据`（92 列 parquet 说明）、`09_分析脚本`（12 个脚本）、`10_组员pilot研究`、`11_工作计划`、`11_结果与报告`（含已训练模型 `model_histgb_ours.joblib`）以及 `06_独立清洗审核`、`07_脚本修订验证`、`04_图表`、**`12_文本层_TFIDF`（2026-11-12：文本层脚本/结果/图/报告/两个可部署模型）**。
> **权威说明请读根目录 `交付说明_期末进度.md`**（十节；2026-11-12 在 §六 补入文本层完成情况），本 README 只作旧版参考。
> **体积提示：** 本 zip **不含数据本体**——`ks_clean.csv`（710 MB）、`ks_model_ready*.parquet`（231/144 MB）、原始 CSV（455 MB）都超出发送与 git 限制；包内给的是**说明 + 脚本 + 结果 + 图 + 模型**，数据可由 `09_分析脚本/_build_model_ready.py` 从原始 CSV 重建。

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

### 1.1 2026-11-12 新增：`12_文本层_TFIDF/`（47 个文件）

```
12_文本层_TFIDF/
├─ README.md            一分钟结论 / 目录 / 复跑命令 / 口径提醒 / 回滚方式
├─ 01_脚本/            10 个脚本：data_gate → run_baseline → run_matrix → evaluate → figures → text_model/resave_models
│                       （cfg / features / run_all 一键串跑；并入校验脚本 merge_to_package.py 留在工作区）
├─ 02_结果/             G0 数据门 21 项断言、G1 基线对账、12 配置指标、bootstrap 增量、
│                       决策影响、文本诊断、PDP 21 点、关键词面板 726 行、test_meta
├─ 03_图表/             fig1 增量森林图 / fig2 基线对比 / fig3 USD 对比 /
│                       fig4 决策影响 / fig5 简介词数 PDP / fig6 关键词 lift
├─ 04_报告/             results_text_layer.md（详版）/ report_section_中文.md（报告第 6 章）/
│                       slides_bullets.md（演示要点）
└─ 05_模型/             model_histgb_text（AUC 0.8847）/ model_lr_text_sparse（AUC 0.8554）+ schema
```

**这条线做了什么**：把 `name_clean + blurb_clean` 做 TF-IDF（3,000 词、1–2 gram、只在 train 拟合），
分别以「稀疏直接进 LR」「SVD128 后进 HistGB/RF」两种方式拼到 18 特征基座上，
严格沿用三段切分（train≤2020 / val 21–22 / test≥2023），12 次拟合一次跑完、无二次挑选。

**关键数字**（test n=100,469，creator 整簇 bootstrap 1,000 次）

| 配对 | ΔAUC | 95%CI | ΔBrier |
|---|---|---|---|
| HistGB + 文本 | **+0.240pp** | [+0.179, +0.299] | −0.155pp |
| LR + 文本 | **+0.898pp** | [+0.758, +1.028] | −0.511pp |
| RF + 文本 | **−1.401pp** | [−1.528, −1.263] | +1.420pp |

三条必须一起引用的限定：① HistGB 的 +0.24pp 在 creator 整簇与行级 i.i.d. 两种区间下**都显著**（i.i.d. [+0.185, +0.297]pp），不存在「区间方法敏感」问题（旧的 [−0.023,+0.518] / p=0.074 是 i.i.d. 权重错位的 bug，已于 2026-11-12 修复并重算）；
② RF 的负增益**方向**来自「128 维稠密特征稀释分裂候选」（v1 打乱对照支持该方向，RF 打乱后仍 −0.99pp、HistGB 仅 −0.10pp），但打乱对照**不能给出百分比**，不要写「78% 来自稀释」；
③ 本模块**不重调超参、不换切分**，基线已用同款超参重训并与已发布指标对账（最大偏差 7.7e-5）。

复跑：在项目工作区 `13_TFIDF文本层_v2\01_脚本` 下 `python -u run_all.py`（约 44 min，峰值内存 <8 GB；包内 `12_文本层_TFIDF/01_脚本` 是审计副本，路径按工作区布局解析，不建议原地跑）。
回滚：删除 `12_文本层_TFIDF/` 可回到并入后、改文档前；要连两份文档一起复原，从 `_backup_交付包_20261111_pre_textlayer/` 还原 `README_包内说明.md` 与 `交付说明_期末进度.md`。

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
