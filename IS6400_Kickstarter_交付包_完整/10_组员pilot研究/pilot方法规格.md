# 组员 Pilot Study —— 方法规格（从 `kickstarter_pilot_study.py` 提取）

> 来源：`https://github.com/Yuuhi115/6400-KickStarter-Project` → `kickstarter/kickstarter_pilot_study.py`（53,145 B）
> 提取时间：2026-11-02（本机网络无法 `git clone`，仅通过 web 通道读取文本）
> 用途：让第二阶段能用**我们的全量 81 列数据**精确复刻这套设置，再与他们的 0.857 对账。

## 一、输入输出

| 项 | 值 |
| --- | --- |
| 输入 | **原始 43 列 CSV**（`Kickstarter Full Campaigns_694062_25_April_2026.csv`）——注意：**不是**我们清洗后的 81 列 |
| 输出 | `pilot_study_results/`（报告 + 15 个表 + 9 张图 + 10% 样本 CSV + 测试集预测） |
| 随机种子 | `RANDOM_STATE = 42` |
| 线程 | 强制 `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS = 1` |

## 二、样本与切分

1. `valid = state ∈ {successful, failed}`（604,629 行）；
2. `train_test_split(test_size=0.10, stratify=state, random_state=42)` → Pilot **60,464 行**（成功 29,754 / 失败 30,710）；
3. `choose_year_cutoffs()`：按年份累计份额选切点（训练 ≈70%、验证 ≈85%）→ 实际得到
   - train `launch_year ≤ 2019`：41,124 行（成功率 43.18%）
   - validation `2020–2022`：9,208 行（60.36%）
   - test `> 2022`：10,132 行（63.55%）
4. **阈值只在验证集上选**，原样套用到测试集。

## 三、泄漏控制

- 明确排除 6 个字段：`pledged`、`usd_pledged`、`backers_count`、`percent_funded`、`state_changed_at`、`profile_state`；
- `staff_pick`、`spotlight` 标为「Potential Leakage / Requires Further Verification」，**未入模**；
- 发起人历史只用「当前项目之前」的记录。

## 四、19 个模型特征

**数值（14）**：`goal`、`goal_log1p`、`campaign_duration_days`、`prep_days`、`launch_year`、`launch_month`、`launch_quarter`、`launch_weekday`、`project_name_length`、`project_name_word_count`、`blurb_length`、`blurb_word_count`、`creator_previous_count`、`creator_previous_success_rate`

**类别（5）**：`category_parent_name`、`category_name`、`country`、`location_type`、`currency`

派生定义：
- `campaign_duration_days = (deadline − launched_at)/86400`
- `prep_days = (launched_at − created_at)/86400` ← **需要注意的是这个特征**
- `goal_log1p = log1p(clip(goal, lower=0))`
- `creator_previous_count` = 按 `(launched_at, id)` 排序后同一 `creator_id` 的 `cumcount()`；`creator_previous_success_rate` = 之前成功数 / 之前项目数（无历史则 NaN）

## 五、预处理管线

| 类型 | 步骤 |
| --- | --- |
| 数值 | `SimpleImputer(median)` → `StandardScaler` |
| 类别 | `SimpleImputer(constant="Unknown")` → `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)` |

## 六、模型与超参数

| 模型 | 配置 |
| --- | --- |
| Baseline | `DummyClassifier(strategy="prior")` |
| Logistic Regression | `max_iter=3000, solver="liblinear", random_state=42` |
| Random Forest | `n_estimators=250, min_samples_leaf=5, n_jobs=4, random_state=42` |
| Gradient Boosting | `HistGradientBoostingClassifier(learning_rate=0.08, max_iter=250, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, validation_fraction=0.10, n_iter_no_change=20, random_state=42)` |

## 七、评估

- 指标：Accuracy / Precision / Recall / F1 / ROC_AUC / PR_AUC / Brier；
- F1 阈值搜索：`np.linspace(0.05, 0.95, 181)`，在验证集上取 F1 最大；
- 实际选定阈值：LR **0.310**、RF **0.390**、GB **0.425**（对应的验证集正例率 0.764 / 0.741 / 0.705）；
- Top-Decile Lift：按预测概率排序取前 10%，比较实际成功率与总体成功率；
- 分档：Top 10% / 10–30% / 30–50% / 50–70% / Bottom 30%；
- 校准：`calibration_curve(n_bins=10, strategy="quantile")` + Brier；
- 稳定性：`TimeSeriesSplit(n_splits=3)` 的 expanding window，阈值固定 0.5；
- 特征重要性：LR 用 `|coef|`，RF 用 `feature_importances_`，GB 用 `permutation_importance(scoring="roc_auc", n_repeats=5, 6000 行抽样)`。

## 八、决策规则（写进报告的那条"情况 A"）

同时满足下列条件才判 A：
- `auc_headroom_share = (AUC_best − AUC_base)/(1 − AUC_base) ≥ 0.50`
- `pr_headroom_share ≥ 0.50`
- `lift_gain = Lift − 1 ≥ 0.20`
- `stability_consistency = min(AUC_folds)/median(AUC_folds) ≥ 0.80`
- 三个正式模型的 AUC 与 PR-AUC 都超过 baseline

## 九、⚠️ 与我们现在口径的差异（复刻时必须替换）

| 他们的做法 | 我们应该的做法 | 理由 |
| --- | --- | --- |
| 用原始 43 列 CSV 自建特征 | 用 `08_建模就绪数据/ks_model_ready_v1.parquet`（81 列 + 11 个 `m_*`） | 我们已完成清洗/审计/修订 |
| `goal`、`log1p(goal)` 混币种 | `m_log_goal_usd`（`goal × static_usd_rate`，USD 缺汇率精确填充） | 非 USD 占 31.26%，>50k 档会被高估 6.6pp |
| `campaign_duration_days`（截断） | `m_dur_days_exact`（带小数） | 旧口径 41.79% 行有截断误差 |
| 文本长度用 `fillna("")`（缺失算 0） | `m_name_len` / `m_blurb_len` / `m_blurb_words`（缺失为 NA） | 缺失感知，不自造 0 |
| `creator_previous_*` 只在 10% 样本内算 | `creator_rank` + `creator_previous_success_rate`（全量、严格 point-in-time） | 他们自标"S 阶段应基于完整数据重建" |
| `category_name`、`country`、`currency` | `subcategory_key`（147）、`country_final`（25）、`currency`（15） | 我们已做大小写归并与口径统一 |
| 年份切点 2019 / 2022 | `m_split`（train ≤2022 / test ≥2023）；验证段可用 2020–2022 | 与交付数据预置一致 |
| 无 TF-IDF | 加 TF-IDF（`name_clean`+`blurb_clean`，3000 维、min_df=20） | 第二阶段明确要求 |

## 十、复刻检查清单

- [ ] 同一套 19 个特征的等价物（用 `m_*` 替换上面 8 处）
- [ ] 同样的 LR/RF/HistGB 超参（保证可比）
- [ ] 阈值只在验证集选，测试集不动
- [ ] 报告 AUC / PR-AUC / Brier / top-decile lift / 分档 / 3 折稳定性
- [ ] 与他们的 0.8571（10% 样本）对账：预期全量 + 正确履历后应**不低于**该值
