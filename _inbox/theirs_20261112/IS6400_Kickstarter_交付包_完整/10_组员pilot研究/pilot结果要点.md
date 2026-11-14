# 组员 Pilot Study —— 结果要点（离线副本）

> 来源：`https://github.com/Yuuhi115/6400-KickStarter-Project` → `kickstarter/pilot_study_results/Kickstarter_Pilot_Study_Report.md`（17,378 B）
> 本文件是**要点的忠实转录**（表格与数字逐项照抄）；原文与 9 张图、两个大数据文件需要能上网的机器 pull。

## 一、样本与时间切分

| 集合 | 年份条件 | 行数 | 成功率 |
| --- | --- | ---: | ---: |
| Pilot 总体 | 10% 分层抽样 | 60,464 | 49.21%（成功 29,754 / 失败 30,710） |
| Training | `launch_year ≤ 2019` | 41,124 | 43.18% |
| Validation | `2020 – 2022` | 9,208 | 60.36% |
| Test | `> 2022` | 10,132 | 63.55% |

清洗审计：初始 60,464 行，因缺失/异常 `launched_at`/`deadline` 排除 **0**，因周期不在 1–365 天排除 **0**，因 `goal` 缺失或为负排除 **0**。

## 二、测试集指标（阈值来自验证集 F1）

| Model | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | Brier | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Gradient Boosting** | 0.7829 | 0.7780 | 0.9213 | 0.8436 | **0.8571** | **0.9083** | **0.1453** | 0.425 |
| Random Forest | 0.7720 | 0.7592 | 0.9391 | 0.8396 | 0.8438 | 0.8987 | 0.1629 | 0.390 |
| Logistic Regression | 0.7493 | 0.7341 | 0.9495 | 0.8280 | 0.8307 | 0.8927 | 0.1612 | 0.310 |
| Baseline（Dummy prior） | 0.3645 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6355 | 0.2731 | 0.500 |

验证集最优 F1 阈值与对应正例率：LR 0.310 / 0.764；RF 0.390 / 0.741；GB 0.425 / 0.705。

## 三、分档与 Top-Decile Lift（最佳模型 GB）

| Group | Projects | Actual Success Rate | Mean Predicted |
| --- | ---: | ---: | ---: |
| Top 10% | 1,014 | **0.9862** | 0.9689 |
| 10–30% | 2,026 | 0.9210 | 0.8937 |
| 30–50% | 2,026 | 0.7813 | 0.7625 |
| 50–70% | 2,027 | 0.6088 | 0.6029 |
| Bottom 30% | 3,039 | 0.2488 | 0.2637 |

| Model | 总体成功率 | Top-Decile 成功率 | **Lift** | 捕获成功数占比 |
| --- | ---: | ---: | ---: | ---: |
| Gradient Boosting | 0.6355 | 0.9862 | **1.5518** | 0.1553 |
| Logistic Regression | 0.6355 | 0.9842 | 1.5487 | 0.1550 |
| Random Forest | 0.6355 | 0.9813 | 1.5441 | 0.1545 |

## 四、稳定性（3 个 expanding-window fold，阈值固定 0.5）

| Fold | Model | Train/Test 行 | F1 | ROC_AUC | PR_AUC |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | LR / RF / GB | 15,116 / 15,116 | 0.4867 / 0.4716 / 0.5081 | 0.7188 / 0.7480 / 0.7615 | 0.5763 / 0.6066 / 0.6308 |
| 2 | LR / RF / GB | 30,232 / 15,116 | 0.5721 / 0.6109 / 0.6870 | 0.7643 / 0.7894 / 0.7962 | 0.7668 / 0.7914 / 0.7997 |
| 3 | LR / RF / GB | 45,348 / 15,116 | 0.8128 / 0.8272 / 0.8401 | 0.8218 / 0.8388 / 0.8542 | 0.8836 / 0.8908 / 0.9015 |

## 五、特征重要性（Top 代表）

- **Logistic Regression**：`category_name_documentary`、`shorts`、`Tabletop Games`、`Hip-Hop`、`Camera Equipment`、`Apps` …
- **Random Forest**：`prep_days`、`goal`、`goal_log1p`、`campaign_duration_days`、`project_name_length`、`launch_year`、`blurb_length` …
- **Gradient Boosting（permutation）**：`goal`、`creator_previous_count`、`category_name`、`prep_days`、`campaign_duration_days`、`country`、`category_parent_name`、`project_name_word_count`

## 六、结论与限制（他们的原文立场）

- 决策：**情况 A —— 值得继续扩大数据**；依据是区分能力、相对基线的提升、Top-Decile 推荐价值、泄漏控制与时间 fold 稳定性。
- 限制：平台偏差、`staff_pick`/`spotlight` 的选择偏差与反向因果、时间漂移、缺失值、**发起人历史只基于 10% 样本**（应在第二阶段用完整数据重建）、特征重要性≠因果。

## 七、与我们当前口径的 4 处冲突（写报告时要说明）

1. **数据来源**：他们读原始 43 列 CSV 自建特征（其 state 计数 307,093 / 69,216 是原始口径）；我们已用 81 列清洗数据（307,091 / 69,210）。
2. **混币种**：`goal`/`log1p(goal)` 未换算；我们已改用 `m_log_goal_usd`。
3. **无 TF-IDF**（他们留到第二阶段）。
4. **Top-decile 98.6% 与 AUC 0.857 略显不协调** → 需核验 `prep_days` 与 `creator_previous_success_rate` 的实现（尤其 `prep_days` 是否隐含结果信息）。

## 八、仍需从仓库补齐的文件（本机网络受限，未能下载）

| 文件 | 大小 | 优先级 |
| --- | ---: | --- |
| `kickstarter_pilot_study.py` | 53 KB | 高（已有方法规格提取，原文仍建议 pull） |
| `kickstarter_pilot_10pct.csv` | 41 MB | 低（可从原始数据复现） |
| `test_predictions_best_model.csv` | 1.1 MB | 中（分档核验用） |
| 9 张 PNG（`plots/01–09`） | 合计约 0.9 MB | 中（报告图表） |
| `data_type_check.csv`、`missing_values.csv`、`yearly_distribution.csv`、`calibration.csv`、`rank_group_success_rates.csv`、`top_decile_lift.csv`、`stability_time_folds.csv`、`validation_thresholds.csv`、`feature_importance_*.csv` ×3 | 合计 < 60 KB | 低（数字已在本文件） |
| `output/` 4 份文档（condensed summary / variable exploration notebook / dataset shortlist / multi-dataset plan） | 约 63 KB | 中 |
