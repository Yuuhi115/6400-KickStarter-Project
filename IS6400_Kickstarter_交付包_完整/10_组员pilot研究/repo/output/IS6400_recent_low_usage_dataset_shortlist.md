# IS6400 近期、低使用量 Kaggle 数据集候选

> 数据核对日期：2026-09-22  
> 筛选目标：数据覆盖 2023–2026、Kaggle 使用量较低、至少能支持五名成员使用不同变量、可以同时完成本周变量关系作业和 Group Project。

## 1. 筛选标准

本次没有继续推荐 Hotel Booking Demand、Telco Churn、Online Retail II 等热门数据，改用以下标准：

1. 数据覆盖最近 1–3 年，或数据更新时间在 2025–2026。
2. Kaggle 下载量低于 1,000，Notebook 数少于 10。
3. 至少提供 15 个不重复的有效变量，或提供足够多张可直接连接的子表。
4. 能完成 shape、dtypes、missing values 和完整属性类型表。
5. 能延展成最终报告需要的描述分析、关系分析、回归、分类、分群或时间序列。
6. 明确标注数据是真实数据、公开数据整理，还是合成数据。

## 2. 五个候选总览

| 数据集 | 时间覆盖 | 数据规模 | Kaggle 下载量 | Notebook 数 | 数据性质 | 本周作业适配 | 项目深度 |
|---|---|---:|---:|---:|---|---|---|
| Kickstarter 2025 + 2026 update | 项目更新至 2026-04 | 694,062 x 43 | 758 | 1 | 真实项目抓取数据 | 高 | 很高 |
| E-Commerce Orders 2023–2025 | 2023–2025 | 100,000 x 25 | 9 | 0 | 模拟研究数据 | 很高 | 高 |
| Employee Attrition 2026 | 2025–2026 场景 | 45,000 x 29 | 146 | 3 | 合成数据 | 最高 | 高 |
| Central Asia Fintech Fraud | 2023–2024 | 150,000 + 12,000 + 2,500 | 164 | 2 | 合成数据 | 高 | 很高 |
| Real Airline Passenger Survey | 2021–2025 | 57,514 x 145 | 679 | 5 | 真实官方调查 | 中高 | 很高 |

综合排序：

1. **Kickstarter 2025**：真实、字段多、时间最新，项目上限最高。
2. **E-Commerce Orders 2023–2025**：使用量极低、零缺失，本周作业最顺。
3. **Employee Attrition 2026**：29 列零缺失，最符合本周结构化报告要求。
4. **Central Asia Fintech Fraud**：数据更新到 2024，适合金融风控项目。
5. **Real Airline Passenger Survey**：真实调查来源最可靠，但 145 列和结构性缺失较重。

## 3. Candidate 1: Kickstarter 2025

Kaggle：<https://www.kaggle.com/datasets/domingosun/kickstarter-dataset-629147-projects-may-2025>

### 3.1 数据情况

- 主文件：694,062 行、43 列。
- 数据更新至 2026 年 4 月，项目发布时间覆盖 2009–2026。
- 2024 年以后的项目有 80,446 个。
- 其中 `successful` 或 `failed` 的二分类样本有 69,865 个。
- 下载量 758，Notebook 数 1，重题风险显著低于常见电商数据。
- 数据类型包括 19 个字符串、15 个整数、7 个浮点数、2 个布尔值。
- `state` 可以作为成功、失败或取消的多分类目标；二分类时只保留 `successful` 和 `failed`。
- 主要缺失字段是 `profile_blurb`、`category_id`、`category_name`、`profile_name` 等。

### 3.2 本周作业的五人变量分配

| 成员 | 不重复变量 | 统计属性类型建议 | 关系分析 |
|---|---|---|---|
| A | `state`, `launched_at`, `deadline` | nominal；interval；interval | 项目状态、发布时间和项目周期 |
| B | `category_parent_name`, `country`, `location_type` | nominal；nominal；nominal | 类别、国家和项目所在地类型 |
| C | `goal`, `usd_pledged`, `backers_count` | ratio；ratio；ratio | 目标金额、实际筹资和支持人数 |
| D | `staff_pick`, `spotlight`, `created_at` | nominal；nominal；interval | 平台推荐、宣传标签和创建时间 |
| E | `name`, `blurb`, `creator_id` | nominal/text；nominal/text；nominal ID | 文本信息和创作者特征 |

如果严格做机器学习预测，`usd_pledged`、`backers_count`、`percent_funded` 和 `state_changed_at` 都不能用于预测项目成功，因为这些信息只在项目结束后才完整出现。

### 3.3 研究方向和可能产出

研究方向：**哪些项目、金额、类别、地区和文本特征能够预测众筹成功，并帮助创业者在发布前优化项目？**

研究方法：

- 描述统计：不同类别、国家、年份和推荐状态的成功率。
- 二分类：Logistic Regression、Decision Tree、Random Forest 和 Gradient Boosting 预测 `successful` 或 `failed`。
- 回归：对 `log(1 + usd_pledged)` 建模，研究目标金额、类别和文本长度与筹资规模的关系。
- 文本分析：对 `name` 和 `blurb` 做 TF-IDF、关键词和主题分析。
- 时间序列：观察项目创建、发布和成功率的年度/月度变化。
- 分群：按类别、目标金额、项目周期和创作者特征做聚类。

预计产出：

- 项目成功预测模型和 ROC-AUC、PR-AUC。
- 不同类别和国家的成功率排名。
- 目标金额区间与成功概率的曲线。
- 项目名称和高成功文案的常见关键词。
- 面向创业者的目标金额、发布时间和文案建议。
- 2024–2026 年众筹趋势图。

## 4. Candidate 2: E-Commerce Orders 2023–2025

Kaggle：<https://www.kaggle.com/datasets/minyooo/e-commerce-orders-consumer-behavior-and-efa>

### 4.1 数据情况

该数据集包含三份数据：

| 文件 | 行数 x 列数 | 缺失 | 时间范围 |
|---|---:|---:|---|
| `international_ecommerce_orders_2023_2025.csv` | 100,000 x 25 | 0 | 2023-01-01 至 2025-12-31 |
| `international_consumer_survey_intention.csv` | 55,000 x 23 | 5,422 | `survey_date` 全部为 2026-09-08 |
| `international_consumer_survey_with_efa_items.csv` | 55,000 x 44 | 5,422 | 同上 |

下载量只有 9，Notebook 数为 0，是本清单里撞题概率最低的数据。

订单文件的关键比例：

- `is_returned=True`：6.92%
- `repeat_purchase_within_90d=True`：2.77%

注意：问卷文件的 `purchase_intent_status_3m` 全部等于 `Has intention`，`survey_date` 也全部相同。它不适合作为真实购买意向的结论依据。建议把 `international_ecommerce_orders_2023_2025.csv` 作为主数据，问卷只用于补充变量类型和 EFA 练习。

### 4.2 本周作业的五人变量分配

| 成员 | 不重复变量 | 统计属性类型建议 | 关系分析 |
|---|---|---|---|
| A | `order_date`, `Year`, `Quarter` | interval；interval；ordinal | 订单时间、年份和季度趋势 |
| B | `region`, `product_category`, `campaign_type` | nominal | 地区、品类和营销活动 |
| C | `product_price`, `quantity`, `discount_percent` | ratio；ratio；ratio | 价格、数量与折扣 |
| D | `delivery_days`, `expected_delivery_days`, `delivery_delay_days` | ratio | 实际、承诺和延迟配送 |
| E | `customer_rating`, `perceived_value_score`, `is_returned` | ordinal/ratio；ordinal/ratio；nominal | 满意度、感知价值与退货 |

### 4.3 研究方向和可能产出

研究方向：**折扣、配送体验和消费者感知如何影响收入、满意度和退货？**

研究方法：

- 描述统计：收入、折扣、配送延迟和退货率的时间趋势。
- 回归：预测 `revenue` 或 `realized_revenue`。
- 分类：预测 `is_returned` 和 `repeat_purchase_within_90d`，报告 PR-AUC 和 Top-Decile Lift。
- 分组分析：比较不同地区、品类和活动类型的收入与退货率。
- 价格与折扣分析：计算折扣区间和平均订单价值、转化代理指标之间的关系。
- 分群：按购买频次、品类和客单价做客户分群。

预计产出：

- 月度收入、销量、折扣率和退货率趋势图。
- 退货预测模型和主要风险特征。
- 配送延迟与客户评分的分组比较。
- 不同折扣区间的收入贡献和可能边际变化。
- 不同地区和品类的运营优先级。
- 重复购买客户的画像和触达建议。

数据限制：该数据具有明显的模拟特征，所有结果应表述为“在模拟商业环境下”，不能直接宣称代表真实消费者。

## 5. Candidate 3: Employee Attrition in the AI & Hybrid-Work Era 2026

Kaggle：<https://www.kaggle.com/datasets/sergionefedov/employee-attrition-in-the-ai-and-hybrid-work-era>

### 5.1 数据情况

- 45,000 行、29 列。
- 无缺失值。
- `attrition=1` 的比例为 18.14%。
- `attrition_reason` 是多分类结果，包括 poor management、RTO inflexibility、burnout、better offer、career growth 和 layoff 等。
- 下载量 146，Notebook 数 3。
- 29 列中有 18 个整数、8 个字符串、3 个浮点数。
- 100% 为合成数据，但变量围绕 2025–2026 的 AI、远程/混合办公和返岗政策设计，时效性明显好于传统 IBM HR 数据。

### 5.2 本周作业的五人变量分配

| 成员 | 不重复变量 | 统计属性类型建议 | 关系分析 |
|---|---|---|---|
| A | `attrition`, `tenure_years`, `generation` | nominal；ratio；ordinal | 任职年限、代际与离职 |
| B | `age`, `department`, `job_level` | ratio；nominal；ordinal | 年龄、部门和职级 |
| C | `salary_usd`, `salary_vs_market_pct`, `promotions_last_2y` | ratio；interval；ratio | 薪资竞争力和晋升 |
| D | `work_arrangement`, `rto_mandate`, `days_in_office_required` | nominal；nominal；ratio | 办公模式与返岗要求 |
| E | `burnout_score`, `manager_relationship_score`, `engagement_score` | ordinal/interval | 倦怠、管理关系和敬业度 |

### 5.3 研究方向和可能产出

研究方向：**2026 年 AI 与混合办公环境下，哪些员工特征和组织政策最可能导致离职？**

研究方法：

- 二分类：预测 `attrition`。
- 多分类：预测 `attrition_reason`。
- Logistic Regression、Random Forest、Gradient Boosting。
- 使用 SHAP 或特征重要性解释管理质量、RTO 和倦怠的影响。
- 比较 hybrid、remote、onsite 三组的离职率和薪资差异。
- 对性别、年龄代际和职级做公平性检查。

预计产出：

- 离职风险模型和 ROC-AUC、PR-AUC。
- 高风险员工画像和管理者风险评分。
- RTO、混合办公和远程办公的离职率对比。
- 管理关系、倦怠和敬业度的局部解释图。
- 不同继任/离职原因的多分类混淆矩阵。
- 针对管理培训和复工政策的留存建议。

数据限制：这是合成数据，只能用于方法训练和情景模拟，不能声称代表真实公司的实际离职率。

## 6. Candidate 4: Central Asia Fintech Fraud

Kaggle：<https://www.kaggle.com/datasets/safar1/central-asia-fintech-fraud-dataset>

### 6.1 数据情况

| 表 | 行数 x 列数 | 缺失 | 主要用途 |
|---|---:|---:|---|
| `transactions.csv` | 150,000 x 13 | 0 | 交易和欺诈标签 |
| `users.csv` | 12,000 x 13 | 0 | 用户画像 |
| `merchants.csv` | 2,500 x 10 | 0 | 商户信息和风险评分 |

- 覆盖 2023-01 至 2024-12。
- 欺诈比例约 3.47%，适合研究不平衡分类。
- 下载量 164，Notebook 数 2。
- 数据为合成数据，但模拟了移动支付、跨城交易、登录次数和商户风险等真实风控结构。

### 6.2 本周作业的五人变量分配

将三张表按照 `user_id` 和 `merchant_id` 连接后，五人使用以下不重复变量：

| 成员 | 不重复变量 | 统计属性类型建议 | 关系分析 |
|---|---|---|---|
| A | `timestamp`, `hour_of_day`, `day_of_week` | interval；interval/cyclic；ordinal/cyclic | 欺诈的时间模式 |
| B | `amount_uzs`, `channel`, `session_duration_sec` | ratio；nominal；ratio | 金额、渠道和会话时长 |
| C | `login_attempts`, `is_cross_city_tx`, `is_fraud` | ratio；nominal；nominal | 登录异常、跨城交易和欺诈 |
| D | `gender`, `age`, `city` | nominal；ratio；nominal | 用户人口特征和地区 |
| E | `category`, `risk_score`, `is_online` | nominal；ratio；nominal | 商户类型、风险和线上属性 |

### 6.3 研究方向和可能产出

研究方向：**如何在极低欺诈率下识别高风险交易和商户，并降低误报？**

研究方法：

- Logistic Regression、Random Forest、LightGBM。
- 类别不平衡处理：class weights、欠采样、SMOTE 对比。
- PR-AUC、ROC-AUC、Precision-Recall 曲线和 Recall at fixed Precision。
- Isolation Forest、DBSCAN 或 Autoencoder 做异常检测。
- 用户和商户风险评分。
- 按小时、星期和月份分析交易与欺诈时间序列。
- 使用 SHAP 解释风险特征。

预计产出：

- 欺诈识别模型和业务阈值表。
- 高风险交易名单和商户风险排行榜。
- 夜间、跨城、登录异常和高金额的风险倍数。
- 不同 Precision 下的漏报和误报成本。
- 按渠道的实时监控规则。
- 面向风控部门的分层处置方案。

数据限制：欺诈信号是人工设计的，模型效果可能高于真实生产环境。报告中必须明确这是合成风控数据。

## 7. Candidate 5: Real Airline Passenger Satisfaction

Kaggle：<https://www.kaggle.com/datasets/rustam32/real-airline-passenger-satisfaction-dataset>

### 7.1 数据情况

- 处理后的平衡数据：57,514 行、145 列。
- 原始来源：巴西交通部机场乘客满意度调查。
- 原始季度文件覆盖 2021–2025，可以只使用 2024 和 2025 年八个季度文件。
- 下载量 679，Notebook 数 5，远低于常见航空满意度数据。
- `liked` 已经平衡为 50% 喜欢、50% 不喜欢，不能把 50% 解释成总体真实满意度。
- 145 列中有 76 列存在缺失，大量空白来自问卷的结构性跳题。
- 数据类型：62 个浮点、60 个整数、23 个字符串。

### 7.2 本周作业的五人变量分配

| 成员 | 不重复变量 | 统计属性类型建议 | 缺失提示 |
|---|---|---|---|
| A | `liked`, `process`, `flight_type` | nominal | 无缺失 |
| B | `traveling_alone`, `age_group`, `household_income` | nominal；ordinal；ordinal | 约 42% 缺失 |
| C | `trip_purpose`, `boarding_lounge_comfort`, `restroom_cleanliness` | nominal；ordinal；ordinal | 42%–87% 缺失 |
| D | `security_screening_process`, `checkin_process`, `airport_internet` | ordinal | 61%–86% 缺失 |
| E | `checkin_queue_wait_time`, `security_queue_wait_time`, `baggage_claim_time` | ordinal | 91%–97% 缺失 |

本数据最重要的作业发现之一就是“结构性缺失”：很多变量只对特定乘客流程或特定机场适用，因此每组都必须先筛选适用样本，再画图和解释相关性。

### 7.3 研究方向和可能产出

研究方向：**机场服务流程中，哪些环节最影响乘客整体满意度，且不同乘客群体关注点是否不同？**

研究方法：

- 描述统计和缺失值结构分析。
- Spearman、卡方检验和 Cramer's V。
- Ordinal Logistic Regression。
- Random Forest、Gradient Boosting 和分类模型。
- 按航班类型、出行目的、年龄组和收入组做亚组分析。
- 使用 SHAP 比较不同乘客群体关注的服务环节。

预计产出：

- 满意度分类模型和 ROC-AUC、PR-AUC。
- 不同机场流程与服务评分的相关矩阵。
- 哪些服务环节对整体满意度贡献最大。
- 商务旅客、休闲旅客和国际/国内旅客的关注点差异。
- 结构性缺失和样本适用范围的说明。
- 面向机场管理者的服务改进优先级。

数据限制：145 列会造成很长的属性类型表；建议在附录中按“主变量、服务评分、适用性指标”三组分块展示，不要只给一张无层次的大表。

## 8. 横向比较

| 比较维度 | 最佳候选 | 原因 |
|---|---|---|
| 真实数据优先 | Kickstarter、Airline | Kickstarter 为真实项目抓取，Airline 为官方调查 |
| 最新覆盖期 | Kickstarter、E-Commerce Orders、Attrition | 覆盖或更新至 2025–2026 |
| 最低撞题概率 | E-Commerce Orders、Attrition、Fraud | 下载量 9、146、164 |
| 本周作业最省力 | Attrition | 29 列、零缺失、变量类型清楚 |
| 零缺失且变量足够 | E-Commerce Orders、Attrition、Fraud | 三者的分析主表均无缺失 |
| 最终报告最丰富 | Kickstarter、E-Commerce Orders、Fraud | 分类、回归、文本、时间序列和分群都可做 |
| 金融/风控方向 | Central Asia Fintech Fraud | 不平衡分类、异常检测和成本分析完整 |
| 调查问卷方向 | Airline Passenger Survey | 真实官方调查，ordinal 变量丰富 |

## 9. 最终选择建议

### 如果团队最重视真实性和低撞题

选择 **Kickstarter 2025**。它包含 694,062 个真实项目、43 个字段，并且可以只抽取 2024–2026 的约 8 万个项目，既保持时效性，又降低计算量。

### 如果团队最重视本周作业顺利完成

选择 **Employee Attrition 2026**。45,000 行、29 列、零缺失，几乎可以直接开始做缺失值表、属性类型表和变量关系图。

### 如果团队想做大消费与营销分析

选择 **E-Commerce Orders 2023–2025**。100,000 行、25 列、零缺失、覆盖 2023–2025，而且 Kaggle 下载量只有 9，撞题风险最低。

### 如果团队想做金融风控

选择 **Central Asia Fintech Fraud**。三张表合计超过 16 万行，数据覆盖 2023–2024，欺诈比例约 3.47%，适合分类、异常检测和成本分析。

### 如果团队想用真实调查并接受较高清洗难度

选择 **Real Airline Passenger Satisfaction**。优先只使用 2024–2025 的季度文件，这样能满足时效性要求，但需要认真处理 145 列和结构性缺失。

## 10. 避免重题的操作建议

1. 在班级群或 Canvas 中先确认是否有人选择同一数据的同一子集。
2. 不要把题目写成宽泛的“某数据集分析”，应写成一个具体管理问题。
3. 同一个数据可以采用完全不同的研究问题来降低相似度。
4. 在提案中明确数据时间范围和最近两年的子样本。
5. 如果使用合成数据，在封面、数据说明和结论中明确披露。
6. 本周作业的五人变量组合必须提前登记，不能两人使用相同变量。
7. 最终报告中所有分类模型至少报告 ROC-AUC、PR-AUC 和 Top-Decile Lift 中的两项。
8. 如果使用 Kickstarter 或 E-Commerce Orders，先分离“建模特征”和“结果发生后才知道的字段”。
