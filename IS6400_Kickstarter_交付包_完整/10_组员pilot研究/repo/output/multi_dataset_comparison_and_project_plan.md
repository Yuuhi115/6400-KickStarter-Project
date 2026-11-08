# IS6400 多数据集对比、家庭作业适配与项目方案

## 1. 结论先行

如果只选一个最稳妥的综合方案，仍然推荐：

**Hotel Booking Demand**

如果本周作业优先、希望最容易完成属性类型表和五人变量拆分，推荐：

**Airline Passenger Satisfaction**

如果团队有较强数据清洗能力，并希望最终报告做得最深入，推荐：

**DataCo SMART Supply Chain**

如果最终项目优先、愿意接受多表处理，推荐：

**Brazilian E-Commerce by Olist**

完整的综合排序如下：

| 排名 | 数据集 | 综合分值/100 | 最适合的场景 |
|---:|---|---:|---|
| 1 | Hotel Booking Demand | 94 | 本周作业与最终报告最平衡 |
| 2 | Airline Passenger Satisfaction | 92 | 本周作业最轻松，建模容易解释 |
| 3 | DataCo SMART Supply Chain | 87 | 项目深度最强，但清洗最重 |
| 4 | Bank Marketing | 85 | 零缺失、营销响应模型清楚 |
| 5 | Brazilian E-Commerce by Olist | 82 | 多表、真实业务、项目上限最高 |
| 6 | Online Shoppers Purchasing Intention | 76 | 小体量电商购买意向分析 |

分值不是客观真理，而是按照以下权重计算的：

- 本周作业适配度：30%
- 五人变量互斥能力：25%
- 最终项目深度：25%
- 数据质量与导入难度：10%
- 业务问题清晰度：10%

## 2. 统一比较表

| 数据集 | 行数 x 列数 | 缺失情况 | 目标变量及比例 | 本周作业适配 | 五人拆分 | 项目深度 | 主要风险 |
|---|---:|---|---|---|---|---|---|
| Hotel Booking Demand | 119,390 x 32 | 主要在 `company`、`agent`、`country`、`children` | `is_canceled`: 37.0% | 很高 | 很高 | 高 | 数据泄漏、`deposit_type` 异常 |
| Airline Passenger Satisfaction | 129,880 x 25 | 仅 `Arrival Delay in Minutes` 缺失 393 条 | `satisfaction`: 满意 43.4% | 最高 | 高 | 高 | 多项服务评分高度相关 |
| DataCo SMART Supply Chain | 180,519 x 53 | 总缺失 336,209 个单元格 | `Late_delivery_risk`: 54.8% | 中等 | 最高 | 最高 | PII、大量缺失、结果泄漏 |
| Bank Marketing | 41,188 x 21 | 0 | `y=yes`: 11.3% | 高 | 高 | 中等 | `duration` 泄漏、无客户 ID |
| Brazilian E-Commerce by Olist | 9 张表，订单主表 99,441 x 8 | 视表而定 | 评论、物流、销售均可作为目标 | 低 | 很高 | 最高 | 多表连接、连接后行数膨胀 |
| Online Shoppers Intention | 12,330 x 18 | 0，另有 125 条重复记录 | `Revenue=True`: 15.5% | 中高 | 中等 | 中低 | 变量仅够五人分配，后续深度有限 |

## 3. Candidate 1: Hotel Booking Demand

Kaggle：<https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand>

### 3.1 为什么适合

- 单一 CSV，119,390 行、32 列，导入和报告 shape、dtypes、missing values 很直接。
- 有 nominal、ordinal、interval、ratio 变量，也有 discrete 和 continuous 变量。
- 15 个以上变量供五人互斥选择，不依赖重复变量。
- 缺失值有层次：`company` 缺失 94.31%，`agent` 缺失 13.69%，`country` 缺失 0.41%，适合展示完整缺失值分析。
- 同一数据可以继续做取消预测、ADR 回归、季节性和收益管理。

### 3.2 五人变量分配

| 成员 | 不重复变量组合 | 关系分析 |
|---|---|---|
| A | `is_canceled`, `lead_time`, `deposit_type` | 提前期、押金类型与取消结果 |
| B | `market_segment`, `distribution_channel`, `customer_type` | 渠道与客户结构 |
| C | `adr`, `hotel`, `arrival_date_month` | 酒店类型、月份与房价 |
| D | `previous_cancellations`, `total_of_special_requests`, `booking_changes` | 历史行为与预订互动 |
| E | `country`, `required_car_parking_spaces`, `days_in_waiting_list` | 来源国、停车需求和等待时间 |

### 3.3 项目和预期产出

研究方向：高取消风险识别与收益管理。

方法：描述统计、卡方检验、Logistic Regression、Random Forest、ADR 多元回归、季节分解、时间序列和收益情景模拟。

预计产出：取消预测模型、ROC-AUC/PR-AUC/Lift、Top 10% 高风险订单、渠道和押金策略、月度需求与价格趋势、收入场景表。

主要风险：`reservation_status` 和 `reservation_status_date` 必须从预测变量中删除。

## 4. Candidate 2: Airline Passenger Satisfaction

Kaggle：<https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction>

### 4.1 为什么适合本周作业

- Kaggle 下载包含 `train.csv` 和 `test.csv`，主表分别为 103,904 x 25 和 25,976 x 25。
- 仅 `Arrival Delay in Minutes` 有 393 条缺失，缺失值分析非常容易。
- 有多个 0–5 分服务评分，是典型的 ordinal discrete 变量，属性类型表会非常完整。
- `Age`、`Flight Distance`、延误分钟数是 ratio 变量，适合做直方图、散点图和相关性分析。
- 五人可以各自分析不同类型的变量关系，而且每个成员都有清晰图形。

### 4.2 五人变量分配

先删除 `Unnamed: 0` 和 `id`，使用以下互不重复组合：

| 成员 | 不重复变量组合 | 关系分析 |
|---|---|---|
| A | `Gender`, `Customer Type`, `Age` | 客户类型、年龄与满意度 |
| B | `Type of Travel`, `Class`, `Flight Distance` | 出行目的、舱位与航程 |
| C | `Inflight wifi service`, `Ease of Online booking`, `Online boarding` | 数字化服务评分之间的关系 |
| D | `Seat comfort`, `Inflight entertainment`, `Leg room service` | 客舱硬件体验组合 |
| E | `Departure Delay in Minutes`, `Arrival Delay in Minutes`, `satisfaction` | 延误与满意度结果 |

### 4.3 项目和预期产出

研究方向：航空公司乘客满意度驱动因素和差异化服务改进。

方法：描述统计、Spearman 相关、卡方检验、Cramer's V、Ordinal Logistic Regression、Decision Tree、Random Forest 和客户分群。

预计产出：满意度分类模型、变量重要性和效应大小、不同客户类型和舱位的满意度差异、服务改进优先级矩阵、按乘客群体设计的服务建议。

主要风险：14 个服务评分彼此高度相关，相关性分析不能全部解释为独立影响；wifi 评分为 0 表示 Not Applicable，不能直接当作最差服务。

## 5. Candidate 3: DataCo SMART Supply Chain

Kaggle：<https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis>

### 5.1 为什么适合项目

- 主表 180,519 行、53 列，变量数量最多，五人拆分空间最大。
- 包含订单、客户、市场、地区、产品、折扣、利润、运费、实际/计划运输天数和交付日期。
- 可以同时做分类、回归、时间序列、地理分析和客户细分。
- `Late_delivery_risk` 为 54.8%，目标相对平衡，适合比较多个分类模型。

### 5.2 五人变量分配

使用 `DataCoSupplyChainDataset.csv`，并删除姓名、邮箱、密码、街道等 PII 字段。

| 成员 | 不重复变量组合 | 关系分析 |
|---|---|---|
| A | `Type`, `Shipping Mode`, `Days for shipment (scheduled)` | 付款方式、运输方式和计划时效 |
| B | `Customer Segment`, `Market`, `Order Region` | 客户细分、市场和地区的差异 |
| C | `Order Item Quantity`, `Order Item Discount Rate`, `Sales` | 数量、折扣和销售额 |
| D | `Benefit per order`, `Order Item Profit Ratio`, `Order Profit Per Order` | 订单利润结构 |
| E | `order date (DateOrders)`, `Category Name`, `Customer Country` | 时间、品类和国家结构 |

### 5.3 项目和预期产出

研究方向：降低延迟交付风险并提升订单利润。

方法：分类模型预测 `Late_delivery_risk`；线性回归或随机森林解释利润；月度时间序列预测订单量；按市场、客户细分、运输方式和地区做分群；地理热力图和情景模拟。

预计产出：延迟风险模型、高风险订单和路线名单、运输方式建议、利润率影响因素、利润下滑原因、月度需求预测和区域运营策略。

主要风险：`Product Description` 全部缺失，`Order Zipcode` 缺失 86.2%；`Delivery Status` 和实际运输天数会造成结果泄漏；若干利润字段高度相关，必须做相关性和多重共线性检查。

## 6. Candidate 4: Bank Marketing

Kaggle：<https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing>

### 6.1 为什么适合

- 41,188 行、21 列，零缺失，报告数据质量非常轻松。
- 有客户人口变量、上次联系信息、历史活动结果和社会经济指标。
- 可以自然做 nominal、ordinal、interval、ratio 变量关系分析。
- 最终项目可以明确转化为“谁更可能认购定期存款、应优先联系谁”。

### 6.2 五人变量分配

| 成员 | 不重复变量组合 | 关系分析 |
|---|---|---|
| A | `age`, `job`, `marital` | 年龄、职业和婚姻状态 |
| B | `education`, `default`, `housing` | 教育、违约和房贷 |
| C | `loan`, `contact`, `month` | 贷款、联系方式和月份 |
| D | `day_of_week`, `campaign`, `previous` | 联系频率和历史联系 |
| E | `poutcome`, `euribor3m`, `y` | 上次结果、利率和目标响应 |

### 6.3 项目和预期产出

研究方向：银行电话营销响应预测和客户优先排序。

方法：描述统计、卡方检验、点二列相关、Logistic Regression、Decision Tree、Random Forest、类别不平衡处理和 Lift Curve。

预计产出：响应预测模型、ROC-AUC/PR-AUC、Top-Decile Lift、最佳联系月份和渠道、不同客户群的响应率、营销 ROI 情景表。

主要风险：`duration` 只有在通话结束后才知道，真实预测场景必须删除；`pdays=999` 是“从未联系”的代码，不是连续数值；数据不包含年份和客户 ID，不能做真实客户留存分析。

## 7. Candidate 5: Brazilian E-Commerce by Olist

Kaggle：<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

### 7.1 为什么项目上限最高

- 99,441 笔订单、9 张表、112,650 条订单商品记录、99,224 条评论。
- 含订单状态、物流时间、客户地区、商品、卖家、支付和评论文本。
- 可以同时完成分类、回归、聚类、时间序列、文本分析和多表关系分析。
- 适合把“业务问题、数据、方法、结论”完整讲给管理层。

### 7.2 五人变量分配

每一组只能使用自己表中的变量，避免跨成员重复：

| 成员 | 使用表 | 不重复变量组合 |
|---|---|---|
| A | orders | `order_status`, `order_purchase_timestamp`, `order_delivered_customer_date` |
| B | customers | `customer_state`, `customer_city`, `customer_zip_code_prefix` |
| C | order_items | `price`, `freight_value`, `shipping_limit_date` |
| D | order_payments | `payment_type`, `payment_installments`, `payment_value` |
| E | order_reviews | `review_score`, `review_comment_message`, `review_creation_date` |

### 7.3 项目和预期产出

研究方向：配送履约、卖家表现和客户满意度。

方法：多表连接、描述统计、RFM/卖家聚类、差评分类、逻辑回归、随机森林、销售时间序列、TF-IDF 和主题分析。

预计产出：订单级主表、延迟与差评模型、高风险卖家列表、销售预测、差评主题和运营整改建议。

主要风险：本周作业需要分别报告多张表的 shape 和 dtypes；合并表可能因一个订单包含多个商品而增加行数。该数据最适合项目优先、团队愿意投入数据工程的组。

## 8. Candidate 6: Online Shoppers Purchasing Intention

Kaggle：<https://www.kaggle.com/datasets/henrysue/online-shoppers-intention>

### 8.1 为什么适合小体量作业

- 12,330 行、18 列，零缺失，125 条重复记录。
- 变量类型简单，适合快速完成作业。
- `Revenue=True` 为 15.5%，适合做购买意向分类。
- 可以分析页面访问、停留时间、跳出率、访问月份和流量类型与购买的关系。

### 8.2 五人变量分配

| 成员 | 不重复变量组合 | 关系分析 |
|---|---|---|
| A | `Administrative`, `Administrative_Duration`, `Informational` | 管理类页面行为 |
| B | `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` | 产品浏览行为 |
| C | `BounceRates`, `ExitRates`, `PageValues` | 会话质量和页面价值 |
| D | `SpecialDay`, `Month`, `VisitorType` | 时间和新老访客 |
| E | `OperatingSystems`, `TrafficType`, `Revenue` | 技术来源与购买结果 |

### 8.3 项目和预期产出

研究方向：电商会话购买意向预测和流量资源分配。

方法：描述统计、卡方检验、Spearman 相关、Logistic Regression、Decision Tree、Random Forest 和会话分群。

预计产出：购买预测模型、Top-Decile Lift、页面行为画像、不同月份和流量来源的转化率、预算优化建议。

主要风险：只有 18 列，五人刚好使用 15 列，后续扩展空间小；`BounceRates` 和 `ExitRates` 高度相关；没有客户 ID，无法分析复购和客户生命周期。

## 9. 怎么根据团队目标选

| 团队目标 | 最佳选择 | 原因 |
|---|---|---|
| 本周作业快速、稳、五人变量类型丰富 | Airline Passenger Satisfaction | 单表、少缺失、ordinal 变量多、图形直观 |
| 作业与最终项目兼顾 | Hotel Booking Demand | 单表好导入，后续能扩展为完整收益管理项目 |
| 最终报告最深入 | DataCo Supply Chain | 53 列、运营与财务变量丰富，适合高复杂度项目 |
| 想讲真实多表电商案例 | Brazilian E-Commerce by Olist | 物流、支付、卖家和评论文本完整 |
| 想使用零缺失、容易解释的营销数据 | Bank Marketing | 变量干净，响应分类业务清楚 |
| 想用最小数据量完成本周作业 | Online Shoppers Intention | 小体量、零缺失、列数刚好够五人 |

## 10. 最终建议

如果课程分数主要取决于最终报告深度，并且团队有 2–3 周时间处理数据，我会优先考虑 **DataCo Supply Chain**。

如果希望从本周作业开始就不出错，并且最终报告也能达到较好水平，我会选择 **Hotel Booking Demand**。

如果本周作业马上就要交，而团队还没有开始清洗数据，我会选择 **Airline Passenger Satisfaction**。它的缺失值少、变量类型清楚、每人都有可画的关系图，是最容易在短期完成高质量作业的候选。

无论最终选择哪一个数据集，本周作业都应遵守以下规则：

1. 先完成全列 shape、dtypes 和 missing values 表。
2. 再完成全列 nominal / ordinal / interval / ratio 与 discrete / continuous 表。
3. 五名成员各自最多选择 3 个变量，且变量不能重复。
4. 每人至少展示两种图，并明确样本量和单位。
5. 每人写 150–200 词时，需要说明发现、局限和它如何影响后续建模。
6. 提前确定并记录哪些变量会造成数据泄漏，不要让它们进入模型。
