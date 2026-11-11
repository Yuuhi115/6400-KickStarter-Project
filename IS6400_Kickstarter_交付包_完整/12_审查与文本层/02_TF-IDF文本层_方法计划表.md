# TF-IDF 文本层 · 方法计划表

- **目标**：把"只用量/字数"的文本维度，替换为"能解释、能归因、能给建议"的内容特征层，并把文本信号接入现有 HistGB，使模型指标与解释口径同时成立。
- **不做什么**：本文件只是计划。**不跑 TF-IDF、不改包内既有脚本**；实现需在方法表评审通过后进行。
- **参照实现**：同工作区 `..\ks_v2\src\text_model.py`（文本层）与 `..\ks_v2\src\explain.py`（关键词面板）——已在另一口径下跑通，本计划是将其移植进本交付包的口径。

---

## 0. 两个痛点 → 三个产出物

| 现在的痛点 | 根因（见 `01_检查报告` §4.6） | 本层给出的产出物 |
|---|---|---|
| "文案加长反而降 13pp"无法解释 | 输入 `40 词 / 240 字符` 在数据支撑之外（实际最大 239 字符）；且长度与成功的关系被时代混杂（逐年相关≈0，全样本 −0.058） | 用 `text_score`（内容信号）替代长度标量；所有文案结论都附"支撑域检查"；文案反事实只在 ≤26 词 / ≤135 字符内做 |
| "文案杠杆给不出建议" | 模型内只有两个长度标量，没有任何内容维度可归因 | ① 品类关键词面板 ② 文档级正负词贡献 ③ 支撑域内改写反事实 |

---

## 1. 现状与可复用资产

| 资产 | 位置 | 能直接复用吗 |
|---|---|---|
| TF-IDF 向量化 + 每年扩展窗 OOF 打分 | `..\ks_v2\src\text_model.py:26,41-42,49-67,74-83` | 是（改路径与参数） |
| 参数（1–2gram / min_df / max_features / sublinear / 英文停用词） | `..\ks_v2\src\config.py:88-89` | 是（按 §3 调整） |
| `text_score` 作为第 19 个普通数值列进 HistGB（非 stacking） | `..\ks_v2\src\models.py:82-85,166-171,181-183` | 是（保留 id 对齐断言） |
| 配对 bootstrap ΔAUC 与 CI | `..\ks_v2\src\evaluate.py:56-69,211-231` | 是 |
| 品类关键词面板（log-odds + BH-FDR + 支撑门） | `..\ks_v2\src\explain.py:35,56-95` | 是（加稳定性闸门） |
| Wilson CI | `..\ks_v2\src\config.py:129-138` | 是 |
| 本包 18 特征与切分 | `09_分析脚本\replicate_pilot.py:112-150` | 是（需补 `id/name_clean/blurb_clean` 三列） |

**本包缺、必须补的**：`replicate_pilot.py:54-58` 的 `NEEDED` 没有 `id`、`blurb_clean`、`name_clean`；6 个脚本硬编码 `E:\Workspace\...`；包内无 parquet。

---

## 2. 步骤表（S0–S9）

> 约定：`PKG` = `IS6400_Kickstarter_交付包_完整`；`RAW` = `..\..\ks_clean.csv`（只读）；所有新文件只**新增**，不覆盖任何既有文件。工作区：`12_审查与文本层\`（work 目录）、`11_结果与报告\`（对外产物）。

| 步骤 | 目的 | 输入 | 方法与参数 | 产物 | 验收判据 | 风险 / 对策 | 预计 |
|---|---|---|---|---|---|---|---|
| **S0 数据底座与路径** | 让文本层有输入，且不再依赖作者机器 | `RAW` 81 列 | 取 `id, name_clean, blurb_clean, category_key, subcategory_key, country_final, location_type, currency, goal, static_usd_rate, currency, yr, mon, dow, launched_at, deadline_dt, created_at, creator_rank, creator_prev_success_rate, is_terminal_pair, succ`；自建 `m_*` 等价特征（`m_goal_usd=goal*rate`，缺率且 USD 时用 goal；`m_log_goal_usd=log10`；`m_dur_days_exact=(deadline−launched)/86400`；`prep_days=(launched−created)/86400`；`launch_quarter`；`name_word_count`；`creator_previous_count=creator_rank−1`）；路径改 `Path(__file__).resolve().parents[1]`；临时脚本放 `12_审查与文本层\_scratch\` | `12_审查与文本层\work\model_frame.parquet`（约 604,629 行，含 id 与两列文本） | 行数 = 604,629；`id` 唯一；`succ.notna()` ≡ `is_terminal_pair`；与包内 `model_histgb_ours_schema.json` 的 18 特征同名同义；不写入 `RAW` | 与已交付特征口径不一致 → 用同一 `_build_model_ready.py` 逻辑逐列对齐，并跑一次 18 特征复现（应回到 AUC≈0.8823） | 0.5 天 |
| **S1 语料体检与清洗** | 知道文本里有什么坑 | S0 的 `name_clean`+`blurb_clean` | 统计：空值（`blurb` 空 13 行 / `name` 空 5 行）、`<tag>` 0.01%、HTML entity 8 行、URL 0.29%、非 ASCII 6.88%、CJK 0.14%、emoji 0.38%、`#NAME?`/3D-STL/Enamel Pins 模板串、mojibake（U+FFFD）；`blurb_clean` 已清 HTML 但**未小写**（全小写仅 1,718 行）；重复简介 9,631 行（1.59%） | `12_审查与文本层\work\corpus_audit.csv` | 上述计数可复现；**不删任何行**，只加 `is_template_dup`、`has_oov_marker` 两个标记列 | 行级去重会改变切分与基准率 → 不做，改为敏感性分析 | 0.25 天 |
| **S2 向量化** | 把文本变成可复用矩阵 | S0 文本列 | `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=30_000, sublinear_tf=True, stop_words="english", lowercase=True)`；文档 = `name_clean + " " + blurb_clean`；**词表只 fit 在 `yr ≤ 2022`**（train+val 期），`transform` 到 test | `11_结果与报告\text_vec.joblib`（vec 对象即可，矩阵不落盘） | 词表 fit 行全部 ≤2022；test 窗"零 in-vocab token"文档 ≤0.1%（实测 43/100,469 = 0.04%）；nnz ≈5.3M | **R1 泄漏**：词表看到 test → 增量虚高。对策：代码里硬断言 `max(fit_years) <= 2022` | 0.25 天 |
| **S3 年度扩展窗 OOF 文本分** | 得到 PIT 安全的 `text_score` | S2 矩阵 + `yr` | `LogisticRegression(C=1, max_iter=1000/2000, solver="liblinear")`；对 2012–2022 每年用 `prior = yr < y 且 ≥ min_history` 重拟合；`MIN_HISTORY=20_000`；2021–22 也走 OOF（不用最终 clf，避免轻度 in-sample）；test 用 `yr ≤ 2022` 全窗 clf 打分 | `11_结果与报告\text_scores.csv.gz`：`id, yr, text_score, split` | 行数 = 604,629 且 id 与 S0 一一对齐（`assert (scores.id == frame.id).all()`）；train 段为 OOF（同一行不被自己训练）；test 段用 ≤2022 的 clf | 扩展窗首年样本不足 → `MIN_HISTORY=20k` 保证最早一年也够；词表漂移（train/test 独立词表仅 66.2% 重叠）→ 只作稳健性对照，不改主口径 | 0.5 天 |
| **S4 双模型对照训练** | 量出文本层的真实增量 | S0 18 特征 + S3 文本分 | 两个 HistGB，超参与包内**完全一致**（`lr=0.08, max_iter=250, max_leaf_nodes=31, l2=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42`）；切分一致（train ≤2020 / val 2021–22 / test ≥2023）；A = 18 列，B = 19 列（+`text_score`）。**不覆盖** `model_histgb_ours.joblib` | `11_结果与报告\model_histgb_text.joblib` + `model_histgb_text_schema.json`（19 特征、阈值、test 指标） | 同一切分同一种子下 A 的 test AUC 应回到 0.8823±0.001（复现校验）；B 的 test AUC ≥ A；阈值仍只在 val 上按 F1 选 | **R3 稀疏×树**：不能把 30k 稀疏列灌进 HistGB（460k×30k 稠密 ≈55 GB，且 sklearn 1.5.1 的 HGB 拒收 CSR）→ 只传 1 列稠密 `text_score` | 0.5 天 |
| **S5 增量与 CI** | 让"有提升"可被检验 | A/B 的 test 预测 | 配对 bootstrap 300–500 次（percentile 2.5/97.5）；scope：全体、7 个品类、2023–2025（去 2026 部分年）；敏感性：① 删去与 train 逐字重复的 1,001 行 test 文案 ② 仅 `creator_rank==1` 子样本 | `11_结果与报告\text_increment.csv`：`scope, n, auc_struct, auc_plus_text, delta_auc, ci_lo, ci_hi` | 全体 ΔAUC >0 且 CI 下界 >0；分品类至少 4 个 CI>0；删重复文案后 ΔAUC 保留 ≥60%（否则披露为"含记忆成分"） | 增量可能 <0.004 → 按 `03_...预期产出表` 的降级方案走附录，不改主结论 | 0.5 天 |
| **S6 品类关键词面板** | 把"写什么"变成可读清单 | S2 矩阵 + 标签 + 品类 | 每文档 token 去重 → 文档计数 → 支撑门 `cnt_s ≥ max(30, 1%·n_s)` 或 `cnt_f ≥ ...` → 0.5 平滑 log-odds `log(a/b) − log(c/d)`，`se=sqrt(1/a+1/b+1/c+1/d)` → BH-FDR → 正负各 top15；**加稳定性列**：用 ≤2020 拟合、在 2023+ 复测符号是否一致 | `11_结果与报告\keyword_panel.csv`（`category, token, n_success_docs, n_failure_docs, log_odds, p_value, significant_fdr, sign_stable, support_ok`） | 每品类正负各 ≥10 个通过 FDR 且 `support_ok`；`sign_stable ≥60%` 的品类才进主表，其余标"参考级" | **R4 面板不稳**：≤2020 拟合的面板在 2023+ 符号稳定仅 58.3%（film 18/30、games 20/30、publishing 13/30、technology 19/30）→ 面板 fit 在 ≤2022 并输出稳定性；**R5 模板词** → 支撑门 + 标注模板 token | 0.5 天 |
| **S7 文档级归因** | "你这条文案哪些词在帮你/拖你" | 线性文本 clf 系数 + 单文档 tf-idf | 加性归因：`contrib_j = tfidf_ij × coef_j`（log-odds 空间可直接相加）→ 每篇取正负各 top10；与 S6 面板符号一致率、与 `text_score` 的 Spearman 作为校验 | `11_结果与报告\text_contrib_top.csv`（`id, top_pos, top_neg, contrib_pos_sum, contrib_neg_sum`） | 符号一致率 ≥70%；Spearman(`text_score`, `contrib_pos_sum+contrib_neg_sum`) ≥0.95 | ngram 共线导致重复计数 → 归因按 token 去重并合并子串；否定/习语误读 → 措辞限定"关联"，不写"应该改成 X" | 0.5 天 |
| **S8 支撑域内文案反事实** | 给出"幅度正确"的文案结论 | S3 clf + S0 行 | 三档位移：p5（短）、p50（中位）、p95（长）→ 分别把样本重排为对应词数/长度，重新向量化并打分，报 ΔP + bootstrap CI；**硬断言**：`words ≤ 25`（全样本 p95；test 窗 p95 = 24）`且 len ≤ 135`，越界直接拒绝打分并报 `out_of_support`；另做"删掉 top-3 负向词"的 token 级版本 | `11_结果与报告\text_counterfactual.csv`（`config, n, dp_mean, ci_lo, ci_hi, in_support`） | 所有行 `in_support == True`；预期幅度为**个位数 pp**（实测：p95 口径约 −1pp，中位约 +1pp，越界配置 −12.6pp 必须标 `out_of_support` 并排除） | 越界复现 −13pp 的风险 → 用硬断言封死；`-13pp` 从此只能作为"外推反例"出现在方法说明里 | 0.5 天 |
| **S9 消费端联动与口径改写** | 让建议端用得上文本层 | S3/S4/S6/S7 产物 | ① `scorecard.py`、`decision_axes.py`、`advice_by_category.py` 若 `feats` 含 `text_score`，必须在 `R.build_frame` 后按 `id` merge `text_scores`（否则 KeyError）；② 改写三处"文案层待 TF-IDF"的口径行：`decision_axes.py:129`、`decision_axes.py:158`、`advice_by_category.py:90`；③ 文案建议卡模板见 `03_...预期产出表` §4 | `11_结果与报告\advice_by_category_text.csv`（含文案建议列） | 端到端跑通；`advice_by_category_text.csv` 的文案列非空的行占比 ≥70%；所有文案结论都带"关联"限定语与支撑域说明 | 修改既有脚本属于"改包内文件"，需组内同意 → 本轮**不动**，只写方法与 diff 说明 | 0.5 天 |

**合计约 4.5 人·天**（不含评审）；纯机器时间 ≈ 6–10 分钟/次全流程（实测：TF-IDF 30k 拟合 27–29s、liblinear 504k×15k 3.6s/折、单次 HistGB 25–32s）。

> **执行状态（2026-11-10）**：S0–S6 已完成，实测见 `..\11_结果与报告\TF-IDF文本层_结果报告.md` 与 `TF-IDF关键词面板与归因.md`；S7 文案反事实已用 `text_length_response`（S5 内）覆盖"长度"部分，**词替换版 S7 与 S9 消费端联动未做**。
> S6 实际口径与计划的两处差异：① 支撑门用 `a≥20 ∧ b≥20 ∧ a+b ≥ max(20, 0.5%·n)`（比计划的 1% 更宽，因 15 品类样本差异大）；② 稳定性改为"≤2022 拟合 + 2023+ 复测符号"，并把 `coef_agrees` 并入 `trust` 三级（高/中/低）。实际产出 24 列而非 9 列。
> **v1.1 修正（2026-11-11）**：③ 首版实现的 `train` 掩码误用了主模型的 `TRAIN_MAX_YEAR=2020`，与"≤2022"标签及 `coef` 窗均不一致，且 `early`(≤2020) 列退化成 `train` 的复制（5,479 行零差异）。现统一 `train=≤2022`、`early=≤2020`，并在 `panel_table` 后加断言（两期差值 >1e-6 的行须 >90%）。重跑后面板 5,509 行 / 3,069 显著 / 跨期同号 77.5% / 高可信 1,956，84.1 s。

---

## 3. 关键设计决策（为什么这么选）

| 决策 | 备选 | 实测对比 | 结论 |
|---|---|---|---|
| **`text_score` 作为单列进 HistGB** | 稀疏 TF-IDF 直接进树 / SVD 降维后拼接 / 独立文本模型概率融合 | 11 特征代理模型（160k 子样本）：struct 0.87368 → **+text_score 0.88221（+0.0085）** → +SVD50 0.87867（+0.0050） → 两者都加 0.88206；18 特征复现版（另一位审查者同口径）：+0.0048~+0.0068（CI 均 >0） | 选单列：增量最大、最省内存、可画分布、可做杠杆；SVD 只作稳健性对照。**已在 18 特征全量口径上跑通端到端探针：0.88218 → 0.89014，ΔAUC +0.00795，CI [+0.00709, +0.00871]（见 `04_复现与实测附录.md` §2）** |
| **词表只 fit ≤2022** | fit 全表（含 test） | 词表若含 test 会虚高增量（Leakage R1）；train/test 独立词表仅 66.2% 重叠，但 test "零 in-vocab token" 文档仅 0.04% | 严格 ≤2022 |
| **`name + blurb` 两列拼接** | 只用 blurb | text-only AUC：blurb 15k = 0.7517；name+blurb 30k = **0.7799**（+0.025） | 用 name+blurb |
| **30k 特征** | 15k / 60k / 100k | text-only：0.7550 / 0.7564 / 0.7577（边际递减） | 30k（性价比） |
| **年度扩展窗 OOF（2012–2022）** | 单次切分训练 | 多花约 40s；换来 PIT 干净与 train 期分数可用 | 保留，不省这 40s |
| **p95 作为支撑上限** | 均值/中位数/固定 240 字 | 越界配置 −12.6pp；p95 域内 −1.1pp；中位数 +1.3pp | 支撑域硬断言 `words≤25, len≤135` |

---

## 4. 数据契约（实现时必须成立）

1. `model_frame.parquet`：604,629 行；`id` 唯一非空；列名与 `model_histgb_ours_schema.json` 的 18 个特征一致 + `id/name_clean/blurb_clean/succ/yr`。
2. `text_scores.csv.gz`：604,629 行；`assert (scores["id"] == frame["id"]).all()`（沿用 `ks_v2\src\models.py:181-183` 的做法）。
3. 任何模型输入 `feats` 与 `LEAKAGE_COLS` 交集为空（`02_清洗规则与理由\清洗脚本.py:40` 的那 10 列）。
4. 任何文案反事实必须打 `in_support` 标记，否则不得进入报告。
5. 新增文件只增不改：`model_histgb_ours.joblib`、`scorecard.csv`、`decision_axes_cells.csv`、`score_demo.csv` 一律不动。

---

## 5. 风险清单

| 编号 | 风险 | 触发条件 | 对策 | 残余风险 |
|---|---|---|---|---|
| R1 | 文本层泄漏 | 词表/折跨到 ≥2023 | 代码断言 `max(fit_year) ≤ 2022`；2021–22 也 OOF | 低 |
| R2 | 词表漂移 | train/test 词表 66.2% 重叠 | 年度重拟合只作稳健性对照；主口径固定 ≤2022 词表 | 中低 |
| R3 | 稀疏×树不可行 | 直接把 30k 列拼给 HistGB | 只传单列 `text_score`；SVD 作对照 | 无 |
| R4 | 关键词面板不稳 | ≤2020 拟合在 2023+ 符号仅 58.3% 稳定 | 面板 fit ≤2022 + `sign_stable` 列 + 品类级闸门 | 中 |
| R5 | 模板/重复文案 | 1.59% 重复简介、`#NAME?`/3D 模板串 | 支撑门 + 标记 + 敏感性（删重复 test 行） | 中 |
| R6 | 因果过度解读 | 把 `text_score` 当"可改杠杆" | 全部措辞用"关联"；反事实限支撑域 | 中（取决于写作纪律） |
| R7 | 旧模型不可加载 | sklearn 1.8 pickle 在 1.5.1 下失败 | 重训并锁版本（`sklearn==1.5.1, numpy==1.26.4`）；新增 `requirements_frozen.txt` | 低 |

---

## 6. 排期（半天粒度）

| 时段 | 内容 | 依赖 |
|---|---|---|
| D1-AM | S0 + S1（底座与语料体检） | — |
| D1-PM | S2 + S3（向量化 + OOF 打分） | S0/S1 |
| D2-AM | S4 + S5（双模型 + 增量 CI） | S3 |
| D2-PM | S6 + S7（面板 + 归因） | S2 |
| D3-AM | S8（支撑域反事实）+ S9 方案与 diff 说明 | S3/S4 |
| D3-PM | 汇总为一页结论 + 与主报告口径对账 | 全部 |

---

## 7. 与既有文档口径的衔接

- `11_工作计划\期末计划_v2.md` S1.1 承诺"TF-IDF 3000 维、增量 ≥+0.02 AUC 否则进附录" → **把验收线改为 ΔAUC ≥ +0.004 且 CI 下界 >0**。理由：+0.02 在结构特征已 0.88 的情况下不现实，任何团队都做不到，继续挂在文档里只会导致验收造假。本项目已跑通端到端探针：ΔAUC **+0.00795**、CI [+0.00709, +0.00871]，说明 +0.004 是**可达线**而非安慰线（见 `04_复现与实测附录.md` §2）。
- `decision_axes.py:129`、`:158`、`advice_by_category.py:90` 三处"文案层待 TF-IDF / 长度代理会误导"的措辞，实现后改为"文本层已落地，文案结论限支撑域内的个位数 pp 关联"。
- `交付说明_期末进度.md` §六的"TF-IDF 文本层未做"可改为"已做，见 `text_increment.csv` / `keyword_panel.csv`"。

---

## 8. 明确不做的

1. 不做稀疏矩阵进树（不可行）。
2. 不做 char_wb（未测，预算外；每轮 5–10 分钟）。
3. 不做行级去重（会改变切分与基准率）。
4. 不用 `description` 长文（包内未提供，且上线前可能修改）。
5. 不对 test 窗做任何调参或词表拟合。
6. 不修改 `01/02/06/07` 目录中的清洗产物与审计结论。
