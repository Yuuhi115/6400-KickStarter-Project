# 五人分工方案（等工时 + 全员编码）

> 前提：全流程代码已跑通（`ks_v2/`，1,430 行，10 个模块），报告/PPT 未写。
> 分工原则：**每人接管 2 个可独立运行的模块 + 各做 1 项能独立跑出结果的小实验**，
> 保证 5 人都写代码、都能在助教抽查时 3 分钟内跑通并解释；写报告/做 PPT 的量按页数严格对齐。

---

## 一、工时分摊（每人 ≈10.4 小时）

| 角色 | 接管模块（行数） | 接管复现 | 扩展实验 | 报告 | PPT/讲稿 | 复核+彩排 | 合计 |
|---|---|---|---|---|---|---|---|
| A 数据与口径 | `data_v2.py` 93 + `eda_numbers.py` 130 | 1.2h | 2.2h | 3.5h | 1.5h | 2.0h | **10.4h** |
| B 特征与文本 | `features.py` 80 + `text_model.py` 95 | 1.0h | 2.3h | 3.5h | 1.5h | 2.0h | **10.3h** |
| C 模型 | `models.py` 291 + `run_all.py` 57 | 1.8h | 1.6h | 3.5h | 1.5h | 2.0h | **10.4h** |
| D 评估与时序 | `evaluate.py` 243 + `tseries.py` 98 | 1.8h | 1.6h | 3.5h | 1.5h | 2.0h | **10.4h** |
| E 双目标与解释 | `dual.py` 144 + `explain.py` 256 | 2.0h | 1.5h | 3.5h | 1.5h | 2.0h | **10.5h** |

代码行数不均是**有意平衡**的：C/D/E 模块本身更重，所以只给 1 个扩展实验；A/B 模块轻，所以给 2 个扩展实验 + 更重的数据/文本复核。

---

## 二、每人具体工作包

### A｜数据与口径（`data_v2.py` + `eda_numbers.py`）
- **接管复现**：`python ks_v2/run_all.py --stage data` → `--stage eda`；核对 408,347 / 353,913 / 54,434 / 0.4724 / 0.6425，与 `outputs/data/v2_stats.json` 一致。
- **扩展实验 1（编码）**：给 `data_v2.py` 加 `--currency {USD,ALL}` 开关，产出"删币种代价表"：样本量、达标率、被丢掉的市场（HK 0.759、JP 0.686 等）、以及 ALL 口径下 log_goal 不可比的具体表现。
- **扩展实验 2（编码）**：用现成的 `eda_simpson_category_goal.csv` 画 Simpson 分层图（每个品类一条金额档达标率曲线），写进报告 §1。
- **交付物**：`eda_headline.csv`、`eda_{year,category,goal_bin,duration,creator_rank,location_type}.csv`、`eda_money_by_goal.csv`、Simpson 图、删币种代价表。
- **报告**：§1 数据与口径（2 页）+ §2 验证设计（2 页）。
- **PPT**：第 1–2 页（封面+一句话结论、问题定义与不做的事）+ 第 3 页（口径与代价）。

### B｜特征与文本（`features.py` + `text_model.py`）
- **接管复现**：`--stage text`；核对 text-only AUC 0.7302、15,000 维词表、nnz 3,808,654、`text_scores.csv.gz` 374,519 行；核对"词表只在训练窗口拟合"。
- **扩展实验 1（编码）**：字符级 `char_wb` 3–5gram 对照（同一时间切分，只替换 vectorizer），报告 ΔAUC；若差距 >0.01 必须在报告里解释为何不换。
- **扩展实验 2（编码）**：关键词面板稳健性——train 期与 test 期分别算 log-odds，Top-30 词重合率 + 相关，输出 `keyword_stability.csv`。
- **交付物**：`text_increment.csv`、`keyword_panel.csv`、4 张关键词图（film&video / music / publishing / games）、`keyword_stability.csv`。
- **报告**：§3 特征与文本通道（2 页）+ §6 文本增量按品类（2 页）。
- **PPT**：第 4 页（验证设计图）+ 第 8 页（文本增量）。

### C｜模型与滚动验证（`models.py` + `run_all.py`）
- **接管复现**：`--stage models`（约 8 分钟），核对 `metrics_rolling_summary.csv` 9 行、`metrics_test.csv` 9 行、AUC 0.8848。
- **扩展实验 1（编码）**：HGB 超参敏感性（`max_depth` 4/6、`learning_rate` 0.04/0.06，各 2 折），输出 `hgb_param_sensitivity.csv`，证明选定参数不是"调出来的偶然"。
- **扩展实验 2（编码）**：给 `run_all.py` 加 `--stage-check`：缺前置产物时给清晰报错（助教复现不踩坑）。
- **交付物**：`metrics_rolling*.csv`、`metrics_test.csv`、`model_selection.json`、参数敏感性表。
- **报告**：§4 模型比较与选择（3 页）+ §11 复现说明（1 页）。
- **PPT**：第 5 页（模型比较表）+ 第 6 页（主结果 AUC/CI）。

### D｜评估、校准与时序（`evaluate.py` + `tseries.py`）
- **接管复现**：`--stage eval` → `--stage ts`；核对 ECE 0.0032、AUC CI [0.8819, 0.8876]、分年表、HW MAE 0.0193。
- **扩展实验 1（编码）**：分组校准——按 4 个金额档各画一条可靠性曲线，诊断"高金额档概率是否更不准"，输出 `calibration_by_goal.csv` + 图。
- **扩展实验 2（编码）**：时序加 ETS / Theta 对照（statsmodels 已装），与 HW、季节朴素同表比较。
- **交付物**：`calibration_summary.csv`、`prior_shift.csv`、`auc_by_year.csv`、`auc_by_subgroup.csv`、`decile_lift.csv`、`headline_ci.csv`、`auc_deltas.csv`、`timeseries_metrics.csv`、HW 图、分组校准图。
- **报告**：§5 校准与稳定性（3 页）+ §8 平台面时序（1 页）。
- **PPT**：第 7 页（校准 + lift）+ 第 9 页（时序）。

### E｜双目标、可解释与落地件（`dual.py` + `explain.py`）
- **接管复现**：`--stage dual` → `--stage explain`；核对 pinball 0.106/0.332/0.186、覆盖率 0.7412、204 格评分表、OR、置换重要性、3 张示例卡。
- **扩展实验 1（编码）**：覆盖率诊断——分品类算 P10–P90 覆盖率，说明 0.7412 的偏差集中在哪，输出 `quantile_coverage_by_cat.csv`。
- **扩展实验 2（编码）**：示例卡加"时长敏感性"曲线（现在只有目标金额曲线），输出 `fig_example_cards_duration.png`。
- **交付物**：`tradeoff_by_goal.csv`、`quantile_{pinball,coverage}.csv`、`scorecard.csv`、`lr_odds_ratios.csv`、`permutation_importance.csv`、`example_cards.csv`、权衡图、重要性图、示例卡 2 图。
- **报告**：§7 双目标权衡（2 页）+ §9 可解释与落地件（2 页）。
- **PPT**：第 10 页（落地三件 + 边界）。

---

## 三、报告 24 页与 PPT 10 页的分配

| 报告章节 | 页 | 负责人 | | PPT 页 | 负责人 |
|---|---|---|---|---|---|
| §0 摘要（全员要点合并）| 1 | A 统稿 | | 1 封面+一句话 | A |
| §1 数据与口径 | 2 | A | | 2 问题定义/不做的事 | A |
| §2 验证设计 | 2 | A | | 3 口径与代价 | A |
| §3 特征与文本通道 | 2 | B | | 4 验证设计图 | B |
| §4 模型比较与选择 | 3 | C | | 5 模型比较表 | C |
| §5 校准与稳定性 | 3 | D | | 6 主结果 AUC/CI | C |
| §6 文本增量 | 2 | B | | 7 校准 + lift | D |
| §7 双目标权衡 | 2 | E | | 8 文本增量 | B |
| §8 平台面时序 | 1 | D | | 9 时序 | D |
| §9 可解释与落地件 | 2 | E | | 10 落地三件 + 边界 | E |
| §10 边界与结论 | 2 | 全员各写 0.5 页，C 合并 | | | |
| §11 复现说明 | 1 | C | | | |

每人正文 **4 页 + 0.5 页边界段 + 2 页 PPT + 90–100 秒讲稿 ≈ 1.5 分钟**（8 分钟 / 5 人）。

---

## 四、排期（10 天，小任务可压缩）

| 天 | 事项 |
|---|---|
| D1 | 30 分钟启动会（过 RESULTS_SUMMARY、认领模块、建分支）；各自跑通自己的 stage（每人 1–1.5h）|
| D2–D3 | 各写扩展实验代码 + 出结果（每人 1.5–2.5h）；结果先写进 `report/§x.md` 骨架 |
| D4–D5 | 报告各节写完（每人 3.5h）；图表编号统一 |
| D6 | **数字审计日**：B 主持，把正文所有数字与 `tables/*.csv` 逐一核对；交叉复核（A↔D、B↔E、C↔A 各 1h）|
| D7 | 合并统稿（A 主持）、排版、附录、参考文献 |
| D8 | PPT 制作 + 讲稿（每人 1.5h）；D 统一模板 |
| D9 | 彩排 ×2（E 计时）；Q&A 抽题 |
| D10 | 定稿提交，`run_all.py --stage all` 最后跑一次存档 |

---

## 五、协作规则（避免冲突）

1. **文件所有权**：每人只改自己那 2 个 `.py` 模块 + 自己的 `report/§x.md` + 自己的 slide；`config.py`、`features.py` 的公共接口改动必须群里先说，由 owner 改。
2. **git 分支**：`pkg-a-data`、`pkg-b-text`、`pkg-c-models`、`pkg-d-eval`、`pkg-e-explain`；合并前必须 `run_all.py --stage all` 通过。
3. **表格/图只读**：谁的模块生成的表就谁负责重新生成，其他人只引用不覆盖。
4. **数字冻结**：D6 审计后所有正文数字冻结，之后只允许"修正不改口径"。
5. **代码贡献可追溯**：每人把 `git log --author` 记录和"负责模块 + 新增实验 + 行数"写进报告附录的"代码分工表"。

---

## 六、验收标准（每人自查 + 助教抽查）

- [ ] 删除自己模块的输出后，`python ks_v2/run_all.py --stage <mine>` 能重新生成、且数字与 `RESULTS_SUMMARY.md` 一致。
- [ ] 能口头解释自己模块：输入什么、拟合什么、为什么这样切分、关键数字含义。
- [ ] 扩展实验有独立输出文件（CSV/PNG），能在报告里找到对应段落。
- [ ] 报告 4 页 + PPT 2 页 + 90 秒讲稿按时交付。
- [ ] 参与 D6 数字审计与交叉复核（记录在 `report/audit_log.md`）。
