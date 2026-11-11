# 11_结果与报告

本目录存放各阶段的结果与图表（`S0_复刻结果.md`、`s0_metrics.csv`、`config_grid_v2.csv`、`tradeoff_curve.png`、`keyword_panels.md`、`trend_forecast.md` …）。
数据类文件（csv/parquet/png）按 `.gitignore` 默认不入库，只提交 md 与小型结果表；大图与中间产物留在本地。

## 文本层（TF-IDF）新增 · 2026-11-09

| 文件 | 内容 |
|---|---|
| `TF-IDF文本层_结果报告.md` | **正式交付报告**：五配置对比、增量与 CI、分品类/分年份、长度响应、局限 |
| `text_layer_metrics.csv` | M0/M0b/M1/M2/M3 全部指标（AUC、CI、PR-AUC、Brier、F1、阈值、top10%、耗时） |
| `text_increment.csv` | 6 组对比 × 全体 + 10 个品类，含配对 bootstrap CI |
| `text_auc_by_year.csv` | 五配置 × 2023–2026 |
| `text_length_response.csv` | 简介截断到 6/10/14/18/22/25 词的响应曲线 |
| `text_scores.csv.gz` | 604,629 行 id + `text_score`（第 19 个特征） |
| `text_model_summary.json` | 语料审计、逐年 fold AUC、五配置指标 |
| `model_histgb_text.joblib` + `model_histgb_text_schema.json` | M2 完整 pipeline（19 特征，阈值 0.430），供评分卡/决策轴后续复用 |

流程与审查文档见 `..\12_审查与文本层\`；图见 `..\04_图表\figt1…figt7*.png`。

## 关键词面板与文档级归因（S6）新增 · 2026-11-10

| 文件 | 内容 |
|---|---|
| `TF-IDF关键词面板与归因.md` | **正式交付文档**：面板口径、15 品类精选词、三条失效模式、归因恒等式、示例、局限 |
| `keyword_panel.csv` | 5,509 行 × 24 列，品类×词面板（`log_odds`/`p_fdr`/`log_odds_test_2023`/`sign_stable`/`trust`/`is_top_pos`） |
| `text_contrib_top.csv` | 100,469 行 × 20 列，test 文档逐条 top5 正/负词贡献 + `Σcontrib+b=logit` 闭合字段 |
| `text_explain_summary.json` | 词表规模、截距、面板统计、每品类 top 词、恒等式误差（6.6e-14）、耗时 84.1 s |

脚本 `..\12_审查与文本层\build_text_explain.py`；面板图 `..\04_图表\figt8_keyword_panel.png`。
自校验：`nnz=8417547`、text-only test AUC `0.7799`（与 S3 一致）。

