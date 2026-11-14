# README · TF-IDF 文本层 v2（工作区）

本目录是文本层组员的完整工作区（`13_TFIDF文本层_v2/`）。成果已按「只新增」的方式并入
交付包的 `12_文本层_TFIDF/`；本目录保留可复跑的脚本、中间产物与更完整的论证细节。

## 一分钟看结论

| 问题 | 结论 |
|---|---|
| 文本层有用吗 | 有用但幅度很小：HistGB **+0.24pp**、LR **+0.90pp** AUC（creator 整簇 CI 均排除 0） |
| 依赖模型吗 | 强烈依赖：同一 SVD128 表示，RF **−1.40pp**（连续特征摊薄树分裂，非文本语义；只给方向不给百分比） |
| USD 口径呢 | 方向一致、幅度略大（HistGB +0.31 / 非 USD +0.17pp），不需单独模型 |
| 决策影响 | top-10% 名单换掉约 17%；**约 4.1% 测试行判定翻转**（上调 2.1% / 下调 2.1%）；单项目 p99 \|Δp\| 18.94pp |
| 与 S1 验收线（+0.02 AUC） | **本次 12 个配置均未达到**，按 S1 约定进附录；是否改口径由组会决定 |

## 目录

```
00_plan/        工作计划（v1.3，已批准）
01_脚本/     cfg / data_gate / features / run_baseline / run_matrix / evaluate
                / figures / run_all / text_model / resave_models / merge_to_package
02_结果/     门检查、指标、bootstrap、决策影响（含 decision_flips.csv）、诊断、preds/（每个配置的 test 预测）
03_图表/        fig1…fig6
04_报告/      results_text_layer.md（详版）/ report_section_中文.md（报告第 6 章）
                / slides_bullets.md（演示要点）
05_模型/      model_histgb_text / model_lr_text_sparse（+ schema + text_model.py 独立加载模块）
```

## 直接用模型（不依赖本工作区）

`05_模型/*.joblib` 是**纯 sklearn dict artifact**，配套的 `05_模型/text_model.py` 是自包含加载器：

```powershell
cd <交付包>\12_文本层_TFIDF\05_模型
python -c "import sys; sys.path.insert(0,'.'); from text_model import load_model; m=load_model('model_histgb_text.joblib'); print(m.threshold, m.algorithm)"
```

`m.predict_proba(df)` 吃 18 个入模特征 + `name_clean`/`blurb_clean`；`m.predict(df)` 按 artifact 内阈值判定。
（旧版是 `features.TextModel` 的 pickle，在包内会 `ModuleNotFoundError`；2026-11-12 已重存为 dict artifact，
重存前后预测 `max|Δp|` = 0（HistGB）/ 3e-8（LR），AUC 不变。）

## 复跑

> 下面这段在工作区 `13_TFIDF文本层_v2/` 下执行。交付包里的 `12_文本层_TFIDF/01_脚本/` 是**审计副本**
> （路径按工作区布局解析：`HERE/..` 当工作区、`../..` 当项目根），不建议在包内原地跑。

```powershell
cd 01_脚本
$env:PYTHONIOENCODING="utf-8"
python -u run_all.py                 # gate → baseline → matrix → evaluate → figures（约 44 min）
python -u run_all.py --only evaluate # 只重跑评估（约 2 min）
python -u run_all.py --from matrix   # 只重跑 12 次拟合（约 35 min）
python -u merge_to_package.py        # 并入交付包（只新增 + 两个文档的进度补充，带 sha256 审计）
```

环境：Python 3.12.7 / scikit-learn 1.5.1 / numpy 1.26.4 / pandas 2.2.2 / pyarrow 16.1.0。

复跑前提（两条，缺一直接 G0 失败）：

1. `IS6400_Kickstarter_交付包_完整/ks_clean.csv`（710 MB）需在位——它按「数据本体不进分发版」的约定
   已从分发版移出，备份在 `_backup_交付包_20261111_pre_textlayer/`；`data_gate.py` 会校验它的 sha256
   `aaf7a89e…b70b3f` 与 710,301,125 字节。
2. 组员建模目录名为 `建模交付`，本机已被改名为 `TF-IDF前的建模交付`；`cfg.py` 对两者都做回退解析，无需手动改。
组员发布件为 sklearn 1.8.0，本机无法加载，故 6 个基线全部重训；对账结果见
`02_结果/g1_baseline_check.csv`（最大偏差 7.7e-5）。

## 关键口径（引用数字前必看）

* 三段：train `yr≤2020` 443,693 / val 2021–22 60,467 / test `yr≥2023` 100,469；USD 327,642 / 33,542 / 54,435。
* parquet 内置 `m_split` 是**两段**口径（504,160 / 100,469），与本工作区无关；本工作区一律用 `split_ours`。
* `prepare_report.json` 的 1,487 / 5 / 47 是**全量 673,839 行**口径；可建模子集内是 850 / 5 / 13。
* 阈值一律 val 选（181 点 F1 网格），test 只看一次；`hgb_full` 本机阈值为 0.380（已发布 0.390，相邻网格点）。
* bootstrap：主口径 = creator 整簇 1,000 次；并行列级 i.i.d. 作对照，两者结论一致（见 4.3）。
  2026-11-12 晚修复了 i.i.d. 分支的权重对齐 bug（`evaluate.py` 原先把“原始行序”的抽样计数直接用于
  “按分数重排后”的数组），旧区间 [−0.023,+0.518]pp / p=0.074 已作废，见 `02_结果/deltas_bootstrap.csv`。

## 已并入交付包的内容

`IS6400_Kickstarter_交付包_完整/12_文本层_TFIDF/`：01_脚本 / 02_结果 / 03_图表 / 04_报告 / 05_模型（共 47 个文件，顶层 `README.md` 即本文件）。
并入前后清单：`02_结果/pkg_manifest_before.sha256.csv` 与 `pkg_manifest_after.sha256.csv`；
并入动作本身对既有 140 个文件零改动（`merge_to_package.py` 会再次断言）。

**2026-11-12 按用户要求补充进度**，改了包内两个既有文档（这是并入之后唯一的两次既有文件改动，
`merge_to_package.py` 的 `ALLOWED_EDITS` 只放行这两个，并把 sha256 变化记到 `02_结果/pkg_docs_updated.csv`）：

* `README_包内说明.md`：顶部新增 2026-11-12 说明；目录数/文件数改为 13 / 186；新增 §1.1 介绍 `12_文本层_TFIDF/`。
* `交付说明_期末进度.md`：新增 §六「文本层：TF-IDF 构建与入模对比」（原 §六–§十 顺延为 §七–§十一）；
  一页速览、尚未实现表、已知局限、版本记录、一句话总结同步更新。

## 回滚

```powershell
Remove-Item -Recurse -Force "<交付包>\12_文本层_TFIDF"
Copy-Item "<备份>\_backup_交付包_20261111_pre_textlayer\README_包内说明.md" "<交付包>\" -Force
Copy-Item "<备份>\_backup_交付包_20261111_pre_textlayer\交付说明_期末进度.md" "<交付包>\" -Force
```

只删目录只能回到「并入后、改文档前」；要完全回到 2026-11-11 状态，还需从 `_backup_交付包_20261111_pre_textlayer`
还原上示两个文档（备份包内其余 138 个文件与现包逐字节一致）。
