# IS6400 Kickstarter · 建模交付说明（2026-11-09）

> 本包是**建模层**的完整交付：训练脚本、数据切分物证、六个已落盘模型、模型结果与报告。
> 面向：接手做文本特征的同学、写报告/做演示的组员、以及任何要复核数字的人。

---

## 一、30 秒上手（最小可用代码）

```python
import joblib, pandas as pd, numpy as np

obj  = joblib.load("03_模型文件/model_hgb_full.joblib")   # 主模型（全样本 HistGB）
pipe, feats = obj["pipeline"], obj["features"]
print(obj["threshold"], obj["metrics_own_test"])          # 0.39 / AUC 0.8823 ...

# 一个"即将上线"的项目：只需 18 个特征的 dict
cfg = {
    "m_log_goal_usd": np.log10(10000), "m_dur_days_exact": 30.0, "prep_days": 14.0,
    "yr": 2026, "mon": 6, "launch_quarter": 2, "dow": 2,
    "m_name_len": 35.0, "m_blurb_len": 120.0, "m_blurb_words": 20.0, "name_word_count": 6.0,
    "creator_previous_count": 0.0, "creator_prev_success_rate": np.nan,
    "category_parent_name": "games", "subcategory_key": "tabletop games",
    "country_final": "US", "location_type": "1", "currency": "USD",
}
print(pipe.predict_proba(pd.DataFrame([cfg])[feats])[:, 1][0])   # 达标概率
```

⚠️ 环境必须 `scikit-learn==1.8.0`（见 `01_训练脚本/requirements.txt` 或包根 `requirements.txt`）；跨版本加载会报
`ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module`。

---

## 二、数据切分（唯一规则，见 `02_数据切分/splits_manifest.json`）

数据源：`08_建模就绪数据/ks_model_ready_v1.parquet`（92 列，本包不含数据本体）。
样本口径：`m_is_modelable == True` 且 `succ` 非空（successful / failed 二分类）。

**ours 协议（本项目主口径）**

| 集合 | 规则 | 全样本 | 仅 USD |
| --- | --- | ---: | ---: |
| train | `yr ≤ 2020` | **443,693** | **327,642** |
| val | `2021 ≤ yr ≤ 2022` | **60,467** | **33,542** |
| test | `yr ≥ 2023` | **100,469** | **54,435** |
| 合计 | | 604,629 | 415,619 |

- **val 只用来选判定阈值（0.39）**；模型族与超参是预先设定的，未在 val 上搜索。
- 币种口径：**全部保留**，金额按 `static_usd_rate` 换算成美元；非 USD 可建模 **189,010 行（31.26%）**。
- **⚠️ 陷阱**：parquet 内置列 `m_split` 是**两段**口径（train `yr≤2022` = 504,160 / test `yr≥2023` = 100,469），
  与上面三段口径不同。2021–22 那 60,467 行在 `m_split` 里属 train、在本协议里属 val。**引用必须说明是哪一套。**
- 行级物证：`split_assignment.parquet`（604,629 行，字段 `id / yr / is_usd / split_ours / m_split`），可直接 join 复核任何一行归属。

---

## 三、六个已落盘模型（`03_模型文件/`）

三种算法 × 两种训练口径，配置与 `S1_模型对比.md` 完全一致；均为 `joblib(..., compress=3)`。

| 文件 | 算法 | 训练口径 | 体积 | 全样本测试行 AUC | 训练耗时 |
| --- | --- | --- | ---: | ---: | ---: |
| `model_hgb_full.joblib` | **HistGB（主模型）** | 全样本 | 0.42 MB | **0.8823** | 288 s |
| `model_hgb_usd.joblib` | HistGB | 仅 USD | 0.42 MB | 0.8769 | 178 s |
| `model_rf_full.joblib` | RF(150) | 全样本 | **92.7 MB** | 0.8749 | 618 s |
| `model_rf_usd.joblib` | RF(150) | 仅 USD | **72.9 MB** | 0.8699 | 327 s |
| `model_lr_full.joblib` | LR | 全样本 | 7 KB | 0.8464 | 39 s |
| `model_lr_usd.joblib` | LR | 仅 USD | 6 KB | 0.8359 | 27 s |

- 每个模型配 `*_schema.json`：`features / numeric / categorical / threshold / metrics_own_test / n_train`。
- `models_manifest.csv`：文件、体积、训练耗时一览。
- ⚠️ **两个 RF 模型（165 MB）不在主 zip 里**，单独打了 `IS6400_模型文件_RF_20261109.zip`；
  如不想传大文件，让同事用 `s5_save_six_models.py` 重训即可（RF 全样本约 10 分钟，单线程）。

---

## 四、结果文件索引（`04_模型结果/`）

| 文件 | 内容 | 关键数字 |
| --- | --- | --- |
| `S1_模型对比.md` + `s0_metrics_*.csv` | 三算法同口径对比（主结果） | LR 0.8464 / RF 0.8749 / HistGB **0.8823** |
| `S1_解释性.md` + `s1_*.csv` | 置换重要性、LR 优势比、7 条 PDP | 履历 52.4%、目标 19.7%、文本 <1% |
| `cells.parquet` | **唯一事实源**：450 格，四口径并排 | `p_profile / p_cell_mean / rate / n` |
| `cells_oos.parquet` | 样本外格子（仅测试期） | 可用格 **85/432**（in-sample 是 250/450） |
| `S3_样本外与价值检验.md` | 价值维度 bootstrap（creator 聚类）+ 取消敏感性 | 推荐级 43/45；49.21% → 44.16% |
| `S4_USD-only对照.md` | USD-only 平行模型 + 干净消融 | 同测试集口径剔除代价 **−0.54pp** |
| `S5_三模型x两口径.md` | **六模型 12 格矩阵** | full−usd：LR +1.05 / RF +0.50 / HistGB +0.54pp |
| `S2_建议引擎.md` + `decision_axes_v2.csv` | 受约束建议引擎（1–60 天 / 目标格 n≥300 / 允许保持原配置） | 负 dp_max 0、61d+ 建议 0 |

---

## 五、复现顺序（单线程，总约 40–50 分钟）

```powershell
cd <包目录>\01_训练脚本
python s6_splits_delivery.py      # 1) 切分物证（~1 min）
python s5_save_six_models.py      # 2) 六模型 + 矩阵（~28 min，RF 是瓶颈）
python s2_decision_engine.py      # 3) cells + 受约束建议（~6 min）
python s3_oos_value.py            # 4) 样本外 + 价值检验（~4 min）
python s4_usd_parallel.py         # 5) USD-only 平行（~6 min）
python s4b_ablation_same_rows.py  # 6) 干净消融（~2 min）
python _interpretability.py       # 7) 解释性三件套（~6 min）
python audit_before_delivery.py   # 8) 交付前门禁，必须 4 PASS
```

> 沙箱/受限环境下 `n_jobs>1` 会报 `PermissionError [WinError 5]`，所有模型均强制单线程。

---

## 六、三条硬约束（引用数字前必须读）

1. **四个"概率"不可混用**：个体 `P` / `p_cell_mean`（格均值）/ `p_profile`（合成画像）/ `rate`（历史观测）。
   同一格可差 20pp（案例：76.6% / 58.8% / 74.4% / 58.1%）。定义见交付说明 §0 口径契约。
2. **ΔP 是模型情景分差，不是因果收益、也不是实测效果**；实测只能来自测试期（`yr≥2023`）。
3. **`p_weighted_q50 = p × Q50` 不是期望值**；`exposure = (1−p) × Q50` 不是期望损失（我们没有成本/投入字段）。

---

## 七、给接手的同学：加文本层时必须遵守的对照协议

1. **必须在同一批测试行上比**：全样本测试行 100,469（`yr≥2023`）；如果做 USD-only 版本，则用 54,435。
   不允许"我 0.8848 vs 你 0.8823"这种跨样本比较。
2. **必须报 ΔAUC + 95% 置信区间**（bootstrap，按 `creator_id` 整簇重抽），并同时给出 AP / Brier。
3. **六个模型都要跑一遍对照**（LR/RF/HistGB × full/usd），否则无法判断文本增益是否依赖模型。
4. **文本特征必须只用训练期拟合词表**，不得用测试期语料拟合（否则是泄漏）。
5. 若增益 <0.5pp，如实写——现有证据（组员 `text_increment.csv`）显示整体 +0.57pp，
   且在 games / technology / comics 三个品类**不显著**。

---

## 八、已知限制

1. 数据本体不在包内（231 MB parquet / 710 MB CSV），需按 `_build_model_ready.py` 从原始 CSV 重建；
2. RF 两个模型不在主 zip；
3. 测试集已被多次查看（选模型、消融），严格说只能称"开发期时间外评估"；
4. 履历特征按 `launch_dt` 截断而非 `created_at`，影响 1.65% 行（9,984 行）；
5. 汇率用项目级静态汇率，属近似；已用 USD-only 子样本做稳健性（AUC −0.54pp）。
