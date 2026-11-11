# S5 六模型：三算法 × 两口径（全部已落盘）

> 三种算法（LR / RF150 / HistGB，配置同 S1）× 两种训练范围（full 全样本 / usd 仅 USD）。
> **六个模型均已保存**到 `11_结果与报告/models/`，命名 `model_{algo}_{scope}.joblib` + 同名 schema。
> 评估只用阈值无关指标（AUC / AP / Brier），两批测试行完全相同。

## 一、12 个评估

| 算法 | 训练范围 | 测试行 | n | AUC | AP | Brier |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LR | full | all_test | 100,469 | 0.8464 | 0.9056 | 0.1507 |
| LR | full | usd_test | 54,435 | 0.8368 | 0.9000 | 0.1556 |
| RF(150) | full | all_test | 100,469 | 0.8749 | 0.9229 | 0.1403 |
| RF(150) | full | usd_test | 54,435 | 0.8698 | 0.9204 | 0.1431 |
| HistGB | full | all_test | 100,469 | 0.8823 | 0.9295 | 0.1316 |
| HistGB | full | usd_test | 54,435 | 0.8777 | 0.9268 | 0.1344 |
| LR | usd | all_test | 100,469 | 0.8359 | 0.8997 | 0.1555 |
| LR | usd | usd_test | 54,435 | 0.8359 | 0.8995 | 0.1574 |
| RF(150) | usd | all_test | 100,469 | 0.8699 | 0.9202 | 0.1433 |
| RF(150) | usd | usd_test | 54,435 | 0.8685 | 0.9193 | 0.1446 |
| HistGB | usd | all_test | 100,469 | 0.8769 | 0.9265 | 0.1348 |
| HistGB | usd | usd_test | 54,435 | 0.8759 | 0.9257 | 0.1353 |

## 二、配对差（同一算法、同一测试行：full − usd）

| 算法 | 测试行 | ΔAUC | ΔAP | ΔBrier（usd−full） |
| --- | --- | ---: | ---: | ---: |
| HistGB | all_test | +0.54pp | +0.29pp | +0.33pp |
| HistGB | usd_test | +0.18pp | +0.11pp | +0.09pp |
| LR | all_test | +1.05pp | +0.59pp | +0.48pp |
| LR | usd_test | +0.08pp | +0.05pp | +0.18pp |
| RF(150) | all_test | +0.50pp | +0.27pp | +0.31pp |
| RF(150) | usd_test | +0.13pp | +0.11pp | +0.15pp |

## 三、结论

- **方向一致性**：全样本测试行上，**三种算法均为 full > usd**。幅度： LR +1.05pp、RF(150) +0.50pp、HistGB +0.54pp。
- **迁移性**：仅 USD 测试行上，LR +0.08pp、RF(150) +0.13pp、HistGB +0.18pp（均为正则为正迁移）。
- **结论不依赖模型选择**：剔除非 USD 在三种算法上都是小幅净损失；真正代价仍是覆盖范围（国家 25→3、可用格 250→219、样本外 85→41）。

## 四、已落盘模型清单

| 文件 | 算法 | 口径 | 训练耗时 | 体积 |
| --- | --- | --- | ---: | ---: |
| `models/model_lr_full.joblib` | Logistic Regression | full | 39s | 0.0 MB |
| `models/model_rf_full.joblib` | Random Forest | full | 618s | 92.7 MB |
| `models/model_hgb_full.joblib` | Gradient Boosting | full | 288s | 0.4 MB |
| `models/model_lr_usd.joblib` | Logistic Regression | usd | 27s | 0.0 MB |
| `models/model_rf_usd.joblib` | Random Forest | usd | 327s | 72.9 MB |
| `models/model_hgb_usd.joblib` | Gradient Boosting | usd | 178s | 0.4 MB |

> ⚠️ 全部 `*.joblib` 被 `.gitignore` 排除，也不进 zip（体积大）；需要时用对应脚本重生成，schema 里记了 features/threshold/metrics。
