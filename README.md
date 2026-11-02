# IS6400 Group Project — Kickstarter 众筹上线前配置优化

课程项目：用 Kickstarter 公开数据构建「上线前配置 → 达标概率」的预测与评分工具。
本仓库只跟踪**文本、脚本与小体积证据**；数据与演示文件不入库（见下）。

## 一、当前工作区结构

```
group\
├─ Kickstarter Full Campaigns_694062_25_April_2026.csv   原始数据（694,062 × 43，455 MB）※不入库
├─ IS6400_Kickstarter_交付包_完整\                         最新交付包（解压自 v1 zip）
│   ├─ 01_清洗数据\         ks_clean.csv（673,839 × 81）※入库被忽略；dropped_rows.csv.gz ※忽略
│   ├─ 02_清洗规则与理由\   清洗脚本.py、数据清洗文档.md、一致性审计报告.md、字段字典.md
│   ├─ 03_方法与预期模型\   方法与预期模型说明_中文.md、选题说明_中文.md
│   ├─ 04_图表\            11 张分析/画像图（PNG 已入库）
│   ├─ 05_简报清单\        简报已完成清单_中文.md
│   ├─ 06_独立清洗审核\    审核报告 + 证据 CSV/JSON（小表已入库）
│   ├─ 07_脚本修订验证\    修订说明、测试脚本、覆盖前备份（备份 CSV ※忽略）
│   └─ README_包内说明.md
└─ _archive_20261101\                                     旧版本与中间产物归档（2.09 GB）※不入库
```

## 二、数据口径与校验值（都可复算）

| 文件 | 规模 | sha256 |
| --- | --- | --- |
| 原始 CSV | 694,062 × 43，454,851,804 B | `598ab8ab…` |
| `01_清洗数据/ks_clean.csv`（最新，81 列） | 673,839 × 81，710,301,125 B | `aaf7a89e…` |
| `07_脚本修订验证/覆盖前备份/…csv`（修订前，68 列） | 673,839 × 68，652,236,755 B | `78f2c2f5…` |

关键口径（写报告时必须遵守）：

- 目标变量 `succ`：1 = successful，0 = failed；**canceled（69,210 行）`succ` 为空，不计入成功率**；
- 成功率分母 = **604,629**（successful 297,538 + failed 307,091），不是 673,839；
- 10 个事后/泄漏列**永不入模**：`pledged`、`usd_pledged`、`backers_count`、`percent_funded`、
  `converted_pledged_amount`、`spotlight`、`profile_state`、`staff_pick`、`state_changed_at`、`profile_state_changed_at`；
- 金额为**项目本币**（非 USD 占 31.65%）；换算方向已验证：`goal_usd = goal × static_usd_rate`；
- 时间切分：训练 `yr <= 2022`（502,975）/ 测试 `yr >= 2023`（100,469）；多数类基线 **0.6441**。

## 三、复现命令

```powershell
# 1) 用原始数据重跑清洗（输出到 07_修订清洗输出，不覆盖现有数据）
cd "IS6400_Kickstarter_交付包_完整\02_清洗规则与理由"
python 清洗脚本.py --raw "..\..\Kickstarter Full Campaigns_694062_25_April_2026.csv"
```

## 四、仓库约定

- **不入库**：`*.csv`、`*.zip`、`*.pptx`、`*.parquet`、`_archive_20261101/`（理由：体积大、可再生成或另有归档）。
- **入库**：`*.md`、`*.py`、`*.json`、交付包内 11 张图、`06_独立清洗审核/*.csv` 小表。
- 每个数字的出处要能追溯（图注写数据来源 + 计算脚本）；不写因果结论，只写关联。
- 版本命名：deck 等二进制交付物用 `_v1/_v2/...` 后缀，不入库，靠文件名与 sha256 区分。

## 五、课程要求摘要（Briefing 5% / Presentation 15% / Report 10%）

- Briefing（Week 5）：≤5 页 PPT、≤3 分钟、必含 Business Problem / Dataset Description（含字段名与数据类型）/ Key Variables / BDA Methods / Potential Contribution；
- Final Presentation（Week 12–13）：8 分钟 + 2 分钟 Q&A，全员到场；
- Final Report（5 月 1 日 23:59）：20–25 页、11pt 双倍行距，含一页 Executive Summary、分析程序与结果、参考文献（附录）。
