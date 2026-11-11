# TF-IDF 文本层 · 关键词面板与文档级归因（S6）

> 2026-11-10 · 承接 `TF-IDF文本层_结果报告.md`（S0–S5，模型侧 +0.00788 AUC）
> 本文只做两件事：**把"文案里哪些词有用"做成可审计的面板**，**把单条项目的文案得分拆到词**。
> 脚本 `12_审查与文本层\build_text_explain.py`，全流程 **84.1 s**，单机、无外部依赖、不联网。
>
> **v1.1（2026-11-11 修订）**：修正面板计数窗与词表窗不一致的问题——`train` 掩码原先误用了主模型的 `2020` 上限，导致 ①"≤2022"标签与实算 ≤2020 不符、②`early`(≤2020) 列与 `train` 完全重合（重复列）、③`log_odds` 与 `coef`(≤2022) 跨窗比较。现 `train` 与词表/LR 统一为 **≤2022**，`early` 保持 **≤2020** 作为真正的第三期对照列；脚本内加断言防止掩码再次塌缩。文中全部数字为修订后重跑结果（84.1 s）；归因部分（`text_contrib_top.csv`）不受影响。

---

## 0. 一页结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | 关键词面板建成：**15 个大品类、5,509 个词、3,069 个 FDR<0.05**，其中 **77.5%** 在独立期（2023+）同号 | `keyword_panel.csv`，§2 |
| 2 | 词级信号与品类生态一致，可解释：Art/Fashion/Crafts 是 **enamel pin** 生态族，Publishing 是 **zine/limited edition/anthology**，Games 是 **stl files/stl/3d printable**，Technology 是 **raspberry pi/dual/compatible** | 表 2，图 `figt8` |
| 3 | 负向词多是**项目类型**而非**文案质量**：`android/mobile/fps`（Games）、`app/platform/website`（Technology/Publishing）、`hip hop/mixtape/rap`（Music）→ 面板可以指路，但不能当"改文案就能加分"的处方 | §3.2 |
| 4 | 文档级归因可闭合：100,469 条 test 文档的 `Σ x·w + b = logit(text_score)`，最大误差 **6.6e-14**；平均 13.94 个有效词，**99.67%** 文档至少有一个正向贡献词 | `text_contrib_top.csv`，§4 |
| 5 | 这把"文案杠杆"从口号变成词级动作：每条项目输出 top-5 正向/负向词 + 贡献值 + 该词所属品类的信任等级 | 表 4 |
| 6 | 面板**不能**证明"加长文案能提分"——它给的是"用哪些词"，与 `figt7` 的"长度在支撑域内波动 ≤3pp"互相印证 | §5 |

一句话：**文本层的用处是"换词"，不是"加长"。** 面板负责说清换什么词、在哪个品类、可不可信；归因负责把单条项目的话具体到几个词。

---

## 1. 口径与方法（与已交付的文本层严格对齐）

### 1.1 先验自校验（避免"另起一套"）

面板与归因必须建立在**与 `text_scores.csv.gz` 完全相同的文本模型**上，否则词级解释与模型行为对不上。因此 `fit_text_model()` 按 `build_text_layer.build_text_score()` 的参数逐项重拟合，并做三条对账：

| 对账项 | 交付管线（S2–S3） | 本次重拟合 | 判定 |
|---|---|---|---|
| 词表拟合窗 | `yr<=2022`，n=504,160 | 同 | ✅ |
| 词表规模 / 非零元 | 30,000 / 8,417,547 | 30,000 / 8,417,547 | ✅ 完全一致 |
| text-only test AUC | 0.7799 | **0.779922** | ✅ 一致 |

> 注：首次实现时误用 `yr<=2020`（443,693 篇）拟合词表与 LR，nnz=8,391,065、test AUC=0.7670，**与交付口径不符**，已修正。词表与最终分类器用的是 **≤2022 全窗**，不是 2020 模型训练窗——这一点在复用代码时极易踩错，特此记录。
>
> 第二处同类错误（v1.1 修正）：面板的 `train` 掩码当时写成了 `B.TRAIN_MAX_YEAR`（其值为 2020），于是 ①`n_docs_category`、`log_odds`、`p_fdr`、支撑门实际都按 ≤2020 计算，与文中"≤2022"不符；②`early`(≤2020) 与 `train` 完全重合，`log_odds_early_2020` 是纯复制列（实测 5,509 行中 0 行不同）；③`coef` 来自 ≤2022 的 LR，与 ≤2020 的 `log_odds` 跨窗比较。现统一为 `train = ≤2022`（与词表/`coef` 同窗）、`early = ≤2020`，并在 `panel_table` 之后加断言：若两期差值 >1e-6 的行不足 90%，直接报错。

### 1.2 面板统计（`keyword_panel.csv`）

对每个品类、每个词，在**训练窗（≤2022）**内做标准 2×2 列联表：

```
                   达标        未达标
含该词              a            b
不含该词            c            d
```

- 校正对数比：`lo = log((a+0.5)/(b+0.5)) − log((c+0.5)/(d+0.5))`
- 标准误：`se = sqrt(1/(a+.5) + 1/(b+.5) + 1/(c+.5) + 1/(d+.5))`，双侧正态 p，**品类内 BH-FDR**
- 支撑门（三条全过才进表）：`a ≥ 20`、`b ≥ 20`、`a+b ≥ max(20, 0.5%·品类文档数)`；品类需 ≥2,000 篇
- **跨期稳定性**：用独立期（`yr≥2023`）重算同一 `lo`，要求两期同号且 `|lo| ≥ 0.10`（`sign_stable`）
- **早期对照**：`log_odds_early_2020` 为 `yr≤2020` 单独重算的第三期存档列，**只用于事后排查漂移，不参与 `trust` 判定**
- **与模型一致性**：`coef` 为该词在 text-only LR 上的系数（与 `log_odds` 同为 ≤2022 窗），`coef_agrees = sign(coef)=sign(lo)`
- **信任分级**：`高` = FDR 显著 ∧ 跨期稳定 ∧ 系数同号；`中` = 显著且（稳定 ∨ 同号）；`低` = 其余
- **杂词标记**：`is_artifact` 标出 URL/HTML 碎片、纯年份、乱码（共 51 行），不进入精选 top 列表与图中

### 1.3 文档级归因（`text_contrib_top.csv`）

线性模型的可加性直接给出逐词贡献：

```
contrib_j = x_ij · w_j          x = tf-idf 值, w = 该词系数
Σ_j contrib_j + b = logit(text_score_i)
```

对 test 窗（`yr≥2023`，n=100,469）逐行取贡献最大的 5 个正向词与 5 个负向词，另存 `sum_pos_contrib`、`sum_neg_contrib`、`logit_p_text`、`intercept`，**用恒等式自检**（实测最大误差 **6.57e-14**，float64 精度级）。

---

## 2. 面板结果

### 2.1 覆盖与可信度

| 品类 | 显著词数 | 跨期同号率 | 高可信词数 | 品类 | 显著词数 | 跨期同号率 | 高可信词数 |
|---|---:|---:|---:|---|---:|---:|---:|
| Games | 342 | **89.2%** | 244 | Publishing | 246 | 70.3% | 148 |
| Technology | 315 | **89.5%** | 228 | Food | 207 | 62.3% | 113 |
| Music | 333 | 79.9% | 206 | Comics | 153 | 82.4% | 107 |
| Design | 284 | 77.5% | 190 | Crafts | 127 | 83.5% | 92 |
| Art | 264 | 76.1% | 150 | Photography | 122 | **63.1%** | 64 |
| Fashion | 244 | 80.7% | 165 | Theater | 102 | **61.8%** | 50 |
| Film & Video | 243 | 70.4% | 141 | Dance | 59 | 71.2% | 30 |
| **合计** | **3,069** | **77.5%** | **1,956** | Journalism | 28 | 75.0% | 18 |

读法：**跨期同号率就是"这个词现在还算不算数"的体检**。Technology/Games 最稳（89.5% / 89.2%），Theater/Photography/Food 最不稳（61.8% / 63.1% / 62.3%）——后者的词更多是当年特定项目类型的产物。

### 2.2 每品类精选词（高可信，按 `log_odds` 排序；完整 top-10 见 `text_explain_summary.json`）

| 品类 | 多用（正向）示例 | 少用（负向）示例 |
|---|---|---|
| Art | `pins featuring`、`pin collection`、`collection enamel`、`pin series`、`enamel pins`(+1.39, lift 1.50, 覆盖 11.7%) | `supplies`、`oil`、`art gallery`、`digital`、`want` |
| Comics | `hardcover`、`collected`、`collecting`、`returns`、`adventure comic` | `super hero`、`looking`、`web`、`dream`、`funny` |
| Crafts | `bowls`、`pens`、`enamel pins`、`edition`、`make 100` | `starting`、`different`、`candles`、`body`、`trying` |
| Dance | `evening`、`season`、`works`、`work`、`dance theater` | `hip hop`、`hip`、`fun`、`learn`、`children` |
| Design | `titanium`、`edc`、`everyday carry`、`rfid`、`japanese` | `children`、`allows`、`safety`、`company`、`start` |
| Fashion | `pin collection`、`pin series`、`enamel pins`、`enamel`、`pins` | `clothing`、`trendy`、`hair`、`fitness`、`clothing line` |
| Film & Video | `thesis film`、`thesis`、`dark comedy`、`short film`、`short` | `fun`、`produce`、`adventures`、`start`、`amazing` |
| Food | `brick mortar`、`mortar`、`brick`、`downtown`、`neighborhood` | `el`、`bbq`、`llc`、`chicken`、`cupcakes` |
| Games | `stl files`、`stl`(+2.65, 覆盖 3.1%)、`3d printable`、`zine`、`printable` | `person`、`multiplayer`、`platform`、`ios`、`shooter` |
| Journalism | `season`、`issue`、`print`、`publication`、`journalist` | `new`、`podcast`、`stories`、`help`、`website` |
| Music | `length record`、`new length`、`pre order`、`pre`、`making new` | `gospel`、`radio`、`artist`、`hop`、`hip` |
| Photography | `art nude`、`nudes`、`nude`、`university`、`zine` | `like`、`travel`、`capture`、`share`、`country` |
| Publishing | `zine`(+1.51, lift 1.85)、`limited edition`、`anthology`、`hardcover`、`limited` | `children books`、`reality`、`want`、`looking`、`write` |
| Technology | `dual`、`raspberry pi`(+1.78, lift 2.49)、`raspberry`、`pi`、`compatible` | `events`、`allow`、`search`、`website`、`application` |
| Theater | `fringe`、`edinburgh`、`edinburgh fringe`、`fringe festival`、`solo` | `halloween`、`issues`、`que`、`talent`、`movie` |

机器可读版：`keyword_panel.csv`（24 列，含 `n_success_docs / n_failure_docs / success_rate_with_token / success_rate_category / lift / log_odds / se / p_fdr / coef / log_odds_test_2023 / sign_stable / coef_agrees / trust / direction / is_artifact / is_top_pos / is_top_neg`）。

精选词的整体力度（150 行 `is_top_pos`）：`log_odds` 中位 **+1.29**（区间 +0.64…+2.67），`lift` 中位 **1.65**，即"含该词的项目达标率约为品类均值的 1.65 倍"。

**但注意覆盖度**：这些高 lift 词的文档覆盖率中位只有 **1.14%**——它们是"识别信号"，不是"加进文案就能复制"的开关。高 lift 常来自稀有词，中频词（如 `enamel pins` 覆盖 11.7%、lift 1.50）才是可操作的量级。

### 2.3 视觉版

`04_图表\figt8_keyword_panel.png`：Music / Food / Fashion / Publishing 四品类红蓝横条（蓝=多用，红=少用，只画 FDR<0.05 且 2023+ 同号的高可信词）。

---

## 3. 怎么读这张面板（三条失效模式）

### 3.1 "当年有效"≠"现在有效"：看 `trust` 列，别只看 `log_odds`

典型：Art 的 `residency`，训练窗 `log_odds=+1.28`（FDR 显著），但 2023+ 独立期是 **−0.42** → `sign_stable=False`，只能判 `中`。同理 Journalism 的 `season`、Theater 的 `edinburgh` 在后期衰减明显。**给用户看的建议只应取 `trust=高`。**

### 3.2 负向词多半在描述"项目类型"，不是"文案写得差"

- Games 负向：`android / mobile / fps / shooter / mmorpg` —— 这是在说"手游/端游项目更难达标"，不是"别写这些词"，**改词没有用，改的是项目定位**。
- Technology 负向：`app / platform / website / services` —— 同上，软件类项目在 KS 上成功率系统性偏低。
- Music 负向：`hip hop / mixtape / rap / talent` —— 品类内定位信号。
- 真正的"泛承诺词"负向：`want / provide / start / trying / help / like` —— 这类才更接近"文案空话"的解释，但也可能是低预算项目的共同话术（**相关≠因果**）。

结论：面板可用于**诊断"这个项目像不像该品类的达标项目"**，不可用于"把负向词删掉就能加分"。后者需要 §5 说的对照实验。

### 3.3 分词与清洗带来的两个已知伪影

- **sklearn 停用词表误删领域词**：`ENGLISH_STOP_WORDS` 含 `full / back / front / fire / find / fill / bill / detail / top / bottom`（318 词），所以 `full-length record` 被切成 `length record`、`feature-length` 剩 `feature length`。面板里 Music 的 `length record`、`new length`、`debut length` 都应按 **full-length record / brand new full-length** 读。下一轮若换自定义停用词表（只去虚词），这些词会恢复为完整短语。
- **模板串与拼写变体**：`make 100`（KS 官方"限量 100 件"活动）、`#NAME?`、`3D-STL`、`Enamel Pins / enamel pins` 大小写已统一但单复数并存；`is_artifact` 已隔离 51 行 URL/年份/乱码，但单复数合并仍需归一化。

---

## 4. 文档级归因（`text_contrib_top.csv`）

### 4.1 规模与闭合性

| 项 | 值 |
|---|---|
| 覆盖文档 | 100,469（test，2023+） |
| 每文档有效词（非零 tf-idf） | 均值 13.94、中位 14、sd 5.05、max 36 |
| 含至少一个正向贡献词 | **99.67%** |
| `Σcontrib + b − logit(p_text)` 最大误差 | **6.57e-14** |
| `p_text` 分位 | 1% 0.085 · 10% 0.257 · 25% 0.407 · 50% 0.607 · 75% 0.792 · 90% 0.906 · 99% 0.983 |
| `p_text<0.3` / `>0.9` | 13.7% / 10.8% |

### 4.2 示例：低分项目被"哪个词扣的分"

| 项目（截断） | 品类 | p_text | 主要负向贡献 | 主要正向贡献 |
|---|---|---:|---|---|
| Titan - Multiplayer Online FPS (development) | Games | 0.021 | `fps` −0.92、`multiplayer` −0.50、`shooter` −0.44 | `titan` +0.23 |
| Valutok App; A great platform to network and monetize | Technology | 0.005 | `app` −1.44、`platform` −0.89、`users` −0.50 | `send` +0.23 |
| Karmic Publications Website | Publishing | 0.043 | `website` −1.16、`publishing company` −0.53 | `help` +0.15 |
| Prophecy of the storm book series | Publishing | 0.033 | `book series` −0.66、`prophecy` −0.55、`publish` −0.51 | `book` +0.31 |

高分侧同理（`p_text>0.9` 的 10.8% 文档）：

| 项目（截断） | 品类 | p_text | 主要正向贡献 |
|---|---|---:|---|
| The Trials of King Arnac - Illustrated edition gamebook | Publishing | 0.907 | `gamebook` +1.41、`edition` +0.82 |
| MAKE 100: Pirate Pins - Enamel badges | Fashion | 0.892 | `make 100` +0.56、`enamel` +0.52 |

### 4.3 这一层能干什么、不能干什么

- **能**：给每条项目一句话的"文案诊断"——"你在 Publishing，`website/promote/writing book` 这类词在你的品类里是负向；同类达标项目更常说 `zine/anthology/art book/hardcover`"。
- **不能**：把这些词当作可直接替换的模板。词级贡献是**线性文本模型**的解释，主模型（HistGB）消费的是 `text_score` 一个标量，所以"改一个词对最终达标概率的影响"只能通过重打分（下一轮 S7 对照实验做）来量化。
- **不能**：跨品类套用。`rap` 在 Music 负向，在 Games 里毫无意义；`hardcover` 在 Publishing 正向，在 Music 无意义。面板的作用域是"品类内"。

---

## 5. 与三个痛点的关系（回到交付目标）

| 痛点 | 本文贡献 | 边界 |
|---|---|---|
| ① "只用字数/词数，加长文案反而 −13pp" | 与 `figt7` 一致：长度在支撑域内不单调、波动 ≤3pp；文本层真正的增量来自**内容词**（+0.00788），不是长度（+0.00145）。面板解释了为什么"加长"没有稳定方向——达标项目的文案并不更长，而是"词更像该品类" | 面板不能证明"缩短能提分"，也不能给出长度最优值 |
| ② "文案杠杆无法给建议" | 现在可以给出**品类内、带信任等级的换词建议**（top-5 正向词 + 需要回避的负向词 + 覆盖度 + 跨期稳定性），并附单条项目的词级归因 | 建议强度只到"关联 + 一致性"，到不了"因果 + 幅度"；幅度需 S7 重打分 |
| ③ 评分卡/决策轴 | 面板可按 `category_parent_name` 直接并入 `advice_by_category.py` 的建议行（品类内选词），但**不要**改动已交付的评分卡口径 | 并入建议卡属于下一步，本文只产出面板 |

---

## 6. 已知局限（用前必读）

1. **观测关联，非因果**：所有 `log_odds` 都来自同一批已结束项目的文本共现；未控制目标金额、时长、履历。高 lift 词很可能是"低成本/系列化/已成型项目"的代理变量（如 `enamel pins` 对应成熟周边生意）。
2. **词级与项目级混在一起**：文档 = `name_clean + " " + blurb_clean`，无法区分"标题里的词"与"简介里的词"的作用；标题权重更高这一点未被建模。
3. **文本重复未剔除**：全表 9,631 行重复简介（1.59%），且 1,001 条 test 简介与 train 逐字相同（0.996%），面板计数对这些模板串略微高估（`make 100` 类模板受影响最大）。
4. **停用词表伪影**：见 §3.3，`full/back/front/find/detail` 等被删，短语退化。
5. **词表上限 30,000**：稀有但专有的词（品牌名、硬件名）可能被 `max_features` 截断；`min_df=5` 也过滤掉极稀有词。
6. **面板只到 15 个大品类**：147 个子品类、25 个国家未分层；Dance/Journalism/Theater 样本小（显著词 27–95 个），建议强度弱。
7. **归因是线性模型的解释**：`coef` 与 `log_odds` 两条口径有时不一致（`coef_agrees=False` 的行已在表中标出），因为前者是"控制其他词后"的边际效应，后者是"单独看这个词"的共现比。

---

## 7. 复现与产物

```powershell
cd "IS6400_Kickstarter_交付包_完整\12_审查与文本层"
python -X utf8 build_text_explain.py        # 84.1 s，峰值 <6 GB
```

| 产物 | 说明 |
|---|---|
| `11_结果与报告\keyword_panel.csv` | 5,509 行 × 24 列，品类×词面板（含 FDR、跨期稳定性、系数、信任等级） |
| `11_结果与报告\text_contrib_top.csv` | 100,469 行 × 20 列，test 文档逐条 top5 正/负词贡献 + 闭合性字段 |
| `11_结果与报告\text_explain_summary.json` | 汇总：词表规模、截距、面板统计、每品类 top 词、恒等式误差、耗时 |
| `04_图表\figt8_keyword_panel.png` | 四品类红蓝横条面板图 |
| `12_审查与文本层\work\text_model_fitted.joblib` | 重拟合的向量器 + LR（供复算/答辩追溯） |
| `12_审查与文本层\build_text_explain.py` | 全部逻辑（含 `fit_text_model` / `build_panel_counts` / `panel_table` / `attribute` / `fig_panel`） |

自校验三件套（跑完必须打印）：`nnz=8417547`、`text-only test AUC=0.7799`、`logit_identity_max_abs_err < 1e-10`。

---

## 8. 下一步（尚未做，建议按序）

1. **S7 文案对照重打分**：对同一批低分项目，用 top-5 正向词替换负向词后重算 `text_score` → 主模型概率差 → 给出"文案改动幅度"的分布（这才是能写进建议卡的数字）。
2. **停用词表修正 + 短语归一**：自定义停用词（仅虚词）+ 单复数/连字符归一，重跑面板，看 `full-length` 类短语是否恢复、面板稳定性是否提升。
3. **去重敏感性**：剔除 1,001 条 train/test 逐字重叠后重估面板，确认 `make 100` 等模板词不驱动结论。
4. **并入建议卡**：`advice_by_category.py` 增加一行"文案候选词（品类内，trust=高）"，并标注"仅诊断、非处方"。
