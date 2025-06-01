# README — *03_QUADAS2* Data Model  
*Systematic Review: Vision-Based AI Systems for Post-Stroke Gait Assessment*  
**Version 1.0 | Last updated:** 2025-06-01  

---

## 1. 目的与定位  
本工作簿依据 **QUADAS-2** 官方指南（2011, Ann Intern Med 155:529-536）扩展，用于系统评价中**单篇研究的质量与可适用性判定**。  
- **`03_QUADAS2`** Sheet：域-级（Domain-level）与整体（Study-level）结论。  
- **`03_QUADAS2_Items`** Sheet：19 个信号问题（Signalling Questions, SQ）逐条原始打分与佐证信息。  

> ✅ **“先细后粗”**：任何汇总结论（Risk / App etc.）必须由 *Items → Domain → Study* 自底向上传递，可机读复核。  

---

## 2. 字段说明  

### 2.1 `03_QUADAS2`（Study-level）  

| 字段 | 类型 | 允许值 / 格式 | 释义 |
|------|------|---------------|------|
| `Study_ID` | string | `Acronym-YY` 或 `SurnameYY` | 元数据主键，与 *01_Studies* 表一致。 |
| `Author_Year` | string | `FirstAuthor (YYYY)` | 显示友好型。 |
| `Reviewer_1 / _2` | string | 姓名缩写 | 首轮独立评审者。 |
| `Consensus_Date` | date | `YYYY-MM-DD` | 完成域级共识的日期。 |
| `D1_Risk` – `D4_Risk` | enum | **L** / **H** / **U** | 四大域偏倚风险：Low / High / Unclear。 |
| `D1_App` – `D4_App` | enum | **L** / **H** / **U** | 四大域可适用性 Concern。 |
| `LowRisk_Count` | int | 0-4 | 4 个域中判为 **L** 的数量。 |
| `Overall_RiskLevel` | enum | **Low** / **Moderate** / **High** | 规则：<br>· **Low** = 全域 *L*；<br>· **High** = 任一域 *H*；<br>· **Moderate** = 其他情况。 |
| `Overall_Score4` | int | 0-4 | `LowRisk_Count` 的复制列，可用于森林图配色。 |
| `Core40_Flag` | bool | 1 / 0 | 是否进入“核心40篇”深度分析池。 |
| `Consensus?` | bool | Y / N | 域级结论是否经双方确认。 |
| `Notes_Expert` | long-text |   | 顶级临床/AI 咨询的额外备注。 |

### 2.2 `03_QUADAS2_Items`（Signalling-question level）  

| 字段 | 类型 | 允许值 | 释义 |
|------|------|---------|------|
| `Study_ID` | string | 见上 |
| `D1_SQ1` ·· `D4_SQ5` | enum | **Y** / **N** / **U** | 19 个信号问题（详见 §3）。 |
| `Response_R1 / _R2` | enum | Y / N / U | 初评答案（与列顺序一一对应）。 |
| `Consensus` | enum | Y / N / U | 共识答案（机器用于自动回填 *SQ* 字段）。 |
| `Justification` | string | 50-250 字符 | 关键语句或数据；可引用表格或原文语句。 |
| `Source_Evidence` | string | “p 3, Table 1” | 页码、图表编号或 DOI-定位。 |
| `Risk_Flag` | enum | L / H / U | 遇到“**任何关键 SQ=No**”则域判 *High*；*All Yes* → Low；否则 *Unclear*。 |
| `Applicability_Flag` | enum | L / H / U | 按 QUADAS-2 指南结合综述 PICO。 |
| `Last_Update` | date | YYYY-MM-DD | 最近一次修改时间戳。 |
| `AI_Specific?` | bool | 1 / 0 | 仅 AI-特有项（如 D2_SQ3/4）标 1。 |
| `Direction_of_Bias` | enum | + / – / ? | 若判 *High*，标注可能方向：+ = 高估性能；– = 低估；? = 不确定。 |
| `Auto_QC_Status` | enum | pass / warn / fail | 由脚本验证值域、留空等。 |

---

## 3. 信号问题（19 Items）与判定规则  

| 域(D) | SQ | 中文简述 | 判 *Yes* 的操作化标准 |
|-------|----|----------|-----------------------|
| **D1 Patient Selection** | 1 | 连续或随机样本？ | 招募过程明示 “连续”/“随机抽取”；无便利样本。 |
| | 2 | 避免病例-对照？ | 研究设计非病例-对照 (case–control)。 |
| | 3 | 无不当排除？ | 排除理由合理且 <10% 为“技术失败/资料缺失”。 |
| | 4 | ≥10 stroke 患者？ | 样本量报告 ≥10；若混合队列统计 stroke≥10。 |
| | 5 | 病程均衡？ | 急 / 亚急 / 慢性 比例有说明且任一阶段 ≤70%。 |
| **D2 Index Test** | 1 | Index 解读盲法？ | 分析者不知参考标准结果；或自动管线。 |
| | 2 | 预设阈值？ | 模型阈值 / 权重在训练集外预先定义。 |
| | 3 | 上线前锁定模型？ | 明言“模型冻结”或外部验证仅推断不调参。 |
| | 4 | 无数据泄漏？ | 训练/验证/测试严格独立；无预处理共享信息。 |
| **D3 Reference Standard** | 1 | 参考标准准确？ | 3D MoCap / GAITRite / 临床确诊 MRI 等一线金标准。 |
| | 2 | 参考盲法？ | 参考评估者未知 Index 结果。 |
| | 3 | 使用金标准装置？ | 如 Vicon、Qualisys、GAITRite、压力板。 |
| | 4 | 报告精度？ | 提供误差(±SD)或 ICC/SEM ≥1 指标。 |
| **D4 Flow & Timing** | 1 | 时间间隔合适？ | Index–Ref ≤30 min；若居家/院外 ≤24 h。 |
| | 2 | 全体接受参考？ | `n_ref == n_total` 或 差异 <5%。 |
| | 3 | 完整分析？ | 报告全部受试者数据；无“per-protocol”删减。 |
| | 4 | 同步采集？ | Index 与 Ref 同步或同一试次。 |
| | 5 | 透明处理缺失？ | 明示缺失原因；采用合适插补/敏感性分析。 |

> ⚠️ “一票否决”：若 **任何** 关键 SQ (粗体) ＝ No → 该域直接标 *High Risk*。  

---

## 4. 域-级与整体评分算法  

```mermaid
graph TD
  Items --规则--> Domain(Risk/App)
  Domain --LowRisk_Count--> Overall_Score4
  Overall_Score4 --> Overall_RiskLevel
