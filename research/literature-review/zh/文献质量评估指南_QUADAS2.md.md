# 文献质量评估指南 — QUADAS-2 数据字典  
*系统综述：基于视觉的 AI 步态评估（卒中后）*  
**版本 1.1 | 最近更新：2025-06-01**  

---

## 1 │ 定位与范围
本指南用于 **单篇原始研究** 的风险偏倚（Risk of Bias，RoB）与可适用性（Applicability Concern，AC）定量标注；扩展自 QUADAS-2 标准流程 [[1]](#ref1)。  
- **Sheet `03_QUADAS2`** : *域级 / 研究级* 结论与聚合指标。  
- **Sheet `03_QUADAS2_Items`** : 19 条信号问题（Signalling Questions, SQ）逐项打分与证据链。  

> **分层逻辑** Items → Domain → Study 任一上游修改须同步刷新下游结果，  
>  脚本可据此自动执行一致性与留空校验（见 § 5）。  

---

## 2 │ 字段定义与取值域  

### 2.1 Sheet `03_QUADAS2` （Study-level）

| 字段 | 类型 | 取值/格式 | 专业释义 |
|------|------|-----------|----------|
| **Study_ID** | `string` | `SurnameYY` 或 `Acrn-YY` | 数据主键，全工作簿唯一；与 *01_Studies* 同步。 |
| **Author_Year** | `string` | `FirstAuthor (YYYY)` | 便于人工检索的可读标签。 |
| **Reviewer_1 / _2** | `string` | 姓名缩写 | 独立初评者标识。 |
| **Consensus_Date** | `date` | `YYYY-MM-DD` | 完成域级共识日期。 |
| **D1_Risk – D4_Risk** | `enum` | **L / H / U** | 四域风险 (Low, High, Unclear)。 |
| **D1_App – D4_App** | `enum` | **L / H / U** | 四域可适用性 Concern。 |
| **LowRisk_Count** | `int` | 0 – 4 | 4 个域中被判 *Low Risk* 的数量。 |
| **Overall_RiskLevel** | `enum` | Low / Moderate / High | 规则：<br>  ➊ 全域 L → Low；➋ 任一域 H → High；➌ 其余 → Moderate。 |
| **Overall_Score4** | `int` | 0 – 4 | 与 *LowRisk_Count* 等值，用于森林图配色。 |
| **Core40_Flag** | `bool` | 1 / 0 | 是否归入“核心 40” 深度分析池。 |
| **Consensus?** | `bool` | Y / N | 域级结论是否已双评一致。 |
| **Notes_Expert** | `text` | — | 院士/领域专家的附加见解（可 Markdown 列表）。 |

---

### 2.2 Sheet `03_QUADAS2_Items` （SQ-level）

| 字段 | 类型 | 取值 | 释义 |
|------|------|------|------|
| **Study_ID** | `string` | — | 主外键。 |
| **D1_SQ1 … D4_SQ5** | `enum` | **Y / N / U** | 19 项信号问题原始判定。 |
| **Response_R1 / _R2** | `enum` | Y / N / U | 两位评审的独立打分（与 SQ 顺序对应）。 |
| **Consensus** | `enum` | Y / N / U | 协商后共识打分；用于覆写 SQ 列。 |
| **Justification** | `string` | 50–250 字 | 关键语句 / 数据；可引用表格、附录。 |
| **Source_Evidence** | `string` | `p x, Fig y` | 页码或图、表、附录索引。 |
| **Risk_Flag** | `enum` | L / H / U | 根据 19 项 SQ→域规则自动生成。 |
| **Applicability_Flag** | `enum` | L / H / U | 依 PICO 匹配度评定。 |
| **Last_Update** | `date` | YYYY-MM-DD | 最后编辑时间戳。 |
| **AI_Specific?** | `bool` | 1 / 0 | D2_SQ3 / SQ4 等 AI 独有条目标 1。 |
| **Direction_of_Bias** | `enum` | + / – / ? | 若 Risk_Flag = H：+ 高估；– 低估；? 不确定。 |
| **Auto_QC_Status** | `enum` | pass / warn / fail | 脚本校验：留空/越界/逻辑冲突。 |

---

## 3 │ 19 条信号问题与操作化标准  

| 域 (D) | SQ | 中文简述 | 判 *Yes* 的 **可操作标准** |
|--------|----|----------|---------------------------|
| **D1 Patient Selection** | 1 | 连续或随机样本？ | 报告“连续”或“随机抽样”；非方便样本。 |
| | 2 | 避免病例-对照？ | 设计非 *case-control*；无显著 spectrum bias。 |
| | 3 | 无不当排除？ | 排除理由充分；排除率 < 10%。 |
| | 4 | ≥10 例卒中？ | 报告卒中子样本 ≥ 10。 |
| | 5 | 病程均衡？ | 急/亚急/慢性 ≤ 70% 集中于单阶段。 |
| **D2 Index Test** | 1 | Index 盲法？ | Index 结果解读时未知参考标准。 |
| | 2 | 预设阈值？ | 阈值/权重在外部验证前已锁定。 |
| | 3 | 上线前锁定模型？ | 报告“模型冻结”或仅推断不调参。 |
| | 4 | 无数据泄漏？ | 训练/验证/测试严格隔离；特征工程不越界。 |
| **D3 Reference Standard** | 1 | 标准准确？ | Vicon、GAITRite 或临床公认诊断依据。 |
| | 2 | 参考盲法？ | 参考评估者不知 Index 结果。 |
| | 3 | 金标准装置？ | MoCap、压力板等。 |
| | 4 | 报告精度？ | 量化误差 ±SD、ICC、SEM 任一。 |
| **D4 Flow & Timing** | 1 | 间隔合适？ | Index-Ref ≤ 30 min（室内）；≤ 24 h（外设）。 |
| | 2 | 全体接受参考？ | `n_ref == n_total` 或 差异 < 5%。 |
| | 3 | 完整分析？ | 无 “per-protocol” 删减；ITT 或全部样本。 |
| | 4 | 同步采集？ | 同步或同一试次完成。 |
| | 5 | 透明缺失？ | 缺失原因列举；有插补/敏感度分析。 |

> **一票否决原则** 任一带 **✱** 号核心 SQ = No → 该域直接判 *High Risk*。  

---

## 4 │ 域级与整体评分算法
flowchart LR
  subgraph SQ级
    A1[19× SQ<br>(Y/N/U)]
  end
  A1 --> B1[域级 Risk / App<br>(D1–D4)]
  B1 --> C1[LowRisk_Count]
  C1 --> D1[Overall_Score4]
  B1 --> D2[Overall_RiskLevel]
