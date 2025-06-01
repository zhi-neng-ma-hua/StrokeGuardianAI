# 文献质量评估指南 — QUADAS-2 数据字典  
_系统综述：Vision-Based AI Systems for Post-Stroke Gait Assessment_  
**Version 1.2 | Last update 2025-06-01**

---

## 目录
1. [工具定位](#sec1)  
2. [数据表概览](#sec2)  
3. [`QUADAS2` 字段定义（Study level）](#sec3)  
4. [`QUADAS2_Items` 字段定义（SQ level）](#sec4)  
5. [19 项信号问题与“是/否”判据](#sec5)  
6. [域级与研究级评分算法](#sec6)  
7. [自动质量控制（Auto-QC）脚本规则](#sec7)  
8. [参考文献](#sec8)  

---

<a id="sec1"></a>
## 1 │ 工具定位
本指南将 **QUADAS-2** (Quality Assessment of Diagnostic Accuracy Studies-2) 扩展为 _深度学习+步态学_ 语境专用模板，用于评价单篇研究的 **风险偏倚 (RoB)** 与 **可适用性顾虑 (AC)**。  
- **Sheet `03_QUADAS2`** 域级/研究级聚合结论  
- **Sheet `03_QUADAS2_Items`** 19 条信号问题逐项打分与证据链  

> **自下而上链式更新**  
> `SQ → Domain → Study` 任何上游改动必须向下游自动同步，保证可追溯性与机读一致性。

---

<a id="sec2"></a>
## 2 │ 数据表概览

| Sheet | 角色 | 粒度 | 关键输出 |
|-------|------|------|----------|
| **QUADAS2** | 结论表 | Study | 4 域 RoB & AC，整体风险级别，核心文献标记 |
| **QUADAS2_Items** | 原始表 | SQ | 19×Y/N/U 判定、佐证、偏倚方向、脚本 QC |

---

<a id="sec3"></a>
## 3 │ 字段定义 — `03_QUADAS2`（研究级）

| 字段 | 数据型 | 合法值 | 说明 |
|------|--------|--------|------|
| `Study_ID` | `string` | `Smith23` / `ABC-23` | 主键，与 _01_Studies_ 对齐 |
| `Author_Year` | `string` | `Smith (2023)` | 可读引用 |
| `Reviewer_1` / `_2` | `string` | 姓名缩写 | 双盲初评者 |
| `Consensus_Date` | `date` | `YYYY-MM-DD` | 域级共识落款 |
| `D1_Risk` – `D4_Risk` | `enum` | **L/H/U** | Patient - Index - Reference - Flow |
| `D1_App` – `D4_App` | `enum` | **L/H/U** | 同上四域的适用性 Concern |
| `LowRisk_Count` | `int` | 0–4 | 四域中 *Low Risk* 数量 |
| `Overall_RiskLevel` | `enum` | Low / Moderate / High | 规则见 § 6 |
| `Overall_Score4` | `int` | 0–4 | 与 `LowRisk_Count` 等值，便于可视化 |
| `Core40_Flag` | `bool` | 1/0 | 是否进入“核心 40” |
| `Consensus?` | `bool` | Y/N | 域级结论已达一致？ |
| `Notes_Expert` | `text` | — | 院士/专业顾问补充 |

---

<a id="sec4"></a>
## 4 │ 字段定义 — `03_QUADAS2_Items`（信号问题级）

| 字段 | 数据型 | 合法值 | 说明 |
|------|--------|--------|------|
| `Study_ID` | `string` | — | 外键 |
| `D1_SQ1 … D4_SQ5` | `enum` | **Y/N/U** | 19 项 SQ 最终共识 |
| `Response_R1` / `_R2` | `enum` | Y/N/U | 初评答案（列顺序 × SQ 顺序） |
| `Consensus` | `enum` | Y/N/U | 若双评不一致需第三方裁决 |
| `Justification` | `text` | 50–250 字 | 关键信息或引用原文 |
| `Source_Evidence` | `string` | `p 3, Fig 2` | 页码/图表/附录编号 |
| `Risk_Flag` | `enum` | L/H/U | 域级 RoB 自动回填 |
| `Applicability_Flag` | `enum` | L/H/U | 域级 AC 自动回填 |
| `Last_Update` | `date` | YYYY-MM-DD | 最近编辑 |
| `AI_Specific?` | `bool` | 1/0 | D2_SQ3/4 设 1 |
| `Direction_of_Bias` | `enum` | + / – / ? | RoB=H 时：+高估 –低估 |
| `Auto_QC_Status` | `enum` | pass / warn / fail | 脚本校验结果 |

---

<a id="sec5"></a>
## 5 │ 19 项信号问题与可操作判据

| 域 | SQ | 简述 | **判 Yes 的实操标准** |
|----|----|------|------------------------|
| **D1 Patient Selection** | 1 | 连续/随机样本？ | 明示“连续”或随机抽样；拒绝便利样本 |
|  | 2 | 避免病例-对照？ | 设计非 case-control；无 spectrum bias |
|  | 3 | 无不当排除？ | 排除率 < 10% 且理由合理 |
|  | 4 | ≥ 10 例卒中？ | 卒中子样本 ≥ 10 |
|  | 5 | 病程均衡？ | 急/亚急/慢性 ≤ 70 % 集中单阶段 |
| **D2 Index Test** | 1 | Index 盲法？ | 结果解读者不知参考标准 |
|  | 2 | 预设阈值？ | 阈值/权重在外部测试前已锁定 |
|  | 3 | 模型冻结？ | 报告“model frozen”或仅推断不调参 |
|  | 4 | 无数据泄漏？ | 训练/验证/测试严格隔离 |
| **D3 Reference Standard** | 1 | 参考准确？ | 3D MoCap、GAITRite、MRI 等一线金标准 |
|  | 2 | 参考盲法？ | 评估者未知 Index 结果 |
|  | 3 | 使用金标准装置？ | Vicon、Qualisys、压力板等 |
|  | 4 | 报告精度？ | 提供 ICC/SEM/±SD ≥ 1 项 |
| **D4 Flow & Timing** | 1 | 间隔合适？ | Index-Ref ≤ 30 min（室内）或 ≤ 24 h |
|  | 2 | 全体接受参考？ | `n_ref ≥ 95 % n_total` |
|  | 3 | 完整分析？ | 采用 ITT 或报告全部样本 |
|  | 4 | 同步采集？ | 同一试次或同步触发 |
|  | 5 | 透明缺失？ | 缺失原因+插补/敏感性分析 |

**一票否决**：★ 标记的关键 SQ 为 **No** 即该域直接判 *High Risk*。  

---

<a id="sec6"></a>
## 6 │ 评分算法

```mermaid
flowchart LR
    %% --- Signalling-question layer ----------------
    subgraph SQ19["Signalling Questions (19 items)"]
        A[19 × SQ<br/>(Y / N / U)]
    end

    %% --- Domain layer ----------------------------
    A -->|scoring rules| B[Domain-level<br/>RoB & Applicability]

    %% --- Study layer -----------------------------
    B --> C[LowRisk_Count]
    C --> D[Overall_Score4]
    B --> E[Overall_RiskLevel]
