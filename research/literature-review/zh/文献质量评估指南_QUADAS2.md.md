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
## 3 │ 字段定义 — `QUADAS2`（研究级）

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
## 4 │ 字段定义 — `QUADAS2_Items`（信号问题级）

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

<!-- ───────────── 19 Signal Questions Quick-Ref ───────────── -->
<div style="overflow-x:auto; font-size: 14px">

<table>
  <colgroup>
    <col style="width: 6%">
    <col style="width: 4%">
    <col style="width: 18%">
    <col style="width: 26%">
    <col style="width: 28%">
    <col style="width: 18%">
  </colgroup>
  <thead>
    <tr>
      <th>域</th>
      <th>SQ</th>
      <th>问题（中文缩写）</th>
      <th><strong>判 Yes 的<strong>“可操作”标准</th>
      <th>Why it matters / 易错示例</th>
      <th>关键判否逻辑</th>
    </tr>
  </thead>
  <tbody>

  <!-- ── D1 ─────────────────────────── -->
  <tr><td rowspan="5"><strong>D1<br>Patient<br>Selection</strong></td>
      <td>1★</td>
      <td>连续 / 随机样本？</td>
      <td>招募段落或流程图明确写出 “consecutive”<br>或使用随机号码表抽样；<br>不接受“方便取样 (convenience)”</td>
      <td>避免 spectrum bias；便利样本往往排除重症/合并症</td>
      <td>若发现“convenience sample”“retrospective chart screen” ⇒ **No**</td></tr>

  <tr><td>2★</td><td>避免病例-对照？</td>
      <td>研究设计非 case-control；若为诊断准确度研究须为 cohort / cross-sectional</td>
      <td>病例-对照易夸大 AUC 与敏感度</td>
      <td>明言 “case–control” 或 组间配对招募 ⇒ **No**</td></tr>

  <tr><td>3★</td><td>无不当排除？</td>
      <td>筛查 → 纳入全流程列出排除理由；总体排除 &lt; 10%</td>
      <td>高排除率 = 选择性偏倚</td>
      <td>排除占样本 ≥ 10% 且无正当理由 ⇒ **No**</td></tr>

  <tr><td>4</td><td>≥ 10 例 stroke？</td>
      <td>样本量表或结果段明示 n ≥ 10</td>
      <td>小样本不具备稳定的准确度估计</td>
      <td>n &lt; 10（或未报告）⇒ **Unclear** / **High**</td></tr>

  <tr><td>5</td><td>病程均衡？</td>
      <td>急 / 亚急 / 慢性 任一类比例 ≤ 70%</td>
      <td>单一病程阶段主导会削弱外部适用性</td>
      <td>≥ 70% 集中同一阶段且无分层分析 ⇒ **Unclear**</td></tr>

  <!-- ── D2 ─────────────────────────── -->
  <tr><td rowspan="4"><strong>D2<br>Index&nbsp;Test</strong></td>
      <td>1★</td><td>Index 盲法？</td>
      <td>算法或读片人员在得出结果时**未获知**参考标准</td>
      <td>防止“诊断决策回溯”偏倚</td>
      <td>同一团队先看 GAITRite 再标注 AI 输出 ⇒ **No**</td></tr>

  <tr><td>2</td><td>预设阈值？</td>
      <td>阈值 / cut-point 在 protocol 或注册表上预声明</td>
      <td>事后调阈会抬高准确度</td>
      <td>“best-Youden” post-hoc 优化 ⇒ **No**</td></tr>

  <tr><td>3</td><td>模型冻结？</td>
      <td>外部测试集仅 forward 推断，无任何再训练</td>
      <td>防止 leakage-fine-tune</td>
      <td>若报告“fine-tuned on test” ⇒ **No**</td></tr>

  <tr><td>4★</td><td>无数据泄漏？</td>
      <td>全流程分区：Train / Val / Test 互斥；<br>无“patient-overlap”或时间穿越</td>
      <td>leakage 可能高估性能 ≥ 20%</td>
      <td>同一患者步态循环跨 Train/Test ⇒ **No**</td></tr>

  <!-- ── D3 ─────────────────────────── -->
  <tr><td rowspan="4"><strong>D3<br>Reference<br>Standard</strong></td>
      <td>1★</td><td>参考准确？</td>
      <td>3D MoCap / GAITRite / MRI / 临床多学科共识诊断</td>
      <td>基准不准＝结果不可信</td>
      <td>自制简易摄像或问卷代替金标准 ⇒ **No**</td></tr>

  <tr><td>2</td><td>参考盲法？</td>
      <td>参考评估者不接触 Index 输出</td>
      <td>减少 review bias</td>
      <td>双盲未说明 ⇒ **Unclear**</td></tr>

  <tr><td>3</td><td>使用金标准装置？</td>
      <td>Vicon、Qualisys、GAITRite、Kistler FP 等</td>
      <td>保证度量物理准确度</td>
      <td>仅手机计步器 ⇒ **No**</td></tr>

  <tr><td>4</td><td>报告精度？</td>
      <td>参考装置给出 ICC、SEM、RMSE 等 ≤ 公认阈值</td>
      <td>便于评估总误差</td>
      <td>未报告任何精度指标 ⇒ **Unclear**</td></tr>

  <!-- ── D4 ─────────────────────────── -->
  <tr><td rowspan="5"><strong>D4<br>Flow & Timing</strong></td>
      <td>1</td><td>间隔合适？</td>
      <td>室内同日 ≤ 30 min；<br>若跨机构 ≤ 24 h 且病情稳定</td>
      <td>病程可能在间隔中变化</td>
      <td>间隔 > 7 天 且无解释 ⇒ **No**</td></tr>

  <tr><td>2</td><td>全体接受参考？</td>
      <td><code>n_ref / n_total ≥ 0.95</code></td>
      <td>避免 verification bias</td>
      <td>大幅缺少参考标准 ⇒ **High**</td></tr>

  <tr><td>3</td><td>完整分析？</td>
      <td>使用 ITT 或 <u>列出</u>排除清单；无“per-protocol only”</td>
      <td>排除失败样本易高估性能</td>
      <td>仅分析“成功帧” ⇒ **No**</td></tr>

  <tr><td>4★</td><td>同步采集？</td>
      <td>Index 与 Ref 同步触发或同一试次</td>
      <td>取消行为差异/疲劳影响</td>
      <td>轮流走两遍、顺序固定 ⇒ **Unclear** / **No**</td></tr>

  <tr><td>5</td><td>透明缺失？</td>
      <td>说明缺失原因 + 采用插补 / 敏感度分析</td>
      <td>缺失非随机 → 偏倚</td>
      <td>未解释缺失 &gt; 5% ⇒ **No**</td></tr>
  </tbody>
</table>
</div>

> **一票否决原则** 带 “★” 的 SQ 若回答 **No**，该域直接标为 **High Risk**。 若所有 SQ = Yes ⇒ **Low Risk**； 否则 **Unclear**。  

---

#### 使用说明
1. 表格中行内 `<br>` 换行已测试通过 GitHub Markdown 渲染。  
2. 如需在本地查看，可直接打开 `.md` 文件或使用 VS Code + *Markdown Preview Enhanced*。  
3. 若需打印版，可把 README 导入 **Typora / Obsidian → PDF**，所有列宽会随 `<colgroup>` 固定而对齐。

---

<a id="sec6"></a>
## 6 │ 评分算法

```mermaid
flowchart TD
    %% ---------- Layer 1 : Signalling Questions ----------
    SQ["信号问题层<br/>(19 项 SQ)"]
    class SQ sqStyle;

    %% ---------- Layer 2 : QUADAS-2 Domains ----------
    D["域层<br/>(D1–D4)"]
    class D domStyle;

    %% ---------- Layer 3 : Study-level Outputs ----------
    subgraph Study["研究层"]
        LR["LowRisk_Count"]
        S4["Overall_Score4"]
        RL["Overall_RiskLevel"]
    end
    class Study studyStyle;

    %% ----------- Edges -----------
    SQ  -->|"规则汇总"| D
    D   --> LR
    LR  --> S4
    D   --> RL

    %% ----------- Custom Styles -----------
    classDef sqStyle   fill:#E9F5FF,stroke:#0B83D9,color:#0B83D9,stroke-width:1px;
    classDef domStyle  fill:#FFF9DB,stroke:#F7A100,color:#F7A100,stroke-width:1px;
    classDef studyStyle fill:#E8F7E0,stroke:#26933B,color:#26933B,stroke-width:1px;
