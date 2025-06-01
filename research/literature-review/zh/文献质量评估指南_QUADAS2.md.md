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
- **Sheet `QUADAS2`** 域级/研究级聚合结论  
- **Sheet `QUADAS2_Items`** 19 条信号问题逐项打分与证据链  

> **自下而上链式更新**  
> `SQ → Domain → Study` 任何上游改动必须向下游自动同步，保证可追溯性与机读一致性。

---

<a id="sec2"></a>
## 2 │ 数据表概览

<!-- ─────────── 数据表一览 ─────────── -->
<div style="overflow-x:auto; font-size:15px; line-height:1.55">

<table>
  <colgroup>
    <col style="width:22%">
    <col style="width:28%">
    <col style="width:20%">
    <col style="width:30%">
  </colgroup>
  <thead>
    <tr style="background:#f8f8f8">
      <th>Sheet 名称</th>
      <th>功能定位</th>
      <th>记录粒度</th>
      <th>核心字段 / 输出</th>
    </tr>
  </thead>

  <tbody>
  <tr>
    <td><strong>03_QUADAS2</strong></td>
    <td>域级 &amp; 研究级<br/>结论汇总表</td>
    <td><u>1 行 = 1 篇文献</u><br/>(Study&nbsp;Level)</td>
    <td>
      • <b>D1–D4_Risk / App</b><br/>
      • <b>Overall_RiskLevel</b><br/>
      • <b>Overall_Score4</b>（0-4）<br/>
      • <b>Core40_Flag</b>（核心样本池）<br/>
      • 专家备注 <code>Notes_Expert</code>
    </td>
  </tr>

  <tr>
    <td><strong>03_QUADAS2_Items</strong></td>
    <td>信号问题原始打分<br/>与佐证追溯表</td>
    <td><u>1 行 = 1 个 SQ</u><br/>(19 × SQ per study)</td>
    <td>
      • 19 × <b>Y / N / U</b> 共识答案<br/>
      • <b>Justification</b> &amp; <b>Source_Evidence</b><br/>
      • 自动派生 <code>Risk_Flag</code>、<code>Applicability_Flag</code><br/>
      • <code>Direction_of_Bias</code>（+ / – / ?）<br/>
      • <code>Auto_QC_Status</code>（pass / warn / fail）
    </td>
  </tr>
  </tbody>
</table>
</div>

%% ======================================
%%   QUADAS-2 数据处理三步流 · 总览示意图
%%   —— 通过 “SPACER” 隐形节点留白
%% ======================================
```mermaid
flowchart LR
    %% ---------- Step 1 ----------
    subgraph STEP1["步骤 1｜原始打分<br/>(03_QUADAS2_Items)"]
        direction TB
        SP1[" "]        %% 占位留白
        A["19 × 信号问题<br/>(每篇文献)"]
    end

    %% ---------- Step 2 ----------
    subgraph STEP2["步骤 2｜域级聚合<br/>(D1–D4)"]
        direction TB
        SP2[" "]
        B["域级结果<br/>Risk + Applicability"]
    end

    %% ---------- Step 3 ----------
    subgraph STEP3["步骤 3｜研究级输出<br/>(03_QUADAS2)"]
        direction TB
        SP3[" "]
        C1["LowRisk_Count"]
        C2["Overall_Score4"]
        C3["Overall_RiskLevel"]
        C4["Core40_Flag"]
    end

    %% ----------- Edges -----------
    A -- 规则映射 --> B
    B --> C1
    C1 --> C2
    B --> C3
    C3 --> C4

    %% ----------- Styles ----------
    classDef box   fill:#E8F7F0,stroke:#2CA58D,color:#145A32,stroke-width:1px;
    classDef blank fill:none,stroke:none;
    class A,B,C1,C2,C3,C4 box;
    class SP1,SP2,SP3 blank;
```

---

<a id="sec3"></a>
## 3 │ 字段定义 — `QUADAS2`（研究级）

<!-- ─────────── 字段定义表：03_QUADAS2 ─────────── -->
<div style="overflow-x:auto; font-size:15px; line-height:1.55">

<table>
  <colgroup>  <!-- 4 列均 25 %；在窄屏自动横向滚动 -->
    <col style="width:25%">
    <col style="width:18%">
    <col style="width:22%">
    <col style="width:35%">
  </colgroup>
  <thead>
    <tr style="background:#f8f8f8">
      <th>字段</th>
      <th>数据类型<br/>(SQL / CSV)</th>
      <th>合法值 / 格式</th>
      <th>专业释义 &nbsp;|&nbsp; 运用示例</th>
    </tr>
  </thead>
  <tbody>

  <tr><td><strong>Study_ID</strong></td>
      <td>VARCHAR(30)</td>
      <td><code>Smith23</code> / <code>ABC-23</code></td>
      <td>主键；必须与 *01_Studies* 的 <code>Study_ID</code> 一致，用作多表连接。</td></tr>

  <tr><td><strong>Author_Year</strong></td>
      <td>VARCHAR(40)</td>
      <td><code>Smith&nbsp;(2023)</code></td>
      <td>可读格式，方便人工检索；机器分析仍以 <code>Study_ID</code> 为准。</td></tr>

  <tr><td><strong>Reviewer_1</strong><br/><strong>Reviewer_2</strong></td>
      <td>VARCHAR(10)</td>
      <td>姓名缩写</td>
      <td>首轮双盲打分者；如团队 <abbr title="表示审稿人">LHZ</abbr>、<abbr title="表示审稿人">MXW</abbr>。</td></tr>

  <tr><td><strong>Consensus_Date</strong></td>
      <td>DATE</td>
      <td><code>YYYY-MM-DD</code></td>
      <td>四域均达成共识的日期；供追溯与版本控制。</td></tr>

  <!-- ──── 域级风险 ──── -->
  <tr style="background:#fafafa"><td colspan="4"><em><strong>域级风险（RoB）与适用性（Concern）</strong></em></td></tr>

  <tr><td><strong>D1_Risk</strong> – <strong>D4_Risk</strong></td>
      <td>ENUM<br/>(L,H,U)</td>
      <td>L = 低风险<br/>H = 高风险<br/>U = 不确定</td>
      <td>四域：<u>Patient Selection</u>、<u>Index Test</u>、<u>Reference Standard</u>、<u>Flow & Timing</u>。</td></tr>

  <tr><td><strong>D1_App</strong> – <strong>D4_App</strong></td>
      <td>ENUM<br/>(L,H,U)</td>
      <td>同左</td>
      <td>面向本综述 PICO 的适用性 Concern；判规则见 QUADAS-2 官方指南。</td></tr>

  <!-- ──── 全局指标 ──── -->
  <tr style="background:#fafafa"><td colspan="4"><em><strong>全局指标</strong></em></td></tr>

  <tr><td><strong>LowRisk_Count</strong></td>
      <td>INT</td>
      <td>0 – 4</td>
      <td>四个域中标记为 <code>L</code> 的计数；可直接驱动热图色阶。</td></tr>

  <tr><td><strong>Overall_RiskLevel</strong></td>
      <td>ENUM<br/>(Low,Moderate,High)</td>
      <td>—</td>
      <td><u>自动计算</u>：<br/>
          - 若任一域 = H → <b>High</b><br/>
          - 全域 = L → <b>Low</b><br/>
          - 其余 → <b>Moderate</b></td></tr>

  <tr><td><strong>Overall_Score4</strong></td>
      <td>INT</td>
      <td>0 – 4</td>
      <td><code>LowRisk_Count</code> 的复制列，便于森林图等可视化调色。</td></tr>

  <tr><td><strong>Core40_Flag</strong></td>
      <td>BOOLEAN</td>
      <td>1 / 0</td>
      <td>置 1 = 进入“核心 40”篇深度定量分析池。</td></tr>

  <tr><td><strong>Consensus?</strong></td>
      <td>BOOLEAN</td>
      <td>Y / N</td>
      <td>四域 Risk 与 Applicability 是否已由评审者确认无分歧。</td></tr>

  <tr><td><strong>Notes_Expert</strong></td>
      <td>TEXT</td>
      <td>—</td>
      <td>院士级/领域专家的附加评注或补救策略（可多行使用 &lt;br/&gt;）。</td></tr>

  </tbody>
</table>
</div>

<details>
<summary><strong>使用注意（点开查看）</strong></summary>

| 场景 | 建议做法 |
|------|---------|
| **批量录入** | 推荐用 *CSV (UTF-8)*；日期字段保持 ISO-8601。 |
| **可视化** | 直接读取 <code>Overall_RiskLevel</code> ；<br/>或以 <code>Overall_Score4</code>/<code>LowRisk_Count</code> 映射 0–4 色标。 |
| **CI/CD 校验** | 修改任何域级字段后自动更新 <code>Consensus_Date</code> 与 <code>Consensus?</code> status。 |
| **API 调用** | 以 <code>Study_ID</code> 为主键，可快速 JOIN *03_QUADAS2_Items* 与 *01_Studies* 以重建全表。 |
</details>

---

<a id="sec4"></a>
## 4 │ 字段定义 — `QUADAS2_Items`（信号问题级）

<!-- ─────────── 字段定义表：03_QUADAS2_Items ─────────── -->
<div style="overflow-x:auto; font-size:15px; line-height:1.55">

<table>
  <colgroup>  <!-- 统一列宽：4 × 25 % -->
    <col style="width:22%">
    <col style="width:13%">
    <col style="width:25%">
    <col style="width:40%">
  </colgroup>
  <thead>
    <tr style="background:#fafafa">
      <th>字段</th>
      <th>数据类型<br/>(SQL / CSV)</th>
      <th>合法值 / 格式</th>
      <th>专业释义 &nbsp;|&nbsp; 示例说明</th>
    </tr>
  </thead>
  <tbody>

  <tr><td><strong>Study_ID</strong></td>
      <td>VARCHAR(30)</td>
      <td>如 <code>Smith24</code></td>
      <td>连接主键，唯一标识单篇文献；与 *01_Studies* 中 <code>Study_ID</code> 完全一致。</td></tr>

  <tr><td><strong>D1_SQ1</strong> – <strong>D4_SQ5</strong></td>
      <td>ENUM<br/>(Y,N,U)</td>
      <td>Y = 是<br/>N = 否<br/>U = 不确定</td>
      <td>19 个信号问题最终 <u>共识</u> 答案。字段顺序 = 表&nbsp;§5 的 SQ 顺序。</td></tr>

  <tr><td><strong>Response_R1</strong><br/><strong>Response_R2</strong></td>
      <td>ENUM<br/>(Y,N,U)</td>
      <td>同上</td>
      <td>两名评审者首次独立打分的原始结果（用于偏差检测）。</td></tr>

  <tr><td><strong>Consensus</strong></td>
      <td>ENUM<br/>(Y,N,U)</td>
      <td>见上</td>
      <td>若 <code>Response_R1</code> ≠ <code>Response_R2</code>，由第三评审裁决的最终值；否则复制原值。</td></tr>

  <tr><td><strong>Justification</strong></td>
      <td>TEXT</td>
      <td>50–250 字</td>
      <td>引用原文关键句、表格或统计量以支持“是/否”判断；鼓励同时提供英/中简释。</td></tr>

  <tr><td><strong>Source_Evidence</strong></td>
      <td>VARCHAR(40)</td>
      <td>如 <code>p 7, Fig 2</code></td>
      <td>定位信息：页码、图表编号、附录，或 PDF 行号 (例如 <code>L256-L270</code>)。</td></tr>

  <tr><td><strong>Risk_Flag</strong></td>
      <td>ENUM<br/>(L,H,U)</td>
      <td>L 低风险<br/>H 高风险<br/>U 不确定</td>
      <td>由脚本根据 19 个 SQ 自动汇总至“域-级风险”（见 §6 算法）。</td></tr>

  <tr><td><strong>Applicability_Flag</strong></td>
      <td>ENUM<br/>(L,H,U)</td>
      <td>同上</td>
      <td>同理自动汇总“域-级适用性 Concern”。</td></tr>

  <tr><td><strong>Last_Update</strong></td>
      <td>DATE</td>
      <td><code>YYYY-MM-DD</code></td>
      <td>任何字段被修改时必须刷新；CI/CD 可用作增量校验触发器。</td></tr>

  <tr><td><strong>AI_Specific?</strong></td>
      <td>BOOLEAN</td>
      <td>1 / 0</td>
      <td>仅对 AI 独有的 SQ（目前 <code>D2_SQ3</code>, <code>D2_SQ4</code>）标记为 <code>1</code>；其余默认 0。</td></tr>

  <tr><td><strong>Direction_of_Bias</strong></td>
      <td>ENUM<br/>(+,&nbsp;−,&nbsp;?)</td>
      <td>+ 高估<br/>− 低估<br/>? 未知</td>
      <td>当 <code>Risk_Flag = H</code> 时填写：<br/>+ = 可能夸大诊断性能；− = 可能保守。</td></tr>

  <tr><td><strong>Auto_QC_Status</strong></td>
      <td>ENUM<br/>(pass,warn,fail)</td>
      <td>—</td>
      <td>自动脚本校验结果：<br/><code>pass</code> = 全部合法值；<br/><code>warn</code> = 非关键字段留空；<br/><code>fail</code> = 枚举越界或逻辑冲突。</td></tr>

  </tbody>
</table>
</div>

<details>
<summary><strong>字段补充说明（点开查看）</strong></summary>

* **ENUM 值区分大小写**：<code>Y</code>/<code>N</code>/<code>U</code> 均需大写，便于 R/Python 解析。  
* **文本字段避免换行**：若需多行，使用 <code>&lt;br/&gt;</code> 以兼容 CSV→HTML 转换。  
* **方向性 (+/−)**：参考 **Whiting et al. 2021** 推荐做法：若无法推断方向请填 “?”，而非留空。  
* **Auto_QC 触发器示例**  
  ```python
  if field not in {'Y','N','U'}:
      qc = 'fail'
  elif justification == '' and field != 'U':
      qc = 'warn'
  else:
      qc = 'pass'
</details>

---

<a id="sec5"></a>
## 5 │ 19 项信号问题与可操作判据

<!-- ───────────── 19 项信号问题速查表（中文完整版）───────────── -->
<div style="overflow-x:auto; font-size:15px; line-height:1.55">

<table>
  <colgroup>
    <col style="width: 6%">
    <col style="width: 4%">
    <col style="width: 28%">
    <col style="width: 30%">
    <col style="width: 32%">
  </colgroup>
  <thead>
    <tr style="background:#fafafa">
      <th>域</th>
      <th>SQ</th>
      <th>完整信号问题</th>
      <th><strong>判定为 “是” 的操作标准</strong></th>
      <th>常见错误示例 / 潜在偏倚</th>
    </tr>
  </thead>
  <tbody>

  <!-- ────────── D1 Patient Selection ────────── -->
  <tr><td rowspan="5"><strong>D1<br>受试者选择</strong></td>
      <td>1★</td>
      <td>研究是否采用<strong>连续或随机抽样</strong>，而非便利样本？</td>
      <td>方案或流程图注明 “consecutive” 或使用随机号码 / 电子随机化。<br>招募时间段完整，无择优挑选。</td>
      <td>便利采样、门诊排队取样 → 人群代表性不足，可能漏掉重症或合并症患者。</td></tr>

  <tr><td>2★</td>
      <td>研究是否<strong>避免病例-对照设计</strong>？</td>
      <td>采用前瞻性或回顾性队列 / 交叉设计。<br>若为诊断准确度研究，不使用“病例组 vs 对照组”明确定义。</td>
      <td>病例-对照设计常导致光谱偏倚，AUC 被人为抬高。</td></tr>

  <tr><td>3★</td>
      <td>研究是否<strong>避免不当排除</strong>受试者？</td>
      <td>给出全部排除理由；排除比例&nbsp;&lt;&nbsp;10 %。</td>
      <td>排除“步态异常严重”或 “影像质量差” 未说明原因 → 选择性偏倚。</td></tr>

  <tr><td>4</td>
      <td>研究样本中是否<strong>包含不少于 10 例脑卒中患者</strong>？</td>
      <td>主文或附录报告脑卒中受试者数量&nbsp;≥ 10。</td>
      <td>样本过少 → 准确度估计置信区间极宽；无法稳定评估。</td></tr>

  <tr><td>5</td>
      <td>研究的<strong>病程阶段（急性 / 亚急性 / 慢性）分布</strong>是否均衡？</td>
      <td>各阶段比例均 &lt; 70 %，或作者提供分层分析。</td>
      <td>若 90 % 均为慢性样本，却用来推断急性护理场景 → 适用性存疑。</td></tr>

  <!-- ────────── D2 Index Test ────────── -->
  <tr><td rowspan="4"><strong>D2<br>索引试验</strong></td>
      <td>1★</td>
      <td>索引试验结果解释时是否<strong>对参考标准保持盲法</strong>？</td>
      <td>人工标注者或算法评估流程在得出索引结果时<strong>未访问</strong>任何参考标准信息。</td>
      <td>同一研究人员先读 GAITRite 再调整 AI 输出 → 解盲偏倚。</td></tr>

  <tr><td>2</td>
      <td>是否在模型开发阶段就<strong>预先设定阈值 / 权重</strong>？</td>
      <td>研究方案 / 注册信息中已写明阈值，或使用临床公认 cut-off。</td>
      <td>事后基于“最大 Youden 指数”调阈 → 浮夸敏感度和特异度。</td></tr>

  <tr><td>3</td>
      <td>在外部验证前，最终模型是否<strong>已冻结</strong>且未再调整参数？</td>
      <td>文中用词 “model frozen / locked weights”；外部测试仅 forward 推理。</td>
      <td>在外部集上继续微调（fine-tune）→ 数据泄漏。</td></tr>

  <tr><td>4★</td>
      <td>整个分析流程中是否<strong>完全避免数据泄漏</strong>？</td>
      <td>Train / Validation / Test 样本<strong>患者级</strong>互斥；无时间穿越；预处理统计量仅基于训练集。</td>
      <td>同一受试者多次步态循环跨库；全数据归一化用全局均值 → 高估性能。</td></tr>

  <!-- ────────── D3 Reference Standard ────────── -->
  <tr><td rowspan="4"><strong>D3<br>参考标准</strong></td>
      <td>1★</td>
      <td>所用参考标准是否<strong>准确且被公认</strong>为金标准？</td>
      <td>3D 光学动作捕捉（Vicon/Qualisys）、GAITRite 压力板、临床影像确诊等。</td>
      <td>仅用智能手机计步器作“真值” → 参考标准不足。</td></tr>

  <tr><td>2</td>
      <td>参考标准评估人员是否<strong>对索引试验结果保持盲法</strong>？</td>
      <td>文中声明“双盲”或 “assessor masked”。</td>
      <td>参考评估者同时参与索引算法开发 → 评估偏倚。</td></tr>

  <tr><td>3</td>
      <td>是否使用<strong>三维光学动作捕捉 / GAITRite / 力平台</strong>等金标准装置？</td>
      <td>设备型号、采样率、标定流程写入方法学。</td>
      <td>自制 Markless 系统当作参考 → 不合格。</td></tr>

  <tr><td>4</td>
      <td>是否<strong>充分报告参考标准的精度或误差</strong>？</td>
      <td>提供 ICC、SEM、RMSE 等至少一项；或引用制造商精度。</td>
      <td>无任何精度信息 → 难以评估误差传播。</td></tr>

  <!-- ────────── D4 Flow & Timing ────────── -->
  <tr><td rowspan="5"><strong>D4<br>流程与时序</strong></td>
      <td>1</td>
      <td>索引试验与参考标准之间的<strong>时间间隔</strong>是否合理？</td>
      <td>室内同日 ≤ 30 min；若跨机构，≤ 24 h 且注明病情稳定。</td>
      <td>长间隔内功能恢复或疲劳改变真实步态。</td></tr>

  <tr><td>2</td>
      <td>是否<strong>所有受试者</strong>都接受了参考标准？</td>
      <td><code>参考标准样本数 / 总样本数 ≥ 95 %</code><br>或明确说明缺失原因。</td>
      <td>只对部分受试者进行金标准测量 → 核实偏倚 (verification bias)。</td></tr>

  <tr><td>3</td>
      <td>是否对<strong>全部入组受试者</strong>进行了完整分析？</td>
      <td>采用 ITT 或报告排除清单并做敏感度分析。</td>
      <td>排除“测试失败”个案 → 人为抬高模型性能。</td></tr>

  <tr><td>4★</td>
      <td>索引试验与参考标准是否<strong>同步采集</strong>？</td>
      <td>同一趟步行同时布控；或硬件触发信号同步。</td>
      <td>两趟步行顺序固定且无随机化 → 顺序效应偏倚。</td></tr>

  <tr><td>5</td>
      <td>是否对<strong>缺失数据</strong>进行了透明报告并说明处理方法？</td>
      <td>缺失比例、原因及插补 / 敏感度策略写入结果。</td>
      <td>缺失占比高且“按可用数据分析” → 偏倚方向未知。</td></tr>
  </tbody>
</table>
</div>

> ★ 一票否决：带 ★ 项若回答 **“否”**，该域立即判定为 **高偏倚风险**；  
> **全部 “是”** → 低风险；其余 → 不确定风险。

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
