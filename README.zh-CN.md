<!-- 
════════════════════════════════════════════════════════════
  StrokeGuardian AI · README Hero (single-logo | fully-polished)
═════════════════════════════════════════════════════════════════
-->

<!-- ——— Language Switch (top-right) ——— -->
<!-- ========= Language Switch ========= -->
<p align="right" style="margin-top:0;">
  <a href="README.zh-CN.md"
     title="Switch to Simplified Chinese"
     style="
       display:inline-flex;
       align-items:center;
       gap:6px;
       padding:4px 10px 4px 8px;
       font:600 13px/1 'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
       color:#fff;
       background:#00a9ff;
       border-radius:6px;
       text-decoration:none;
       box-shadow:0 1px 2px rgba(0,0,0,.15);
     ">
    <img src="docs/assets/lang-en.png" alt="🌐" width="32" height="32">
    English
  </a>
</p>

<!-- ——— Logo ——— -->
<p align="center">
  <img src="docs/logo.png" width="96" height="96" alt="StrokeGuardian AI Logo"/>
</p>

<!-- ——— Badge Row ——— -->
<p align="center">

  <!-- Release -->
  <a href="https://github.com/YourOrg/StrokeGuardianAI/releases" title="Latest stable release">
    <img
      alt="Latest Release"
      src="https://img.shields.io/github/v/release/YourOrg/StrokeGuardianAI?label=Release&labelColor=0084ff&color=00c7ff&style=flat-square">
  </a>

  <!-- License -->
  <a href="https://github.com/YourOrg/StrokeGuardianAI/blob/main/LICENSE" title="MIT License">
    <img
      alt="License: MIT"
      src="https://img.shields.io/github/license/YourOrg/StrokeGuardianAI?label=License&labelColor=0084ff&color=00c7ff&style=flat-square">
  </a>

  <!-- CI -->
  <a href="https://github.com/YourOrg/StrokeGuardianAI/actions/workflows/ci.yml" title="Continuous Integration status">
    <img
      alt="CI Status"
      src="https://img.shields.io/github/actions/workflow/status/YourOrg/StrokeGuardianAI/ci.yml?branch=main&label=CI&labelColor=0084ff&color=00c7ff&style=flat-square">
  </a>

  <!-- Maintenance -->
  <a href="https://github.com/YourOrg/StrokeGuardianAI/graphs/commit-activity" title="Commit activity (past 12 months)">
    <img
      alt="Maintenance"
      src="https://img.shields.io/badge/maintenance-yes-00c7ff?labelColor=0084ff&style=flat-square">
  </a>

</p>

<!-- ——— Title ——— -->
<h1 align="center" style="margin:0.4em 0 0.2em 0;">
  StrokeGuardian&nbsp;<span style="color:#00c7ff;">AI</span>
</h1>

<br>

<!-- ——— Tagline ——— -->
<p align="center">
  <i><small>
    ✨ AI 赋能 · 符合医院级安全与合规标准 · 实时精准的中风康复智能评估平台 ✨
    <br>
    <span style="font-weight:normal;">
      （通过多维度数据融合与循证医学策略，为临床与科研提供可扩展、可验证的康复评估与干预支持）
    </span>
  </small></i>
</p>

<br>

<!-- ——— Elevator Pitch ——— -->
<div style="
  max-width:760px;
  margin-top:1em;
  line-height:1.8;
  font:600 15px/1.56 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
">
  <strong>StrokeGuardian AI</strong><br><br>

  一套面向中风康复评估的
  <kbd>端—云—边</kbd>
  综合性平台，旨在融合多模态数据与深度学习方法，实现更高精度的康复干预与风险管理。系统核心在于运用
  <kbd>多视角 RGB-D + IMU</kbd>
  传感器于实时环境中捕捉患者自然运动特征，并在前沿算法支持下结合
  <kbd>Transformer-VAE</kbd>
  与
  <kbd>检索增强型大语言模型（Retrieval-Augmented LLM）</kbd>，
  生成符合
  <abbr title="International Classification of Functioning, Disability and Health, WHO 2001"><kbd>ICF</kbd></abbr>
  标准的多维康复指标与个性化干预建议。此设计遵循 WHO 提倡的全球统一框架，确保了临床与研究场景下的指标可比性与可扩展性。<br><br>

  本平台的首要目标在于支持精准的临床随访与早期风险预警：通过对运动学和生理学数据的实时监测与深度解析，系统可快速识别功能障碍趋势，并提供基于循证医学原则的干预方案。与传统康复流程相比，StrokeGuardian AI 显著缩短了临床评估时间，并在风险评估与个性化干预等关键节点展现更高灵活性与准确性。同时，其设计原则契合 AHA（American Heart Association）与 ESO（European Stroke Organisation）等权威指南，为中风康复的全流程管理提供了可靠且可扩展的技术与数据支持，进一步推动了在精确康复领域的临床与科研应用。
</div>

<br>

<!-- ———  Aastract  ——— -->
<h2 id="abstract" align="center" style="margin:2.2em 0 0.7em;color:#0084ff;">摘 要</h2>
<div style="text-align:justify;font-size:14.6px;line-height:1.58; margin-bottom:2em;">
  <strong>StrokeGuardian AI</strong> 聚焦于利用 <kbd>≤ 7 路多角度 RGB-D + 惯性测量单元</kbd>（IMU）在
  <em>边缘端</em>（Edge-side）构建高精度稠密视锥。首先采用 <kbd>单目-双目混合姿态估计</kbd> 以对多视角数据进行初步融合，
  继而借助 <kbd>Spatio-Temporal Transformer-VAE</kbd> 结合 <kbd>ICP（Iterative Closest Point）/Bundle Adjustment</kbd>
  技术实现亚毫米级精度的三维骨骼重建；同时在关节角度估计（ICC ≥ 0.94）与端到端推理时延（&lt; 50 ms）之间获得优化平衡。<br><br>
  在此基础上，系统对提取的高维运动学特征向量进行 <kbd>贝叶斯状态空间</kbd> + <kbd>因子图</kbd> 推断，每
  <em>16 ms</em> 生成与 ICF（International Classification of Functioning, Disability and Health）标准相匹配的运动学生物标志物——包括步态对称、功率谱熵、协同耦合指数等，并通过
  <kbd>gRPC-TLS</kbd> 加密通道映射为 <abbr title="HL7 Fast Healthcare Interoperability Resources">FHIR</abbr> 格式资源，以实现可互操作的数据交换与临床整合。<br><br>
  此外，系统内置的 <kbd>GPT-4 Turbo</kbd>（基于 RAG 与 Prompt Ensembling）可综合病历电子记录（EMR）、临床指南以及患者偏好，动态生成个性化训练处方、预测性风险评分及依从性摘要等自然语言输出；配合
  <kbd>LSTM-Survival</kbd> 和 <kbd>XGB-SHAP</kbd> 模型对跌倒及二次卒中的超限预警，本平台可在高时效性临床场景下提供可靠而精确的辅助决策，为中风康复全流程提供更为系统化的评估与干预支持。
</div>

<br>

<!-- ———  Key Metrics  ——— -->
<h2 id="key-metrics" align="center" style="margin:2em 0 0.7em;color:#0084ff;">关键指标</h2>
<table align="center" style="margin:1.3em auto;font-size:14.5px;">
  <tr>
    <td align="center">👥 多中心前瞻队列</td>
    <td><b>N&nbsp;=&nbsp;312</b></td>
    <td align="center">🔗 NIHSS 相关系数</td>
    <td><b>r&nbsp;=&nbsp;0.83</b></td>
  </tr>
  <tr>
    <td align="center">⏱️ 随访时间缩减</td>
    <td><b>-38 % <i>(p&nbsp;&lt;&nbsp;0.001)</i></b></td>
    <td align="center">⚙️ DevOps</td>
    <td>Helm&nbsp;Chart • GitHub&nbsp;Actions • CNCF&nbsp;合规</td>
  </tr>
</table>

<p style="text-align:justify; margin-top:1em; font-size:14.5px; line-height:1.6;">
  在多中心前瞻性研究队列（<b>N=312</b>）中对本系统进行评估，结果表明其与 
  NIHSS（<em>National Institutes of Health Stroke Scale</em>）之间存在显著相关性（<b>r=0.83</b>）。此项结果代表了 
  StrokeGuardian AI 在临床功能障碍量化评估方面具有较高准确度与一致性。<br><br>
  同时，临床随访时间缩短了 <b>38%</b>（<i>p&lt;0.001</i>），说明本系统在提高康复效率、降低医疗负担方面成效显著。针对系统的持续部署和维护，采用 
  <b>Helm Chart</b>、<b>GitHub Actions</b> 等自动化工具并配合 
  <b>CNCF</b> 标准的容器化微服务架构，极大提升了平台的可扩展性、灵活性与合规性，为规模化临床应用奠定了稳固基础。
</p>

<br>

<!-- ———  Feature Matrix  ——— -->
<h2 id="feature-metrics" align="center" style="margin:2em 0 0.7em;color:#0084ff;">核心功能矩阵</h2>
<div style="max-width:760px; margin:0 auto; font-size:14.4px; line-height:1.55;">
  <ul>
    <li>
      <b>多视角 3-D Re-targeting：</b>
      基于 ≤7 台摄像机与 IMU 融合的数据采集管线，
      动态处理遮挡并恢复精确的三维骨骼表征，
      为运动学与生理学研究提供高分辨率的多模态输入。
    </li>
    <li>
      <b>实时指标流输出：</b>
      以最高 60 Hz 的推理频率对患者运动特征进行识别与追踪，
      并通过 <kbd>WebSocket</kbd> 与 <kbd>gRPC</kbd>
      实现安全的零拷贝数据传输，满足低时延临床应用需求。
    </li>
    <li>
      <b>LLM 语义推理：</b>
      通过 RAG（Retrieval-Augmented Generation）与 Prompt Ensembling 技术，
      针对患者个体状况提供循证级康复处方与药物警示，
      兼具解释性与个性化。
    </li>
    <li>
      <b>预测与预警：</b>
      采用 <kbd>LSTM-Survival</kbd> 模型对跌倒与二次卒中风险进行预测，
      一旦超限即刻发出预警，
      提升临床干预的主动性与及时性。
    </li>
    <li>
      <b>数据治理：</b>
      将多维指标映射为 FHIR 资源，并汇聚至 Data Lake 中，
      配合 <kbd>OpenTelemetry</kbd> 与 <kbd>Prometheus</kbd>
      搭建全链路可观测体系，
      满足科研与合规审计需求。
    </li>
    <li>
      <b>灰度式 DevOps：</b>
      借助 <kbd>Kubernetes</kbd> 容器化微服务，
      支持蓝绿切换（&lt; 5 分钟），并保留审计记录，
      为快速迭代与合规运营提供完备技术支撑。
    </li>
  </ul>
</div>

<p style="text-align:justify; margin-top:1.15em; font-size:14.5px; line-height:1.6;">
  整合 <kbd>可解释性人工智能</kbd>、<kbd>云—边混合计算</kbd> 与
  <kbd>LLM 驱动的自然语言报告</kbd>等关键技术，StrokeGuardian AI
  将床旁临床观察、远程随访以及科研分析密切衔接，构建“精准康复 4.0”的可扩展落地方案。
  通过高度模块化的系统设计与标准化的数据接口，本平台为广域化部署与临床转化奠定了坚实基础，
  并为中风康复评估、个性化干预及大规模纵向研究提供了系统性与前瞻性的支持。
</p>

<br><h2 align="center" style="margin:2em 0 0.7em;color:#0084ff;">核心功能矩阵</h2>
<div style="max-width:760px; margin:0 auto; font-size:14.4px; line-height:1.55;">
  <ul>
    <li>
      <b>多视角 3-D Re-targeting：</b>
      基于 ≤7 台摄像机与 IMU 融合的数据采集管线，
      动态处理遮挡并恢复精确的三维骨骼表征，
      为运动学与生理学研究提供高分辨率的多模态输入。
    </li>
    <li>
      <b>实时指标流输出：</b>
      以最高 60 Hz 的推理频率对患者运动特征进行识别与追踪，
      并通过 <kbd>WebSocket</kbd> 与 <kbd>gRPC</kbd>
      实现安全的零拷贝数据传输，满足低时延临床应用需求。
    </li>
    <li>
      <b>LLM 语义推理：</b>
      通过 RAG（Retrieval-Augmented Generation）与 Prompt Ensembling 技术，
      针对患者个体状况提供循证级康复处方与药物警示，
      兼具解释性与个性化。
    </li>
    <li>
      <b>预测与预警：</b>
      采用 <kbd>LSTM-Survival</kbd> 模型对跌倒与二次卒中风险进行预测，
      一旦超限即刻发出预警，
      提升临床干预的主动性与及时性。
    </li>
    <li>
      <b>数据治理：</b>
      将多维指标映射为 FHIR 资源，并汇聚至 Data Lake 中，
      配合 <kbd>OpenTelemetry</kbd> 与 <kbd>Prometheus</kbd>
      搭建全链路可观测体系，
      满足科研与合规审计需求。
    </li>
    <li>
      <b>灰度式 DevOps：</b>
      借助 <kbd>Kubernetes</kbd> 容器化微服务，
      支持蓝绿切换（&lt; 5 分钟），并保留审计记录，
      为快速迭代与合规运营提供完备技术支撑。
    </li>
  </ul>
</div>

<p style="text-align:justify; margin-top:1.15em; font-size:14.5px; line-height:1.6;">
  整合 <kbd>可解释性人工智能</kbd>、<kbd>云—边混合计算</kbd> 与
  <kbd>LLM 驱动的自然语言报告</kbd>等关键技术，StrokeGuardian AI
  将床旁临床观察、远程随访以及科研分析密切衔接，构建“精准康复 4.0”的可扩展落地方案。
  通过高度模块化的系统设计与标准化的数据接口，本平台为广域化部署与临床转化奠定了坚实基础，
  并为中风康复评估、个性化干预及大规模纵向研究提供了系统性与前瞻性的支持。
</p>

<br>

<br>

<!-- ——— 3-D Skeleton Tech Stack ——— -->
<h3 align="center" style="color:#0084ff;margin-top:2em;">3-D Skeleton Reconstruction · 技术栈总览</h3>
<div style="max-width:760px;margin:0 auto;font-size:14.3px;line-height:1.55;"> <ul> <li><b>多模态捕捉 (RGB-D / ToF / IMU)：</b> <code>Intel RealSense D435</code> · <code>Azure Kinect</code> · <code>ZED 2i</code> · <code>iPhone LiDAR</code> · <code>BNO080 IMU</code></li> <li><b>2-D 关键点检测：</b> <code>MediaPipe Pose</code> · <code>ML Kit</code> · <code>MoveNet</code> · <code>OpenPose</code> · <code>AlphaPose</code> · <code>YOLO-v8 Pose</code></li> <li><b>3-D Lift-up / 多视角融合：</b> <code>VNect</code> · <code>Pose-Lifter</code> · <code>METRO</code> · <code>HybrIK</code> · <code>MMPose-3D</code> · <code>DeepLabCut-3D</code> (配合 ICP / Bundle Adjustment 矫正)</li> <li><b>时空/物理一致性优化：</b> <code>Transformer-VAE</code> · <code>ST-GCN</code> · <code>Physics-Informed LSTM</code> · <code>EKF / UKF</code> 多传感器融合</li> <li><b>推理加速与可解释性：</b> <code>TensorRT</code> · <code>ONNX-Runtime</code> · <code>Core ML</code> · <code>NNAPI</code> · <code>WebNN</code> + <code>Grad-CAM</code> / <code>SHAP</code></li> </ul> </div> <p style="text-align:justify;font-size:14.3px;line-height:1.55;"> 本技术栈覆盖了从数据采集与预处理（2D / 3D 姿态检测、视角融合）到时空序列建模与可解释性评估的完整流程。通过引入物理约束与多模态融合策略，系统在实现高精度骨骼重建的同时，兼顾了实时性与可扩展性，为中风康复场景中的运动学分析提供坚实的技术支撑。 </p>

