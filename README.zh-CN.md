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

<!-- ——— Title & Tagline ——— -->
<h1 align="center" style="margin:0.4em 0 0.2em 0;">
  StrokeGuardian&nbsp;<span style="color:#00c7ff;">AI</span>
</h1>

<p align="center">
  <i><small>✨ AI 赋能 · 医 院 级 · 实 时 精 准 的 中 风 康 复 智 能 评 估 平 台 ✨</small></i>
</p>

<!-- ——— Elevator Pitch ——— -->
<div style="
  max-width:760px;
  margin-top:1em;
  font:600 15px/1.8 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height:1.8;
">
  <strong>StrokeGuardian AI</strong><br><br>
  — 一套面向中风康复评估的综合性
  <span style="color:#e91e63; font-weight:bold;"><strong>端—云—边</strong></span>
  协同平台，旨在通过多源信息融合与深度学习方法，实现更加精准的康复干预与风险管理。系统核心在于利用
  <span style="color:#9c27b0; font-weight:bold;">多视角 RGB-D + IMU</span>
  传感器实时捕捉患者的自然运动数据，并通过结合
  <span style="color:#2196f3; font-weight:bold;">Transformer-VAE</span>
  与
  <span style="color:#009688; font-weight:bold;">检索增强型大语言模型</span>
  （Retrieval-Augmented LLM）的前沿技术框架，生成符合
  <abbr title="International Classification of Functioning, Disability and Health, WHO 2001">
    <span style="color:#ff5722; font-weight:bold;">ICF</span>
  </abbr>
  标准的多维康复指标序列与个性化干预建议。此设计在借助 WHO 所倡导的全球统一标准（ICF）的同时，保证了可比性与可推广性。<br><br>
  本平台的首要目标在于支持临床实践的精准随访与风险预警：通过对运动学与生理学特征的实时监测和深度分析，系统可帮助医生快速识别功能障碍趋势并生成量化指标，进而为个案提供依据充分且循证严谨的康复方案。与传统康复流程相比，StrokeGuardian AI 不仅显著缩短了临床评估周期，而且在风险评估与干预建议等关键环节上展现出更高的灵活性与准确性，为中风康复的全流程管理提供了稳健、有效且高效率的技术支撑。
</div>

<!-- ———  A B S T R A C T  ——— -->
<h2 id="摘要" align="center" style="margin:2.2em 0 0.7em;color:#0084ff;">摘 要</h2>

<div style="text-align:justify;font-size:14.6px;line-height:1.58;"> <strong>StrokeGuardian AI</strong> 致力于利用<kbd>≤ 7 路多角度 RGB-D + 惯性测量单元</kbd>（IMU）构建高精度稠密视锥。在<em>边缘端</em>，本系统首先采用<kbd>单目-双目混合姿态估计</kbd>对多视角数据进行初步处理；随后，通过<kbd>Spatio-Temporal Transformer-VAE</kbd>结合<kbd>ICP（Iterative Closest Point）/Bundle Adjustment</kbd>技术，重建亚毫米级精度的三维骨骼序列，实现关节角度估计的可靠性（ICC ≥ 0.94）及端到端推理延迟（< 50 ms）的平衡。 <br><br> 基于所获取的高维运动学特征向量，本系统采用<kbd>贝叶斯状态空间</kbd>与<kbd>因子图</kbd>推断方法，每<em>16 ms</em>即可输出与 ICF 标准相对齐的康复生物学标志物（如步态对称性、功率谱熵、协同耦合指数等），并通过<kbd>gRPC-TLS</kbd>隧道将其安全映射为<abbr title="HL7 Fast Healthcare Interoperability Resources">FHIR</abbr>格式资源。 <br><br> 此外，嵌入式<kbd>GPT-4 Turbo</kbd>（基于 RAG 与 Prompt Ensembling）可综合病历电子记录（EMR）、临床指南与患者偏好，动态生成个性化训练处方、预测性风险评分以及遵从性摘要等自然语言输出；而<kbd>LSTM-Survival</kbd>与<kbd>XGB-SHAP</kbd>模型则用于跌倒及二次卒中的超限预警，使本系统能够在时间敏感的临床场景中提供精确、可靠的决策辅助。 </div>

<!-- ———  Key Metrics  ——— -->
<table align="center" style="margin:1.3em auto;font-size:14.5px;"> <tr> <td align="center">👥&nbsp;多中心前瞻队列</td><td><b>N&nbsp;=&nbsp;312</b></td> <td align="center">🔗&nbsp;NIHSS&nbsp;相关系数</td><td><b>r&nbsp;=&nbsp;0.83</b></td> </tr> <tr> <td align="center">⏱️&nbsp;随访时间缩减</td><td><b>-38 % <i>(p&nbsp;&lt;&nbsp;0.001)</i></b></td> <td align="center">⚙️&nbsp;DevOps</td><td>Helm&nbsp;Chart • GitHub&nbsp;Actions • CNCF&nbsp;合规</td> </tr> </table> <p style="text-align:justify;margin-top:1em;font-size:14.5px;line-height:1.6;"> 通过在多中心的前瞻性研究队列中测试，本系统达成了与 NIHSS（<em>National Institutes of Health Stroke Scale</em>）评分较高的相关性（r = 0.83），并显著缩短了临床随访所需时间（-38%，p &lt; 0.001）。在 DevOps 体系中采用 Helm Chart、GitHub Actions，以及符合 CNCF 标准的容器化微服务部署模式，进一步提升了整体服务的可扩展性与可维护性。 </p>

<!-- ———  Feature Matrix  ——— -->
<<div style="max-width:760px;margin:0 auto;font-size:14.4px;line-height:1.55;"> <ul> <li><b>多视角 3-D Re-targeting：</b>基于≤7台摄像机与 IMU 的融合数据，实现动态遮挡补偿并提取精确三维骨骼信息。</li> <li><b>实时指标流输出：</b>最高 60 Hz 的推理频率，利用<kbd>WebSocket</kbd> 和 <kbd>gRPC</kbd>实现安全、零拷贝式数据传输。</li> <li><b>LLM 语义推理：</b>通过 RAG 与 Prompt Ensembling 机制，为临床提供循证级康复处方及药物警示。</li> <li><b>预测与预警：</b>应用<kbd>LSTM-Survival</kbd>预测跌倒与再次卒中的可能性，一旦超限即刻发出预警。</li> <li><b>数据治理：</b>系统将采集的多维指标映射为 FHIR 资源并汇聚至 Data Lake，配合<kbd>OpenTelemetry</kbd>与<kbd>Prometheus</kbd>实现全链路可观测性。</li> <li><b>灰度式 DevOps：</b>基于<kbd>Kubernetes</kbd>的容器化微服务部署，蓝绿切换时间低于 5 分钟，并保留合规审计记录。</li> </ul> </div> <p style="text-align:justify;margin-top:1.15em;font-size:14.5px;line-height:1.6;"> 通过整合<kbd>可解释性人工智能</kbd>、<kbd>云—边混合计算</kbd>以及<kbd>LLM 驱动的自然语言报告</kbd>，StrokeGuardian AI 有效地将床旁观察、远程随访与科研需求无缝衔接。该平台为“精准康复 4.0”提供可大规模部署与推广的技术基础与数据支撑，进一步拓宽了中风康复评估与干预的研究与实践边界。 </p>

<!-- ——— 3-D Skeleton Tech Stack ——— -->
<h3 align="center" style="color:#0084ff;margin-top:2em;">3-D Skeleton Reconstruction · 技术栈总览</h3>
<div style="max-width:760px;margin:0 auto;font-size:14.3px;line-height:1.55;"> <ul> <li><b>多模态捕捉 (RGB-D / ToF / IMU)：</b> <code>Intel RealSense D435</code> · <code>Azure Kinect</code> · <code>ZED 2i</code> · <code>iPhone LiDAR</code> · <code>BNO080 IMU</code></li> <li><b>2-D 关键点检测：</b> <code>MediaPipe Pose</code> · <code>ML Kit</code> · <code>MoveNet</code> · <code>OpenPose</code> · <code>AlphaPose</code> · <code>YOLO-v8 Pose</code></li> <li><b>3-D Lift-up / 多视角融合：</b> <code>VNect</code> · <code>Pose-Lifter</code> · <code>METRO</code> · <code>HybrIK</code> · <code>MMPose-3D</code> · <code>DeepLabCut-3D</code> (配合 ICP / Bundle Adjustment 矫正)</li> <li><b>时空/物理一致性优化：</b> <code>Transformer-VAE</code> · <code>ST-GCN</code> · <code>Physics-Informed LSTM</code> · <code>EKF / UKF</code> 多传感器融合</li> <li><b>推理加速与可解释性：</b> <code>TensorRT</code> · <code>ONNX-Runtime</code> · <code>Core ML</code> · <code>NNAPI</code> · <code>WebNN</code> + <code>Grad-CAM</code> / <code>SHAP</code></li> </ul> </div> <p style="text-align:justify;font-size:14.3px;line-height:1.55;"> 本技术栈覆盖了从数据采集与预处理（2D / 3D 姿态检测、视角融合）到时空序列建模与可解释性评估的完整流程。通过引入物理约束与多模态融合策略，系统在实现高精度骨骼重建的同时，兼顾了实时性与可扩展性，为中风康复场景中的运动学分析提供坚实的技术支撑。 </p>

