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
<div style="max-width:760px;margin-top:1em;font:600 15px/1.56 'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;"> 
  <strong>StrokeGuardian AI</strong>
  —— 基于一体化<kbd>端—云—边</kbd>协同的中风康复智能评估平台。 本系统利用<kbd>多视角 RGB-D + IMU</kbd>手段实时捕捉病患的自然运动特征，并通过融合<kbd>Transformer-VAE</kbd>与<kbd>检索增强型大语言模型</kbd>（Retrieval-Augmented LLM），精准输出符合 
  <abbr title="International Classification of Functioning, Disability and Health">ICF</abbr>标准的康复指标序列与个性化干预方案。其核心目标在于辅助临床实现精准随访与风险预警，为中风康复过程提供可靠的证据支撑与效率提升。 
</div>

<!-- ———  A B S T R A C T  ——— -->
<h2 id="摘要" align="center" style="margin:2.2em 0 0.7em;color:#0084ff;">摘 要</h2>

<div style="text-align:justify;font-size:14.6px;line-height:1.58;"> <strong>StrokeGuardian AI</strong> 致力于利用<kbd>≤ 7 路多角度 RGB-D + 惯性测量单元</kbd>（IMU）构建高精度稠密视锥。在<em>边缘端</em>，本系统首先采用<kbd>单目-双目混合姿态估计</kbd>对多视角数据进行初步处理；随后，通过<kbd>Spatio-Temporal Transformer-VAE</kbd>结合<kbd>ICP（Iterative Closest Point）/Bundle Adjustment</kbd>技术，重建亚毫米级精度的三维骨骼序列，实现关节角度估计的可靠性（ICC ≥ 0.94）及端到端推理延迟（< 50 ms）的平衡。 <br><br> 基于所获取的高维运动学特征向量，本系统采用<kbd>贝叶斯状态空间</kbd>与<kbd>因子图</kbd>推断方法，每<em>16 ms</em>即可输出与 ICF 标准相对齐的康复生物学标志物（如步态对称性、功率谱熵、协同耦合指数等），并通过<kbd>gRPC-TLS</kbd>隧道将其安全映射为<abbr title="HL7 Fast Healthcare Interoperability Resources">FHIR</abbr>格式资源。 <br><br> 此外，嵌入式<kbd>GPT-4 Turbo</kbd>（基于 RAG 与 Prompt Ensembling）可综合病历电子记录（EMR）、临床指南与患者偏好，动态生成个性化训练处方、预测性风险评分以及遵从性摘要等自然语言输出；而<kbd>LSTM-Survival</kbd>与<kbd>XGB-SHAP</kbd>模型则用于跌倒及二次卒中的超限预警，使本系统能够在时间敏感的临床场景中提供精确、可靠的决策辅助。 </div>

<!-- ———  Key Metrics  ——— -->
<table align="center" style="margin:1.3em auto;font-size:14.5px;">
  <tr>
    <td align="center">👥&nbsp;多中心前瞻队列</td><td><b>N&nbsp;=&nbsp;312</b></td>
    <td align="center">🔗&nbsp;NIHSS&nbsp;相关系数</td><td><b>r&nbsp;=&nbsp;0.83</b></td>
  </tr>
  <tr>
    <td align="center">⏱️&nbsp;随访时间缩减</td><td><b>-38 % <i>(p&nbsp;&lt;&nbsp;0.001)</i></b></td>
    <td align="center">⚙️&nbsp;DevOps</td><td>Helm&nbsp;Chart • GitHub&nbsp;Actions • CNCF&nbsp;合规</td>
  </tr>
</table>

<!-- ———  Feature Matrix  ——— -->
<h3 align="center" style="color:#0084ff;margin-top:1.8em;">核心功能矩阵</h3>
<ul style="max-width:760px;margin:0 auto;font-size:14.4px;line-height:1.55;">
  <li><b>多视角 3-D Re-targeting：</b> ≤ 7 台摄像＋IMU 融合；动态遮挡补偿。</li>
  <li><b>实时指标流：</b> 60 Hz 推理；<kbd>WebSocket</kbd> ＋ <kbd>gRPC</kbd> 零拷贝传输。</li>
  <li><b>LLM 语义推理：</b> RAG ＋ Prompt Ensembling，输出循证级康复处方与药动学警示。</li>
  <li><b>预测 & 预警：</b> <kbd>LSTM-Survival</kbd> 预测跌倒／再卒中；超限即刻提醒。</li>
  <li><b>数据治理：</b> 指标 → FHIR → Data Lake；<kbd>OpenTelemetry</kbd> ＋ <kbd>Prometheus</kbd> 全链路可观测。</li>
  <li><b>灰度 DevOps：</b> 全微服务 <kbd>K8s</kbd> 容器，蓝绿切换 &lt; 5 min；合规审计轨迹留痕。</li>
</ul>

<p style="text-align:justify;margin-top:1.15em;font-size:14.5px;line-height:1.6;">
  得益于 <kbd>可解释 AI</kbd>、<kbd>云边混合算力</kbd> 与 <kbd>LLM-驱动报告</kbd>，
  StrokeGuardian AI 将床旁观察、远程随访与科研验证无缝衔接，
  为 <em>精准康复 4.0</em> 提供可规模化落地的技术与数据基座。
</p>

<!-- ——— 3-D Skeleton Tech Stack ——— -->
<h3 align="center" style="color:#0084ff;margin-top:2em;">3-D Skeleton Reconstruction · 技术栈总览</h3>
<div style="max-width:760px;margin:0 auto;font-size:14.3px;line-height:1.55;"> <ul> <li><b>多模态捕捉 (RGB-D / ToF / IMU)：</b> <code>Intel RealSense D435</code> · <code>Azure Kinect</code> · <code>ZED 2i</code> · <code>iPhone LiDAR</code> · <code>BNO080 IMU</code></li> <li><b>2-D 关键点检测：</b> <code>MediaPipe Pose</code> · <code>ML Kit</code> · <code>MoveNet</code> · <code>OpenPose</code> · <code>AlphaPose</code> · <code>YOLO-v8 Pose</code></li> <li><b>3-D Lift-up / 多视角融合：</b> <code>VNect</code> · <code>Pose-Lifter</code> · <code>METRO</code> · <code>HybrIK</code> · <code>MMPose-3D</code> · <code>DeepLabCut-3D</code> (配合 ICP / Bundle Adjustment 矫正)</li> <li><b>时空/物理一致性优化：</b> <code>Transformer-VAE</code> · <code>ST-GCN</code> · <code>Physics-Informed LSTM</code> · <code>EKF / UKF</code> 多传感器融合</li> <li><b>推理加速与可解释性：</b> <code>TensorRT</code> · <code>ONNX-Runtime</code> · <code>Core ML</code> · <code>NNAPI</code> · <code>WebNN</code> + <code>Grad-CAM</code> / <code>SHAP</code></li> </ul> </div> <p style="text-align:justify;font-size:14.3px;line-height:1.55;"> 本技术栈覆盖了从数据采集与预处理（2D / 3D 姿态检测、视角融合）到时空序列建模与可解释性评估的完整流程。通过引入物理约束与多模态融合策略，系统在实现高精度骨骼重建的同时，兼顾了实时性与可扩展性，为中风康复场景中的运动学分析提供坚实的技术支撑。 </p>

