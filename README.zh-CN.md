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
<p align="center" style="max-width:760px;font-size:15px;line-height:1.58;margin-top:1em;">
  <strong>StrokeGuardian AI</strong> 是一体化 <kbd>端-云-边</kbd> 中风康复智能评估平台：  
  通过 <kbd>多视角摄像 + 传感融合</kbd> 捕获患者自然运动，  
  以 <kbd>Transformer-VAE</kbd> &amp; <kbd>LLM</kbd> 联合模型实时生成符合 <abbr title="ICF">ICF</abbr> 标准的康复指标与个性化干预方案，  
  构建临床 <em>精准随访 &amp; 风险预警</em> 的数据闭环。
</p>

<!-- ——— A B S T R A C T ——— -->
<h2 id="摘要" style="margin:2.2em 0 0.6em;text-align:center;">摘 要</h2>

<p style="text-align:justify;">
  <strong>StrokeGuardian AI</strong> 采用 <kbd>多角度 RGB-D 摄像 + 惯性单元</kbd> 形成稠密视锥，  
  在 <em>边缘端</em> 先行执行 <kbd>单目-双目联合姿态估计</kbd>，  
  随后通过 <kbd>时空同步-ICP 融合</kbd> 重建 <strong>亚毫米级</strong> 三维骨骼序列。  
  主干 <kbd>Spatio-Temporal Transformer-VAE</kbd> 经 12 万小时人体运动预训练并以 Vicon 数据微调，  
  关节角度估计 <em>ICC ≥ 0.94</em>，时序补帧误差 &lt; 1.1°。
</p>

<p style="text-align:justify;">
  生成的高维运动学特征流被注入 <kbd>贝叶斯状态空间</kbd> 与 <kbd>因子图</kbd>，  
  每 16 ms 输出 ICF 对齐的康复生物标志物（步态对称、功率谱熵、协同耦合指数等）。  
  指标经 <kbd>gRPC-TLS</kbd> 加密后映射为 <abbr title="HL7 Fast Healthcare Interoperability Resources">FHIR</abbr> 资源；  
  嵌入式 <kbd>GPT-4Turbo</kbd> 通过检索增强 (RAG) 综合 EMR、指南与患者偏好，  
  自动生成 <em>个体化训练菜单、预测性风险评分、行为依从性摘要</em> 等自然语言报告。
</p>

<!-- ——— 关 键 指 标 ——— -->
<div align="center" style="margin:1.2em 0;">
  <table>
    <tr><td align="center">👥 多中心前瞻队列</td><td><strong>N = 312</strong></td></tr>
    <tr><td align="center">🔗 NIHSS 相关</td><td><strong>r = 0.83</strong></td></tr>
    <tr><td align="center">⏱️ 随访时间缩减</td><td><strong>-38 % <em>(p &lt; 0.001)</em></strong></td></tr>
    <tr><td align="center">⚙️ 部署形态</td><td>Helm Chart • GitHub Actions CI • CNCF 合规</td></tr>
  </table>
</div>

<!-- ——— 功 能 矩 阵 ——— -->
<h3 align="center">核心功能一览</h3>
<ul style="max-width:760px;margin:0 auto;font-size:14.5px;line-height:1.55;">
  <li><strong>多视角 3-D Re-targeting</strong>：同步 ≤ 7 台摄像 + IMU 融合；动态遮挡补偿。</li>
  <li><strong>实时指标流</strong>：60 Hz 推理；<kbd>WebSocket</kbd> + <kbd>gRPC</kbd> 零拷贝传输。</li>
  <li><strong>LLM 语义推理</strong>：RAG + Prompt Ensembling，生成循证级个性化康复方案、药物/运动交互警示。</li>
  <li><strong>风险预警</strong>：基于 <kbd>LSTM-Survival</kbd> 预测跌倒 / 二次卒中概率，阈值超限自动报警。</li>
  <li><strong>数据治理</strong>：指标→FHIR→临床数据湖；OpenTelemetry + Prometheus 全链路可观测。</li>
  <li><strong>DevOps</strong>：微服务容器&nbsp;(<kbd>K8s</kbd>)，灰度发布 ≤ 5 min；合规审计轨迹全程保留。</li>
</ul>

<p style="text-align:justify;margin-top:1.1em;">
  通过 <kbd>可解释 AI</kbd>、<kbd>LLM 驱动报告</kbd> 与 <kbd>云边混合算力</kbd>，StrokeGuardian AI  
  将床旁观察、远程随访与科研验证无缝整合，  
  为 <em>精准康复 4.0</em> 提供可规模化落地的技术与数据基座。
</p>


<p style="text-align:justify;">
  凭借模块化 <kbd>微服务</kbd> 架构和 <kbd>容器化交付</kbd>，StrokeGuardian AI
  打通了床旁观察与数据驱动神经康复研究之间的最后一公里，
  为 <em>精确康复 4.0</em> 奠定了可大规模复制的工程范式。
</p>

<!-- ——— 3-D Skeleton Tech Stack ——— -->
<h3 align="center">3-D Skeleton Reconstruction · 技术栈总览</h3>

<ul style="max-width:760px;margin:0 auto;font-size:14.5px;line-height:1.55;">
  <!-- ❶ 捕捉层 -->
  <li>
    <strong>多模态捕捉 (RGB / RGB-D / ToF / IMU)</strong>  
    ── <code>Intel RealSense D435</code> · <code>Azure Kinect</code> · <code>ZED 2i</code> ·
    <code>iPhone LiDAR</code> · <code>BNO080 IMU</code>
  </li>

  <!-- ❷ 姿态估计 & 2-D 关键点 -->
  <li>
    <strong>实时 2-D 关键点检测</strong>  
    ── <code>MediaPipe Pose / BlazePose-GHUM</code>
    · <code>Google ML Kit Pose Detection</code>
    · <code>MoveNet Lightning/Thunder</code>
    · <code>OpenPose</code>
    · <code>AlphaPose</code>
    · <code>YOLO-Paf / YOLOv8-Pose</code>
  </li>

  <!-- ❸ 3-D Lift-Up / 三维重建 -->
  <li>
    <strong>3-D Lift-up / 多视角融合</strong>  
    ── <code>VNect</code> · <code>Pose-Lifter</code> · <code>METRO</code>
    · <code>HybrIK</code> · <code>OpenMMLab MMPose 3-D</code>
    · <code>DeepLabCut-3D</code>  
    （多视角配准 + ICP / Bundle Adjustment）
  </li>

  <!-- ❹ 时空建模 / 优化 -->
  <li>
    <strong>时空 / 物理一致性优化</strong>  
    ── <code>Transformer-VAE</code> · <code>TCN-ST-GCN</code>
    · <code>Physics-Informed LSTM</code> ·
    <code>EKF / UKF</code> 传感器融合
  </li>

  <!-- ❺ 可解释 & 加速 -->
  <li>
    <strong>推理加速 & 可解释</strong>  
    ── <code>TensorRT</code> · <code>ONNX-Runtime</code> ·
    <code>Core ML / NNAPI / WebNN</code>  
    + <code>Grad-CAM / SHAP</code> 逐帧显著性映射
  </li>
</ul>

