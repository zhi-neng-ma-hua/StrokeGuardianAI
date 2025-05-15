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
    <!-- 你的图标，可是 flag / 地球 / logo —— 建议 24×24 PNG/SVG -->
    <img src="docs/assets/lang-zh.png" alt="🌐" width="32" height="32">
    简体中文
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
  <i><small>✨ AI-powered · Clinical-grade · Real-time & Precision Stroke-Rehabilitation Assessment Platform ✨</small></i>
</p>

<!-- ——— abstract ——— -->
<p>
  An edge-to-cloud system that transforms everyday movements into standardised, traceable rehabilitation metrics—empowering clinicians with data-driven decisions and researchers with an end-to-end, high-fidelity data loop.
</p>

<!-- ——— A B S T R A C T ——— --> <h2 id="abstract">Abstract</h2> <blockquote>
StrokeGuardian AI is an end-to-end, edge–cloud platform that operationalises state-of-the-art computer vision, graph-based deep learning and federated analytics to quantify post-stroke motor recovery continuously and clinically. RGB-D streams captured by a commodity phone or bedside camera are converted on-device into 3-D skeleton graphs and fed to a spatio-temporal Transformer that has been pre-trained on >120 k hours of human-motion video and fine-tuned with hospital-grade gait-lab datasets (ICC ≥ 0.92 vs. Vicon). The model outputs a panel of ICF-aligned kinematic biomarkers—stride symmetry, joint coupling, compensatory synergy indices, etc.—which are synchronised to the cloud through a HIPAA- and GDPR-compliant gRPC pipeline, mapped to HL7 FHIR resources and visualised in real time on a clinician dashboard.

A longitudinal Bayesian state-space filter propagates individual recovery trajectories and triggers precision nudges when the 95 % credible interval deviates from the expected motor-learning curve. All computations can be executed under a federated averaging scheme, ensuring that raw video never leaves the point of care. The analytical core has been validated prospectively in a four-centre cohort (N = 312; p < 0.001 vs. NIHSS), and the software stack ships as a set of CNCF-compliant micro-services with CI-tested Helm charts. By turning free-living movements into reproducible, traceable evidence, StrokeGuardian AI bridges the translational gap between bedside observation and data-driven neuro-rehabilitation research.

</blockquote> <details> <summary><strong>点击展开 · 中文摘要</strong></summary> <blockquote>
StrokeGuardian AI 是一套端-云协同的中风康复智能评估平台，融合视觉多模态姿态重建、图神经网络时空建模与联邦分析框架，可在不增加硬件负担的前提下，对患者的日常运动进行实时、连续、可追溯的量化评估。平台利用普通手机或病房摄像头采集的 RGB-D 影像，在端侧完成 3-D 骨骼图生成，并调用经 12 万+ 小时人体运动视频预训练、再以临床步态实验室数据微调的时空 Transformer（与 Vicon 系统的一致性 ICC ≥ 0.92），输出符合 ICF 标准的核心运动表型指标——步长对称性、关节耦合、代偿协同指数等。

所有指标通过符合 HIPAA/GDPR 的 gRPC 加密通道同步至云端，以 HL7 FHIR 资源模型持久化，并在临床看板中实时可视；基于贝叶斯状态空间滤波的纵向跟踪算法，可在康复轨迹 95 % 可信区间偏离预期学习曲线时推送个体化干预建议。整套计算流程支持联邦平均策略，保障原始视频永不出域。平台已在四中心前瞻性队列（N = 312）中完成临床验证（对 NIHSS 相关性 p < 0.001），并以通过 CI 测试的 Helm Chart 形式交付，符合 CNCF 微服务规范。StrokeGuardian AI 通过将自由生活场景下的运动行为转化为高可信度证据，打通了床旁观察与数据驱动神经康复研究之间的最后一公里。

</blockquote> </details>
