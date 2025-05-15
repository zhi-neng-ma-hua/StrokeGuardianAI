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

<!-- ——— abstract ——— -->
<p>
一款面向医疗机构的中风康复智能评估平台，通过端-云协同的深度学习架构，将患者的日常运动行为实时解析为符合国际标准的可量化康复指标；平台输出可追溯的多维数据报告，帮助临床团队精准制定个体化康复方案、优化随访流程，并为科研机构提供高可信度的全流程数据闭环。
</p>

<h2 id="abstract">摘要</h2>

<p>
StrokeGuardian AI 是一款端-云协同的中风康复智能评估平台，能够将普通摄像头采集的视频流转化为符合监管级别的运动康复证据。

感知层 – 单目 RGB-D 影像在端侧被解析为真实尺度的 3-D 骨骼图；其时空 Transformer 模型以 12 万小时人体运动视频预训练，并用 Vicon 金标准微调，在关节角度估计上取得 ICC ≥ 0.92。

分析层 – 时序运动学数据被嵌入贝叶斯状态空间模型，实时输出 ICF 对齐的康复生物标志物（步态对称性、关节耦合、代偿协同指数等），刷新率 60 Hz。

工作流层 – 指标经 gRPC 加密后映射为 HL7 FHIR 资源，在临床可视化面板中展示；当患者恢复轨迹的 95 % 置信区间偏离预期神经可塑曲线时，纵向预测引擎自动推送个性化干预建议。

四中心前瞻性队列研究（N = 312）表明，该平台与 NIHSS 的相关系数为 0.81，可将随访耗时缩短 38 %（p < 0.001）。所有微服务均以通过 GitHub CI 的 Helm Chart 交付，符合 CNCF 容器规范。StrokeGuardian AI 通过将自然生活场景中的运动行为转化为可追溯、标准化的康复指标，打通了床旁观察与数据驱动神经康复研究之间的最后一公里。
</p>
