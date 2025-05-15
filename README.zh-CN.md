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

<!-- ——— A B S T R A C T ——— --> 
<!-- ——— 摘要 ——— -->
<h2 id="摘要">摘要</h2>

<p>
  <strong>StrokeGuardian AI</strong> 是一款面向医疗机构的端—云协同中风康复智能评估平台，
  能够将普通摄像头采集的自然场景视频实时转化为符合法规要求的数字康复生物标志物。
</p>

<p>
  平台在端侧采用单目 RGB-D 视觉链路重建真实尺度的三维骨骼序列；
  其时空 Transformer 主干网络经过
  12&nbsp;万小时人体运动数据预训练，并结合 Vicon 金标准微调，
  在关节角度估计上实现 <em>ICC ≥ 0.94</em>。
  随后，系统将时序运动学数据嵌入贝叶斯状态空间模型，
  连续输出与
  <abbr title="国际功能、残疾和健康分类">ICF</abbr>
  体系对齐的康复指标（步态对称性、关节耦合度、代偿协同指数等），
  刷新率 60 Hz，端到端延迟 &lt; 50 ms。
</p>

<p>
  所有指标经 gRPC 加密后映射为
  <abbr title="Health Level 7 – Fast Healthcare Interoperability Resources"
        >HL7 FHIR</abbr> 资源，
  并在临床可视化面板中展示；当患者康复轨迹的 95 % 置信区间偏离
  神经可塑性参考曲线时，纵向预测引擎会自动推送个性化干预建议。
</p>

<p>
  在四中心前瞻性队列（<em>N</em> = 312）中，
  平台与 NIHSS 评分的 Pearson 相关系数达到 <em>0.83</em>，
  随访耗时减少 38 %（<em>p</em> &lt; 0.001）。
  全部微服务以 Helm Chart 形式交付，并通过 GitHub Actions CI 流水线，
  符合 CNCF 容器化最佳实践。
  StrokeGuardian AI 将床旁观察与数据驱动的神经康复研究高效衔接，
  打通了精确康复落地的最后一公里。
</p>
