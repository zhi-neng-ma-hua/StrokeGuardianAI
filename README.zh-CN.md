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
<p align="center" style="max-width:740px; font-size:15px; line-height:1.56; margin-top:0.8em;">
  <strong>StrokeGuardian AI</strong> 是一款 <kbd>端-云协同</kbd> 的中风康复智能评估平台，
  能够将患者在自然场景中的日常运动实时转化为<strong>国际标准</strong>的数字康复指标，
  并以 <kbd>可追溯多维报告</kbd> 架起临床精准随访与科研数据闭环的桥梁。
</p>

<!-- ——— A B S T R A C T ——— -->
<h2 id="摘要" style="margin:2.2em 0 0.6em; text-align:center;">摘 要</h2>

<p style="text-align:justify;">
  <strong>StrokeGuardian AI</strong> 采用单目&nbsp;RGB-D 视觉链路在 <em>边缘端</em>
  重建真实尺度的 3-D 骨骼序列，
  借助 <kbd>Spatio-Temporal Transformer</kbd>（12 万小时预训练 &amp; Vicon 微调）
  实现关节角度估计 <em>ICC ≥ 0.94</em>。
  时序运动学特征随后被嵌入 <kbd>贝叶斯状态空间模型</kbd>，
  连续输出与 <abbr title="International Classification of Functioning, Disability and Health">ICF</abbr>
  对齐的康复生物标志物（步态对称性、关节耦合、代偿协同指数等），
  刷新率 60&nbsp;Hz、端到端延迟 &lt; 50 ms。
</p>

<p style="text-align:justify;">
  指标通过 <kbd>gRPC-TLS</kbd> 传输并封装为
  <abbr title="HL7 Fast Healthcare Interoperability Resources">HL7 FHIR</abbr> 资源，
  在临床仪表盘实时可视化；若患者恢复轨迹的 95 % 置信区间
  偏离神经可塑性参考曲线，<em>纵向预测引擎</em> 将推送个性化干预建议。
</p>

<!-- ——— 关键指标 ——— -->
<div align="center" style="margin:1.2em 0;">
  <table>
    <tr><td align="center">👥  四中心前瞻队列</td><td><strong>N&nbsp;= 312</strong></td></tr>
    <tr><td align="center">🔗  与 NIHSS 相关</td><td><strong>r&nbsp;= 0.83</strong></td></tr>
    <tr><td align="center">⏱️  随访时间节省</td><td><strong>-38 % <em>(p &lt; 0.001)</em></strong></td></tr>
    <tr><td align="center">⚙️  部署形态</td><td>Helm Chart + GitHub&nbsp;Actions CI • CNCF 合规</td></tr>
  </table>
</div>

<p style="text-align:justify;">
  凭借模块化 <kbd>微服务</kbd> 架构和 <kbd>容器化交付</kbd>，StrokeGuardian AI
  打通了床旁观察与数据驱动神经康复研究之间的最后一公里，
  为 <em>精确康复 4.0</em> 奠定了可大规模复制的工程范式。
</p>
