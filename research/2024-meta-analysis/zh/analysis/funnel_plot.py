#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名称：funnel_plot_meta_analysis.py

模块概述：
    本模块面向元分析（Meta-Analysis）流程，为研究人员和企业级数据团队提供完整、
    专业且可定制的“漏斗图”（Funnel Plot）可视化解决方案。涵盖从数据模拟、
    样式配置到图形生成的端到端支持，确保图表在科研报告、自动化流水线及 Web 服务
    中均可稳定、可复现地呈现高质量视觉效果。

主要功能：
  1. 数据模拟（FunnelDataSimulator）：基于用户指定的研究数量、随机种子和发表偏倚
     比例，生成效应量 (yi)、标准误 (se) 及精度 (1/se) 数据集，并附带完整日志和异常
     捕获；
  2. 样式管理（FunnelPlotStyle + 全局 StyleManager）：集中定义图表尺寸、留白、分辨率、
     散点、网格、漏斗等高线、文本、图例、外框等视觉参数，实现与全局绘图风格解耦；
  3. 绘图核心（FunnelPlotter）：分步严格执行漏斗区域、散点、标签、平均线、主/次网格、
     坐标轴、图例和外框绘制，支持 Y 轴反转、网格细分和等边距布局，并将图像保存或
     直观展示；
  4. 企业级日志（LoggerFactory）与全局风格（StyleManager）集成：自动识别脚本或包模式，
     控制台与文件双通道输出，日志按天或大小切割并保留历史，保证流程可追溯性与可诊断性；
  5. 异常管理：在每一关键环节采用 try/except 捕获、记录并向上抛出自定义异常，确保
     主流程友好退出或告警。

典型场景：
    - 检测元分析中的发表偏倚，可视化诊断；
    - 自动化科研报告与企业级 BI 平台的插图生成；
    - 教学示例及敏感性分析流水线中的可复现演示；
    - Python Web 服务或 Jupyter Notebook 中实时绘图。

主要依赖：
    - numpy, pandas：数据生成与处理
    - matplotlib：底层绘图
    - seaborn：全局主题支持（通过 StyleManager）
    - utils.logger_factory：企业级日志配置
    - utils.plt_style_manager：全局绘图风格管理

运行方式：
    python funnel_plot_meta_analysis.py

Author: 智能麻花
Date: 2025-05-15
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.logger_factory import LoggerFactory
from utils.plt_style_manager import StyleConfig, StyleManager

# Todo # --- 日志模块初始化 --- #
# 自动判断：作为脚本运行时，用文件名作为 Logger 名；否则用模块名 __name__
module_name = __name__ if __name__ != "__main__" else Path(__file__).stem

# 获取 Logger
#    - root 级别由环境变量 LOG_LEVEL 控制，默认 DEBUG
#    - 控制台输出 INFO 及以上，启用彩色、高亮
#    - 文件输出 DEBUG 及以上，按天切割、保留15天日志
#    - 格式中包含时间、线程、级别、模块:行号、函数名、消息
logger = LoggerFactory.get_logger(
    module_name,
    level=logging.DEBUG,  # Logger 总级别
    console_level=logging.INFO,  # 控制台 INFO+
    logfile="logs/funnel_plot_meta_analysis.log",  # 写入文件
    file_level=logging.DEBUG,  # 文件 DEBUG+
    max_bytes=None,  # 文件大小上限
    backup_count_bytes=3,  # 保留最近 3 个文件
    when="midnight",  # 按天切割
    backup_count_time=7  # 保留最近 7 天日志
)

# Todo # --- 全局样式配置 --- #
# 样式配置：补全 DPI、上下文、字体候选、网格、调色板、色盲配色等
try:
    cfg = StyleConfig(
        grid=False,
        palette="colorblind",
        context="talk"
    )
    StyleManager.apply(cfg)
    logger.info("全局绘图风格配置应用成功：dpi=%d, palette=%s, context=%s", cfg.dpi, cfg.palette, cfg.context)
except Exception as e:
    logger.error("应用全局绘图风格失败，将使用默认 Matplotlib 设置：%s", str(e), exc_info=True)


@dataclass
class FunnelDataSimulator:
    """
    漏斗图数据模拟器（FunnelDataSimulator）

    业务背景：
      元分析 (Meta-Analysis) 中常需绘制漏斗图以检测发表偏倚。此类
      提供高度可定制、可复现的数据模拟能力，支持科研与企业级上
      线需求，包括可控的研究数量、随机种子和偏倚比例。

    核心功能：
      1. 参数校验：确保 n_studies >=1, seed >=0, biased_portion ∈ [0,1]
      2. 效应量生成：按 biased_portion 比例分段生成偏低/偏高效应量
      3. 标准误生成：均匀分布在 [0.05,0.15) 范围内
      4. 精度计算：1 / se
      5. 中间变量统计日志：均值、方差、最大最小值等，辅助诊断

    方法：
      simulate() -> pd.DataFrame:
        返回带 ["study","yi","se","precision"] 列的 DataFrame。
    """
    n_studies: int = field(default=15, metadata={"help": "研究总数 (>=1)"})
    seed: int = field(default=42, metadata={"help": "随机种子 (>=0)"})
    biased_portion: float = field(default=0.5, metadata={"help": "偏倚研究比例 [0,1]"})

    def __post_init__(self):
        # 参数与业务合理性校验
        if self.n_studies < 1:
            logger.error("初始化失败：研究数量 n_studies=%d 小于 1", self.n_studies)
            raise ValueError("n_studies 必须 >= 1")
        if self.seed < 0:
            logger.error("初始化失败：随机种子 seed=%d 小于 0", self.seed)
            raise ValueError("seed 必须 >= 0")
        if not (0.0 <= self.biased_portion <= 1.0):
            logger.error("初始化失败：偏倚比例 biased_portion=%.4f 不在 [0,1]", self.biased_portion)
            raise ValueError("biased_portion 必须在 [0,1] 区间内")
        logger.info(
            "FunnelDataSimulator 已初始化：n_studies=%d, seed=%d, biased_portion=%.2f",
            self.n_studies, self.seed, self.biased_portion
        )

    def _generate_effects(self) -> np.ndarray:
        """
        生成效应量 (yi)：先按 biased_portion 划分数量，再分别从
        两个中心波动区间随机抽样，最后拼接返回。

        Returns:
            yi: np.ndarray, 长度为 n_studies

        Raises:
            RuntimeError: 生成过程中出现数值异常
        """
        try:
            low_n = int(self.n_studies * self.biased_portion)
            high_n = self.n_studies - low_n
            # 偏低效应量：中心 0.2，波动 ±0.3
            yi_low = 0.2 + 0.3 * (np.random.rand(low_n) - 0.5) * 2
            # 偏高效应量：中心 0.6，波动 ±0.2
            yi_high = 0.6 + 0.2 * (np.random.rand(high_n) - 0.5) * 2
            yi = np.concatenate([yi_low, yi_high])
            logger.debug(
                "效应量生成统计：low_n=%d, high_n=%d, mean=%.4f, std=%.4f",
                low_n, high_n, yi.mean(), yi.std()
            )
            return yi
        except Exception as ex:
            logger.exception("生成效应量失败：%s", ex)
            raise RuntimeError("效应量生成出错") from ex

    def _generate_se(self) -> np.ndarray:
        """
        生成标准误 (se)：从 [0.05, 0.15) 均匀分布中采样。

        Returns:
            se: np.ndarray, 长度为 n_studies

        Raises:
            RuntimeError: 生成过程中出现数值异常
        """
        try:
            se = 0.05 + 0.1 * np.random.rand(self.n_studies)
            logger.debug("标准误生成统计：min=%.4f, max=%.4f, mean=%.4f",
                         se.min(), se.max(), se.mean())
            return se
        except Exception as ex:
            logger.exception("生成标准误失败：%s", ex)
            raise RuntimeError("标准误生成出错") from ex

    def simulate(self) -> pd.DataFrame:
        """
        执行完整的漏斗图数据模拟流程。

        1) 设定随机种子；
        2) 生成效应量 yi；
        3) 生成标准误 se；
        4) 构建 DataFrame 并计算 precision；
        5) 验证 precision 合法性 (不可为零或负)；
        6) 返回结果。

        Returns:
            pd.DataFrame: 包含列 ["study","yi","se","precision"]

        Raises:
            RuntimeError: 模拟过程中出现未知错误，或 precision 非法
        """
        try:
            logger.info(
                "开始模拟数据：n_studies=%d, seed=%d, biased_portion=%.2f",
                self.n_studies, self.seed, self.biased_portion
            )
            # 1) 设定随机种子
            np.random.seed(self.seed)

            # 2) 生成效应量 yi
            yi = self._generate_effects()

            # 3) 生成标准误 se
            se = self._generate_se()

            # 4) 构建 DataFrame 并计算 precision
            df = pd.DataFrame({
                "study": [f"Study_{i + 1}" for i in range(self.n_studies)],
                "yi": yi,
                "se": se,
            })
            # 计算精度
            df["precision"] = 1.0 / df["se"]

            # 5) 验证 precision 合法性 (不可为零或负)
            # 数据质量检查
            if (df["precision"] <= 0).any():
                logger.error("计算精度异常：存在非正值，数据预览：\n%s", df.head().to_string(index=False))
                raise RuntimeError("precision 中存在非正数，数据模拟失败")

            logger.info("数据模拟完成：共 %d 条记录", len(df))
            return df
        except (ValueError, RuntimeError):
            # 参数校验及已知运行时错误直接向上抛
            raise
        except Exception as ex:
            # 捕获所有未知异常
            logger.exception("未知错误导致模拟失败：%s", ex)
            raise RuntimeError("数据模拟执行遇到未知错误") from ex


@dataclass
class FunnelPlotStyle:
    """
    漏斗图样式配置类（FunnelPlotStyle）

    本类集中管理漏斗图的所有视觉参数，并提供校验与一键应用功能，
    以保证样式集中、可维护、易重用。适用于科研报告与企业 BI
    平台中高质量、可复现的漏斗图渲染。

    属性:
        fig_width, fig_height             图表尺寸（英寸），必须正数
        dpi                               输出分辨率，正整数
        margin                            四边留白比例 {left, right, top, bottom}，值 ∈ [0,0.5]
        marker                            散点样式参数字典:
            size, edgecolor, facecolor, alpha
        label_text                        数值标签样式参数:
            offset, fontsize, color, bbox(dict)
        mean_line                         平均效应线样式参数:
            color, linestyle, linewidth
        funnel_region                     漏斗等高线与填充样式:
            fill_color, fill_alpha,
            line_style, line_width, z_value
        axes_label                        轴标签和标题文字与样式:
            xlabel, ylabel, title,
            xlabel_kwargs(dict), ylabel_kwargs(dict), title_kwargs(dict)
        grid                              {'major': {...}, 'minor': {...}} 两级网格样式
        legend                            图例样式参数:
            loc, fontsize, frameon, handlelength, labelspacing
        border                            图外框样式:
            color, linewidth

    方法:
        __post_init__()      ：参数校验并记录日志
        apply(fig, ax)       ：将本样式应用到给定 Figure/Axes
    """

    # 画布与留白
    fig_width: float = 12.0
    fig_height: float = 7.0
    dpi: int = 300
    margin: Dict[str, float] = field(default_factory=lambda: {
        "left": 0.10,  # 左侧留白 10%
        "right": 0.10,  # 右侧留白 10%
        "bottom": 0.15,  # 底部留白 15%
        "top": 0.15  # 顶部留白 15%
    })

    # 散点样式
    marker: Dict[str, Any] = field(default_factory=lambda: {
        "s": 60.0,
        "edgecolors": "#2A6F97",
        "facecolors": "#A6CEE3",
        "alpha": 0.8,
        "linewidths": 0.8
    })

    # 数值标签样式
    label_text: Dict[str, Any] = field(default_factory=lambda: {
        "offset": 0.5,
        "fontsize": 9,
        "color": "#1F78B4",
        "bbox": {"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1}
    })

    # 平均效应线样式
    mean_line: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#E31A1C",
        "linestyle": "--",
        "linewidth": 2.0
    })

    # 漏斗区域样式
    funnel_region: Dict[str, Any] = field(default_factory=lambda: {
        "fill_color": "#B2DF8A",
        "fill_alpha": 0.2,
        "line_style": "--",
        "line_width": 1.0,
        "z_value": 1.96
    })

    # 轴标签与标题
    axes_label: Dict[str, Any] = field(default_factory=lambda: {
        "xlabel": "效应量 (yi)",
        "ylabel": "精度 (1/SE)",
        "title": "漏斗图模拟 (检测发表偏倚)",
        "xlabel_kwargs": {"fontsize": 14, "labelpad": 20},
        "ylabel_kwargs": {"fontsize": 14, "labelpad": 20},
        "title_kwargs": {"fontsize": 18, "weight": "semibold", "pad": 30}
    })

    # 网格（主/次）
    grid: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "major": {"which": "major", "linestyle": "-", "linewidth": 0.8, "color": "#DDDDDD", "alpha": 0.6},
        "minor": {"which": "minor", "linestyle": ":", "linewidth": 0.5, "color": "#EEEEEE", "alpha": 0.3}
    })

    # 图例
    legend: Dict[str, Any] = field(default_factory=lambda: {
        "loc": "upper right",
        "fontsize": 12,
        "frameon": False,
        "handlelength": 2.5,
        "labelspacing": 1
    })

    # 外框
    border: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#909090",
        "linewidth": 1.5
    })

    def __post_init__(self):
        """
        参数校验：
          1. fig_width, fig_height 必须 > 0；
          2. dpi 必须 > 0；
          3. margin['left'], margin['right'], margin['bottom'], margin['top'] 都必须 ∈ [0,1]；
          4. margin['left'] + margin['right'] < 1 （保证子图宽度 > 0）；
          5. margin['bottom'] + margin['top'] < 1 （保证子图高度 > 0）；
          6. marker.alpha 与 funnel_region.fill_alpha 必须 ∈ [0,1]；
        并在通过校验后输出初始化日志。
        """
        # 1) 图表尺寸与 DPI
        if self.fig_width <= 0 or self.fig_height <= 0:
            logger.error("样式初始化失败：fig_width=%.2f, fig_height=%.2f 必须 > 0", self.fig_width, self.fig_height)
            raise ValueError("fig_width 和 fig_height 必须为正数")
        if self.dpi <= 0:
            logger.error("样式初始化失败：dpi=%d 必须 > 0", self.dpi)
            raise ValueError("dpi 必须为正整数")

        # 2) 每侧留白范围校验
        for side in ("left", "right", "bottom", "top"):
            val = self.margin[side]
            if not (0.0 <= val <= 1.0):
                logger.error("样式初始化失败：margin['%s']=%.2f 不在 [0,1]", side, val)
                raise ValueError(f"margin['{side}'] 必须在 [0,1] 区间内")

        # 3) 保证子图有宽度与高度
        if self.margin["left"] + self.margin["right"] >= 1.0:
            logger.error(
                "样式初始化失败：margin['left'] + margin['right']=%.2f >= 1.0",
                self.margin["left"] + self.margin["right"]
            )
            raise ValueError("margin['left'] + margin['right'] 必须 < 1.0")
        if self.margin["bottom"] + self.margin["top"] >= 1.0:
            logger.error(
                "样式初始化失败：margin['bottom'] + margin['top']=%.2f >= 1.0",
                self.margin["bottom"] + self.margin["top"]
            )
            raise ValueError("margin['bottom'] + margin['top'] 必须 < 1.0")

        # 4) 透明度校验
        alpha_marker = self.marker.get("alpha", 1.0)
        if not (0.0 <= alpha_marker <= 1.0):
            logger.error("样式初始化失败：marker.alpha=%.2f 不在 [0,1]", alpha_marker)
            raise ValueError("marker.alpha 必须在 [0,1]")
        alpha_funnel = self.funnel_region.get("fill_alpha", 1.0)
        if not (0.0 <= alpha_funnel <= 1.0):
            logger.error("样式初始化失败：funnel_region.fill_alpha=%.2f 不在 [0,1]", alpha_funnel)
            raise ValueError("funnel_region.fill_alpha 必须在 [0,1]")

        # 5) 初始化成功日志
        logger.info(
            "FunnelPlotStyle 已初始化：fig(%.1f×%.1f@%d DPI), margin=%s, marker.size=%.1f",
            self.fig_width, self.fig_height, self.dpi, self.margin, self.marker.get("s", 0.0)
        )

    def apply(self, fig: plt.Figure, ax: plt.Axes):
        """
        将本样式应用到给定的 Figure/Axes 上。

        包含：
          1. 应用留白、DPI、外框；
          2. 隐藏上/右脊柱，粗化底/左脊柱；
          3. 设置网格主次级别、轴标签、标题；
          4. 图例参数配置（其它元素绘制由调用者完成）。

        参数:
            fig (plt.Figure): 目标 Figure 对象
            ax  (plt.Axes):   目标 Axes 对象

        异常:
            不捕获异常，上层自行处理绘图失败情形。
        """
        # 1) Figure 布局与外框
        fig.set_size_inches(self.fig_width, self.fig_height)
        fig.set_dpi(self.dpi)
        fig.patch.set_facecolor("white")
        fig.patch.set_edgecolor(self.border["color"])
        fig.patch.set_linewidth(self.border["linewidth"])

        # 2) 将 "留白比例" 转换为子图边界：
        lm = self.margin["left"]
        rm = self.margin["right"]
        bm = self.margin["bottom"]
        tm = self.margin["top"]
        fig.subplots_adjust(
            left=lm,  # 子图左边界 = 留白左侧
            right=1.0 - rm,  # 子图右边界 = 1 - 留白右侧
            bottom=bm,  # 子图底部边界 = 留白底部
            top=1.0 - tm  # 子图顶部边界 = 1 - 留白顶部
        )

        # 3) 脊柱样式
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"): ax.spines[sp].set_linewidth(1.2)

        # 4) 网格
        ax.grid(**self.grid["major"])
        ax.minorticks_on()
        ax.grid(**self.grid["minor"])

        # 5) 轴标签与标题
        ax.set_xlabel(self.axes_label["xlabel"], **self.axes_label["xlabel_kwargs"])
        ax.set_ylabel(self.axes_label["ylabel"], **self.axes_label["ylabel_kwargs"])
        ax.set_title(self.axes_label["title"], **self.axes_label["title_kwargs"])

        logger.info("已将 FunnelPlotStyle 应用到 Figure/Axes")


class FunnelPlotter:
    """
    漏斗图绘制器（FunnelPlotter）

    业务背景：
        在元分析（Meta-Analysis）中，漏斗图用于检测发表偏倚（publication bias），
        通过研究效应量与其精度的分布可视化辅助诊断。本类实现了从原始数据到
        高质量漏斗图的端到端封装，满足科研与企业级可复现绘图需求。

    核心功能：
      1. 初始化校验：确保输入 DataFrame 含有 "yi" 和 "precision" 列。
      2. 分步绘制流程：
         - 应用全局/自定义样式
         - 绘制漏斗等高线区域
         - 绘制个体研究散点及数值标签
         - 绘制平均效应线
         - 配置主/次网格、轴标签、标题、图例
      3. 支持保存到文件或直接展示，异常时给出清晰日志并抛出 RuntimeError。

    使用示例：
        plotter = FunnelPlotter(df, style=my_style)
        plotter.plot(save_path="funnel.png")
    """

    def __init__(self, data: pd.DataFrame, style):
        """
        初始化 FunnelPlotter。

        :param data: 必须包含 "yi"（效应量）和 "precision"（精度 1/se）两列的 DataFrame。
        :param style: FunnelPlotStyle 实例，集中管理全图视觉参数。
        :raises ValueError: 当 data 丢失必要列时抛出。
        """
        required = {"yi", "precision"}
        missing = required - set(data.columns)
        if missing:
            logger.error("FunnelPlotter 初始化失败：缺少列 %s", missing)
            raise ValueError(f"输入 DataFrame 必须包含列: {missing}")
        self.data = data.copy()
        self.style = style
        logger.info("FunnelPlotter 初始化完成，数据行数=%d", len(self.data))

    def _draw_funnel_region(self, ax: plt.Axes):
        """
        绘制漏斗图的置信区间填充区域：mean ± z * SE。

        :param ax: matplotlib Axes 对象
        """
        df = self.data
        s = self.style.funnel_region
        mean = df["yi"].mean()
        # 从最小到最大 SE 生成等高线
        se_vals = np.linspace(df["precision"].min() ** -1,
                              df["precision"].max() ** -1, 200)
        lower = mean - s["z_value"] * se_vals
        upper = mean + s["z_value"] * se_vals

        # 填充区域
        ax.fill_betweenx(
            1 / se_vals, lower, upper,
            color=s["fill_color"],
            alpha=s["fill_alpha"],
            zorder=1,
            label="95% 漏斗区域"
        )

        # 边界线
        ax.plot(
            lower, 1 / se_vals,
            linestyle=s["line_style"],
            linewidth=s["line_width"],
            color=s["fill_color"],
            zorder=2
        )
        ax.plot(
            upper, 1 / se_vals,
            linestyle=s["line_style"],
            linewidth=s["line_width"],
            color=s["fill_color"],
            zorder=2
        )

        logger.debug("已绘制漏斗等高线区域(mean=%.4f)", mean)

    def _draw_points_and_labels(self, ax: plt.Axes):
        """
        绘制个体研究散点及对应的数值标签。

        :param ax: matplotlib Axes 对象
        """
        df = self.data
        m = self.style.marker
        lbl = self.style.label_text

        # 绘制散点
        ax.scatter(
            df["yi"], df["precision"],
            s=m["s"],
            edgecolors=m["edgecolors"],
            facecolors=m["facecolors"],
            alpha=m["alpha"],
            linewidths=m["linewidths"],
            zorder=3,
            label="个体研究"
        )
        logger.debug("散点绘制完成，共 %d 个点", len(df))

        # 添加数值标签
        for _, row in df.iterrows():
            ax.text(
                row["yi"],
                row["precision"] + lbl["offset"],
                f"{row['yi']:.2f}",
                fontsize=lbl["fontsize"],
                ha="center", va="bottom",
                color=lbl["color"],
                zorder=4,
                bbox=lbl["bbox"]
            )
        logger.debug("散点标签绘制完成")

    def _draw_mean_line(self, ax: plt.Axes):
        """
        绘制平均效应垂直参考线。

        :param ax: matplotlib Axes 对象
        """
        mean = self.data["yi"].mean()
        ml = self.style.mean_line
        ax.axvline(
            mean,
            color=ml["color"],
            linestyle=ml["linestyle"],
            linewidth=ml["linewidth"],
            zorder=5,
            label=f"平均效应: {mean:.2f}"
        )
        logger.debug("平均效应线绘制完成(mean=%.4f)", mean)

    def plot(self, save_path: Optional[str] = None):
        """
        绘制漏斗图并保存或展示。

        :param save_path: 可选，若提供则将图保存到该路径；否则直接弹窗展示。
        :raises RuntimeError: 绘图过程中发生任何异常时抛出。
        """
        logger.info("开始执行 FunnelPlotter.plot()")
        try:
            # 1) 新建 Figure/Axes
            fig, ax = plt.subplots(figsize=(self.style.fig_width, self.style.fig_height), dpi=self.style.dpi)

            # 2) 应用样式（留白、外框、脊柱、网格、标签、标题）
            self.style.apply(fig, ax)

            # 3) 绘制漏斗区域
            self._draw_funnel_region(ax)

            # 4) 绘制散点与标签
            self._draw_points_and_labels(ax)

            # 5) 绘制平均效应线
            self._draw_mean_line(ax)

            # 6) 图例
            ax.legend(**self.style.legend)
            logger.debug("图例配置完成")

            # 7) 保存或展示
            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
                logger.info("漏斗图已保存至: %s", save_path)
            else:
                plt.show()
                logger.info("漏斗图展示完成")

        except Exception as ex:
            logger.error("漏斗图绘制失败：%s", ex, exc_info=True)
            raise RuntimeError("FunnelPlotter.plot 执行失败") from ex


def parse_args(argv=None) -> argparse.Namespace:
    """
    解析命令行参数。

    :param argv: 参数列表（默认: None，使用 sys.argv[1:]）
    :return: 包含以下属性的 Namespace：
        - n_studies (int): 研究总数 (>=1)
        - seed (int): 随机种子 (>=0)
        - biased_portion (float): 偏倚研究比例 [0,1]
        - output (str|None): 输出文件路径；若为 None 则弹窗展示
    :raises SystemExit: 当 --help 被调用或参数校验失败时退出
    """
    parser = argparse.ArgumentParser(
        prog="funnel_plot_meta_analysis.py",
        description="Funnel Plot 模拟与绘制工具：生成含发表偏倚的数据并绘制漏斗图"
    )
    parser.add_argument(
        "-n", "--n-studies",
        type=int, default=15,
        help="模拟研究数量 (正整数，默认: 15)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int, default=42,
        help="随机种子 (非负整数，默认: 42)"
    )
    parser.add_argument(
        "-b", "--biased-portion",
        type=float, default=0.5,
        help="偏倚研究比例 [0,1] (默认: 0.5)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str, default=None,
        help="输出图像路径；不指定则屏幕展示"
    )
    args = parser.parse_args(argv)
    # 参数校验
    if args.n_studies < 1:
        parser.error(f"--n-studies 必须 >= 1，当前: {args.n_studies}")
    if args.seed < 0:
        parser.error(f"--seed 必须 >= 0，当前: {args.seed}")
    if not (0.0 <= args.biased_portion <= 1.0):
        parser.error(f"--biased-portion 必须在 [0,1]，当前: {args.biased_portion}")
    return args


def main(argv=None) -> int:
    """
    主程序入口：解析参数 → 模拟数据 → 绘制漏斗图 → 保存或展示。

    :param argv: 参数列表（默认: None，使用 sys.argv[1:]）
    :return: 退出码
        0 成功
        1 参数错误或运行时错误
        2 用户中断
    """
    try:
        # 1) 解析命令行
        args = parse_args(argv)
        logger.info(
            "启动参数解析完成：n_studies=%d, seed=%d, biased_portion=%.2f, output=%s",
            args.n_studies, args.seed, args.biased_portion,
            args.output or "屏幕展示"
        )

        # 2) 数据模拟
        sim = FunnelDataSimulator(
            n_studies=args.n_studies,
            seed=args.seed,
            biased_portion=args.biased_portion
        )
        df = sim.simulate()

        # 3) 样式与绘图
        style = FunnelPlotStyle()  # 可进一步从配置文件或 CLI 扩展
        plotter = FunnelPlotter(df, style=style)
        plotter.plot(save_path=args.output)

        logger.info("主流程执行成功，退出码=0")
        return 0
    except KeyboardInterrupt:
        logger.warning("检测到用户中断 (KeyboardInterrupt)，退出码=2")
        return 2
    except SystemExit as se:
        # argparse 使用 parser.error() 时会触发 SystemExit
        logger.error("参数解析或帮助退出：%s", se)
        return se.code if isinstance(se.code, int) else 1
    except ValueError as ve:
        logger.error("参数校验失败：%s", ve)
        return 1
    except RuntimeError as re:
        logger.error("运行时错误：%s", re, exc_info=True)
        return 1
    except Exception as ex:
        logger.exception("未知错误导致程序异常终止：%s", ex)
        return 1


if __name__ == "__main__":
    # 将 main() 的返回值作为退出码，确保外部调用可捕获
    sys.exit(main())
