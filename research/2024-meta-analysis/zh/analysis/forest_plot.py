#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名称：forest_plot_meta_analysis.py

文件描述：
    基于固定效应（Fixed Effect, FE）和 DerSimonian–Laird 随机效应（Random Effect, RE）模型，实现以下功能：
      1. 调用数据模拟模块生成元分析研究的模拟数据（效应量 yi、标准误 se、95% 置信区间）；
      2. 计算并封装 FE 与 RE 模型的合并效应、方差、异质性指标（τ²、Q、I²）及预测区间；
      3. 绘制专业美观的横向森林图（Forest Plot），可同时展示 FE/RE 汇总结果及个体研究数据；
      4. 集成企业级日志（LoggerFactory）与全局可视化风格（StyleManager），
         并在关键环节添加严格异常捕获，确保脚本在生产环境中的健壮性与可复现性。

适用场景：
    - 医学、公共卫生、社会科学等领域的元分析研究结果可视化；
    - 自动化科研报告生成或 Web 服务实时绘图；
    - 高级学术论文与企业级数据报告。

主要依赖：
    numpy, pandas, scipy, matplotlib, seaborn, utils.data_simulation, utils.logger_factory, utils.plt_style_manager

使用方法：
    直接运行：
        python forest_plot_meta_analysis.py
    或在其他脚本中按需调用其中的 MetaAnalysis 与 ForestPlotter 类。

作者：智能麻花
日期：2025-05-14
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from utils.data_simulation import generate_simulated_data
from utils.logger_factory import LoggerFactory
from utils.plt_style_manager import StyleConfig, StyleManager

# Todo # --- 日志模块初始化 + 全局样式配置 --- #
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
    logfile="logs/forest_plot_meta_analysis.log",  # 写入文件
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


@dataclass(frozen=True)
class ModelResult:
    """
    元分析模型结果容器，封装单次模型计算的所有输出指标。

    Attributes:
        model (str):                   模型类型，"FE" 或 "RE"。
        effect (float):                合并效应量估计值 μ。
        var (float):                   合并效应量方差 Var(μ)。
        ci_lower (float):              置信区间下限 μ - z*sqrt(var)。
        ci_upper (float):              置信区间上限 μ + z*sqrt(var)。
        tau2 (float|None):             异质性方差 τ²（FE 模型为 None）。
        Q (float|None):                Cochran Q 统计量（FE 模型为 None）。
        I2 (float|None):               I² 异质性比例（FE 模型为 None）。
        prediction_lower (float|None): 预测区间下限（RE 模型专用）。
        prediction_upper (float|None): 预测区间上限（RE 模型专用）。
        weights_sum (float):           权重之和 ∑w_i。
        n_studies (int):               研究数量。
        ci_level (float):              置信水平 α，例如 0.95。
    """
    model: str
    effect: float
    var: float
    ci_lower: float
    ci_upper: float
    tau2: Optional[float]
    Q: Optional[float]
    I2: Optional[float]
    prediction_lower: Optional[float]
    prediction_upper: Optional[float]
    weights_sum: float
    n_studies: int
    ci_level: float

    def __post_init__(self):
        # 基本一致性校验
        if not (0 < self.ci_level < 1):
            logger.error("ModelResult 初始化失败：ci_level=%s 不在 (0,1)", self.ci_level)
            raise ValueError(f"ci_level 应在 (0,1)，当前={self.ci_level}")
        if self.ci_lower > self.ci_upper:
            logger.error("ModelResult 初始化失败：下限 %.4f 大于上限 %.4f", self.ci_lower, self.ci_upper)
            raise ValueError("ci_lower 不应大于 ci_upper")
        if self.n_studies < 1:
            logger.error("ModelResult 初始化失败：n_studies=%d < 1", self.n_studies)
            raise ValueError("n_studies 必须 >= 1")


class MetaAnalysis:
    """元分析计算器，支持固定效应(FE)和 DerSimonian–Laird 随机效应(RE)模型。"""

    def __init__(self, df: pd.DataFrame, ci_level: float = 0.95):
        """
        初始化元分析实例。

        :param df: 包含列 ["study","yi","se"] 的 DataFrame。
        :param ci_level: 置信水平 (0,1)，如 0.95。
        :raises ValueError: 当输入格式不正确或 ci_level 越界时抛出。
        """
        required = {"study", "yi", "se"}
        missing = required - set(df.columns)
        if missing:
            msg = f"输入 DataFrame 缺失列: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        if not (0 < ci_level < 1):
            msg = f"ci_level 必须在 (0,1)，当前={ci_level}"
            logger.error(msg)
            raise ValueError(msg)

        # 深拷贝并缓存
        self._df = df.copy()
        self.n = len(df)
        self.ci_level = ci_level
        self._cache = {}
        logger.info("MetaAnalysis 初始化完成：n_studies=%d, ci_level=%.2f", self.n, self.ci_level)

    def _z_value(self) -> float:
        """根据置信水平计算正态分布分位数 z。"""
        alpha = 1.0 - self.ci_level
        return st.norm.ppf(1 - alpha / 2)

    def fixed_effects(self) -> ModelResult:
        """
        计算固定效应模型 (FE) 的合并效应及统计量。

        :return: ModelResult 对象，model="FE"。
        :raises ZeroDivisionError: 若所有权重和为零时抛出。
        """
        if "FE" in self._cache:
            return self._cache["FE"]

        logger.info("开始计算 Fixed Effect 模型")
        try:
            yi = self._df["yi"].to_numpy()
            se = self._df["se"].to_numpy()
            if np.any(se <= 0):
                msg = "标准误 se 必须均为正数"
                logger.error(msg + f"，发现值={se}")
                raise ValueError(msg)

            # 计算权重与加权平均
            w = 1.0 / (se ** 2)
            W = w.sum()
            if W <= 0:
                logger.error("FE 模型计算失败：权重之和 W=%.4e", W)
                raise ZeroDivisionError("权重之和为零，无法计算 FE 模型")

            effect = (w * yi).sum() / W
            var = 1.0 / W
            z = self._z_value()
            ci_low, ci_high = effect - z * np.sqrt(var), effect + z * np.sqrt(var)

            logger.debug("FE 中间量：W=%.4e, effect=%.4f, var=%.4e, z=%.4f", W, effect, var, z)

            result = ModelResult(
                model="FE",
                effect=effect, var=var,
                ci_lower=ci_low, ci_upper=ci_high,
                tau2=None, Q=None, I2=None,
                prediction_lower=None, prediction_upper=None,
                weights_sum=W, n_studies=self.n,
                ci_level=self.ci_level
            )
            logger.info("FE 计算完成：effect=%.4f, 95%% CI=[%.4f, %.4f]", effect, ci_low, ci_high)
            self._cache["FE"] = result
            return result
        except Exception as ex:
            logger.error("FE 模型计算异常：%s", str(ex), exc_info=True)
            raise

    def random_effects(self) -> ModelResult:
        """
        计算 DerSimonian–Laird 随机效应模型 (RE) 的合并效应及统计量。

        :return: ModelResult 对象，model="RE"。
        :raises ZeroDivisionError: 若随机效应权重和为零时抛出。
        """
        if "RE" in self._cache:
            return self._cache["RE"]

        logger.info("开始计算 Random Effect 模型")
        try:
            fe = self.fixed_effects()
            yi = self._df["yi"].to_numpy()
            se = self._df["se"].to_numpy()

            # Cochran Q
            w = 1.0 / (se ** 2)
            Q = (w * (yi - fe.effect) ** 2).sum()
            dfree = self.n - 1
            denom = w.sum() - (w ** 2).sum() / w.sum()
            tau2 = max(0.0, (Q - dfree) / denom) if denom > 0 else 0.0

            # RE 权重与合并效应
            w_re = 1.0 / (se ** 2 + tau2)
            W_re = w_re.sum()
            if W_re <= 0:
                logger.error("RE 模型计算失败：权重之和 W_re=%.4e", W_re)
                raise ZeroDivisionError("随机效应权重之和为零，无法计算 RE 模型")

            effect_re = (w_re * yi).sum() / W_re
            var_re = 1.0 / W_re

            # 异质性指标
            I2 = max(0.0, (Q - dfree) / Q) if Q > dfree else 0.0

            # 预测区间: μ ± t_{dfree,1-α/2} * sqrt(var_re + τ²)
            t = st.t.ppf(1 - (1 - self.ci_level) / 2, dfree)
            pred_se = np.sqrt(var_re + tau2)
            pred_low, pred_high = effect_re - t * pred_se, effect_re + t * pred_se

            # 置信区间
            z = self._z_value()
            ci_low, ci_high = effect_re - z * np.sqrt(var_re), effect_re + z * np.sqrt(var_re)

            result = ModelResult(
                model="RE",
                effect=effect_re, var=var_re,
                ci_lower=ci_low, ci_upper=ci_high,
                tau2=tau2, Q=Q, I2=I2,
                prediction_lower=pred_low, prediction_upper=pred_high,
                weights_sum=W_re, n_studies=self.n,
                ci_level=self.ci_level
            )
            logger.info("RE 计算完成：effect=%.4f, 95%% CI=[%.4f, %.4f], τ²=%.4e", effect_re, ci_low, ci_high, tau2)
            self._cache["RE"] = result
            return result
        except Exception as ex:
            logger.error("RE 模型计算异常：%s", str(ex), exc_info=True)
            raise

    @property
    def df(self) -> pd.DataFrame:
        """用于绘图的原始 DataFrame（返回副本以保护内部状态）。"""
        return self._df.copy()


@dataclass(frozen=True)
class ForestPlotStyle:
    """
    森林图专属样式配置类（仅管理布局与视觉元素），与全局字体、网格、背景分离，确保样式集中、易维护。

    Attributes:
        fig_width (float):               图表宽度，单位英寸。
        min_fig_height (float):          最小图表高度，单位英寸。
        row_height (float):              每条研究行高度。
        summary_height (float):          每条汇总行高度。
        summary_vgap (float):            汇总区与研究区垂直间距。

        row_even_color (str):            行背景-偶数行颜色。
        row_odd_color (str):             行背景-奇数行颜色。

        individual_marker (str):         个体研究点标记形状。
        individual_marker_size (float):  个体研究点标记大小。
        individual_capsize (float):      误差条端帽长度。
        individual_color (str):          个体研究点和误差条颜色。
        individual_ecolor (str):         个体研究误差条颜色。
        ci_z (float):                    置信区间 z 值（通常 1.96）。
        font_size_individual (int):      个体研究数值标注字号。

        summary_markers (Dict[str,str]): 汇总模型标记形状 {"FE":..., "RE":...}。
        summary_colors (Dict[str,str]):  汇总模型颜色 {"FE":..., "RE":...}。
        summary_linewidth (float):       汇总区间主线宽度。
        summary_alpha (float):           汇总菱形透明度。
        summary_marker_size (float):     汇总菱形中心大标记大小。

        zero_line_color (str):           零参考线颜色。
        font_size_label (int):           轴标签与图例字号。
        font_size_title (int):           标题字号。
        legend_fontsize (int):           图例文字字号。
    """
    # 画布与布局
    fig_width: float = 12.0
    min_fig_height: float = 6.0
    row_height: float = 0.6
    summary_height: float = 1.2
    summary_vgap: float = 0.8

    # 行背景
    row_even_color: str = "#f7f7f7"
    row_odd_color: str = "#ffffff"

    # 个体研究
    individual_marker: str = "o"
    individual_marker_size: float = 6.0
    individual_capsize: float = 3.0
    individual_color: str = "#2A6F97"
    individual_ecolor: str = "#A0A0A0"
    ci_z: float = 1.96
    font_size_individual: int = 10

    # 汇总模型
    summary_markers: Dict[str, str] = field(default=None)
    summary_colors: Dict[str, str] = field(default=None)
    summary_linewidth: float = 3.0
    summary_alpha: float = 0.5
    summary_marker_size: float = 100.0

    # 其它
    zero_line_color: str = "#555555"
    font_size_label: int = 14
    font_size_title: int = 20
    legend_fontsize: int = 12

    def __post_init__(self):
        # 校验数值均非负
        for attr in ["fig_width", "min_fig_height", "row_height", "summary_height",
                     "row_even_color", "row_odd_color",
                     "individual_marker_size", "individual_capsize",
                     "summary_linewidth", "summary_alpha", "summary_marker_size"]:
            val = getattr(self, attr)
            if isinstance(val, (int, float)) and val < 0:
                logger.error("ForestPlotStyle 初始化失败：%s=%.4f 非法", attr, val)
                raise ValueError(f"{attr} 必须非负，当前={val}")

        # 默认色盲友好配色
        if self.summary_colors is None:
            object.__setattr__(self, "summary_colors", {
                "FE": "#E69F00",  # 橙色（固定效应）
                "RE": "#56B4E9",  # 蓝色（随机效应）
            })

        if self.summary_markers is None:
            object.__setattr__(self, "summary_markers", {
                "FE": "D",  # Diamond
                "RE": "D",
            })


class ForestPlotter:
    """
    横向森林图绘制器，将 MetaAnalysis 的 FE/RE 结果可视化。

    主要职责：
      1. 数据预处理：倒序研究顺序并替换中文标签；
      2. 画布初始化：动态计算高度，隐藏多余脊柱；
      3. 背景渲染：交替行背景提升可读性；
      4. 个体研究绘制：误差条、散点及数值标注；
      5. 汇总模型绘制：置信区间主线、菱形区域、大号标记及数值注释；
      6. 最终美化：坐标轴、标题、零线、图例及保存/展示。
    """

    def __init__(
            self,
            analysis: "MetaAnalysis",
            style: ForestPlotStyle = ForestPlotStyle(),
            show_fe: bool = True,
            show_re: bool = True
    ):
        """
        初始化绘图器。

        :param analysis: MetaAnalysis 实例，必须提供 fixed_effects() 和 random_effects() 方法。
        :param style: ForestPlotStyle 实例，所有视觉参数集中管理。
        :param show_fe: 是否绘制固定效应汇总（FE）。
        :param show_re: 是否绘制随机效应汇总（RE）。
        :raises TypeError: 当 analysis 不包含预期方法时。
        """
        # 验证 analysis
        if not (hasattr(analysis, "fixed_effects") and hasattr(analysis, "random_effects")):
            logger.error("ForestPlotter 初始化失败：analysis 不支持固定/随机效应方法")
            raise TypeError("analysis 必须为 MetaAnalysis 实例")

        self.analysis = analysis
        self.style = style
        self.show_fe = show_fe
        self.show_re = show_re

        logger.info("ForestPlotter 初始化成功：show_fe=%s, show_re=%s", show_fe, show_re)

    def plot(self, save_path: Optional[str] = None):
        """
        主入口：按流程绘制森林图并保存或展示。

        :param save_path: 若指定则将图输出到文件，否则直接 plt.show()。
        :raises RuntimeError: 绘图或保存过程中出现任何异常时抛出。
        """
        logger.info("开始绘制森林图")
        try:
            # 1) 数据预处理
            df = self._prepare_data()
            logger.debug("处理后研究数：%d", len(df))

            # 2) 收集汇总模型
            summaries = self._collect_summaries()
            logger.debug("待绘制汇总模型：%s", [key for _, key in summaries])

            # 3) 初始化 Figure/Axis
            fig, ax = self._init_figure(n_rows=len(df), n_summaries=len(summaries))
            logger.debug("画布尺寸：%s", fig.get_size_inches())

            # 4) 渲染背景
            self._shade_background(ax, count=len(df))

            # 5) 绘制个体研究
            indiv_handle = self._draw_individuals(ax, df)

            # 6) 绘制汇总模型
            summary_handles = self._draw_summaries(ax, summaries)

            # 7) 最终美化 & 输出
            self._finalize(ax, df, summaries, indiv_handle, summary_handles)
            fig.tight_layout(pad=2)

            # 灰色外边
            border = mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=fig.transFigure,  # 坐标系统为 figure 相对坐标
                facecolor="none",  # 透明填充
                edgecolor="#909090",  # 边框颜色：中灰
                linewidth=1.5  # 边框粗细
            )
            fig.add_artist(border)

            if save_path:
                fig.savefig(save_path, dpi=300)
                logger.info("森林图已保存至：%s", save_path)
            else:
                plt.show()
                logger.info("森林图展示完成")
        except Exception as ex:
            logger.error("森林图绘制失败：%s", str(ex), exc_info=True)
            raise RuntimeError("ForestPlotter.plot 执行失败") from ex

    def _prepare_data(self) -> pd.DataFrame:
        """
        数据预处理：深拷贝、倒序、英文标签替换为“研究 N”。

        :return: 处理后的 DataFrame 副本。
        """
        df = self.analysis.df.copy().iloc[::-1].reset_index(drop=True)
        # 将 "Study N" 转为中文 "研究 N"
        df["study"] = df["study"].str.replace(r"Study (\d+)", r"研究 \1", regex=True)
        return df

    def _collect_summaries(self) -> List[Tuple["ModelResult", str]]:
        """
        根据 show_fe 和 show_re 标志，收集 FE / RE 模型结果及其标识。

        :return: 列表元素为 (ModelResult, 模型标签)。
        """
        out = []
        if self.show_fe:
            out.append((self.analysis.fixed_effects(), "FE"))
        if self.show_re:
            out.append((self.analysis.random_effects(), "RE"))
        return out

    def _init_figure(self, n_rows: int, n_summaries: int):
        """
        动态计算画布高度并创建 Figure/Axis。

        :param n_rows: 个体研究数。
        :param n_summaries: 汇总模型数。
        :return: (fig, ax)
        """
        base_h = n_rows * self.style.row_height + n_summaries * self.style.summary_height
        height = max(self.style.min_fig_height, base_h) + self.style.summary_vgap
        fig, ax = plt.subplots(figsize=(self.style.fig_width, height))
        # 调整边距并隐藏上/右脊柱，突出底/左脊柱
        fig.subplots_adjust(left=0.32, right=0.94, top=0.88, bottom=0.12)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.1)
        ax.spines["left"].set_linewidth(1.1)
        return fig, ax

    def _shade_background(self, ax, count: int):
        """
        交替绘制行背景，提升可读性。

        :param ax: matplotlib Axes 对象。
        :param count: 行数。
        """
        for i in range(count):
            c = self.style.row_even_color if i % 2 == 0 else self.style.row_odd_color
            # 每行高度 0.8，居中绘制
            ax.axhspan(i - 0.4, i + 0.4, color=c, zorder=0)

    def _draw_individuals(self, ax, df: pd.DataFrame) -> Line2D:
        """
        绘制个体研究的误差条、散点及数值标注。

        :param ax: matplotlib Axes 对象。
        :param df: 预处理后的 DataFrame，包含 "yi", "se"。
        :return: 最后一个散点的 handle，用于图例。
        """
        last_handle = None
        z = self.style.ci_z
        # 预先获取 x 轴范围用于数值文本偏移
        x0, x1 = ax.get_xlim()
        for idx, row in df.iterrows():
            ci = z * row["se"]

            # 绘制误差条
            eb = ax.errorbar(
                row["yi"], idx,
                xerr=ci,
                fmt=self.style.individual_marker,
                color=self.style.individual_color,
                ecolor=self.style.individual_ecolor,
                capsize=self.style.individual_capsize,
                lw=1.0, zorder=3
            )

            # 绘制散点
            sc = ax.scatter(
                row["yi"], idx,
                s=self.style.individual_marker_size ** 2,
                marker=self.style.individual_marker,
                facecolor=self.style.individual_color,
                edgecolor="white",
                lw=0.8, zorder=4
            )

            # 数值标注
            ax.text(
                row["yi"] + ci + 0.02 * (x1 - x0),
                idx, f"{row['yi']:.2f}",
                va="center", ha="left",
                fontsize=self.style.font_size_individual,
                color=self.style.individual_color,
                zorder=5
            )
            last_handle = sc
        return last_handle

    def _draw_summaries(self, ax, summaries: List[Tuple[ModelResult, str]]) -> List[Line2D]:
        """
        绘制 FE/RE 汇总区间主线、菱形区域及大号中心标记，并注释汇总效应值。

        :param ax: matplotlib Axes 对象。
        :param summaries: (ModelResult, key) 列表。
        :return: 中心标记的 handles 列表，用于图例。
        """
        handles = []
        for i, (res, key) in enumerate(summaries):
            y = -(i + 1)
            lo, hi = res.ci_lower, res.ci_upper
            col = self.style.summary_colors[key]

            # 主线
            ax.hlines(y, lo, hi, colors=col, linewidth=self.style.summary_linewidth, zorder=4)

            # 菱形区域
            mid = res.effect
            diamond = [
                ((lo + mid) / 2, y),
                (mid, y + 0.3),
                ((hi + mid) / 2, y),
                (mid, y - 0.3),
            ]
            ax.add_patch(Polygon(
                diamond,
                closed=True,
                facecolor=col,
                edgecolor=col,
                alpha=self.style.summary_alpha,
                zorder=5
            ))

            # 中心大标记
            sc = ax.scatter(
                mid, y,
                s=self.style.summary_marker_size,
                marker=self.style.summary_markers[key],
                facecolor="white",
                edgecolor=col,
                linewidth=2.0,
                zorder=6
            )

            # 效应值标注
            ax.text(
                mid, y + 0.45,
                f"{mid:.2f}",
                ha="center", va="bottom",
                fontsize=self.style.font_size_label,
                color=col,
                weight="bold",
                zorder=7
            )
            handles.append(sc)
        return handles

    def _finalize(self,
                  ax,
                  df: pd.DataFrame,
                  summaries: List[Tuple[ModelResult, str]],
                  indiv_handle: Line2D,
                  summary_handles: List[Line2D]):
        """
        设置坐标轴范围、标签、标题、零参考线及图例。

        :param ax: matplotlib Axes 对象。
        :param df: 预处理后的 DataFrame。
        :param summaries: 汇总模型列表。
        :param indiv_handle: 个体研究散点 handle。
        :param summary_handles: 汇总模型散点 handle 列表。
        """
        # X 轴范围及 padding
        lowers = df["ci_lower"].tolist() + [r.ci_lower for r, _ in summaries]
        uppers = df["ci_upper"].tolist() + [r.ci_upper for r, _ in summaries]
        xmin, xmax = min(lowers), max(uppers)
        pad = 0.12 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)

        # Y 轴范围
        ax.set_ylim(-len(summaries) - 0.5, len(df) - 0.5)

        # Y 轴标签
        yt = list(range(len(df))) + [-(i + 1) for i in range(len(summaries))]
        ylabels = df["study"].tolist()
        if self.show_fe: ylabels.append("固定效应 汇总")
        if self.show_re: ylabels.append("随机效应 汇总")
        ax.set_yticks(yt)
        ax.set_yticklabels(ylabels, fontsize=self.style.font_size_label)

        # 标题与坐标轴标签
        ax.set_title("森林图 (Forest Plot)", fontsize=self.style.font_size_title, weight="semibold", pad=12)
        ax.set_xlabel("效应量及 95% 置信区间", fontsize=self.style.font_size_label, labelpad=8)

        # 零参考线
        ax.axvline(0, linestyle="--", linewidth=1.0, color=self.style.zero_line_color, zorder=2)

        # 图例
        legend_handles = [indiv_handle] + summary_handles
        legend_labels = ["个体研究"] + [
            "固定效应 汇总" if key == "FE" else "随机效应 汇总"
            for _, key in summaries
        ]
        ax.legend(
            legend_handles, legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=self.style.legend_fontsize
        )


class CLIConfig:
    """
    命令行参数配置类。

    负责：
      1. 定义并解析所有命令行选项；
      2. 校验参数合法性；
      3. 将解析结果保存为属性，供主流程调用。

    Attributes:
        n_studies (int):   模拟研究数量，正整数。
        seed (int):        随机种子，非负整数。
        ci_level (float):  置信水平，0<ci_level<1。
        output (str|None): 输出文件路径；若为 None 则直接屏幕展示。
    """

    def __init__(self, argv: Optional[list] = None):
        """
        初始化并解析命令行参数。

        :param argv: 可选参数列表，默认使用 sys.argv[1:]。
        :raises SystemExit: 由 argparse 在 --help 或参数错误时触发退出。
        """
        parser = argparse.ArgumentParser(
            prog="forest_plot_meta_analysis.py",
            description="Forest Plot Meta-Analysis 工具：生成模拟数据并绘制 FE/RE 森林图"
        )
        parser.add_argument(
            "-n", "--n-studies",
            type=int, default=8,
            help="模拟研究数量 (正整数，默认: 8)"
        )
        parser.add_argument(
            "-s", "--seed",
            type=int, default=42,
            help="随机种子 (非负整数，默认: 42)"
        )
        parser.add_argument(
            "-c", "--ci-level",
            type=float, default=0.95,
            help="置信水平 (0<ci_level<1，默认: 0.95)"
        )
        parser.add_argument(
            "-o", "--output",
            type=str, default=None,
            help="森林图输出路径；不指定则直接在屏幕展示"
        )

        args = parser.parse_args(argv)

        # 参数校验
        if args.n_studies < 1:
            parser.error(f"--n-studies 参数必须 >=1，当前：{args.n_studies}")
        if args.seed < 0:
            parser.error(f"--seed 参数必须 >=0，当前：{args.seed}")
        if not (0.0 < args.ci_level < 1.0):
            parser.error(f"--ci-level 参数必须在 (0,1) 之间，当前：{args.ci_level}")

        # 赋值到实例属性
        self.n_studies: int = args.n_studies
        self.seed: int = args.seed
        self.ci_level: float = args.ci_level
        self.output: Optional[str] = args.output

        logger.debug(
            "CLIConfig 已解析：n_studies=%d, seed=%d, ci_level=%.2f, output=%s",
            self.n_studies, self.seed, self.ci_level,
            self.output or "None(屏幕展示)"
        )


def main(argv=None) -> int:
    """
    主程序：解析命令行参数 → 生成模拟数据 → 执行 MetaAnalysis → 绘制森林图。

    :param argv: 可选参数列表，默认使用 sys.argv[1:]。
    :return: 退出码，0=成功，1=错误，2=用户中断。
    """
    # 1) 解析命令行配置
    try:
        config = CLIConfig(argv)
        logger.info(
            "启动参数：n_studies=%d, seed=%d, ci_level=%.2f, output=%s",
            config.n_studies,
            config.seed,
            config.ci_level,
            config.output or "（屏幕展示）"
        )
    except SystemExit:
        # argparse 已打印帮助或错误，此处直接返回
        return 1
    except Exception as ex:
        logger.error("参数解析失败：%s", ex, exc_info=True)
        return 1

    try:
        # 2) 模拟数据
        df = generate_simulated_data(
            n_studies=config.n_studies,
            seed=config.seed,
            ci_level=config.ci_level
        )
        logger.info("模拟数据生成完成，共 %d 条记录。", len(df))
        logger.debug("模拟数据预览：\n%s", df.head().to_string(index=False))

        # 3) 元分析
        ma = MetaAnalysis(df, ci_level=config.ci_level)
        logger.info("MetaAnalysis 实例化成功 (ci_level=%.2f)", config.ci_level)

        # 4) 绘图
        fp = ForestPlotter(ma, show_fe=True, show_re=True)
        fp.plot(save_path=config.output)

        logger.info("森林图处理流程完成。")
        return 0
    except KeyboardInterrupt:
        logger.warning("用户中断程序 (KeyboardInterrupt)。")
        return 2
    except Exception as ex:
        logger.exception("主流程执行失败：%s", ex)
        return 1


if __name__ == "__main__":
    # 调用 main 并以其返回值作为退出码
    sys.exit(main())
