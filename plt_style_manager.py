#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块：plt_style_manager.py

提供一个线程安全的全局可视化风格管理器，适用于 Matplotlib 和 Seaborn。

职责：
  - 定义并存储统一的可视化风格配置。
  - 动态探测可用字体，并按顺序回退（中文、符号、英文字体）。
  - 设置 Seaborn 的主题、调色板和上下文。
  - 更新 Matplotlib rcParams，包括 DPI、背景色、字体、字号、间距和网格配置。
  - 提供预设风格（例如海报风格）。

示例：
    from utils.plt_style_manager import StyleConfig, StyleManager

    cfg = StyleConfig(
        dpi=150,
        theme="ticks",
        palette="pastel",
        context="notebook",
        grid=False,
        font_family_cn=["Microsoft YaHei", "SimHei"],
        font_family_symbol=["Segoe UI Symbol", "Symbola"],
        font_family_en=["DejaVu Sans", "Arial"],
        title_size=20,
        label_size=14,
        tick_label_size=12,
        legend_size=12,
        face_color="#f9f9f9",
        ax_face_color="#ffffff"
    )
    StyleManager.apply(cfg)
"""

from dataclasses import dataclass, field
from threading import RLock
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm


@dataclass(frozen=True)
class StyleConfig:
    """
    全局可视化风格配置数据容器。

    属性:
        theme:              Seaborn 主题风格名称（如 "ticks", "whitegrid" 等）。
        palette:            Seaborn 调色板名称（如 "muted", "deep" 等）。
        context:            Seaborn 上下文环境（如 "notebook", "talk" 等）。

        dpi:                图形分辨率（dots per inch）。
        face_color:         整体 Figure 背景色。
        ax_face_color:      Axes 区域背景色。
        ax_edge_color:      Axes 边框颜色。

        font_family_cn:     中文字体候选列表，按优先级排列。
        font_family_symbol: 符号/Emoji 字体候选列表。
        font_family_en:     英文字体候选列表。

        title_size:         坐标轴标题字号。
        label_size:         坐标轴标签字号。
        tick_label_size:    刻度标签字号。
        legend_size:        图例文字字号。
        title_pad:          标题与 Axes 上边缘的距离。
        label_pad:          标签与轴线之间的距离。

        grid:               是否显示网格。
        grid_color:         网格主线颜色。
        grid_linestyle:     网格主线线型。
        grid_linewidth:     网格主线宽度。
        grid_alpha:         网格主线透明度。
    """
    # —— Seaborn 风格 —— #
    theme: str = "ticks"
    palette: str = "muted"
    context: str = "notebook"

    # —— 分辨率与背景 —— #
    dpi: int = 150
    face_color: str = "#f9f9f9"
    ax_face_color: str = "#ffffff"
    ax_edge_color: str = "#cccccc"

    # —— 字体列表（候选） —— #
    font_family_cn: List[str] = field(default_factory=lambda: ["Microsoft YaHei", "SimHei", "STSong"])
    font_family_symbol: List[str] = field(default_factory=lambda: ["Segoe UI Symbol", "Symbola"])
    font_family_en: List[str] = field(default_factory=lambda: ["DejaVu Sans", "Arial"])

    # —— 尺寸与间距 —— #
    title_size: int = 18
    label_size: int = 14
    tick_label_size: int = 12
    legend_size: int = 12
    title_pad: float = 15.0
    label_pad: float = 10.0

    # —— 网格 —— #
    grid: bool = True
    grid_color: str = "#dddddd"
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.6
    grid_alpha: float = 0.4


class StyleManager:
    """
    线程安全的可视化风格管理器，将 StyleConfig 应用到 Seaborn 和 Matplotlib。

    特性:
      - 使用 RLock 保证多线程环境下的安全。
      - 动态探测并回落可用字体（中文、符号、英文）。
      - 统一管理 DPI、背景、字体、字号、间距、网格等全局样式。
    """
    _lock = RLock()

    @staticmethod
    def _find_installed(preferred: List[str]) -> List[str]:
        """
        从候选字体列表中筛选出系统实际安装且 Matplotlib 可识别的字体名称。

        :param preferred: 字体名称候选列表，按优先级顺序排列。
        :return: 系统中已安装并被 Matplotlib 识别的字体列表，保持原有顺序。
        """
        # 获取系统所有字体文件对应的字体名称
        available = {fm.FontProperties(fname=fp).get_name() for fp in fm.findSystemFonts()}
        # 仅保留可用字体
        return [name for name in preferred if name in available]

    @classmethod
    def apply(cls, cfg: StyleConfig) -> None:
        """
        线程安全地应用给定的可视化风格配置。

        步骤:
          1) 设置 Seaborn 全局主题、调色板和上下文。
          2) 动态探测并合并字体列表：中文 → 符号 → 英文。
          3) 更新 Matplotlib rcParams：
             - 分辨率 (dpi)、背景色 (figure/axes)
             - 字体族 (font.family, font.sans-serif) 及负号显示
             - 标题/标签/刻度/图例 大小与间距
          4) 根据 cfg.grid 打开或关闭网格，并应用主/次网格样式。

        :param cfg: StyleConfig 实例，包含所有样式设置。
        :return: None
        """
        with cls._lock:
            # 1) 设置 Seaborn 主题
            sns.set_theme(style=cfg.theme, palette=cfg.palette, context=cfg.context)

            # 2) 探测可用字体列表
            cn_fonts = cls._find_installed(cfg.font_family_cn)
            sym_fonts = cls._find_installed(cfg.font_family_symbol)
            en_fonts = cls._find_installed(cfg.font_family_en)

            # 3) 更新 Matplotlib 全局 rcParams
            rc = plt.rcParams
            rc.update({
                "figure.dpi": cfg.dpi,
                "figure.facecolor": cfg.face_color,
                "axes.facecolor": cfg.ax_face_color,
                "axes.edgecolor": cfg.ax_edge_color,
                "font.family": "sans-serif",
                "font.sans-serif": cn_fonts + sym_fonts + en_fonts,
                "axes.unicode_minus": False,
                "axes.titlesize": cfg.title_size,
                "axes.titlepad": cfg.title_pad,
                "axes.labelsize": cfg.label_size,
                "axes.labelpad": cfg.label_pad,
                "xtick.labelsize": cfg.tick_label_size,
                "ytick.labelsize": cfg.tick_label_size,
                "legend.fontsize": cfg.legend_size,
            })

            # 4) 网格配置
            if cfg.grid:
                rc.update({
                    "axes.grid": True,
                    "grid.color": cfg.grid_color,
                    "grid.linestyle": cfg.grid_linestyle,
                    "grid.linewidth": cfg.grid_linewidth,
                    "grid.alpha": cfg.grid_alpha,
                })
                # 确保网格绘制在数据后面
                sns.set_style(rc={"grid.which": "both", "grid.axis": "both"})
            else:
                rc["axes.grid"] = False

    @classmethod
    def preset_poster(cls) -> StyleConfig:
        """
        返回适用于海报或大屏展示的大尺寸风格预设。

        特点:
          - 高分辨率 (dpi=300)
          - 更大字号的标题与标签
          - 关闭网格使背景更简洁

        :return: 一个预配置好的 StyleConfig 实例
        """
        return StyleConfig(
            dpi=300,
            title_size=24,
            label_size=20,
            tick_label_size=16,
            legend_size=18,
            grid=False,
            face_color="#ffffff",
            ax_face_color="#f0f0f0",
        )
