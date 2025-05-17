#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: plt_style_manager.py

Provides a thread-safe global style manager for Matplotlib and Seaborn.

Responsibilities:
  - Define and store unified visualization style settings.
  - Dynamically detect and fall back through available fonts (Chinese, symbols, English).
  - Apply Seaborn theme, palette, and context.
  - Apply Matplotlib rcParams for DPI, background colors, fonts, font sizes, paddings, and grid.
  - Offer preset configurations (e.g., poster style).

Example:
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

import warnings
from dataclasses import dataclass, field
from threading import RLock
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm


@dataclass(frozen=True)
class StyleConfig:
    """
    Data container for global visualization style settings.

    Attributes:
        theme:              Seaborn style theme (e.g., "ticks", "whitegrid").
        palette:            Seaborn color palette (e.g., "muted", "deep").
        context:            Seaborn context (e.g., "notebook", "talk", "paper").

        dpi:                Figure resolution in dots per inch.
        face_color:         Figure background color.
        ax_face_color:      Axes background color.
        ax_edge_color:      Axes edge (frame) color.

        font_family_cn:     List of candidate Chinese fonts, in priority order.
        font_family_symbol: List of candidate symbol/emoji fonts.
        font_family_en:     List of candidate English fonts.

        title_size:         Font size for axis titles.
        label_size:         Font size for axis labels.
        tick_label_size:    Font size for tick labels.
        legend_size:        Font size for legend text.
        title_pad:          Padding between title and top of axes.
        label_pad:          Padding between labels and axes.

        grid:               Whether to display grid lines.
        grid_color:         Color of major grid lines.
        grid_linestyle:     Line style for major grid lines.
        grid_linewidth:     Line width for major grid lines.
        grid_alpha:         Transparency for major grid lines.
    """
    # —— Seaborn style —— #
    theme: str = "ticks"
    palette: str = "muted"
    context: str = "notebook"

    # —— Resolution and Background —— #
    dpi: int = 150
    face_color: str = "#f9f9f9"
    ax_face_color: str = "#ffffff"
    ax_edge_color: str = "#cccccc"

    # —— Font list (candidates) —— #
    font_family_cn: List[str] = field(default_factory=lambda: ["Microsoft YaHei", "SimHei", "STSong"])
    font_family_symbol: List[str] = field(default_factory=lambda: ["Segoe UI Symbol", "Symbola"])
    font_family_en: List[str] = field(default_factory=lambda: ["DejaVu Sans", "Arial"])

    # —— Size and padding —— #
    title_size: int = 18
    label_size: int = 14
    tick_label_size: int = 12
    legend_size: int = 12
    title_pad: float = 15.0
    label_pad: float = 10.0

    # —— Grid —— #
    grid: bool = True
    grid_color: str = "#dddddd"
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.6
    grid_alpha: float = 0.4


class StyleManager:
    """
    Thread-safe manager to apply StyleConfig settings to Seaborn and Matplotlib.

    Methods:
        apply:           Apply the given StyleConfig globally.
        preset_poster:   Return a StyleConfig optimized for poster-size figures.
    """
    _lock = RLock()

    @staticmethod
    def _find_installed(preferred: List[str]) -> List[str]:
        """
        Filter the given font list to only those installed and recognized by Matplotlib.

        :param preferred: List of preferred font names in priority order.
        :return: A filtered list containing only installed fonts, preserving order.
        """
        available = {fm.FontProperties(fname=fp).get_name() for fp in fm.findSystemFonts()}
        return [name for name in preferred if name in available]

    @classmethod
    def apply(cls, cfg: StyleConfig) -> None:
        """
        Apply the style configuration to Seaborn and Matplotlib in a thread-safe way.

        Steps:
          1) Suppress Matplotlib "Glyph missing" warnings.
          2) Set Seaborn style, palette, and context.
          3) Detect installed fonts and compose final font.sans-serif list.
          4) Update Matplotlib rcParams:
             - DPI, figure and axes backgrounds/edges.
             - Font family settings and unicode minus.
             - Title/label/tick/legend sizes and paddings.
          5) Enable or disable grid and set major/minor grid styles.

        :param cfg: The StyleConfig instance containing desired settings.
        :return: None
        """
        with cls._lock:
            # 1) Suppress missing-glyph warnings to reduce noise
            warnings.filterwarnings(
                "ignore",
                message=r"Glyph .* missing from font.*",
                category=UserWarning,
            )

            # 2) Apply Seaborn global theme
            sns.set_theme(style=cfg.theme, palette=cfg.palette, context=cfg.context)

            # 3) Compose font lists by detecting installed fonts
            cn_fonts = cls._find_installed(cfg.font_family_cn)
            sym_fonts = cls._find_installed(cfg.font_family_symbol)
            en_fonts = cls._find_installed(cfg.font_family_en)

            # 4) Update Matplotlib rcParams
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

            # 5) Grid configuration
            if cfg.grid:
                rc.update({
                    "axes.grid": True,
                    "grid.color": cfg.grid_color,
                    "grid.linestyle": cfg.grid_linestyle,
                    "grid.linewidth": cfg.grid_linewidth,
                    "grid.alpha": cfg.grid_alpha,
                })
                sns.set_style(rc={"grid.which": "both", "grid.axis": "both"})
            else:
                rc["axes.grid"] = False

    @classmethod
    def preset_poster(cls) -> StyleConfig:
        """
        Return a poster-optimized StyleConfig.

        Suitable for large-format outputs (posters, slides):
          - High DPI (300)
          - Increased font sizes
          - No grid for cleaner background

        :return: A StyleConfig pre-populated for poster use.
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
