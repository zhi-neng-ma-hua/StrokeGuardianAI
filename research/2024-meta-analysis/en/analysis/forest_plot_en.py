#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name: forest_plot_meta_analysis.py

Description:
    Based on the Fixed Effect (FE) and DerSimonian–Laird Random Effect (RE) models, this script provides:
      1. Invocation of a data simulation module to generate meta-analysis study data
         (effect sizes yi, standard errors se, and 95% confidence intervals);
      2. Computation and encapsulation of combined effects, variances,
         heterogeneity metrics (τ², Q, I²), and prediction intervals for both FE and RE models;
      3. Rendering of a professional and aesthetically pleasing horizontal forest plot
         that can simultaneously display FE/RE summary results along with individual study data;
      4. Integration of enterprise-grade logging (LoggerFactory) and a global visualization style
         (StyleManager), with strict exception handling in key steps to ensure robustness
         and reproducibility in production environments.

Applicable Scenarios:
    - Visualization of meta-analysis results in medicine, public health, social sciences, etc.;
    - Automated scientific report generation or real-time plotting in web services;
    - Advanced academic publications and enterprise-level data reporting.

Key Dependencies:
    numpy, pandas, scipy, matplotlib, seaborn, utils.data_simulation,
    utils.logger_factory, utils.plt_style_manager

Usage:
    Run directly:
        python forest_plot_meta_analysis.py
    Or import and use the MetaAnalysis and ForestPlotter classes in other scripts as needed.

Author: Intelligent Mahua
Date: 2025-05-14
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

# Todo # --- Log module initialization --- #
# Auto-detect: when run as a script, use the filename as the logger name; otherwise use __name__
module_name = __name__ if __name__ != "__main__" else Path(__file__).stem

# Acquire Logger
#    - Root level controlled by LOG_LEVEL env var, default DEBUG
#    - Console outputs INFO and above, with color/highlight
#    - File outputs DEBUG and above, rotated daily, retaining 15 days of logs
#    - Format includes timestamp, thread, level, module:line, function, message
logger = LoggerFactory.get_logger(
    module_name,
    level=logging.DEBUG,  # Overall logger level
    console_level=logging.INFO,  # Console: INFO+
    logfile="logs/forest_plot_meta_analysis.log",
    file_level=logging.DEBUG,  # File: DEBUG+
    max_bytes=None,  # Maximum file size
    backup_count_bytes=3,  # Retain last 3 file rotations
    when="midnight",  # Rotate at midnight
    backup_count_time=7  # Retain 7 days of logs
)

# Todo # --- Global Style Configuration --- #
# Style configuration: fill in DPI, context, font options, grid, palette, colorblind-friendly scheme, etc.
try:
    cfg = StyleConfig(
        grid=False,
        palette="colorblind",
        context="talk"
    )
    StyleManager.apply(cfg)
    logger.info(
        "Global plotting style applied successfully: dpi=%d, palette=%s, context=%s",
        cfg.dpi, cfg.palette, cfg.context
    )
except Exception as e:
    logger.error(
        "Failed to apply global plotting style; using Matplotlib defaults: %s",
        str(e), exc_info=True
    )


@dataclass(frozen=True)
class ModelResult:
    """
    Container for meta-analysis model results, encapsulating all output metrics from a single computation.

    Attributes:
        model (str):                   Model type, "FE" or "RE".
        effect (float):                Estimated combined effect size μ.
        var (float):                   Variance of the combined effect Var(μ).
        ci_lower (float):              Lower bound of confidence interval μ - z*sqrt(var).
        ci_upper (float):              Upper bound of confidence interval μ + z*sqrt(var).
        tau2 (float|None):             Heterogeneity variance τ² (None for FE model).
        Q (float|None):                Cochran’s Q statistic (None for FE model).
        I2 (float|None):               I² heterogeneity proportion (None for FE model).
        prediction_lower (float|None): Lower bound of prediction interval (RE only).
        prediction_upper (float|None): Upper bound of prediction interval (RE only).
        weights_sum (float):           Sum of weights ∑wᵢ.
        n_studies (int):               Number of studies.
        ci_level (float):              Confidence level α, e.g. 0.95.
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
        # Basic consistency check
        if not (0 < self.ci_level < 1):
            logger.error("ModelResult initialization failed: ci_level=%s not in (0,1)", self.ci_level)
            raise ValueError(f"ci_level should be between (0,1), currently={self.ci_level}")
        if self.ci_lower > self.ci_upper:
            logger.error(
                "ModelResult initialization failed: lower bound %.4f is greater than upper bound %.4f",
                self.ci_lower, self.ci_upper
            )
            raise ValueError("ci_lower should not be greater than ci_upper")
        if self.n_studies < 1:
            logger.error("ModelResult initialization failed: n_studies=%d < 1", self.n_studies)
            raise ValueError("n_studies must be >= 1")


class MetaAnalysis:
    """Meta-analysis calculator supporting Fixed Effect (FE) and DerSimonian–Laird Random Effect (RE) models."""

    def __init__(self, df: pd.DataFrame, ci_level: float = 0.95):
        """
        Initialize a MetaAnalysis instance.

        :param df: A DataFrame containing columns ["study","yi","se"].
        :param ci_level: Confidence level (0,1), e.g., 0.95.
        :raises ValueError: Raised if the DataFrame format is incorrect or ci_level is out of bounds.
        """
        required = {"study", "yi", "se"}
        missing = required - set(df.columns)
        if missing:
            msg = f"Input DataFrame is missing columns: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        if not (0 < ci_level < 1):
            msg = f"ci_level must be within (0,1), current={ci_level}"
            logger.error(msg)
            raise ValueError(msg)

        # Deep copy and cache initialization
        self._df = df.copy()
        self.n = len(df)
        self.ci_level = ci_level
        self._cache = {}
        logger.info("MetaAnalysis initialized: n_studies=%d, ci_level=%.2f", self.n, self.ci_level)

    def _z_value(self) -> float:
        """Compute the normal distribution quantile z based on the confidence level."""
        alpha = 1.0 - self.ci_level
        return st.norm.ppf(1 - alpha / 2)

    def fixed_effects(self) -> ModelResult:
        """
        Compute the Fixed Effect (FE) model combined effect and statistics.

        :return: A ModelResult object with model="FE".
        :raises ZeroDivisionError: If the sum of weights is zero.
        """
        if "FE" in self._cache:
            return self._cache["FE"]

        logger.info("Starting Fixed Effect model computation")
        try:
            yi = self._df["yi"].to_numpy()
            se = self._df["se"].to_numpy()
            if np.any(se <= 0):
                msg = "Standard errors (se) must all be positive"
                logger.error(msg + f", found values={se}")
                raise ValueError(msg)

            # Calculate weights and weighted mean
            w = 1.0 / (se ** 2)
            W = w.sum()
            if W <= 0:
                logger.error("FE computation failed: sum of weights W=%.4e", W)
                raise ZeroDivisionError("Sum of weights is zero; cannot compute FE model")

            effect = (w * yi).sum() / W
            var = 1.0 / W
            z = self._z_value()
            ci_low, ci_high = effect - z * np.sqrt(var), effect + z * np.sqrt(var)

            logger.debug("FE intermediate values: W=%.4e, effect=%.4f, var=%.4e, z=%.4f", W, effect, var, z)

            result = ModelResult(
                model="FE",
                effect=effect, var=var,
                ci_lower=ci_low, ci_upper=ci_high,
                tau2=None, Q=None, I2=None,
                prediction_lower=None, prediction_upper=None,
                weights_sum=W, n_studies=self.n,
                ci_level=self.ci_level
            )
            logger.info("FE computation completed: effect=%.4f, 95%% CI=[%.4f, %.4f]", effect, ci_low, ci_high)
            self._cache["FE"] = result
            return result
        except Exception as ex:
            logger.error("Exception during FE computation: %s", str(ex), exc_info=True)
            raise

    def random_effects(self) -> ModelResult:
        """
        Compute the DerSimonian–Laird Random Effect (RE) model combined effect and statistics.

        :return: A ModelResult object with model="RE".
        :raises ZeroDivisionError: If the sum of random-effect weights is zero.
        """
        if "RE" in self._cache:
            return self._cache["RE"]

        logger.info("Starting Random Effect model computation")
        try:
            fe = self.fixed_effects()
            yi = self._df["yi"].to_numpy()
            se = self._df["se"].to_numpy()

            # Cochran’s Q statistic
            w = 1.0 / (se ** 2)
            Q = (w * (yi - fe.effect) ** 2).sum()
            dfree = self.n - 1
            denom = w.sum() - (w ** 2).sum() / w.sum()
            tau2 = max(0.0, (Q - dfree) / denom) if denom > 0 else 0.0

            # Random-effects weights and combined effect
            w_re = 1.0 / (se ** 2 + tau2)
            W_re = w_re.sum()
            if W_re <= 0:
                logger.error("RE computation failed: sum of weights W_re=%.4e", W_re)
                raise ZeroDivisionError("Sum of random-effect weights is zero; cannot compute RE model")

            effect_re = (w_re * yi).sum() / W_re
            var_re = 1.0 / W_re

            # Heterogeneity metric I²
            I2 = max(0.0, (Q - dfree) / Q) if Q > dfree else 0.0

            # Prediction interval: μ ± t_{dfree,1-α/2} * sqrt(var_re + τ²)
            t = st.t.ppf(1 - (1 - self.ci_level) / 2, dfree)
            pred_se = np.sqrt(var_re + tau2)
            pred_low, pred_high = effect_re - t * pred_se, effect_re + t * pred_se

            # Confidence interval
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
            logger.info(
                "RE computation completed: effect=%.4f, 95%% CI=[%.4f, %.4f], τ²=%.4e",
                effect_re, ci_low, ci_high, tau2
            )
            self._cache["RE"] = result
            return result
        except Exception as ex:
            logger.error("Exception during RE computation: %s", str(ex), exc_info=True)
            raise

    @property
    def df(self) -> pd.DataFrame:
        """Original DataFrame for plotting (returns a copy to protect internal state)."""
        return self._df.copy()


@dataclass(frozen=True)
class ForestPlotStyle:
    """
    Forest plot style configuration class (managing only layout and visual elements),
    separated from global fonts, grids, and backgrounds to ensure centralized, maintainable styling.

    Attributes:
        fig_width (float):               Width of the figure in inches.
        min_fig_height (float):          Minimum height of the figure in inches.
        row_height (float):              Height per study row.
        summary_height (float):          Height per summary row.
        summary_vgap (float):            Vertical gap between summary rows and study rows.

        row_even_color (str):            Background color for even rows.
        row_odd_color (str):             Background color for odd rows.

        individual_marker (str):         Marker style for individual studies.
        individual_marker_size (float):  Marker size for individual studies.
        individual_capsize (float):      Cap size for individual error bars.
        individual_color (str):          Color for individual markers and error bars.
        individual_ecolor (str):         Edge color for individual error bars.
        ci_z (float):                    Z-value for confidence interval (typically 1.96).
        font_size_individual (int):      Font size for individual study labels.

        summary_markers (Dict[str,str]): Marker shapes for summary models {"FE":..., "RE":...}.
        summary_colors (Dict[str,str]):  Colors for summary models {"FE":..., "RE":...}.
        summary_linewidth (float):       Line width for summary intervals.
        summary_alpha (float):           Alpha transparency for summary diamond.
        summary_marker_size (float):     Marker size for summary diamond center.

        zero_line_color (str):           Color of the zero reference line.
        font_size_label (int):           Font size for axis labels and legend text.
        font_size_title (int):           Font size for the figure title.
        legend_fontsize (int):           Font size for legend entries.
    """
    # Canvas and Layout
    fig_width: float = 12.0
    min_fig_height: float = 6.0
    row_height: float = 0.6
    summary_height: float = 1.2
    summary_vgap: float = 0.8

    # Row Background
    row_even_color: str = "#f7f7f7"
    row_odd_color: str = "#ffffff"

    # Individual research
    individual_marker: str = "o"
    individual_marker_size: float = 6.0
    individual_capsize: float = 3.0
    individual_color: str = "#2A6F97"
    individual_ecolor: str = "#A0A0A0"
    ci_z: float = 1.96
    font_size_individual: int = 10

    # Aggregate Model
    summary_markers: Dict[str, str] = field(default=None)
    summary_colors: Dict[str, str] = field(default=None)
    summary_linewidth: float = 3.0
    summary_alpha: float = 0.5
    summary_marker_size: float = 100.0

    # other
    zero_line_color: str = "#555555"
    font_size_label: int = 14
    font_size_title: int = 20
    legend_fontsize: int = 12

    def __post_init__(self):
        # Validate that all numeric values are non-negative
        for attr in ["fig_width", "min_fig_height", "row_height", "summary_height",
                     "individual_marker_size", "individual_capsize",
                     "summary_linewidth", "summary_alpha", "summary_marker_size"]:
            val = getattr(self, attr)
            if isinstance(val, (int, float)) and val < 0:
                logger.error("ForestPlotStyle initialization failed: %s=%.4f is invalid", attr, val)
                raise ValueError(f"{attr} must be non-negative, current={val}")

        # Default to a colorblind-friendly palette if not provided
        if self.summary_colors is None:
            object.__setattr__(self, "summary_colors", {
                "FE": "#E69F00",  # Orange (Fixed Effect)
                "RE": "#56B4E9",  # Blue (Random Effect)
            })
        if self.summary_markers is None:
            object.__setattr__(self, "summary_markers", {
                "FE": "D",  # Diamond
                "RE": "D",
            })


class ForestPlotter:
    """
    Horizontal forest plot renderer, visualizing the FE/RE results from a MetaAnalysis.

    Core responsibilities:
      1. Data preprocessing: reverse the order of studies and replace labels;
      2. Canvas initialization: dynamically calculate height and hide redundant spines;
      3. Background rendering: alternate row backgrounds to enhance readability;
      4. Individual study rendering: error bars, scatter points, and numeric annotations;
      5. Summary model rendering: confidence interval lines, diamond patches, large markers, and annotations;
      6. Final styling: configure axes, title, zero-reference line, legend, and save/display the figure.
    """

    def __init__(
            self,
            analysis: "MetaAnalysis",
            style: ForestPlotStyle = ForestPlotStyle(),
            show_fe: bool = True,
            show_re: bool = True
    ):
        """
        Initialize the ForestPlotter.

        :param analysis: A MetaAnalysis instance, must provide fixed_effects() and random_effects() methods.
        :param style: A ForestPlotStyle instance managing all visual parameters centrally.
        :param show_fe: Whether to render the Fixed Effect summary.
        :param show_re: Whether to render the Random Effect summary.
        :raises TypeError: If the analysis object does not support the required methods.
        """
        # Validate analysis interface
        if not (hasattr(analysis, "fixed_effects") and hasattr(analysis, "random_effects")):
            logger.error("ForestPlotter initialization failed: analysis lacks FE/RE methods")
            raise TypeError("analysis must be a MetaAnalysis instance")

        self.analysis = analysis
        self.style = style
        self.show_fe = show_fe
        self.show_re = show_re

        logger.info("ForestPlotter successfully initialized: show_fe=%s, show_re=%s", show_fe, show_re)

    def plot(self, save_path: Optional[str] = None):
        """
        Main entry point: render the forest plot following the full workflow,
        then save or display the figure.

        :param save_path: If specified, save the figure to file; otherwise, call plt.show().
        :raises RuntimeError: Raised if any error occurs during rendering or saving.
        """
        logger.info("Beginning forest plot rendering")
        try:
            # 1) Data preprocessing
            df = self._prepare_data()
            logger.debug("Number of studies after processing: %d", len(df))

            # 2) Collect summary models
            summaries = self._collect_summaries()
            logger.debug("Summary models to render: %s", [key for _, key in summaries])

            # 3) Initialize figure/axes
            fig, ax = self._init_figure(n_rows=len(df), n_summaries=len(summaries))
            logger.debug("Canvas size: %s", fig.get_size_inches())

            # 4) Shade background
            self._shade_background(ax, count=len(df))

            # 5) Draw individual studies
            indiv_handle = self._draw_individuals(ax, df)

            # 6) Draw summary models
            summary_handles = self._draw_summaries(ax, summaries)

            # 7) Final styling & output
            self._finalize(ax, df, summaries, indiv_handle, summary_handles)
            fig.tight_layout(pad=2)

            # Gray outside
            border = mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=fig.transFigure,  # Coordinate system is figure relative
                facecolor="none",  # Transparent fill
                edgecolor="#909090",  # Border color: medium gray
                linewidth=1.5  # Border thickness
            )
            fig.add_artist(border)

            if save_path:
                fig.savefig(save_path, dpi=300)
                logger.info("Forest plot saved to: %s", save_path)
            else:
                plt.show()
                logger.info("Forest plot displayed successfully")
        except Exception as ex:
            logger.error("Forest plot rendering failed: %s", str(ex), exc_info=True)
            raise RuntimeError("ForestPlotter.plot execution failed") from ex

    def _prepare_data(self) -> pd.DataFrame:
        """
        Data preprocessing: perform a deep copy, reverse the order, and replace English labels "Study N" with Chinese "研究 N".

        :return: A processed copy of the DataFrame.
        """
        df = self.analysis.df.copy().iloc[::-1].reset_index(drop=True)
        # Replace "Study N" with the Chinese "研究 N"
        # df["study"] = df["study"].str.replace(r"Study (\d+)", r"研究 \1", regex=True)
        return df

    def _collect_summaries(self) -> List[Tuple["ModelResult", str]]:
        """
        Collect FE/RE model results and their identifiers based on the show_fe and show_re flags.

        :return: A list of tuples, each containing (ModelResult, model label).
        """
        out = []
        if self.show_fe:
            out.append((self.analysis.fixed_effects(), "FE"))
        if self.show_re:
            out.append((self.analysis.random_effects(), "RE"))
        return out

    def _init_figure(self, n_rows: int, n_summaries: int):
        """
        Dynamically calculate the canvas height and create the Figure/Axis.

        :param n_rows: Number of individual studies.
        :param n_summaries: Number of summary models.
        :return: (fig, ax)
        """
        base_h = n_rows * self.style.row_height + n_summaries * self.style.summary_height
        height = max(self.style.min_fig_height, base_h) + self.style.summary_vgap
        fig, ax = plt.subplots(figsize=(self.style.fig_width, height))
        # Adjust margins and hide the top/right spines, emphasize the bottom/left spines
        fig.subplots_adjust(left=0.32, right=0.94, top=0.88, bottom=0.12)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.1)
        ax.spines["left"].set_linewidth(1.1)
        return fig, ax

    def _shade_background(self, ax, count: int):
        """
        Alternate row background shading to enhance readability.

        :param ax: Matplotlib Axes object.
        :param count: Number of rows.
        """
        for i in range(count):
            c = self.style.row_even_color if i % 2 == 0 else self.style.row_odd_color
            # Each row spans a height of 0.8 units, centered on the row index
            ax.axhspan(i - 0.4, i + 0.4, color=c, zorder=0)

    def _draw_individuals(self, ax, df: pd.DataFrame) -> Line2D:
        """
        Plot individual study error bars, scatter points, and numeric annotations.

        :param ax: Matplotlib Axes object.
        :param df: Preprocessed DataFrame containing "yi" and "se".
        :return: Handle of the last scatter point, used for the legend.
        """
        last_handle = None
        z = self.style.ci_z
        # Retrieve the x-axis limits first to calculate text offset
        x0, x1 = ax.get_xlim()
        for idx, row in df.iterrows():
            ci = z * row["se"]

            # Draw the error bar
            eb = ax.errorbar(
                row["yi"], idx,
                xerr=ci,
                fmt=self.style.individual_marker,
                color=self.style.individual_color,
                ecolor=self.style.individual_ecolor,
                capsize=self.style.individual_capsize,
                lw=1.0, zorder=3
            )

            # Draw the scatter point
            sc = ax.scatter(
                row["yi"], idx,
                s=self.style.individual_marker_size ** 2,
                marker=self.style.individual_marker,
                facecolor=self.style.individual_color,
                edgecolor="white",
                lw=0.8, zorder=4
            )

            # Numeric annotation
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
        Render the FE/RE summary confidence interval lines, diamond patches, and large central markers, then annotate the summary effect values.

        :param ax: Matplotlib Axes object.
        :param summaries: List of (ModelResult, key) tuples.
        :return: List of handles for the central markers, used in the legend.
        """
        handles = []
        for i, (res, key) in enumerate(summaries):
            y = -(i + 1)
            lo, hi = res.ci_lower, res.ci_upper
            col = self.style.summary_colors[key]

            # Main line
            ax.hlines(y, lo, hi, colors=col, linewidth=self.style.summary_linewidth, zorder=4)

            # Diamond patch
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

            # Large central marker
            sc = ax.scatter(
                mid, y,
                s=self.style.summary_marker_size,
                marker=self.style.summary_markers[key],
                facecolor="white",
                edgecolor=col,
                linewidth=2.0,
                zorder=6
            )

            # Effect value annotation
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
        Configure axis limits, labels, title, zero-reference line, and legend.

        :param ax: Matplotlib Axes object.
        :param df: Preprocessed DataFrame.
        :param summaries: List of summary model tuples.
        :param indiv_handle: Handle of the individual study scatter point.
        :param summary_handles: List of handles for summary model scatter points.
        """
        # X-axis range and padding
        lowers = df["ci_lower"].tolist() + [r.ci_lower for r, _ in summaries]
        uppers = df["ci_upper"].tolist() + [r.ci_upper for r, _ in summaries]
        xmin, xmax = min(lowers), max(uppers)
        pad = 0.12 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)

        # Y-axis range
        ax.set_ylim(-len(summaries) - 0.5, len(df) - 0.5)

        # Y-axis labels
        yt = list(range(len(df))) + [-(i + 1) for i in range(len(summaries))]
        ylabels = df["study"].tolist()
        if self.show_fe: ylabels.append("Fixed Effect Summary")
        if self.show_re: ylabels.append("Random Effect Summary")
        ax.set_yticks(yt)
        ax.set_yticklabels(ylabels, fontsize=self.style.font_size_label)

        # Title and axis labels
        ax.set_title("Forest Plot", fontsize=self.style.font_size_title, weight="semibold", pad=12)
        ax.set_xlabel("Effect Size and 95% Confidence Interval", fontsize=self.style.font_size_label, labelpad=8)

        # Zero reference line
        ax.axvline(0, linestyle="--", linewidth=1.0, color=self.style.zero_line_color, zorder=2)

        # Legend
        legend_handles = [indiv_handle] + summary_handles
        legend_labels = ["Individual Study"] + [
            "Fixed Effect Summary" if key == "FE" else "Random Effect Summary"
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
    Command-line parameters configuration class.

    Responsible for:
      1. Defining and parsing all command-line options;
      2. Validating parameter correctness;
      3. Storing parsed results as attributes for the main workflow to use.

    Attributes:
        n_studies (int):   Number of simulated studies, a positive integer.
        seed (int):        Random seed, a non-negative integer.
        ci_level (float):  Confidence level, 0 < ci_level < 1.
        output (str|None): Output file path; if None, the plot is displayed on screen.
    """

    def __init__(self, argv: Optional[list] = None):
        """
        Initialize and parse command-line arguments.

        :param argv: Optional list of arguments; defaults to sys.argv[1:].
        :raises SystemExit: Raised by argparse when --help is requested or on argument errors.
        """
        parser = argparse.ArgumentParser(
            prog="forest_plot_meta_analysis.py",
            description="Forest Plot Meta-Analysis Tool: Generate simulated data and draw FE/RE forest plots"
        )
        parser.add_argument(
            "-n", "--n-studies",
            type=int, default=8,
            help="Number of simulated studies (positive integer, default: 8)"
        )
        parser.add_argument(
            "-s", "--seed",
            type=int, default=42,
            help="Random seed (non-negative integer, default: 42)"
        )
        parser.add_argument(
            "-c", "--ci-level",
            type=float, default=0.95,
            help="Confidence level (0 < ci_level < 1, default: 0.95)"
        )
        parser.add_argument(
            "-o", "--output",
            type=str, default=None,
            help="Output path for the forest plot; if not specified, display on screen"
        )

        args = parser.parse_args(argv)

        # Parameter validation
        if args.n_studies < 1:
            parser.error(f"--n-studies must be >= 1, current: {args.n_studies}")
        if args.seed < 0:
            parser.error(f"--seed must be >= 0, current: {args.seed}")
        if not (0.0 < args.ci_level < 1.0):
            parser.error(f"--ci-level must be in (0,1), current: {args.ci_level}")

        # Assign to instance attributes
        self.n_studies: int = args.n_studies
        self.seed: int = args.seed
        self.ci_level: float = args.ci_level
        self.output: Optional[str] = args.output

        logger.debug(
            "CLIConfig parsed: n_studies=%d, seed=%d, ci_level=%.2f, output=%s",
            self.n_studies, self.seed, self.ci_level,
            self.output or "None (display on screen)"
        )


def main(argv=None) -> int:
    """
    Main routine: parse CLI arguments → generate simulated data → perform MetaAnalysis → render forest plot.

    :param argv: Optional argument list; defaults to sys.argv[1:].
    :return: Exit code, 0=success, 1=error, 2=user interruption.
    """
    # 1) Parse command-line configuration
    try:
        config = CLIConfig(argv)
        logger.info(
            "Startup parameters: n_studies=%d, seed=%d, ci_level=%.2f, output=%s",
            config.n_studies,
            config.seed,
            config.ci_level,
            config.output or "(display on screen)"
        )
    except SystemExit:
        # argparse has printed help or error; return directly
        return 1
    except Exception as ex:
        logger.error("Failed to parse arguments: %s", ex, exc_info=True)
        return 1

    try:
        # 2) Generate simulated data
        df = generate_simulated_data(
            n_studies=config.n_studies,
            seed=config.seed,
            ci_level=config.ci_level
        )
        logger.info("Simulated data generation complete: %d records.", len(df))
        logger.debug("Preview of simulated data:\n%s", df.head().to_string(index=False))

        # 3) Meta-analysis
        ma = MetaAnalysis(df, ci_level=config.ci_level)
        logger.info("MetaAnalysis instantiated successfully (ci_level=%.2f)", config.ci_level)

        # 4) Plotting
        fp = ForestPlotter(ma, show_fe=True, show_re=True)
        fp.plot(save_path=config.output)

        logger.info("Forest plot workflow completed successfully.")
        return 0
    except KeyboardInterrupt:
        logger.warning("Program interrupted by user (KeyboardInterrupt).")
        return 2
    except Exception as ex:
        logger.exception("Main workflow failed: %s", ex)
        return 1


if __name__ == "__main__":
    # Invoke main() and use its return value as the exit code
    sys.exit(main())
