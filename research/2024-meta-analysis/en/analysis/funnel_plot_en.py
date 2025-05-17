#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name: funnel_plot_meta_analysis.py

Module Overview:
    This module targets the Meta-Analysis workflow, delivering a comprehensive, professional, and customizable
    Funnel Plot visualization solution for researchers and enterprise data teams. It spans end-to-end support—from
    data simulation and style configuration to figure generation—ensuring that plots remain high-quality,
    stable, and reproducible in scientific reports, automated pipelines, and web services.

Key Features:
    1. Data Simulation (FunnelDataSimulator): Generates effect sizes (yi), standard errors (se), and precision (1/se)
    datasets according to user-specified study count, random seed, and publication-bias proportion, with full logging
    and exception handling.
    2. Style Management (FunnelPlotStyle + global StyleManager): Centralizes figure size, margins, resolution, scatter
    styling, grid, funnel contours, text, legend, and border parameters, decoupling them from global plotting themes.
    3. Core Plotting (FunnelPlotter):Sequentially renders the funnel region, scatter points, labels, mean line,
    major/minor grids, axes, legend, and borders; supports Y-axis inversion, minor-tick detail, and equal-margin layout;
    outputs either to file or to screen.
    4. Enterprise-Grade Logging & Styling:Integrates LoggerFactory and StyleManager to auto-detect script vs.
       package mode,output logs to console and file, rotate logs daily or by size,
       and retain history for traceability and diagnostics.
    5. Exception Management:Wraps each critical step in try/except blocks to capture, log,
       and rethrow custom exceptions, ensuring graceful exits or alerts in the main workflow.。

Typical Use Cases:
    - Visual diagnostics for publication bias in meta-analyses;
    - Automated figure generation for scientific reports and enterprise BI platforms;
    - Reproducible demonstrations in teaching examples and sensitivity-analysis pipelines;
    - Real-time plotting in Python web services or Jupyter Notebooks.

Dependencies:
    - numpy, pandas: Data generation and processing
    - matplotlib: Core plotting library
    - seaborn: Global theming support (via StyleManager)
    - utils.logger_factory: Enterprise-grade logging configuration
    - utils.plt_style_manager: Global plotting style management

Usage:
    python funnel_plot_meta_analysis.py

Author: zhinengmahua
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

# Todo # --- Initialize Logging Module --- #
# Auto-detect: if run as a script, use the filename as the logger name; otherwise use __name__
module_name = __name__ if __name__ != "__main__" else Path(__file__).stem

# Obtain Logger
#    - Root level controlled by LOG_LEVEL env var, default DEBUG
#    - Console: INFO+ with color/highlight
#    - File: DEBUG+ with daily rotation, keep 15 days
#    - Format includes time, thread, level, module:line, function, message
logger = LoggerFactory.get_logger(
    module_name,
    level=logging.DEBUG,  # overall logger level
    console_level=logging.INFO,  # console INFO+
    logfile="logs/funnel_plot_meta_analysis.log",
    file_level=logging.DEBUG,  # file DEBUG+
    max_bytes=None,  # no size limit
    backup_count_bytes=3,  # keep last 3 files
    when="midnight",  # rotate daily
    backup_count_time=7  # keep last 7 days
)

# Todo # --- Global Style Configuration --- #
# Style settings: finalize DPI, context, font fallbacks, grid, palette, color-blind support, etc.
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
        "Failed to apply global plotting style; using default Matplotlib settings: %s",
        str(e), exc_info=True
    )


@dataclass
class FunnelDataSimulator:
    """
    Funnel Plot Data Simulator (FunnelDataSimulator)

    Background:
      Funnel plots are commonly used in Meta-Analysis to detect publication bias.
      This class provides a highly customizable and reproducible data simulation
      utility suitable for both research and enterprise deployment, allowing control
      over study count, random seed, and bias proportion.

    Core Functions:
      1. Parameter validation: ensures n_studies >= 1, seed >= 0, biased_portion ∈ [0,1]
      2. Effect size generation: partitions data by biased_portion into lower/higher effects
      3. Standard error generation: samples uniformly from [0.05, 0.15)
      4. Precision calculation: 1 / se
      5. Logging of summary statistics (mean, variance, min/max) for diagnostics

    Method:
      simulate() -> pd.DataFrame:
        Returns a DataFrame with columns ["study", "yi", "se", "precision"].
    """
    n_studies: int = field(default=15, metadata={"help": "Total number of studies (>=1)"})
    seed: int = field(default=42, metadata={"help": "Random seed (>=0)"})
    biased_portion: float = field(default=0.5, metadata={"help": "Proportion of biased studies [0,1]"})

    def __post_init__(self):
        # Validate parameters
        if self.n_studies < 1:
            logger.error("Initialization failed: n_studies=%d < 1", self.n_studies)
            raise ValueError("n_studies must be >= 1")
        if self.seed < 0:
            logger.error("Initialization failed: seed=%d < 0", self.seed)
            raise ValueError("seed must be >= 0")
        if not (0.0 <= self.biased_portion <= 1.0):
            logger.error("Initialization failed: biased_portion=%.4f not in [0,1]", self.biased_portion)
            raise ValueError("biased_portion must be within [0,1]")

        logger.info(
            "FunnelDataSimulator initialized: n_studies=%d, seed=%d, biased_portion=%.2f",
            self.n_studies, self.seed, self.biased_portion
        )

    def _generate_effects(self) -> np.ndarray:
        """
        Generate effect sizes (yi): first split the total number of studies according to `biased_portion`,
        then randomly sample from two intervals centered at different means, and finally concatenate them.

        Returns:
            yi: np.ndarray, of length n_studies

        Raises:
            RuntimeError: if a numerical error occurs during generation
        """
        try:
            low_n = int(self.n_studies * self.biased_portion)
            high_n = self.n_studies - low_n
            # Lower effect sizes: centered at 0.2 with ±0.3 variation
            yi_low = 0.2 + 0.3 * (np.random.rand(low_n) - 0.5) * 2
            # Higher effect sizes: centered at 0.6 with ±0.2 variation
            yi_high = 0.6 + 0.2 * (np.random.rand(high_n) - 0.5) * 2
            yi = np.concatenate([yi_low, yi_high])
            logger.debug(
                "Effect size generation stats: low_n=%d, high_n=%d, mean=%.4f, std=%.4f",
                low_n, high_n, yi.mean(), yi.std()
            )
            return yi
        except Exception as ex:
            logger.exception("Failed to generate effect sizes: %s", ex)
            raise RuntimeError("Error generating effect sizes") from ex

    def _generate_se(self) -> np.ndarray:
        """
        Generate standard errors (se): sample uniformly from the interval [0.05, 0.15).

        Returns:
            se: np.ndarray, of length n_studies

        Raises:
            RuntimeError: if a numerical error occurs during generation
        """
        try:
            se = 0.05 + 0.1 * np.random.rand(self.n_studies)
            logger.debug(
                "Standard error generation stats: min=%.4f, max=%.4f, mean=%.4f",
                se.min(), se.max(), se.mean()
            )
            return se
        except Exception as ex:
            logger.exception("Failed to generate standard errors: %s", ex)
            raise RuntimeError("Error generating standard errors") from ex

    def simulate(self) -> pd.DataFrame:
        """
        Execute the full data simulation workflow for the funnel plot.

        1) Set the random seed;
        2) Generate effect sizes yi;
        3) Generate standard errors se;
        4) Build a DataFrame and compute precision;
        5) Validate that precision values are positive and non-zero;
        6) Return the result.

        Returns:
            pd.DataFrame: containing columns ["study", "yi", "se", "precision"]

        Raises:
            RuntimeError: if an unknown error occurs during simulation or if precision is invalid
        """
        try:
            logger.info(
                "Starting data simulation: n_studies=%d, seed=%d, biased_portion=%.2f",
                self.n_studies, self.seed, self.biased_portion
            )
            # 1) Set random seed
            np.random.seed(self.seed)

            # 2) Generate effect sizes yi
            yi = self._generate_effects()

            # 3) Generate standard errors se
            se = self._generate_se()

            # 4) Build DataFrame and compute precision
            df = pd.DataFrame({
                "study": [f"Study_{i + 1}" for i in range(self.n_studies)],
                "yi": yi,
                "se": se,
            })
            # Compute precision
            df["precision"] = 1.0 / df["se"]

            # 5) Validate precision legality (must not be zero or negative)
            # Data quality check
            if (df["precision"] <= 0).any():
                logger.error(
                    "Precision calculation error: non-positive values found, data preview:\n%s",
                    df.head().to_string(index=False)
                )
                raise RuntimeError("Non-positive values in precision; data simulation failed")

            logger.info("Data simulation completed: %d records", len(df))
            return df
        except (ValueError, RuntimeError):
            # Parameter validation and known runtime errors are thrown upwards directly
            raise
        except Exception as ex:
            # Catch all unknown exceptions
            logger.exception("Simulation failed due to unknown error: %s", ex)
            raise RuntimeError("Data simulation execution encountered an unknown error") from ex


@dataclass
class FunnelPlotStyle:
    """
    Funnel Plot Style Configuration Class (FunnelPlotStyle)

    This class centralizes all visual parameters for funnel plots and provides
    validation and a one-step apply method to ensure maintainable, reusable styling.
    It is suitable for high-quality, reproducible funnel plot rendering in
    scientific publications and enterprise BI platforms.

    Attributes:
        fig_width, fig_height             Figure size in inches (must be positive)
        dpi                               Output resolution (must be positive integer)
        margin                            Margin proportions {left, right, top, bottom} in [0, 0.5]
        marker                            Scatter style dict:
            size, edgecolor, facecolor, alpha
        label_text                        Data label style dict:
            offset, fontsize, color, bbox(dict)
        mean_line                         Mean effect line style dict:
            color, linestyle, linewidth
        funnel_region                     Funnel contour and fill style dict:
            fill_color, fill_alpha,
            line_style, line_width, z_value
        axes_label                        Axis and title text/style dict:
            xlabel, ylabel, title,
            xlabel_kwargs(dict), ylabel_kwargs(dict), title_kwargs(dict)
        grid                              Grid style dict with 'major' and 'minor'
        legend                            Legend style dict:
            loc, fontsize, frameon, handlelength, labelspacing
        border                            Border style dict:
            color, linewidth

    Methods:
        __post_init__(): Validate parameters and log initialization
        apply(fig, ax):  Apply this style to the given Figure/Axes
    """

    # Canvas and margins
    fig_width: float = 12.0
    fig_height: float = 7.0
    dpi: int = 300
    margin: Dict[str, float] = field(default_factory=lambda: {
        "left": 0.10,  # 10% left margin
        "right": 0.10,  # 10% right margin
        "bottom": 0.15,  # 15% bottom margin
        "top": 0.15  # 15% top margin
    })

    # Scatter style parameters
    marker: Dict[str, Any] = field(default_factory=lambda: {
        "s": 60.0,
        "edgecolors": "#2A6F97",
        "facecolors": "#A6CEE3",
        "alpha": 0.8,
        "linewidths": 0.8
    })

    # Data label style parameters
    label_text: Dict[str, Any] = field(default_factory=lambda: {
        "offset": 0.5,
        "fontsize": 9,
        "color": "#1F78B4",
        "bbox": {"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1}
    })

    # Mean effect line style parameters
    mean_line: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#E31A1C",
        "linestyle": "--",
        "linewidth": 2.0
    })

    # Funnel contour and fill style parameters
    funnel_region: Dict[str, Any] = field(default_factory=lambda: {
        "fill_color": "#B2DF8A",
        "fill_alpha": 0.2,
        "line_style": "--",
        "line_width": 1.0,
        "z_value": 1.96
    })

    # Axis labels and title style parameters
    axes_label: Dict[str, Any] = field(default_factory=lambda: {
        "xlabel": "Effect size (yi)",
        "ylabel": "Precision (1/SE)",
        "title": "Funnel Plot Simulation (Publication Bias Detection)",
        "xlabel_kwargs": {"fontsize": 14, "labelpad": 20},
        "ylabel_kwargs": {"fontsize": 14, "labelpad": 20},
        "title_kwargs": {"fontsize": 18, "weight": "semibold", "pad": 30}
    })

    # Grid styles (major/minor)
    grid: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "major": {"which": "major", "linestyle": "-", "linewidth": 0.8, "color": "#DDDDDD", "alpha": 0.6},
        "minor": {"which": "minor", "linestyle": ":", "linewidth": 0.5, "color": "#EEEEEE", "alpha": 0.3}
    })

    # Legend style parameters
    legend: Dict[str, Any] = field(default_factory=lambda: {
        "loc": "upper right",
        "fontsize": 12,
        "frameon": False,
        "handlelength": 2.5,
        "labelspacing": 1
    })

    # Border style parameters
    border: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#909090",
        "linewidth": 1.5
    })

    def __post_init__(self):
        """
        Parameter validation:
          1. fig_width and fig_height must be > 0;
          2. dpi must be > 0;
          3. margin['left'], margin['right'], margin['bottom'], margin['top'] must each be in [0,1];
          4. margin['left'] + margin['right'] < 1 (ensure subplot width > 0);
          5. margin['bottom'] + margin['top'] < 1 (ensure subplot height > 0);
          6. marker.alpha and funnel_region.fill_alpha must be in [0,1];
        Logs initialization success after validation.
        """
        # 1) Validate figure size and DPI
        if self.fig_width <= 0 or self.fig_height <= 0:
            logger.error(
                "Style initialization failed: fig_width=%.2f, fig_height=%.2f must be > 0",
                self.fig_width, self.fig_height
            )
            raise ValueError("fig_width and fig_height must be positive")

        if self.dpi <= 0:
            logger.error("Style initialization failed: dpi=%d must be > 0", self.dpi)
            raise ValueError("dpi must be a positive integer")

        # 2) Validate each margin is within [0,1]
        for side in ("left", "right", "bottom", "top"):
            val = self.margin[side]
            if not (0.0 <= val <= 1.0):
                logger.error("Style initialization failed: margin['%s']=%.2f not in [0,1]", side, val)
                raise ValueError(f"margin['{side}'] must be within [0,1]")

        # 3) Ensure subplot has positive width and height
        if self.margin["left"] + self.margin["right"] >= 1.0:
            logger.error(
                "Style initialization failed: margin['left'] + margin['right']=%.2f >= 1.0",
                self.margin["left"] + self.margin["right"]
            )
            raise ValueError("margin['left'] + margin['right'] must be < 1.0")

        if self.margin["bottom"] + self.margin["top"] >= 1.0:
            logger.error(
                "Style initialization failed: margin['bottom'] + margin['top']=%.2f >= 1.0",
                self.margin["bottom"] + self.margin["top"]
            )
            raise ValueError("margin['bottom'] + margin['top'] must be < 1.0")

        # 4) Validate alpha values
        alpha_marker = self.marker.get("alpha", 1.0)
        if not (0.0 <= alpha_marker <= 1.0):
            logger.error("Style initialization failed: marker.alpha=%.2f not in [0,1]", alpha_marker)
            raise ValueError("marker.alpha must be within [0,1]")

        alpha_funnel = self.funnel_region.get("fill_alpha", 1.0)
        if not (0.0 <= alpha_funnel <= 1.0):
            logger.error("Style initialization failed: funnel_region.fill_alpha=%.2f not in [0,1]", alpha_funnel)
            raise ValueError("funnel_region.fill_alpha must be within [0,1]")

        # 5) Log successful initialization
        logger.info(
            "FunnelPlotStyle initialized: fig(%.1f×%.1f@%d DPI), margin=%s, marker.size=%.1f",
            self.fig_width, self.fig_height, self.dpi, self.margin, self.marker.get("s", 0.0)
        )

    def apply(self, fig: plt.Figure, ax: plt.Axes):
        """
        Apply this style to the given Figure/Axes.

        Includes:
          1. Setting margins, DPI, and border;
          2. Hiding top/right spines, thickening bottom/left spines;
          3. Applying major/minor grids, axis labels, and title;
          4. Configuring legend parameters (other artists drawn by caller).

        Parameters:
            fig (plt.Figure): Target Figure object
            ax  (plt.Axes):   Target Axes object

        Exceptions:
            This method does not catch exceptions; callers must handle drawing errors.
        """
        # 1) Configure figure layout and border
        fig.set_size_inches(self.fig_width, self.fig_height)
        fig.set_dpi(self.dpi)
        fig.patch.set_facecolor("white")
        fig.patch.set_edgecolor(self.border["color"])
        fig.patch.set_linewidth(self.border["linewidth"])

        # 2) Convert margin proportions to subplot boundaries
        lm = self.margin["left"]
        rm = self.margin["right"]
        bm = self.margin["bottom"]
        tm = self.margin["top"]
        fig.subplots_adjust(
            left=lm,  # subplot left = left margin
            right=1.0 - rm,  # subplot right = 1 - right margin
            bottom=bm,  # subplot bottom = bottom margin
            top=1.0 - tm  # subplot top = 1 - top margin
        )

        # 3) Spine styling
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"): ax.spines[sp].set_linewidth(1.2)

        # 4) Grid styling
        ax.grid(**self.grid["major"])
        ax.minorticks_on()
        ax.grid(**self.grid["minor"])

        # 5) Axis labels and title
        ax.set_xlabel(self.axes_label["xlabel"], **self.axes_label["xlabel_kwargs"])
        ax.set_ylabel(self.axes_label["ylabel"], **self.axes_label["ylabel_kwargs"])
        ax.set_title(self.axes_label["title"], **self.axes_label["title_kwargs"])

        logger.info("Applied FunnelPlotStyle to Figure/Axes")


class FunnelPlotter:
    """
    Funnel Plotter (FunnelPlotter)

    Background:
        In Meta-Analysis, funnel plots are used to detect publication bias by visualizing
        the distribution of study effect sizes against their precision. This class
        provides an end-to-end encapsulation from raw data to high-quality funnel plots,
        suitable for reproducible scientific and enterprise-grade rendering.

    Core Functions:
      1. Initialization validation: ensures the input DataFrame contains "yi" and "precision" columns.
      2. Stepwise plotting workflow:
         - Apply global/custom style
         - Draw funnel contour region
         - Plot individual study points and labels
         - Draw the mean effect line
         - Configure major/minor grids, axes, title, and legend
      3. Supports saving to file or direct display, with clear logging and RuntimeError on failure.

    Usage Example:
        plotter = FunnelPlotter(df, style=my_style)
        plotter.plot(save_path="funnel.png")
    """

    def __init__(self, data: pd.DataFrame, style):
        """
        Initialize the FunnelPlotter.

        :param data: A DataFrame that must include columns "yi" (effect size) and "precision" (1/se).
        :param style: A FunnelPlotStyle instance that centrally manages all visual parameters.
        :raises ValueError: If the input DataFrame is missing required columns.
        """
        required = {"yi", "precision"}
        missing = required - set(data.columns)
        if missing:
            logger.error("FunnelPlotter initialization failed: missing columns %s", missing)
            raise ValueError(f"Input DataFrame must contain columns: {missing}")
        self.data = data.copy()
        self.style = style
        logger.info("FunnelPlotter initialized successfully, number of records = %d", len(self.data))

    def _draw_funnel_region(self, ax: plt.Axes):
        """
        Draw the funnel contour region representing the confidence interval: mean ± z * SE.

        :param ax: matplotlib Axes object
        """
        df = self.data
        s = self.style.funnel_region
        mean = df["yi"].mean()
        # Generate contour lines from minimum to maximum SE
        se_vals = np.linspace(df["precision"].min() ** -1, df["precision"].max() ** -1, 200)
        lower = mean - s["z_value"] * se_vals
        upper = mean + s["z_value"] * se_vals

        # Fill the funnel region
        ax.fill_betweenx(
            1 / se_vals, lower, upper,
            color=s["fill_color"],
            alpha=s["fill_alpha"],
            zorder=1,
            label="95% funnel region"
        )

        # Boundary lines
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

        logger.debug("Funnel contour region drawn (mean=%.4f)", mean)

    def _draw_points_and_labels(self, ax: plt.Axes):
        """
        Plot individual study points and their numeric labels.

        :param ax: matplotlib Axes object
        """
        df = self.data
        m = self.style.marker
        lbl = self.style.label_text

        # Plot individual study points
        ax.scatter(
            df["yi"], df["precision"],
            s=m["s"],
            edgecolors=m["edgecolors"],
            facecolors=m["facecolors"],
            alpha=m["alpha"],
            linewidths=m["linewidths"],
            zorder=3,
            label="Individual Studies"
        )
        logger.debug("Scatter points plotted, total %d points", len(df))

        # Add numeric labels
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
        logger.debug("Scatter point labels drawn")

    def _draw_mean_line(self, ax: plt.Axes):
        """
        Draw a vertical reference line at the mean effect.

        :param ax: matplotlib Axes object
        """
        mean = self.data["yi"].mean()
        ml = self.style.mean_line
        ax.axvline(
            mean,
            color=ml["color"],
            linestyle=ml["linestyle"],
            linewidth=ml["linewidth"],
            zorder=5,
            label=f"Mean effect: {mean:.2f}"
        )
        logger.debug("Mean effect line drawn (mean=%.4f)", mean)

    def plot(self, save_path: Optional[str] = None):
        """
        Render the funnel plot and either save it to file or display it.

        :param save_path: Optional path; if provided, save the figure there; otherwise display it.
        :raises RuntimeError: If any error occurs during plotting.
        """
        logger.info("Starting FunnelPlotter.plot()")
        try:
            # 1) Create Figure/Axes
            fig, ax = plt.subplots(figsize=(self.style.fig_width, self.style.fig_height), dpi=self.style.dpi)

            # 2) Apply style (margins, border, spines, grid, labels, title)
            self.style.apply(fig, ax)

            # 3) Draw funnel region
            self._draw_funnel_region(ax)

            # 4) Draw points and labels
            self._draw_points_and_labels(ax)

            # 5) Draw mean effect line
            self._draw_mean_line(ax)

            # 6) Legend
            ax.legend(**self.style.legend)
            logger.debug("Legend configured")

            # 7) Save or display
            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
                logger.info("Funnel plot saved to: %s", save_path)
            else:
                plt.show()
                logger.info("Funnel plot displayed successfully")
        except Exception as ex:
            logger.error("Funnel plot rendering failed: %s", ex, exc_info=True)
            raise RuntimeError("FunnelPlotter.plot execution failed") from ex


def parse_args(argv=None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param argv: Argument list (default: None, uses sys.argv[1:])
    :return: A Namespace with attributes:
        - n_studies (int): number of studies (>=1)
        - seed (int): random seed (>=0)
        - biased_portion (float): proportion of biased studies [0,1]
        - output (str|None): output file path; if None, display on screen
    :raises SystemExit: on --help or argument validation failure
    """
    parser = argparse.ArgumentParser(
        prog="funnel_plot_meta_analysis.py",
        description="Funnel Plot simulation and rendering tool: generate data with publication bias and draw a funnel plot"
    )
    parser.add_argument(
        "-n", "--n-studies",
        type=int, default=15,
        help="Number of simulated studies (positive integer, default: 15)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int, default=42,
        help="Random seed (non-negative integer, default: 42)"
    )
    parser.add_argument(
        "-b", "--biased-portion",
        type=float, default=0.5,
        help="Proportion of biased studies [0,1] (default: 0.5)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str, default=None,
        help="Output image path; if not specified, display on screen"
    )
    args = parser.parse_args(argv)
    # 参数校验
    if args.n_studies < 1:
        parser.error(f"--n-studies must be >= 1, current: {args.n_studies}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, current: {args.seed}")
    if not (0.0 <= args.biased_portion <= 1.0):
        parser.error(f"--biased-portion must be within [0,1], current: {args.biased_portion}")
    return args


def main(argv=None) -> int:
    """
    Main entry point: parse arguments → simulate data → draw funnel plot → save or display.

    :param argv: Argument list (default: None, uses sys.argv[1:])
    :return: Exit code
        0 on success
        1 on argument or runtime error
        2 on user interruption
    """
    try:
        # 1) Parse command-line arguments
        args = parse_args(argv)
        logger.info(
            "Startup argument parsing complete: n_studies=%d, seed=%d, biased_portion=%.2f, output=%s",
            args.n_studies, args.seed, args.biased_portion,
            args.output or "screen display"
        )

        # 2) Data simulation
        sim = FunnelDataSimulator(
            n_studies=args.n_studies,
            seed=args.seed,
            biased_portion=args.biased_portion
        )
        df = sim.simulate()

        # 3) Style and plotting
        style = FunnelPlotStyle()  # can be further extended via config file or CLI
        plotter = FunnelPlotter(df, style=style)
        plotter.plot(save_path=args.output)

        logger.info("Main workflow completed successfully, exit code=0")
        return 0
    except KeyboardInterrupt:
        logger.warning("User interruption detected (KeyboardInterrupt), exit code=2")
        return 2
    except SystemExit as se:
        # argparse triggers SystemExit when parser.error() is called
        logger.error("Argument parsing or help exit: %s", se)
        return se.code if isinstance(se.code, int) else 1
    except ValueError as ve:
        logger.error("Argument validation failed: %s", ve)
        return 1
    except RuntimeError as re:
        logger.error("Runtime error: %s", re, exc_info=True)
        return 1
    except Exception as ex:
        logger.exception("Unknown error caused program to terminate abnormally: %s", ex)
        return 1


if __name__ == "__main__":
    # Use the return value of main() as the exit code so that external callers can detect it
    sys.exit(main())
