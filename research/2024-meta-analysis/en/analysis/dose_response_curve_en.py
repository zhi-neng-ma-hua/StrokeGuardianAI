#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name: dose_response_curve_meta_analysis.py

Purpose: To simulate and visualize dose–response curves, integrating data simulation, polynomial fitting, confidence interval calculation, and high-quality rendering.

Key Features:
  1. DoseResponseDataSimulator: Generates noisy dose–response data using an exponential model;
  2. Fitting & Evaluation: Applies polynomial regression to fit the curve and computes statistical metrics such as R² and RMSE;
  3. Visualization: Renders scatter points, fitted curve, confidence interval, statistical annotations, and an external legend;
  4. Style Configuration: Manages canvas dimensions, color palettes, grid settings, and other visual parameters via DoseResponseStyle;
  5. Logging & Exception Handling: Integrates enterprise-level logging and comprehensive error capture to ensure debuggability and stability.

Dependencies:
  - Python 3.7+
  - numpy, pandas, matplotlib, seaborn, scikit-learn

Usage Examples:
  # Generate default 8-level dose data and display
  python dose_response_curve_meta_analysis.py
  # Customize 12 dose levels, random seed, noise scale, and save the figure
  python dose_response_curve_meta_analysis.py --n_levels 12 --seed 123 --noise_scale 0.1 --output result.png

Exit Codes:
  0: Success
  1: Parameter error or runtime exception
  2: User interruption (KeyboardInterrupt)

Author: zhinengmahua
Date: 2025-05-15
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from utils.logger_factory import LoggerFactory
from utils.plt_style_manager import StyleConfig, StyleManager

# TODO # --- Logging Module Initialization --- #
module_name = __name__ if __name__ != "__main__" else Path(__file__).stem
logger = LoggerFactory.get_logger(
    module_name,
    level=logging.DEBUG,
    console_level=logging.INFO,
    logfile="logs/dose_response_curve_meta_analysis.log",
    file_level=logging.DEBUG,
    max_bytes=None,
    backup_count_bytes=3,
    when="midnight",
    backup_count_time=7
)

# TODO # --- Global Style Configuration --- #
try:
    cfg = StyleConfig(grid=False, palette="colorblind", context="talk")
    StyleManager.apply(cfg)
except Exception as e:
    logger.error(
        "Failed to apply global plotting style; default Matplotlib settings will be used: %s",
        str(e), exc_info=True
    )


@dataclass
class DoseResponseDataSimulator:
    """
    Class: Dose-Response Data Simulator (DoseResponseDataSimulator)

    Description:
        Generates Gaussian‐noised dose–response synthetic data based on an exponential growth model
        (effect = 0.5 + 0.8*(1 - exp(-dose/3))). Suitable for scenarios such as drug dose exploration,
        sensitivity analysis of engineering parameters, and preliminary data diagnostics.

    :param dose_min: Minimum dose value; must be < dose_max.
    :param dose_max: Maximum dose value; must be > dose_min.
    :param n_levels: Number of dose levels; integer ≥ 2. Excessively large values will trigger performance warnings.
    :param seed: Random number seed for reproducibility.
    :param noise_scale: Noise standard deviation; float ≥ 0. Excessively large values will trigger warnings.

    :raises ValueError: Raised if any parameter is invalid, with detailed diagnostic messages.
    """
    dose_min: float = field(default=0.0, metadata={"help": "Minimum dose"})
    dose_max: float = field(default=10.0, metadata={"help": "Maximum dose"})
    n_levels: int = field(default=8, metadata={"help": "Number of dose levels"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    noise_scale: float = field(default=0.05, metadata={"help": "Noise standard deviation"})

    def __post_init__(self):
        # Validate dose range
        if not (self.dose_min < self.dose_max):
            msg = (
                f"ParameterError: dose_min({self.dose_min}) must be less than dose_max({self.dose_max}); "
                "please adjust the input range"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Validate number of levels
        if self.n_levels < 2:
            msg = f"ParameterError: n_levels({self.n_levels}) must be ≥ 2"
            logger.error(msg)
            raise ValueError(msg)
        if self.n_levels > 10000:
            logger.warning(
                "Warning: n_levels=%d is very large and may cause memory or performance issues", self.n_levels
            )

        # Validate noise scale
        if self.noise_scale < 0:
            msg = f"ParameterError: noise_scale({self.noise_scale}) must be ≥ 0"
            logger.error(msg)
            raise ValueError(msg)
        if self.noise_scale > (self.dose_max - self.dose_min):
            logger.warning(
                "Noise scale noise_scale=%.3f is large relative to dose range and may obscure the trend",
                self.noise_scale
            )

        logger.info("Parameter validation passed; simulator initialization complete")

    def simulate(self) -> pd.DataFrame:
        """
        Perform dose-response data simulation.

        Process:
          1. Set random seed for reproducibility;
          2. Generate equally spaced dose sequence;
          3. Compute base effect via the exponential model: 0.5 + 0.8*(1 - exp(-dose/3));
          4. Add Gaussian noise (mean=0, standard deviation=noise_scale);
          5. Construct and return a DataFrame.

        :return: DataFrame with columns 'dose' and 'effect', number of rows = n_levels
        :raises RuntimeError: Raised if an unexpected error occurs during simulation
        """
        logger.info(
            "Starting simulation: dose in [%.3f, %.3f], levels=%d, noise_scale=%.3f",
            self.dose_min, self.dose_max, self.n_levels, self.noise_scale
        )

        try:
            # 1) Set random seed
            np.random.seed(self.seed)

            # 2) Generate equally spaced dose sequence
            doses = np.linspace(self.dose_min, self.dose_max, self.n_levels)
            logger.debug("Dose sequence generated (first 5): %s", np.round(doses[:5], 4))

            # 3) Compute base effect
            base_effect = 0.5 + 0.8 * (1 - np.exp(-doses / 3.0))
            logger.debug("Base effect (noise-free) example: %s", np.round(base_effect[:5], 4))

            # 4) Add Gaussian noise
            noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=self.n_levels)
            effects = base_effect + noise
            logger.debug("Noise sample: %s", np.round(noise[:5], 4))
            logger.debug("Noisy effect sample: %s", np.round(effects[:5], 4))

            # 5) Build DataFrame
            df = pd.DataFrame({"dose": doses, "effect": effects})
            logger.info("Simulation complete; generated %d records", len(df))
            return df
        except MemoryError as me:
            logger.exception("MemoryError: n_levels=%d", self.n_levels)
            raise RuntimeError("Data size too large, out of memory") from me
        except Exception as ex:
            logger.exception("Unexpected error during data simulation: %s", ex)
            raise RuntimeError("Dose-response data simulation failed; please check logs") from ex


@dataclass
class DoseResponseStyle:
    """
    Class: Dose-Response Curve Style Configuration (DoseResponseStyle)

    Description:
        This class centrally defines and applies high-quality visualization parameters
        for dose–response plots, including canvas size, DPI, margin ratios, scatter and
        fit line styles, grid line styles, axis labels, title properties, etc., ensuring
        a professionally consistent visual standard in scientific reports or enterprise
        presentations.

    Usage:
        When invoked by plotter.apply(), this style is automatically applied to the
        matplotlib Figure/Axes objects.
    """

    fig_width: float = 12.0  # Canvas width (inches)
    fig_height: float = 8.0  # Canvas height (inches)
    dpi: int = 300  # Image resolution (DPI)

    # Subplot margins; note that left+right < 1 and top+bottom < 1
    margin: Dict[str, float] = field(default_factory=lambda: {
        "left": 0.10, "right": 0.10, "top": 0.12, "bottom": 0.12
    })

    # Scatter plot style configuration
    scatter: Dict[str, Any] = field(default_factory=lambda: {
        "s": 100,
        "cmap": "viridis",
        "edgecolor": "#ffffff",
        "linewidth": 1.2,
        "alpha": 0.9
    })

    # Fit line style
    line: Dict[str, Any] = field(default_factory=lambda: {
        "linestyle": "-",
        "linewidth": 2.2,
        "color": "#d62728",
        "alpha": 0.9
    })

    # Major/minor grid line styles
    grid_major: Dict[str, Any] = field(default_factory=lambda: {
        "which": "major",
        "linestyle": "--",
        "linewidth": 0.8,
        "color": "#999999",
        "alpha": 0.7
    })

    grid_minor: Dict[str, Any] = field(default_factory=lambda: {
        "which": "minor",
        "linestyle": ":",
        "linewidth": 0.5,
        "color": "#cccccc",
        "alpha": 0.5
    })

    # Title style parameters
    title_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "fontsize": 18,
        "weight": "bold",
        "color": "#333333",
        "pad": 20
    })

    # Axis label style parameters
    label_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "fontsize": 14,
        "color": "#333333",
        "labelpad": 15
    })

    # Border (canvas edge)
    border: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#909090",
        "linewidth": 1.5
    })

    ticks_color: str = "#666666"  # Axis spine color

    def __post_init__(self):
        """After initialization, automatically validate parameters and log the result."""
        try:
            # Validate canvas size and DPI
            if self.fig_width <= 0 or self.fig_height <= 0:
                raise ValueError(f"Invalid canvas dimensions: fig_width={self.fig_width}, fig_height={self.fig_height}")
            if self.dpi <= 0:
                raise ValueError(f"DPI must be a positive integer: dpi={self.dpi}")

            # Validate margin ratios
            total_h = self.margin["left"] + self.margin["right"]
            total_v = self.margin["top"] + self.margin["bottom"]
            for k, v in self.margin.items():
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"Margin parameter margin['{k}']={v} out of [0,1] range")
            if total_h >= 1.0 or total_v >= 1.0:
                raise ValueError(f"Invalid margin sums: horizontal={total_h:.2f}, vertical={total_v:.2f}; must be < 1")

            # Log: successful initialization
            logger.info(
                "DoseResponseStyle initialized successfully: canvas=%.1f×%.1f inches @%d DPI, margin=%s",
                self.fig_width, self.fig_height, self.dpi, self.margin
            )
        except ValueError as ve:
            logger.error("[Style Initialization Failed] %s", ve)
            raise

    def apply(self, fig: plt.Figure, ax: plt.Axes):
        """
        Apply the current style to specified chart components (Figure and Axes).

        :param fig: matplotlib Figure object
        :param ax: matplotlib Axes object
        :raises RuntimeError: if the style application fails
        """
        logger.debug("Starting to apply DoseResponseStyle")
        try:
            # Set canvas size and resolution
            fig.set_size_inches(self.fig_width, self.fig_height)
            fig.set_dpi(self.dpi)
            fig.patch.set_facecolor("#fbfbfb")
            fig.patch.set_edgecolor(self.border["color"])
            fig.patch.set_linewidth(self.border["linewidth"])
            ax.set_facecolor("#f7f7f7")

            # Hide top/right spines; style bottom/left spines
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_color(self.ticks_color)
                ax.spines[side].set_linewidth(1.2)

            # Add major/minor grid lines
            ax.grid(**self.grid_major)
            ax.minorticks_on()
            ax.grid(**self.grid_minor)

            # Set title and axis labels
            ax.set_title("Simulated Dose–Response Curve", **self.title_kwargs)
            ax.set_xlabel("Dose Level (Dose)", **self.label_kwargs)
            ax.set_ylabel("Effect Value (Effect)", **self.label_kwargs)

            # Apply subplot margins
            fig.subplots_adjust(
                left=self.margin["left"],
                right=1.0 - self.margin["right"],
                top=1.0 - self.margin["top"],
                bottom=self.margin["bottom"]
            )
            logger.debug("DoseResponseStyle applied successfully")
        except Exception as e:
            logger.exception("Style application failed: %s", e)
            raise RuntimeError(f"Style application failed: {e}") from e


@dataclass
class DoseResponseCurvePlotter:
    """
    Class: Dose–Response Curve Plotter (DoseResponseCurvePlotter)

    Description:
        Based on the provided dose–effect data and style configuration, renders a professional-grade
        dose–response scatter plot with a polynomial fit, including confidence intervals, numeric labels,
        a color bar, statistical annotations, and an external legend, suitable for scientific reports
        and enterprise-level BI dashboards.

    :param data: A DataFrame containing the 'dose' and 'effect' columns
    :param style: An instance of DoseResponseStyle for consistent visualization styling

    :raises ValueError: If the input data is missing required columns
    """
    data: pd.DataFrame
    style: DoseResponseStyle

    def __post_init__(self):
        # Validate that the input DataFrame contains the required columns
        missing = {"dose", "effect"} - set(self.data.columns)
        if missing:
            msg = f"[Initialization Failed] Missing required columns: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        # Copy the data to prevent external modifications
        self.data = self.data[["dose", "effect"]].copy()
        logger.info("DoseResponseCurvePlotter initialization complete; total records: %d", len(self.data))

    def plot(self, save_path: Optional[str] = None, fit_degree: int = 3) -> None:
        """
        Render the dose–response curve and either display it interactively or save to file
        based on the input parameters.

        Workflow:
          1. Create Figure/Axes and apply the unified style;
          2. Draw a gradient scatter plot (size and color mapping);
          3. Add a color bar;
          4. Perform polynomial fitting and fill a fixed-width confidence interval;
          5. Add numeric labels to each data point;
          6. Calculate and annotate R² and RMSE;
          7. Configure an external legend;
          8. Call tight_layout to adjust the layout;
          9. Save or display the figure.

        :param save_path: Output file path; if None, display interactively
        :param fit_degree: Polynomial fit degree, integer >= 1
        :return: None
        :raises ValueError: If fit_degree is not a positive integer or data count is insufficient
        :raises RuntimeError: If an unexpected error occurs during plotting
        """
        logger.info("Begin plotting: polynomial fit degree = %d", fit_degree)

        # Validate the polynomial fit degree
        if fit_degree < 1:
            msg = f"[Parameter Error] fit_degree={fit_degree} must be >=1"
            logger.error(msg)
            raise ValueError(msg)
        if len(self.data) <= fit_degree:
            msg = f"[Insufficient Data] record count={len(self.data)} <= fit_degree={fit_degree}"
            logger.error(msg)
            raise ValueError(msg)

        try:
            # 1) Create the figure and axes, then apply the style
            fig, ax = plt.subplots()
            self.style.apply(fig, ax)
            doses = self.data["dose"].to_numpy()
            effects = self.data["effect"].to_numpy()
            logger.debug(
                "剂量范围 [%.3f, %.3f], 效应范围 [%.3f, %.3f]",
                doses.min(), doses.max(), effects.min(), effects.max()
            )

            # 2) Gradient scatter: map sizes and colors linearly
            span = np.ptp(effects) + 1e-9
            norm = (effects - effects.min()) / span
            sizes = 60 + 120 * norm
            sc = ax.scatter(
                doses, effects,
                c=effects, s=sizes,
                cmap="viridis",
                edgecolor="white", linewidth=1.2, alpha=0.9
            )
            logger.debug("Scatter plot complete: size range [%.1f, %.1f]", sizes.min(), sizes.max())

            # 3) Add a color bar
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("Effect Value", fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            logger.debug("Colorbar added successfully")

            # 4) Polynomial fit & confidence interval
            coeffs = np.polyfit(doses, effects, fit_degree)
            poly = np.poly1d(coeffs)
            xs = np.linspace(doses.min(), doses.max(), 300)
            y_fit = poly(xs)
            ci_err = 0.1  # Fixed error width
            ax.fill_between(
                xs, y_fit - ci_err, y_fit + ci_err,
                color=self.style.line["color"], alpha=0.2,
                label="95% Confidence Interval"
            )
            ax.plot(
                xs, y_fit,
                linestyle=self.style.line["linestyle"],
                linewidth=self.style.line["linewidth"],
                color=self.style.line["color"],
                alpha=self.style.line.get("alpha", 1.0),
                label=f"Fitted Curve (deg={fit_degree})"
            )
            logger.debug("Fitted curve and confidence interval rendered")

            # 5) Data labels
            for x, y in zip(doses, effects):
                ax.text(
                    x, y + 0.04, f"{y:.2f}",
                    ha="center", va="bottom",
                    fontsize=12, color="#1f77b4",
                    bbox=dict(boxstyle="round", fc="white", ec="#1f77b4", alpha=0.7)
                )
            logger.debug("Numeric labels added")

            # 6) Statistical metrics
            r2 = r2_score(effects, poly(doses))
            rmse = math.sqrt(mean_squared_error(effects, poly(doses)))
            ax.text(
                0.95, 0.05,
                f"$R^2$={r2:.2f}, RMSE={rmse:.2f}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=12,
                bbox=dict(boxstyle="round", fc="white", alpha=0.6)
            )
            logger.debug("Statistical annotation: R²=%.3f, RMSE=%.3f", r2, rmse)

            # 7) External legend
            legend = ax.legend(
                frameon=True, facecolor="white", edgecolor="#cccccc",
                loc="upper left", bbox_to_anchor=(1.02, 1)
            )
            legend.get_frame().set_alpha(0.8)
            for lh in legend.get_lines():
                lh.set_linewidth(1.5)
            logger.debug("Legend configured")

            # 8) Layout and output
            plt.tight_layout(pad=3.0)
            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
                logger.info("Image saved to: %s", save_path)
            else:
                plt.show()
                logger.info("Plot displayed successfully")
        except np.linalg.LinAlgError as le:
            logger.exception("Polynomial fitting failed: %s", le)
            raise RuntimeError("Error during polynomial fitting") from le
        except Exception as ex:
            logger.exception("An unexpected error occurred during plotting: %s", ex)
            raise RuntimeError("Plot execution failed; please check the logs") from ex


def parse_args():
    """
    Parse command-line arguments.

    :return: argparse.Namespace instance containing dose_min, dose_max, n_levels, seed, noise_scale, and output
    """
    parser = argparse.ArgumentParser(
        prog="dose_response_curve_meta_analysis.py",
        description="Simulate and visualize dose–response curves, supporting custom dose levels, random seed, noise intensity, and output path."
    )
    parser.add_argument("--dose-min", type=float, default=0.0,
                        help="Minimum dose; must be less than --dose-max (default: 0.0)")
    parser.add_argument("--dose-max", type=float, default=10.0,
                        help="Maximum dose; must be greater than --dose-min (default: 10.0)")
    parser.add_argument("--n-levels", type=int, default=8,
                        help="Number of dose levels; integer ≥ 2 (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--noise-scale", type=float, default=0.05,
                        help="Gaussian noise standard deviation, ≥ 0 (default: 0.05)")
    parser.add_argument("--output", type=str, default=None,
                        help="If provided, save the figure to this path; otherwise display interactively")
    return parser.parse_args()


def main():
    """
    Main program entry point: parse arguments, perform data simulation, initialize style, and render the plot.

    Workflow:
      1. Parse command-line arguments;
      2. Initialize the simulator and generate data;
      3. Initialize style configuration;
      4. Instantiate the plotter and render the figure;
      5. Save or display based on --output.

    :return: None
    :raises ValueError: If argument validation fails
    :raises RuntimeError: If an error occurs during execution
    :raises KeyboardInterrupt: If interrupted by the user
    """
    args = parse_args()
    logger.info("=== Program start: Dose–response curve simulation and rendering ===")
    logger.debug("Command-line arguments: %s", args)

    try:
        # 2) Data simulation
        logger.info("1. Data simulation stage")
        sim = DoseResponseDataSimulator(
            dose_min=args.dose_min,
            dose_max=args.dose_max,
            n_levels=args.n_levels,
            seed=args.seed,
            noise_scale=args.noise_scale
        )
        df = sim.simulate()

        # 3) Style configuration
        logger.info("2. Style initialization stage")
        style = DoseResponseStyle()

        # 4) Plot rendering
        logger.info("3. Plot rendering stage")
        plotter = DoseResponseCurvePlotter(df, style)
        plotter.plot(save_path=args.output)

        logger.info("=== Program executed successfully, exit code 0 ===")
        sys.exit(0)
    except KeyboardInterrupt:
        # Capture user interruption
        logger.warning("Program interrupted by user, exit code 2")
        sys.exit(2)
    except ValueError as ve:
        # Argument or data validation error
        logger.error("Argument error: %s, exit code 1", ve)
        sys.exit(1)
    except Exception as ex:
        # Other runtime errors
        logger.exception("Runtime exception: %s, exit code 1", ex)
        sys.exit(1)


if __name__ == "__main__":
    main()
