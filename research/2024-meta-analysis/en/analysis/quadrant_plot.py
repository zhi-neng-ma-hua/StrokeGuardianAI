#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadrant-based Performance-Comparison Visualisation Tool

Module purpose:
    Using Pandas, Matplotlib and Seaborn, this script simulates performance
    data for multiple interventions/algorithms on four metrics—Accuracy,
    Time, Safety and Resource.  It then focuses on the two key factors,
    Accuracy (higher-is-better) and Time (lower-is-better), dynamically
    partitions the plane into four quadrants, and visualises their
    distribution to aid research and business decision-making.

Key features:
    - Reproducible simulation: user-defined random seed; automatic validation
      and de-duplication of algorithm/intervention names.
    - Flexible thresholding: automatic or manual Accuracy/Time cut-offs with
      semi-transparent quadrant backgrounds.
    - Publication-grade styling: soft grids, dashed separators, highlighted
      intersection, central quadrant labels, dual-colour scatter points.
    - Enterprise-level robustness: structured Chinese logging, comprehensive
      exception handling and global style management.
    - Multi-mode output: interactive display and high-resolution image export.

Usage examples:
  # Interactive display
  python quadrant_plot_meta_analysis.py
  # Specify output path, seed and thresholds
  python quadrant_plot_meta_analysis.py \
    --output output/quadrant.png \
    --seed 42 \
    --x-threshold 0.85 \
    --y-threshold 10.0 \
    --verbose

Runtime requirements:
  Python ≥ 3.8
    ├─ numpy
    ├─ pandas
    ├─ matplotlib
    ├─ seaborn
    ├─ utils/logger_factory
    └─ utils/plt_style_manager

Author: zhinengmahua <zhinengmahua@gmail.com>
Date  : 2025-05-19
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from utils.logger_factory import LoggerFactory
from utils.plt_style_manager import StyleConfig, StyleManager


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    """
    Initialise and return a logger.

    :param name: Logger name, typically the module name.
    :param log_dir: Directory for log files; created if absent.
    :return: Configured logging.Logger instance.
    :raises RuntimeError: Raised if the log directory cannot be created or LoggerFactory fails to initialise.
    """
    # Ensure the log directory exists
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Unable to create log directory: {log_dir}") from e

    # Use LoggerFactory to configure console and file handlers uniformly
    try:
        logger = LoggerFactory.get_logger(
            name=name,
            level=logging.DEBUG,
            console_level=logging.INFO,
            logfile=str(log_dir / f"{name}.log"),
            file_level=logging.DEBUG,
            when="midnight",
            backup_count_time=7,
            max_bytes=None
        )
    except Exception as e:
        raise RuntimeError(f"LoggerFactory initialisation failed: {e}") from e

    return logger


def setup_style(cfg: Optional[StyleConfig] = None) -> None:
    """
    Apply the global plotting style.

    :param cfg: StyleConfig instance; defaults to (grid=False, palette="colorblind", context="talk") if None.
    :raises RuntimeError: Raised if an error occurs while applying the style.
    """
    style = cfg or StyleConfig(grid=False, palette="colorblind", context="talk")
    try:
        StyleManager.apply(style)
    except Exception as e:
        # It is up to the caller to decide how to log or handle this exception
        raise RuntimeError(f"Failed to apply plotting style: {e}") from e


class DataSimulator:
    """
    Multidimensional Performance Data Generator

    Given a list of algorithms/interventions, this generator creates reproducible
    random data for four metrics—Accuracy (higher-is-better), Time (lower-is-better),
    Safety (higher-is-better) and Resource (lower-is-better)—for subsequent visualisation and analysis.

    Features:
        - Thorough input validation with de-duplication while preserving order;
        - Configurable metric ranges and random seed for research-grade reproducibility;
        - Structured Chinese logging for production monitoring and local debugging;
        - Exception wrapping to guarantee robustness of upstream workflows.
    """

    def __init__(
            self,
            methods: List[str],
            acc_range: Tuple[float, float] = (0.7, 0.95),
            time_range: Tuple[float, float] = (5.0, 15.0),
            safety_range: Tuple[float, float] = (0.8, 0.99),
            resource_range: Tuple[float, float] = (50.0, 120.0),
            seed: int = 42,
            logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialise the data generator and validate input arguments.

        :param methods: Non-empty list of algorithm/intervention names (strings); duplicates are removed automatically.
        :param acc_range: Accuracy value range [min, max] – higher is better.
        :param time_range: Time value range [min, max] – lower is better.
        :param safety_range: Safety value range [min, max] – higher is better.
        :param resource_range: Resource value range [min, max] – lower is better.
        :param seed: Random seed to guarantee reproducibility.
        :param logger: Logger instance; defaults to the module-level logger.
        :raises TypeError: If methods is not a list of strings, or any range argument is malformed.
        :raises ValueError: If methods is empty (before or after de-duping) or any range has invalid bounds.
        """
        # Logging
        self.logger = logger or logging.getLogger(__name__)
        # Validate methods
        if not isinstance(methods, list): raise TypeError("methods must be a list")
        cleaned = []
        for m in methods:
            if not isinstance(m, str): raise TypeError(f"Method name must be a string; received:{m!r}")
            name = m.strip()
            if name: cleaned.append(name)
        self.methods = list(dict.fromkeys(cleaned))
        if not self.methods: raise ValueError("The methods list must contain at least one non-empty string")
        # Validate each metric range
        for name, rng in (("Accuracy", acc_range), ("Time", time_range), ("Safety", safety_range),
                          ("Resource", resource_range)):
            if (not isinstance(rng, tuple) or len(rng) != 2
                    or not all(isinstance(v, (int, float)) for v in rng)
                    or rng[0] >= rng[1]):
                raise ValueError(f"{name} range must be (min, max) with min < max; received:{rng}")
        self.acc_range = acc_range
        self.time_range = time_range
        self.safety_range = safety_range
        self.resource_range = resource_range
        self.seed = int(seed)
        self.logger.debug(
            "Initialise DataSimulator: methods=%s, seed=%d, acc_range=%s, time_range=%s, safety_range=%s, resource_range=%s",
            self.methods, self.seed, acc_range, time_range, safety_range, resource_range
        )

    def simulate(self) -> pd.DataFrame:
        """
        Generate the multidimensional performance-metric dataset.

        :return: A DataFrame with columns ["Method", "Accuracy", "Time", "Safety", "Resource"].
        :raises RuntimeError: Raised if any exception occurs while generating or assembling the data.
        """
        try:
            self.logger.info(
                "Begin generating multidimensional performance data, number of methods = %d",
                len(self.methods)
            )
            np.random.seed(self.seed)

            # Generate all metric arrays in one batch
            metrics = {
                "Accuracy": np.random.uniform(*self.acc_range, size=len(self.methods)),
                "Time": np.random.uniform(*self.time_range, size=len(self.methods)),
                "Safety": np.random.uniform(*self.safety_range, size=len(self.methods)),
                "Resource": np.random.uniform(*self.resource_range, size=len(self.methods)),
            }
            df = pd.DataFrame(metrics, index=self.methods).rename_axis("Method").reset_index()

            # Log sample and return
            sample = df.head(1).to_dict(orient="records")[0]
            self.logger.debug("Sample generated data: %s", sample)
            self.logger.info("Multidimensional performance data generation completed — %d records", len(df))
            return df
        except Exception as ex:
            self.logger.exception("Data generation failed: %s", ex)
            raise RuntimeError("DataSimulator.simulate execution failed") from ex


class QuadrantPlotter:
    """
    Quadrant Scatter-plot Visualiser

    Designed for research and enterprise scenarios, this tool focuses on two key metrics—Accuracy (higher-is-better)
    and Time (lower-is-better). It partitions the plane into four quadrants using user-defined or
    automatically computed mean thresholds, then employs soft colour blocks,  bold reference lines,
    and highlighted labels to emphasise methods and coordinates.

    Features:
        - Supports custom thresholds or automatic mean-based partitioning;
        - Configurable semi-transparent background colours for each quadrant;
        - Clear scatter points with arrow-style annotations that highlight method names and values;
        - Comprehensive logging and exception handling to guarantee robustness for upstream calls.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialise the quadrant plotter.

        :param data: DataFrame containing “Method”, x_col, y_col and other columns;
        :param x_col: Name of the X-axis column (higher values are better, e.g. Accuracy);
        :param y_col: Name of the Y-axis column (lower values are better, e.g. Time);
        :param logger: Logger instance; defaults to the module-level logger.
        :raises ValueError: Raised when *data* lacks x_col or y_col.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col

        missing = {x_col, y_col, "Method"} - set(self.data.columns)
        if missing:
            raise ValueError(f"Required columns missing from data: {missing}")
        self.logger.debug("QuadrantPlotter initialised successfully | x=%s, y=%s", x_col, y_col)

    def plot(
            self,
            x_threshold: Optional[float] = None,
            y_threshold: Optional[float] = None,
            figsize: Tuple[int, int] = (12, 8),
            save_path: Optional[str] = None
    ) -> None:
        """
        Draw the quadrant scatter plot and either display or save it.

        :param x_threshold: Reference threshold on the X-axis; defaults to the mean of x_col when None;
        :param y_threshold: Reference threshold on the Y-axis; defaults to the mean of y_col when None;
        :param figsize: Figure size (width, height) in inches;
        :param save_path: Save to this path if provided; otherwise display in a pop-up window;
        :raises RuntimeError: Raised if drawing or saving the plot fails.
        """
        try:
            # 1. Compute thresholds
            x_ref = x_threshold if x_threshold is not None else self.data[self.x_col].mean()
            y_ref = y_threshold if y_threshold is not None else self.data[self.y_col].mean()
            self.logger.info("Reference thresholds | %s=%.3f, %s=%.3f", self.x_col, x_ref, self.y_col, y_ref)

            # 2. Prepare figure canvas
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("#f2f2f2")
            fig.patch.set_edgecolor("#cccccc")
            fig.patch.set_linewidth(1.5)
            ax.set_facecolor("#ffffff")
            ax.grid(which="major", color="#cccccc", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.minorticks_on()
            ax.grid(which="minor", color="#e6e6e6", linestyle=":", linewidth=0.5, alpha=0.4)
            sns.despine(ax=ax, top=True, right=True)

            # 3. Calculate and set axis limits with 20% padding
            x_vals = self.data[self.x_col]
            y_vals = self.data[self.y_col]
            pad_x = (x_vals.max() - x_vals.min()) * 0.2
            pad_y = (y_vals.max() - y_vals.min()) * 0.2
            ax.set_xlim(x_vals.min() - pad_x, x_vals.max() + pad_x)
            ax.set_ylim(y_vals.min() - pad_y, y_vals.max() + pad_y)

            # 4. Render four-quadrant background
            quadrant_colors = ["#d0f0c0", "#f0d0d0", "#d0e0f0", "#f0e0c0"]  # Q4, Q3, Q2, Q1
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            spans = [
                (x_ref, x_max, y_min, y_ref),  # Q4
                (x_min, x_ref, y_min, y_ref),  # Q3
                (x_min, x_ref, y_ref, y_max),  # Q2
                (x_ref, x_max, y_ref, y_max),  # Q1
            ]
            for color, (x0, x1, y0, y1) in zip(quadrant_colors, spans):
                ax.axvspan(x0, x1, y0, y1, facecolor=color, alpha=0.35, zorder=0)

            # Compute centre positions of each quadrant
            x_mid = (x_min + x_ref) / 2
            x_mid2 = (x_ref + x_max) / 2
            y_mid = (y_min + y_ref) / 2
            y_mid2 = (y_ref + y_max) / 2

            ax.text(x_mid2, y_mid2, "Best\n(Q4)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#5577aa", alpha=0.3)
            ax.text(x_mid, y_mid2, "Good\n(Q3)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#aa7755", alpha=0.3)
            ax.text(x_mid, y_mid, "Poor\n(Q2)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#5588aa", alpha=0.3)
            ax.text(x_mid2, y_mid, "Worst\n(Q1)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#aa5555", alpha=0.3)

            # 5. Draw reference lines
            ax.axvline(x_ref, color="#555555", linestyle="--", linewidth=1)
            ax.axhline(y_ref, color="#555555", linestyle="--", linewidth=1)
            ax.scatter([x_ref], [y_ref], s=50, color="#555555", marker="o", zorder=5)

            # 6. Scatter points and labels
            palette = sns.color_palette("tab10", n_colors=len(self.data))
            for idx, row in self.data.iterrows():
                xi, yi = row[self.x_col], row[self.y_col]
                color = palette[idx]
                ax.scatter(xi, yi, s=200, color=color, edgecolors="white", linewidth=1.8, zorder=3)
                ax.annotate(
                    f"{row['Method']}\n({xi:.2f}, {yi:.2f})",
                    xy=(xi, yi),
                    xytext=(12, 12),
                    textcoords="offset points",
                    ha="left", va="bottom",
                    fontsize=12, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                    zorder=4
                )

            # 7. Axis labels and title
            ax.set_xlabel(f"{self.x_col} ↑", weight="bold", labelpad=12)
            ax.set_ylabel(f"{self.y_col} ↓", weight="bold", labelpad=12)
            ax.set_title(f"{self.x_col} vs {self.y_col} - Quadrant Distribution", weight="bold", pad=24)

            # 8. Custom legend
            handles = [
                Patch(facecolor=quadrant_colors[i], label=label)
                for i, label in enumerate(["Best (Q4)", "Good (Q3)", "Poor (Q2)", "Worst (Q1)"])
            ]
            ax.legend(
                handles=handles,
                title="Quadrant",
                loc="upper right",
                fontsize=12,
                title_fontsize=14,
                frameon=True,
                edgecolor="#888",
                facecolor="white"
            )

            # 9. Output or display
            plt.tight_layout(pad=1.2)
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info("Quadrant plot saved: %s", save_path)
            else:
                plt.show()
                self.logger.info("Quadrant plot displayed successfully")

        except Exception as err:
            self.logger.exception("QuadrantPlotter drawing failed: %s", err)
            raise RuntimeError("QuadrantPlotter.plot execution failed") from err


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    This tool generates an “Accuracy vs Time” quadrant scatter-plot, supporting either interactive display or
    high-resolution file export, and offers a random seed plus custom threshold options to satisfy both research
    reproducibility and enterprise deployment needs.

    Supported options:
        - -o, --output  : Path to save the image (optional); if omitted, the chart is shown in a pop-up window
        - -s, --seed    : Random seed (int) that guarantees reproducible simulation; default 42
        - --x-threshold : Accuracy reference threshold (float); defaults to the mean of the Accuracy column
        - --y-threshold : Time reference threshold (float); defaults to the mean of the Time column
        - -v, --verbose : Enable DEBUG-level logging; default is INFO
        - --version     : Show script version and exit

    :return: argparse.Namespace containing the following attributes:
        - output (Path|None) Image save path
        - seed (int) Random seed
        - x_threshold (Optional[float]) Accuracy reference threshold
        - y_threshold (Optional[float]) Time reference threshold
        - verbose(bool) Whether DEBUG logging is enabled
    :raises SystemExit: Prints an error and exits when arguments are invalid
    """
    parser = argparse.ArgumentParser(
        prog="quadrant_plot_meta_analysis",
        description="Quadrant-based Performance-Comparison Visualisation Tool (Accuracy vs Time)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Version information
    parser.add_argument(
        "--version",
        action="version",
        version="quadrant_plot_meta_analysis 1.0.0",
        help="Show script version and exit"
    )
    # Output path
    parser.add_argument(
        "-o", "--output",
        dest="output",
        metavar="PATH",
        type=Path,
        help="Path to save the high-resolution image; if not specified, the chart is displayed in a pop-up window"
    )
    # Random seed
    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed that ensures reproducible data simulation"
    )
    # Custom thresholds
    parser.add_argument(
        "--x-threshold",
        dest="x_threshold",
        type=float,
        help="Accuracy reference threshold (higher is better); defaults to the column mean"
    )
    parser.add_argument(
        "--y-threshold",
        dest="y_threshold",
        type=float,
        help="Time reference threshold (lower is better); defaults to the column mean"
    )
    # Debug mode
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable DEBUG-level log output for development debugging"
    )

    args = parser.parse_args()

    # Validate that the output directory exists
    if args.output:
        out_dir = args.output.parent
        if not out_dir.exists(): parser.error(f"The output directory does not exist or is not writable: {out_dir}")

    return args


def main() -> None:
    """
    Main workflow: parse arguments → initialise logging → apply plotting style → simulate data → draw quadrant plot.

    Exit codes:
        0  Success
        1  Main-flow exception
        2  Plot-style initialisation failure
    """
    args = parse_args()

    # 1. Configure logging
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("logs"))
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if hasattr(handler, "setLevel"): handler.setLevel(console_level)
    logger.info(
        "Quadrant visualisation started | seed=%d | x_threshold=%s | y_threshold=%s | output=%s | verbose=%s",
        args.seed,
        args.x_threshold if args.x_threshold is not None else "(mean)",
        args.y_threshold if args.y_threshold is not None else "(mean)",
        args.output or "(on-screen display)",
        args.verbose
    )

    # 2. Style initialisation
    try:
        setup_style()
        logger.debug("Global plotting style applied successfully")
    except Exception as err:
        logger.error("Failed to initialise plotting style: %s", err)
        sys.exit(2)

    # 3. Data simulation and plotting
    try:
        methods = ["Method_A", "Method_B", "Method_C", "Method_D", "Method_E"]
        # Data simulation
        simulator = DataSimulator(methods, seed=args.seed, logger=logger)
        df = simulator.simulate()
        # Quadrant plotting
        plotter = QuadrantPlotter(df, x_col="Accuracy", y_col="Time", logger=logger)
        plotter.plot(
            x_threshold=args.x_threshold,
            y_threshold=args.y_threshold,
            save_path=str(args.output) if args.output else None
        )
        logger.info("Main workflow completed; program exited normally")
        sys.exit(0)
    except Exception as err:
        logger.exception("Main workflow failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
