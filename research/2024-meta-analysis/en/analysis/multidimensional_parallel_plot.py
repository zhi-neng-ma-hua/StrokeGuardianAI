#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multidimensional Parallel Coordinates Plot Generation Tool

Module function: uses Pandas and Seaborn to simulate and visualise multidimensional performance metrics
        (Accuracy, Time, Safety, Resource) across multiple methods/interventions, producing reusable,
        enterprise- and research-grade parallel-coordinates comparison plots.

Key features:
    - Randomly simulate multidimensional metric data with custom ranges and random seed for full reproducibility;
    - Render clear, intuitive multidimensional comparison plots with a parallel-coordinates system (parallel_coordinates);
    - Comprehensive logging and debug output, robust exception handling with user-friendly messages;
    - Support both high-resolution image export and interactive display modes.

Usage examples:
    # Default display mode
     python multidimensional_parallel_plot_meta_analysis.py
    # Specify output file
    python multidimensional_parallel_plot_meta_analysis.py --save-path output/parallel_plot.png
    # Enable debug mode
    python multidimensional_parallel_plot_meta_analysis.py --verbose

Runtime requirements:
  Python ≥ 3.8
  ├─ numpy
  ├─ pandas
  ├─ matplotlib
  ├─ seaborn
  ├─ utils/logger_factory
  └─ utils/plt_style_manager

Author: zhinengmahua <zhinengmahua@gmail.com>
Date:   2025-05-17
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

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


from typing import List, Tuple, Optional
import logging

import numpy as np
import pandas as pd


class MultiDimensionalDataSimulator:
    """
    Multidimensional Performance Data Simulator.

    Generates reproducible multidimensional performance data based on the given list of methods / interventions
    and metric ranges, suitable for algorithm/intervention performance evaluation and research visualisation.
    """

    def __init__(self, methods: List[str], seed: int = 42, logger: Optional[logging.Logger] = None) -> None:
        """
        Constructor: validate and initialise the simulator.

        :param methods: List of method or intervention names to simulate;
                        each element must be a non-empty string;
                        duplicates are removed automatically.
        :param seed: Random seed to ensure reproducibility.
        :param logger: Logger instance; if None, the module-level logger is used.
        :raises ValueError: Raised when the methods argument does not meet requirements.
        """
        # Validate the methods list
        if not isinstance(methods, list) or not methods: raise ValueError("methods parameter must be a non-empty list")
        cleaned = [m.strip() for m in methods if isinstance(m, str) and m.strip()]
        if not cleaned: raise ValueError("methods must contain at least one non-empty string")
        # Remove duplicates while preserving the original order
        self.methods = list(dict.fromkeys(cleaned))
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug("Initialise MultiDimensionalDataSimulator: methods=%s, seed=%d", self.methods, self.seed)

    def simulate(self,
                 acc_range: Tuple[float, float] = (0.7, 0.95),
                 time_range: Tuple[float, float] = (5.0, 15.0),
                 safety_range: Tuple[float, float] = (0.8, 0.99),
                 resource_range: Tuple[float, float] = (50.0, 120.0)) -> pd.DataFrame:
        """
        Generate multidimensional simulated data.

        :param acc_range: Accuracy metric range [min, max].
        :param time_range: Time metric range [min, max].
        :param safety_range: Safety metric range [min, max].
        :param resource_range: Resource metric range [min, max].
        :return: A DataFrame containing the columns ["Method", "Accuracy", "Time", "Safety", "Resource"].
        :raises RuntimeError: Raised if an exception occurs during simulation; the original information is preserved.
        """
        try:
            n = len(self.methods)
            self.logger.info("Begin simulating multidimensional performance data; number of methods = %d", n)
            np.random.seed(self.seed)

            # Generate all metric arrays in a single step
            metrics = {
                "Accuracy": np.random.uniform(*acc_range, size=n),
                "Time": np.random.uniform(*time_range, size=n),
                "Safety": np.random.uniform(*safety_range, size=n),
                "Resource": np.random.uniform(*resource_range, size=n),
            }

            # Build the DataFrame and set method names as the first column
            df = pd.DataFrame(metrics, index=self.methods).reset_index()
            df.rename(columns={"index": "Method"}, inplace=True)

            self.logger.debug("Sample simulated data:", df.head().to_dict(orient="records"))
            self.logger.info("Multidimensional performance data simulation completed, total %d records", n)
            return df
        except Exception as e:
            self.logger.exception("Failed to simulate multidimensional performance data: %s", e)
            raise RuntimeError("MultiDimensionalDataSimulator.simulate execution failed") from e


class ParallelCoordinatesPlotter:
    """
    Parallel-coordinates visualiser: intuitively compares multidimensional performance data in a single chart.

    Supports:
        - Automatic detection or manual selection of numeric columns to display;
        - Multiple colour palettes, either custom-defined or Seaborn presets;
        - High-quality styling: title, axes, spines, legend, etc., rendered to enterprise / research standards;
        - Ability to save to file or show interactively.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 class_column: str = "Method",
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Constructor: validates and initialises the plotter.

        :param data: DataFrame containing class_column and at least one numeric metric column.
        :param class_column: Name of the grouping column, default "Method".
        :param logger: Logger instance; defaults to the module-level logger.
        :raises ValueError: Raised if class_column is absent from data or no numeric columns are found.
        """
        self.logger = logger or logging.getLogger(__name__)
        # Validate the grouping column
        if class_column not in data.columns: raise ValueError(f"Missing grouping column: {class_column}")
        # Automatically identify numeric columns
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        if not numeric_cols: raise ValueError("No numeric metric columns found in the DataFrame")
        # Save a copy together with metadata
        self.data = data.copy()
        self.class_column = class_column
        self.numeric_cols = numeric_cols
        self.logger.debug(
            "ParallelCoordinatesPlotter initialised: class column=%s, numeric columns=%s",
            class_column, numeric_cols
        )

    def plot(self,
             cols: Optional[List[str]] = None,
             palette: Optional[List] = None,
             figsize: Tuple[float, float] = (12, 8),
             save_path: Optional[str] = None) -> None:
        """
        Draw a parallel-coordinates plot and either display or save it.

        :param cols: List of numeric columns to plot; defaults to all auto-detected numeric columns.
        :param palette: List of colours, one per group; if None, Seaborn’s tab10 is used.
        :param figsize: Figure size (width, height) in inches.
        :param save_path: Path to save the figure; if None, the plot is shown in a popup window.
        :raises RuntimeError: Raised if an error occurs while plotting or saving.
        """
        self.logger.info("Start drawing parallel-coordinates plot")
        try:
            df = self.data.copy()
            # Determine columns to plot
            cols_to_plot = cols or self.numeric_cols
            missing = [c for c in cols_to_plot if c not in df.columns]
            if missing: raise ValueError(f"Specified columns to plot do not exist: {missing}")

            # Colour selection: choose colours according to number of groups
            n_groups = df.shape[0]
            if palette is None: palette = sns.color_palette("tab10", n_colors=n_groups)
            self.logger.debug("Using colour palette: %s", palette)

            # Step 1: Create canvas & automatic layout
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("#ffffff")
            fig.patch.set_edgecolor("#cccccc")
            fig.patch.set_linewidth(1.5)

            # Step 2: Alternate background bands to enhance separation
            for idx in range(len(cols_to_plot)):
                if idx % 2 == 0:
                    ax.axvspan(idx - 0.5, idx + 0.5, color="#f0f0f0", alpha=0.3, zorder=0)

            # Step 3: Draw main polylines
            parallel_coordinates(
                df,
                class_column=self.class_column,
                cols=cols_to_plot,
                color=palette,
                linewidth=2.5,
                alpha=0.85,
                ax=ax
            )

            # Step 4: Add “glow” shadow & white highlights
            for line in ax.get_lines():
                # Draw a wide, translucent shadow first
                line.set_path_effects([
                    pe.Stroke(linewidth=6, foreground="black", alpha=0.1),
                    pe.Normal()
                ])
                # Then style main line and markers
                line.set_linewidth(2.5)
                line.set_alpha(0.9)
                line.set_marker("o")
                line.set_markersize(6)
                line.set_markeredgewidth(0.8)
                line.set_markerfacecolor("white")

            #  Step 5: Highlight first/last points & add text labels
            xticks = ax.get_xticks()
            x_first, x_last = xticks[0], xticks[-1]
            for i, (_, row) in enumerate(df.iterrows()):
                y0, y1 = row[cols_to_plot[0]], row[cols_to_plot[-1]]
                # Starting point
                ax.scatter(x_first, y0, s=50, facecolors=palette[i], edgecolors="black", linewidths=1.0, zorder=5)
                # Ending point
                ax.scatter(x_last, y1, s=50, facecolors=palette[i], edgecolors="black", linewidths=1.0, zorder=5)
                # Text label
                ax.text(
                    x_last + 0.05, y1,
                    f"{y1:.2f}",
                    fontsize=12, fontweight="bold",
                    color=palette[i],
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=palette[i], lw=0.7, alpha=0.9)
                )

            # tep 6: Dual-layer grid & tick embellishment
            ax.minorticks_on()
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)

            # Step 7: Outline title & axis labels to ensure legibility
            ax.set_title(
                "Multidimensional Parallel Coordinates Plot",
                weight="bold", color="#333333",
                path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()]
            )
            ax.set_xlabel("Metric", labelpad=10, path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
            ax.set_ylabel("Value", labelpad=10, path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

            # Step 8: Rotate X-axis labels to avoid overlap
            ax.set_xticklabels(cols_to_plot, rotation=30, ha="right")

            # Step 9: Spine styling: keep only left/bottom
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#666666")
                ax.spines[spine].set_linewidth(1.0)

            # Step 10: Place legend outside to avoid occlusion
            legend = ax.legend(
                title=self.class_column,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=min(n_groups, 5),
                frameon=True
            )

            plt.tight_layout(pad=2.0)

            # Step 11: Save or display
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info("Parallel-coordinates plot saved: %s", save_path)
            else:
                plt.show()
                self.logger.info("Parallel-coordinates plot displayed successfully")
        except Exception as e:
            self.logger.exception("Failed to draw parallel-coordinates plot: %s", e)
            raise RuntimeError("ParallelCoordinatesPlotter.plot execution failed") from e


def parse_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Supported options:
        - -o, --save-path: Path to save the output image (optional); if omitted, the plot is shown in a pop-up window.
        - -s, --seed: Random seed for reproducible data simulation (optional, default = 42).
        - -v, --verbose: Enable DEBUG-level logging for development debugging.
        - --version: Show the script version and exit.

    :return: Namespace containing save_path (Path | None), seed (int), and verbose (bool).
    :raises SystemExit: Prints an error and exits when arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        prog="multidimensional_parallel_plot_meta_analysis",
        description="Simulate multidimensional performance data and render enterprise / research-grade parallel-coordinate plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--version", action="version",
        version="multidimensional_parallel_plot_meta_analysis 1.0.0",
        help="Show script version and exit"
    )
    parser.add_argument(
        "-o", "--save-path",
        metavar="PATH",
        type=Path,
        help="Output image save path; if not specified, the diagram is displayed in a pop-up window"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed to ensure reproducible data simulation"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for development debugging"
    )
    args = parser.parse_args()
    # Validation: if a save path is specified, ensure that its directory is writable
    if args.save_path:
        parent = args.save_path.parent
        if not parent.exists(): parser.error(f"Save-path directory does not exist: {parent}")
    return args


def main() -> None:
    """
    Main workflow: argument parsing → logging & style initialisation → data simulation → parallel-coordinates plotting → finish.

    :exitcode 1: Program terminated due to argument or runtime error
    """
    args = parse_args()

    # Configure log level
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("logs"))
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        if hasattr(h, "setLevel"): h.setLevel(console_level)
    logger.info(
        "Starting parallel-coordinates plot tool: seed=%d, save_path=%s, verbose=%s",
        args.seed, args.save_path or "(not saved)", args.verbose
    )

    # Style initialisation
    try:
        setup_style()
        logger.debug("Plotting style configured successfully")
    except RuntimeError as err:
        logger.error("Failed to configure plotting style: %s", err)
        sys.exit(1)

    try:
        # Data simulation
        methods = ["Method_A", "Method_B", "Method_C", "Method_D", "Method_E"]
        simulator = MultiDimensionalDataSimulator(methods, seed=args.seed, logger=logger)
        df = simulator.simulate()
        # Visualisation
        plotter = ParallelCoordinatesPlotter(df, class_column="Method", logger=logger)
        plotter.plot(save_path=str(args.save_path) if args.save_path else None)
        logger.info("Main workflow completed successfully")
    except Exception as e:
        logger.exception("Program execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
