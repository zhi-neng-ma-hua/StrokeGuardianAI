#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar-Chart Meta-analysis Visualisation Tool

Overview of functionality:
    Built on Pandas and Matplotlib, the tool simulates performance data for multiple interventions/algorithms
    across four key metrics (Accuracy, Time, Safety, Resource) and produces polar-coordinate radar charts.

Key features:
    1. Metric preprocessing: automatically invert Time and Resource (where lower is better);
    2. Full normalisation: scales all metrics to [0, 1] to remove unit effects and enhance comparability;
    3. Publication-grade styling: concentric background rings, polar grids, highlighted data points, outlines, and value labels;
    4. Flexible output: supports both interactive display and high-resolution image export;
    5. Robust and reliable: Chinese structured logging, style management, and comprehensive exception handling
       built-in—ready for enterprise deployment and research-grade reproducibility.

Usage examples:
    # Interactive display
    python radar_chart_meta_analysis.py
    # Save to file
    python radar_chart_meta_analysis.py -o output/radar.png
    # Debug mode
    python radar_chart_meta_analysis.py --verbose

Prerequisites:
  Python ≥ 3.8
  ├─ numpy
  ├─ pandas
  ├─ matplotlib
  ├─ seaborn
  ├─ utils/logger_factory
  └─ utils/plt_style_manager

Author: zhinengmahua <zhinengmahua@gmail.com>
Date: 2025-05-18
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patheffects import Stroke, Normal
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


class MultiDimensionalDataSimulator:
    """
    Multidimensional Performance Data Simulator

    Generates reproducible multidimensional performance data in bulk from a given list of interventions/algorithms
    and metric ranges—ideal for algorithm benchmarking, enterprise deployment, and research reproducibility.
    Strict input validation, structured logging, and exception handling are built-in to ensure production reliability.
    """

    def __init__(self, methods: List[str], seed: int = 42, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialise the simulator and validate input arguments

        :param methods: Non-empty list of strings—intervention or algorithm names; duplicates removed while preserving order
        :param seed: Random seed to guarantee reproducible simulations
        :param logger: Optional logging.Logger instance; defaults to the module-level logger
        :raises TypeError: Raised if methods is not a list or contains non-string elements
        :raises ValueError: Raised if the methods list is empty before or after deduplication
        """
        # Parameter validation
        if not isinstance(methods, list): raise TypeError("methods must be of list type")
        cleaned = []
        for m in methods:
            if not isinstance(m, str): raise TypeError(f"Method name must be a string, received: {m!r}")
            name = m.strip()
            if name: cleaned.append(name)
        # Deduplicate while preserving the original order
        unique_methods = list(dict.fromkeys(cleaned))
        if not unique_methods: raise ValueError("The methods list must contain at least one non-empty string")
        self.methods = unique_methods
        self.seed = int(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(
            "Initialise MultiDimensionalDataSimulator: methods=%s, seed=%d",
            self.methods, self.seed
        )

    def simulate(
            self,
            acc_range: Tuple[float, float] = (0.7, 0.95),
            time_range: Tuple[float, float] = (5.0, 15.0),
            safety_range: Tuple[float, float] = (0.8, 0.99),
            resource_range: Tuple[float, float] = (50.0, 120.0)
    ) -> pd.DataFrame:
        """
        Generate multidimensional performance simulation data

        :param acc_range: Accuracy range [min, max]—higher is better
        :param time_range: Time range [min, max]—lower is better
        :param safety_range: Safety range [min, max]—higher is better
        :param resource_range: Resource consumption range [min, max]—lower is better
        :return: pandas.DataFrame with columns ["Method", "Accuracy", "Time", "Safety", "Resource"]
        :raises RuntimeError: Raised if an exception occurs while generating or assembling data
        """
        try:
            n = len(self.methods)
            self.logger.info("Start simulating multidimensional performance data—%d methods total", n)

            # Set random seed to ensure reproducibility
            np.random.seed(self.seed)

            # Generate all metric arrays in a single step
            metrics = {
                "Accuracy": np.random.uniform(acc_range[0], acc_range[1], n),
                "Time": np.random.uniform(time_range[0], time_range[1], n),
                "Safety": np.random.uniform(safety_range[0], safety_range[1], n),
                "Resource": np.random.uniform(resource_range[0], resource_range[1], n),
            }

            # Build DataFrame and insert the Method column
            df = pd.DataFrame(metrics, index=self.methods).reset_index()
            df.rename(columns={"index": "Method"}, inplace=True)

            # Log sample data for quick verification
            preview = df.head(1).to_dict(orient="records")[0]
            self.logger.debug("Sample simulated data: %s", preview)
            self.logger.info("Multidimensional performance data simulation complete—%d records generated", n)

            return df
        except Exception as e:
            # Record the full exception stack in a structured way to facilitate troubleshooting
            self.logger.exception("Failed to simulate multidimensional performance data: %s", e)
            raise RuntimeError("MultiDimensionalDataSimulator.simulate execution failed") from e


class RadarChartPlotter:
    """
    Radar-chart Visualiser

    Features:
        - Automatically invert metrics where “lower-is-better”;
        - Normalise every metric to the range [0, 1];
        - Produce publication-quality polar radar charts with concentric rings, highlighted vertices, outlines,
          value labels, and flexible legend layout;
        - Support both interactive display and high-resolution file export.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            reverse_cols: List[str],
            logger: Optional[logging.Logger] = None,
            figsize: tuple = (12, 8),
            title: str = "Multi-intervention / algorithm Multidimensional Radar Chart",
            palette_name: str = "Set2"
    ) -> None:
        """
        Initialise the plotter and validate inputs

        :param data: Raw dataset containing a “Method” column plus one or more metric columns
        :param reverse_cols: List of column names that should be inverted (metrics where lower values are preferable)
        :param logger: Logger instance; defaults to the module-level logger
        :param figsize: Figure size in inches
        :param title: Radar-chart title
        :param palette_name: Name of the seaborn colour palette
        :raises ValueError: Raised if any reverse_cols are not found in data
        """
        self.logger = logger or logging.getLogger(__name__)
        self.data = data.copy()
        self.methods = data["Method"].tolist()
        # Automatically identify all metric columns
        self.metrics = [c for c in data.columns if c != "Method"]
        missing = set(reverse_cols) - set(self.metrics)
        if missing: raise ValueError(f"Missing columns to invert: {missing}")
        self.reverse_cols = reverse_cols
        self.figsize = figsize
        self.title = title
        self.palette_name = palette_name
        self.logger.debug(
            "Initialise RadarChartPlotter: methods=%s, metrics=%s, reverse_cols=%s, figsize=%s, title=%r, palette=%s",
            self.methods, self.metrics, self.reverse_cols, self.figsize, self.title, self.palette_name
        )

    def _normalize(self) -> pd.DataFrame:
        """
        Invert specified metrics and normalise all metrics to [0, 1]

        :return: Normalised DataFrame (Method column preserved)
        """
        df = self.data.copy()
        # Inverse mapping
        for col in self.reverse_cols:
            df[col] = df[col].max() - df[col]
            self.logger.debug("Column inverted: %s", col)
        # Normalisation
        for col in self.metrics:
            min_val, max_val = df[col].min(), df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
            self.logger.debug("Normalise %s: min = %.3f, max = %.3f", col, min_val, max_val)
        return df

    def plot(self, save_path: Optional[str] = None, legend_loc: str = "lower center") -> None:
        """
        Render the radar chart

        :param save_path: Path to save the figure; if None, the chart is shown interactively
        :param legend_loc: Legend location (see matplotlib’s legend loc argument)
        """
        try:
            df_norm = self._normalize()
            n_vars = len(self.metrics)
            # 1. Compute the angles and close the loop
            angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
            angles += angles[:1]

            # 2. Create the polar coordinate canvas
            fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
            fig.patch.set_facecolor("#f8f8f8")
            fig.patch.set_edgecolor("#cccccc")
            fig.patch.set_linewidth(1.5)
            ax.set_facecolor("#ffffff")
            ax.patch.set_alpha(0.9)
            # Primary grid
            ax.grid(which="major", linestyle="-.", linewidth=0.6, color="#bbbbbb")
            ax.grid(which="minor", linestyle=":", linewidth=0.4, color="#dddddd")
            # Secondary grid
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.125))
            ax.yaxis.grid(which="minor", color="#eeeeee", linestyle=":", linewidth=0.4)

            # 3. Concentric background rings
            for r, color in zip([1, 0.75, 0.5, 0.25], sns.light_palette("#dddddd", n_colors=4)):
                ax.fill_between(angles, r if r > 0 else 0, r - 0.25 if r > 0.25 else 0, color=color, zorder=0)

            # 4. Angles and tick labels
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.metrics, fontsize=12, fontweight="bold", color="#333")
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels([f"{v:.2f}" for v in [0.25, 0.5, 0.75, 1.0]], fontsize=10, color="#555")
            ax.set_ylim(0, 1)
            ax.spines["polar"].set_linewidth(1.5)

            # 5. Colour palette
            palette = sns.color_palette(self.palette_name, n_colors=len(df_norm))

            # 6. Plot each method
            for idx, (method, row) in enumerate(df_norm.set_index("Method").iterrows()):
                values = row[self.metrics].tolist() + [row[self.metrics[0]]]
                color = palette[idx]
                # Polyline
                ax.plot(angles, values, color=color, linewidth=3.0, alpha=0.9, label=method, zorder=3)
                # Halo outline
                line = ax.lines[-1]
                line.set_path_effects([Stroke(linewidth=6, foreground="white", alpha=0.5), Normal()])
                # Fill area
                ax.fill(angles, values, color=color, alpha=0.2, zorder=2)
                # Vertices and value labels
                for angle, val in zip(angles, values):
                    # Dual-ring highlighted vertex
                    ax.scatter(angle, val, s=80, edgecolors=color, facecolors="white", linewidth=1.5, zorder=4)
                    ax.scatter(angle, val, s=30, color=color, zorder=5)
                    # Shadowed label
                    txt = ax.text(
                        angle, val + 0.05, f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=12, color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1, alpha=0.8),
                        zorder=6
                    )
                    txt.set_path_effects([Stroke(linewidth=2, foreground="white", alpha=0.8), Normal()])

            # 7. Title and legend
            ax.set_title(self.title, pad=45, weight="bold", color="#222222")
            legend = ax.legend(
                loc=legend_loc,
                bbox_to_anchor=(0.5, -0.18),
                ncol=len(df_norm),
                frameon=True,
                facecolor="white",
                edgecolor="#666666"
            )
            legend.get_frame().set_alpha(0.7)
            for text in legend.get_texts():
                text.set_fontsize(12)

            # 8. Layout and output
            plt.tight_layout(pad=2)
            plt.subplots_adjust(bottom=0.15, top=0.85)

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info("Radar chart saved to: %s", save_path)
            else:
                plt.show()
        except Exception as e:
            self.logger.exception("Failed to draw radar chart: %s", e)
            raise RuntimeError("RadarChartPlotter.plot execution failed") from e


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Supported options:
        - -o, --output: Path to save the output image (optional). If omitted, the chart is shown in a pop-up window;
        - -s, --seed: Random seed that guarantees reproducible data simulation (default 42);
        - -r, --reverse-cols: Comma-separated metric columns to be inverted (“lower-is-better” metrics, default "Time,Resource");
        - -v, --verbose: Enable DEBUG-level logging for development debugging;
        - --version: Show the script version and exit.

    :return: argparse.Namespace with the following fields:
        - output (Path | None): image save path
        - seed (int) : random seed
        - reverse_cols (List[str]): list of column names to invert
        - verbose (bool) : whether DEBUG logging is enabled
    :raises SystemExit: prints an error and exits when arguments are invalid
    """
    parser = argparse.ArgumentParser(
        prog="radar_chart_meta_analysis",
        description="Visualisation tool for multidimensional radar charts of multiple interventions / algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--version",
        action="version",
        version="radar_chart_meta_analysis 1.0.0",
        help="Display the script version and exit"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output",
        metavar="PATH",
        type=Path,
        help="Path to save the output image; if not specified, the chart is displayed in a pop-up window"
    )

    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed to ensure reproducible data simulation"
    )

    parser.add_argument(
        "-r", "--reverse-cols",
        dest="reverse_cols",
        type=lambda s: [item.strip() for item in s.split(",") if item.strip()],
        default=["Time", "Resource"],
        help="Comma-separated metric column names that should be inverted (lower values are better)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable DEBUG-level log output for development debugging"
    )

    args = parser.parse_args()

    # The directory for the output path must exist or be creatable
    if args.output:
        out_dir = args.output.parent
        if not out_dir.exists(): parser.error(f"The output directory does not exist: {out_dir}")

    # reverse_cols must not be empty
    if not args.reverse_cols: parser.error("At least one --reverse-cols argument must be provided")

    return args


def main() -> None:
    """
    Main workflow: parse arguments → initialise logging & style → simulate data → draw radar chart → finish.

    Exit codes:
        0   Successful execution
        1   Runtime error
        2   Style-initialisation failure
    """
    args = parse_args()

    # 1. Configure logging
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("../logs"))
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if hasattr(handler, "setLevel"): handler.setLevel(console_level)
    logger.info(
        "Launching radar-chart visualisation tool | seed=%d | reverse_cols=%s | output=%s | verbose=%s",
        args.seed,
        ",".join(args.reverse_cols),
        args.output or "(not saved)",
        args.verbose
    )

    # 2. Initialise plotting style
    try:
        setup_style()
        logger.debug("Global plotting style applied successfully")
    except Exception as err:
        logger.error("Failed to initialise plotting style: %s", err)
        sys.exit(2)

    # 3. Data simulation & plotting
    try:
        # Fixed method list—replace as needed for your use case
        methods = ["Method_A", "Method_B", "Method_C", "Method_D", "Method_E"]
        simulator = MultiDimensionalDataSimulator(methods, seed=args.seed, logger=logger)
        df = simulator.simulate()
        plotter = RadarChartPlotter(
            data=df,
            reverse_cols=args.reverse_cols,
            logger=logger,
            figsize=(12, 8),
            title="Multi-intervention / Algorithm Multidimensional Radar Chart",
            palette_name="Set2"
        )
        plotter.plot(save_path=str(args.output) if args.output else None, legend_loc="lower center")
        logger.info("Radar chart generated and output completed")
        sys.exit(0)
    except Exception as err:
        logger.exception("Main workflow failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
