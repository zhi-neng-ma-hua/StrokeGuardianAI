#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Meta-analysis Visualization Tool

Module function:
  Using NetworkX and Matplotlib/Seaborn, it builds a weighted undirected graph from multi-arm (Network Meta) data and produces publication-quality network figures.

Key features:
  - Load intervention lists and comparison weights from JSON
  - Construct and summarise node and edge statistics
  - Support three layouts: circular, spring, kamada_kawai
  - Export high-resolution images or display interactive plots

Usage examples:
  python network_plot_meta_analysis.py  # built-in demo
  python network_plot_meta_analysis.py --config path/to/config.json --layout spring --save-path output/network.png  # custom setup

Runtime requirements:
  Python ≥ 3.8
  ├─ networkx
  ├─ matplotlib
  ├─ seaborn
  ├─ utils/logger_factory
  └─ utils/plt_style_manager

Author: zhinengmahua <zhinengmahua@gmail.com>
Date: 2025-05-16
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


@dataclass
class ComparisonConfig:
    """
    Input configuration for multi-arm Network Meta-analysis.

    Attributes:
        interventions (List[str]): A list of unique intervention names; each item is a non-empty string.
        comparison_dict (Dict[Tuple[str, str], int]): Mapping of comparison weights;
                                                      keys are ("Intervention A", "Intervention B") tuples,
                                                      and values are positive integers greater than 0.
    """
    interventions: List[str] = field(default_factory=list)
    comparison_dict: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Automatically validates the configuration after instantiation.

        Validation rules:
            1. The intervention list must be non-empty, with unique, non-empty strings.
            2. The comparison dictionary must be non-empty; keys must be correctly formatted and values positive integers.

        :raises ValueError: Raised when the configuration is invalid; the exception message lists all detected errors.
        """
        errors: Dict[str, str] = {}
        self._validate_interventions(errors)
        self._validate_comparisons(errors)
        if errors: raise ValueError(f"Configuration validation failed: {errors}")

    def _validate_interventions(self, errors: Dict[str, str]) -> None:
        """
        Validate self.interventions.

        - The list must not be empty.
        - Each element must be a non-empty string.
        - No duplicate items are allowed.
        """
        if not self.interventions:
            errors["interventions.empty"] = "Intervention list cannot be empty"
            return

        # Collect invalid names
        invalid_names = [x for x in self.interventions if not isinstance(x, str) or not x.strip()]
        if invalid_names: errors["interventions.format"] = f"Contains invalid names {invalid_names}"

        # Detect duplicate items
        if len(self.interventions) != len(set(self.interventions)):
            errors["interventions.dup"] = "Duplicate names detected"

    def _validate_comparisons(self, errors: Dict[str, str]) -> None:
        """
        Validate self.comparison_dict.

        - The dictionary must not be empty.
        - Keys must be two-element string tuples.
        - Values must be integers greater than 0.
        """
        if not self.comparison_dict:
            errors["comparison_dict.empty"] = "Comparison dictionary cannot be empty"
            return
        for pair, w in self.comparison_dict.items():
            # Key format check
            if not (isinstance(pair, tuple) and len(pair) == 2):
                errors[f"pair.{pair}.format"] = "Key must be a two-element tuple"
                continue

            # Element format check
            a, b = pair
            if not all(isinstance(x, str) and x.strip() for x in (a, b)):
                errors[f"pair.{pair}.value"] = "Tuple elements must be non-empty strings"

            # Value format check
            if not (isinstance(w, int) and w > 0):
                errors[f"weight.{pair}"] = f"Weight must be an integer greater than 0, current = {w}"

    @classmethod
    def from_json(cls, filepath: Path) -> "ComparisonConfig":
        """
        Load configuration from a JSON file and validate it.

        Example JSON structure:
            {
                "interventions": ["A", "B", "C"],
                "comparisons": {"A-B": 5, "B-C": 3}
            }

        :param filepath: Path to the configuration file, expected to be UTF-8 encoded and present.
        :return: A validated ComparisonConfig instance.
        :raises FileNotFoundError: Raised if the file is not found.
        :raises ValueError: Raised if reading, parsing, or content validation fails.
        """
        # Read the raw JSON content
        try:
            raw = filepath.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise Exception(f"Configuration file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to read configuration: {e}") from e

        # Parse the JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON format error: {e}") from e

        # Validate the top-level structure
        interventions = data.get("interventions")
        comparisons = data.get("comparisons")
        if not isinstance(interventions, list) or not isinstance(comparisons, dict):
            raise ValueError("JSON must contain an 'interventions' list and a 'comparisons' dictionary")

        # Build comparison_dict: convert keys and values in a single comprehension
        try:
            comp_dict = {
                tuple(part.strip() for part in key.split("-", 1)): int(val)
                for key, val in comparisons.items()
                if isinstance(key, str) and "-" in key
            }
        except Exception as e:
            raise ValueError(f"Failed to build comparison_dict: {e}") from e

        # Instantiate and trigger __post_init__ validation
        return cls(interventions=interventions, comparison_dict=comp_dict)


class ComparisonNetworkBuilder:
    """
    Build a weighted undirected network graph based on ComparisonConfig.

    Attributes:
        interventions (List[str]): List of intervention names.
        comparison_dict (Dict[Tuple[str, str], int]): Mapping of intervention comparison weights.
        logger (logging.Logger): Logger used to output workflow and error information.
    """

    def __init__(self, config: ComparisonConfig, logger: logging.Logger) -> None:
        """
        Initialise the network builder and validate the inputs.

        :param config: A validated ComparisonConfig instance.
        :param logger: A configured Logger for recording logs.
        :raises TypeError: Raised when argument types are incorrect.
        :raises ValueError: Raised when configuration content is incomplete.
        """
        if not isinstance(config, ComparisonConfig): raise TypeError("config must be a ComparisonConfig instance")
        if not isinstance(logger, logging.Logger): raise TypeError("logger must be a logging.Logger instance")

        # Double safeguard: ensure that the core configuration is not empty
        if not config.interventions: raise ValueError("At least one intervention name must be specified")
        if not config.comparison_dict: raise ValueError("At least one set of comparison weights must be provided")

        self.interventions = config.interventions
        self.comparison_dict = config.comparison_dict
        self.logger = logger

        self.logger.info(
            "Initialisation complete: %d interventions, %d comparison weights",
            len(self.interventions), len(self.comparison_dict)
        )

    def build_graph(self) -> nx.Graph:
        """
        Build and return a weighted undirected graph.

        Procedure:
            1. Add all intervention names as nodes.
            2. Traverse comparison_dict, keeping only valid comparisons and collecting statistics.
            3. Assign edge weights via the weight attribute.

        :return: networkx.Graph whose edges contain the "weight" attribute.
        :raises RuntimeError: Raised if an unexpected exception occurs during traversal.
        """
        self.logger.info("Starting to build the network graph")
        G = nx.Graph()
        G.add_nodes_from(self.interventions)
        self.logger.debug("Node addition completed: %s", self.interventions)

        valid, invalid = 0, 0
        try:
            for (u, v), w in self.comparison_dict.items():
                if u in self.interventions and v in self.interventions:
                    G.add_edge(u, v, weight=w)
                    valid += 1
                    self.logger.debug("Edge added: %s - %s, weight = %d", u, v, w)
                else:
                    invalid += 1
                    self.logger.warning("Skipped invalid edge: %s - %s", u, v)
        except Exception as e:
            self.logger.exception("Exception occurred while building the network")
            raise RuntimeError("Network construction failed") from e

        self.logger.info(
            "Build finished: %d valid edges, %d invalid edges, nodes = %d, edges = %d",
            valid, invalid, G.number_of_nodes(), G.number_of_edges()
        )
        return G


class NetworkMetaAnalysisVisualizer:
    """
    Professional-grade Network Meta-analysis visualiser.

    Generates publication-quality network diagrams from a weighted undirected graph, supports three layouts,
    and allows saving or interactive display.
    """

    def __init__(self, graph: nx.Graph, logger: logging.Logger) -> None:
        """
        Constructor: validates inputs and logs basic information.

        :param graph: networkx.Graph instance whose edges contain a "weight" attribute
        :param logger: Configured Logger for outputting logs
        :raises TypeError: Raised when parameter types are incorrect
        :raises ValueError: Raised when an edge lacks the "weight" attribute
        """
        if not isinstance(graph, nx.Graph): raise TypeError("graph must be a networkx.Graph instance")
        if not isinstance(logger, logging.Logger): raise TypeError("logger must be a logging.Logger instance")

        missing = [(u, v) for u, v, d in graph.edges(data=True) if "weight" not in d]
        if missing: raise ValueError(f"Edge(s) missing the 'weight' attribute: {missing}")

        self.graph = graph
        self.logger = logger
        self.logger.info(
            "Visualiser initialised: nodes = %d, edges = %d",
            graph.number_of_nodes(), graph.number_of_edges()
        )

    def plot(self,
             layout: str = "circular",
             base_node_size: int = 200,
             node_scale: float = 120.0,
             edge_scale: float = 1.2,
             cmap_nodes: str = "plasma",
             cmap_edges: str = "viridis",
             annotate_top: int = 3,
             save_path: Optional[str] = None) -> None:
        """
        Draw and output the network diagram.

        :param layout: Layout algorithm — choose from "circular", "spring", or "kamada_kawai"
        :param base_node_size: Base node size
        :param node_scale: Scaling factor that links node size to degree
        :param edge_scale: Scaling factor for edge width
        :param cmap_nodes: Colormap name for nodes
        :param cmap_edges: Colormap name for edges
        :param annotate_top: Label only the top-n nodes by degree
        :param save_path: Save the figure if a path is provided; otherwise, display it
        """
        # 1. Compute layout (dictionary-driven)
        layout_funcs = {
            "circular": nx.circular_layout,
            "spring": lambda g: nx.spring_layout(g, seed=42),
            "kamada_kawai": nx.kamada_kawai_layout
        }
        if layout not in layout_funcs: raise RuntimeError(f"Unknown layout type: {layout}")
        pos = layout_funcs[layout](self.graph)
        self.logger.debug("Layout computation finished: %s", layout)

        # 2. Canvas & style
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        fig.patch.set_facecolor("#f9f9f9")
        fig.patch.set_edgecolor("#909090")
        fig.patch.set_linewidth(1.5)
        ax.set_facecolor("#f9f9f9")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", linewidth=0.4, color="#cccccc")
        ax.margins(0.05)
        ax.set_aspect("equal", "box")

        # 3. Compute node degrees
        degrees = dict(self.graph.degree())
        deg_vals = list(degrees.values())
        sizes = [base_node_size + node_scale * degrees[n] for n in self.graph.nodes()]

        # 4. Draw degree labels below nodes
        degrees = dict(self.graph.degree())
        y_offset = 0.15  # Downward offset, can be fine-tuned per layout
        for n, (x, y) in pos.items():
            ax.text(
                x, y - y_offset,
                str(degrees[n]),
                fontsize=12,
                fontweight="bold",
                ha="center", va="top",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="#666666", lw=0.8, alpha=0.8),
                zorder=4
            )

        # 5. Draw nodes: map values to colormap
        cmap_n = colormaps[cmap_nodes]
        norm_n = plt.Normalize(vmin=min(deg_vals), vmax=max(deg_vals))
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=sizes,
            node_color=deg_vals,
            cmap=cmap_n,
            vmin=min(deg_vals),
            vmax=max(deg_vals),
            edgecolors="#333333",
            linewidths=1.2,
            alpha=0.9,
            ax=ax
        )

        # 6. Label only the highest-degree nodes
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:annotate_top]
        annotate_nodes = set(top_nodes) | {"D"}
        labels = {n: n for n in annotate_nodes if n in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=labels,
            font_size=12, font_weight="bold", font_color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666666", alpha=0.8),
            ax=ax
        )

        # 7. Draw edges: colour by weight
        edges_data = list(self.graph.edges(data=True))
        weights = [d["weight"] for _, _, d in edges_data]
        widths = [edge_scale * w for w in weights]
        cmap_e = colormaps[cmap_edges]
        norm_e = plt.Normalize(vmin=min(weights), vmax=max(weights))
        edge_colors = [cmap_e(norm_e(d["weight"])) for _, _, d in edges_data]

        nx.draw_networkx_edges(
            self.graph, pos,
            width=widths,
            edge_color=edge_colors,
            alpha=0.8,
            ax=ax
        )

        # 8. Add numeric labels on edges
        edge_labels = {(u, v): d["weight"] for u, v, d in edges_data}
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=12, font_color="#555555", font_weight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7),
            label_pos=0.6, ax=ax
        )

        # 9. Add colour bars: node degree and edge weight
        divider = make_axes_locatable(ax)
        # Left colour bar: node degree
        cax_left = divider.append_axes("left", size="3%", pad=0.1)
        cb_nodes = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_n, cmap=cmap_nodes),
            cax=cax_left, orientation="vertical"
        )
        cb_nodes.set_label("Node Degree", rotation=270, labelpad=15)
        cb_nodes.ax.yaxis.set_label_position("left")
        cb_nodes.ax.yaxis.tick_left()

        # Right colour bar: edge weight
        cax_right = divider.append_axes("right", size="3%", pad=0.1)
        cb_edges = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_e, cmap=cmap_edges),
            cax=cax_right, orientation="vertical"
        )
        cb_edges.set_label("Edge Weight", rotation=270, labelpad=15)
        cb_edges.ax.yaxis.set_label_position("right")
        cb_edges.ax.yaxis.tick_right()

        # 10. Title and layout finalisation
        ax.set_title("Network Meta-analysis Diagram", fontsize=18, weight="bold", pad=20)
        ax.axis("off")
        plt.tight_layout()

        # 11. Save or display
        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=300, bbox_inches="tight")
            self.logger.info("Network diagram saved: %s", out)
        else:
            plt.show()
            self.logger.info("Network diagram displayed successfully")


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Supports:
        - Specify a JSON configuration file (optional)
        - Choose the network layout algorithm
        - Specify the output path for the figure (optional)
        - Enable debug mode (DEBUG logging)
        - Display script version information

    :return: argparse.Namespace containing the following fields:
        - config (Path|None): Path to the JSON configuration file
        - layout (str): Network layout algorithm, supports ["circular", "spring", "kamada_kawai"]
        - save_path (Path|None): Path to save the figure
        - verbose (bool): Whether DEBUG logging is enabled

    :raises SystemExit: Prints an error and exits when arguments are invalid
    """
    parser = argparse.ArgumentParser(
        prog="network_plot_meta_analysis",
        description="Network Meta-analysis Visualization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Support viewing version
    parser.add_argument(
        "--version",
        action="version",
        version="network_plot_meta_analysis 1.0.0",
        help="Show script version and exit"
    )

    # Path to JSON configuration file (optional)
    parser.add_argument(
        "-c", "--config",
        metavar="JSON",
        type=Path,
        help="Path to the JSON configuration file containing the 'interventions' and 'comparisons' fields"
    )

    # Layout algorithm options
    LAYOUT_CHOICES = ["circular", "spring", "kamada_kawai"]
    parser.add_argument(
        "-l", "--layout",
        choices=LAYOUT_CHOICES,
        default="circular",
        help="Network layout algorithm"
    )

    # Output path for the figure (optional)
    parser.add_argument(
        "-o", "--save-path",
        dest="save_path",
        metavar="PATH",
        type=Path,
        help="Path to save the network figure; if not specified, the diagram is shown in a popup window"
    )

    # Debug mode switch
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for development debugging"
    )

    args = parser.parse_args()

    # Argument validation: check that the configuration file exists
    if args.config and not args.config.is_file():
        parser.error(f"Configuration file does not exist or is unreadable: {args.config}")

    return args


def main() -> None:
    """
    Main workflow entry point.

    1. Parse and validate command-line arguments;
    2. Initialise logging (supports DEBUG/INFO) and the plotting style;
    3. Load the ComparisonConfig (user-specified or built-in default);
    4. Build a weighted undirected network graph;
    5. Render and either save or display the network diagram;
    6. Catch and categorise all exceptions, then exit according to specification.

    :exitcode 1: Configuration or parameter error
    :exitcode 2: Runtime error
    :exitcode 3: Unknown error
    """
    args = parse_args()

    # Determine the console log level according to the verbose switch
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("logs"))
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if hasattr(handler, "setLevel"): handler.setLevel(console_level)

    logger.info("Starting Network Meta-analysis Visualization Tool (verbose=%s)", args.verbose)

    # Style initialisation
    try:
        setup_style()
        logger.info("Plotting style initialised successfully")
    except RuntimeError as err:
        logger.error("Failed to initialise plotting style: %s", err)
        sys.exit(2)

    # Load configuration
    try:
        if args.config:
            cfg = ComparisonConfig.from_json(args.config)
            logger.info("External configuration loaded: %s", args.config)
        else:
            default_map = {("A", "B"): 5, ("A", "C"): 2, ("B", "C"): 6, ("B", "D"): 3, ("C", "D"): 4}
            cfg = ComparisonConfig(interventions=["A", "B", "C", "D"], comparison_dict=default_map)
            logger.info("No configuration file specified; using built-in defaults")
    except ValueError as ve:
        logger.error("Failed to load configuration: %s", ve)
        sys.exit(1)

    # Build the network
    try:
        builder = ComparisonNetworkBuilder(cfg, logger)
        graph = builder.build_graph()
    except RuntimeError as re:
        logger.error("Network construction failed: %s", re)
        sys.exit(2)

    # Visualisation and output
    try:
        visualizer = NetworkMetaAnalysisVisualizer(graph, logger)
        visualizer.plot(layout=args.layout, save_path=str(args.save_path) if args.save_path else None)
    except RuntimeError as re:
        logger.error("Failed to generate network diagram: %s", re)
        sys.exit(2)
    except Exception as e:
        logger.exception("Unexpected error; program terminated: %s", e)
        sys.exit(3)


if __name__ == "__main__":
    main()
