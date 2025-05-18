#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络 Meta 分析可视化工具

模块功能：
  基于 NetworkX 与 Matplotlib/Seaborn，对多干预比较（Network Meta）数据构建带权无向图并生成高质量出版级网络图。

核心特点：
  - 从 JSON 加载干预列表与对比权重
  - 构建并统计节点、边数据
  - 支持 circular、spring、kamada_kawai 三种布局
  - 输出可保存的高分辨率图像或交互展示

使用示例：
  # 内置示例运行
  python network_plot_meta_analysis.py
  # 自定义配置及布局
  python network_plot_meta_analysis.py --config path/to/config.json --layout spring --save-path output/network.png

依赖环境：
  Python ≥ 3.8
  ├─ networkx
  ├─ matplotlib
  ├─ seaborn
  ├─ utils/logger_factory
  └─ utils/plt_style_manager

作者：智能麻花 <zhinengmahua@gmail.com>
日期：2025-05-16
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
    初始化并返回日志记录器。

    :param name: 日志器名称，通常使用模块名。
    :param log_dir: 日志文件目录，若不存在则创建。
    :return: 已配置的 logging.Logger 实例。
    :raises RuntimeError: 日志目录创建失败或 LoggerFactory 初始化失败。
    """
    # 确保日志目录存在
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"无法创建日志目录：{log_dir}") from e

    # 由 LoggerFactory 统一配置控制台与文件 Handler
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
        raise RuntimeError(f"LoggerFactory 初始化失败：{e}") from e

    return logger


def setup_style(cfg: Optional[StyleConfig] = None) -> None:
    """
    应用全局绘图样式。

    :param cfg: 样式配置实例，None 时使用默认(grid=False, palette="colorblind", context="talk")。
    :raises RuntimeError: 样式应用过程中出现错误时抛出。
    """
    style = cfg or StyleConfig(grid=False, palette="colorblind", context="talk")
    try:
        StyleManager.apply(style)
    except Exception as e:
        # 由调用者决定如何记录或处理该异常
        raise RuntimeError(f"绘图样式应用失败：{e}") from e


@dataclass
class ComparisonConfig:
    """
    多干预网络 Meta 分析输入配置。

    Attributes:
        interventions (List[str]): 不重复的干预名称列表，每项为非空字符串。
        comparison_dict (Dict[Tuple[str, str], int]]): 干预对比权重映射，键为 ("干预 A","干预 B") 元组，值为大于 0 的整数权重。
    """
    interventions: List[str] = field(default_factory=list)
    comparison_dict: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        在实例化后自动执行配置校验。

        校验逻辑：
          1. 干预列表非空、元素为非空字符串且唯一；
          2. 对比字典非空、键格式正确、值为正整数。

        :raises ValueError: 配置不合法时，异常消息中包含所有错误明细。
        """
        errors: Dict[str, str] = {}
        self._validate_interventions(errors)
        self._validate_comparisons(errors)
        if errors: raise ValueError(f"配置校验失败：{errors}")

    def _validate_interventions(self, errors: Dict[str, str]) -> None:
        """
        校验 self.interventions。

        - 列表非空；
        - 元素为非空字符串；
        - 无重复项。
        """
        if not self.interventions:
            errors["interventions.empty"] = "干预列表不能为空"
            return

        # 非法名称收集
        invalid_names = [x for x in self.interventions if not isinstance(x, str) or not x.strip()]
        if invalid_names: errors["interventions.format"] = f"含无效名称 {invalid_names}"

        # 重复项检测
        if len(self.interventions) != len(set(self.interventions)): errors["interventions.dup"] = "存在重复名称"

    def _validate_comparisons(self, errors: Dict[str, str]) -> None:
        """
        校验 self.comparison_dict。

        - 字典非空；
        - 键为二元字符串元组；
        - 值为大于0的整数。
        """
        if not self.comparison_dict:
            errors["comparison_dict.empty"] = "对比字典不能为空"
            return
        for pair, w in self.comparison_dict.items():
            # 键格式检查
            if not (isinstance(pair, tuple) and len(pair) == 2):
                errors[f"pair.{pair}.format"] = "键须为二元元组"
                continue

            # 元素格式检查
            a, b = pair
            if not all(isinstance(x, str) and x.strip() for x in (a, b)):
                errors[f"pair.{pair}.value"] = "元组元素须为非空字符串"

            # 值格式检查
            if not (isinstance(w, int) and w > 0):
                errors[f"weight.{pair}"] = f"权重须为大于0的整数，当前={w}"

    @classmethod
    def from_json(cls, filepath: Path) -> "ComparisonConfig":
        """
        从 JSON 文件加载配置并校验。

        JSON 格式示例：
            {
              "interventions": ["A", "B", "C"],
              "comparisons": {"A-B": 5, "B-C": 3}
            }

        :param filepath: 配置文件路径，需存在且为 UTF-8 编码；
        :return: 校验通过的 ComparisonConfig 实例；
        :raises FileNotFoundError: 文件未找到；
        :raises ValueError: 读取或解析失败，或内容校验不通过。
        """
        # 读取原始 JSON 内容
        try:
            raw = filepath.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise Exception(f"配置文件未找到：{filepath}")
        except Exception as e:
            raise ValueError(f"读取配置失败：{e}") from e

        # 解析 JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 格式错误：{e}") from e

        # 验证顶层结构
        interventions = data.get("interventions")
        comparisons = data.get("comparisons")
        if not isinstance(interventions, list) or not isinstance(comparisons, dict):
            raise ValueError("JSON 必须包含 'interventions' 列表 与 'comparisons' 字典")

        # 构建 comparison_dict：一行推导式完成键和值的转换
        try:
            comp_dict = {
                tuple(part.strip() for part in key.split("-", 1)): int(val)
                for key, val in comparisons.items()
                if isinstance(key, str) and "-" in key
            }
        except Exception as e:
            raise ValueError(f"comparison_dict 构建失败：{e}") from e

        # 实例化并触发 __post_init__ 校验
        return cls(interventions=interventions, comparison_dict=comp_dict)


class ComparisonNetworkBuilder:
    """
    基于 ComparisonConfig 构建带权无向网络图。

    Attributes:
        interventions (List[str]): 干预名称列表。
        comparison_dict (Dict[Tuple[str, str], int]): 干预对比权重映射。
        logger (logging.Logger): 日志记录器，用于输出流程和错误信息。
    """

    def __init__(self, config: ComparisonConfig, logger: logging.Logger) -> None:
        """
        初始化网络构建器并校验输入。

        :param config: 已校验的 ComparisonConfig 实例。
        :param logger: 已配置的 Logger，用于记录日志。
        :raises TypeError: 参数类型错误时。
        :raises ValueError: 配置内容不全时。
        """
        if not isinstance(config, ComparisonConfig): raise TypeError("config 必须为 ComparisonConfig 实例")
        if not isinstance(logger, logging.Logger): raise TypeError("logger 必须为 logging.Logger 实例")

        # 二次保险：确保核心配置非空
        if not config.interventions: raise ValueError("至少需指定一个干预名称")
        if not config.comparison_dict: raise ValueError("至少需提供一组对比权重")

        self.interventions = config.interventions
        self.comparison_dict = config.comparison_dict
        self.logger = logger

        self.logger.info("初始化完成：%d 个干预，%d 条对比权重", len(self.interventions), len(self.comparison_dict))

    def build_graph(self) -> nx.Graph:
        """
        构建并返回带权无向图。

        流程：
          1. 添加所有干预名称为节点；
          2. 遍历 comparison_dict，仅保留合法对比并累积统计；
          3. 边权直接使用 weight 属性。

        :return: networkx.Graph，边带 "weight" 属性。
        :raises RuntimeError: 遍历过程出现意外异常时。
        """
        self.logger.info("开始构建网络图")
        G = nx.Graph()
        G.add_nodes_from(self.interventions)
        self.logger.debug("节点添加完成：%s", self.interventions)

        valid, invalid = 0, 0
        try:
            for (u, v), w in self.comparison_dict.items():
                if u in self.interventions and v in self.interventions:
                    G.add_edge(u, v, weight=w)
                    valid += 1
                    self.logger.debug("边添加：%s - %s, 权重=%d", u, v, w)
                else:
                    invalid += 1
                    self.logger.warning("跳过无效边：%s - %s", u, v)
        except Exception as e:
            self.logger.exception("构建网络时发生异常")
            raise RuntimeError("网络构建失败") from e

        self.logger.info(
            "构建完成：%d 条有效边，%d 条无效边，节点数=%d，边数=%d",
            valid, invalid, G.number_of_nodes(), G.number_of_edges()
        )
        return G


class NetworkMetaAnalysisVisualizer:
    """
    专业级网络 Meta 分析可视化器。

    根据带权无向图生成学术出版级网络图，支持三种布局，并提供保存或交互式展示。
    """

    def __init__(self, graph: nx.Graph, logger: logging.Logger) -> None:
        """
        构造函数，校验输入并记录基本信息。

        :param graph: 含 "weight" 属性的 networkx.Graph 实例
        :param logger: 已配置的 Logger，用于输出日志
        :raises TypeError: 参数类型不符
        :raises ValueError: 边缺少 "weight" 属性
        """
        if not isinstance(graph, nx.Graph): raise TypeError("graph 必须为 networkx.Graph 实例")
        if not isinstance(logger, logging.Logger): raise TypeError("logger 必须为 logging.Logger 实例")

        missing = [(u, v) for u, v, d in graph.edges(data=True) if "weight" not in d]
        if missing: raise ValueError(f"边缺少权重属性：{missing}")

        self.graph = graph
        self.logger = logger
        self.logger.info("初始化可视化器：节点=%d，边=%d", graph.number_of_nodes(), graph.number_of_edges())

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
        绘制并输出网络图。

        :param layout: 布局算法，可选 "circular","spring","kamada_kawai"
        :param base_node_size: 节点基准大小
        :param node_scale: 节点大小与度的放大系数
        :param edge_scale: 边宽缩放系数
        :param cmap_nodes: 节点色图名称
        :param cmap_edges: 边色图名称
        :param annotate_top: 仅标注度最高的前 n 个节点
        :param save_path: 若提供则保存，否则展示
        """
        # 1. 布局计算（字典驱动）
        layout_funcs = {
            "circular": nx.circular_layout,
            "spring": lambda g: nx.spring_layout(g, seed=42),
            "kamada_kawai": nx.kamada_kawai_layout
        }
        if layout not in layout_funcs: raise RuntimeError(f"未知布局类型：{layout}")
        pos = layout_funcs[layout](self.graph)
        self.logger.debug("布局计算完成：%s", layout)

        # 2. 画布 & 样式
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        fig.patch.set_facecolor("#f9f9f9")
        fig.patch.set_edgecolor("#909090")
        fig.patch.set_linewidth(1.5)
        ax.set_facecolor("#f9f9f9")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", linewidth=0.4, color="#cccccc")
        ax.margins(0.05)
        ax.set_aspect("equal", "box")

        # 3. 节点度计算
        degrees = dict(self.graph.degree())
        deg_vals = list(degrees.values())
        sizes = [base_node_size + node_scale * degrees[n] for n in self.graph.nodes()]

        # 4. 在节点下方绘制度数标签
        degrees = dict(self.graph.degree())
        y_offset = 0.15  # 向下偏移量，可根据布局微调
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

        # 5. 节点绘制：数值映射到 colormap
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

        # 6. 只标注度最高的若干节点
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

        # 7. 边绘制：根据权重着色
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

        # 8. 在边上添加数值标签
        edge_labels = {(u, v): d["weight"] for u, v, d in edges_data}
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=12, font_color="#555555", font_weight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7),
            label_pos=0.6, ax=ax
        )

        # 8. 添加色条：节点度与边权
        # 复用同一个 ax，创建 divider
        divider = make_axes_locatable(ax)
        # 左侧色条：节点度
        cax_left = divider.append_axes("left", size="3%", pad=0.1)
        cb_nodes = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_n, cmap=cmap_nodes),
            cax=cax_left, orientation="vertical"
        )
        cb_nodes.set_label("节点度", rotation=270, labelpad=15)
        cb_nodes.ax.yaxis.set_label_position("left")
        cb_nodes.ax.yaxis.tick_left()

        # 右侧色条：边权重
        cax_right = divider.append_axes("right", size="3%", pad=0.1)
        cb_edges = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_e, cmap=cmap_edges),
            cax=cax_right, orientation="vertical"
        )
        cb_edges.set_label("边权重", rotation=270, labelpad=15)
        cb_edges.ax.yaxis.set_label_position("right")
        cb_edges.ax.yaxis.tick_right()

        # 9. 标题与布局收尾
        ax.set_title("网络 Meta 分析图", fontsize=18, weight="bold", pad=20)
        ax.axis("off")
        plt.tight_layout()

        # 10. 保存或展示
        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=300, bbox_inches="tight")
            self.logger.info("网络图已保存：%s", out)
        else:
            plt.show()
            self.logger.info("网络图展示完成")


def parse_args() -> argparse.Namespace:
    """
    解析并校验命令行参数。

    支持：
      - 指定 JSON 配置文件（可选）
      - 选择网络布局算法
      - 指定图像输出路径（可选）
      - 开启调试模式（DEBUG 日志）
      - 查看脚本版本信息

    :return: argparse.Namespace，包含以下字段：
      - config    (Path|None) : JSON 配置文件路径
      - layout    (str)       : 网络布局算法，支持 ["circular","spring","kamada_kawai"]
      - save_path (Path|None) : 图像保存路径
      - verbose   (bool)      : 是否开启 DEBUG 日志
    :raises SystemExit: 参数不合法时打印错误并退出
    """
    parser = argparse.ArgumentParser(
        prog="network_plot_meta_analysis",
        description="网络 Meta 分析可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 支持查看版本
    parser.add_argument(
        "--version",
        action="version",
        version="network_plot_meta_analysis 1.0.0",
        help="显示脚本版本并退出"
    )

    # JSON 配置文件路径（可选）
    parser.add_argument(
        "-c", "--config",
        metavar="JSON",
        type=Path,
        help="JSON 配置文件路径，包含 'interventions' 与 'comparisons' 字段"
    )

    # 布局算法选项
    LAYOUT_CHOICES = ["circular", "spring", "kamada_kawai"]
    parser.add_argument(
        "-l", "--layout",
        choices=LAYOUT_CHOICES,
        default="circular",
        help="网络布局算法"
    )

    # 图像输出路径（可选）
    parser.add_argument(
        "-o", "--save-path",
        dest="save_path",
        metavar="PATH",
        type=Path,
        help="网络图像保存路径；若不指定则弹窗展示"
    )

    # 调试模式开关
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="开启 DEBUG 级别日志输出，用于开发调试"
    )

    args = parser.parse_args()

    # 参数校验：配置文件存在性
    if args.config and not args.config.is_file():
        parser.error(f"配置文件不存在或不可读：{args.config}")

    return args


def main() -> None:
    """
    主流程入口。

    1. 解析并校验命令行参数；
    2. 初始化日志（支持 DEBUG/INFO）与绘图样式；
    3. 加载 ComparisonConfig（用户指定或默认内置）；
    4. 构建带权无向网络图；
    5. 渲染并保存或展示网络图；
    6. 捕获并分类处理所有异常，按规范退出。

    :exitcode 1: 配置或参数错误
    :exitcode 2: 运行时错误
    :exitcode 3: 未知错误
    """
    args = parse_args()

    # 根据 verbose 开关决定控制台日志级别
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("logs"))
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if hasattr(handler, "setLevel"): handler.setLevel(console_level)

    logger.info("启动网络 Meta 分析可视化工具（verbose=%s）", args.verbose)

    # 样式初始化
    try:
        setup_style()
        logger.info("绘图样式初始化成功")
    except RuntimeError as err:
        logger.error("绘图样式初始化失败：%s", err)
        sys.exit(2)

    # 加载配置
    try:
        if args.config:
            cfg = ComparisonConfig.from_json(args.config)
            logger.info("已加载外部配置：%s", args.config)
        else:
            # 内置默认配置
            default_map = {("A", "B"): 5, ("A", "C"): 2, ("B", "C"): 6, ("B", "D"): 3, ("C", "D"): 4}
            cfg = ComparisonConfig(interventions=["A", "B", "C", "D"], comparison_dict=default_map)
            logger.info("未指定配置文件，使用内置默认配置")
    except ValueError as ve:
        logger.error("配置加载失败：%s", ve)
        sys.exit(1)

    # 构建网络
    try:
        builder = ComparisonNetworkBuilder(cfg, logger)
        graph = builder.build_graph()
    except RuntimeError as re:
        logger.error("网络构建失败：%s", re)
        sys.exit(2)

    # 可视化及输出
    try:
        visualizer = NetworkMetaAnalysisVisualizer(graph, logger)
        visualizer.plot(layout=args.layout, save_path=str(args.save_path) if args.save_path else None)
    except RuntimeError as re:
        logger.error("网络图生成失败：%s", re)
        sys.exit(2)
    except Exception as e:
        logger.exception("意外错误，程序终止：%s", e)
        sys.exit(3)


if __name__ == "__main__":
    main()
