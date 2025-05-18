#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多维度平行坐标图生成工具

模块功能：基于 Pandas 与 Seaborn 模拟并可视化多方法/多干预的多维度性能指标（Accuracy、Time、Safety、Resource），
        生成企业级、科研级可复用的平行坐标对比图。

核心特点：
  - 随机模拟多维度指标数据，支持自定义范围与随机种子，保证可复现性；
  - 使用平行坐标系（parallel_coordinates）渲染清晰直观的多维度对比图；
  - 全面日志记录与调试输出，完善的异常捕获与友好提示；
  - 支持高分辨率图像保存与交互式展示两种模式。

使用示例：
  # 默认展示模式
  python multidimensional_parallel_plot_meta_analysis.py
  # 指定输出文件
  python multidimensional_parallel_plot_meta_analysis.py --save-path output/parallel_plot.png
  # 开启调试模式
  python multidimensional_parallel_plot_meta_analysis.py --verbose

依赖环境：
  Python ≥ 3.8
  ├─ numpy
  ├─ pandas
  ├─ matplotlib
  ├─ seaborn
  ├─ utils/logger_factory
  └─ utils/plt_style_manager

作者：智能麻花 <zhinengmahua@gmail.com>
日期：2025-05-17
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates

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


class MultiDimensionalDataSimulator:
    """
    多维度性能数据模拟器。

    根据给定的方法/干预列表及指标范围，生成可复现的多维度性能数据，适用于算法/干预方案的性能评估与科研可视化。
    """

    def __init__(self, methods: List[str], seed: int = 42, logger: Optional[logging.Logger] = None) -> None:
        """
        构造函数：校验并初始化模拟器。

        :param methods: 待模拟的方法或干预名称列表，元素为非空字符串，自动去重；
        :param seed: 随机种子，保证结果可复现；
        :param logger: 日志记录器，若无则使用模块级 logger。
        :raises ValueError: 当 methods 参数不符合要求时抛出。
        """
        # 校验 methods 列表
        if not isinstance(methods, list) or not methods: raise ValueError("methods 参数应为非空列表")
        cleaned = [m.strip() for m in methods if isinstance(m, str) and m.strip()]
        if not cleaned: raise ValueError("methods 中至少需包含一个非空字符串")
        # 去重且保持原序
        self.methods = list(dict.fromkeys(cleaned))
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug("初始化 MultiDimensionalDataSimulator：methods=%s, seed=%d", self.methods, self.seed)

    def simulate(self,
                 acc_range: Tuple[float, float] = (0.7, 0.95),
                 time_range: Tuple[float, float] = (5.0, 15.0),
                 safety_range: Tuple[float, float] = (0.8, 0.99),
                 resource_range: Tuple[float, float] = (50.0, 120.0)) -> pd.DataFrame:
        """
        生成多维度模拟数据。

        :param acc_range: Accuracy 指标范围 [min, max]；
        :param time_range: Time 指标范围 [min, max]；
        :param safety_range: Safety 指标范围 [min, max]；
        :param resource_range: Resource 指标范围 [min, max]；
        :return: 包含 ["Method","Accuracy","Time","Safety","Resource"] 列的 DataFrame；
        :raises RuntimeError: 模拟过程中出现异常时抛出，保留原始信息。
        """
        try:
            n = len(self.methods)
            self.logger.info("开始模拟多维度性能数据，方法数=%d", n)
            np.random.seed(self.seed)

            # 一次性生成所有指标数组
            metrics = {
                "Accuracy": np.random.uniform(*acc_range, size=n),
                "Time": np.random.uniform(*time_range, size=n),
                "Safety": np.random.uniform(*safety_range, size=n),
                "Resource": np.random.uniform(*resource_range, size=n),
            }

            # 构造 DataFrame，并将方法名设为首列
            df = pd.DataFrame(metrics, index=self.methods).reset_index()
            df.rename(columns={"index": "Method"}, inplace=True)

            self.logger.debug("模拟数据示例：\n%s", df.head().to_dict(orient="records"))
            self.logger.info("多维度性能数据模拟完成，共 %d 条记录", n)
            return df
        except Exception as e:
            self.logger.exception("多维度性能数据模拟失败：%s", e)
            raise RuntimeError("MultiDimensionalDataSimulator.simulate 执行失败") from e


class ParallelCoordinatesPlotter:
    """
    平行坐标系可视化器：用于将多维度性能数据在同一图表中直观对比。

    支持：
      - 自动识别或自定义要展示的数值列；
      - 多种调色方案，可自定义或使用 Seaborn 预设；
      - 高度美化：标题、坐标、脊线、图例等均按企业/科研级标准渲染；
      - 支持保存至文件或交互式展示。
    """

    def __init__(self,
                 data: pd.DataFrame,
                 class_column: str = "Method",
                 logger: Optional[logging.Logger] = None) -> None:
        """
        构造函数：校验并初始化绘图器。

        :param data: 包含 class_column 及至少一列数值型指标的 DataFrame；
        :param class_column: 分组列名，默认 "Method"；
        :param logger: 日志记录器，默认使用模块级 logger；
        :raises ValueError: 当 class_column 不在 data 中或无数值列时抛出。
        """
        self.logger = logger or logging.getLogger(__name__)
        # 校验分组列
        if class_column not in data.columns: raise ValueError(f"缺少分组列：{class_column}")
        # 自动识别数值列
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        if not numeric_cols: raise ValueError("DataFrame 中未发现数值型指标列")
        # 保存副本与元数据
        self.data = data.copy()
        self.class_column = class_column
        self.numeric_cols = numeric_cols
        self.logger.debug("ParallelCoordinatesPlotter 初始化：分组列=%s，数值列=%s", class_column, numeric_cols)

    def plot(self,
             cols: Optional[List[str]] = None,
             palette: Optional[List] = None,
             figsize: Tuple[float, float] = (12, 8),
             save_path: Optional[str] = None) -> None:
        """
        绘制平行坐标图并展示或保存。

        :param cols: 指定要绘制的数值列列表，默认使用所有自动识别的数值列；
        :param palette: 颜色列表，长度应与组数相同，None 时使用 Seaborn tab10；
        :param figsize: 图像大小（宽, 高），单位英寸；
        :param save_path: 保存路径，若为 None 则弹窗展示；
        :raises RuntimeError: 绘图或保存过程中发生错误时抛出。
        """
        self.logger.info("开始绘制平行坐标图")
        try:
            df = self.data.copy()
            # 确定绘制列
            cols_to_plot = cols or self.numeric_cols
            missing = [c for c in cols_to_plot if c not in df.columns]
            if missing: raise ValueError(f"指定绘制列不存在：{missing}")

            # 调色：按分组数量选色
            n_groups = df.shape[0]
            if palette is None: palette = sns.color_palette("tab10", n_colors=n_groups)
            self.logger.debug("使用调色板：%s", palette)

            # Step 1: 创建画布 & 自动布局
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("#ffffff")
            fig.patch.set_edgecolor("#cccccc")
            fig.patch.set_linewidth(1.5)

            # Step 2: 交替背景色带增强区隔
            for idx in range(len(cols_to_plot)):
                if idx % 2 == 0:
                    ax.axvspan(idx - 0.5, idx + 0.5, color="#f0f0f0", alpha=0.3, zorder=0)

            # Step 3: 主曲线绘制
            parallel_coordinates(
                df,
                class_column=self.class_column,
                cols=cols_to_plot,
                color=palette,
                linewidth=2.5,
                alpha=0.85,
                ax=ax
            )

            # Step 4: 添加 "发光" 阴影 & 白色
            for line in ax.get_lines():
                # 先画一层宽、透明的阴影
                line.set_path_effects([
                    pe.Stroke(linewidth=6, foreground="black", alpha=0.1),
                    pe.Normal()
                ])
                # 再调整主线和 marker
                line.set_linewidth(2.5)
                line.set_alpha(0.9)
                line.set_marker("o")
                line.set_markersize(6)
                line.set_markeredgewidth(0.8)
                line.set_markerfacecolor("white")

            #  Step 5: 高亮首末端点 & 文本标签
            xticks = ax.get_xticks()
            x_first, x_last = xticks[0], xticks[-1]
            for i, (_, row) in enumerate(df.iterrows()):
                y0, y1 = row[cols_to_plot[0]], row[cols_to_plot[-1]]
                # 首端点
                ax.scatter(x_first, y0, s=50, facecolors=palette[i], edgecolors="black", linewidths=1.0, zorder=5)
                # 末端点
                ax.scatter(x_last, y1, s=50, facecolors=palette[i], edgecolors="black", linewidths=1.0, zorder=5)
                # 文本标签
                ax.text(
                    x_last + 0.05, y1,
                    f"{y1:.2f}",
                    fontsize=12, fontweight="bold",
                    color=palette[i],
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=palette[i], lw=0.7, alpha=0.9)
                )

            # tep 6: 双层网格 & 刻度美化
            ax.minorticks_on()
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)

            # Step 7: 标题 & 轴标签描边确保可读
            ax.set_title(
                "多维度平行坐标图",
                weight="bold", color="#333333",
                path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()]
            )
            ax.set_xlabel("指标", labelpad=10, path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
            ax.set_ylabel("数值", labelpad=10, path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

            # Step 8: 旋转 X 轴标签 防止重叠
            ax.set_xticklabels(cols_to_plot, rotation=30, ha="right")

            # Step 9: 脊线美化：仅保留左/下
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#666666")
                ax.spines[spine].set_linewidth(1.0)

            # Step 10: 图例外置 保证不遮挡
            legend = ax.legend(
                title=self.class_column,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=min(n_groups, 5),
                frameon=True
            )

            plt.tight_layout(pad=2.0)

            # Step 11: 保存或展示
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info("平行坐标图已保存：%s", save_path)
            else:
                plt.show()
                self.logger.info("平行坐标图展示完成")
        except Exception as e:
            self.logger.exception("平行坐标图绘制失败：%s", e)
            raise RuntimeError("ParallelCoordinatesPlotter.plot 执行失败") from e


def parse_args() -> argparse.Namespace:
    """
    解析并返回命令行参数。

    支持以下选项：
      - -o, --save-path: 网络图像保存路径（可选），若不指定则弹窗展示；
      - -s, --seed: 随机种子，用于数据模拟的可复现性（可选，默认42）；
      - -v, --verbose: 开启 DEBUG 级别日志输出，用于开发调试；
      - --version:   显示脚本版本并退出。

    :return: 包含 save_path (Path|None)、seed (int)、verbose (bool) 的 Namespace；
    :raises SystemExit: 参数不合法时打印错误并退出。
    """
    parser = argparse.ArgumentParser(
        prog="multidimensional_parallel_plot_meta_analysis",
        description="模拟多维度性能数据并绘制企业/科研级平行坐标图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--version", action="version",
        version="multidimensional_parallel_plot_meta_analysis 1.0.0",
        help="显示脚本版本并退出"
    )
    parser.add_argument(
        "-o", "--save-path",
        metavar="PATH",
        type=Path,
        help="输出图像保存路径；若不指定则弹窗展示"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="随机种子，保证数据模拟可复现"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="开启 DEBUG 级别日志，用于开发调试"
    )
    args = parser.parse_args()
    # 校验：若指定保存路径，确保其目录可写
    if args.save_path:
        parent = args.save_path.parent
        if not parent.exists(): parser.error(f"保存路径目录不存在：{parent}")
    return args


def main() -> None:
    """
    主流程：参数解析 → 日志与样式初始化 → 数据模拟 → 平行坐标图绘制 → 结束。

    ：exitcode 1：参数或运行错误导致程序终止
    """
    args = parse_args()

    # 配置日志级别
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("logs"))
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        if hasattr(h, "setLevel"): h.setLevel(console_level)
    logger.info(
        "启动平行坐标图工具：seed=%d, save_path=%s, verbose=%s",
        args.seed, args.save_path or "（不保存）", args.verbose
    )

    # 样式初始化
    try:
        setup_style()
        logger.debug("绘图样式配置完成")
    except RuntimeError as err:
        logger.error("绘图样式配置失败：%s", err)
        sys.exit(1)

    try:
        # 数据模拟
        methods = ["Method_A", "Method_B", "Method_C", "Method_D", "Method_E"]
        simulator = MultiDimensionalDataSimulator(methods, seed=args.seed, logger=logger)
        df = simulator.simulate()
        # 可视化
        plotter = ParallelCoordinatesPlotter(df, class_column="Method", logger=logger)
        plotter.plot(save_path=str(args.save_path) if args.save_path else None)
        logger.info("主流程执行成功")
    except Exception as e:
        logger.exception("程序执行失败：%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
