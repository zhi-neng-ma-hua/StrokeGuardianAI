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
    多维度性能数据生成器

    基于给定的算法/干预列表，按 Accuracy（越高越好）、Time（越低越好）、Safety（越高越好）、Resource（越低越好）
    四项指标生成可复现的随机数据，以便后续性能可视化与分析。

    特性：
      - 全面输入校验，自动去重并保持原序列；
      - 可配置指标分布区间及随机种子，支持科研级复现性；
      - 结构化中文日志，方便线上监控与本地调试；
      - 异常捕获包装，确保上层流程可靠性。
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
        初始化数据生成器并校验输入参数。

        :param methods: 非空算法/干预名称列表，元素为字符串，支持自动去重；
        :param acc_range: Accuracy 数据范围 [min, max]，值越大越好；
        :param time_range: Time 数据范围 [min, max]，值越小越好；
        :param safety_range: Safety 数据范围 [min, max]，值越大越好；
        :param resource_range: Resource 数据范围 [min, max]，值越小越好；
        :param seed: 随机数种子，保证结果可复现；
        :param logger: 日志记录器，默认使用模块级 logger。
        :raises TypeError: methods 不是字符串列表，或 range 参数格式不正确；
        :raises ValueError: methods 列表为空或所有元素去重后为空，或 range 参数上下界不合法。
        """
        # 日志
        self.logger = logger or logging.getLogger(__name__)
        # 校验 methods
        if not isinstance(methods, list): raise TypeError("methods 必须为列表")
        cleaned = []
        for m in methods:
            if not isinstance(m, str): raise TypeError(f"方法名称应为字符串，收到：{m!r}")
            name = m.strip()
            if name: cleaned.append(name)
        self.methods = list(dict.fromkeys(cleaned))
        if not self.methods: raise ValueError("methods 列表至少包含一个非空字符串")
        # 校验各指标范围
        for name, rng in (("Accuracy", acc_range), ("Time", time_range), ("Safety", safety_range),
                          ("Resource", resource_range)):
            if (not isinstance(rng, tuple) or len(rng) != 2
                    or not all(isinstance(v, (int, float)) for v in rng)
                    or rng[0] >= rng[1]):
                raise ValueError(f"{name} 范围应为 (min, max) 且 min < max，收到：{rng}")
        self.acc_range = acc_range
        self.time_range = time_range
        self.safety_range = safety_range
        self.resource_range = resource_range
        self.seed = int(seed)
        self.logger.debug(
            "初始化 DataSimulator: methods=%s, seed=%d, acc_range=%s, time_range=%s, safety_range=%s, resource_range=%s",
            self.methods, self.seed, acc_range, time_range, safety_range, resource_range
        )

    def simulate(self) -> pd.DataFrame:
        """
        生成多维度性能指标数据。

        :return: 包含 ["Method","Accuracy","Time","Safety","Resource"] 列的 DataFrame；
        :raises RuntimeError: 数据生成或组装过程中出现异常。
        """
        try:
            self.logger.info("开始生成多维度性能数据，方法数=%d", len(self.methods))
            np.random.seed(self.seed)

            # 一次性批量生成各项指标
            metrics = {
                "Accuracy": np.random.uniform(*self.acc_range, size=len(self.methods)),
                "Time": np.random.uniform(*self.time_range, size=len(self.methods)),
                "Safety": np.random.uniform(*self.safety_range, size=len(self.methods)),
                "Resource": np.random.uniform(*self.resource_range, size=len(self.methods)),
            }
            df = pd.DataFrame(metrics, index=self.methods).rename_axis("Method").reset_index()

            # 日志示例并返回
            sample = df.head(1).to_dict(orient="records")[0]
            self.logger.debug("生成数据示例：%s", sample)
            self.logger.info("多维度性能数据生成完成，共 %d 条记录", len(df))
            return df

        except Exception as ex:
            self.logger.exception("数据生成失败：%s", ex)
            raise RuntimeError("DataSimulator.simulate 执行失败") from ex


class QuadrantPlotter:
    """
    四象限散点图可视化器

    面向科研与企业级应用，聚焦 Accuracy（越高越好）与 Time（越低越好）两项指标，
    按指定或自动均值阈值划分四象限，并用柔和配色、加粗参考线、高亮标注方法及坐标。

    特性：
      - 支持自定义阈值或使用均值自动划分；
      - 四象限背景色块可配置、半透明渲染；
      - 清晰散点与箭头式标签，突出方法名称与数值；
      - 完善日志记录与异常捕获，确保上层调用稳定性。
    """

    def __init__(
            self,
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            logger: Optional[logging.Logger] = None
    ) -> None:
        """
        初始化四象限绘图器。

        :param data: 包含 "Method"、x_col、y_col 等列的 DataFrame；
        :param x_col: X 轴列名（值越大越优，例如 Accuracy）；
        :param y_col: Y 轴列名（值越小越优，例如 Time）；
        :param logger: 日志记录器，默认使用模块级 logger。
        :raises ValueError: 当 data 缺少 x_col 或 y_col 时抛出。
        """
        self.logger = logger or logging.getLogger(__name__)
        self.data = data.copy()
        self.x_col = x_col
        self.y_col = y_col

        missing = {x_col, y_col, "Method"} - set(self.data.columns)
        if missing:
            raise ValueError(f"数据缺少必要列：{missing}")
        self.logger.debug("QuadrantPlotter 初始化成功 | x=%s, y=%s", x_col, y_col)

    def plot(
            self,
            x_threshold: Optional[float] = None,
            y_threshold: Optional[float] = None,
            figsize: Tuple[int, int] = (12, 8),
            save_path: Optional[str] = None
    ) -> None:
        """
        绘制四象限散点图并展示或保存。

        :param x_threshold: X 轴参考阈值，None 则使用 x_col 均值；
        :param y_threshold: Y 轴参考阈值，None 则使用 y_col 均值；
        :param figsize: 图像尺寸（宽, 高，英寸）；
        :param save_path: 若指定则保存至该路径，否则弹窗展示；
        :raises RuntimeError: 绘制或保存失败时抛出。
        """
        try:
            # 1. 计算阈值
            x_ref = x_threshold if x_threshold is not None else self.data[self.x_col].mean()
            y_ref = y_threshold if y_threshold is not None else self.data[self.y_col].mean()
            self.logger.info("参考阈值 | %s=%.3f, %s=%.3f", self.x_col, x_ref, self.y_col, y_ref)

            # 2. 准备画布
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("#f2f2f2")
            fig.patch.set_edgecolor("#cccccc")
            fig.patch.set_linewidth(1.5)
            ax.set_facecolor("#ffffff")
            ax.grid(which="major", color="#cccccc", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.minorticks_on()
            ax.grid(which="minor", color="#e6e6e6", linestyle=":", linewidth=0.5, alpha=0.4)
            sns.despine(ax=ax, top=True, right=True)

            # 3. 计算并设置轴范围，留白20%
            x_vals = self.data[self.x_col]
            y_vals = self.data[self.y_col]
            pad_x = (x_vals.max() - x_vals.min()) * 0.2
            pad_y = (y_vals.max() - y_vals.min()) * 0.2
            ax.set_xlim(x_vals.min() - pad_x, x_vals.max() + pad_x)
            ax.set_ylim(y_vals.min() - pad_y, y_vals.max() + pad_y)

            # 4. 渲染四象限背景
            quadrant_colors = ["#d0f0c0", "#f0d0d0", "#d0e0f0", "#f0e0c0"]  # Q4,Q3,Q2,Q1
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

            # 计算象限中心位置
            x_mid = (x_min + x_ref) / 2
            x_mid2 = (x_ref + x_max) / 2
            y_mid = (y_min + y_ref) / 2
            y_mid2 = (y_ref + y_max) / 2

            ax.text(x_mid2, y_mid2, '最优\n(Q4)', ha='center', va='center', fontsize=14, fontweight='bold',
                    color='#5577aa', alpha=0.3)
            ax.text(x_mid, y_mid2, '次优\n(Q3)', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#aa7755', alpha=0.3)
            ax.text(x_mid, y_mid, '次差\n(Q2)', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#5588aa', alpha=0.3)
            ax.text(x_mid2, y_mid, '最差\n(Q1)', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#aa5555', alpha=0.3)

            # 5. 绘制参考线
            ax.axvline(x_ref, color="#555555", linestyle="--", linewidth=1)
            ax.axhline(y_ref, color="#555555", linestyle="--", linewidth=1)
            ax.scatter([x_ref], [y_ref], s=50, color="#555555", marker="o", zorder=5)

            # 6. 散点与标签
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

            # 7. 轴标签与标题
            ax.set_xlabel(f"{self.x_col} ↑", weight="bold", labelpad=12)
            ax.set_ylabel(f"{self.y_col} ↓", weight="bold", labelpad=12)
            ax.set_title(f"{self.x_col} vs {self.y_col} — 四象限分布", weight="bold", pad=24)

            # 8. 自定义图例
            handles = [
                Patch(facecolor=quadrant_colors[i], label=label)
                for i, label in enumerate(["最优(Q4)", "次优(Q3)", "次差(Q2)", "最差(Q1)"])
            ]
            ax.legend(
                handles=handles,
                title="象限",
                loc="upper right",
                fontsize=12,
                title_fontsize=14,
                frameon=True,
                edgecolor="#888",
                facecolor="white"
            )

            # 9. 输出或展示
            plt.tight_layout(pad=1.2)
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info("四象限图已保存：%s", save_path)
            else:
                plt.show()
                self.logger.info("四象限图展示完成")

        except Exception as err:
            self.logger.exception("QuadrantPlotter 绘制失败：%s", err)
            raise RuntimeError("QuadrantPlotter.plot 执行失败") from err


def parse_args() -> argparse.Namespace:
    """
    解析并校验命令行参数。

    本工具用于生成 "Accuracy vs. Time" 四象限散点图，支持交互式展示或高分辨率文件输出，
    并提供随机种子和自定义参考阈值配置，以满足科研复现与企业级上线需求。

    支持以下参数：
      - -o, --output  : 图像保存路径（可选），若不指定则弹窗展示
      - -s, --seed    : 随机数种子，保证数据模拟可复现，类型 int，默认 42
      - --x-threshold : Accuracy 参考阈值（float），默认使用 Accuracy 列均值
      - --y-threshold : Time 参考阈值（float），默认使用 Time 列均值
      - -v, --verbose : 启用 DEBUG 级别日志输出，默认 INFO
      - --version     : 显示脚本版本并退出

    :return: argparse.Namespace，包含以下属性：
      - output (Path|None) 图像保存路径
      - seed (int) 随机数种子
      - x_threshold (Optional[float])Accuracy 参考阈值
      - y_threshold (Optional[float])Time 参考阈值
      - verbose(bool) 是否启用 DEBUG 日志
    :raises SystemExit: 参数不合法时打印错误并退出
    """
    parser = argparse.ArgumentParser(
        prog="quadrant_plot_meta_analysis",
        description="四象限性能对比可视化工具（Accuracy vs. Time）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 版本信息
    parser.add_argument(
        "--version",
        action="version",
        version="quadrant_plot_meta_analysis 1.0.0",
        help="显示脚本版本并退出"
    )
    # 输出路径
    parser.add_argument(
        "-o", "--output",
        dest="output",
        metavar="PATH",
        type=Path,
        help="高分辨率图像保存路径；若不指定则弹窗展示"
    )
    # 随机种子
    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        type=int,
        default=42,
        help="随机数种子，保证数据模拟可复现"
    )
    # 自定义阈值
    parser.add_argument(
        "--x-threshold",
        dest="x_threshold",
        type=float,
        help="Accuracy 参考阈值（越高越好），默认使用列均值"
    )
    parser.add_argument(
        "--y-threshold",
        dest="y_threshold",
        type=float,
        help="Time 参考阈值（越低越好），默认使用列均值"
    )
    # 调试模式
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="启用 DEBUG 级别日志输出，用于开发调试"
    )

    args = parser.parse_args()

    # 校验输出路径所在目录
    if args.output:
        out_dir = args.output.parent
        if not out_dir.exists(): parser.error(f"输出目录不存在或不可写：{out_dir}")

    return args


def main() -> None:
    """
    主流程：解析参数 → 初始化日志 → 应用绘图样式 → 数据模拟 → 四象限图绘制。

    退出码：
      0   执行成功
      1   主流程异常
      2   绘图样式初始化失败
    """
    args = parse_args()

    # 1. 配置日志
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("logs"))
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if hasattr(handler, "setLevel"): handler.setLevel(console_level)
    logger.info(
        "启动四象限可视化 | seed=%d | x_threshold=%s | y_threshold=%s | output=%s | verbose=%s",
        args.seed,
        args.x_threshold if args.x_threshold is not None else "（均值）",
        args.y_threshold if args.y_threshold is not None else "（均值）",
        args.output or "（屏幕展示）",
        args.verbose
    )

    # 2. 样式初始化
    try:
        setup_style()
        logger.debug("全局绘图样式应用成功")
    except Exception as err:
        logger.error("绘图样式初始化失败：%s", err)
        sys.exit(2)

    # 3. 数据模拟与绘图
    try:
        methods = ["Method_A", "Method_B", "Method_C", "Method_D", "Method_E"]
        # 数据模拟
        simulator = DataSimulator(methods, seed=args.seed, logger=logger)
        df = simulator.simulate()
        # 四象限绘制
        plotter = QuadrantPlotter(df, x_col="Accuracy", y_col="Time", logger=logger)
        plotter.plot(
            x_threshold=args.x_threshold,
            y_threshold=args.y_threshold,
            save_path=str(args.output) if args.output else None
        )
        logger.info("主流程执行完毕，程序正常退出")
        sys.exit(0)
    except Exception as err:
        logger.exception("主流程执行失败：%s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
