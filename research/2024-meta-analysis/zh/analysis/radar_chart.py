#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达图 Meta 分析可视化工具

功能概述：
    基于 Pandas 与 Matplotlib，模拟多干预/多算法在四项关键指标（Accuracy、Time、Safety、Resource）上的性能数据，并生成极坐标雷达图。

核心特点：
  1. 指标预处理：对 Time、Resource（越低越好）自动反向映射；
  2. 全量归一：将所有指标标准化至 [0,1]，消除量纲影响，提升对比性；
  3. 出版级美化：同心背景环、极坐标网格、高亮数据点、描边与数值标签；
  4. 灵活输出：支持交互式展示与高分辨率图像保存两种模式；
  5. 稳定可靠：内置中文结构化日志、样式管理、全面异常捕获，满足企业级部署与科研级可复现性需求。

使用示例：
  # 交互式展示
  python radar_chart_meta_analysis.py
  # 保存到文件
  python radar_chart_meta_analysis.py -o output/radar.png
  # 调试模式
  python radar_chart_meta_analysis.py --verbose

先决条件：
  Python ≥ 3.8
  ├─ numpy
  ├─ pandas
  ├─ matplotlib
  ├─ seaborn
  ├─ utils/logger_factory
  └─ utils/plt_style_manager

作者：智能麻花 <zhinengmahua@gmail.com>
日期：2025-05-18
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
    多维性能数据模拟器

    基于给定的干预/算法列表和各项指标范围，批量生成可复现的多维性能数据，适用于算法性能评估、企业级部署与科研可复现。
    内置严格的输入校验、结构化日志和异常捕获，保证生产环境可靠性。
    """

    def __init__(
            self,
            methods: List[str],
            seed: int = 42,
            logger: Optional[logging.Logger] = None
    ) -> None:
        """
        初始化模拟器并校验输入参数

        :param methods: 非空字符串列表，干预或算法名称，自动去重保留原序
        :param seed: 随机种子，保证模拟结果可复现
        :param logger: 可选的 logging.Logger 实例；默认使用模块级 logger
        :raises TypeError: methods 不是列表或包含非字符串元素
        :raises ValueError: methods 列表为空或去重后为空
        """
        # 参数校验
        if not isinstance(methods, list): raise TypeError("methods 必须为列表类型")
        cleaned = []
        for m in methods:
            if not isinstance(m, str): raise TypeError(f"方法名称必须为字符串，收到：{m!r}")
            name = m.strip()
            if name: cleaned.append(name)
        # 去重并保持原序
        unique_methods = list(dict.fromkeys(cleaned))
        if not unique_methods: raise ValueError("methods 列表至少包含一个非空字符串")
        self.methods = unique_methods
        self.seed = int(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(
            "初始化 MultiDimensionalDataSimulator：methods=%s, seed=%d",
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
        生成多维度性能模拟数据

        :param acc_range: 准确率范围[min, max]，值越大越好
        :param time_range: 时间范围[min, max]，值越小越好
        :param safety_range: 安全性范围[min, max]，值越大越好
        :param resource_range: 资源消耗范围[min, max]，值越小越好
        :return: 包含 ["Method","Accuracy","Time","Safety","Resource"] 列的 pandas.DataFrame
        :raises RuntimeError: 在数据生成或组装过程中发生异常
        """
        try:
            n = len(self.methods)
            self.logger.info("开始模拟多维度性能数据，共 %d 个方法", n)

            # 设置随机种子，保证可复现
            np.random.seed(self.seed)

            # 一次性生成所有指标数组
            metrics = {
                "Accuracy": np.random.uniform(acc_range[0], acc_range[1], n),
                "Time": np.random.uniform(time_range[0], time_range[1], n),
                "Safety": np.random.uniform(safety_range[0], safety_range[1], n),
                "Resource": np.random.uniform(resource_range[0], resource_range[1], n),
            }

            # 构建 DataFrame 并插入 Method 列
            df = pd.DataFrame(metrics, index=self.methods).reset_index()
            df.rename(columns={"index": "Method"}, inplace=True)

            # 记录示例数据以便快速验证
            preview = df.head(1).to_dict(orient="records")[0]
            self.logger.debug("模拟数据示例：%s", preview)
            self.logger.info("多维度性能数据模拟完成，共 %d 条记录", n)

            return df
        except Exception as e:
            # 结构化记录完整异常栈，便于问题定位
            self.logger.exception("多维度性能数据模拟失败：%s", e)
            raise RuntimeError("MultiDimensionalDataSimulator.simulate 执行失败") from e


class RadarChartPlotter:
    """
    雷达图可视化器

    功能：
      - 对 "越低越好" 的指标自动反向处理；
      - 对所有指标归一化至 [0,1]；
      - 生成出版级极坐标雷达图，带同心背景环、高亮顶点、描边、数值标签及灵活图例布局；
      - 支持交互式展示与高分辨率文件保存。
    """

    def __init__(
            self,
            data: pd.DataFrame,
            reverse_cols: List[str],
            logger: Optional[logging.Logger] = None,
            figsize: tuple = (12, 8),
            title: str = "多干预/算法多维度雷达图",
            palette_name: str = "Set2"
    ) -> None:
        """
        初始化绘图器并校验输入

        :param data: 原始数据，包含 "Method" 列和若干指标列
        :param reverse_cols: 需要反向映射的列名列表（越低越好）
        :param logger: 日志记录器；默认获取模块级 logger
        :param figsize: 图像尺寸，单位英寸
        :param title: 雷达图标题
        :param palette_name: seaborn 调色板名称
        :raises ValueError: 当 reverse_cols 在 data 中不存在时
        """
        self.logger = logger or logging.getLogger(__name__)
        self.data = data.copy()
        self.methods = data["Method"].tolist()
        # 自动识别所有指标列
        self.metrics = [c for c in data.columns if c != "Method"]
        missing = set(reverse_cols) - set(self.metrics)
        if missing: raise ValueError(f"反向处理列缺失：{missing}")
        self.reverse_cols = reverse_cols
        self.figsize = figsize
        self.title = title
        self.palette_name = palette_name
        self.logger.debug(
            "初始化 RadarChartPlotter：methods=%s, metrics=%s, reverse_cols=%s, figsize=%s, title=%r, palette=%s",
            self.methods, self.metrics, self.reverse_cols, self.figsize, self.title, self.palette_name
        )

    def _normalize(self) -> pd.DataFrame:
        """
        反向映射并归一化所有指标到 [0,1]

        :return: 归一化后的 DataFrame（保留 Method 列）
        """
        df = self.data.copy()
        # 反向映射
        for col in self.reverse_cols:
            df[col] = df[col].max() - df[col]
            self.logger.debug("已反向映射列：%s", col)
        # 归一化
        for col in self.metrics:
            min_val, max_val = df[col].min(), df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
            self.logger.debug("归一化 %s：min=%.3f, max=%.3f", col, min_val, max_val)
        return df

    def plot(self, save_path: Optional[str] = None, legend_loc: str = "lower center") -> None:
        """
        渲染雷达图

        :param save_path: 图像保存路径，None 则弹窗展示
        :param legend_loc: 图例位置，参考 matplotlib legend loc 参数
        """
        try:
            df_norm = self._normalize()
            n_vars = len(self.metrics)
            # 1. 计算角度并闭合
            angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
            angles += angles[:1]

            # 2. 创建极坐标画布
            fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
            fig.patch.set_facecolor("#f8f8f8")
            fig.patch.set_edgecolor("#cccccc")
            fig.patch.set_linewidth(1.5)
            ax.set_facecolor("#ffffff")
            ax.patch.set_alpha(0.9)
            # 主网格
            # ax.grid(which="major", color="#cccccc", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.grid(which="major", linestyle="-.", linewidth=0.6, color="#bbbbbb")
            ax.grid(which="minor", linestyle=":", linewidth=0.4, color="#dddddd")
            # 次级网格
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.125))
            ax.yaxis.grid(which="minor", color="#eeeeee", linestyle=":", linewidth=0.4)

            # 3. 同心背景环
            for r, color in zip([1, 0.75, 0.5, 0.25], sns.light_palette("#dddddd", n_colors=4)):
                ax.fill_between(angles, r if r > 0 else 0, r - 0.25 if r > 0.25 else 0, color=color, zorder=0)

            # 4. 角度与刻度
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.metrics, fontsize=12, fontweight="bold", color="#333")
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels([f"{v:.2f}" for v in [0.25, 0.5, 0.75, 1.0]], fontsize=10, color="#555")
            ax.set_ylim(0, 1)
            ax.spines["polar"].set_linewidth(1.5)

            # 5. 调色板
            palette = sns.color_palette(self.palette_name, n_colors=len(df_norm))

            # 6. 绘制各方法
            for idx, (method, row) in enumerate(df_norm.set_index("Method").iterrows()):
                values = row[self.metrics].tolist() + [row[self.metrics[0]]]
                color = palette[idx]
                # 折线
                ax.plot(angles, values, color=color, linewidth=3.0, alpha=0.9, label=method, zorder=3)
                # 光晕描边
                line = ax.lines[-1]
                line.set_path_effects([Stroke(linewidth=6, foreground="white", alpha=0.5), Normal()])
                # 填充
                ax.fill(angles, values, color=color, alpha=0.2, zorder=2)
                # 顶点与标签
                for angle, val in zip(angles, values):
                    # 双圈高亮顶点
                    ax.scatter(angle, val, s=80, edgecolors=color, facecolors="white", linewidth=1.5, zorder=4)
                    ax.scatter(angle, val, s=30, color=color, zorder=5)
                    # 阴影标签
                    txt = ax.text(
                        angle, val + 0.05, f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=12, color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1, alpha=0.8),
                        zorder=6
                    )
                    txt.set_path_effects([Stroke(linewidth=2, foreground="white", alpha=0.8), Normal()])

            # 7. 标题与图例
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

            # 8. 布局与输出
            plt.tight_layout(pad=2)
            plt.subplots_adjust(bottom=0.15, top=0.85)

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info("雷达图已保存至：%s", save_path)
            else:
                plt.show()
        except Exception as e:
            self.logger.exception("雷达图绘制失败：%s", e)
            raise RuntimeError("RadarChartPlotter.plot 执行失败") from e


def parse_args() -> argparse.Namespace:
    """
    解析并校验命令行参数。

    支持以下选项：
      - -o, --output        : 图像保存路径（可选），若不指定则弹窗展示；
      - -s, --seed          : 随机数种子，保证数据模拟可复现（默认 42）；
      - -r, --reverse-cols  : 需要反向处理的指标列，逗号分隔（默认 "Time,Resource"）；
      - -v, --verbose       : 开启 DEBUG 级别日志，用于开发调试；
      - --version           : 显示脚本版本并退出。

    :return: argparse.Namespace，包含以下字段：
      - output       (Path|None) : 图像保存路径
      - seed         (int)       : 随机种子
      - reverse_cols (List[str]): 反向处理的列名列表
      - verbose      (bool)      : 是否启用 DEBUG 日志
    :raises SystemExit: 参数不合法时打印错误并退出
    """
    parser = argparse.ArgumentParser(
        prog="radar_chart_meta_analysis",
        description="多干预/算法多维度雷达图可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--version",
        action="version",
        version="radar_chart_meta_analysis 1.0.0",
        help="显示脚本版本并退出"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output",
        metavar="PATH",
        type=Path,
        help="输出图像保存路径；若不指定则弹窗展示"
    )

    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        type=int,
        default=42,
        help="随机数种子，保证数据模拟可复现"
    )

    parser.add_argument(
        "-r", "--reverse-cols",
        dest="reverse_cols",
        type=lambda s: [item.strip() for item in s.split(",") if item.strip()],
        default=["Time", "Resource"],
        help="逗号分隔的需反向处理指标列名（越低越好）"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="开启 DEBUG 日志输出，用于开发调试"
    )

    args = parser.parse_args()

    # 校验：输出路径所在目录必须存在或可创建
    if args.output:
        out_dir = args.output.parent
        if not out_dir.exists(): parser.error(f"输出目录不存在：{out_dir}")

    # 校验：reverse_cols 必须非空
    if not args.reverse_cols: parser.error("必须指定至少一个 --reverse-cols 参数")

    return args


def main() -> None:
    """
    主流程：解析参数 → 初始化日志与样式 → 数据模拟 → 绘制雷达图 → 结束。

    退出码：
      0   执行成功
      1   程序运行异常
      2   样式初始化失败
    """
    args = parse_args()

    # 1. 配置日志
    console_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name=Path(__file__).stem, log_dir=Path("logs"))
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if hasattr(handler, "setLevel"): handler.setLevel(console_level)
    logger.info(
        "启动雷达图可视化工具 | seed=%d | reverse_cols=%s | output=%s | verbose=%s",
        args.seed,
        ",".join(args.reverse_cols),
        args.output or "（不保存）",
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
        # 固定方法列表，可根据实际需求替换
        methods = ["Method_A", "Method_B", "Method_C", "Method_D", "Method_E"]
        simulator = MultiDimensionalDataSimulator(methods, seed=args.seed, logger=logger)
        df = simulator.simulate()
        plotter = RadarChartPlotter(
            data=df,
            reverse_cols=args.reverse_cols,
            logger=logger,
            figsize=(12, 8),
            title="多干预/算法多维度雷达图",
            palette_name="Set2"
        )
        plotter.plot(save_path=str(args.output) if args.output else None, legend_loc="lower center")
        logger.info("雷达图生成并输出完成")
        sys.exit(0)
    except Exception as err:
        logger.exception("主流程执行失败：%s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
