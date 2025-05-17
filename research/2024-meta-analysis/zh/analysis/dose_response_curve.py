#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名称：dose_response_curve_meta_analysis.py

模块用途：用于模拟与可视化 "剂量-反应" 关系曲线，集成数据模拟、多项式拟合、置信区间计算及高质量图形渲染。

主要功能：
  1. DoseResponseDataSimulator：基于指数模型，生成带噪声的剂量–反应数据；
  2. 拟合与评估：使用多项式回归拟合曲线，计算 R²、RMSE 等统计指标；
  3. 可视化：渲染散点、拟合曲线、置信区间、统计注释及外置图例；
  4. 样式配置：通过 DoseResponseStyle 统一管理画布、色彩、网格等视觉参数；
  5. 日志与异常：内置企业级日志与全流程异常捕获，确保可调试性与稳定性。

依赖环境：
  - Python 3.7+
  - numpy, pandas, matplotlib, seaborn, scikit-learn

使用示例：
  # 生成默认 8 级剂量数据并展示
  python dose_response_curve_meta_analysis.py
  # 自定义 12 级剂量、随机种子、噪声强度并保存图像
  python dose_response_curve_meta_analysis.py --n_levels 12 --seed 123 --noise_scale 0.1 --output result.png

退出码：
  0：执行成功
  1：参数错误或运行时异常
  2：用户中断（KeyboardInterrupt）

作者：智能麻花
日期：2025-05-15
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

# Todo # --- 日志模块初始化 --- #
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

# Todo # --- 全局样式配置 --- #
try:
    cfg = StyleConfig(grid=False, palette="colorblind", context="talk")
    StyleManager.apply(cfg)
except Exception as e:
    logger.error("应用全局绘图风格失败，将使用默认 Matplotlib 设置：%s", str(e), exc_info=True)


@dataclass
class DoseResponseDataSimulator:
    """
    类：剂量-反应数据模拟器 (DoseResponseDataSimulator)

    描述：
        基于指数增长模型 (effect = 0.5 + 0.8*(1 - exp(-dose/3))) 生成
        带高斯噪声的剂量-反应仿真数据，适用于药物剂量探索、
        工程参数敏感性分析、初步数据诊断等场景。

    :param dose_min: 最小剂量值，必须 < dose_max
    :param dose_max: 最大剂量值，必须 > dose_min
    :param n_levels: 剂量梯度数，整数且 >=2；过大时会触发性能警告
    :param seed: 随机数种子，用于结果可复现
    :param noise_scale: 噪声标准差，浮点数且 >=0；过大时会触发警告

    :raises ValueError: 参数不合法时抛出，包含详细诊断信息
    """
    dose_min: float = field(default=0.0, metadata={"help": "最低剂量"})
    dose_max: float = field(default=10.0, metadata={"help": "最高剂量"})
    n_levels: int = field(default=8, metadata={"help": "剂量梯度数"})
    seed: int = field(default=42, metadata={"help": "随机数种子"})
    noise_scale: float = field(default=0.05, metadata={"help": "噪声标准差"})

    def __post_init__(self):
        # 日志：记录初始化参数
        logger.info(
            "初始化 DoseResponseDataSimulator → dose_min=%.3f, dose_max=%.3f, "
            "n_levels=%d, noise_scale=%.3f, seed=%d",
            self.dose_min, self.dose_max, self.n_levels, self.noise_scale, self.seed
        )

        # 校验剂量范围
        if not (self.dose_min < self.dose_max):
            msg = (
                f"参数错误：dose_min({self.dose_min}) 必须小于 dose_max({self.dose_max})，"
                "请调整输入范围"
            )
            logger.error(msg)
            raise ValueError(msg)

        # 校验梯度数
        if self.n_levels < 2:
            msg = f"参数错误：n_levels({self.n_levels}) 必须 >= 2"
            logger.error(msg)
            raise ValueError(msg)
        if self.n_levels > 10000:
            logger.warning(
                "警告：n_levels=%d 过大，可能导致内存或计算性能问题", self.n_levels
            )

        # 校验噪声水平
        if self.noise_scale < 0:
            msg = f"参数错误：noise_scale({self.noise_scale}) 必须 >= 0"
            logger.error(msg)
            raise ValueError(msg)
        if self.noise_scale > (self.dose_max - self.dose_min):
            logger.warning(
                "噪声水平 noise_scale=%.3f 相对剂量范围过大，可能导致信号掩盖趋势",
                self.noise_scale
            )

        logger.info("参数校验通过, 模拟器初始化完成")

    def simulate(self) -> pd.DataFrame:
        """
        执行剂量-反应数据模拟。

        流程：
          1. 设置随机数种子，确保结果可复现；
          2. 生成等距剂量序列；
          3. 计算指数模型基准效应：0.5 + 0.8*(1 - exp(-dose/3))；
          4. 添加高斯噪声（均值=0，标准差=noise_scale）；
          5. 构建并返回 DataFrame。

        :return: 包含 'dose'、'effect' 两列的 DataFrame，行数 = n_levels
        :raises RuntimeError: 模拟过程中发生未知错误时抛出
        """
        logger.info(
            "开始模拟：dose in [%.3f, %.3f], levels=%d, noise_scale=%.3f",
            self.dose_min, self.dose_max, self.n_levels, self.noise_scale
        )

        try:
            # 1) 设置随机数种子
            np.random.seed(self.seed)

            # 2) 等距生成剂量序列
            doses = np.linspace(self.dose_min, self.dose_max, self.n_levels)
            logger.debug("生成剂量序列（前5项示例）: %s", np.round(doses[:5], 4))

            # 3) 计算基准效应
            base_effect = 0.5 + 0.8 * (1 - np.exp(-doses / 3.0))
            logger.debug("基准效应（无噪声）示例: %s", np.round(base_effect[:5], 4))

            # 4) 添加高斯噪声
            noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=self.n_levels)
            effects = base_effect + noise
            logger.debug("噪声示例: %s", np.round(noise[:5], 4))
            logger.debug("带噪声效应示例: %s", np.round(effects[:5], 4))

            # 5) 构建 DataFrame
            df = pd.DataFrame({"dose": doses, "effect": effects})
            logger.info("模拟完成，共生成 %d 条数据", len(df))
            return df

        except MemoryError as me:
            logger.exception("内存溢出：n_levels=%d", self.n_levels)
            raise RuntimeError("数据量过大，内存不足") from me

        except Exception as ex:
            logger.exception("数据模拟过程中发生未知错误：%s", ex)
            raise RuntimeError("剂量-反应数据模拟失败，请检查日志") from ex


@dataclass
class DoseResponseStyle:
    """
    类：剂量-反应曲线样式配置 (DoseResponseStyle)

    描述：
        该类用于集中定义并应用高质量剂量-反应图的可视化参数，
        包括画布尺寸、DPI、边距比例、散点与拟合线风格、网格线样式、
        坐标轴标签、标题属性等，确保在科研报告或企业展示中保持专业一致的视觉规范。

    用途：
        被 plotter.apply() 调用后，可自动将样式应用于 matplotlib 的 Figure/Axes。
    """

    fig_width: float = 12.0  # 图像宽度（单位：英寸）
    fig_height: float = 8.0  # 图像高度（单位：英寸）
    dpi: int = 300  # 图像分辨率

    # 子图边距，注意 left+right < 1, top+bottom < 1
    margin: Dict[str, float] = field(default_factory=lambda: {
        "left": 0.10, "right": 0.10, "top": 0.12, "bottom": 0.12
    })

    # 散点图样式配置
    scatter: Dict[str, Any] = field(default_factory=lambda: {
        "s": 100,
        "cmap": "viridis",
        "edgecolor": "#ffffff",
        "linewidth": 1.2,
        "alpha": 0.9
    })

    # 拟合曲线样式
    line: Dict[str, Any] = field(default_factory=lambda: {
        "linestyle": "-",
        "linewidth": 2.2,
        "color": "#d62728",
        "alpha": 0.9
    })

    # 主/次网格线样式
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

    # 标题样式
    title_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "fontsize": 18,
        "weight": "bold",
        "color": "#333333",
        "pad": 20
    })

    # 轴标签样式
    label_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "fontsize": 14,
        "color": "#333333",
        "labelpad": 15
    })

    # 外框
    border: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#909090",
        "linewidth": 1.5
    })

    ticks_color: str = "#666666"  # 坐标轴脊柱颜色

    def __post_init__(self):
        """初始化后自动进行参数合法性校验，并记录日志。"""
        try:
            # 校验尺寸与 DPI 合法性
            if self.fig_width <= 0 or self.fig_height <= 0:
                raise ValueError(f"画布尺寸非法：fig_width={self.fig_width}, fig_height={self.fig_height}")
            if self.dpi <= 0:
                raise ValueError(f"DPI 必须为正整数：dpi={self.dpi}")

            # 校验边距合法性
            total_h = self.margin["left"] + self.margin["right"]
            total_v = self.margin["top"] + self.margin["bottom"]
            for k, v in self.margin.items():
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"边距参数 margin['{k}']={v} 不在 [0,1] 范围内")
            if total_h >= 1.0 or total_v >= 1.0:
                raise ValueError(f"边距总和非法：左右={total_h:.2f}, 上下={total_v:.2f}，必须 < 1")

            # 日志：成功信息
            logger.info(
                "DoseResponseStyle 初始化成功：画布=%.1f×%.1f inches @%d DPI，边距=%s",
                self.fig_width, self.fig_height, self.dpi, self.margin
            )

        except ValueError as ve:
            logger.error("[样式初始化失败] %s", ve)
            raise

    def apply(self, fig: plt.Figure, ax: plt.Axes):
        """
        将当前样式应用于指定的图表组件（Figure 和 Axes）。

        :param fig: matplotlib Figure 对象
        :param ax: matplotlib Axes 对象
        :raises RuntimeError: 若应用失败则抛出异常
        """
        logger.debug("开始应用 DoseResponseStyle 样式")
        try:
            # 设置画布尺寸与分辨率
            fig.set_size_inches(self.fig_width, self.fig_height)
            fig.set_dpi(self.dpi)
            fig.patch.set_facecolor("#fbfbfb")
            fig.patch.set_edgecolor(self.border["color"])
            fig.patch.set_linewidth(self.border["linewidth"])
            ax.set_facecolor("#f7f7f7")

            # 隐藏上/右脊柱，调整左/下脊柱样式
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_color(self.ticks_color)
                ax.spines[side].set_linewidth(1.2)

            # 添加主/次网格线
            ax.grid(**self.grid_major)
            ax.minorticks_on()
            ax.grid(**self.grid_minor)

            # 设置标题和标签
            ax.set_title("模拟剂量-反应曲线", **self.title_kwargs)
            ax.set_xlabel("剂量水平 (Dose)", **self.label_kwargs)
            ax.set_ylabel("效应值 (Effect)", **self.label_kwargs)

            # 应用子图边距
            fig.subplots_adjust(
                left=self.margin["left"],
                right=1.0 - self.margin["right"],
                top=1.0 - self.margin["top"],
                bottom=self.margin["bottom"]
            )

            logger.debug("DoseResponseStyle 样式应用成功")
        except Exception as e:
            logger.exception("应用样式失败：%s", e)
            raise RuntimeError(f"应用样式失败：{e}") from e


@dataclass
class DoseResponseCurvePlotter:
    """
    类：剂量-反应曲线绘制器 (DoseResponseCurvePlotter)

    描述：
        基于给定的剂量-效应数据与样式配置，绘制专业级剂量-反应散点图及多项式拟合曲线，
        并添加置信区间、数值标签、色条、统计指标与外置图例，适用于科研报告与企业级 BI 展示。

    :param data: 包含 'dose' 与 'effect' 列的 DataFrame
    :param style: DoseResponseStyle 实例，用于统一可视化风格配置

    :raises ValueError: 若输入数据缺少必需列
    """
    data: pd.DataFrame
    style: DoseResponseStyle

    def __post_init__(self):
        # 校验输入 DataFrame 是否包含必要列
        missing = {"dose", "effect"} - set(self.data.columns)
        if missing:
            msg = f"[初始化失败] 数据缺少必需列: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        # 复制数据，避免外部修改
        self.data = self.data[["dose", "effect"]].copy()
        logger.info("DoseResponseCurvePlotter 初始化完成，数据条数：%d", len(self.data))

    def plot(self, save_path: Optional[str] = None, fit_degree: int = 3) -> None:
        """
        绘制剂量-反应曲线，并根据参数决定展示或保存。

        流程：
          1. 创建 Figure/Axes 并应用统一样式；
          2. 绘制渐变散点（大小与颜色映射）；
          3. 添加 Colorbar；
          4. 多项式拟合及固定误差置信区间填充；
          5. 为每点添加数值标签；
          6. 计算并注释 R²、RMSE；
          7. 配置外置图例；
          8. 调用 tight_layout 调整布局；
          9. 保存或展示图像。

        :param save_path: 输出文件路径；若为 None，则弹窗展示
        :param fit_degree: 多项式拟合阶数，整数且 >=1
        :return: None
        :raises ValueError: fit_degree 非正整数或数据量不足以拟合时
        :raises RuntimeError: 绘图过程中发生未知错误
        """
        logger.info("开始绘图：多项式拟合阶数 = %d", fit_degree)

        # 校验拟合阶数
        if fit_degree < 1:
            msg = f"[参数错误] 拟合阶数 fit_degree={fit_degree} 必须 >=1"
            logger.error(msg)
            raise ValueError(msg)
        if len(self.data) <= fit_degree:
            msg = f"[数据量不足] 数据条数={len(self.data)} <= 拟合阶数={fit_degree}"
            logger.error(msg)
            raise ValueError(msg)

        try:
            # 1) 创建画布与坐标轴，应用样式
            fig, ax = plt.subplots()
            self.style.apply(fig, ax)
            doses = self.data["dose"].to_numpy()
            effects = self.data["effect"].to_numpy()
            logger.debug(
                "剂量范围 [%.3f, %.3f], 效应范围 [%.3f, %.3f]",
                doses.min(), doses.max(), effects.min(), effects.max()
            )

            # 2) 渐变散点：大小线性映射、颜色映射
            span = np.ptp(effects) + 1e-9
            norm = (effects - effects.min()) / span
            sizes = 60 + 120 * norm
            sc = ax.scatter(
                doses, effects,
                c=effects, s=sizes,
                cmap="viridis",
                edgecolor="white", linewidth=1.2, alpha=0.9
            )
            logger.debug("散点绘制完成：size 范围 [%.1f, %.1f]", sizes.min(), sizes.max())

            # 3) 添加色条
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("效应值 (Effect)", fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            logger.debug("Colorbar 添加完成")

            # 4) 多项式拟合 & 置信区间
            coeffs = np.polyfit(doses, effects, fit_degree)
            poly = np.poly1d(coeffs)
            xs = np.linspace(doses.min(), doses.max(), 300)
            y_fit = poly(xs)
            ci_err = 0.1  # 固定误差宽度
            ax.fill_between(
                xs, y_fit - ci_err, y_fit + ci_err,
                color=self.style.line["color"], alpha=0.2,
                label="95% 置信区间"
            )
            ax.plot(
                xs, y_fit,
                linestyle=self.style.line["linestyle"],
                linewidth=self.style.line["linewidth"],
                color=self.style.line["color"],
                alpha=self.style.line.get("alpha", 1.0),
                label=f"拟合曲线 (deg={fit_degree})"
            )
            logger.debug("拟合曲线及置信区间绘制完成")

            # 5) 数据标签
            for x, y in zip(doses, effects):
                ax.text(
                    x, y + 0.04, f"{y:.2f}",
                    ha="center", va="bottom",
                    fontsize=12, color="#1f77b4",
                    bbox=dict(boxstyle="round", fc="white", ec="#1f77b4", alpha=0.7)
                )
            logger.debug("数值标签添加完成")

            # 6) 统计指标
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
            logger.debug("统计指标注释：R²=%.3f, RMSE=%.3f", r2, rmse)

            # 7) 外置图例
            legend = ax.legend(
                frameon=True, facecolor="white", edgecolor="#cccccc",
                loc="upper left", bbox_to_anchor=(1.02, 1)
            )
            legend.get_frame().set_alpha(0.8)
            for lh in legend.get_lines():
                lh.set_linewidth(1.5)
            logger.debug("图例配置完成")

            # 8) 布局与输出
            plt.tight_layout(pad=3.0)
            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
                logger.info("图像已保存至：%s", save_path)
            else:
                plt.show()
                logger.info("图形展示完成")
        except np.linalg.LinAlgError as le:
            logger.exception("多项式拟合失败：%s", le)
            raise RuntimeError("多项式拟合过程出错") from le
        except Exception as ex:
            logger.exception("绘图过程中发生未知错误：%s", ex)
            raise RuntimeError("Plot 执行失败，请检查日志") from ex


def parse_args():
    """
    解析命令行参数。

    :return: argparse.Namespace 对象，包含 dose_min, dose_max, n_levels, seed, noise_scale, output
    """
    parser = argparse.ArgumentParser(
        prog="dose_response_curve_meta_analysis.py",
        description="模拟并可视化剂量-反应曲线，支持自定义剂量级数、随机种子、噪声强度和输出路径。"
    )
    parser.add_argument("--dose-min", type=float, default=0.0,
                        help="最小剂量，必须小于 --dose-max （默认: 0.0）")
    parser.add_argument("--dose-max", type=float, default=10.0,
                        help="最大剂量，必须大于 --dose-min （默认: 10.0）")
    parser.add_argument("--n-levels", type=int, default=8,
                        help="剂量梯度级数，整数且>=2（默认: 8）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机数种子，用于结果复现（默认: 42）")
    parser.add_argument("--noise-scale", type=float, default=0.05,
                        help="高斯噪声标准差，>=0（默认: 0.05）")
    parser.add_argument("--output", type=str, default=None,
                        help="若提供，保存图像到该路径；否则弹窗展示")
    return parser.parse_args()


def main():
    """
    主程序入口：解析参数，执行数据模拟、样式初始化与绘图。

    流程：
      1. 解析命令行参数；
      2. 初始化模拟器并生成数据；
      3. 初始化样式配置；
      4. 创建绘制器并渲染图像；
      5. 根据 --output 保存或展示。

    :return: None
    :raises ValueError: 参数校验失败
    :raises RuntimeError: 模块执行过程出错
    :raises KeyboardInterrupt: 用户中断
    """
    args = parse_args()
    logger.info("=== 程序开始：剂量-反应曲线模拟与渲染 ===")
    logger.debug("命令行参数：%s", args)

    try:
        # 2) 数据模拟
        logger.info("1. 数据模拟阶段")
        sim = DoseResponseDataSimulator(
            dose_min=args.dose_min,
            dose_max=args.dose_max,
            n_levels=args.n_levels,
            seed=args.seed,
            noise_scale=args.noise_scale
        )
        df = sim.simulate()

        # 3) 样式配置
        logger.info("2. 样式初始化阶段")
        style = DoseResponseStyle()

        # 4) 绘图渲染
        logger.info("3. 绘图渲染阶段")
        plotter = DoseResponseCurvePlotter(df, style)
        plotter.plot(save_path=args.output)

        logger.info("=== 程序执行成功，退出码 0 ===")
        sys.exit(0)
    except KeyboardInterrupt:
        # 捕获用户中断
        logger.warning("程序被用户中断，退出码 2")
        sys.exit(2)
    except ValueError as ve:
        # 参数或数据校验错误
        logger.error("参数错误：%s，退出码 1", ve)
        sys.exit(1)
    except Exception as ex:
        # 其他运行时错误
        logger.exception("运行时异常：%s，退出码 1", ex)
        sys.exit(1)


if __name__ == "__main__":
    main()
