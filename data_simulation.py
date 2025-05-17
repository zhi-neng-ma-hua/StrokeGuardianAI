#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_simulation.py

提供元分析模拟数据生成函数，避免在多个脚本中重复定义。

函数:
    generate_simulated_data:
        生成一组模拟研究的效应值和标准误，并计算置信区间。
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .logger_factory import LoggerFactory

# 自动判断：如果是作为脚本运行，就用文件名作为 Logger 名；否则用 __name__
module_name = __name__ if __name__ != "__main__" else Path(__file__).stem

logger = LoggerFactory.get_logger(
    module_name,
    level=logging.DEBUG,  # Logger 总级别
    console_level=logging.INFO,  # 控制台 INFO+
    logfile="logs/data_simulation.log",  # 写入文件
    file_level=logging.DEBUG,  # 文件 DEBUG+
    max_bytes=10 * 1024 * 1024,  # 文件大小上限 10MB
    backup_count_bytes=3,  # 保留最近 3 个文件
    when="midnight",  # 按天切割
    backup_count_time=7  # 保留最近 7 天日志
)


def generate_simulated_data(
        n_studies: int = 8,
        effect_loc: float = 0.3,
        effect_scale: float = 0.5,
        se_low: float = 0.05,
        se_high: float = 0.15,
        ci_level: float = 0.95,
        seed: int = 42
) -> pd.DataFrame:
    """
    生成用于元分析的模拟数据。

    效应量 yi ~ 正态分布 N(effect_loc, effect_scale^2)；
    标准误 se ~ 均匀分布 U(se_low, se_high)；
    并基于 ci_level 计算双侧置信区间。

    :param n_studies: 要模拟的研究数，整数且 >=1。
    :param effect_loc: 效应量分布的均值 mu。
    :param effect_scale: 效应量分布的标准差 sigma，>0。
    :param se_low: 标准误下限，>=0 且 < se_high。
    :param se_high: 标准误上限，> se_low。
    :param ci_level: 置信水平，(0,1)，如 0.95 对应 95% CI。
    :param seed: 随机数种子。
    :return: 包含以下列的 DataFrame：
             - study: 研究名称 ("Study 1", "Study 2", …)
             - yi: 模拟效应量
             - se: 模拟标准误
             - ci_lower: 双侧下限
             - ci_upper: 双侧上限
    :raises ValueError: 当任一参数不符合预期时。
    """
    # 参数校验
    if not isinstance(n_studies, int) or n_studies < 1:
        raise ValueError(f"n_studies 必须为正整数，当前：{n_studies}")
    if effect_scale <= 0:
        raise ValueError(f"effect_scale 必须 >0，当前：{effect_scale}")
    if se_low < 0 or se_high <= se_low:
        raise ValueError(f"se_low/se_high 必须满足 0<=se_low<se_high，当前：{se_low},{se_high}")
    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level 必须在 (0,1) 之间，当前：{ci_level}")

    logger.info(
        "开始生成模拟数据: n_studies=%d, effect_loc=%.3f, effect_scale=%.3f, se_range=[%.3f,%.3f], ci_level=%.2f",
        n_studies, effect_loc, effect_scale, se_low, se_high, ci_level
    )

    # 使用现代 RNG 接口
    rng = np.random.default_rng(seed)
    # 效应量与标准误
    yi = rng.normal(loc=effect_loc, scale=effect_scale, size=n_studies)
    se = rng.uniform(low=se_low, high=se_high, size=n_studies)

    # 计算置信区间因子
    alpha = 1.0 - ci_level
    z = abs(np.round(np.quantile(rng.standard_normal(1000000), [1 - alpha / 2])[0], 4))
    # 或者直接用 scipy.stats.norm.ppf(1-alpha/2) 如果可用

    # 构造 DataFrame
    studies = [f"Study {i + 1}" for i in range(n_studies)]
    df = pd.DataFrame({
        "study": studies,
        "yi": yi,
        "se": se,
        "ci_lower": yi - z * se,
        "ci_upper": yi + z * se
    })

    logger.debug("模拟数据示例：\n%s", df.head(3).to_string(index=False))
    logger.info("模拟数据生成完成，共 %d 条记录。", n_studies)
    return df
