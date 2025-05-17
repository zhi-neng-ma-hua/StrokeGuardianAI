#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_simulation.py

Provides a function to generate simulated data for meta-analysis,
avoiding repeated definitions across multiple scripts.

Functions:
    generate_simulated_data:
        Generates a set of simulated study effect sizes and standard errors,
        and computes confidence intervals.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .logger_factory import LoggerFactory

# Automatically choose logger name: use module name when imported, or file stem if run as a script
module_name = __name__ if __name__ != "__main__" else Path(__file__).stem

logger = LoggerFactory.get_logger(
    module_name,
    level=logging.DEBUG,  # Root logger level
    console_level=logging.INFO,  # Console handler level
    logfile="logs/data_simulation.log",  # File to write logs
    file_level=logging.DEBUG,  # File handler level
    max_bytes=10 * 1024 * 1024,  # Max file size 10MB
    backup_count_bytes=3,  # Keep last 3 rotated files
    when="midnight",  # Time-based rotation frequency
    backup_count_time=7  # Keep logs for last 7 days
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
    Generate simulated data for meta-analysis.

    Effect sizes yi ~ N(effect_loc, effect_scale^2);
    Standard errors se ~ U(se_low, se_high);
    Compute two-sided confidence intervals at the specified level.

    :param n_studies: Number of studies to simulate (integer >= 1).
    :param effect_loc: Mean of the effect size distribution.
    :param effect_scale: Standard deviation of the effect size distribution (>0).
    :param se_low: Lower bound for standard error (>=0 and < se_high).
    :param se_high: Upper bound for standard error (> se_low).
    :param ci_level: Confidence level between 0 and 1 (e.g., 0.95 for 95% CI).
    :param seed: Random seed for reproducibility.
    :return: DataFrame with columns:
             - study: Study identifier ("Study 1", "Study 2", …)
             - yi: Simulated effect size
             - se: Simulated standard error
             - ci_lower: Lower bound of the CI
             - ci_upper: Upper bound of the CI
    :raises ValueError: If any parameter is out of expected range.
    """
    # Parameter validation
    if not isinstance(n_studies, int) or n_studies < 1:
        raise ValueError(f"n_studies must be a positive integer, got: {n_studies}")
    if effect_scale <= 0:
        raise ValueError(f"effect_scale must be > 0, got: {effect_scale}")
    if se_low < 0 or se_high <= se_low:
        raise ValueError(f"se_low and se_high must satisfy 0 <= se_low < se_high, got: {se_low}, {se_high}")
    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level must be between 0 and 1, got: {ci_level}")

    logger.info(
        "Generating simulated data: n_studies=%d, effect_loc=%.3f, effect_scale=%.3f, se_range=[%.3f,%.3f], ci_level=%.2f",
        n_studies, effect_loc, effect_scale, se_low, se_high, ci_level
    )

    # Use the modern RNG interface
    rng = np.random.default_rng(seed)
    # Draw effect sizes and standard errors
    yi = rng.normal(loc=effect_loc, scale=effect_scale, size=n_studies)
    se = rng.uniform(low=se_low, high=se_high, size=n_studies)

    # Compute the z-factor for the given confidence level
    alpha = 1.0 - ci_level
    z = abs(np.round(np.quantile(rng.standard_normal(1000000), [1 - alpha / 2])[0], 4))
    # Alternatively: use scipy.stats.norm.ppf(1-alpha/2)

    #  Build the DataFrame
    studies = [f"Study {i + 1}" for i in range(n_studies)]
    df = pd.DataFrame({
        "study": studies,
        "yi": yi,
        "se": se,
        "ci_lower": yi - z * se,
        "ci_upper": yi + z * se
    })

    logger.debug("Sample simulated data:\n%s", df.head(3).to_string(index=False))
    logger.info("Simulation complete: generated %d records.", n_studies)
    return df
