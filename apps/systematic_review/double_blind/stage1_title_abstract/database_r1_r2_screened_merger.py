# -*- coding: utf-8 -*-
"""
database_r1_r2_screened_merger.py

Stage 1 (Title / Abstract) Database-specific Double-Blind R1 / R2 Merging Tool.

Main Features:
1. Reads the R1 / R2 screening results from different databases (including Decision / Notes columns).
2. Checks the completeness of the Decision / Notes fields before merging to prevent null or erroneous data from entering the merge.
3. Merges R1 and R2 data strictly by primary key fields (Article Title / Publication Year / DOI).
4. Generates a merged result with suffix columns (Decision_R1 / Notes_R1 / Decision_R2 / Notes_R2) for subsequent arbitration or R3 intervention.

Author: Aiden Cao <zhinengmahua@gmail.com>
Date: 2025-05-22
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from utils.exceptions import DataValidationError
from utils.data_io import load_table, save_table
from utils.logger_manager import LoggerManager

PathLike = Union[str, Path]

# Key primary key fields (used to uniquely identify the literature)
KEY_FIELDS: Tuple[str, ...] = ("Article Title", "Publication Year", "DOI")

# Mapping of source files (for each database) and the output file names for this script
SOURCE_FILE_MAP: Dict[str, Dict[str, str]] = {
    "scopus": {
        "r1": "screened_scopus_papers_R1.xlsx",
        "r2": "screened_scopus_papers_R2.xlsx",
        "merged": "screened_scopus_papers_R1_R2.xlsx",
    },
    "wos": {
        "r1": "screened_web_of_science_papers_R1.xlsx",
        "r2": "screened_web_of_science_papers_R2.xlsx",
        "merged": "screened_web_of_science_papers_R1_R2.xlsx",
    },
    "ieee": {
        "r1": "screened_ieee_papers_R1.xlsx",
        "r2": "screened_ieee_papers_R2.xlsx",
        "merged": "screened_ieee_papers_R1_R2.xlsx",
    },
    "pubmed": {
        "r1": "screened_pubmed_papers_R1.xlsx",
        "r2": "screened_pubmed_papers_R2.xlsx",
        "merged": "screened_pubmed_papers_R1_R2.xlsx",
    },
}


def setup_logger(
        name: str = "stage1_database_r1r2_screened_merger",
        verbose: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger.

    :param name: Logger name
    :param verbose: Whether to enable DEBUG level logging
    :return: logging.Logger instance
    """
    return LoggerManager.setup_logger(
        logger_name=name,
        module_name=__name__,
        verbose=verbose,
    )


class PreMergeValidator:
    """
    Pre-merge data integrity validator:
    - Checks if Decision / Notes columns exist.
    - Checks if Decision / Notes contain missing or "pseudo-missing" values (e.g., empty strings or whitespace).
    """

    REQUIRED_COLS = ["Decision", "Notes"]

    @staticmethod
    def _is_missing(series: pd.Series) -> pd.Series:
        """
        Checks whether a column contains missing values (including NaN, None, empty strings, all-whitespace, or 'nan' string).

        :param series: Series to check
        :return: Boolean Series, True indicates missing or invalid values
        """
        return series.isnull() | (series.astype(str).str.strip().replace("nan", "") == "")

    @classmethod
    def validate(
            cls,
            df: pd.DataFrame,
            label: str,
            logger: logging.Logger,
    ) -> None:
        """
        Performs strict validation of Decision / Notes fields before merging.

        :param df: Input DataFrame, must contain Decision / Notes columns
        :param label: Label (e.g., "Scopus R1") for logging prefix
        :param logger: Logger instance
        :raises DataValidationError: If fields are missing or contain invalid values
        """
        # 1. Check if the necessary columns exist
        missing_cols = [c for c in cls.REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            logger.error(f"【✗】{label}: Missing required fields: {missing_cols}")
            raise DataValidationError(f"{label} missing required fields: {missing_cols}")

        # 2. Check for missing values
        decision_missing = cls._is_missing(df["Decision"])
        notes_missing = cls._is_missing(df["Notes"])
        mask = decision_missing | notes_missing

        if mask.any():
            show_cols = [c for c in (*KEY_FIELDS, *cls.REQUIRED_COLS) if c in df.columns]
            bad_rows = df.loc[mask, show_cols]
            logger.error(
                f"【✗】{label}: Missing or invalid Decision / Notes values, examples (top 5 rows):\n"
                f"{bad_rows.head().to_string(index=False)}"
            )
            logger.error(f"【✗】{label}: Found {int(mask.sum())} invalid records.")
            raise DataValidationError(
                f"{label} data integrity validation failed: {int(mask.sum())} Decision/Notes missing or invalid.")

        logger.info(f"【✓】{label}: Decision / Notes integrity check passed, total {len(df)} rows.")


class DatabaseR1R2ScreenedMerger:
    """Controller for merging Stage 1 R1 / R2 double-blind screening results by database."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the merging controller.

        :param logger: Logger instance, if None, one will be created automatically
        :raises DataValidationError: If the data directories do not exist
        """
        self.logger = logger or setup_logger()
        self._init_dirs()

    def _init_dirs(self) -> None:
        """
        Initializes and checks the data directory structure.

        :raises DataValidationError: If the R1 or R2 directory does not exist
        """
        # Current file path: apps/systematic_review/double_blind/stage1_title_abstract/database_r1_r2_screened_merger.py
        project_root = Path(__file__).resolve().parents[4]
        stage_root = (
                project_root
                / "data"
                / "systematic_review"
                / "double_blind"
                / "stage1_title_abstract"
        )

        self.r1_dir = stage_root / "R1"
        self.r2_dir = stage_root / "R2"
        self.merged_dir = stage_root / "merged"

        if not self.r1_dir.is_dir():
            raise DataValidationError(f"R1 data directory does not exist: {self.r1_dir}")
        if not self.r2_dir.is_dir():
            raise DataValidationError(f"R2 data directory does not exist: {self.r2_dir}")

        self.merged_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"[Paths] R1 directory: {self.r1_dir} | R2 directory: {self.r2_dir} | Merged output directory: {self.merged_dir}")

    @staticmethod
    def _normalize_keys(df: pd.DataFrame, key_fields: List[str]) -> pd.DataFrame:
        """
        Standardize the primary key columns to ensure consistent types and strip whitespace.

        :param df: Input DataFrame
        :param key_fields: List of primary key column names
        :return: Standardized DataFrame
        """
        df = df.copy()
        for col in key_fields:
            if col not in df.columns:
                continue
            if col == "Publication Year":
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .astype("Int64")
                    .astype(str)
                )
            else:
                df[col] = df[col].astype(str).str.strip()
        return df

    def _merge_single_source(
            self,
            source: str,
            r1_path: Path,
            r2_path: Path,
            key_fields: List[str],
    ) -> pd.DataFrame:
        """
        Merge R1 / R2 screening results for a single source (e.g., Scopus).

        :param source: Source identifier (e.g., "scopus")
        :param r1_path: Path to R1 result file
        :param r2_path: Path to R2 result file
        :param key_fields: List of primary key column names
        :return: Merged DataFrame
        :raises DataValidationError: If the primary key sets are inconsistent or missing key columns
        """
        label_r1 = f"{source.upper()} R1"
        label_r2 = f"{source.upper()} R2"

        # 1. Read and validate R1 / R2
        df_r1 = load_table(r1_path, logger=self.logger)
        df_r2 = load_table(r2_path, logger=self.logger)

        PreMergeValidator.validate(df_r1, label_r1, self.logger)
        PreMergeValidator.validate(df_r2, label_r2, self.logger)

        # 2. Standardize primary keys and check for consistency
        df1 = self._normalize_keys(df_r1, key_fields)
        df2 = self._normalize_keys(df_r2, key_fields)

        # Check if the key columns exist
        missing_key_cols = [c for c in key_fields if c not in df1.columns or c not in df2.columns]
        if missing_key_cols:
            raise DataValidationError(f"{source.upper()} R1/R2 missing primary key columns: {missing_key_cols}")

        keys1 = set(tuple(row) for row in df1[key_fields].values)
        keys2 = set(tuple(row) for row in df2[key_fields].values)

        if keys1 != keys2:
            only_in_r1 = keys1 - keys2
            only_in_r2 = keys2 - keys1
            if only_in_r1:
                self.logger.error(
                    f"【✗】{source.upper()}: {len(only_in_r1)} keys are unique to R1: {list(only_in_r1)[:5]} ..."
                )
            if only_in_r2:
                self.logger.error(
                    f"【✗】{source.upper()}: {len(only_in_r2)} keys are unique to R2: {list(only_in_r2)[:5]} ..."
                )
            raise DataValidationError(f"{source.upper()} R1/R2 primary key sets do not match, merge aborted.")

        self.logger.info(f"【✓】{source.upper()}: R1 / R2 primary key sets are identical, starting merge.")

        # 3. Add suffixes to Decision / Notes to avoid conflicts
        r1_suffix_cols = [c for c in ("Decision", "Notes") if c in df1.columns]
        r2_suffix_cols = [c for c in ("Decision", "Notes") if c in df2.columns]

        meta_cols = [c for c in df1.columns if c not in r1_suffix_cols]

        df1_renamed = df1[meta_cols + r1_suffix_cols].rename(columns={col: f"{col}_R1" for col in r1_suffix_cols})
        df2_renamed = df2[list(key_fields) + r2_suffix_cols].rename(
            columns={col: f"{col}_R2" for col in r2_suffix_cols})

        # 4. Strict one-to-one merge
        merged = pd.merge(
            df1_renamed,
            df2_renamed,
            on=key_fields,
            how="left",
            validate="one_to_one",
        )

        # 5. Simple post-merge integrity check (record potential null values)
        for col in ("Decision_R1", "Notes_R1", "Decision_R2", "Notes_R2"):
            if col not in merged.columns:
                self.logger.warning(f"[Notice] {source.upper()}: Column {col} missing in merge result.")
                continue
            missing = merged[col].astype(str).str.strip().replace("nan", "") == ""
            if missing.any():
                cnt = int(missing.sum())
                self.logger.warning(f"【!】{source.upper()}: {cnt} null values found in {col} column.")

        self.logger.info(f"【✓】{source.upper()}: R1 / R2 merge complete, {len(merged)} rows.")
        return merged

    def run(self) -> None:
        """
        Executes the full process: merges R1 / R2 for each database and outputs to the merged directory.

        :return: None
        :raises DataValidationError: If all sources fail to merge
        """
        merged_any = False

        for source, names in SOURCE_FILE_MAP.items():
            r1_path = self.r1_dir / names["r1"]
            r2_path = self.r2_dir / names["r2"]
            out_path = self.merged_dir / names["merged"]

            if not r1_path.is_file():
                self.logger.warning(f"[Missing] R1 file for source {source} does not exist: {r1_path}")
                continue
            if not r2_path.is_file():
                self.logger.warning(f"[Missing] R2 file for source {source} does not exist: {r2_path}")
                continue

            self.logger.info(f"[Starting] Source {source}: R1={r1_path.name} | R2={r2_path.name}")

            merged_df = self._merge_single_source(source, r1_path, r2_path, list(KEY_FIELDS))
            save_table(merged_df, out_path, logger=self.logger)
            self.logger.info(
                f"[Completed] Merged R1/R2 for source {source}, output file: {out_path} (rows: {len(merged_df)})")
            merged_any = True

        if not merged_any:
            raise DataValidationError(
                "All sources' R1 / R2 files are missing or unreadable, no merged result generated.")


def main() -> None:
    """
    Main function: Configures logging and executes the merging process for each database's R1 / R2 results.

    :return: None
    """
    logger = setup_logger(verbose=True)
    logger.info("[Main] Stage 1 database R1/R2 merging process started.")

    merger = DatabaseR1R2ScreenedMerger(logger=logger)
    merger.run()

    logger.info("[Main] Stage 1 database R1/R2 merging process completed.")


if __name__ == "__main__":
    main()