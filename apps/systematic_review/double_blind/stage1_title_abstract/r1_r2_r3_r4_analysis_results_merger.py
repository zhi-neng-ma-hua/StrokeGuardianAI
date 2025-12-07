# -*- coding: utf-8 -*-
"""
r1_r2_r3_r4_analysis_results_merger.py

Overview
--------
Merge the independent R4 screening results into the Stage 1
title/abstract screening summary.

Source files
------------
Under: project_root / data / systematic_review / double_blind / stage1_title_abstract

- R1_R2_R3_analysis_results.xlsx  (base table: R1 / R2 / R3)
- R4_analysis_results.xlsx        (R4-only table: contains ["Decision", "Notes"])

Merge key (row identity)
------------------------
- ["No.", "Article Title", "Publication Year", "DOI"]

Output
------
- R1_R2_R3_R4_analysis_results.xlsx
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from utils.logger_manager import LoggerManager
from utils.data_io import load_table, save_table
from utils.exceptions import DataValidationError

# Key fields for normalisation, identity checks and alignment
KEY_FIELDS: Tuple[str, ...] = ("No.", "Article Title", "Publication Year", "DOI")

# Stage 1 filenames
BASE_FILENAME = "R1_R2_R3_analysis_results.xlsx"      # Base table (R1 / R2 / R3)
R4_FILENAME = "R4_analysis_results.xlsx"              # R4-only table (Decision / Notes)
OUTPUT_FILENAME = "R1_R2_R3_R4_analysis_results.xlsx" # Merged R1 / R2 / R3 / R4 table


def setup_logger(
    name: str = "r1_r2_r3_r4_analysis_results_merger",
    verbose: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger for this module.

    :param name: Logger name.
    :param verbose: If True, enable DEBUG-level logging.
    :return: Configured logging.Logger instance.
    """
    return LoggerManager.setup_logger(
        logger_name=name,
        module_name=__name__,
        verbose=verbose,
    )


class R1R2R3R4AnalysisResultsMerger:
    """
    Merge controller for integrating R4 decisions into the Stage 1 R1/R2/R3 summary.

    Responsibilities:
    - Resolve the Stage 1 directory.
    - Load base (R1_R2_R3) and R4 results tables.
    - Normalise key fields and check key uniqueness / set equality.
    - Align R4 rows to base row order via KEY_FIELDS.
    - Add Decision_R4 / Notes_R4 to the base table.
    - Write merged results to disk.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialise the merger and resolve the Stage 1 data directory.

        :param logger: Optional logger; if None, a default logger is created.
        :raises DataValidationError: If the Stage 1 directory does not exist.
        """
        self.logger = logger or setup_logger()

        # Expected script location:
        # apps/systematic_review/double_blind/stage1_title_abstract/r1_r2_r3_r4_analysis_results_merger.py
        # parents[0] = stage1_title_abstract
        # parents[1] = double_blind
        # parents[2] = systematic_review
        # parents[3] = data
        # parents[4] = project_root
        project_root = Path(__file__).resolve().parents[4]
        systematic_review_root = project_root / "data" / "systematic_review"
        self.stage_root = systematic_review_root / "double_blind" / "stage1_title_abstract"

        if not self.stage_root.is_dir():
            raise DataValidationError(f"Stage 1 directory does not exist: {self.stage_root}")

        self.logger.info("[PATH] Stage 1 directory: %s", self.stage_root)

    # ------------------------------------------------------------------ #
    # Generic utilities
    # ------------------------------------------------------------------ #

    def _load_file(self, filename: str) -> pd.DataFrame:
        """
        Load a file from the Stage 1 directory with logging and error wrapping.

        :param filename: Filename relative to the Stage 1 directory.
        :return: Loaded DataFrame.
        :raises DataValidationError: If the file does not exist or cannot be read.
        """
        path = self.stage_root / filename
        if not path.is_file():
            raise DataValidationError(f"Expected file does not exist: {path}")

        try:
            df = load_table(path, logger=self.logger)
            self.logger.info(
                "[LOAD] %s | rows=%d | columns=%s",
                filename,
                len(df),
                list(df.columns),
            )
            return df
        except Exception as exc:
            raise DataValidationError(f"Error loading file {path}: {exc}") from exc

    @staticmethod
    def _require_columns(df: pd.DataFrame, filename: str, required: List[str]) -> None:
        """
        Ensure that a DataFrame contains the required columns.

        :param df: DataFrame to check.
        :param filename: Logical filename (for error messages).
        :param required: List of required column names.
        :raises DataValidationError: If any required column is missing.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"{filename} is missing required columns: {missing}")

    # ------------------------------------------------------------------ #
    # Key normalisation and MultiIndex construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise key fields for robust alignment and comparison.

        Rules:
        - "Article Title" / "DOI":
            * cast to string
            * strip leading/trailing whitespace
            * convert to lowercase
        - "Publication Year":
            * convert to numeric (invalid values → NaN)
            * store as Int64
        - All KEY_FIELDS:
            * cast to string
            * strip leading/trailing whitespace

        :param df: DataFrame to normalise.
        :return: Normalised copy of the DataFrame.
        """
        df = df.copy()

        if "Article Title" in df.columns:
            df["Article Title"] = (
                df["Article Title"].astype(str).str.strip().str.lower()
            )

        if "DOI" in df.columns:
            df["DOI"] = df["DOI"].astype(str).str.strip().str.lower()

        if "Publication Year" in df.columns:
            df["Publication Year"] = pd.to_numeric(
                df["Publication Year"],
                errors="coerce",
            ).astype("Int64")

        for col in KEY_FIELDS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    @staticmethod
    def _build_key_index(df: pd.DataFrame) -> pd.MultiIndex:
        """
        Build a MultiIndex from KEY_FIELDS for alignment / comparison.

        :param df: DataFrame containing KEY_FIELDS.
        :return: MultiIndex constructed from KEY_FIELDS.
        :raises DataValidationError: If any key column is missing.
        """
        missing = [c for c in KEY_FIELDS if c not in df.columns]
        if missing:
            raise DataValidationError(f"DataFrame is missing key columns: {missing}")
        return pd.MultiIndex.from_frame(df[list(KEY_FIELDS)])

    @staticmethod
    def _ensure_unique_keys(index: pd.MultiIndex, label: str) -> None:
        """
        Ensure that a MultiIndex has no duplicated key combinations.

        :param index: MultiIndex built from KEY_FIELDS.
        :param label: Label identifying the source (for error reporting).
        :raises DataValidationError: If duplicate key rows are found.
        """
        dup_mask = index.duplicated(keep=False)
        if dup_mask.any():
            duplicated_keys = list(index[dup_mask])
            sample = duplicated_keys[:5]
            raise DataValidationError(
                f"{label} contains duplicate key rows (total={len(duplicated_keys)}). "
                f"Sample duplicates: {sample}"
            )

    # ------------------------------------------------------------------ #
    # Key-set consistency check and R4 alignment
    # ------------------------------------------------------------------ #

    def _validate_keys_and_align(
        self,
        df_base_raw: pd.DataFrame,
        df_r4_raw: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalise keys, validate key-set equality, and align R4 rows to the base key order.

        Returns:
        - df_base_norm  : Normalised copy of the base table (for logging / debugging).
        - df_r4_aligned : Copy of the R4 table aligned to the base key order
                          (original columns preserved, RangeIndex).

        :param df_base_raw: Raw base DataFrame (R1_R2_R3_analysis_results).
        :param df_r4_raw: Raw R4 DataFrame (R4_analysis_results).
        :return: (df_base_norm, df_r4_aligned)
        :raises DataValidationError: If key sets differ or keys are not unique.
        """
        # 1. Normalise key fields
        df_base_norm = self._normalize_keys(df_base_raw)
        df_r4_norm = self._normalize_keys(df_r4_raw)

        # 2. Build MultiIndex keys and check uniqueness
        idx_base = self._build_key_index(df_base_norm)
        idx_r4 = self._build_key_index(df_r4_norm)

        self._ensure_unique_keys(idx_base, "R1_R2_R3_analysis_results")
        self._ensure_unique_keys(idx_r4, "R4_analysis_results")

        # 3. Compare key sets
        keys_base = set(idx_base)
        keys_r4 = set(idx_r4)

        if keys_base != keys_r4:
            only_in_base = keys_base - keys_r4
            only_in_r4 = keys_r4 - keys_base

            self.logger.error(
                "[KEYS] Key sets differ between R1_R2_R3_analysis_results and R4_analysis_results.\n"
                "  only_in_base : %d\n"
                "  only_in_R4   : %d",
                len(only_in_base),
                len(only_in_r4),
            )
            if only_in_base:
                self.logger.error("  Sample keys only in base (up to 5): %s", list(only_in_base)[:5])
            if only_in_r4:
                self.logger.error("  Sample keys only in R4 (up to 5): %s", list(only_in_r4)[:5])
            raise DataValidationError("Key sets of base and R4 files are not identical after normalisation.")

        self.logger.info("[KEYS] Base and R4 files have identical key sets after normalisation.")

        # 4. Align R4 table to base key order
        df_r4_indexed = df_r4_raw.copy()
        df_r4_indexed.index = idx_r4
        df_r4_aligned = df_r4_indexed.reindex(idx_base)

        # 5. Safety check: warn if any row becomes all-NaN after reindexing
        if df_r4_aligned.isna().all(axis=1).any():
            n_all_nan = int(df_r4_aligned.isna().all(axis=1).sum())
            self.logger.warning(
                "[ALIGN] %d rows in aligned R4 table are all-NaN after reindex; "
                "please verify key normalisation and input files.",
                n_all_nan,
            )

        # 6. Reset to RangeIndex for position-based assignment
        df_r4_aligned = df_r4_aligned.reset_index(drop=True)

        return df_base_norm, df_r4_aligned

    # ------------------------------------------------------------------ #
    # Merge R4 Decision / Notes into base table
    # ------------------------------------------------------------------ #

    def _merge_r4_into_base(self, df_base_raw: pd.DataFrame, df_r4_aligned: pd.DataFrame) -> pd.DataFrame:
        """
        Merge aligned R4 ["Decision", "Notes"] columns into the base table as
        ["Decision_R4", "Notes_R4"].

        Notes:
        - Only Decision_R4 / Notes_R4 are added or overwritten; other columns remain unchanged.
        - R4 table must contain "Decision" and "Notes" columns.
        - Base and aligned R4 tables must have identical row counts.

        :param df_base_raw: Original base DataFrame (non-normalised).
        :param df_r4_aligned: R4 DataFrame aligned to base row order.
        :return: New DataFrame with Decision_R4 / Notes_R4 merged in.
        :raises DataValidationError: If R4 columns are missing or row counts mismatch.
        """
        # Ensure required columns in R4 table
        self._require_columns(df_r4_aligned, R4_FILENAME, ["Decision", "Notes"])

        df_out = df_base_raw.copy()

        # Row counts must match for position-based assignment
        if len(df_out) != len(df_r4_aligned):
            raise DataValidationError(
                "Row count mismatch between base and aligned R4 table "
                f"(base={len(df_out)}, R4_aligned={len(df_r4_aligned)})."
            )

        # Position-based assignment to avoid index alignment issues
        df_out["Decision_R4"] = df_r4_aligned["Decision"].to_numpy()
        df_out["Notes_R4"] = df_r4_aligned["Notes"].to_numpy()

        self.logger.info("[MERGE] Decision_R4 / Notes_R4 have been merged from R4_analysis_results.xlsx.")
        return df_out

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Execute the R1_R2_R3 + R4 merge workflow.

        Steps:
        1. Load R1_R2_R3_analysis_results.xlsx and R4_analysis_results.xlsx.
        2. Normalise keys and validate key-set equality / uniqueness.
        3. Align R4 table to the base row order via KEY_FIELDS.
        4. Merge R4.Decision / R4.Notes into the base as Decision_R4 / Notes_R4.
        5. Save merged results as R1_R2_R3_R4_analysis_results.xlsx.
        """
        # 1. Load source tables
        df_base_raw = self._load_file(BASE_FILENAME)
        df_r4_raw = self._load_file(R4_FILENAME)

        # 2. Key normalisation + consistency check + alignment
        _, df_r4_aligned = self._validate_keys_and_align(df_base_raw, df_r4_raw)

        # 3. Merge R4 decisions/notes
        df_merged = self._merge_r4_into_base(df_base_raw, df_r4_aligned)

        # 4. Save merged results (do not overwrite the original base table)
        output_path = self.stage_root / OUTPUT_FILENAME
        try:
            save_table(df_merged, output_path, logger=self.logger)
            self.logger.info(
                "[SAVE] R1/R2/R3/R4 merged analysis results saved to: %s",
                output_path,
            )
        except Exception as exc:
            raise DataValidationError(f"Error saving merged results to {output_path}: {exc}")


def main() -> None:
    """
    Script entry point: run the R1_R2_R3 + R4 merge workflow.
    """
    logger = setup_logger(verbose=True)
    merger = R1R2R3R4AnalysisResultsMerger(logger=logger)
    merger.run()


if __name__ == "__main__":
    main()