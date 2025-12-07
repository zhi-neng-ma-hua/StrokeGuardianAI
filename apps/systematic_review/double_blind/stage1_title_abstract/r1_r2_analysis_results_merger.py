# -*- coding: utf-8 -*-
"""
r1_r2_analysis_results_merger.py

Merge Stage 1 (Title / Abstract) R1 and R2 screening results on a shared key space:

    - "No."
    - "Article Title"
    - "Publication Year"
    - "DOI"

Design:
- Treat R1_analysis_results.xlsx as the base table (all non-decision columns come from R1).
- Attach only the Decision / Notes columns from R2_analysis_results.xlsx.
- Normalize key fields in both R1 / R2 before any comparison.
- Validate that R1 and R2 share an identical key set after normalization.
- Produce:

      Decision_R1, Notes_R1  (from R1_analysis_results.xlsx)
      Decision_R2, Notes_R2  (from R2_analysis_results.xlsx)
      Need_R3                (derived flag for R3 adjudication)

Need_R3 rules:
- "Yes" if Decision_R1 = Decision_R2 = "unsure"
- "Yes" if Decision_R1 != Decision_R2
- "No" otherwise

Before writing the final file, Notes_R1 and Notes_R2 are formatted as
pretty JSON (4-space indentation) where valid JSON strings are detected.

Output:
- R1_R2_analysis_results.xlsx in the Stage 1 directory.

Author: Aiden Cao <zhinengmahua@gmail.com>
Date  : 2025-12-06
"""

import logging
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from utils.logger_manager import LoggerManager
from utils.data_io import load_table, save_table
from utils.exceptions import DataValidationError

# File names (relative to Stage 1 root)
R1_ANALYSIS_FILE = "R1_analysis_results.xlsx"
R2_ANALYSIS_FILE = "R2_analysis_results.xlsx"
OUTPUT_FILE = "R1_R2_analysis_results.xlsx"

# Key fields for alignment / consistency checks
KEY_FIELDS: List[str] = ["No.", "Article Title", "Publication Year", "DOI"]
KeyTuple = Tuple[str, ...]


def setup_logger(name: str = "r1r2_analysis_results_merger", verbose: bool = True) -> logging.Logger:
    """
    Configure and return a logger instance.

    :param name: Logger name.
    :param verbose: Whether to enable DEBUG-level logs.
    :return: Configured logging.Logger instance.
    """
    return LoggerManager.setup_logger(
        logger_name=name,
        module_name=__name__,
        verbose=verbose,
    )


class R1R2AnalysisMerger:
    """
    Merge R1 / R2 analysis results and derive Need_R3 on a common key space.

    Responsibilities:
    - Resolve Stage 1 directory.
    - Load raw R1 / R2 analysis tables.
    - Normalize and validate key sets.
    - Prepare R1 base and R2 decision slices.
    - Merge R1+R2 decisions and compute Need_R3.
    - Normalize Notes_R1 / Notes_R2 to pretty JSON where applicable.
    - Save the final merged results.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the merger and resolve the Stage 1 directory.

        :param logger: Optional logger instance; if None, a default logger is created.
        :raises DataValidationError: If the Stage 1 directory does not exist.
        """
        self.logger = logger or setup_logger()

        project_root = Path(__file__).resolve().parents[4]
        self.stage_root = project_root / "data" / "systematic_review" / "double_blind" / "stage1_title_abstract"

        if not self.stage_root.is_dir():
            raise DataValidationError(f"Stage 1 directory does not exist: {self.stage_root}")

        self.logger.info(f"[PATH] Stage 1 directory: {self.stage_root}")

    # -------------------------------------------------------------------------
    # Generic utilities
    # -------------------------------------------------------------------------

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a table from disk and wrap low-level exceptions in DataValidationError.

        :param file_path: Path to the file.
        :return: Loaded DataFrame.
        :raises DataValidationError: If the file cannot be loaded.
        """
        try:
            return load_table(file_path, logger=self.logger)
        except Exception as exc:
            raise DataValidationError(f"Error loading file {file_path}: {exc}")

    @staticmethod
    def _require_columns(df: pd.DataFrame, filename: str, required: List[str]) -> None:
        """
        Ensure that a DataFrame contains the required columns.

        :param df: DataFrame to check.
        :param filename: Name of the source file (for error messages).
        :param required: List of required column names.
        :raises DataValidationError: If any required columns are missing.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"{filename} is missing required columns: {missing}")

    @staticmethod
    def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize key fields ("Article Title", "Publication Year", "DOI")
        for robust comparison / merging.

        Standardization rules:
        - 'Article Title' and 'DOI':
            * cast to string
            * strip leading/trailing whitespace
            * convert to lowercase
        - 'Publication Year':
            * convert to numeric (invalid values become NaN)
            * store as Int64
        - All KEY_FIELDS:
            * cast to string
            * strip leading/trailing whitespace

        :param df: DataFrame to normalize.
        :return: Normalized DataFrame (copy).
        """
        df = df.copy()

        if "Article Title" in df.columns:
            df["Article Title"] = df["Article Title"].astype(str).str.strip().str.lower()

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
    def _build_key_tuples(df: pd.DataFrame) -> List[KeyTuple]:
        """
        Build a list of key tuples from KEY_FIELDS for each row.

        :param df: DataFrame containing KEY_FIELDS.
        :return: List of key tuples (one per row).
        """
        return [tuple(row) for row in df[KEY_FIELDS].to_numpy()]

    # -------------------------------------------------------------------------
    # Key consistency validation
    # -------------------------------------------------------------------------

    def _validate_key_consistency(self, df_r1_raw: pd.DataFrame, df_r2_raw: pd.DataFrame) -> None:
        """
        Validate that R1 and R2 analysis results have identical key sets (after normalization).

        Steps:
        1. Normalize key fields in both R1 / R2 DataFrames.
        2. Ensure that all KEY_FIELDS exist.
        3. Build key sets and compare:
           - Report row counts and unique key counts.
           - If key sets differ, log samples and raise DataValidationError.

        :param df_r1_raw: Raw R1 analysis DataFrame.
        :param df_r2_raw: Raw R2 analysis DataFrame.
        :raises DataValidationError: If key sets are not exactly identical.
        """
        df_r1_norm = self._normalize_keys(df_r1_raw)
        df_r2_norm = self._normalize_keys(df_r2_raw)

        for df, name in ((df_r1_norm, R1_ANALYSIS_FILE), (df_r2_norm, R2_ANALYSIS_FILE)):
            missing = [c for c in KEY_FIELDS if c not in df.columns]
            if missing:
                raise DataValidationError(f"{name} is missing key columns: {missing}")

        keys_r1 = set(self._build_key_tuples(df_r1_norm))
        keys_r2 = set(self._build_key_tuples(df_r2_norm))

        # Basic statistics
        self.logger.info("[KEYS] R1/R2 key consistency check")
        self.logger.info(f"  R1: rows={len(df_r1_raw)}, unique keys={len(keys_r1)}")
        self.logger.info(f"  R2: rows={len(df_r2_raw)}, unique keys={len(keys_r2)}")

        only_in_r1 = keys_r1 - keys_r2
        only_in_r2 = keys_r2 - keys_r1

        if only_in_r1 or only_in_r2:
            self.logger.error(
                f"[KEYS] Key sets differ: only_in_R1={len(only_in_r1)}, only_in_R2={len(only_in_r2)}"
            )
            if only_in_r1:
                sample_r1 = list(only_in_r1)[:5]
                self.logger.error(f"  Sample keys only in R1 (up to 5): {sample_r1}")
            if only_in_r2:
                sample_r2 = list(only_in_r2)[:5]
                self.logger.error(f"  Sample keys only in R2 (up to 5): {sample_r2}")
            raise DataValidationError("R1 and R2 analysis results do not share identical key sets.")
        else:
            self.logger.info("[KEYS] R1 and R2 have identical key sets (after normalization).")

    # -------------------------------------------------------------------------
    # Preparation of R1 base and R2 decision slices
    # -------------------------------------------------------------------------

    def _prepare_r1_base(self, df_r1_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare R1 base DataFrame from the raw R1 analysis results.

        This method:
        - Verifies presence of KEY_FIELDS + ['Decision', 'Notes'].
        - Renames 'Decision' → 'Decision_R1', 'Notes' → 'Notes_R1'.

        :param df_r1_raw: Raw R1 analysis DataFrame.
        :return: Prepared R1 base DataFrame.
        """
        self._require_columns(df_r1_raw, R1_ANALYSIS_FILE, KEY_FIELDS + ["Decision", "Notes"])

        df_r1 = df_r1_raw.copy()
        df_r1.rename(
            columns={
                "Decision": "Decision_R1",
                "Notes": "Notes_R1",
            },
            inplace=True,
        )

        self.logger.info(
            f"[R1] Prepared base table from {R1_ANALYSIS_FILE} with {len(df_r1)} rows"
        )
        return df_r1

    def _prepare_r2_decisions(self, df_r2_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare R2 decision slice from the raw R2 analysis results.

        This method:
        - Verifies presence of KEY_FIELDS + ['Decision', 'Notes'].
        - Returns a slim DataFrame with:
          KEY_FIELDS + ['Decision_R2', 'Notes_R2'].

        :param df_r2_raw: Raw R2 analysis DataFrame.
        :return: R2 decision/notes DataFrame.
        """
        self._require_columns(df_r2_raw, R2_ANALYSIS_FILE, KEY_FIELDS + ["Decision", "Notes"])

        df_r2 = df_r2_raw.copy()
        df_r2 = df_r2[KEY_FIELDS + ["Decision", "Notes"]]
        df_r2.rename(
            columns={
                "Decision": "Decision_R2",
                "Notes": "Notes_R2",
            },
            inplace=True,
        )

        self.logger.info(
            f"[R2] Prepared decision slice from {R2_ANALYSIS_FILE} with {len(df_r2)} rows"
        )
        return df_r2

    # -------------------------------------------------------------------------
    # Merge + Need_R3 computation
    # -------------------------------------------------------------------------

    def _merge_r1_r2(self, df_r1: pd.DataFrame, df_r2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge prepared R1 base with prepared R2 decisions on KEY_FIELDS.

        R1 is used as the left/base DataFrame. The result includes:
        - All columns from R1 (with Decision_R1 / Notes_R1).
        - Decision_R2 / Notes_R2 appended from R2 where keys match.

        :param df_r1: Prepared R1 base DataFrame.
        :param df_r2: Prepared R2 decisions DataFrame.
        :return: Merged DataFrame.
        """
        merged = pd.merge(
            df_r1,
            df_r2,
            on=KEY_FIELDS,
            how="left",
        )

        if len(merged) != len(df_r1):
            self.logger.warning(
                f"[MERGE] Row count changed after merging R2 (R1={len(df_r1)}, merged={len(merged)}). "
                f"Check for duplicate or missing keys in R2."
            )
        else:
            self.logger.info(f"[MERGE] R1 + R2 merged with {len(merged)} rows (row count unchanged).")

        return merged

    def _set_need_r3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Need_R3 flag based on Decision_R1 / Decision_R2.

        Rules:
        - Default 'Need_R3' = "No"
        - Set to "Yes" if:
          * Decision_R1 == Decision_R2 == "unsure", OR
          * Decision_R1 != Decision_R2

        :param df: DataFrame with Decision_R1 and Decision_R2 columns.
        :return: DataFrame with Need_R3 column added.
        """
        df = df.copy()

        for col in ("Decision_R1", "Decision_R2"):
            if col not in df.columns:
                raise DataValidationError(f"Expected column '{col}' is missing for Need_R3 computation.")

        df["Need_R3"] = "No"
        unsure_both = (df["Decision_R1"] == "unsure") & (df["Decision_R2"] == "unsure")
        diff_decision = df["Decision_R1"] != df["Decision_R2"]

        df.loc[unsure_both | diff_decision, "Need_R3"] = "Yes"

        self._log_need_r3(df)
        return df

    def _log_need_r3(self, df: pd.DataFrame) -> None:
        """
        Log summary statistics for the Need_R3 column.

        Logs:
        - Value counts of Need_R3.
        - List of 'No.' values where Need_R3 == "Yes".

        :param df: DataFrame with the 'Need_R3' column.
        """
        counts = df["Need_R3"].value_counts()
        self.logger.info(f"[INFO] 'Need_R3' value counts:\n{counts}")

        yes_rows = df[df["Need_R3"] == "Yes"]
        self.logger.info(
            f"[INFO] 'Need_R3' = 'Yes' for the following No. values:\n{yes_rows['No.'].tolist()}"
        )

    # -------------------------------------------------------------------------
    # Notes JSON formatting
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_notes_cell(val: object) -> str:
        """
        Format a single Notes cell as pretty JSON (4-space indentation) if possible.

        Rules:
        - If val is None or NaN → return empty string.
        - If val is a str:
            * strip it; if empty → return empty string.
            * try json.loads; on success → json.dumps(..., indent=4, ensure_ascii=False).
            * on failure → return original string (unchanged).
        - For non-str (e.g., dict/list):
            * try json.dumps(..., indent=4, ensure_ascii=False).
            * on failure → cast to str and return.

        :param val: Original Notes cell value.
        :return: Formatted string.
        """
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""

        # Try to keep original text if not JSON
        if isinstance(val, str):
            text = val.strip()
            if not text:
                return ""
            try:
                parsed = json.loads(text)
                return json.dumps(parsed, ensure_ascii=False, indent=4)
            except Exception:
                # Not valid JSON: return original string
                return val

        # Non-str: try to serialize as JSON
        try:
            return json.dumps(val, ensure_ascii=False, indent=4)
        except Exception:
            return str(val)

    def _normalize_notes_columns(self, df: pd.DataFrame, note_columns: List[str]) -> pd.DataFrame:
        """
        Normalize given Notes columns to pretty JSON format where applicable.

        :param df: DataFrame containing the Notes columns.
        :param note_columns: List of column names to format (e.g., ['Notes_R1', 'Notes_R2']).
        :return: DataFrame with Notes columns formatted.
        """
        df = df.copy()
        for col in note_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._format_notes_cell)
            else:
                self.logger.warning(f"[WARN] Notes column '{col}' not found; JSON formatting skipped.")
        return df

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def _save_results(self, df: pd.DataFrame) -> None:
        """
        Save the final merged DataFrame to the Stage 1 directory.

        :param df: DataFrame to save.
        :raises DataValidationError: If the file cannot be saved.
        """
        try:
            output_path = self.stage_root / OUTPUT_FILE
            save_table(df, output_path, logger=self.logger)
            self.logger.info(f"[INFO] Merged R1/R2 analysis results saved to {output_path}")
        except Exception as exc:
            raise DataValidationError(f"Error saving merged results to {OUTPUT_FILE}: {exc}")

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the full R1/R2 analysis merge process:

        1. Load raw R1 and R2 analysis tables.
        2. Normalize keys and validate that R1 / R2 share identical key sets.
        3. Prepare R1 base and R2 decision slices from the raw tables.
        4. Merge them on KEY_FIELDS.
        5. Compute Need_R3 based on Decision_R1 / Decision_R2.
        6. Normalize Notes_R1 / Notes_R2 as pretty JSON where applicable.
        7. Save the final merged results.
        """
        # 1. Load raw tables
        r1_path = self.stage_root / R1_ANALYSIS_FILE
        r2_path = self.stage_root / R2_ANALYSIS_FILE

        df_r1_raw = self._load_file(r1_path)
        df_r2_raw = self._load_file(r2_path)

        # 2. Validate key consistency after normalization
        self._validate_key_consistency(df_r1_raw, df_r2_raw)

        # 3. Prepare R1 base and R2 decisions from raw tables
        df_r1 = self._prepare_r1_base(df_r1_raw)
        df_r2 = self._prepare_r2_decisions(df_r2_raw)

        # 4. Merge and compute Need_R3
        merged_df = self._merge_r1_r2(df_r1, df_r2)
        merged_df = self._set_need_r3(merged_df)

        # 6. Normalize Notes_R1 / Notes_R2 to pretty JSON (4-space indentation)
        merged_df = self._normalize_notes_columns(merged_df, ["Notes_R1", "Notes_R2"])

        # 7. Save
        self._save_results(merged_df)


def main() -> None:
    """
    Main entry point of the script.
    """
    logger = setup_logger(verbose=True)
    logger.info("[MAIN] Starting R1/R2 analysis results merge process")

    merger = R1R2AnalysisMerger(logger=logger)
    merger.run()

    logger.info("[MAIN] R1/R2 analysis results merge process completed")


if __name__ == "__main__":
    main()