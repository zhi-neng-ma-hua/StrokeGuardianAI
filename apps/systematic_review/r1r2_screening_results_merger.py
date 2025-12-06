# -*- coding: utf-8 -*-
"""
r1r2_screening_results_merger.py

This script merges standardized bibliographic data (`standardized_papers.xlsx`) with
R1 and R2 screening results from the Stage 1 (Title/Abstract) double-blind review.

Main features:
1. Load and standardize the data from standardized papers and the R1/R2 screening results.
2. Perform de-duplication and left merge based on key fields (Article Title, Publication Year, DOI).
3. Generate R1/R2 summary results with the same row order as `standardized_papers.xlsx`.
4. Perform missing value statistics, JSON formatting of the Notes field, and handle NaN values.
5. Check if the merged results match the standardized papers using ["No.", "Article Title", "Publication Year", "DOI"].

Author: Aiden Cao <zhinengmahua@gmail.com>
Date: 2025-05-22
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

from utils.logger_manager import LoggerManager
from utils.exceptions import DataValidationError
from utils.data_io import load_table, save_table

# Key fields for merging and consistency check
KEY_FIELDS: List[str] = ["Article Title", "Publication Year", "DOI"]

# Column order for the final merged dataset
COLUMN_ORDER: List[str] = [
    "No.", "Article Title", "Publication Year", "DOI", "Source Title", "Times Cited",
    "Record Link", "Abstract", "Author Keywords", "Index Keywords", "Mesh Terms",
    "Document Type", "ISSN", "Unique ID", "Decision", "Notes",
]

# Paths for R1, R2, and standardized data (relative to `stage_root / standardized_root`)
R1_FILE_PATHS: List[str] = [
    "R1/screened_scopus_papers_R1.xlsx",
    "R1/screened_web_of_science_papers_R1.xlsx",
    "R1/screened_ieee_papers_R1.xlsx",
    "R1/screened_pubmed_papers_R1.xlsx",
]

R2_FILE_PATHS: List[str] = [
    "R2/screened_scopus_papers_R2.xlsx",
    "R2/screened_web_of_science_papers_R2.xlsx",
    "R2/screened_ieee_papers_R2.xlsx",
    "R2/screened_pubmed_papers_R2.xlsx",
]

STANDARDIZED_FILE_PATH: str = "standardized_papers.xlsx"


def setup_logger(name: str = "r1r2_screening_results_merger", verbose: bool = True) -> logging.Logger:
    """
    Setup and return the logger instance.

    :param name: Logger name.
    :param verbose: Whether to enable DEBUG level logs.
    :return: Configured logging.Logger instance.
    """
    return LoggerManager.setup_logger(
        logger_name=name,
        module_name=__name__,
        verbose=verbose,
    )


class R1R2ScreeningMerger:
    """
    Controller for merging R1/R2 screening results and performing consistency checks.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the controller, configure logger, and setup directories.

        :param logger: Optional logger instance. If None, default logger will be used.
        """
        self.logger = logger or setup_logger()
        self._init_dirs()

    def _init_dirs(self) -> None:
        """
        Initialize and verify the existence of necessary directories.

        :raises DataValidationError: If directories do not exist.
        """
        project_root = Path(__file__).resolve().parents[4]
        systematic_review_root = project_root / "data" / "systematic_review"

        self.stage_root = systematic_review_root / "double_blind" / "stage1_title_abstract"
        self.standardized_root = systematic_review_root / "standardized"

        if not self.stage_root.is_dir():
            raise DataValidationError(f"Stage directory does not exist: {self.stage_root}")
        if not self.standardized_root.is_dir():
            raise DataValidationError(f"Standardized directory does not exist: {self.standardized_root}")

        self.logger.info(f"[PATH] Stage directory: {self.stage_root}")
        self.logger.info(f"[PATH] Standardized directory: {self.standardized_root}")

    def _normalize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize key fields ("Article Title", "Publication Year", "DOI") for easy merging.

        Standardization rules:
        - 'Article Title' and 'DOI' are converted to lowercase and trimmed.
        - 'Publication Year' is converted to numeric, invalid values become NaN.
        - KEY_FIELDS are standardized by trimming spaces.

        :param df: DataFrame to normalize.
        :return: Normalized DataFrame.
        """
        df = df.copy()

        if "Article Title" in df.columns:
            df["Article Title"] = df["Article Title"].str.strip().str.lower()

        if "DOI" in df.columns:
            df["DOI"] = df["DOI"].str.strip().str.lower()

        if "Publication Year" in df.columns:
            df["Publication Year"] = pd.to_numeric(df["Publication Year"], errors="coerce").astype("Int64")

        for col in KEY_FIELDS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    def _load_and_merge_data(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load and merge all data files for R1 or R2.

        :param file_paths: List of file paths (relative to `stage_root`).
        :return: Merged DataFrame.
        """
        data_frames: List[pd.DataFrame] = []
        total_rows = 0

        for file_path in file_paths:
            path = self.stage_root / file_path
            df = load_table(path, logger=self.logger)
            data_frames.append(df)
            total_rows += len(df)

        merged_df = pd.concat(data_frames, ignore_index=True)
        self.logger.info(f"[LOAD] Merged {len(file_paths)} files, {total_rows} rows → {len(merged_df)} rows.")
        return merged_df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows based on the key fields.

        :param df: DataFrame to de-duplicate (should already be normalized).
        :return: DataFrame with duplicates removed.
        """
        before = len(df)
        df_dedup = df.drop_duplicates(subset=KEY_FIELDS, keep="first")
        removed = before - len(df_dedup)

        self.logger.info(f"[DEDUP] Removed {removed} duplicate rows, remaining {len(df_dedup)} rows.")
        return df_dedup

    def _merge_data(self, df_standardized: pd.DataFrame, df_r1: pd.DataFrame, df_r2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge standardized data with R1 and R2 data using left joins.

        The standardized data is used as the base, and the R1/R2 columns (e.g., Unique ID, Decision, Notes)
        are added as suffixes.

        :param df_standardized: Standardized data (with added "No." column).
        :param df_r1: R1 data (normalized and de-duplicated).
        :param df_r2: R2 data (normalized and de-duplicated).
        :return: Merged R1 and R2 DataFrames.
        """
        merged_r1 = pd.merge(df_standardized, df_r1, on=KEY_FIELDS, how="left", suffixes=("", "_r1"))
        merged_r2 = pd.merge(df_standardized, df_r2, on=KEY_FIELDS, how="left", suffixes=("", "_r2"))

        missing_r1 = merged_r1.isna().sum()
        missing_r2 = merged_r2.isna().sum()
        self.logger.info(f"[MISSING] Missing values in R1 merge: \n{missing_r1}")
        self.logger.info(f"[MISSING] Missing values in R2 merge: \n{missing_r2}")

        return merged_r1, merged_r2

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder the columns based on predefined COLUMN_ORDER.

        :param df: DataFrame to reorder.
        :return: Reordered DataFrame.
        """
        return df[COLUMN_ORDER].copy()

    def _save_results(self, df: pd.DataFrame, round_label: str) -> Path:
        """
        Save the merged results to an Excel file.

        :param df: Final DataFrame to save.
        :param round_label: Label for the round (e.g., "R1" or "R2").
        :return: Path to the saved file.
        """
        file_name = f"{round_label}_analysis_results.xlsx"
        output_path = self.stage_root / file_name
        save_table(df, output_path, logger=self.logger)
        self.logger.info(f"[SAVE] {round_label} results saved to: {output_path}")
        return output_path

    def _convert_notes_to_json(self, notes: object) -> str:
        """
        Convert the Notes column content to formatted JSON (4-space indentation).

        :param notes: Notes field value.
        :return: JSON formatted string or original notes.
        """
        if notes is None or (isinstance(notes, float) and pd.isna(notes)):
            return ""
        if not isinstance(notes, str):
            notes = str(notes)

        text = notes.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=4, ensure_ascii=False)
        except Exception:
            return notes

    def _check_no_and_keys_alignment(self, df_std: pd.DataFrame, df_r1: pd.DataFrame, df_r2: pd.DataFrame) -> None:
        """
        Check if the rows in the standardized data, R1, and R2 match based on ["No.", "Article Title", "Publication Year", "DOI"].

        :param df_std: Standardized data (with No. column).
        :param df_r1: R1 results.
        :param df_r2: R2 results.
        """
        key_cols = ["No."] + KEY_FIELDS
        std_keys = df_std[key_cols]
        r1_keys = df_r1[key_cols]
        r2_keys = df_r2[key_cols]

        ok_r1 = std_keys.equals(r1_keys)
        ok_r2 = std_keys.equals(r2_keys)

        if ok_r1 and ok_r2:
            self.logger.info("[CHECK] Standardized matches R1/R2 on ['No.', 'Article Title', 'Publication Year', 'DOI'].")
        else:
            if not ok_r1:
                self.logger.warning("[CHECK] Standardized and R1 mismatch on ['No.', 'Article Title', 'Publication Year', 'DOI'].")
            if not ok_r2:
                self.logger.warning("[CHECK] Standardized and R2 mismatch on ['No.', 'Article Title', 'Publication Year', 'DOI'].")

    def run(self) -> None:
        """
        Run the full Stage 1 process: data merging, consistency checking, and final result saving.
        """
        standardized_path = self.standardized_root / STANDARDIZED_FILE_PATH
        if not standardized_path.is_file():
            raise DataValidationError(f"Standardized document file does not exist: {standardized_path}")

        df_standardized = load_table(standardized_path, logger=self.logger)
        self.logger.info(f"[LOAD] Loaded standardized_papers.xlsx with {len(df_standardized)} rows.")

        # Add "No." column to the standardized data to preserve the row order
        df_standardized.insert(0, "No.", range(1, len(df_standardized) + 1))
        self.logger.info(f"[INFO] Added 'No.' column to standardized data, total rows: {len(df_standardized)}")

        # Load and merge R1/R2 data
        df_r1_raw = self._load_and_merge_data(R1_FILE_PATHS)
        df_r2_raw = self._load_and_merge_data(R2_FILE_PATHS)

        # Normalize and remove duplicates
        df_r1_norm = self._normalize_keys(df_r1_raw)
        df_r2_norm = self._normalize_keys(df_r2_raw)

        df_r1 = self._remove_duplicates(df_r1_norm)
        df_r2 = self._remove_duplicates(df_r2_norm)

        # Merge the data
        merged_r1, merged_r2 = self._merge_data(df_standardized, df_r1, df_r2)

        # Reorder columns and save
        merged_r1 = self._reorder_columns(merged_r1)
        merged_r2 = self._reorder_columns(merged_r2)

        # Missing value statistics
        self.logger.info(f"[MISSING] R1 final missing values: \n{merged_r1.isna().sum()}")
        self.logger.info(f"[MISSING] R2 final missing values: \n{merged_r2.isna().sum()}")

        # Handle Notes field and fill NaN with empty strings
        merged_r1["Notes"] = merged_r1["Notes"].apply(self._convert_notes_to_json)
        merged_r2["Notes"] = merged_r2["Notes"].apply(self._convert_notes_to_json)

        merged_r1 = merged_r1.fillna("")
        merged_r2 = merged_r2.fillna("")

        # Check consistency
        self._check_no_and_keys_alignment(df_standardized, merged_r1, merged_r2)

        # Save the results
        self._save_results(merged_r1, "R1")
        self._save_results(merged_r2, "R2")


def main() -> None:
    """
    Main entry point to start Stage 1 data merging and consistency analysis.
    """
    logger = setup_logger(verbose=True)
    logger.info("[MAIN] Stage 1 data merging and consistency analysis started.")

    merger = R1R2ScreeningMerger(logger=logger)
    merger.run()

    logger.info("[MAIN] Stage 1 data merging and consistency analysis completed.")


if __name__ == "__main__":
    main()