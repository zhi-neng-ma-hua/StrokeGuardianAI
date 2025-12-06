# -*- coding: utf-8 -*-
"""
standardized_literature_merger.py

Batch merging and deduplication tool for standardized literature result files.

Main Features:
1. Vertically merges multiple standardized literature data files.
2. Removes duplicates based on specified fields (such as title, year, DOI).
3. Automatically fills in missing columns to ensure uniform output column structure.
4. Outputs a merged result file for further systematic review and bibliometric analysis.

Author: Aiden Cao <zhinengmahua@gmail.com>
Date: 2025-07-13
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from utils.exceptions import DataValidationError
from utils.data_io import load_table, save_table

# Path type alias
PathLike = Union[str, Path]

# Default columns to retain (order corresponds to the output column order)
DEFAULT_FIELDS_TO_KEEP: List[str] = [
    "Article Title",
    "Publication Year",
    "Source Title",
    "Times Cited",
    "DOI",
    "Record Link",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "Mesh Terms",
    "Document Type",
    "ISSN",
]

# Default deduplication key fields
DEFAULT_DEDUP_KEYS: List[str] = ["Article Title", "Publication Year", "DOI"]


def merge_and_deduplicate_std_files(
    std_file_paths: Sequence[PathLike],
    output_path: PathLike,
    fields_to_keep: Optional[Sequence[str]] = None,
    dedup_keys: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
    encoding: str = "utf-8-sig",
) -> None:
    """
    Batch merges and deduplicates multiple standardized literature files.

    :param std_file_paths: List of standardized file paths (supports str / Path)
    :param output_path: Output file path for the merged result (supports .xlsx / .csv, depending on save_table implementation)
    :param fields_to_keep: List of fields to retain; if None, use DEFAULT_FIELDS_TO_KEEP
    :param dedup_keys: List of key fields for deduplication; if None, use DEFAULT_DEDUP_KEYS
    :param logger: Logger; if None, no logging will occur
    :param encoding: Output file encoding, passed to save_table (usually "utf-8-sig")
    :return: None
    :raises DataValidationError: If no files are provided, no files are successfully read, or required deduplication fields are missing
    """
    # Normalize parameters
    fields_to_keep = list(fields_to_keep) if fields_to_keep else DEFAULT_FIELDS_TO_KEEP
    dedup_keys = list(dedup_keys) if dedup_keys else DEFAULT_DEDUP_KEYS

    if not std_file_paths:
        raise DataValidationError("No standardized file paths provided, unable to proceed with merging.")

    output_path = Path(output_path)
    data_frames: List[pd.DataFrame] = []
    failed_files: List[str] = []

    if logger:
        logger.info(f"[Merging] Starting to merge standardized files, target file count: {len(std_file_paths)}")

    # Read each file and ensure consistent columns
    for file_path in std_file_paths:
        file_path = Path(file_path)
        try:
            df = load_table(file_path, logger=logger)

            # Ensure missing columns are filled to match the structure
            for col in fields_to_keep:
                if col not in df.columns:
                    df[col] = ""

            # Trim to the unified column set and maintain the column order
            df = df[fields_to_keep]
            data_frames.append(df)

            if logger:
                logger.info(f"[Merging] Loaded file: {file_path} | Rows: {len(df)}")
        except Exception as exc:
            failed_files.append(str(file_path))
            if logger:
                logger.error(f"[Merging] Failed to read file: {file_path} | Error: {exc}")

    # If no files were successfully read, raise an error
    if not data_frames:
        raise DataValidationError("All standardized files failed to load, unable to proceed with merging.")

    if failed_files and logger:
        logger.warning(f"[Merging] {len(failed_files)} files failed to read, skipped: {failed_files}")

    # Merge all DataFrames
    merged_df = pd.concat(data_frames, ignore_index=True, copy=False)
    before_rows = len(merged_df)

    # Check if deduplication keys exist
    missing_keys = [key for key in dedup_keys if key not in merged_df.columns]
    if missing_keys:
        raise DataValidationError(f"Cannot deduplicate: Missing required key fields: {missing_keys}")

    # Deduplicate based on key fields
    merged_df = merged_df.drop_duplicates(subset=dedup_keys, keep="first")
    after_rows = len(merged_df)

    if logger:
        logger.info(
            f"[Deduplication] Rows before merging: {before_rows}, Rows after deduplication: {after_rows}, "
            f"Removed {before_rows - after_rows} duplicate records."
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"[Saving] Writing merged result to: {output_path}")

    # Save the merged result
    save_table(merged_df, output_path, logger=logger, encoding=encoding)

    if logger:
        logger.info(f"[Completed] Merging and deduplication complete. Output file: {output_path} | Final rows: {after_rows}")