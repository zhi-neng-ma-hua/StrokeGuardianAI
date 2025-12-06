# -*- coding: utf-8 -*-
"""
literature_column_standardizer.py

Literature multi-source citation data column name and field batch standardization tool.

Main Features:
1. Standardizes column names and fields of raw literature data from different sources (e.g., Scopus, Web of Science, IEEE Xplore, PubMed) based on field mapping configuration.
2. Cleans and deduplicates columns such as title, year, and DOI.
3. Merges the standardized results from each source into a unified summary table for subsequent systematic review and bibliometric analysis.

Author: Aiden Cao <zhinengmahua@gmail.com>
Date: 2025-05-22
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from utils.exceptions import DataValidationError, FieldMappingError
from apps.systematic_review.data_processing.standardized_literature_merger import (
    merge_and_deduplicate_std_files,
)
from utils.data_io import load_table, save_table
from utils.logger_manager import LoggerManager


def setup_logger(
    name: str = "literature_column_standardizer",
    verbose: bool = True,
) -> logging.Logger:
    """
    Configures and returns a logger.

    :param name: The name of the logger (used to distinguish between different modules)
    :param verbose: Whether to enable DEBUG level logging (True → DEBUG, False → INFO)
    :return: A configured logging.Logger instance
    :raises RuntimeError: If the logger initialization fails
    """
    try:
        return LoggerManager.setup_logger(
            logger_name=name,
            module_name=__name__,
            verbose=verbose,
        )
    except Exception as exc:
        raise RuntimeError(f"Logger initialization failed: {exc}") from exc


class LiteratureColumnStandardizer:
    """
    A class for standardizing literature column names and fields:

    - Parses directories.
    - Loads field mappings.
    - Executes file standardization for each source.
    - Merges standardized results.
    """

    # Raw source files: source identifier → raw Excel file name
    RAW_SOURCE_FILES: Dict[str, str] = {
        "scopus": "scopus_papers.xlsx",
        "wos": "web_of_science_papers.xlsx",
        "ieee": "ieee_xplore_papers.xlsx",
        "pubmed": "pubmed_papers.xlsx",
    }

    # Standardized files: source identifier → standardized Excel file name
    STD_SOURCE_FILES: Dict[str, str] = {
        "scopus": "std_scopus_papers.xlsx",
        "wos": "std_web_of_science_papers.xlsx",
        "ieee": "std_ieee_xplore_papers.xlsx",
        "pubmed": "std_pubmed_papers.xlsx",
    }

    # Final merged output file name
    MERGED_FILENAME: str = "standardized_papers.xlsx"

    # Field mapping configuration file name
    MAPPING_FILENAME: str = "column_mapping.json"

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initializes the standardization tool, parses directories, and loads field mappings.

        :param logger: Logger instance
        :raises DataValidationError: If the base directories do not exist
        :raises FieldMappingError: If the field mapping file is missing or invalid
        """
        self.logger = logger
        self._init_dirs()
        self.field_mapping: Dict[str, dict] = self._load_mapping()

    # =========================
    # Path and Configuration
    # =========================

    def _init_dirs(self) -> None:
        """
        Initializes the directories for data (raw, mappings, standardized).

        :raises DataValidationError: If the raw data or mappings directories do not exist
        """
        project_root = Path(__file__).resolve().parents[3]
        root_dir = project_root / "data" / "systematic_review"

        self.raw_dir = root_dir / "raw"
        self.mappings_dir = root_dir / "mappings"
        self.std_dir = root_dir / "standardized"
        self.mapping_path = self.mappings_dir / self.MAPPING_FILENAME

        if not self.raw_dir.is_dir():
            raise DataValidationError(f"Raw data directory does not exist: {self.raw_dir}")
        if not self.mappings_dir.is_dir():
            raise DataValidationError(f"Field mappings directory does not exist: {self.mappings_dir}")

        self.logger.info(f"[PATH] Stage directory       : {self.stage_root}")
        self.logger.info(f"[PATH] Standardized directory: {self.standardized_root}")

    def _load_mapping(self) -> Dict[str, dict]:
        """
        Loads the field mapping configuration file (JSON).

        :return: A dictionary containing the field mapping (logical field → {source column names, standard_name})
        :raises DataValidationError: If the mapping file is missing
        :raises FieldMappingError: If the mapping content is empty or in an invalid format
        """
        if not self.mapping_path.is_file():
            raise DataValidationError(f"Field mapping file does not exist: {self.mapping_path}")

        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if not mapping or not isinstance(mapping, dict):
                raise FieldMappingError("Field mapping configuration is empty or invalid format.")
            self.logger.info(f"[CONFIG] Field mapping loaded: {self.mapping_path}")
            return mapping
        except (OSError, json.JSONDecodeError) as exc:
            raise FieldMappingError(f"Failed to load field mapping: {exc}") from exc

    def _get_src2std_map(self, source: str) -> Dict[str, str]:
        """
        Generates a mapping of "raw column name → standard column name" for the given source from the field mapping configuration.

        :param source: Source identifier (e.g., scopus, wos, ieee, pubmed)
        :return: A dictionary mapping source column names to standard column names (returns an empty dictionary if the source is not configured)
        """
        src2std: Dict[str, str] = {}
        for _, info in self.field_mapping.items():
            src_col = info.get(source)
            std_col = info.get("standard_name")
            if src_col and std_col: src2std[src_col] = std_col
        return src2std

    # =========================
    # DataFrame Cleaning Tools
    # =========================

    @staticmethod
    def _clean_title_doi_year(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans title, DOI, and year columns: trims spaces, converts to lowercase, and filters out invalid years.

        :param df: Input DataFrame
        :return: Cleaned DataFrame (a copy to avoid chained assignments)
        """
        df = df.copy()

        if "Article Title" in df.columns:
            df["Article Title"] = (df["Article Title"].astype(str).str.strip().str.lower())

        if "DOI" in df.columns: df["DOI"] = df["DOI"].astype(str).str.strip().str.lower()

        if "Publication Year" in df.columns:
            df["Publication Year"] = pd.to_numeric( df["Publication Year"], errors="coerce").astype("Int64")
            df = df[df["Publication Year"].ge(2000) & df["Publication Year"].notnull()].copy()

        return df

    @staticmethod
    def _normalize_keywords(val: object) -> str:
        """
        Normalizes keyword fields (Index Keywords).

        :param val: Original cell value
        :return: Normalized keyword string (semicolon-separated, Title Case), returns empty string if not a string
        """
        if not isinstance(val, str): return ""
        items = [item.strip() for item in val.split(";") if item.strip()]
        return "; ".join(word.title() for word in items)

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate records based on title, year, and DOI.

        :param df: Input DataFrame
        :return: Deduplicated DataFrame (returns original data if key columns are missing)
        """
        dedup_keys = ["Article Title", "Publication Year", "DOI"]
        existing_keys = [col for col in dedup_keys if col in df.columns]
        if not existing_keys: return df
        return df.drop_duplicates(subset=existing_keys, keep="first")

    @staticmethod
    def _sort_by_year(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the literature records by year in descending order (if the year column exists).

        :param df: Input DataFrame
        :return: Sorted DataFrame (returns original if no year column)
        """
        if "Publication Year" in df.columns:
            return df.sort_values(
                by="Publication Year",
                ascending=False,
                na_position="last",
            )
        return df

    # =========================
    # Single File Standardization
    # =========================

    def standardize_single_file(
        self,
        in_path: Path,
        out_path: Path,
        src2std_map: Dict[str, str],
    ) -> None:
        """
        Standardizes, cleans, deduplicates, and saves a single literature data file.

        :param in_path: Input raw data file path
        :param out_path: Standardized result file path
        :param src2std_map: Mapping dictionary from source column names to standard column names
        :return: None
        :raises Exception: If reading or saving fails
        """
        try:
            df = load_table(in_path, logger=self.logger)
            self.logger.info(f"[LOAD] {in_path.name}: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as exc:
            self.logger.error(f"[LOAD FAILED] File {in_path.name}, error: {exc}")
            raise

        # Rename columns based on the mapping (only for columns present in the mapping)
        if not src2std_map:
            self.logger.warning(f"[MAPPING MISSING] Source {in_path.name} has no column mapping, keeping original column names.")
            df_std = df.copy()
        else:
            rename_dict = {
                col: src2std_map[col]
                for col in df.columns
                if col in src2std_map
            }
            df_std = df.rename(columns=rename_dict)

        # Clean title, DOI, and year columns
        df_std = self._clean_title_doi_year(df_std)

        # Clean keyword field
        if "Index Keywords" in df_std.columns:
            df_std["Index Keywords"] = df_std["Index Keywords"].apply(self._normalize_keywords)

        # Deduplicate and sort by year
        df_dedup = self._remove_duplicates(df_std)
        df_sorted = self._sort_by_year(df_dedup)

        # Save result
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_table(df_sorted, out_path, logger=self.logger)
            self.logger.info(f"[SAVE] {out_path.name}: {len(df_sorted)} rows")
        except Exception as exc:
            self.logger.error(f"[SAVE FAILED] File {out_path.name}, error: {exc}")
            raise

    # =========================
    # Batch Processing and Merging
    # =========================

    def run(self) -> None:
        """
        Executes the batch process: standardizes raw files by source, and merges the standardized results.
        :return: None
        :raises DataValidationError: If all raw files or standardized results are missing
        """
        std_files = []

        # 1. Standardize raw files from each source
        for source, raw_filename in self.RAW_SOURCE_FILES.items():
            in_path = self.raw_dir / raw_filename
            if not in_path.is_file():
                self.logger.warning(f"[MISSING] Raw file for source {source} does not exist: {in_path}")
                continue

            std_filename = self.STD_SOURCE_FILES.get(source)
            if not std_filename:
                self.logger.warning(f"[CONFIG MISSING] No output file configured for source {source}, skipping.")
                continue

            out_path = self.std_dir / std_filename
            src2std_map = self._get_src2std_map(source)

            self.logger.info(f"[STANDARDIZE] Source {source}: {in_path.name} → {out_path.name}")
            self.standardize_single_file(in_path, out_path, src2std_map)
            std_files.append(out_path)

        if not std_files:
            raise DataValidationError(f"All sources' raw files are missing or unavailable, no standardized results generated: {self.raw_dir}")

        # 2. Merge all standardized files
        merged_output = self.std_dir / self.MERGED_FILENAME
        self.logger.info("[MAIN] Starting to merge standardized result files...")
        try:
            merge_and_deduplicate_std_files(
                std_file_paths=std_files,
                output_path=merged_output,
                logger=self.logger,
            )
            self.logger.info(f"[MAIN] Standardized results merged, output file: {merged_output.name}")
        except Exception as exc:
            self.logger.error(f"[MAIN] Failed to merge standardized results: {exc}")
            raise


def main() -> None:
    """
    Main function: Configure logging and execute column name standardization and merging for multi-source literature data.
    :return: None
    """
    logger = setup_logger(verbose=True)
    logger.info("[MAIN] Literature column standardization process started")

    standardizer = LiteratureColumnStandardizer(logger)
    standardizer.run()

    logger.info("[MAIN] Literature column standardization process completed")


if __name__ == "__main__":
    main()