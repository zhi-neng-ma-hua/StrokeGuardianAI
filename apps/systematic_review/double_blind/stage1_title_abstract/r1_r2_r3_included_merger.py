# -*- coding: utf-8 -*-
"""
r1_r2_r3_final_included_merger.py

Stage 1  R1/R2/R3 screening results – final inclusion statistics and export tool.

Main functions
--------------
1. Read the Stage 1 results file:
   data/systematic_review/double_blind/stage1_title_abstract/R1_R2_R3_analysis_results.xlsx

2. After normalizing decision columns (Decision_R1 / Decision_R2 / Decision_R3),
   classify finally included studies and collect No. lists:
   (1) Decision_R1 = Decision_R2 = "include"
   (2) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 = "include"
   (3) Decision_R1 ≠ Decision_R2 AND Decision_R3 = "include"

3. Check records where Decision_R3 is missing and output No. lists for:
   (1) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 is blank
   (2) Decision_R1 ≠ Decision_R2 AND Decision_R3 is blank

4. Write all finally included studies (union of the three include categories) to:
   data/systematic_review/double_blind/stage1_title_abstract/R1_R2_R3_final_included_studies.xlsx

5. Write a concise, structured TXT summary report to:
   data/systematic_review/double_blind/stage1_title_abstract/R1_R2_R3_final_included_summary.txt

Author: Aiden Cao <zhinengmahua@gmail.com>
Date  : 2025-07-13
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger_manager import LoggerManager

# Canonical value in Remark column that is treated as "access restrictions"
ACCESS_RESTRICTIONS_VALUE = "access restrictions"


def setup_logger(
    name: str = "r1_r2_r3_final_included_merger",
    verbose: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger.

    :param name: Logger name.
    :param verbose: Whether to enable DEBUG-level logging.
    :return: Configured logging.Logger instance.
    """
    return LoggerManager.setup_logger(
        logger_name=name,
        module_name=__name__,
        verbose=verbose,
    )


class Stage1R1R2R3FinalInclusionAnalyzer:
    """
    Controller for final inclusion statistics and export of Stage 1 R1 R2 R3
    R1/R2/R3 screening results.
    """

    INPUT_FILENAME = "R1_R2_R3_analysis_results.xlsx"
    OUTPUT_FILENAME = "R1_R2_R3_final_included_studies.xlsx"
    SUMMARY_FILENAME = "R1_R2_R3_final_included_summary.txt"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the analyzer, resolve paths, and set up logging.

        :param logger: Logger instance; if None, a default logger is created.
        """
        self.logger = logger or setup_logger()
        self._init_paths()

    def _init_paths(self) -> None:
        """
        Initialize and validate input/output paths under Stage 1 R1 R2 R3 directory.
        """
        # apps/systematic_review/double_blind/stage1_title_abstract/r1_r2_r3_final_included_merger_final_included_merger.py
        project_root = Path(__file__).resolve().parents[4]
        stage_root = (
            project_root
            / "data"
            / "systematic_review"
            / "double_blind"
            / "stage1_title_abstract"
        )

        self.stage_root = stage_root
        self.input_path = stage_root / self.INPUT_FILENAME
        self.output_path = stage_root / self.OUTPUT_FILENAME
        self.summary_path = stage_root / self.SUMMARY_FILENAME

        if not self.input_path.is_file():
            raise FileNotFoundError(f"Analysis results file not found: {self.input_path}")

        self.logger.info(
            "[PATH] Input file: %s | Output file: %s | Summary report: %s",
            self.input_path,
            self.output_path,
            self.summary_path,
        )

    # -------------------------------------------------------------------------
    # Normalization utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_decision_column(series: pd.Series) -> pd.Series:
        """
        Normalize a decision column by casting to string, stripping whitespace,
        and converting to lowercase.

        :param series: Decision column as a Series.
        :return: Normalized Series (lowercase, stripped).
        """
        return series.astype(str).str.strip().str.lower()

    @staticmethod
    def _normalize_remark_column(series: pd.Series) -> pd.Series:
        """
        Normalize the Remark column by stripping whitespace, removing trailing
        periods, and converting to lowercase.

        :param series: Remark column as a Series.
        :return: Normalized Remark Series.
        """
        return (
            series.astype(str)
            .str.strip()
            .str.rstrip(".")
            .str.lower()
        )

    @staticmethod
    def _is_blank(series: pd.Series) -> pd.Series:
        """
        Determine whether values are considered blank.

        A value is treated as blank if:
        - it is NaN / None
        - or it becomes an empty string after strip
        - or it is the literal string "nan" (case-insensitive)

        :param series: Series to check.
        :return: Boolean Series where True indicates a blank / missing value.
        """
        s_stripped = series.astype(str).str.strip()
        return series.isna() | s_stripped.eq("") | s_stripped.str.lower().eq("nan")

    @staticmethod
    def _extract_no_list(series: pd.Series) -> List[str]:
        """
        Extract a list of No. values as normalized strings.

        Rules:
        - NaN values are skipped.
        - Values that can be cast to int are formatted as integer strings.
        - Other values are converted to stripped strings.

        :param series: Series containing No. values.
        :return: List of normalized No. strings.
        """
        nos: List[str] = []
        for v in series:
            if pd.isna(v):
                continue
            try:
                nos.append(str(int(v)))
            except (ValueError, TypeError):
                nos.append(str(v).strip())
        return nos

    # -------------------------------------------------------------------------
    # Data loading and decision normalization
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_decision_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Resolve the actual column names for R1/R2/R3 decisions.

        Canonical naming style (preferred):
        - "Decision_R1"
        - "Decision_R2"
        - "Decision_R3"

        Backward-compatible alternative style:
        - "R1_Decision"
        - "R2_Decision"
        - "R3_Decision"

        :param df: Input DataFrame.
        :return: Mapping {"R1": col_name, "R2": col_name, "R3": col_name}.
        :raises KeyError: If any decision column cannot be found.
        """
        candidates = {
            "R1": ("Decision_R1", "R1_Decision"),
            "R2": ("Decision_R2", "R2_Decision"),
            "R3": ("Decision_R3", "R3_Decision"),
        }

        resolved: Dict[str, str] = {}
        for key, options in candidates.items():
            for col in options:
                if col in df.columns:
                    resolved[key] = col
                    break
            if key not in resolved:
                raise KeyError(
                    f"Input file is missing required decision column for {key}: "
                    f"tried {options}"
                )
        return resolved

    def load_results(self) -> pd.DataFrame:
        """
        Read Stage 1 R1 R2 R3 results and add normalized helper columns _R1 / _R2 / _R3.

        The original decision columns are left unchanged. Normalized decisions are
        stored in _R1 / _R2 / _R3 for downstream logic.

        :return: DataFrame with normalized decision helper columns.
        """
        df = pd.read_excel(self.input_path, dtype=str).fillna("")

        if "No." not in df.columns:
            self.logger.warning(
                "[WARN] Input file is missing 'No.' column; No. lists in summary will be empty."
            )

        decision_cols = self._resolve_decision_columns(df)

        df["_R1"] = self._normalize_decision_column(df[decision_cols["R1"]])
        df["_R2"] = self._normalize_decision_column(df[decision_cols["R2"]])
        df["_R3"] = self._normalize_decision_column(df[decision_cols["R3"]])

        self.logger.info(
            "[LOAD] Normalized decisions using columns: Decision_R1=%s, Decision_R2=%s, Decision_R3=%s",
            decision_cols["R1"],
            decision_cols["R2"],
            decision_cols["R3"],
        )

        return df

    # -------------------------------------------------------------------------
    # Final inclusion classification
    # -------------------------------------------------------------------------

    def classify_final_included(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Classify finally included studies into three categories and return masks,
        counts, and No. lists for each category.

        Categories (based on normalized _R1 / _R2 / _R3):
        (1) Decision_R1 = Decision_R2 = "include"
        (2) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 = "include"
        (3) Decision_R1 ≠ Decision_R2           AND Decision_R3 = "include"

        :param df: DataFrame with _R1 / _R2 / _R3 helper columns.
        :return: Dict keyed by 'cat1'/'cat2'/'cat3' with fields:
                 - mask  : boolean Series
                 - count : int
                 - nos   : list of No. strings
        """
        r1 = df["_R1"]
        r2 = df["_R2"]
        r3 = df["_R3"]

        mask_cat1 = (r1 == "include") & (r2 == "include")
        mask_cat2 = (r1 == "unsure") & (r2 == "unsure") & (r3 == "include")
        mask_cat3 = (r1 != r2) & (r3 == "include")

        stats: Dict[str, Dict[str, Any]] = {}

        def _make_stat(key: str, mask: pd.Series, label: str) -> None:
            count = int(mask.sum())
            if "No." in df.columns:
                nos = self._extract_no_list(df.loc[mask, "No."])
            else:
                nos = []
            self.logger.info("[INCLUDE] %s: %d records", label, count)
            if nos:
                self.logger.debug("[INCLUDE] %s No. list: %s", label, nos)
            stats[key] = {"mask": mask, "count": count, "nos": nos}

        _make_stat("cat1", mask_cat1, "(1) Decision_R1 = Decision_R2 = include")
        _make_stat("cat2", mask_cat2, "(2) Decision_R1 = Decision_R2 = unsure, Decision_R3 = include")
        _make_stat("cat3", mask_cat3, "(3) Decision_R1 ≠ Decision_R2, Decision_R3 = include")

        return stats

    # -------------------------------------------------------------------------
    # Decision_R3 missing checks
    # -------------------------------------------------------------------------

    def check_missing_r3(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Check records with missing Decision_R3 under two scenarios:

        (1) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 is blank
        (2) Decision_R1 ≠ Decision_R2           AND Decision_R3 is blank

        :param df: DataFrame with _R1 / _R2 / _R3 helper columns.
        :return: Dict with counts and No. lists for each missing category.
        """
        r1 = df["_R1"]
        r2 = df["_R2"]

        # Prefer canonical Decision_R3; fall back to R3_Decision or _R3 if necessary.
        r3_raw = df.get("Decision_R3", df.get("R3_Decision", df["_R3"]))
        r3_blank = self._is_blank(r3_raw)

        mask_unsure_both = (r1 == "unsure") & (r2 == "unsure") & r3_blank
        mask_diff = (r1 != r2) & r3_blank

        stats: Dict[str, Dict[str, Any]] = {}

        def _make_stat(key: str, mask: pd.Series, label: str) -> None:
            count = int(mask.sum())
            self.logger.info("[MISSING R3] %s: %d records", label, count)
            nos: List[str] = []
            if "No." in df.columns and count > 0:
                nos = self._extract_no_list(df.loc[mask, "No."])
                self.logger.warning("[MISSING R3] %s No. list: %s", label, nos)
            stats[key] = {"count": count, "nos": nos}

        _make_stat(
            "unsure_no_r3",
            mask_unsure_both,
            "Decision_R1 = Decision_R2 = unsure AND Decision_R3 is blank",
        )
        _make_stat(
            "diff_no_r3",
            mask_diff,
            "Decision_R1 ≠ Decision_R2 AND Decision_R3 is blank",
        )

        return stats

    # -------------------------------------------------------------------------
    # Access restrictions statistics
    # -------------------------------------------------------------------------

    def count_access_restrictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Count records where Remark equals "Access restrictions" (case-insensitive,
        trailing periods ignored).

        :param df: DataFrame.
        :return: Dict with keys: mask, count, nos.
        """
        if "Remark" not in df.columns:
            self.logger.warning(
                "[ACCESS] Input file is missing Remark column; skipping Access restrictions statistics."
            )
            return {"mask": pd.Series(False, index=df.index), "count": 0, "nos": []}

        remark_norm = self._normalize_remark_column(df["Remark"])
        mask = remark_norm == ACCESS_RESTRICTIONS_VALUE
        count = int(mask.sum())
        self.logger.info(
            "[ACCESS] Remark = 'Access restrictions' (normalized) count: %d",
            count,
        )

        nos: List[str] = []
        if count > 0 and "No." in df.columns:
            nos = self._extract_no_list(df.loc[mask, "No."])
            self.logger.debug("[ACCESS] Remark = 'Access restrictions' No. list: %s", nos)

        return {"mask": mask, "count": count, "nos": nos}

    # -------------------------------------------------------------------------
    # Export of final included studies
    # -------------------------------------------------------------------------

    @staticmethod
    def _reorder_columns_for_output(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally define the final column order for export.

        Currently returns the DataFrame unchanged, preserving the original column order.

        :param df: DataFrame to reorder.
        :return: DataFrame with desired column order.
        """
        return df

    def write_final_included(self, df: pd.DataFrame, final_mask: pd.Series) -> None:
        """
        Export finally included studies to an Excel file.

        :param df: Original DataFrame.
        :param final_mask: Boolean mask indicating final inclusion.
        """
        final_df = df.loc[final_mask].copy()
        final_df = self._reorder_columns_for_output(final_df)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_excel(self.output_path, index=False)
        self.logger.info(
            "[EXPORT] Final included studies written to: %s | rows=%d",
            self.output_path,
            len(final_df),
        )

    # -------------------------------------------------------------------------
    # Summary TXT report
    # -------------------------------------------------------------------------

    def _build_summary_text(
        self,
        total_count: int,
        final_stats: Dict[str, Dict[str, Any]],
        excluded_count: int,
        access_stats: Dict[str, Any],
        access_excluded_stats: Dict[str, Any],
        missing_r3_stats: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Build the TXT summary report text.

        :return: Summary report as a formatted string.
        """
        lines: List[str] = []

        lines.append("=" * 58)
        lines.append("Stage 1 R1 R2 R3 – Final Inclusion Summary")
        lines.append("=" * 58)
        lines.append("")

        # 1. Overall counts
        lines.append("1. Overall counts")
        lines.append(f"- Total records                   : {total_count}")
        lines.append(
            f"- Final included (any category)   : {final_stats['total']['count']}"
        )
        lines.append(f"- Final excluded                  : {excluded_count}")
        lines.append("")

        # 2. Final inclusion categories
        lines.append("2. Final inclusion categories")
        lines.append(
            f"- (1) Decision_R1 = Decision_R2 = include                        : {final_stats['cat1']['count']}"
        )
        lines.append(
            f"- (2) Decision_R1 = Decision_R2 = unsure, Decision_R3 = include : {final_stats['cat2']['count']}"
        )
        lines.append(
            f"- (3) Decision_R1 ≠ Decision_R2, Decision_R3 = include          : {final_stats['cat3']['count']}"
        )
        lines.append("")

        # 3. Decision_R3 missing checks
        lines.append("3. Decision_R3 missing checks")
        lines.append(
            f"- Decision_R1 = Decision_R2 = unsure, Decision_R3 missing : {missing_r3_stats['unsure_no_r3']['count']}"
        )
        lines.append(
            f"- Decision_R1 ≠ Decision_R2, Decision_R3 missing          : {missing_r3_stats['diff_no_r3']['count']}"
        )
        lines.append("")

        # 4. Access restrictions
        lines.append("4. Access restrictions (Remark)")
        lines.append(
            f"- All records with Remark = 'Access restrictions'     : {access_stats['count']}"
        )
        lines.append(
            f"- Excluded records with Remark = 'Access restrictions': {access_excluded_stats['count']}"
        )
        lines.append("")

        # 5. No. lists (if available)
        lines.append("5. No. lists (if available)")
        if final_stats["cat1"]["nos"]:
            lines.append(
                f"- Final included (1) R1 = R2 = include              : {final_stats['cat1']['nos']}"
            )
        if final_stats["cat2"]["nos"]:
            lines.append(
                f"- Final included (2) R1 = R2 = unsure, R3 = include : {final_stats['cat2']['nos']}"
            )
        if final_stats["cat3"]["nos"]:
            lines.append(
                f"- Final included (3) R1 ≠ R2, R3 = include          : {final_stats['cat3']['nos']}"
            )
        if missing_r3_stats["unsure_no_r3"]["nos"]:
            lines.append(
                f"- Decision_R1 = Decision_R2 = unsure, R3 missing    : {missing_r3_stats['unsure_no_r3']['nos']}"
            )
        if missing_r3_stats["diff_no_r3"]["nos"]:
            lines.append(
                f"- Decision_R1 ≠ Decision_R2, R3 missing             : {missing_r3_stats['diff_no_r3']['nos']}"
            )
        if access_stats["nos"]:
            lines.append(
                f"- All Access restrictions                          : {access_stats['nos']}"
            )
        if access_excluded_stats["nos"]:
            lines.append(
                f"- Excluded Access restrictions                     : {access_excluded_stats['nos']}"
            )

        if (
            not final_stats["cat1"]["nos"]
            and not final_stats["cat2"]["nos"]
            and not final_stats["cat3"]["nos"]
            and not missing_r3_stats["unsure_no_r3"]["nos"]
            and not missing_r3_stats["diff_no_r3"]["nos"]
            and not access_stats["nos"]
            and not access_excluded_stats["nos"]
        ):
            lines.append(
                "- Column 'No.' is missing or empty; No. lists are not printed."
            )

        lines.append("")
        lines.append("=" * 58)
        return "\n".join(lines)

    def _write_summary(self, text: str) -> None:
        """
        Write the summary text to a TXT file.

        :param text: Summary report text.
        """
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(text, encoding="utf-8")
        self.logger.info(
            "[SUMMARY] Statistics written to TXT report: %s", self.summary_path
        )

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the full Stage 1 pipeline:

        1. Load Stage 1 R1 R2 R3 results.
        2. Classify final inclusion into three categories and compute masks.
        3. Derive final inclusion / exclusion counts.
        4. Check cases where Decision_R3 is missing.
        5. Count Access restrictions (overall and among excluded).
        6. Export final included studies to Excel.
        7. Generate and write a structured TXT summary report.
        """
        # 1. Load and normalize decisions
        df = self.load_results()
        total_count = len(df)

        # 2. Final inclusion classification and masks
        final_stats = self.classify_final_included(df)
        included_mask = (
            final_stats["cat1"]["mask"]
            | final_stats["cat2"]["mask"]
            | final_stats["cat3"]["mask"]
        )
        included_count = int(included_mask.sum())
        excluded_mask = ~included_mask
        excluded_count = int(excluded_mask.sum())
        final_stats["total"] = {"mask": included_mask, "count": included_count}

        self.logger.info("[TOTAL] Final included studies: %d", included_count)
        self.logger.info("[TOTAL] Final excluded studies: %d", excluded_count)

        # 3. Decision_R3 missing checks
        missing_r3_stats = self.check_missing_r3(df)

        # 4. Remark = Access restrictions (all records)
        access_stats = self.count_access_restrictions(df)

        # 5. Remark = Access restrictions among finally excluded
        if access_stats["count"] > 0 and "Remark" in df.columns:
            remark_norm = self._normalize_remark_column(df["Remark"])
            access_mask_all = remark_norm == ACCESS_RESTRICTIONS_VALUE
            access_excluded_mask = access_mask_all & excluded_mask
            access_excluded_count = int(access_excluded_mask.sum())
            access_excluded_nos: List[str] = []
            if access_excluded_count > 0 and "No." in df.columns:
                access_excluded_nos = self._extract_no_list(
                    df.loc[access_excluded_mask, "No."]
                )
            access_excluded_stats = {
                "count": access_excluded_count,
                "nos": access_excluded_nos,
            }
            self.logger.info(
                "[ACCESS] Remark = 'Access restrictions' AND finally excluded: %d",
                access_excluded_count,
            )
        else:
            access_excluded_stats = {"count": 0, "nos": []}

        # 6. Export final included studies
        self.write_final_included(df, included_mask)

        # 7. Build and write TXT summary report
        summary_text = self._build_summary_text(
            total_count=total_count,
            final_stats=final_stats,
            excluded_count=excluded_count,
            access_stats=access_stats,
            access_excluded_stats=access_excluded_stats,
            missing_r3_stats=missing_r3_stats,
        )
        self._write_summary(summary_text)


def main() -> None:
    """
    Main entry point: run Stage 1 R1 R2 R3 final inclusion / exclusion
    statistics and export.
    """
    logger = setup_logger(verbose=True)
    logger.info("[MAIN] Stage 1 R1 R2 R3 final inclusion summary started")

    analyzer = Stage1R1R2R3FinalInclusionAnalyzer(logger=logger)
    analyzer.run()

    logger.info("[MAIN] Stage 1 R1 R2 R3 final inclusion summary finished")


if __name__ == "__main__":
    main()
