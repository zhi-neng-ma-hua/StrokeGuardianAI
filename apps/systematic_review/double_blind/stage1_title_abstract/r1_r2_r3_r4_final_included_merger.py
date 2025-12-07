# -*- coding: utf-8 -*-
"""
r1_r2_r3_r4_final_included_merger.py

Stage 1 (Title / Abstract) R1/R2/R3/R4 screening – final inclusion
classification and export based on the *final* analysis results.

Main functions
--------------
1. Read the Stage 1 results file:
   data/systematic_review/double_blind/stage1_title_abstract/R1_R2_R3_R4_final_analysis_results.xlsx

2. Normalise decision columns (Decision_R1 / Decision_R2 / Decision_R3 /
   Decision_R4 / Decision_R1_R2_R3_R4) and Notes_R4, then classify each record
   into three pattern groups:

   Include patterns
   ----------------
   (1) Decision_R1 = Decision_R2 = Decision_R4 = "include"
   (2) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 = Decision_R4 = "include"
   (3) Decision_R1 ≠ Decision_R2 AND Decision_R3 = Decision_R4 = "include"
   (4) Decision_R1_R2_R3_R4 = "include"

   Unsure patterns
   ---------------
   (5) Decision_R4 = "unsure" AND Notes_R4 (strip+lower) = "no access"
   (6) Decision_R4 = "unsure" AND Notes_R4 (strip+lower) = "chinese article; paid access"

   Exclude patterns
   ----------------
   (7) Decision_R1 = Decision_R2 = Decision_R4 = "exclude"
   (8) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 = Decision_R4 = "exclude"
   (9) Decision_R1 ≠ Decision_R2 AND Decision_R3 = Decision_R4 = "exclude"
   (10) Decision_R1_R2_R3_R4 = "exclude"

3. Final inclusion set:
   - All records matching patterns (1)–(4) (Include patterns).

4. Export all finally included studies to:
   data/systematic_review/double_blind/stage1_title_abstract/R1_R2_R3_R4_final_included_studies.xlsx

5. Write a concise, structured TXT summary report to:
   data/systematic_review/double_blind/stage1_title_abstract/R1_R2_R3_R4_final_included_summary.txt
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger_manager import LoggerManager


def setup_logger(
    name: str = "r1_r2_r3_r4_final_included_merger",
    verbose: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger

    :param name: Logger name
    :param verbose: Whether to enable DEBUG level logging
    :return: logging.Logger instance
    """
    return LoggerManager.setup_logger(
        logger_name=name,
        module_name=__name__,
        verbose=verbose,
    )


class Stage1R1R2R3R4FinalInclusionAnalyzer:
    """
    Final inclusion classifier and exporter for Stage 1
    R1/R2/R3/R4 *final* screening results.
    """

    INPUT_FILENAME = "R1_R2_R3_R4_final_analysis_results.xlsx"
    OUTPUT_FILENAME = "R1_R2_R3_R4_final_included_studies.xlsx"
    SUMMARY_FILENAME = "R1_R2_R3_R4_final_included_summary.txt"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialise the analyzer and resolve Stage 1 paths

        :param logger: Logger instance; if None, a default logger is created
        """
        self.logger = logger or setup_logger()
        self._init_paths()

    # -------------------------------------------------------------------------
    # Path resolution
    # -------------------------------------------------------------------------

    def _init_paths(self) -> None:
        """
        Resolve and validate Stage 1 input/output paths
        """
        # apps/systematic_review/double_blind/stage1_title_abstract/...
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
            raise FileNotFoundError(f"Final analysis results file not found: {self.input_path}")

        self.logger.info(
            "[PATH] Input file: %s | Output file: %s | Summary report: %s",
            self.input_path,
            self.output_path,
            self.summary_path,
        )

    # -------------------------------------------------------------------------
    # Normalisation utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_decision_column(series: pd.Series) -> pd.Series:
        """
        Normalise a decision column

        :param series: Decision column as a Series
        :return: Normalised Series (lowercase, stripped)
        """
        return series.astype(str).str.strip().str.lower()

    @staticmethod
    def _normalize_notes_column(series: pd.Series) -> pd.Series:
        """
        Normalise Notes_R4

        :param series: Notes column as a Series
        :return: Normalised Notes_R4 Series
        """
        return series.astype(str).str.strip().str.lower()

    @staticmethod
    def _extract_no_list(series: pd.Series) -> List[str]:
        """
        Extract a list of 'No.' values as normalised strings

        NaN values are skipped; numeric-like values are cast to int strings.

        :param series: Series containing 'No.' values
        :return: List of normalised No. strings
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
    # Data loading and normalisation
    # -------------------------------------------------------------------------

    def _require_columns(self, df: pd.DataFrame) -> None:
        """
        Ensure that the input table contains required decision/notes columns

        :param df: Input DataFrame
        :raises KeyError: If any required column is missing
        """
        required = [
            "Decision_R1",
            "Decision_R2",
            "Decision_R3",
            "Decision_R4",
            "Decision_R1_R2_R3_R4",
            "Notes_R4",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Input file is missing required columns: {missing}")

    def load_results(self) -> pd.DataFrame:
        """
        Read final Stage 1 R1/R2/R3/R4 results and add helper columns

        Helper columns:
            _R1, _R2, _R3, _R4   : normalised decisions
            _RAgg                : normalised Decision_R1_R2_R3_R4
            _N4                  : normalised Notes_R4

        :return: DataFrame with helper columns
        """
        df = pd.read_excel(self.input_path, dtype=str).fillna("")

        if "No." not in df.columns:
            self.logger.warning(
                "[WARN] Input file is missing 'No.' column; No. lists in summary will be empty"
            )

        self._require_columns(df)

        df["_R1"] = self._normalize_decision_column(df["Decision_R1"])
        df["_R2"] = self._normalize_decision_column(df["Decision_R2"])
        df["_R3"] = self._normalize_decision_column(df["Decision_R3"])
        df["_R4"] = self._normalize_decision_column(df["Decision_R4"])
        df["_RAgg"] = self._normalize_decision_column(df["Decision_R1_R2_R3_R4"])
        df["_N4"] = self._normalize_notes_column(df["Notes_R4"])

        self.logger.info(
            "[LOAD] Normalised decisions for Decision_R1/R2/R3/R4 and Decision_R1_R2_R3_R4, plus Notes_R4"
        )
        return df

    # -------------------------------------------------------------------------
    # Pattern classification (include / unsure / exclude)
    # -------------------------------------------------------------------------

    def classify_patterns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Classify records into 10 patterns based on helper columns

        Include patterns
        ----------------
        (1) Decision_R1 = Decision_R2 = Decision_R4 = "include"
        (2) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 = Decision_R4 = "include"
        (3) Decision_R1 ≠ Decision_R2 AND Decision_R3 = Decision_R4 = "include"
        (4) Decision_R1_R2_R3_R4 = "include"

        Unsure patterns
        ---------------
        (5) Decision_R4 = "unsure" AND Notes_R4 = "no access"
        (6) Decision_R4 = "unsure" AND Notes_R4 = "chinese article; paid access"

        Exclude patterns
        ----------------
        (7) Decision_R1 = Decision_R2 = Decision_R4 = "exclude"
        (8) Decision_R1 = Decision_R2 = "unsure" AND Decision_R3 = Decision_R4 = "exclude"
        (9) Decision_R1 ≠ Decision_R2 AND Decision_R3 = Decision_R4 = "exclude"
        (10) Decision_R1_R2_R3_R4 = "exclude"

        :param df: DataFrame with _R* and _N4 helpers
        :return: Dict keyed by pattern id with mask, count and No. list
        """
        r1 = df["_R1"]
        r2 = df["_R2"]
        r3 = df["_R3"]
        r4 = df["_R4"]
        ragg = df["_RAgg"]
        n4 = df["_N4"]

        same_r1_r2 = r1 == r2
        diff_r1_r2 = r1 != r2

        # Include patterns (1–4)
        inc1 = same_r1_r2 & (r1 == "include") & (r4 == "include")
        inc2 = same_r1_r2 & (r1 == "unsure") & (r3 == "include") & (r4 == "include")
        inc3 = diff_r1_r2 & (r3 == "include") & (r4 == "include")
        inc4 = ragg == "include"

        # Unsure patterns (5–6)
        uns5 = (r4 == "unsure") & (n4 == "no access")
        uns6 = (r4 == "unsure") & (n4 == "chinese article; paid access")

        # Exclude patterns (7–10)
        exc7 = same_r1_r2 & (r1 == "exclude") & (r4 == "exclude")
        exc8 = same_r1_r2 & (r1 == "unsure") & (r3 == "exclude") & (r4 == "exclude")
        exc9 = diff_r1_r2 & (r3 == "exclude") & (r4 == "exclude")
        exc10 = ragg == "exclude"

        patterns: Dict[str, Dict[str, Any]] = {}

        def _add_pattern(key: str, mask: pd.Series, label: str) -> None:
            """
            Add one pattern's mask / count / No. list to the dictionary

            :param key: Pattern identifier key, e.g. "inc1"
            :param mask: Boolean mask for this pattern
            :param label: Human-readable pattern label for logging
            """
            count = int(mask.sum())
            if "No." in df.columns:
                nos = self._extract_no_list(df.loc[mask, "No."])
            else:
                nos = []
            patterns[key] = {"mask": mask, "count": count, "nos": nos}
            self.logger.info("[PATTERN] %s: %d records", label, count)

        _add_pattern("inc1", inc1, "Include (1) R1 = R2 = R4 = include")
        _add_pattern("inc2", inc2, "Include (2) R1 = R2 = unsure, R3 = R4 = include")
        _add_pattern("inc3", inc3, "Include (3) R1 ≠ R2, R3 = R4 = include")
        _add_pattern("inc4", inc4, "Include (4) Decision_R1_R2_R3_R4 = include")

        _add_pattern("uns5", uns5, "Unsure (5) R4 = unsure, Notes_R4 = 'no access'")
        _add_pattern("uns6", uns6, "Unsure (6) R4 = unsure, Notes_R4 = 'chinese article; paid access'")

        _add_pattern("exc7", exc7, "Exclude (7) R1 = R2 = R4 = exclude")
        _add_pattern("exc8", exc8, "Exclude (8) R1 = R2 = unsure, R3 = R4 = exclude")
        _add_pattern("exc9", exc9, "Exclude (9) R1 ≠ R2, R3 = R4 = exclude")
        _add_pattern("exc10", exc10, "Exclude (10) Decision_R1_R2_R3_R4 = exclude")

        return patterns

    # -------------------------------------------------------------------------
    # Export of final included studies
    # -------------------------------------------------------------------------

    @staticmethod
    def _reorder_columns_for_output(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally define the final column order for export

        :param df: DataFrame to reorder
        :return: DataFrame with desired column order
        """
        return df

    def write_final_included(self, df: pd.DataFrame, final_mask: pd.Series) -> int:
        """
        Export finally included studies (Include patterns 1–4) to an Excel file

        :param df: Original DataFrame
        :param final_mask: Boolean mask defining final inclusion
        :return: Number of included rows
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
        return len(final_df)

    # -------------------------------------------------------------------------
    # Summary TXT report
    # -------------------------------------------------------------------------

    def _build_summary_text(
        self,
        total_count: int,
        patterns: Dict[str, Dict[str, Any]],
        included_count: int,
        unsure_count: int,
        exclude_count: int,
        agg_include_count: int,
        agg_exclude_count: int,
        unmatched_count: int,
    ) -> str:
        """
        Build the TXT summary report text

        :return: Summary report as a formatted string
        """
        lines: List[str] = []

        lines.append("=" * 72)
        lines.append("Stage 1 R1/R2/R3/R4 – Final Inclusion Summary (Final Analysis)")
        lines.append("=" * 72)
        lines.append("")

        # 1. Overall counts
        lines.append("1. Overall counts")
        lines.append(f"- Total records                         : {total_count}")
        lines.append(f"- Final included (patterns 1–4)         : {included_count}")
        lines.append(f"- Unsure-only (patterns 5–6)            : {unsure_count}")
        lines.append(f"- Excluded (patterns 7–10)              : {exclude_count}")
        lines.append(f"- Aggregated include (pattern 4)        : {agg_include_count}")
        lines.append(f"- Aggregated exclude (pattern 10)       : {agg_exclude_count}")
        lines.append(f"- Unmatched (no pattern hit)            : {unmatched_count}")
        lines.append("")

        def _append_no_block(label: str, nos: List[str]) -> None:
            """
            Append a compact No. list block, if any

            :param label: Pattern label, e.g. "(1)"
            :param nos: List of No. values
            """
            if not nos:
                return
            lines.append(f"  · {label} No. list:")
            lines.append(f"    {nos}")
            lines.append("")

        # 2. Include patterns (1–4) – final inclusion set
        lines.append("2. Include patterns (final inclusion set, patterns 1–4)")
        lines.append(f"- (1) R1 = R2 = R4 = include                           : {patterns['inc1']['count']}")
        _append_no_block("(1)", patterns["inc1"]["nos"])

        lines.append(f"- (2) R1 = R2 = unsure, R3 = R4 = include              : {patterns['inc2']['count']}")
        _append_no_block("(2)", patterns["inc2"]["nos"])

        lines.append(f"- (3) R1 ≠ R2, R3 = R4 = include                       : {patterns['inc3']['count']}")
        _append_no_block("(3)", patterns["inc3"]["nos"])

        lines.append(f"- (4) Decision_R1_R2_R3_R4 = include                   : {patterns['inc4']['count']}")
        _append_no_block("(4)", patterns["inc4"]["nos"])

        # 3. Unsure patterns (5–6)
        lines.append("3. Unsure patterns (not counted as final included)")
        lines.append(
            f"- (5) R4 = unsure, Notes_R4 = 'no access'              : {patterns['uns5']['count']}"
        )
        _append_no_block("(5)", patterns["uns5"]["nos"])

        lines.append(
            f"- (6) R4 = unsure, Notes_R4 = 'chinese article; paid access' : {patterns['uns6']['count']}"
        )
        _append_no_block("(6)", patterns["uns6"]["nos"])

        # 4. Exclude patterns (7–10)
        lines.append("4. Exclude patterns")
        lines.append(f"- (7) R1 = R2 = R4 = exclude                           : {patterns['exc7']['count']}")
        _append_no_block("(7)", patterns["exc7"]["nos"])

        lines.append(
            f"- (8) R1 = R2 = unsure, R3 = R4 = exclude              : {patterns['exc8']['count']}"
        )
        _append_no_block("(8)", patterns["exc8"]["nos"])

        lines.append(f"- (9) R1 ≠ R2, R3 = R4 = exclude                       : {patterns['exc9']['count']}")
        _append_no_block("(9)", patterns["exc9"]["nos"])

        lines.append(f"- (10) Decision_R1_R2_R3_R4 = exclude                  : {patterns['exc10']['count']}")
        _append_no_block("(10)", patterns["exc10"]["nos"])

        lines.append("=" * 72)
        return "\n".join(lines)

    def _write_summary(self, text: str) -> None:
        """
        Write the summary text to a TXT file

        :param text: Summary report text
        """
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(text, encoding="utf-8")
        self.logger.info(
            "[SUMMARY] Final inclusion summary written to TXT report: %s",
            self.summary_path,
        )

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the full Stage 1 final inclusion pipeline on final analysis results

        Steps:
        1. Load Stage 1 R1/R2/R3/R4 final analysis results
        2. Classify records into 10 patterns (1–4 include, 5–6 unsure, 7–10 exclude)
        3. Define final inclusion set as patterns 1–4
        4. Export final included studies to Excel
        5. Generate and write a structured TXT summary report
        """
        # 1. Load and normalise decisions/notes
        df = self.load_results()
        total_count = len(df)

        # 2. Classify patterns
        patterns = self.classify_patterns(df)

        include_mask = (
            patterns["inc1"]["mask"]
            | patterns["inc2"]["mask"]
            | patterns["inc3"]["mask"]
            | patterns["inc4"]["mask"]
        )
        unsure_mask = patterns["uns5"]["mask"] | patterns["uns6"]["mask"]
        exclude_mask = (
            patterns["exc7"]["mask"]
            | patterns["exc8"]["mask"]
            | patterns["exc9"]["mask"]
            | patterns["exc10"]["mask"]
        )

        all_pattern_mask = include_mask | unsure_mask | exclude_mask
        unmatched_mask = ~all_pattern_mask

        included_count = int(include_mask.sum())
        unsure_count = int(unsure_mask.sum())
        exclude_count = int(exclude_mask.sum())
        agg_include_count = int(patterns["inc4"]["count"])
        agg_exclude_count = int(patterns["exc10"]["count"])
        unmatched_count = int(unmatched_mask.sum())

        self.logger.info("[TOTAL] Final included (patterns 1–4)         : %d", included_count)
        self.logger.info("[TOTAL] Unsure-only (patterns 5–6)            : %d", unsure_count)
        self.logger.info("[TOTAL] Excluded (patterns 7–10)              : %d", exclude_count)
        self.logger.info("[TOTAL] Aggregated include (pattern 4)        : %d", agg_include_count)
        self.logger.info("[TOTAL] Aggregated exclude (pattern 10)       : %d", agg_exclude_count)
        self.logger.info("[TOTAL] Unmatched (no pattern hit)            : %d", unmatched_count)

        # 3. Export final included studies (patterns 1–4)
        self.write_final_included(df, include_mask)

        # 4. Build and write TXT summary
        summary_text = self._build_summary_text(
            total_count=total_count,
            patterns=patterns,
            included_count=included_count,
            unsure_count=unsure_count,
            exclude_count=exclude_count,
            agg_include_count=agg_include_count,
            agg_exclude_count=agg_exclude_count,
            unmatched_count=unmatched_count,
        )
        self._write_summary(summary_text)


def main() -> None:
    """
    Main entry point to run the Stage 1 R1/R2/R3/R4 final inclusion workflow
    """
    logger = setup_logger(verbose=True)
    logger.info("[MAIN] Stage 1 R1/R2/R3/R4 final inclusion summary (final analysis) started")

    analyzer = Stage1R1R2R3R4FinalInclusionAnalyzer(logger=logger)
    analyzer.run()

    logger.info("[MAIN] Stage 1 R1/R2/R3/R4 final inclusion summary (final analysis) finished")


if __name__ == "__main__":
    main()