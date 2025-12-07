# -*- coding: utf-8 -*-
"""
r1_r2_r3_r4_included_merger.py

Stage 1 (Title / Abstract) R1/R2/R3/R4 screening – final inclusion
classification and export.

Main functions
--------------
1. Read the Stage 1 results file:
   data/systematic_review/double_blind/stage1_title_abstract/
       R1_R2_R3_R4_analysis_results.xlsx

2. Normalise decision columns (Decision_R1 / Decision_R2 / Decision_R3 / Decision_R4)
   and Notes_R4, then classify each record into three pattern groups:

   Include patterns
   ----------------
   (1) Decision_R1 = Decision_R2 = Decision_R4 = "include"
   (2) Decision_R1 = Decision_R2 = "unsure" AND
       Decision_R3 = Decision_R4 = "include"
   (3) Decision_R1 ≠ Decision_R2 AND
       Decision_R3 = Decision_R4 = "include"

   Unsure patterns
   ---------------
   (4) Decision_R4 = "unsure" AND Notes_R4 (strip+lower) = "no access"
   (5) Decision_R4 = "unsure" AND Notes_R4 (strip+lower) =
       "chinese article; paid access"

   Exclude patterns
   ----------------
   (6) Decision_R1 = Decision_R2 = Decision_R4 = "exclude"
   (7) Decision_R1 = Decision_R2 = "unsure" AND
       Decision_R3 = Decision_R4 = "exclude"
   (8) Decision_R1 ≠ Decision_R2 AND
       Decision_R3 = Decision_R4 = "exclude"

3. Final inclusion set:
   - All records matching patterns (1)–(3) (Include patterns only).

4. Export all finally included studies to:
   data/systematic_review/double_blind/stage1_title_abstract/
       R1_R2_R3_R4_included_studies.xlsx

5. Write a concise, structured TXT summary report to:
   data/systematic_review/double_blind/stage1_title_abstract/
       R1_R2_R3_R4_included_summary.txt
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger_manager import LoggerManager


def setup_logger(
    name: str = "r1_r2_r3_r4_included_merger",
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


class Stage1R1R2R3R4InclusionAnalyzer:
    """
    Final inclusion classifier and exporter for Stage 1
    R1/R2/R3/R4 screening results.
    """

    INPUT_FILENAME = "R1_R2_R3_R4_analysis_results.xlsx"
    OUTPUT_FILENAME = "R1_R2_R3_R4_included_studies.xlsx"
    SUMMARY_FILENAME = "R1_R2_R3_R4_included_summary.txt"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialise the analyzer and resolve Stage 1 paths.

        :param logger: Logger instance; if None, a default logger is created.
        """
        self.logger = logger or setup_logger()
        self._init_paths()

    # ---------------------------------------------------------------------
    # Path resolution
    # ---------------------------------------------------------------------

    def _init_paths(self) -> None:
        """
        Resolve and validate Stage 1 input/output paths.
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
            raise FileNotFoundError(f"Analysis results file not found: {self.input_path}")

        self.logger.info(
            "[PATH] Input file: %s | Output file: %s | Summary report: %s",
            self.input_path,
            self.output_path,
            self.summary_path,
        )

    # ---------------------------------------------------------------------
    # Normalisation utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_decision_column(series: pd.Series) -> pd.Series:
        """
        Normalise a decision column: cast to string, strip whitespace, lowercase.
        """
        return series.astype(str).str.strip().str.lower()

    @staticmethod
    def _normalize_notes_column(series: pd.Series) -> pd.Series:
        """
        Normalise a notes column: strip whitespace and lowercase.
        """
        return series.astype(str).str.strip().str.lower()

    @staticmethod
    def _extract_no_list(series: pd.Series) -> List[str]:
        """
        Extract a list of 'No.' values as normalised strings.

        - NaN values are skipped.
        - If a value can be cast to int, use the integer string.
        - Otherwise, use the stripped string representation.
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

    # ---------------------------------------------------------------------
    # Data loading and normalisation
    # ---------------------------------------------------------------------

    def _require_decision_columns(self, df: pd.DataFrame) -> None:
        """
        Ensure that the input table contains the required decision/notes columns.
        """
        required = [
            "Decision_R1",
            "Decision_R2",
            "Decision_R3",
            "Decision_R4",
            "Notes_R4",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Input file is missing required columns: {missing}")

    def load_results(self) -> pd.DataFrame:
        """
        Read Stage 1 R1/R2/R3/R4 results and add normalised helper columns.

        Helper columns:
            _R1, _R2, _R3, _R4   : normalised decisions
            _N4                  : normalised Notes_R4
        """
        df = pd.read_excel(self.input_path, dtype=str).fillna("")

        if "No." not in df.columns:
            self.logger.warning(
                "[WARN] Input file is missing 'No.' column; No. lists in summary will be empty."
            )

        self._require_decision_columns(df)

        df["_R1"] = self._normalize_decision_column(df["Decision_R1"])
        df["_R2"] = self._normalize_decision_column(df["Decision_R2"])
        df["_R3"] = self._normalize_decision_column(df["Decision_R3"])
        df["_R4"] = self._normalize_decision_column(df["Decision_R4"])
        df["_N4"] = self._normalize_notes_column(df["Notes_R4"])

        self.logger.info(
            "[LOAD] Normalized decisions for Decision_R1/2/3/4 and Notes_R4."
        )
        return df

    # ---------------------------------------------------------------------
    # Pattern classification (include / unsure / exclude)
    # ---------------------------------------------------------------------

    def classify_patterns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Classify records into 8 patterns (3 include, 2 unsure, 3 exclude)
        based on the normalised helper columns _R1 / _R2 / _R3 / _R4 / _N4.

        Returns a dict keyed by pattern id ("inc1", "inc2", ...), each with:
            - mask  : boolean Series
            - count : int
            - nos   : list of No. strings (if available).
        """
        r1 = df["_R1"]
        r2 = df["_R2"]
        r3 = df["_R3"]
        r4 = df["_R4"]
        n4 = df["_N4"]

        same_r1_r2 = r1 == r2
        diff_r1_r2 = r1 != r2

        # Include patterns
        inc1 = same_r1_r2 & (r1 == "include") & (r4 == "include")
        inc2 = same_r1_r2 & (r1 == "unsure") & (r3 == "include") & (r4 == "include")
        inc3 = diff_r1_r2 & (r3 == "include") & (r4 == "include")

        # Unsure patterns
        uns4 = (r4 == "unsure") & (n4 == "no access")
        uns5 = (r4 == "unsure") & (n4 == "chinese article; paid access")

        # Exclude patterns
        exc6 = same_r1_r2 & (r1 == "exclude") & (r4 == "exclude")
        exc7 = same_r1_r2 & (r1 == "unsure") & (r3 == "exclude") & (r4 == "exclude")
        exc8 = diff_r1_r2 & (r3 == "exclude") & (r4 == "exclude")

        patterns: Dict[str, Dict[str, Any]] = {}

        def _add_pattern(key: str, mask: pd.Series, label: str) -> None:
            count = int(mask.sum())
            if "No." in df.columns:
                nos = self._extract_no_list(df.loc[mask, "No."])
            else:
                nos = []
            patterns[key] = {"mask": mask, "count": count, "nos": nos}
            self.logger.info("[PATTERN] %s: %d records", label, count)

        # Register patterns
        _add_pattern("inc1", inc1, "Include (1) R1 = R2 = R4 = include")
        _add_pattern("inc2", inc2, "Include (2) R1 = R2 = unsure, R3 = R4 = include")
        _add_pattern("inc3", inc3, "Include (3) R1 ≠ R2, R3 = R4 = include")

        _add_pattern("uns4", uns4, "Unsure (4) R4 = unsure, Notes_R4 = 'no access'")
        _add_pattern("uns5", uns5, "Unsure (5) R4 = unsure, Notes_R4 = 'chinese article; paid access'")

        _add_pattern("exc6", exc6, "Exclude (6) R1 = R2 = R4 = exclude")
        _add_pattern("exc7", exc7, "Exclude (7) R1 = R2 = unsure, R3 = R4 = exclude")
        _add_pattern("exc8", exc8, "Exclude (8) R1 ≠ R2, R3 = R4 = exclude")

        return patterns

    # ---------------------------------------------------------------------
    # Export of final included studies
    # ---------------------------------------------------------------------

    @staticmethod
    def _reorder_columns_for_output(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally define the final column order for export.

        Currently returns the DataFrame unchanged, preserving the original column order.
        """
        return df

    def write_final_included(self, df: pd.DataFrame, final_mask: pd.Series) -> int:
        """
        Export finally included studies (Include patterns 1–3) to an Excel file.

        :param df: Original DataFrame.
        :param final_mask: Boolean mask indicating final inclusion.
        :return: Number of included rows.
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

    # ---------------------------------------------------------------------
    # Summary TXT report
    # ---------------------------------------------------------------------

    def _build_summary_text(
        self,
        total_count: int,
        patterns: Dict[str, Dict[str, Any]],
        included_count: int,
        unsure_count: int,
        exclude_count: int,
        unmatched_count: int,
    ) -> str:
        """
        Build the TXT summary report text.

        The summary groups patterns into Include / Unsure / Exclude sections,
        reports counts, and lists No. values where appropriate.
        """
        lines: List[str] = []

        lines.append("=" * 70)
        lines.append("Stage 1 R1/R2/R3/R4 – Final Inclusion Summary")
        lines.append("=" * 70)
        lines.append("")

        # 1. Overall counts
        lines.append("1. Overall counts")
        lines.append(f"- Total records                    : {total_count}")
        lines.append(f"- Final included (patterns 1–3)    : {included_count}")
        lines.append(f"- Unsure only (patterns 4–5)       : {unsure_count}")
        lines.append(f"- Excluded (patterns 6–8)          : {exclude_count}")
        lines.append(f"- Unmatched (no pattern hit)       : {unmatched_count}")
        lines.append("")

        def _fmt_nos(label: str, nos: List[str]) -> None:
            """Append a nicely formatted No. list block, if any."""
            if not nos:
                return
            lines.append(f"  · {label} No. list:")
            lines.append(f"    {nos}")
            lines.append("")

        # 2. Include patterns (1–3)
        lines.append("2. Include patterns (final included)")
        lines.append(f"- (1) R1 = R2 = R4 = include                     : {patterns['inc1']['count']}")
        _fmt_nos("(1)", patterns["inc1"]["nos"])

        lines.append(f"- (2) R1 = R2 = unsure, R3 = R4 = include        : {patterns['inc2']['count']}")
        _fmt_nos("(2)", patterns["inc2"]["nos"])

        lines.append(f"- (3) R1 ≠ R2, R3 = R4 = include                 : {patterns['inc3']['count']}")
        _fmt_nos("(3)", patterns["inc3"]["nos"])

        # 3. Unsure patterns (4–5)
        lines.append("3. Unsure patterns (not counted as final included)")
        lines.append(
            f"- (4) R4 = unsure, Notes_R4 = 'no access'        : {patterns['uns4']['count']}"
        )
        _fmt_nos("(4)", patterns["uns4"]["nos"])

        lines.append(
            f"- (5) R4 = unsure, Notes_R4 = 'chinese article; paid access' : {patterns['uns5']['count']}"
        )
        _fmt_nos("(5)", patterns["uns5"]["nos"])

        # 4. Exclude patterns (6–8)
        lines.append("4. Exclude patterns")
        lines.append(f"- (6) R1 = R2 = R4 = exclude                     : {patterns['exc6']['count']}")
        _fmt_nos("(6)", patterns["exc6"]["nos"])

        lines.append(
            f"- (7) R1 = R2 = unsure, R3 = R4 = exclude        : {patterns['exc7']['count']}"
        )
        _fmt_nos("(7)", patterns["exc7"]["nos"])

        lines.append(f"- (8) R1 ≠ R2, R3 = R4 = exclude                 : {patterns['exc8']['count']}")
        _fmt_nos("(8)", patterns["exc8"]["nos"])

        lines.append("=" * 70)
        return "\n".join(lines)

    def _write_summary(self, text: str) -> None:
        """
        Write the summary text to a TXT file.
        """
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(text, encoding="utf-8")
        self.logger.info(
            "[SUMMARY] Final inclusion summary written to TXT report: %s",
            self.summary_path,
        )

    # ---------------------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the full Stage 1 final inclusion pipeline:

        1. Load Stage 1 R1/R2/R3/R4 results.
        2. Classify records into 8 patterns (1–3 include, 4–5 unsure, 6–8 exclude).
        3. Define final inclusion set as patterns 1–3.
        4. Export final included studies to Excel.
        5. Generate and write a structured TXT summary report.
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
        )
        unsure_mask = patterns["uns4"]["mask"] | patterns["uns5"]["mask"]
        exclude_mask = (
            patterns["exc6"]["mask"]
            | patterns["exc7"]["mask"]
            | patterns["exc8"]["mask"]
        )
        unmatched_mask = ~(include_mask | unsure_mask | exclude_mask)

        included_count = int(include_mask.sum())
        unsure_count = int(unsure_mask.sum())
        exclude_count = int(exclude_mask.sum())
        unmatched_count = int(unmatched_mask.sum())

        self.logger.info("[TOTAL] Final included (patterns 1–3): %d", included_count)
        self.logger.info("[TOTAL] Unsure only (patterns 4–5)   : %d", unsure_count)
        self.logger.info("[TOTAL] Excluded (patterns 6–8)      : %d", exclude_count)
        self.logger.info("[TOTAL] Unmatched                     : %d", unmatched_count)

        # 3. Export final included studies
        self.write_final_included(df, include_mask)

        # 4. Build and write TXT summary
        summary_text = self._build_summary_text(
            total_count=total_count,
            patterns=patterns,
            included_count=included_count,
            unsure_count=unsure_count,
            exclude_count=exclude_count,
            unmatched_count=unmatched_count,
        )
        self._write_summary(summary_text)


def main() -> None:
    """
    Main entry point: run Stage 1 R1/R2/R3/R4 final inclusion
    classification and export.
    """
    logger = setup_logger(verbose=True)
    logger.info("[MAIN] Stage 1 R1/R2/R3/R4 final inclusion summary started")

    analyzer = Stage1R1R2R3R4InclusionAnalyzer(logger=logger)
    analyzer.run()

    logger.info("[MAIN] Stage 1 R1/R2/R3/R4 final inclusion summary finished")


if __name__ == "__main__":
    main()