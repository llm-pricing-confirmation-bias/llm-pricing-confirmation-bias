"""Canonical filesystem paths for this package."""

from __future__ import annotations

from pathlib import Path

# config/paths.py → package root is one level up
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PACKAGE_ROOT / "results"
REPORTS_DIR = PACKAGE_ROOT / "reports"
RUNS_DIR = PACKAGE_ROOT / "results_runs"

SINGLE_TURN_DIR = RESULTS_DIR / "single_turn"
ITERATIVE_DIR = RESULTS_DIR / "iterative"
ITERATIVE_SOURCE_DIR = ITERATIVE_DIR / "source_per_round"
PROMPT_VARIANTS_DIR = RESULTS_DIR / "prompt_variants"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
PER_ROUND_ANALYSIS_DIR = ITERATIVE_DIR / "per_round_analysis"
BIMODALITY_PLOTS_DIR = PER_ROUND_ANALYSIS_DIR / "bimodality_plots"

# Fresh API outputs (gitignored). Curated paper artifacts live under results/.
DEFAULT_SINGLE_TURN_RUN_DIR = RUNS_DIR / "single_turn"
DEFAULT_ITERATIVE_RUN_DIR = RUNS_DIR / "iterative"
DEFAULT_BASELINE_SOURCE = DEFAULT_SINGLE_TURN_RUN_DIR / "single_turn_trials.jsonl"

REPORT_TABLES_TEX = REPORTS_DIR / "report_tables.tex"
