#!/usr/bin/env python3
"""
Unified entry point for *Confirmation Bias in LLM Pricing Recommendations*.

Usage (from this directory)::

    python main.py --help
    python main.py baseline control --runs 30
    python main.py static anchor --conditions MCKINSEY --anchor-levels MID --runs 30
    python main.py iterative run --conditions MCKINSEY --anchor-levels MID --max-rounds 5
    python main.py consolidate
    python main.py dump-report
    python main.py analyze-rounds
    python main.py plot-bimodality
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMMANDS: dict[str, tuple[str, str]] = {
    "baseline": (
        "experiments.baseline",
        "CONTROL single-turn baseline (no anchor)",
    ),
    "static": (
        "experiments.static",
        "Static prompting — anchor injected up front",
    ),
    "iterative": (
        "experiments.iterative",
        "Iterative reconsideration loop",
    ),
    "consolidate": (
        "analysis.consolidate_iterative",
        "Flatten source_per_round/ → iterative_all_rounds.*",
    ),
    "dump-report": (
        "analysis.dump_report",
        "Emit reports/report_tables.tex from curated results/",
    ),
    "analyze-rounds": (
        "analysis.analyze_iterative_rounds",
        "Per-round iterative CSVs, plots, and wording note",
    ),
    "plot-bimodality": (
        "analysis.plot_bimodality",
        "Publication-ready bimodality histograms",
    ),
}


def _print_help() -> None:
    print("Confirmation Bias in LLM Pricing Recommendations — single entry point.\n")
    print("usage: python main.py <command> [<args>]\n")
    print("commands:")
    width = max(len(k) for k in COMMANDS)
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<{width}}  {desc}")
    print("\nexamples:")
    print("  python main.py baseline show-prompt")
    print("  python main.py baseline control --runs 30 --yes")
    print("  python main.py static anchor --help")
    print("  python main.py dump-report")
    print("  python main.py plot-bimodality")
    print("  python main.py iterative run --conditions MCKINSEY --anchor-levels MID --max-rounds 5")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command!r}\n", file=sys.stderr)
        _print_help()
        return 2

    forwarded = argv[1:]
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        print(f"ERROR: {module_name} has no main()", file=sys.stderr)
        return 1

    sys.argv = [f"main.py {command}"] + forwarded
    result = module.main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
