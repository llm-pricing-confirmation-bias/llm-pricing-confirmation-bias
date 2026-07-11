"""
Static prompting pipeline  —  anchor injected UP FRONT, single turn.
====================================================================

Same single mechanic as the baseline, except the authority anchor is placed
inside the very first (and only) prompt, on the "Please also note that ..."
line, before the model has committed to any price. This reproduces the
``anchor`` arm of the original ``anchor_assert.py``.

It reuses the single-turn machinery from :mod:`baseline` (``TrialSpec``,
``run_single_turn_trial``, the CSV/summary writers and the run orchestration)
and writes to the SAME results directory / JSONL as the baseline, so the
per-cell summary's anchoring index is computed against each model's own
CONTROL median.

Usage
-----
    python main.py static anchor --conditions MCKINSEY,INTERN \
        --assertions standard --anchor-levels LOW,MID,HIGH --runs 300

    python main.py static anchor --conditions ALL --assertions ALL \
        --anchor-levels MID --runs 300

    python main.py static summarize
    python main.py static show-prompt --condition MCKINSEY --assertion strong --level HIGH
"""

import argparse
import sys

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.anchors import get_anchor_value, require_anchor_values
from config.prompts import (
    ALL_CONDITIONS,
    INJECTIONS,
    RECONSTRUCTED_FIRST_TURN,
    SYSTEM_PROMPT,
)
from config.settings import (
    ANCHOR_LEVEL_KEYS,
    API_KEY,
    CONTROL_TOKEN,
    CURRENCY,
    DEFAULT_MODELS,
    EPS,
    MAX_RETRIES,
    NONE_TOKEN,
    PROFILE_ID,
)
from config.paths import (
    DEFAULT_BASELINE_SOURCE as _BASELINE_SRC,
    DEFAULT_ITERATIVE_RUN_DIR,
    DEFAULT_SINGLE_TURN_RUN_DIR,
)
from services.batch import run_batch
from services.openrouter import call_api, call_with_retries
from services.prompt_builder import (
    build_injection,
    build_round_prompt,
    build_user_prompt,
)
from utils.cli import (
    add_common_run_args,
    resolve_assertions,
    resolve_conditions,
    resolve_levels,
)
from utils.io import (
    dedup_records,
    filter_todo,
    load_records,
    load_source_records,
    paths,
    write_dict_csv,
)
from utils.parse import compute_reasoning_words, parse_response
from utils.stats import summarize_vals

import experiments.baseline as B

DEFAULT_OUTPUT_DIR    = B.DEFAULT_OUTPUT_DIR   # share output dir with baseline
DEFAULT_RUNS_PER_CELL = 300
DEFAULT_ANCHOR_LEVELS = "MID"


# ══════════════════════════════════════════════════════════════════════════════
#  SPEC BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_anchor_specs(models, conditions, assertions, levels, runs):
    specs = []
    for m in models:
        for cond in conditions:
            for a in assertions:
                for lvl in levels:
                    val = float(get_anchor_value(lvl, m))
                    for i in range(1, runs + 1):
                        specs.append(B.TrialSpec(m, cond, a, lvl, val, i))
    return specs


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="Static prompting (anchor up front, single turn).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("anchor", help="Run anchor conditions.")
    pa.add_argument("--conditions", default="ALL",
                    help=f"Comma list or ALL. Options: {', '.join(ALL_CONDITIONS)}.")
    pa.add_argument("--assertions", default="standard",
                    help="Comma list of weak/standard/strong or ALL.")
    pa.add_argument("--anchor-levels", default=DEFAULT_ANCHOR_LEVELS,
                    help=f"Comma list. Options: {', '.join(ANCHOR_LEVEL_KEYS)}.")
    pa.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_CELL)
    pa.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    add_common_run_args(pa)

    ps = sub.add_parser("summarize", help="Rebuild CSVs from JSONL (no API).")
    ps.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ps.add_argument("--include-raw", action="store_true")

    pp = sub.add_parser("show-prompt", help="Print an exact rendered prompt and exit.")
    pp.add_argument("--condition", default="MCKINSEY")
    pp.add_argument("--assertion", default="standard")
    pp.add_argument("--level", default="MID")
    pp.add_argument("--model", default=None,
                    help="Resolve the anchor value for this model slug (defaults to first model).")
    pp.add_argument("--anchor-value", type=float, default=None,
                    help="Override the dollar value just for previewing.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "anchor":
        conditions = resolve_conditions(args.conditions)
        assertions = resolve_assertions(args.assertions)
        levels = resolve_levels(args.anchor_levels)
        require_anchor_values(args.models, levels)
        specs = build_anchor_specs(args.models, conditions, assertions, levels, args.runs)

        def _show(lvl):
            return ", ".join(f"{m.split('/')[-1]}={get_anchor_value(lvl, m)}" for m in args.models)
        label = ("STATIC ANCHOR  conditions={}  assertions={}  ".format(conditions, assertions)
                 + "  ".join(f"{lvl}:[{_show(lvl)}]" for lvl in levels))
        B.confirm_and_run(specs, args, label)

    elif args.command == "summarize":
        records, summary_rows = B.rebuild_outputs(args.output_dir, args.include_raw)
        B.print_report(records, summary_rows)
        _, csv_file, summary_file = paths(args.output_dir, B.STEM)
        print(f"  Rebuilt: {csv_file}\n           {summary_file}\n")

    elif args.command == "show-prompt":
        cond = args.condition.upper().replace(" ", "_").replace("-", "_")
        if cond in (CONTROL_TOKEN, NONE_TOKEN):
            inj = ""
        else:
            if cond not in INJECTIONS:
                sys.exit(f"ERROR: unknown condition '{args.condition}'.")
            val = args.anchor_value
            if val is None:
                model = args.model or DEFAULT_MODELS[0]
                val = get_anchor_value(args.level.upper(), model)
            if val is None:
                val = 65  # preview placeholder only
                print(f"[preview note] no value for {args.level.upper()}; using $65 placeholder]\n")
            inj = build_injection(cond, args.assertion.lower(), val)
        print("===== SYSTEM PROMPT =====")
        print(SYSTEM_PROMPT)
        print("\n===== USER PROMPT =====")
        print(build_user_prompt(inj))


if __name__ == "__main__":
    main()
