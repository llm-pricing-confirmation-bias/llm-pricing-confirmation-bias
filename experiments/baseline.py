"""
Baseline pipeline  —  CONTROL, single turn, no anchor.
======================================================

This is the foundation of the whole study: the model is shown the product
profile ONCE and returns a single price, with no authority anchor and no
"Please also note that ..." injection line at all. It reproduces the CONTROL
arm of the original ``anchor_assert.py`` exactly.

Besides the ``control`` runner, this module owns the **single-turn machinery**
(``TrialSpec`` / ``TrialResult`` / :func:`run_single_turn_trial` and the
CSV/summary writers) that ``static.py`` reuses for anchored single-turn runs.
Both write to the SAME results directory / JSONL so that the per-cell summary
can compute an anchoring index against each model's own CONTROL median.

Usage
-----
    python main.py baseline control --runs 30
    python main.py baseline control --runs 300
    python main.py baseline summarize
    python main.py baseline show-prompt
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median
from typing import Optional

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


DEFAULT_RUNS_PER_CELL = 300
DEFAULT_OUTPUT_DIR    = str(DEFAULT_SINGLE_TURN_RUN_DIR)
STEM                  = "single_turn"


# ══════════════════════════════════════════════════════════════════════════════
#  TRIAL SPEC + RESULT  (shared by baseline + static)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrialSpec:
    model:        str
    condition_id: str               # "CONTROL" for baseline
    assertion:    str               # "na" for control
    anchor_level: str               # "NA" for control
    anchor_value: Optional[float]
    run_index:    int

    @property
    def trial_id(self) -> str:
        return "|".join([
            self.model, PROFILE_ID, self.condition_id,
            self.assertion, self.anchor_level, f"{self.run_index:04d}",
        ])


@dataclass
class TrialResult:
    trial_id:             str
    model:                str
    profile_id:           str
    condition_id:         str
    assertion:            str
    anchor_level:         str
    anchor_value:         Optional[float]
    currency:             str
    run_index:            int
    injection_text:       str             = ""
    price_recommendation: Optional[float] = None
    confidence:           Optional[int]   = None
    justification:        Optional[str]   = None
    reasoning_words:      Optional[int]   = None
    raw_text:             Optional[str]   = None
    parse_error:          Optional[str]   = None
    api_error:            Optional[str]   = None
    attempts:             int             = 0
    latency_s:            Optional[float] = None
    temperature:          Optional[float] = None
    timestamp:            str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-TURN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def run_single_turn_trial(session, spec: TrialSpec, temperature, max_tokens) -> TrialResult:
    injection = build_injection(spec.condition_id, spec.assertion, spec.anchor_value)
    prompt = build_user_prompt(injection)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    result = TrialResult(
        trial_id=spec.trial_id, model=spec.model, profile_id=PROFILE_ID,
        condition_id=spec.condition_id, assertion=spec.assertion,
        anchor_level=spec.anchor_level, anchor_value=spec.anchor_value,
        currency=CURRENCY, run_index=spec.run_index,
        injection_text=injection, temperature=temperature,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        result.attempts = attempt
        t0 = time.monotonic()
        content, api_err, retryable = await call_api(
            session, spec.model, messages, temperature, max_tokens)
        result.latency_s = round(time.monotonic() - t0, 2)
        if api_err:
            result.api_error = api_err
            if retryable and attempt < MAX_RETRIES:
                await asyncio.sleep(min(60, 2 ** attempt))
                continue
            return result
        result.api_error = None
        result.raw_text = content
        price, conf, just, parse_err = parse_response(content)
        if parse_err:
            result.parse_error = parse_err
            if attempt < MAX_RETRIES:
                continue
            return result
        result.parse_error = None
        result.price_recommendation = price
        result.confidence = conf
        result.justification = just
        result.reasoning_words = compute_reasoning_words(content)
        return result
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SPEC BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_control_specs(models, runs):
    return [TrialSpec(m, "CONTROL", "na", "NA", None, i)
            for m in models for i in range(1, runs + 1)]


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT  (shared by baseline + static)
# ══════════════════════════════════════════════════════════════════════════════

CSV_COLS = [
    "trial_id", "model", "profile_id", "condition_id", "assertion",
    "anchor_level", "anchor_value", "currency", "run_index",
    "price_recommendation", "confidence", "justification", "reasoning_words",
    "injection_text", "parse_error", "api_error", "attempts", "latency_s",
    "temperature", "timestamp",
]


def write_trials_csv(records, path, include_raw):
    records = sorted(records, key=lambda r: (
        r.get("model", ""), r.get("condition_id", ""), r.get("assertion", ""),
        r.get("anchor_level", ""), r.get("run_index", 0)))
    write_dict_csv(records, path, CSV_COLS, include_raw=include_raw)


def write_summary_csv(records, path):
    usable = [r for r in records if r.get("price_recommendation") is not None]
    ctrl = {}
    for r in usable:
        if r["condition_id"] == "CONTROL":
            ctrl.setdefault(r["model"], []).append(r["price_recommendation"])
    ctrl = {m: median(v) for m, v in ctrl.items()}
    cells = {}
    for r in usable:
        key = (r["model"], r["condition_id"], r.get("assertion", "na"),
               r.get("anchor_level", "NA"), r.get("anchor_value"))
        cells.setdefault(key, []).append(r["price_recommendation"])
    rows = []
    for (model, cond, a, lvl, aval), vals in cells.items():
        s = summarize_vals(vals)
        cm = ctrl.get(model)
        ai = None
        if cond != "CONTROL" and cm is not None and aval is not None:
            denom = aval - cm
            if abs(denom) > 1e-9:
                ai = round((s["median"] - cm) / denom, 4)
        rows.append({
            "model": model, "condition_id": cond, "assertion": a,
            "anchor_level": lvl, "anchor_value": aval,
            "control_median": round(cm, 2) if cm is not None else None,
            "anchoring_index": ai, **s,
        })
    rows.sort(key=lambda x: (x["model"], x["condition_id"], x["assertion"], str(x["anchor_level"])))
    cols = ["model", "condition_id", "assertion", "anchor_level", "anchor_value",
            "control_median", "anchoring_index", "n", "mean", "median", "sd",
            "cv", "min", "max", "p25", "p75"]
    write_dict_csv(rows, path, cols)
    return rows


def rebuild_outputs(output_dir, include_raw):
    jsonl_file, csv_file, summary_file = paths(output_dir, STEM)
    records = dedup_records(load_records(jsonl_file))
    write_trials_csv(records, csv_file, include_raw)
    summary_rows = write_summary_csv(records, summary_file)
    return records, summary_rows


def print_report(records, summary_rows, elapsed=None):
    usable = [r for r in records if r.get("price_recommendation") is not None]
    bar = "\u2500" * 78
    print(f"\n{bar}\n  CONFIRMATION BIAS STUDY \u2014 SINGLE-TURN RESULTS\n{bar}")
    line = f"  Total trials: {len(records)}  |  Usable: {len(usable)}  |  Failed: {len(records) - len(usable)}"
    if elapsed is not None:
        line += f"  |  Elapsed: {elapsed:.0f}s"
    print(line)
    if summary_rows:
        print(f"\n  {'model':<32}{'cond':<17}{'assert':<9}{'lvl':<5}"
              f"{'n':>5}{'median':>9}{'sd':>8}{'AI':>8}")
        for r in summary_rows:
            ai = "" if r["anchoring_index"] is None else f"{r['anchoring_index']:.2f}"
            print(f"  {r['model']:<32}{r['condition_id']:<17}{r['assertion']:<9}"
                  f"{str(r['anchor_level']):<5}{r['n']:>5}{r['median']:>9}{r['sd']:>8}{ai:>8}")
    print(f"\n{bar}\n  AI = anchoring index vs that model's CONTROL median\n{bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  RUN ORCHESTRATION  (shared by baseline + static)
# ══════════════════════════════════════════════════════════════════════════════

def confirm_and_run(specs, args, label):
    jsonl_file, csv_file, summary_file = paths(args.output_dir, STEM)
    print(f"\nConfirmation Bias study \u2014 {label}")
    print(f"  Models:      {args.models}")
    cells = len(set((s.model, s.condition_id, s.assertion, s.anchor_level) for s in specs))
    print(f"  Cells:       {cells}")
    print(f"  Total specs: {len(specs)}")
    print(f"  Temperature: {args.temperature}   Max tokens: {args.max_tokens}")
    print(f"  Workers:     {args.concurrency}")
    print(f"  Output dir:  {args.output_dir}")
    todo, already = filter_todo(specs, jsonl_file)
    if already:
        print(f"  Resume:      {already} usable trials on disk; {len(todo)} left to run.")
    if args.dry_run:
        print("\n  --dry-run set: no API calls made.")
        return
    if not todo:
        print("\nNothing new to run \u2014 rebuilding CSVs from existing data.")
        records, summary_rows = rebuild_outputs(args.output_dir, args.include_raw)
        print_report(records, summary_rows)
        return
    if not API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not found. Put it in your .env file.")
    if not args.yes:
        ans = input(f"\nLaunch {len(todo)} API calls? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            sys.exit("Aborted. (Use --yes to skip this gate.)")
    _, elapsed = asyncio.run(run_batch(
        todo, jsonl_file, args.concurrency, args.temperature, args.max_tokens,
        run_single_turn_trial))
    records, summary_rows = rebuild_outputs(args.output_dir, args.include_raw)
    print(f"\n  Saved: {jsonl_file}\n         {csv_file}\n         {summary_file}")
    print_report(records, summary_rows, elapsed)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="CONTROL baseline (single turn, no anchor).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("control", help="Run CONTROL baseline (no anchor).")
    pc.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_CELL)
    pc.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    add_common_run_args(pc)

    ps = sub.add_parser("summarize", help="Rebuild CSVs from JSONL (no API).")
    ps.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ps.add_argument("--include-raw", action="store_true")

    pp = sub.add_parser("show-prompt", help="Print the exact CONTROL prompt and exit.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "control":
        confirm_and_run(build_control_specs(args.models, args.runs), args, "CONTROL baseline")

    elif args.command == "summarize":
        records, summary_rows = rebuild_outputs(args.output_dir, args.include_raw)
        print_report(records, summary_rows)
        jsonl_file, csv_file, summary_file = paths(args.output_dir, STEM)
        print(f"  Rebuilt: {csv_file}\n           {summary_file}\n")

    elif args.command == "show-prompt":
        print("===== SYSTEM PROMPT =====")
        print(SYSTEM_PROMPT)
        print("\n===== USER PROMPT (CONTROL) =====")
        print(build_user_prompt(""))


if __name__ == "__main__":
    main()
