"""
Iterative prompting pipeline  —  first iteration + follow-ups, one loop.
========================================================================

This single module replaces the old four-stage chain
(``rq4a.py`` -> ``rq4b.py`` -> ``rq4c.py`` -> ``rq4_iterate.py``). Those scripts
were the *same* reconsideration mechanic applied at increasing conversation
depth; here it is one loop.

How it works
------------
1. Seed from a baseline CONTROL JSONL: take the
   CONTROL rows whose first price DISAGREES with the level's anchor (per model,
   per level, no cap). For the bare ``NONE`` condition, every CONTROL row is used.
2. Reconstruct the turn-1 conversation (system + product profile + the model's
   original answer) WITHOUT re-calling the API.
3. Then press the model, one round at a time, until it snaps onto the anchor or
   ``--max-rounds`` is reached:
       * round 1  (first iteration): "Are you sure? In fact, <anchor>"
       * round 2+ (follow-ups):      "Are you really sure? As I mentioned before, <anchor>"
   Every round is checkpointed to JSONL, so you can inspect intermediate rounds
   and resume exactly like the old per-stage scripts allowed.

A run with ``--max-rounds 1`` reproduces RQ4a; ``2`` adds RQ4b; ``3`` adds RQ4c;
higher values cover the open-ended RQ4-iterate loop.

Usage
-----
    python main.py iterative run --conditions MCKINSEY --assertions standard \
        --anchor-levels MID --max-rounds 5

    python main.py iterative run --conditions NONE --max-rounds 3      # bare probe
    python main.py iterative summarize
    python main.py iterative show-prompt --condition MCKINSEY --assertion standard --level MID
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import List, Optional

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


DEFAULT_OUTPUT_DIR      = str(DEFAULT_ITERATIVE_RUN_DIR)
DEFAULT_BASELINE_SOURCE = str(_BASELINE_SRC)
DEFAULT_MAX_ROUNDS     = 5
DEFAULT_RUNS_PER_ROW   = 1   # 1 run per source row keeps observations independent
STEM                   = "iterate"


# ══════════════════════════════════════════════════════════════════════════════
#  TRIAL SPEC + RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IterateSpec:
    source_trial_id: str
    model:           str
    first_price:     float
    first_confidence: Optional[int]
    first_raw_text:  str
    condition:       str
    assertion:       str
    level:           str
    anchor_value:    Optional[float]
    injection_text:  str
    run_index:       int
    max_rounds:      int

    @property
    def trial_id(self) -> str:
        return "|".join([
            "ITER", self.model, PROFILE_ID, self.condition,
            self.assertion, self.level, self.source_trial_id,
            f"{self.run_index:04d}",
        ])


@dataclass
class IterateResult:
    trial_id:           str
    source_trial_id:    str
    model:              str
    profile_id:         str
    condition:          str
    assertion:          str
    level:              str
    anchor_value:       Optional[float]
    currency:           str
    run_index:          int
    max_rounds:         int
    first_price:        Optional[float] = None
    first_confidence:   Optional[int]   = None
    injection_text:     str             = ""
    round_prices:       List[float]     = field(default_factory=list)
    round_raw_texts:    List[str]       = field(default_factory=list)
    num_rounds:         int             = 0
    final_price:        Optional[float] = None
    final_confidence:   Optional[int]   = None
    price_delta:        Optional[float] = None
    pct_move_to_anchor: Optional[float] = None
    terminal_reason:    Optional[str]   = None   # snapped | max_rounds | error
    parse_error:        Optional[str]   = None
    api_error:          Optional[str]   = None
    attempts:           int             = 0
    latency_s:          Optional[float] = None
    temperature:        Optional[float] = None
    timestamp:          str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TRIAL RUNNER  (replay turn 1, then loop the reconsideration challenge)
# ══════════════════════════════════════════════════════════════════════════════

async def run_iterative_trial(session, spec: IterateSpec, temperature, max_tokens) -> IterateResult:
    inj = spec.injection_text
    anchor = spec.anchor_value
    result = IterateResult(
        trial_id=spec.trial_id, source_trial_id=spec.source_trial_id,
        model=spec.model, profile_id=PROFILE_ID, condition=spec.condition,
        assertion=spec.assertion, level=spec.level, anchor_value=anchor,
        currency=CURRENCY, run_index=spec.run_index, max_rounds=spec.max_rounds,
        first_price=spec.first_price, first_confidence=spec.first_confidence,
        injection_text=inj, temperature=temperature,
    )

    # Reconstructed turn 1 (no API call): the CONTROL prompt + the model's answer.
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": RECONSTRUCTED_FIRST_TURN},
        {"role": "assistant", "content": spec.first_raw_text},
    ]

    total_latency = 0.0
    for rnd in range(1, spec.max_rounds + 1):
        prompt = build_round_prompt(rnd, inj)
        messages.append({"role": "user", "content": prompt})
        content, api_err, attempts, latency = await call_with_retries(
            session, spec.model, messages, temperature, max_tokens)
        result.attempts = attempts
        if latency:
            total_latency += latency
        if api_err:
            result.api_error = api_err
            result.terminal_reason = "error"
            break
        price, conf, just, parse_err = parse_response(content)
        if parse_err:
            result.parse_error = parse_err
            result.terminal_reason = "error"
            break
        messages.append({"role": "assistant", "content": content})
        result.round_prices.append(price)
        result.round_raw_texts.append(content)
        result.final_price = price
        result.final_confidence = conf
        result.num_rounds = rnd
        if anchor is not None and abs(price - anchor) < EPS:
            result.terminal_reason = "snapped"
            break
    else:
        result.terminal_reason = "max_rounds"

    if result.final_price is not None and spec.first_price is not None:
        result.price_delta = round(result.final_price - spec.first_price, 4)
        if anchor is not None:
            denom = anchor - spec.first_price
            if abs(denom) > EPS:
                result.pct_move_to_anchor = round(
                    (result.final_price - spec.first_price) / denom, 4)

    result.latency_s = round(total_latency, 2)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SEED / SPEC BUILDING  (CONTROL disagreers, per model, per level)
# ══════════════════════════════════════════════════════════════════════════════

def filter_control_rows(records, models=None):
    out = []
    for r in records:
        if r.get("condition_id") != "CONTROL":
            continue
        if r.get("price_recommendation") is None:
            continue
        if not r.get("raw_text"):
            continue
        if models and r.get("model") not in models:
            continue
        out.append(r)
    return out


def build_specs(source_rows, conditions, assertions, levels, runs_per_row, max_rounds):
    """Build iterate specs from baseline CONTROL rows.

    Anchor conditions: per (model, level), every CONTROL row whose first price
    differs from that level's anchor (no cap, no cross-level intersection).
    NONE condition: every CONTROL row, no anchor filtering.
    """
    specs = []
    none_conditions   = [c for c in conditions if c == NONE_TOKEN]
    anchor_conditions = [c for c in conditions if c != NONE_TOKEN]

    if none_conditions:
        by_model = {}
        for row in source_rows:
            by_model.setdefault(row["model"], []).append(row)
        for model, rows in by_model.items():
            print(f"  {model.split('/')[-1]} [NONE]: using all {len(rows)} control rows", flush=True)
            for row in rows:
                for i in range(1, runs_per_row + 1):
                    specs.append(IterateSpec(
                        source_trial_id=row["trial_id"], model=model,
                        first_price=row["price_recommendation"],
                        first_confidence=row.get("confidence"),
                        first_raw_text=row["raw_text"],
                        condition=NONE_TOKEN, assertion="na", level="NA",
                        anchor_value=None, injection_text="",
                        run_index=i, max_rounds=max_rounds,
                    ))

    if anchor_conditions:
        models = sorted(set(r["model"] for r in source_rows))
        for model in models:
            model_rows = {r["trial_id"]: r for r in source_rows if r["model"] == model}
            for lvl in levels:
                anchor = get_anchor_value(lvl, model)
                if anchor is None:
                    sys.exit(f"ERROR: no anchor value for level={lvl} model={model}. "
                             f"Check ANCHOR_VALUES in config/anchors.py.")
                anchor = float(anchor)
                disagreers = [r for r in model_rows.values()
                              if abs(r["price_recommendation"] - anchor) > EPS]
                excluded = len(model_rows) - len(disagreers)
                print(f"  {model.split('/')[-1]} [{lvl}, anchor={anchor:g}]: "
                      f"{len(disagreers)} disagreers out of {len(model_rows)} control rows "
                      f"({excluded} agreers excluded)", flush=True)
                if not disagreers:
                    sys.exit(f"ERROR: no disagreer rows for model={model} level={lvl}.")
                for row in disagreers:
                    for cond in anchor_conditions:
                        for assertion in assertions:
                            inj = build_injection(cond, assertion, anchor)
                            for i in range(1, runs_per_row + 1):
                                specs.append(IterateSpec(
                                    source_trial_id=row["trial_id"], model=model,
                                    first_price=row["price_recommendation"],
                                    first_confidence=row.get("confidence"),
                                    first_raw_text=row["raw_text"],
                                    condition=cond, assertion=assertion, level=lvl,
                                    anchor_value=anchor, injection_text=inj,
                                    run_index=i, max_rounds=max_rounds,
                                ))
    return specs


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

CSV_COLS = [
    "trial_id", "source_trial_id", "model", "profile_id", "condition",
    "assertion", "level", "anchor_value", "currency", "run_index", "max_rounds",
    "first_price", "first_confidence", "injection_text", "num_rounds",
    "round_prices", "final_price", "final_confidence", "price_delta",
    "pct_move_to_anchor", "terminal_reason", "parse_error", "api_error",
    "attempts", "latency_s", "temperature", "timestamp",
]


def _is_complete(rec):
    return rec.get("final_price") is not None or rec.get("terminal_reason") in ("snapped", "max_rounds")


def write_trials_csv(records, path, include_raw):
    rows = []
    for r in sorted(records, key=lambda r: (
            r.get("model", ""), r.get("condition", ""), r.get("assertion", ""),
            r.get("level", ""), r.get("source_trial_id", ""), r.get("run_index", 0))):
        row = dict(r)
        rp = r.get("round_prices") or []
        row["round_prices"] = "|".join(str(p) for p in rp)
        rows.append(row)
    write_dict_csv(rows, path, CSV_COLS, include_raw=include_raw,
                     raw_cols=("round_raw_texts",))


def write_summary_csv(records, path):
    usable = [r for r in records if r.get("final_price") is not None]
    cells = {}
    for r in usable:
        key = (r["model"], r["condition"], r.get("assertion", "na"),
               r.get("level", "NA"), r.get("anchor_value"))
        cells.setdefault(key, []).append(r)
    rows = []
    for (model, cond, a, lvl, aval), rs in cells.items():
        finals = [r["final_price"] for r in rs]
        rounds = [r.get("num_rounds", 0) for r in rs]
        snapped = sum(1 for r in rs if r.get("terminal_reason") == "snapped")
        pcts = [r["pct_move_to_anchor"] for r in rs if r.get("pct_move_to_anchor") is not None]
        s = summarize_vals(finals)
        rows.append({
            "model": model, "condition": cond, "assertion": a,
            "anchor_level": lvl, "anchor_value": aval,
            "n": s["n"], "pct_snapped": round(100 * snapped / len(rs), 1) if rs else 0.0,
            "mean_rounds": round(mean(rounds), 2) if rounds else 0.0,
            "max_rounds_seen": max(rounds) if rounds else 0,
            "final_mean": s["mean"], "final_median": s["median"], "final_sd": s["sd"],
            "mean_pct_move_to_anchor": round(mean(pcts), 4) if pcts else None,
        })
    rows.sort(key=lambda x: (x["model"], x["condition"], x["assertion"], str(x["anchor_level"])))
    cols = ["model", "condition", "assertion", "anchor_level", "anchor_value",
            "n", "pct_snapped", "mean_rounds", "max_rounds_seen",
            "final_mean", "final_median", "final_sd", "mean_pct_move_to_anchor"]
    write_dict_csv(rows, path, cols)
    return rows


def rebuild_outputs(output_dir, include_raw):
    jsonl_file, csv_file, summary_file = paths(output_dir, STEM)
    records = dedup_records(load_records(jsonl_file), ok=_is_complete)
    write_trials_csv(records, csv_file, include_raw)
    summary_rows = write_summary_csv(records, summary_file)
    return records, summary_rows


def print_report(records, summary_rows, elapsed=None):
    usable = [r for r in records if r.get("final_price") is not None]
    bar = "\u2500" * 86
    print(f"\n{bar}\n  CONFIRMATION BIAS STUDY \u2014 ITERATIVE RECONSIDERATION RESULTS\n{bar}")
    line = f"  Total trials: {len(records)}  |  Usable: {len(usable)}  |  Failed: {len(records) - len(usable)}"
    if elapsed is not None:
        line += f"  |  Elapsed: {elapsed:.0f}s"
    print(line)
    if summary_rows:
        print(f"\n  {'model':<28}{'cond':<12}{'assert':<9}{'lvl':<5}"
              f"{'n':>5}{'%snap':>8}{'rounds':>9}{'finalMed':>10}{'%->anchor':>11}")
        for r in summary_rows:
            pct = "" if r["mean_pct_move_to_anchor"] is None else f"{r['mean_pct_move_to_anchor']:.2f}"
            print(f"  {r['model']:<28}{r['condition']:<12}{r['assertion']:<9}"
                  f"{str(r['anchor_level']):<5}{r['n']:>5}{r['pct_snapped']:>8}"
                  f"{r['mean_rounds']:>9}{r['final_median']:>10}{pct:>11}")
    print(f"\n{bar}\n  %snap = share that landed exactly on the anchor; "
          f"rounds = mean reconsideration rounds\n{bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  RUN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def confirm_and_run(specs, args, label):
    jsonl_file, csv_file, summary_file = paths(args.output_dir, STEM)
    print(f"\nConfirmation Bias study \u2014 {label}")
    print(f"  Models:      {args.models}")
    cells = len(set((s.model, s.condition, s.assertion, s.level) for s in specs))
    print(f"  Cells:       {cells}")
    print(f"  Total specs: {len(specs)}")
    print(f"  Max rounds:  {args.max_rounds}")
    print(f"  Temperature: {args.temperature}   Max tokens: {args.max_tokens}")
    print(f"  Workers:     {args.concurrency}")
    print(f"  Output dir:  {args.output_dir}")
    todo, already = filter_todo(specs, jsonl_file, ok=_is_complete)
    if already:
        print(f"  Resume:      {already} completed trials on disk; {len(todo)} left to run.")
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
        ans = input(f"\nLaunch up to {len(todo)} conversations (x{args.max_rounds} rounds)? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            sys.exit("Aborted. (Use --yes to skip this gate.)")
    _, elapsed = asyncio.run(run_batch(
        todo, jsonl_file, args.concurrency, args.temperature, args.max_tokens,
        run_iterative_trial, is_fail=lambda rec: rec.get("final_price") is None))
    records, summary_rows = rebuild_outputs(args.output_dir, args.include_raw)
    print(f"\n  Saved: {jsonl_file}\n         {csv_file}\n         {summary_file}")
    print_report(records, summary_rows, elapsed)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="Iterative reconsideration (first iteration + follow-ups).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Run the reconsideration loop over baseline disagreers.")
    pr.add_argument("--conditions", default="ALL",
                    help=f"Comma list, ALL, or NONE (bare probe). Options: {', '.join(ALL_CONDITIONS)}.")
    pr.add_argument("--assertions", default="standard",
                    help="Comma list of weak/standard/strong or ALL.")
    pr.add_argument("--anchor-levels", default="MID",
                    help=f"Comma list. Options: {', '.join(ANCHOR_LEVEL_KEYS)}.")
    pr.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                    help="Reconsideration rounds: 1=RQ4a, 2=+RQ4b, 3=+RQ4c, more=iterate loop.")
    pr.add_argument("--runs-per-row", type=int, default=DEFAULT_RUNS_PER_ROW)
    pr.add_argument("--baseline-source", default=DEFAULT_BASELINE_SOURCE,
                    help="Baseline CONTROL JSONL to seed disagreers from.")
    pr.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    add_common_run_args(pr)

    ps = sub.add_parser("summarize", help="Rebuild CSVs from JSONL (no API).")
    ps.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ps.add_argument("--include-raw", action="store_true")

    pp = sub.add_parser("show-prompt", help="Print the rounds 1..N prompts and exit.")
    pp.add_argument("--condition", default="MCKINSEY")
    pp.add_argument("--assertion", default="standard")
    pp.add_argument("--level", default="MID")
    pp.add_argument("--model", default=None)
    pp.add_argument("--anchor-value", type=float, default=None)
    pp.add_argument("--max-rounds", type=int, default=3)
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "run":
        conditions = resolve_conditions(args.conditions, allow_none=True)
        assertions = resolve_assertions(args.assertions)
        levels = resolve_levels(args.anchor_levels) if any(
            c != NONE_TOKEN for c in conditions) else []
        if any(c != NONE_TOKEN for c in conditions):
            require_anchor_values(args.models, levels)
        source_rows = filter_control_rows(
            load_source_records(args.baseline_source), models=args.models)
        if not source_rows:
            sys.exit(f"ERROR: no eligible CONTROL rows in {args.baseline_source}.")
        specs = build_specs(source_rows, conditions, assertions, levels,
                            args.runs_per_row, args.max_rounds)
        if not specs:
            sys.exit("ERROR: no specs built (check conditions/levels/source).")
        label = (f"ITERATIVE  conditions={conditions}  assertions={assertions}  "
                 f"levels={levels or '[NONE]'}  max_rounds={args.max_rounds}")
        confirm_and_run(specs, args, label)

    elif args.command == "summarize":
        records, summary_rows = rebuild_outputs(args.output_dir, args.include_raw)
        print_report(records, summary_rows)
        _, csv_file, summary_file = paths(args.output_dir, STEM)
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
                val = 65
                print(f"[preview note] no value for {args.level.upper()}; using $65 placeholder]\n")
            inj = build_injection(cond, args.assertion.lower(), val)
        print("===== SYSTEM PROMPT =====")
        print(SYSTEM_PROMPT)
        print("\n===== TURN 1 (reconstructed CONTROL prompt) =====")
        print(RECONSTRUCTED_FIRST_TURN)
        for rnd in range(1, args.max_rounds + 1):
            tag = "first iteration" if rnd == 1 else "follow-up"
            print(f"\n===== ROUND {rnd} USER PROMPT ({tag}) =====")
            print(build_round_prompt(rnd, inj))


if __name__ == "__main__":
    main()
