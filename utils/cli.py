"""Shared argparse helpers for experiment CLIs."""

from __future__ import annotations

import sys

from config.anchors import ANCHOR_LEVEL_KEYS, require_anchor_values  # noqa: F401 — re-export
from config.prompts import ALL_CONDITIONS
from config.settings import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODELS,
    DEFAULT_TEMPERATURE,
    NONE_TOKEN,
    VALID_ASSERTIONS,
)


def parse_csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_models(s):
    return parse_csv_list(s) if s else list(DEFAULT_MODELS)


def resolve_conditions(arg, allow_none=False):
    valid = list(ALL_CONDITIONS) + ([NONE_TOKEN] if allow_none else [])
    if not arg or arg.strip().upper() == "ALL":
        return list(ALL_CONDITIONS)
    out = []
    for tok in parse_csv_list(arg):
        key = tok.upper().replace(" ", "_").replace("-", "_")
        if key not in valid:
            sys.exit(f"ERROR: unknown condition '{tok}'. Valid: {', '.join(valid)} (or ALL).")
        out.append(key)
    return out


def resolve_assertions(arg):
    if not arg or arg.strip().upper() == "ALL":
        return list(VALID_ASSERTIONS)
    out = []
    for tok in parse_csv_list(arg):
        key = tok.lower()
        if key not in VALID_ASSERTIONS:
            sys.exit(f"ERROR: unknown assertion '{tok}'. Valid: {', '.join(VALID_ASSERTIONS)} (or ALL).")
        out.append(key)
    return out


def resolve_levels(arg):
    out = []
    for tok in parse_csv_list(arg):
        key = tok.upper()
        if key not in ANCHOR_LEVEL_KEYS:
            sys.exit(f"ERROR: unknown anchor level '{tok}'. Valid: {', '.join(ANCHOR_LEVEL_KEYS)}.")
        out.append(key)
    if not out:
        sys.exit("ERROR: specify at least one anchor level.")
    return out


def add_common_run_args(p):
    p.add_argument("--models", type=parse_models, default=list(DEFAULT_MODELS),
                   help="Comma-separated OpenRouter model slugs.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--include-raw", action="store_true",
                   help="Include the full raw model text in the trials CSV.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true",
                   help="Skip the interactive launch confirmation.")
