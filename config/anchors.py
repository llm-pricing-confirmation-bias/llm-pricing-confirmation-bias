"""Per-model anchor dollar values (per-model design-doc anchor ladder)."""

from __future__ import annotations

import sys

from config.settings import ANCHOR_LEVEL_KEYS

# Dollar values substituted for "$X" in the injection sentence, per model, per
# level. Only the (model, level) pairs you actually run are required.
ANCHOR_VALUES: dict = {
    "UNREASONABLY_LOW": {
        "openai/gpt-5.4-mini": 10,
        "anthropic/claude-haiku-4.5": 10,
    },
    "VERY_LOW": {
        "openai/gpt-5.4-mini": 26,
        "anthropic/claude-haiku-4.5": 30,
    },
    "LOW": {
        "openai/gpt-5.4-mini": 47,
        "anthropic/claude-haiku-4.5": 52,
    },
    "MID": {
        "openai/gpt-5.4-mini": 58,
        "anthropic/claude-haiku-4.5": 65,
    },
    "HIGH": {
        "openai/gpt-5.4-mini": 70,
        "anthropic/claude-haiku-4.5": 78,
    },
    "VERY_HIGH": {
        "openai/gpt-5.4-mini": 90,
        "anthropic/claude-haiku-4.5": 100,
    },
    "UNREASONABLY_HIGH": {
        "openai/gpt-5.4-mini": 160,
        "anthropic/claude-haiku-4.5": 160,
    },
    "RIDICULOUSLY_HIGH": {
        "openai/gpt-5.4-mini": 1000,
        "anthropic/claude-haiku-4.5": 1000,
    },
}


def get_anchor_value(level, model):
    """Resolve an anchor value for a (level, model). Supports scalar or per-model dict."""
    v = ANCHOR_VALUES.get(level)
    if isinstance(v, dict):
        return v.get(model)
    return v


def require_anchor_values(models, levels):
    """Guard: every (model, level) pair about to run must resolve to a number."""
    problems = []
    for lvl in levels:
        for m in models:
            v = get_anchor_value(lvl, m)
            if v is None:
                problems.append(f"{lvl} not set for {m}")
                continue
            try:
                float(v)
            except (TypeError, ValueError):
                problems.append(f"{lvl} for {m} = {v!r} is not a number")
    if problems:
        sys.exit(
            "ERROR: missing/invalid anchor values in config/anchors.py for this run:\n  - "
            + "\n  - ".join(problems)
            + "\n(Only the levels you run are required.)"
        )


__all__ = [
    "ANCHOR_LEVEL_KEYS",
    "ANCHOR_VALUES",
    "get_anchor_value",
    "require_anchor_values",
]
