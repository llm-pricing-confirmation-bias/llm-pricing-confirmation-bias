"""Prompt construction: single-turn, injection lines, and reconsideration rounds."""

from __future__ import annotations

from config.prompts import (
    FIRST_ITER_BARE,
    FIRST_ITER_WITH_INJECTION,
    FOLLOWUP_BARE,
    FOLLOWUP_WITH_INJECTION,
    INJECTION_WRAPPER,
    INJECTIONS,
    JSON_FORMAT_TAIL,
    RECONSTRUCTED_FIRST_TURN,
    USER_PROMPT_HEAD,
    USER_PROMPT_TAIL,
)
from config.settings import CONTROL_TOKEN, NONE_TOKEN


def build_user_prompt(injection_sentence: str) -> str:
    """Single-turn user prompt. Empty injection => CONTROL (no note line)."""
    middle = INJECTION_WRAPPER.format(sentence=injection_sentence) if injection_sentence else ""
    return USER_PROMPT_HEAD + middle + USER_PROMPT_TAIL


def build_first_iteration_prompt(injection_sentence: str) -> str:
    head = (FIRST_ITER_WITH_INJECTION.format(sentence=injection_sentence)
            if injection_sentence else FIRST_ITER_BARE)
    return head + JSON_FORMAT_TAIL


def build_followup_prompt(injection_sentence: str) -> str:
    head = (FOLLOWUP_WITH_INJECTION.format(sentence=injection_sentence)
            if injection_sentence else FOLLOWUP_BARE)
    return head + JSON_FORMAT_TAIL


def build_round_prompt(round_index: int, injection_sentence: str) -> str:
    """Round 1 uses the first-iteration wording; later rounds repeat the follow-up."""
    if round_index <= 1:
        return build_first_iteration_prompt(injection_sentence)
    return build_followup_prompt(injection_sentence)


def format_price(value) -> str:
    if value is None:
        return ""
    v = float(value)
    return f"${int(v)}" if v.is_integer() else f"${v:,.2f}"


def build_injection(condition: str, assertion: str, anchor_value) -> str:
    """Render an authority sentence. CONTROL / NONE => empty (bare prompt)."""
    if condition in (None, "", NONE_TOKEN, CONTROL_TOKEN):
        return ""
    template = INJECTIONS[condition][assertion]
    return template.replace("$X", format_price(anchor_value))


__all__ = [
    "RECONSTRUCTED_FIRST_TURN",
    "build_followup_prompt",
    "build_first_iteration_prompt",
    "build_injection",
    "build_round_prompt",
    "build_user_prompt",
    "format_price",
]
