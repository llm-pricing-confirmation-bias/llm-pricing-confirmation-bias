"""Tolerant parsing of model completions into price / confidence / justification."""

from __future__ import annotations

import json
import re

_DECODER = json.JSONDecoder()


def iter_json_objects(text: str):
    idx = 0
    while True:
        brace = text.find("{", idx)
        if brace == -1:
            return
        try:
            obj, end = _DECODER.raw_decode(text[brace:])
            if isinstance(obj, dict):
                yield obj, brace
            idx = brace + max(end, 1)
        except json.JSONDecodeError:
            idx = brace + 1


def parse_response(raw_text: str):
    """Return (price, confidence, justification, error). Prefers the last valid
    JSON object carrying a price; falls back to a trailing regex scan."""
    text = re.sub(r"```(?:json)?", "", raw_text.strip())
    candidates = [
        (obj, pos) for obj, pos in iter_json_objects(text)
        if "price_recommendation" in obj or "price" in obj
    ]
    for obj, _pos in reversed(candidates):
        try:
            price = float(obj.get("price_recommendation", obj.get("price")))
        except (TypeError, ValueError):
            continue
        conf = obj.get("confidence")
        try:
            conf = max(0, min(100, int(conf))) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        just = obj.get("justification")
        just = str(just) if just is not None else None
        return price, conf, just, None

    for m in re.finditer(
        r'(?:price[_\s]*recommendation|recommend(?:ed)?\s+(?:price|rate))["\s:$]*'
        r'([0-9]+(?:\.[0-9]+)?)',
        text[-800:], re.IGNORECASE,
    ):
        try:
            return float(m.group(1)), None, None, None
        except ValueError:
            continue
    return None, None, None, "No parseable price found"


def compute_reasoning_words(raw_text: str) -> int:
    last_brace = raw_text.rfind('{"price_recommendation"')
    if last_brace == -1:
        last_brace = raw_text.rfind("{")
    head = raw_text[:last_brace] if last_brace > 0 else raw_text
    return len(head.split())
