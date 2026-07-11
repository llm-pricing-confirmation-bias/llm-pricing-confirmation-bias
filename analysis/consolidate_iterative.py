"""
Consolidate every iterative-reconsideration stage into ONE long-format table.

The original experiment was split across many directories (results_rq4,
results_rq4b, results_rq4c, results_rq4_iterate_full), each with a different
schema. This script flattens all of them into a single tidy file with **one row
per conversation per round** and an explicit ``round`` column, so per-iteration
results are trivial to slice (``round == 1`` is the first iteration,
``round <= 3`` reproduces the old RQ4a/b/c, etc.).

There are no "plausible" vs "unreasonable" anchor sets: every anchor is run once
per model. (The old results_rq4_unreasonable/ directory was a byte-for-byte copy
of the main run's claude VERY_LOW=$30 / VERY_HIGH=$100 cells, relabeled
LOW/HIGH; it is dropped here as a pure duplicate and contributes no unique data.)
``anchor_level`` is therefore unambiguous: one level name -> one dollar value per
model.

Round <-> conversation-turn mapping:
    round 1  = turn 2  (first iteration:  "Are you sure? In fact, ...")
    round 2  = turn 3  (follow-up:        "Are you really sure? As I mentioned before, ...")
    round 3  = turn 4  (follow-up, same wording)
    round 4+ = turn 5+ (open loop, same wording, repeated)

Inputs : results/iterative/source_per_round/round{1,2,3,4plus}.jsonl  (untouched stage copies)
Outputs: results/iterative/iterative_all_rounds.csv    (lean, one row per round)
         results/iterative/iterative_all_rounds.jsonl  (same rows + raw model text)
         results/iterative/iterative_round_summary.csv (per cell x round aggregates)
"""

import csv
import json
import os
import re
from statistics import mean

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config.paths import ITERATIVE_DIR, ITERATIVE_SOURCE_DIR

EPS = 1e-9
HERE = str(_ROOT)
SRC_DIR = str(ITERATIVE_SOURCE_DIR)
OUT_DIR = str(ITERATIVE_DIR)

# Stage files to fold in: (filename, stage_kind). The small open_loop_pilot is
# intentionally excluded from the merged table (it is an early pilot of the same
# round-4+ loop and would double-count conversations); its raw copy is kept under
# source_per_round/ for reference.
STAGES = [
    ("round1.jsonl",     "round1"),
    ("round2.jsonl",     "round2"),
    ("round3.jsonl",     "round3"),
    ("round4plus.jsonl", "round4plus"),
]

COLS = [
    "conversation_key", "model", "condition", "assertion",
    "anchor_level", "anchor_value", "round", "conversation_turn",
    "baseline_price", "price_before", "price_after", "delta",
    "pct_move_to_anchor", "snapped", "is_last_observed_round",
    "terminal_reason", "confidence", "justification",
    "parse_error", "api_error", "stage_trial_id", "source_trial_id",
    "source_stage_file", "timestamp",
]

_DECODER = json.JSONDecoder()


def _iter_json_objects(text):
    idx = 0
    while True:
        brace = text.find("{", idx)
        if brace == -1:
            return
        try:
            obj, end = _DECODER.raw_decode(text[brace:])
            if isinstance(obj, dict):
                yield obj
            idx = brace + max(end, 1)
        except json.JSONDecodeError:
            idx = brace + 1


def parse_conf_just(raw_text):
    """Pull (confidence, justification) out of a model reply, preferring the last
    valid JSON object carrying a price. Mirrors common.parse_response so round-4+
    rows get the same fields the earlier rounds already store."""
    if not raw_text:
        return None, None
    text = re.sub(r"```(?:json)?", "", raw_text.strip())
    candidates = [obj for obj in _iter_json_objects(text)
                  if "price_recommendation" in obj or "price" in obj]
    for obj in reversed(candidates):
        try:
            float(obj.get("price_recommendation", obj.get("price")))
        except (TypeError, ValueError):
            continue
        conf = obj.get("confidence")
        try:
            conf = max(0, min(100, int(conf))) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        just = obj.get("justification")
        just = str(just) if just is not None else None
        return conf, just
    return None, None


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# The "new price" field that signals a usable result, per stage kind.
USABLE_FIELD = {
    "round1": "second_price",
    "round2": "third_price",
    "round3": "fourth_price",
    "round4plus": "final_price",
}


def dedup(records, kind):
    """Collapse duplicate trial_ids the way the original pipeline did: prefer a
    record with a usable price, then the newest timestamp. The source JSONLs are
    append-only, so failed attempts and retries can share a trial_id."""
    field = USABLE_FIELD[kind]
    best = {}
    for r in records:
        tid = r.get("trial_id")
        if not tid:
            continue
        cur = best.get(tid)
        if cur is None:
            best[tid] = r
            continue
        cur_ok = cur.get(field) is not None
        new_ok = r.get(field) is not None
        if new_ok and not cur_ok:
            best[tid] = r
        elif new_ok == cur_ok and r.get("timestamp", "") >= cur.get("timestamp", ""):
            best[tid] = r
    return list(best.values())


def cond_fields(r):
    """Normalize the condition/assertion/level/value field names across stages."""
    return (
        r.get("followup_condition"),
        r.get("followup_assertion"),
        r.get("followup_level"),
        r.get("followup_value"),
    )


def conv_key(model, cond, assertion, level, source_trial_id):
    return "|".join(str(x) for x in
                    [model, cond, assertion, level, source_trial_id])


def snapped_flag(price, anchor):
    return bool(anchor is not None and price is not None and abs(price - anchor) < EPS)


def pct_move(price, baseline, anchor):
    if price is None or baseline is None or anchor is None:
        return None
    denom = anchor - baseline
    if abs(denom) <= EPS:
        return None
    return round((price - baseline) / denom, 4)


def base_row(r, source_file):
    cond, assertion, level, anchor = cond_fields(r)
    src = r.get("source_trial_id")
    return {
        "conversation_key": conv_key(r.get("model"), cond, assertion, level, src),
        "model": r.get("model"),
        "condition": cond, "assertion": assertion,
        "anchor_level": level, "anchor_value": anchor,
        "stage_trial_id": r.get("trial_id"), "source_trial_id": src,
        "source_stage_file": source_file, "timestamp": r.get("timestamp"),
        "parse_error": r.get("parse_error"), "api_error": r.get("api_error"),
    }


def rows_from_record(r, kind, source_file):
    """Yield one normalized row per NEW round contained in this stage record."""
    cond, assertion, level, anchor = cond_fields(r)

    if kind == "round1":
        base = r.get("first_price")
        before, after = r.get("first_price"), r.get("second_price")
        row = base_row(r, source_file)
        row.update(round=1, conversation_turn=2, baseline_price=base,
                   price_before=before, price_after=after,
                   delta=(round(after - before, 4) if after is not None and before is not None else None),
                   pct_move_to_anchor=pct_move(after, base, anchor),
                   snapped=snapped_flag(after, anchor), terminal_reason=None,
                   confidence=r.get("second_confidence"),
                   justification=r.get("second_justification"),
                   raw_text=r.get("second_raw_text"))
        yield row

    elif kind == "round2":
        base = r.get("first_price")
        before, after = r.get("second_price"), r.get("third_price")
        row = base_row(r, source_file)
        row.update(round=2, conversation_turn=3, baseline_price=base,
                   price_before=before, price_after=after,
                   delta=(round(after - before, 4) if after is not None and before is not None else None),
                   pct_move_to_anchor=pct_move(after, base, anchor),
                   snapped=snapped_flag(after, anchor), terminal_reason=None,
                   confidence=r.get("third_confidence"),
                   justification=r.get("third_justification"),
                   raw_text=r.get("third_raw_text"))
        yield row

    elif kind == "round3":
        base = r.get("first_price")
        before, after = r.get("third_price"), r.get("fourth_price")
        row = base_row(r, source_file)
        row.update(round=3, conversation_turn=4, baseline_price=base,
                   price_before=before, price_after=after,
                   delta=(round(after - before, 4) if after is not None and before is not None else None),
                   pct_move_to_anchor=pct_move(after, base, anchor),
                   snapped=snapped_flag(after, anchor), terminal_reason=None,
                   confidence=r.get("fourth_confidence"),
                   justification=r.get("fourth_justification"),
                   raw_text=r.get("fourth_raw_text"))
        yield row

    elif kind == "round4plus":
        traj = r.get("price_trajectory") or []
        pushes = r.get("push_prices") or []
        raws = r.get("push_raw_texts") or []
        base = traj[0] if traj else None
        start = len(traj) - len(pushes)   # index of first push within the trajectory
        terminal_reason = r.get("terminal_reason")
        for i, after in enumerate(pushes):
            rnd = 4 + i
            before = traj[start - 1 + i] if 0 <= start - 1 + i < len(traj) else None
            raw = raws[i] if i < len(raws) else None
            conf, just = parse_conf_just(raw)
            row = base_row(r, source_file)
            row.update(round=rnd, conversation_turn=rnd + 1, baseline_price=base,
                       price_before=before, price_after=after,
                       delta=(round(after - before, 4) if after is not None and before is not None else None),
                       pct_move_to_anchor=pct_move(after, base, anchor),
                       snapped=snapped_flag(after, anchor),
                       terminal_reason=(terminal_reason if i == len(pushes) - 1 else None),
                       confidence=conf, justification=just,
                       raw_text=raw)
            yield row


def main():
    rows = []
    for fname, kind in STAGES:
        path = os.path.join(SRC_DIR, fname)
        records = dedup(list(load_jsonl(path)), kind)
        n = 0
        for r in records:
            for row in rows_from_record(r, kind, fname):
                rows.append(row)
                n += 1
        print(f"  {fname:<20} {kind:<11} -> {len(records)} trials, {n} round-rows")

    # Mark the last observed round per conversation.
    max_round = {}
    for row in rows:
        k = row["conversation_key"]
        if row["round"] > max_round.get(k, -1):
            max_round[k] = row["round"]
    for row in rows:
        row["is_last_observed_round"] = (row["round"] == max_round[row["conversation_key"]])

    rows.sort(key=lambda x: (x["model"], x["condition"], x["assertion"],
                             str(x["anchor_level"]),
                             x["source_trial_id"] or "", x["round"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "iterative_all_rounds.csv")
    jsonl_path = os.path.join(OUT_DIR, "iterative_all_rounds.jsonl")

    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

    # Lean JSONL mirror of the CSV (raw model text is NOT duplicated here; it
    # stays in source_per_round/, reachable via stage_trial_id / source_trial_id).
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps({k: v for k, v in row.items() if k != "raw_text"}) + "\n")

    write_summary(rows)
    print(f"\n  Wrote {len(rows)} rows")
    print(f"    {csv_path}")
    print(f"    {jsonl_path}")


def write_summary(rows):
    cells = {}
    for r in rows:
        key = (r["model"], r["condition"], r["assertion"],
               r["anchor_level"], r["anchor_value"], r["round"])
        cells.setdefault(key, []).append(r)
    out = []
    for (model, cond, a, lvl, aval, rnd), rs in cells.items():
        afters = [x["price_after"] for x in rs if x["price_after"] is not None]
        pcts = [x["pct_move_to_anchor"] for x in rs if x["pct_move_to_anchor"] is not None]
        snaps = sum(1 for x in rs if x["snapped"])
        out.append({
            "model": model, "condition": cond, "assertion": a,
            "anchor_level": lvl, "anchor_value": aval, "round": rnd,
            "n": len(rs), "n_snapped": snaps,
            "pct_snapped": round(100 * snaps / len(rs), 1) if rs else 0.0,
            "mean_price_after": round(mean(afters), 2) if afters else None,
            "mean_pct_move_to_anchor": round(mean(pcts), 4) if pcts else None,
        })
    out.sort(key=lambda x: (x["model"], x["condition"], x["assertion"],
                            str(x["anchor_level"]), x["round"]))
    cols = ["model", "condition", "assertion", "anchor_level",
            "anchor_value", "round", "n", "n_snapped", "pct_snapped",
            "mean_price_after", "mean_pct_move_to_anchor"]
    path = os.path.join(OUT_DIR, "iterative_round_summary.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for row in out:
            wr.writerow(row)
    print(f"    {path}  ({len(out)} cells)")


if __name__ == "__main__":
    main()
