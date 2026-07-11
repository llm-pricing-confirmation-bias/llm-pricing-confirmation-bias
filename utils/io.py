"""JSONL / CSV persistence and resume helpers."""

from __future__ import annotations

import csv
import json
import os
import sys


def paths(output_dir, stem):
    """Return (jsonl, csv, summary) paths for a pipeline rooted at ``output_dir``."""
    return (
        os.path.join(output_dir, f"{stem}_trials.jsonl"),
        os.path.join(output_dir, f"{stem}_trials.csv"),
        os.path.join(output_dir, f"{stem}_summary.csv"),
    )


def load_records(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def load_source_records(path):
    """Like :func:`load_records` but aborts if the file is missing."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: source JSONL not found at '{path}'.")
    return load_records(path)


def dedup_records(records, key="trial_id", ok=lambda r: r.get("price_recommendation") is not None):
    """Keep the best record per ``key``: prefer a usable result, then newest timestamp."""
    best = {}
    for r in records:
        tid = r.get(key)
        if not tid:
            continue
        cur = best.get(tid)
        if cur is None:
            best[tid] = r
            continue
        cur_ok, new_ok = ok(cur), ok(r)
        if new_ok and not cur_ok:
            best[tid] = r
        elif new_ok == cur_ok and r.get("timestamp", "") >= cur.get("timestamp", ""):
            best[tid] = r
    return list(best.values())


def completed_ids(records, ok=lambda r: r.get("price_recommendation") is not None):
    return {r["trial_id"] for r in records if r.get("trial_id") and ok(r)}


def filter_todo(specs, jsonl_file, ok=lambda r: r.get("price_recommendation") is not None):
    """Drop specs whose trial_id already has a usable record on disk."""
    done = completed_ids(load_records(jsonl_file), ok=ok)
    return [s for s in specs if s.trial_id not in done], len(done)


def write_dict_csv(rows, path, cols, include_raw=False, raw_cols=("raw_text",)):
    """Write ``rows`` (list of dicts) to ``path`` with the given column order."""
    out_cols = list(cols)
    if include_raw:
        out_cols += [c for c in raw_cols if c not in out_cols]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
