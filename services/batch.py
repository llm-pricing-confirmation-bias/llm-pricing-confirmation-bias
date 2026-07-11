"""Bounded async worker pool for experiment runners."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass

try:
    import aiohttp
except ImportError:
    sys.exit("ERROR: aiohttp is required.  pip install aiohttp")


async def _worker(queue, session, jf, write_lock, progress, total, start,
                  temperature, max_tokens, new_records, run_trial_fn, is_fail):
    while True:
        spec = await queue.get()
        try:
            if spec is None:
                return
            result = await run_trial_fn(session, spec, temperature, max_tokens)
            rec = asdict(result) if is_dataclass(result) else result
            async with write_lock:
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                new_records.append(rec)
                progress["done"] += 1
                if is_fail(rec):
                    progress["fail"] += 1
                d = progress["done"]
                if d % 25 == 0 or d == total:
                    rate = d / max(1e-9, time.monotonic() - start)
                    eta = (total - d) / max(rate, 1e-9)
                    print(f"  [{d}/{total}]  fails={progress['fail']}  "
                          f"{rate:.1f} trials/s  ETA {eta/60:.1f} min", flush=True)
        finally:
            queue.task_done()


async def run_batch(specs, jsonl_file, concurrency, temperature, max_tokens,
                    run_trial_fn, is_fail=lambda rec: rec.get("price_recommendation") is None):
    """Run ``run_trial_fn`` over ``specs`` with a bounded async pool, appending
    each result as a JSON line to ``jsonl_file``."""
    queue: asyncio.Queue = asyncio.Queue()
    for spec in specs:
        queue.put_nowait(spec)
    for _ in range(concurrency):
        queue.put_nowait(None)
    progress = {"done": 0, "fail": 0}
    write_lock = asyncio.Lock()
    new_records = []
    start = time.monotonic()
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    os.makedirs(os.path.dirname(jsonl_file) or ".", exist_ok=True)
    with open(jsonl_file, "a") as jf:
        async with aiohttp.ClientSession(connector=connector) as session:
            workers = [
                asyncio.create_task(_worker(
                    queue, session, jf, write_lock, progress, len(specs),
                    start, temperature, max_tokens, new_records, run_trial_fn, is_fail))
                for _ in range(concurrency)
            ]
            await queue.join()
            await asyncio.gather(*workers)
    return new_records, time.monotonic() - start
