"""Small descriptive-stat helpers shared by experiment runners."""

from __future__ import annotations

from statistics import mean, median, quantiles, stdev


def summarize_vals(vals):
    if not vals:
        return {"n": 0}
    s = {
        "n": len(vals),
        "mean": round(mean(vals), 2),
        "median": round(median(vals), 2),
        "sd": round(stdev(vals), 2) if len(vals) > 1 else 0.0,
        "cv": round(stdev(vals) / mean(vals), 4) if len(vals) > 1 and mean(vals) else 0.0,
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }
    if len(vals) >= 4:
        q = quantiles(vals, n=4)
        s["p25"], s["p75"] = round(q[0], 2), round(q[2], 2)
    else:
        s["p25"] = s["p75"] = None
    return s
