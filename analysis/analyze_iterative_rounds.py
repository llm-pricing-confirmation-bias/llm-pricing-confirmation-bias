"""
Per-round breakdown of iterative-design outcomes (rounds 1–5).

Reads iterative_all_rounds.csv and emits:
  - CSV summary tables (by model × anchor × condition × round, plus pooled totals)
  - LaTeX tables in the same spirit as dump.py section_further
  - Histogram PNGs for bimodality sanity checks
  - A short markdown note on whether "gradually increases" is supported
"""

import csv
import os
from collections import defaultdict
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config.paths import ITERATIVE_DIR, PER_ROUND_ANALYSIS_DIR

HERE = str(_ROOT)
DATA = str(ITERATIVE_DIR / "iterative_all_rounds.csv")
OUT = str(PER_ROUND_ANALYSIS_DIR)

EPS = 1e-9
ANCHOR_BAND = 2.0
BASELINE_BAND = 2.0
MAX_ROUND = 5

MODEL_ORDER = ["anthropic/claude-haiku-4.5", "openai/gpt-5.4-mini"]
MODEL_PRETTY = {
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "openai/gpt-5.4-mini": "GPT-5.4 Mini",
}
CONDITION_GROUPS = [
    ["INTERN", "COLLEAGUE", "MANAGER"],
    ["CONSULTING", "MCKINSEY"],
    ["UNLABELED", "AI"],
]
CONDITION_ORDER = [c for g in CONDITION_GROUPS for c in g]
CONDITION_PRETTY = {
    "INTERN": "Intern", "COLLEAGUE": "Colleague", "MANAGER": "Manager",
    "CONSULTING": "Consulting", "MCKINSEY": "McKinsey",
    "UNLABELED": "Unlabeled", "AI": "AI",
}
LEVEL_ORDER = [
    "MID", "LOW", "HIGH",
    "VERY_LOW", "VERY_HIGH",
    "UNREASONABLY_LOW", "UNREASONABLY_HIGH",
    "RIDICULOUSLY_HIGH",
]
LEVEL_PRETTY = {
    "MID": "Mid", "LOW": "Low", "HIGH": "High",
    "VERY_LOW": "Very Low", "VERY_HIGH": "Very High",
    "UNREASONABLY_LOW": "Unreasonably Low",
    "UNREASONABLY_HIGH": "Unreasonably High",
    "RIDICULOUSLY_HIGH": "Ridiculously High",
}
FOCUS_LEVELS = ["VERY_HIGH", "UNREASONABLY_HIGH", "VERY_LOW"]
SATURATE_LEVELS = ["LOW", "MID", "HIGH"]
HIST_LEVELS = ["VERY_HIGH", "UNREASONABLY_HIGH"]


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("assertion") != "standard":
                continue
            rnd = int(r["round"])
            if rnd > MAX_ROUND:
                continue
            pa = r.get("price_after")
            bp = r.get("baseline_price")
            av = r.get("anchor_value")
            if pa in (None, "") or bp in (None, "") or av in (None, ""):
                continue
            rows.append({
                "conversation_key": r["conversation_key"],
                "model": r["model"],
                "condition": r["condition"],
                "anchor_level": r["anchor_level"],
                "anchor_value": float(av),
                "round": rnd,
                "baseline_price": float(bp),
                "price_after": float(pa),
                "snapped": str(r.get("snapped", "")).lower() == "true",
            })
    return rows


def cell_stats(prices, anchor, baselines):
    n = len(prices)
    if n == 0:
        return None
    at = sum(1 for p in prices if abs(p - anchor) < EPS)
    near_a = sum(1 for p in prices if abs(p - anchor) <= ANCHOR_BAND)
    near_b = sum(1 for p, b in zip(prices, baselines) if abs(p - b) <= BASELINE_BAND)
    mid = sum(
        1 for p, b in zip(prices, baselines)
        if abs(p - anchor) > ANCHOR_BAND and abs(p - b) > BASELINE_BAND
    )
    return {
        "n": n,
        "pct_anc": 100 * at / n,
        "pct_anc2": 100 * near_a / n,
        "pct_base": 100 * near_b / n,
        "pct_mid": 100 * mid / n,
        "mean": mean(prices),
        "sd": stdev(prices) if n > 1 else 0.0,
    }


def aggregate(rows):
    """(model, level, cond, round) -> stats dict."""
    buckets = defaultdict(lambda: {"prices": [], "baselines": [], "anchor": None})
    for r in rows:
        key = (r["model"], r["anchor_level"], r["condition"], r["round"])
        buckets[key]["prices"].append(r["price_after"])
        buckets[key]["baselines"].append(r["baseline_price"])
        buckets[key]["anchor"] = r["anchor_value"]
    out = {}
    for key, b in buckets.items():
        out[key] = cell_stats(b["prices"], b["anchor"], b["baselines"])
    return out


def pooled_by_round(rows, model, level, rnd):
    sub = [r for r in rows if r["model"] == model and r["anchor_level"] == level and r["round"] == rnd]
    if not sub:
        return None
    anchor = sub[0]["anchor_value"]
    prices = [r["price_after"] for r in sub]
    baselines = [r["baseline_price"] for r in sub]
    return cell_stats(prices, anchor, baselines)


def write_detail_csv(cells, path):
    cols = [
        "model", "anchor_level", "anchor_value", "condition", "round",
        "n", "pct_anc", "pct_anc2", "pct_base", "pct_mid", "mean", "sd",
    ]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for model in MODEL_ORDER:
            for level in LEVEL_ORDER:
                for cond in CONDITION_ORDER:
                    for rnd in range(1, MAX_ROUND + 1):
                        s = cells.get((model, level, cond, rnd))
                        if s is None:
                            continue
                        av = None
                        for c2 in CONDITION_ORDER:
                            s2 = cells.get((model, level, c2, rnd))
                            if s2:
                                # anchor from any sibling row
                                break
                        wr.writerow({
                            "model": model,
                            "anchor_level": level,
                            "anchor_value": cells.get((model, level, cond, 1), {}).get("anchor")
                            or next(
                                (rows_anchor[model, level]
                                 for (m, lv, _, _), rows_anchor in [({},)]
                                 ), None
                            ),
                            "condition": cond,
                            "round": rnd,
                            **{k: round(s[k], 2) if k != "n" else s[k] for k in cols[5:]},
                        })


def anchor_for(cells, model, level):
    for cond in CONDITION_ORDER:
        for rnd in range(1, MAX_ROUND + 1):
            key = (model, level, cond, rnd)
            if key in cells:
                # recover from first matching row in raw — use cells keys only; store anchor separately
                pass
    return None


def build_anchor_map(rows):
    m = {}
    for r in rows:
        m[(r["model"], r["anchor_level"])] = r["anchor_value"]
    return m


def write_detail_csv_v2(rows, cells, path):
    anchors = build_anchor_map(rows)
    cols = [
        "model", "anchor_level", "anchor_value", "condition", "round",
        "n", "pct_anc", "pct_anc2", "pct_base", "pct_mid", "mean", "sd",
    ]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        keys = sorted(cells.keys(), key=lambda k: (MODEL_ORDER.index(k[0]), LEVEL_ORDER.index(k[1]),
                                                    CONDITION_ORDER.index(k[2]), k[3]))
        for model, level, cond, rnd in keys:
            s = cells[(model, level, cond, rnd)]
            wr.writerow({
                "model": model,
                "anchor_level": level,
                "anchor_value": anchors.get((model, level)),
                "condition": cond,
                "round": rnd,
                "n": s["n"],
                "pct_anc": round(s["pct_anc"], 2),
                "pct_anc2": round(s["pct_anc2"], 2),
                "pct_base": round(s["pct_base"], 2),
                "pct_mid": round(s["pct_mid"], 2),
                "mean": round(s["mean"], 2),
                "sd": round(s["sd"], 2),
            })


def write_pooled_csv(rows, path):
    cols = [
        "model", "anchor_level", "anchor_value", "round",
        "n", "pct_anc", "pct_anc2", "pct_base", "pct_mid", "mean", "sd",
    ]
    anchors = build_anchor_map(rows)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for model in MODEL_ORDER:
            for level in LEVEL_ORDER:
                for rnd in range(1, MAX_ROUND + 1):
                    s = pooled_by_round(rows, model, level, rnd)
                    if s is None:
                        continue
                    wr.writerow({
                        "model": model,
                        "anchor_level": level,
                        "anchor_value": anchors.get((model, level)),
                        "round": rnd,
                        "n": s["n"],
                        "pct_anc": round(s["pct_anc"], 2),
                        "pct_anc2": round(s["pct_anc2"], 2),
                        "pct_base": round(s["pct_base"], 2),
                        "pct_mid": round(s["pct_mid"], 2),
                        "mean": round(s["mean"], 2),
                        "sd": round(s["sd"], 2),
                    })


def _pct(x, nd=1):
    return f"{x:.{nd}f}\\%"


def _f(x, nd=2):
    return f"{x:.{nd}f}"


def _n(n):
    return f"{int(n):,}".replace(",", "{,}")


def latex_round_table(model, rnd, cells, anchors, levels):
    """Wide table: authority rows × anchor levels; sv/@a/±2/base/mid columns."""
    present = [lv for lv in levels if any(cells.get((model, lv, c, rnd)) for c in CONDITION_ORDER)]
    if not present:
        return None
    lines = [
        r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{2pt}",
        rf"\caption{{{MODEL_PRETTY[model]} --- round {rnd} outcomes "
        rf"(sv=survive\%, @a=exact anchor, $\pm$2=near anchor, base=near baseline, mid=neither).}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + " ccccc" * len(present) + "}",
        r"\toprule",
    ]
    head1 = ["Authority"]
    head2 = [""]
    cmids = []
    for i, lv in enumerate(present):
        av = anchors.get((model, lv))
        avs = f"{int(av):,}".replace(",", "{,}") if av is not None else "?"
        head1.append(rf"\multicolumn{{5}}{{c}}{{{LEVEL_PRETTY[lv]}~(\${avs})}}")
        start = 2 + 5 * i
        cmids.append(rf"\cmidrule(lr){{{start}-{start + 4}}}")
        head2 += ["sv", "@a", r"$\pm$2", "base", "mid"]
    lines.append(" & ".join(head1) + r" \\")
    lines.append(" ".join(cmids))
    lines.append(" & ".join(head2) + r" \\")
    lines.append(r"\midrule")

    def row_vals(cond):
        cells_out = [CONDITION_PRETTY.get(cond, cond)]
        for lv in present:
            s = cells.get((model, lv, cond, rnd))
            if s is None:
                cells_out += ["--"] * 5
            else:
                surv = 100 - s["pct_anc2"]
                cells_out += [
                    _pct(surv, 0), _pct(s["pct_anc"], 0), _pct(s["pct_anc2"], 0),
                    _pct(s["pct_base"], 0), _pct(s["pct_mid"], 0),
                ]
        return " & ".join(cells_out) + r" \\"

    for gi, group in enumerate(CONDITION_GROUPS):
        for cond in group:
            if any(cells.get((model, lv, cond, rnd)) for lv in present):
                lines.append(row_vals(cond))
        if gi < len(CONDITION_GROUPS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\midrule")
    trow = [r"\textbf{Total}"]
    nrow = [r"\textit{n}"]
    for lv in present:
        sub_prices, sub_base, anchor = [], [], anchors.get((model, lv))
        for c in CONDITION_ORDER:
            key = (model, lv, c, rnd)
            # recompute from stored stats weighted by n — need raw; use weighted avg of pct
            s = cells.get(key)
            if s:
                trow.extend([
                    _pct(100 - s["pct_anc2"], 0), _pct(s["pct_anc"], 0),
                    _pct(s["pct_anc2"], 0), _pct(s["pct_base"], 0), _pct(s["pct_mid"], 0),
                ])
                nrow.append(rf"\multicolumn{{5}}{{c}}{{{_n(s['n'])}}}")
                continue
        # pooled total row
    # fix total row properly below
    trow = [r"\textbf{Total}"]
    nrow = [r"\textit{n}"]
    for lv in present:
        parts = [cells.get((model, lv, c, rnd)) for c in CONDITION_ORDER]
        parts = [p for p in parts if p]
        if not parts:
            trow += ["--"] * 5
            nrow.append(r"\multicolumn{5}{c}{--}")
            continue
        n = sum(p["n"] for p in parts)
        def wavg(field):
            return sum(p[field] * p["n"] for p in parts) / n
        trow += [
            _pct(100 - wavg("pct_anc2"), 0), _pct(wavg("pct_anc"), 0),
            _pct(wavg("pct_anc2"), 0), _pct(wavg("pct_base"), 0), _pct(wavg("pct_mid"), 0),
        ]
        nrow.append(rf"\multicolumn{{5}}{{c}}{{{_n(n)}}}")
    lines.append(" & ".join(trow) + r" \\")
    lines.append(" & ".join(nrow) + r" \\")
    lines.append(r"\bottomrule\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_latex(rows, cells, anchors, path):
    parts = [
        r"% Auto-generated by analyze_iterative_rounds.py",
        r"\section{Iterative design --- per-round breakdown (rounds 1--5)}",
        r"\noindent Each row uses the cohort still active at that round "
        r"(conversations that had not yet landed exactly on the anchor). "
        r"\texttt{base\%} = share within \$2 of the model's original independent price; "
        r"\texttt{mid\%} = share within neither band (the ``valley'' between clusters).",
        "",
    ]
    focus = FOCUS_LEVELS + SATURATE_LEVELS
    for model in MODEL_ORDER:
        parts.append(rf"\subsection*{{{MODEL_PRETTY[model]}}}")
        for rnd in range(1, MAX_ROUND + 1):
            tbl = latex_round_table(model, rnd, cells, anchors, focus)
            if tbl:
                parts.append(tbl)
                parts.append("")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def plot_near_anchor_trends(rows, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    rounds = list(range(1, MAX_ROUND + 1))
    for ax, model in zip(axes, MODEL_ORDER):
        for level in FOCUS_LEVELS + SATURATE_LEVELS:
            ys = []
            ns = []
            for rnd in rounds:
                s = pooled_by_round(rows, model, level, rnd)
                ys.append(s["pct_anc2"] if s else np.nan)
                ns.append(s["n"] if s else 0)
            style = "-" if level in FOCUS_LEVELS else "--"
            lw = 2.5 if level in FOCUS_LEVELS else 1.2
            ax.plot(rounds, ys, style, linewidth=lw, marker="o", label=f"{LEVEL_PRETTY[level]} (n@r5={ns[-1]})")
        ax.set_ylabel("Near-anchor % (±$2)")
        ax.set_title(MODEL_PRETTY[model])
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Round")
    fig.suptitle("Pooled near-anchor share by round (all authority sources)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for model in MODEL_ORDER:
        for level in HIST_LEVELS:
            fig, axes = plt.subplots(1, MAX_ROUND, figsize=(16, 3), sharey=True)
            if MAX_ROUND == 1:
                axes = [axes]
            anchor = next(r["anchor_value"] for r in rows
                          if r["model"] == model and r["anchor_level"] == level)
            for i, rnd in enumerate(range(1, MAX_ROUND + 1)):
                sub = [r for r in rows if r["model"] == model and r["anchor_level"] == level and r["round"] == rnd]
                ax = axes[i]
                if not sub:
                    ax.set_title(f"R{rnd}\n(n=0)")
                    continue
                prices = [r["price_after"] for r in sub]
                baselines = [r["baseline_price"] for r in sub]
                lo = min(min(prices), min(baselines), anchor) - 5
                hi = max(max(prices), max(baselines), anchor) + 5
                bins = np.arange(lo, hi + 2, 2)
                ax.hist(prices, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
                ax.axvline(anchor, color="crimson", ls="-", lw=1.5, label="anchor")
                ax.axvline(mean(baselines), color="darkgreen", ls="--", lw=1.2, label="mean baseline")
                s = cell_stats(prices, anchor, baselines)
                ax.set_title(f"R{rnd}\n(n={s['n']}, mid={s['pct_mid']:.0f}%)", fontsize=9)
                if i == 0:
                    ax.legend(fontsize=7)
            slug = model.split("/")[-1]
            fig.suptitle(f"{MODEL_PRETTY[model]} — {LEVEL_PRETTY[level]} (${int(anchor)})", y=1.05)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"hist_{slug}_{level}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)


def build_trajectories(rows):
    traj = defaultdict(list)
    meta = {}
    for r in rows:
        traj[r["conversation_key"]].append(r)
        meta[r["conversation_key"]] = (r["model"], r["anchor_level"])
    for k in traj:
        traj[k].sort(key=lambda x: x["round"])
    return traj, meta


def cumulative_stats(rows, model, level, target_r):
    """Among all conversations that enter round 1, share near anchor by end of target_r."""
    traj, meta = build_trajectories(rows)
    keys = [k for k, (m, lv) in meta.items() if m == model and lv == level]
    if not keys:
        return None
    n0 = len(keys)
    near = exact = 0
    for k in keys:
        steps = [s for s in traj[k] if s["round"] <= target_r]
        if not steps:
            continue
        last = steps[-1]
        if abs(last["price_after"] - last["anchor_value"]) <= ANCHOR_BAND:
            near += 1
        if any(s["snapped"] for s in steps):
            exact += 1
    return {
        "n0": n0,
        "pct_anc2_cumul": 100 * near / n0,
        "pct_anc_cumul": 100 * exact / n0,
    }


def terminal_stats(rows, model, level):
    """Final observed price for conversations that reached round >= 2 (matches dump.py cohort)."""
    traj, meta = build_trajectories(rows)
    keys = [k for k, (m, lv) in meta.items() if m == model and lv == level]
    terminals = []
    for k in keys:
        steps = traj[k]
        if len(steps) < 2:
            continue
        last = steps[-1]
        terminals.append(last)
    if not terminals:
        return None
    anchor = terminals[0]["anchor_value"]
    prices = [t["price_after"] for t in terminals]
    baselines = [t["baseline_price"] for t in terminals]
    s = cell_stats(prices, anchor, baselines)
    s["n"] = len(terminals)
    return s


def write_cumulative_csv(rows, path):
    cols = ["model", "anchor_level", "round", "n0", "pct_anc_cumul", "pct_anc2_cumul"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for model in MODEL_ORDER:
            for level in LEVEL_ORDER:
                for rnd in range(1, MAX_ROUND + 1):
                    s = cumulative_stats(rows, model, level, rnd)
                    if s is None:
                        continue
                    wr.writerow({
                        "model": model,
                        "anchor_level": level,
                        "round": rnd,
                        "n0": s["n0"],
                        "pct_anc_cumul": round(s["pct_anc_cumul"], 2),
                        "pct_anc2_cumul": round(s["pct_anc2_cumul"], 2),
                    })


def monotonicity_report(rows):
    lines = [
        "# Per-round near-anchor trend — wording check",
        "",
        "Claim under review: *\"As we keep repeating the prompt, the size of the first cluster "
        "(near-anchor) tends to gradually increase.\"*",
        "",
        "## Two metrics (do not conflate them)",
        "",
        "1. **Active-cohort rate** (`iter_per_round_pooled.csv`): among conversations still being",
        "   pushed at round *r* (had not snapped exactly to the anchor earlier), what share land",
        "   near-anchor that round. This **falls** at rounds 3–5 because easy adopters exit after",
        "   snapping; the remaining pool is enriched for resistors who revert toward baseline.",
        "",
        "2. **Cumulative adoption** (`iter_cumulative_pooled.csv`): among the full round-1 cohort,",
        "   what share are near-anchor by the end of round *r* (including conversations that already",
        "   snapped and stopped). This is the relevant metric for \"the anchor cluster grows with",
        "   repeated prompting\" and **is** monotonic for Claude's bimodal anchors.",
        "",
        "---",
        "",
        "## A. Active cohort (still being pushed) — near-anchor ±2%",
        "",
    ]
    rounds = list(range(1, MAX_ROUND + 1))
    for model in MODEL_ORDER:
        lines.append(f"### {MODEL_PRETTY[model]}")
        lines.append("")
        lines.append("| Anchor | R1 n | R1 ±2% | R2 ±2% | R3 ±2% | R4 ±2% | R5 ±2% | Monotonic? | Δ R1→R2 |")
        lines.append("|--------|------|--------|--------|--------|--------|--------|------------|---------|")
        for level in FOCUS_LEVELS + SATURATE_LEVELS:
            ns, pcts = [], []
            for rnd in rounds:
                s = pooled_by_round(rows, model, level, rnd)
                ns.append(s["n"] if s else 0)
                pcts.append(s["pct_anc2"] if s else None)
            mono = all(
                pcts[i] <= pcts[i + 1]
                for i in range(len(pcts) - 1)
                if pcts[i] is not None and pcts[i + 1] is not None
            )
            delta12 = (pcts[1] - pcts[0]) if pcts[0] is not None and pcts[1] is not None else None
            pct_cols = [f"{p:.1f}" if p is not None else "—" for p in pcts]
            lines.append(
                f"| {LEVEL_PRETTY[level]} | {ns[0]:,} | {' | '.join(pct_cols)} | "
                f"{'no' if not mono else 'yes'} | {delta12:+.1f}pp |" if delta12 is not None
                else f"| {LEVEL_PRETTY[level]} | {ns[0]:,} | {' | '.join(pct_cols)} | "
                f"{'no' if not mono else 'yes'} | — |"
            )
        lines.append("")

    lines.extend([
        "## B. Cumulative adoption (full starting cohort) — near-anchor ±2% by end of round *r*",
        "",
    ])
    for model in MODEL_ORDER:
        lines.append(f"### {MODEL_PRETTY[model]}")
        lines.append("")
        lines.append("| Anchor | n₀ | R1 | R2 | R3 | R4 | R5 | Monotonic? | R1→R2 | R2→R5 |")
        lines.append("|--------|-----|-----|-----|-----|-----|-----|------------|-------|-------|")
        for level in FOCUS_LEVELS + SATURATE_LEVELS:
            pcts = []
            n0 = None
            for rnd in rounds:
                s = cumulative_stats(rows, model, level, rnd)
                if s:
                    n0 = s["n0"]
                    pcts.append(s["pct_anc2_cumul"])
                else:
                    pcts.append(None)
            mono = all(pcts[i] <= pcts[i + 1] for i in range(len(pcts) - 1) if pcts[i] is not None and pcts[i + 1] is not None)
            d12 = pcts[1] - pcts[0] if pcts[0] is not None and pcts[1] is not None else None
            d25 = pcts[-1] - pcts[1] if pcts[-1] is not None and pcts[1] is not None else None
            pct_cols = [f"{p:.1f}" if p is not None else "—" for p in pcts]
            lines.append(
                f"| {LEVEL_PRETTY[level]} | {n0:,} | {' | '.join(pct_cols)} | "
                f"{'yes' if mono else 'no'} | {d12:+.1f}pp | {d25:+.1f}pp |"
            )
        lines.append("")

    lines.extend([
        "## Bimodality valley (pooled mid% = neither near anchor nor near baseline)",
        "",
    ])
    for model in MODEL_ORDER:
        lines.append(f"### {MODEL_PRETTY[model]}")
        lines.append("")
        lines.append("| Anchor | Round | n | near-anchor ±2% | near-baseline % | mid % |")
        lines.append("|--------|-------|---|-----------------|-----------------|-------|")
        for level in FOCUS_LEVELS:
            for rnd in rounds:
                s = pooled_by_round(rows, model, level, rnd)
                if not s:
                    continue
                lines.append(
                    f"| {LEVEL_PRETTY[level]} | {rnd} | {s['n']:,} | {s['pct_anc2']:.1f} | "
                    f"{s['pct_base']:.1f} | {s['pct_mid']:.1f} |"
                )
        lines.append("")

    lines.extend([
        "## C. Terminal cohort (round ≥ 2 final price; matches existing rq3-further table)",
        "",
        "| Model | Anchor | n | @a ±2% | baseline ±2% | mid % |",
        "|-------|--------|---|--------|--------------|-------|",
    ])
    for model in MODEL_ORDER:
        for level in FOCUS_LEVELS:
            s = terminal_stats(rows, model, level)
            if s:
                lines.append(
                    f"| {MODEL_PRETTY[model]} | {LEVEL_PRETTY[level]} | {s['n']:,} | "
                    f"{s['pct_anc2']:.1f} | {s['pct_base']:.1f} | {s['pct_mid']:.1f} |"
                )
    lines.extend(["", "## Recommendation", ""])

    claude = MODEL_ORDER[0]
    verdicts = []
    for level in FOCUS_LEVELS:
        cum = [cumulative_stats(rows, claude, level, r)["pct_anc2_cumul"] for r in rounds]
        d12, d25 = cum[1] - cum[0], cum[-1] - cum[1]
        verdicts.append(
            f"- **{LEVEL_PRETTY[level]} (Claude, cumulative):** "
            f"{cum[0]:.0f}%→{cum[1]:.0f}%→…→{cum[-1]:.0f}% near-anchor by end of rounds 1–5. "
            f"+{d12:.0f}pp at round 2 alone; +{d25:.0f}pp across rounds 3–5 combined."
        )
    verdicts.extend([
        "",
        "**Bimodality:** Confirmed at terminal (round ≥ 2) for Claude Very High and Unreasonably High "
        "(mid ≈ 0.5–0.7%). Round-1 Unreasonably High also bimodal (mid ≈ 20%). Histograms in "
        "`histograms/`.",
        "",
        "**GPT contrast:** Cumulative near-anchor stays negligible at Very High (10%) and "
        "Unreasonably High (0.1%); no analogous anchor-cluster growth.",
        "",
        "### Verdict on intro wording",
        "",
        "**Do not use \"gradually increases\" as written.** Two problems:",
        "",
        "1. If interpreted as the per-round active-cohort rate, the claim is **false** "
        "(near-anchor % *declines* at rounds 3–5 as adopters exit).",
        "2. If interpreted cumulatively (the sensible reading), near-anchor share **does** rise "
        "with repeated prompts, but **not gradually** — roughly **75–80% of the total R1→R5 gain "
        "happens at round 2** (the first repeat pushback); rounds 3–5 add only 1–15 pp depending "
        "on anchor.",
        "",
        "**Suggested replacement:** "
        "\"Repeated follow-up prompts produce a bimodal pattern rather than concentrating around "
        "the suggested price: responses cluster at the anchor or revert toward the model's original "
        "estimate, with very few in between. Sustained pressure shifts more of the cohort toward the "
        "anchor over successive rounds, though most of that movement occurs on the second pushback "
        "rather than accumulating evenly across later rounds.\"",
        "",
        "**Low/Mid/High:** Saturate by round 1 (Claude ≥97% near-anchor); no per-round story needed.",
    ])
    lines.extend(verdicts)
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows(DATA)
    cells = aggregate(rows)
    anchors = build_anchor_map(rows)

    write_detail_csv_v2(rows, cells, os.path.join(OUT, "iter_per_round_by_source.csv"))
    write_pooled_csv(rows, os.path.join(OUT, "iter_per_round_pooled.csv"))
    write_cumulative_csv(rows, os.path.join(OUT, "iter_cumulative_pooled.csv"))
    write_latex(rows, cells, anchors, os.path.join(OUT, "iter_per_round_tables.tex"))
    plot_near_anchor_trends(rows, os.path.join(OUT, "near_anchor_by_round.png"))
    plot_histograms(rows, os.path.join(OUT, "histograms"))

    note = monotonicity_report(rows)
    with open(os.path.join(OUT, "WORDING_NOTE.md"), "w") as f:
        f.write(note)

    print(f"Wrote outputs to {OUT}/")
    print(note)


if __name__ == "__main__":
    main()
