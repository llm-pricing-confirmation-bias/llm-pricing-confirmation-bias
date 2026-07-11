r"""
dump.py — report tables for the refactored results
===================================================
Emits a single, compilable LaTeX document with three focused sections, read off
the cleaned/consolidated results in ``results/``:

  1. RQ1 — single-shot anchoring.  One block PER ANCHOR LEVEL holding a 2x2 grid:
        Claude stats | Claude regression   (top row)
        GPT stats    | GPT regression       (bottom row)

  2. First iteration (round 1).  One block PER ANCHOR LEVEL: Claude vs GPT side by
     side (descriptive cells: how far each authority pushes the model on the first
     "Are you sure?" challenge).

  3. Further iterations (rounds 2+).  Deliberately compact: one wide table per
     model giving, for every anchor level x authority, the share that
     SURVIVE / COLLAPSE-to-anchor / COLLAPSE within +/-$2 of the anchor.

Data sources (no extra dependencies — pure Python):
  results/single_turn/single_turn_trials.jsonl   (baseline CONTROL + static anchors)
  results/iterative/iterative_all_rounds.csv      (round 1 = first iteration, 2+ = further)

The RQ1 "regression" is an authority-vs-control group-means model
(price ~ C(source), control = reference). For that saturated design the HC3
heteroskedasticity-robust SE has a closed form (robust SE of a group mean =
s / sqrt(n-1)), so we compute it directly rather than pulling in statsmodels.

Usage:
    python -m analysis.dump_report                 # → reports/report_tables.tex
    python -m analysis.dump_report --out my.tex
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path as _Path
from statistics import mean, stdev

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import (
    ITERATIVE_DIR, PROMPT_VARIANTS_DIR, REPORT_TABLES_TEX, SINGLE_TURN_DIR,
)

csv.field_size_limit(10 ** 9)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

HERE = str(_ROOT)
RESULTS = str(_ROOT / "results")
SINGLE_TURN = str(SINGLE_TURN_DIR / "single_turn_trials.jsonl")
ITERATIVE = str(ITERATIVE_DIR / "iterative_all_rounds.csv")
VARIANT_AMBIGUOUS = str(PROMPT_VARIANTS_DIR / "ambiguous_ranges" / "trials.jsonl")
VARIANT_NOMARGIN = str(PROMPT_VARIANTS_DIR / "no_margin_no_cac" / "trials.jsonl")

MODEL_ORDER = ["anthropic/claude-haiku-4.5", "openai/gpt-5.4-mini"]
MODEL_PRETTY = {
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "openai/gpt-5.4-mini":        "GPT-5.4 Mini",
}

# Authority sources, grouped (midrules separate the groups in every table).
CONDITION_GROUPS = [
    ["INTERN", "COLLEAGUE", "MANAGER"],
    ["CONSULTING", "MCKINSEY"],
    ["UNLABELED", "AI"],
]
CONDITION_ORDER = [c for g in CONDITION_GROUPS for c in g]
CONDITION_PRETTY = {
    "INTERN": "Intern", "COLLEAGUE": "Colleague", "MANAGER": "Manager",
    "CONSULTING": "Consulting", "MCKINSEY": "McKinsey",
    "UNLABELED": "Unlabeled", "AI": "AI", "CONTROL": "Control",
}

# Anchor levels in the report's reading order (Mid first, then out to the extremes).
LEVEL_ORDER = [
    "MID", "LOW", "HIGH",
    "VERY_LOW", "VERY_HIGH",
    "UNREASONABLY_LOW", "UNREASONABLY_HIGH",
    "RIDICULOUSLY_HIGH",
]
LEVEL_PRETTY = {
    "MID": "Mid", "LOW": "Low", "HIGH": "High",
    "VERY_LOW": "Very Low", "VERY_HIGH": "Very High",
    "UNREASONABLY_LOW": "Unreasonably Low", "UNREASONABLY_HIGH": "Unreasonably High",
    "RIDICULOUSLY_HIGH": "Ridiculously High",
}

ANCHOR_BAND = 2.0   # dollars, for the "within +/-$2" collapse metric
EPS = 1e-9

# RQ1/RQ2: moment-matched subsample of the full n=1,000 control for n-matched
# comparisons against treatment cells of n=300. RQ3 uses the full pool; RQ4
# variant controls were collected at n=300 (no subsampling).
CONTROL_SUBSAMPLE = {"n": 300, "seed": 42, "match_moments": True}


def cond_pretty(c):
    return CONDITION_PRETTY.get(c, str(c).replace("_", " ").title())


# ══════════════════════════════════════════════════════════════════════════════
#  STATS  (pure Python; HC3 robust contrast in closed form)
# ══════════════════════════════════════════════════════════════════════════════
def pct_at(vals, anchor):
    if anchor is None or not vals:
        return None
    return 100 * sum(1 for v in vals if abs(v - anchor) < EPS) / len(vals)


def pct_within(vals, anchor, band=ANCHOR_BAND):
    if anchor is None or not vals:
        return None
    return 100 * sum(1 for v in vals if abs(v - anchor) <= band) / len(vals)


def _var_of_mean(vals):
    """HC3 robust variance of a group mean for a saturated group model: s^2/(n-1)."""
    n = len(vals)
    if n < 2:
        return None
    s = stdev(vals)
    return (s * s) / (n - 1)


def _z_p(z):
    """Two-sided p-value from a z statistic (normal approx; n is large here)."""
    return math.erfc(abs(z) / math.sqrt(2))


def contrast(auth_vals, ctrl_vals):
    """Authority-minus-control mean, HC3 robust SE, and two-sided p.
    Identical to the coefficient on the authority dummy in price ~ C(source)."""
    if len(auth_vals) < 2 or len(ctrl_vals) < 2:
        return None
    coef = mean(auth_vals) - mean(ctrl_vals)
    va, vc = _var_of_mean(auth_vals), _var_of_mean(ctrl_vals)
    se = math.sqrt(va + vc)
    if se < EPS:
        p = 0.0 if abs(coef) > EPS else 1.0
    else:
        p = _z_p(coef / se)
    return {"coef": coef, "se": se, "p": p}


def descriptive(vals, anchor=None, ctrl_vals=None):
    if not vals:
        return None
    n = len(vals)
    c = contrast(vals, ctrl_vals) if ctrl_vals else None
    return {
        "n": n, "mean": mean(vals), "sd": stdev(vals) if n > 1 else 0.0,
        "pct_anc": pct_at(vals, anchor), "pct_anc2": pct_within(vals, anchor),
        "p": c["p"] if c else None,
    }


def subsample_indices(values, k, seed=42, match_moments=False, restarts=20, swap_iters=15000):
    """Pick k indices from values. If match_moments, hill-climb to match mean and SD."""
    import random
    n = len(values)
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []

    rng = random.Random(seed)
    full_mean = mean(values)
    full_sd = stdev(values) if n > 1 else 0.0

    def objective(idxs):
        sub = [values[i] for i in idxs]
        m = mean(sub)
        s = stdev(sub) if len(sub) > 1 else 0.0
        return (m - full_mean) ** 2 + (s - full_sd) ** 2

    best = None
    best_obj = float("inf")
    for _ in range(restarts if match_moments else 1):
        chosen = rng.sample(range(n), k)
        if match_moments and k >= 2:
            in_list = list(chosen)
            out_list = [i for i in range(n) if i not in chosen]
            local = set(chosen)
            local_obj = objective(local)
            for _ in range(swap_iters):
                if not out_list:
                    break
                i_idx = rng.choice(in_list)
                o_idx = rng.choice(out_list)
                trial = set(local)
                trial.remove(i_idx)
                trial.add(o_idx)
                obj = objective(trial)
                if obj < local_obj - 1e-12:
                    local = trial
                    local_obj = obj
                    in_list = list(local)
                    out_list = [i for i in range(n) if i not in local]
            chosen = sorted(local)
        obj = objective(chosen)
        if obj < best_obj:
            best_obj = obj
            best = chosen
    return best


def subsample_control(vals, cfg=None):
    """Return control prices, optionally subsampled per CONTROL_SUBSAMPLE."""
    cfg = cfg or CONTROL_SUBSAMPLE
    k = cfg.get("n")
    if k is None or len(vals) <= k:
        return vals, False
    idxs = subsample_indices(vals, k, cfg["seed"], cfg["match_moments"])
    return [vals[i] for i in idxs], True


def r_squared(groups):
    """R^2 of the group-means model across pooled values of {label: [vals]}."""
    pooled = [v for vs in groups.values() for v in vs]
    if len(pooled) < 2:
        return None
    gm = mean(pooled)
    ss_tot = sum((v - gm) ** 2 for v in pooled)
    if ss_tot < EPS:
        return None
    ss_res = 0.0
    for vs in groups.values():
        if not vs:
            continue
        m = mean(vs)
        ss_res += sum((v - m) ** 2 for v in vs)
    return 1 - ss_res / ss_tot


# ══════════════════════════════════════════════════════════════════════════════
#  LOADING + AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
def load_single_turn(path):
    """RQ1 aggregates: control prices per model, anchored prices per (model,level,cond),
    and the anchor dollar value per (model,level)."""
    ctrl = defaultdict(list)
    cells = defaultdict(list)              # (model, level, cond) -> [price]
    anchor = {}                            # (model, level) -> value
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            price = r.get("price_recommendation")
            if price is None or r.get("assertion") not in ("na", "standard"):
                continue
            model = r["model"]
            cond = r.get("condition_id")
            if cond == "CONTROL":
                ctrl[model].append(float(price))
                continue
            level = r.get("anchor_level")
            cells[(model, level, cond)].append(float(price))
            av = r.get("anchor_value")
            if av is not None:
                anchor[(model, level)] = float(av)
    return ctrl, cells, anchor


def load_iterative(path):
    """First-iteration (round 1) price cells and the further-iteration terminal cohort.

    Returns:
      fi_cells[(model,level,cond)]  -> [round-1 price_after]
      anchor[(model,level)]         -> anchor value
      term_cells[(model,level,cond)]-> [terminal price for conversations that reached round >= 2]
    """
    fi_cells = defaultdict(list)
    term_cells = defaultdict(list)
    anchor = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            model, level, cond = r["model"], r["anchor_level"], r["condition"]
            av = r.get("anchor_value")
            if av not in (None, ""):
                anchor[(model, level)] = float(av)
            rnd = int(r["round"])
            pa = r.get("price_after")
            pa = float(pa) if pa not in (None, "") else None
            if rnd == 1 and pa is not None:
                fi_cells[(model, level, cond)].append(pa)
            # terminal price of conversations that survived past the first challenge
            if r.get("is_last_observed_round") == "True" and rnd >= 2 and pa is not None:
                term_cells[(model, level, cond)].append((pa, anchor.get((model, level))))
    return fi_cells, anchor, term_cells


# ══════════════════════════════════════════════════════════════════════════════
#  LATEX PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════
def _f(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def _pct(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}\\%"


def _signed(x, nd=2):
    if x is None:
        return "--"
    return f"+{x:.{nd}f}" if x >= 0 else f"{x:.{nd}f}"


def _money(v):
    if v is None:
        return "?"
    v = float(v)
    s = f"{int(v):,}" if v.is_integer() else f"{v:,.2f}"
    return s.replace(",", "{,}")


def _n(n):
    return "--" if n is None else f"{int(n):,}".replace(",", "{,}")


def _stars(p):
    if p is None:
        return "--"
    if p < 0.001:
        return r"\sig{***}"
    if p < 0.01:
        return r"\sig{**}"
    if p < 0.05:
        return r"\sig{*}"
    return r"\sig{ns}"


def _pval(p):
    if p is None:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


GRAY = "gray!22"


def shade_row(cells):
    """A fully-shaded control row using per-cell \\cellcolor (strictly local, so
    the shade can never bleed onto following rows the way \\rowcolor can in
    nested tabulars)."""
    return " & ".join(rf"\cellcolor{{{GRAY}}}{c}" for c in cells) + r" \\"


def _present_groups(rows_by_cond):
    """Yield (group_index, [conds present]) so we can place midrules correctly."""
    out = []
    for gi, group in enumerate(CONDITION_GROUPS):
        present = [c for c in group if rows_by_cond.get(c)]
        if present:
            out.append((gi, present))
    return out


# ── descriptive tabular (control row + authority rows) ───────────────────────
def desc_tabular(rows_by_cond, ctrl_vals, anchor):
    cs = descriptive(ctrl_vals, anchor=anchor) if ctrl_vals else None
    out = [r"\begin{tabular}{@{}l r r r r r r c@{}}", r"\toprule",
           r"Source & $n$ & Mean & SD & \%anc & \%$\pm$2 & $p$ & sig \\", r"\midrule"]
    if cs:
        out.append(shade_row(["Control", _n(cs['n']), _f(cs['mean']), _f(cs['sd']),
                              _pct(cs['pct_anc']), _pct(cs['pct_anc2']), "--", "--"]))
        out.append(r"\midrule")
    groups = _present_groups(rows_by_cond)
    for j, (gi, conds) in enumerate(groups):
        for cond in conds:
            s = descriptive(rows_by_cond[cond], anchor=anchor, ctrl_vals=ctrl_vals)
            out.append(rf"{cond_pretty(cond)} & {_n(s['n'])} & {_f(s['mean'])} & {_f(s['sd'])} & "
                       rf"{_pct(s['pct_anc'])} & {_pct(s['pct_anc2'])} & {_pval(s['p'])} & "
                       rf"{_stars(s['p'])} \\")
        if j < len(groups) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


# ── regression tabular (authority vs control, HC3 robust SE) ─────────────────
def reg_tabular(rows_by_cond, ctrl_vals):
    if len(ctrl_vals) < 2:
        return None
    groups = {"CONTROL": ctrl_vals}
    for c in CONDITION_ORDER:
        if rows_by_cond.get(c):
            groups[c] = rows_by_cond[c]
    if len(groups) < 2:
        return None
    out = [r"\begin{tabular}{@{}l r r r c@{}}", r"\toprule",
           r"Authority & Coef & SE & $p$ & sig \\", r"\midrule"]
    cm = mean(ctrl_vals)
    cse = math.sqrt(_var_of_mean(ctrl_vals))
    out.append(shade_row(["Control (icpt)", _f(cm), _f(cse), "--", "--"]))
    pg = _present_groups(rows_by_cond)
    for j, (gi, conds) in enumerate(pg):
        out.append(r"\midrule")
        for cond in conds:
            c = contrast(rows_by_cond[cond], ctrl_vals)
            if c is None:
                continue
            out.append(rf"{cond_pretty(cond)} & {_signed(c['coef'])} & {_f(c['se'])} & "
                       rf"{_pval(c['p'])} & {_stars(c['p'])} \\")
    r2 = r_squared(groups)
    nobs = sum(len(v) for v in groups.values())
    out.append(r"\midrule")
    out.append(rf"\multicolumn{{5}}{{c}}{{\scriptsize $n={_n(nobs)}$, $R^2={_f(r2, 3)}$}} \\")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


# ── minipage grids ───────────────────────────────────────────────────────────
def _cell(label, body, width=0.48):
    if not body:
        body = r"\textit{\scriptsize no data}"
    return (rf"\begin{{minipage}}[t]{{{width}\textwidth}}\centering" + "\n"
            + rf"\textbf{{\footnotesize {label}}}\par\vspace{{2pt}}" + "\n"
            + body + "\n" + r"\end{minipage}")


def grid_2x2(caption, tl, tr, bl, br):
    """4 minipages (label, body) in a 2x2 float with one caption."""
    w = 0.49
    return "\n".join([
        r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{7pt}",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{@{}c@{\hspace{0.5em}}c@{}}",
        _cell(*tl, width=w) + " &", _cell(*tr, width=w) + r" \\[1.1em]",
        _cell(*bl, width=w) + " &", _cell(*br, width=w) + r" \\",
        r"\end{tabular}", r"\end{table}", "",
    ])


def grid_1x2(caption, left, right):
    return "\n".join([
        r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{@{}c@{\hspace{1.4em}}c@{}}",
        _cell(*left) + " &", _cell(*right) + r" \\",
        r"\end{tabular}", r"\end{table}", "",
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT PARTS
# ══════════════════════════════════════════════════════════════════════════════
def preamble():
    return r"""\documentclass[10pt]{article}
\usepackage[margin=0.55in]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{caption}
\usepackage{float}
\usepackage{graphicx}
\usepackage{enumitem}
\captionsetup{font=small,labelfont=bf}
\newcommand{\sig}[1]{\textsuperscript{#1}}
\renewcommand{\arraystretch}{1.1}
\title{\vspace{-1.5em}LLM Pricing: Anchoring \& Authority --- Result Tables}
\date{}
\begin{document}
\maketitle
\vspace{-2.5em}
"""


def reading_guide():
    return r"""
\section*{Reading guide}
\begin{description}[leftmargin=2.4cm,font=\normalfont\bfseries]
  \item[Models] Claude Haiku 4.5 and GPT-5.4 Mini; standard assertion throughout.
  \item[Anchor levels] Eight, in order: Mid, Low, High, Very Low, Very High,
    Unreasonably Low/High, Ridiculously High. Each level is one dollar value per
    model (no plausible/unreasonable split; each anchor is run once per model).
  \item[\%anc] Share landing exactly on the anchor. \quad
        \textbf{\%$\pm$2} Share within $\pm\$2$ of the anchor.
  \item[$n$] Treatment cells $n{=}300$. Single-shot control (RQ1/RQ2): moment-matched
        subsample of $n{=}300$ drawn from the full $n{=}1{,}000$ unprompted baseline
        (seed 42). First-iteration reference control (RQ3): full $n{=}1{,}000$.
        Prompt-variant controls (RQ4): $n{=}300$ collected directly under each variant
        prompt (no subsampling).
  \item[$p$ / Coef] Single-shot: authority mean vs.\ control, HC3 heteroskedasticity-robust.
        Stars on raw $p$: \sig{*}$<.05$, \sig{**}$<.01$, \sig{***}$<.001$, ns otherwise.
  \item[Sections] (1) Single-shot anchoring: per anchor, Claude stats/regression on top,
        GPT below. (2) First iteration (round 1): Claude vs.\ GPT per anchor.
        (3) Further iterations (rounds 2+): per anchor $\times$ authority share that
        \emph{survive} / \emph{collapse to anchor} / \emph{collapse within $\pm\$2$}.
        (4--5) Prompt-sensitivity variants (ambiguous inputs; no margin / no CAC),
        same single-shot design with a modified prompt, three core anchors.
\end{description}
"""


def _single_shot_grids(ctrl, cells, anchor, levels, cap_prefix=""):
    """Per-anchor 2x2 grids (Claude stats|reg on top, GPT stats|reg below)."""
    out = []
    for level in levels:
        blocks = {}
        for model in MODEL_ORDER:
            rbc = {c: cells.get((model, level, c), []) for c in CONDITION_ORDER}
            if not any(rbc.values()):
                blocks[model] = (None, None)
                continue
            a = anchor.get((model, level))
            blocks[model] = (desc_tabular(rbc, ctrl.get(model, []), a),
                             reg_tabular(rbc, ctrl.get(model, [])))
        if all(d is None for d, _ in blocks.values()):
            continue
        ac = anchor.get((MODEL_ORDER[0], level))
        ag = anchor.get((MODEL_ORDER[1], level))
        cap = (rf"{cap_prefix}{LEVEL_PRETTY[level]} anchor "
               rf"(Claude \${_money(ac)} / GPT \${_money(ag)}).")
        cl, gp = blocks[MODEL_ORDER[0]], blocks[MODEL_ORDER[1]]
        out.append(grid_2x2(
            cap,
            (f"{MODEL_PRETTY[MODEL_ORDER[0]]} --- stats", cl[0]),
            (f"{MODEL_PRETTY[MODEL_ORDER[0]]} --- regression", cl[1]),
            (f"{MODEL_PRETTY[MODEL_ORDER[1]]} --- stats", gp[0]),
            (f"{MODEL_PRETTY[MODEL_ORDER[1]]} --- regression", gp[1]),
        ))
    return out


def section_single_shot(ctrl, cells, anchor):
    out = [r"\clearpage",
           r"\section{Single-shot anchoring (full-information prompt)}",
           r"\noindent The anchor is shown \emph{before} the model answers. One block "
           r"per anchor level: descriptive stats and the authority-vs-control regression "
           r"(HC3 robust SE), Claude on top, GPT below. "
           r"\textbf{Control baseline:} moment-matched subsample of $n{=}300$ from the "
           r"full $n{=}1{,}000$ unprompted control pool (seed 42), for fair "
           r"$n$-matched comparisons against $n{=}300$ treatment cells.",
           ""]
    out += _single_shot_grids(ctrl, cells, anchor, LEVEL_ORDER)
    return "\n".join(out)


def _ctrl_compare_table(full_ctrl, var_ctrl, label):
    """Compact unprompted-control comparison: full-information vs the variant prompt."""
    out = [r"\begin{table}[H]\centering\footnotesize",
           rf"\caption{{Unprompted control: full-information vs.\ {label} prompt "
           rf"(no anchor; does the prompt change shift the baseline?).}}",
           r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
           r"Model & Prompt & $n$ & Mean & SD \\", r"\midrule"]
    for mi, model in enumerate(MODEL_ORDER):
        for name, vals in [("Full information", full_ctrl.get(model, [])),
                           (label, var_ctrl.get(model, []))]:
            if not vals:
                continue
            sd = stdev(vals) if len(vals) > 1 else 0.0
            out.append(rf"{MODEL_PRETTY[model]} & {name} & {_n(len(vals))} & "
                       rf"{_f(mean(vals))} & {_f(sd)} \\")
        if mi < len(MODEL_ORDER) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(out)


def section_prompt_variant(title, label, ctrl, cells, anchor, full_ctrl):
    """A single-shot prompt-robustness variant, presented like the main section but
    against the variant's own unprompted control."""
    out = [r"\clearpage",
           rf"\section{{{title}}}",
           rf"\noindent Same single-shot design with a modified product prompt "
           rf"({label}); three core anchors only. Each level's shaded Control row is "
           r"this prompt's own unanchored baseline ($n{=}300$ per model, collected "
           r"directly under the variant prompt---not subsampled). The comparison table "
           r"below contrasts this variant baseline against the full-information "
           r"control pool ($n{=}1{,}000$).",
           "",
           _ctrl_compare_table(full_ctrl, ctrl, label),
           ""]
    out += _single_shot_grids(ctrl, cells, anchor, ["MID", "LOW", "HIGH"],
                              cap_prefix=rf"{label} --- ")
    return "\n".join(out)


def _first_iter_tabular(rows_by_cond, ctrl_vals, anchor):
    cs = descriptive(ctrl_vals, anchor=anchor) if ctrl_vals else None
    out = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
           r"Source & $n$ & Mean & SD & \%anc & \%$\pm$2 \\", r"\midrule"]
    if cs:
        out.append(shade_row(["Control", _n(cs['n']), _f(cs['mean']), _f(cs['sd']),
                              _pct(cs['pct_anc']), _pct(cs['pct_anc2'])]))
        out.append(r"\midrule")
    groups = _present_groups(rows_by_cond)
    for j, (gi, conds) in enumerate(groups):
        for cond in conds:
            s = descriptive(rows_by_cond[cond], anchor=anchor)
            out.append(rf"{cond_pretty(cond)} & {_n(s['n'])} & {_f(s['mean'])} & {_f(s['sd'])} & "
                       rf"{_pct(s['pct_anc'])} & {_pct(s['pct_anc2'])} \\")
        if j < len(groups) - 1:
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


def section_first_iter(fi_cells, anchor, ctrl):
    out = [r"\clearpage",
           r"\section{First iteration --- round 1 (``Are you sure? In fact, \ldots'')}",
           r"\noindent The anchor is injected \emph{after} a committed first answer; "
           r"the price below is the model's revised second-pass price. Claude vs.\ GPT "
           r"per anchor. Control = the full $n{=}1{,}000$ single-shot unprompted "
           r"baseline for that model (no anchor; no subsampling).",
           ""]
    for level in LEVEL_ORDER:
        sides = {}
        for model in MODEL_ORDER:
            rbc = {c: fi_cells.get((model, level, c), []) for c in CONDITION_ORDER}
            sides[model] = _first_iter_tabular(rbc, ctrl.get(model, []),
                                               anchor.get((model, level))) \
                if any(rbc.values()) else None
        if all(v is None for v in sides.values()):
            continue
        ac = anchor.get((MODEL_ORDER[0], level))
        ag = anchor.get((MODEL_ORDER[1], level))
        cap = (rf"First iteration --- {LEVEL_PRETTY[level]} anchor "
               rf"(Claude \${_money(ac)} / GPT \${_money(ag)}).")
        out.append(grid_1x2(
            cap,
            (MODEL_PRETTY[MODEL_ORDER[0]], sides[MODEL_ORDER[0]]),
            (MODEL_PRETTY[MODEL_ORDER[1]], sides[MODEL_ORDER[1]]),
        ))
    return "\n".join(out)


def _outcome_pcts(terminals):
    """(survive%, collapse@anchor%, collapse within ±2%) for a list of (price, anchor)."""
    rows = [(p, a) for p, a in terminals if a is not None]
    if not rows:
        return None
    n = len(rows)
    at = sum(1 for p, a in rows if abs(p - a) < EPS)
    near = sum(1 for p, a in rows if abs(p - a) <= ANCHOR_BAND)
    return {"n": n, "surv": 100 * (n - near) / n,
            "anc": 100 * at / n, "anc2": 100 * near / n}


def section_further(term_cells):
    out = [r"\clearpage",
           r"\section{Further iterations --- rounds 2+ (sustained pressure)}",
           r"\noindent Cohort: conversations that did \emph{not} settle on the first "
           r"challenge and were pushed again (round $\geq 2$). For each anchor "
           r"$\times$ authority: share whose \emph{final} price \textbf{survives} "
           r"(more than \$2 from the anchor), \textbf{collapses to anchor} (exactly on it), "
           r"or \textbf{collapses within \$\boldmath$\pm2$}. By construction "
           r"survive\% $=100-{}$($\pm2$)\%, and anchor\% $\leq$ ($\pm2$)\%.",
           ""]
    for model in MODEL_ORDER:
        levels = [lv for lv in LEVEL_ORDER
                  if any(term_cells.get((model, lv, c)) for c in CONDITION_ORDER)]
        if not levels:
            continue
        out.append(rf"\subsection*{{{MODEL_PRETTY[model]}}}")
        # header: per level, 3 subcols (surv / anc / ±2)
        head1 = [r"Authority"]
        head2 = [r""]
        cmids = []
        for i, lv in enumerate(levels):
            av = term_cells_anchor(term_cells, model, lv)
            head1.append(rf"\multicolumn{{3}}{{c}}{{{LEVEL_PRETTY[lv]}~(\${_money(av)})}}")
            start = 2 + 3 * i
            cmids.append(rf"\cmidrule(lr){{{start}-{start + 2}}}")
            head2 += [r"sv", r"@a", r"$\pm$2"]
        out.append(r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}")
        out.append(rf"\caption{{{MODEL_PRETTY[model]} --- further-iteration outcomes "
                   rf"(sv=survive\%, @a=collapse-to-anchor\%, $\pm$2=collapse within \$2\%).}}")
        out.append(r"\resizebox{\textwidth}{!}{%")
        out.append(r"\begin{tabular}{l" + " ccc" * len(levels) + "}")
        out.append(r"\toprule")
        out.append(" & ".join(head1) + r" \\")
        out.append(" ".join(cmids))
        out.append(" & ".join(head2) + r" \\")
        out.append(r"\midrule")
        for gi, group in enumerate(CONDITION_GROUPS):
            for cond in group:
                row = [cond_pretty(cond)]
                for lv in levels:
                    o = _outcome_pcts(term_cells.get((model, lv, cond), []))
                    if o is None:
                        row += ["--", "--", "--"]
                    else:
                        row += [_pct(o["surv"], 0), _pct(o["anc"], 0), _pct(o["anc2"], 0)]
                out.append(" & ".join(row) + r" \\")
            if gi < len(CONDITION_GROUPS) - 1:
                out.append(r"\midrule")
        # Total + cohort n
        out.append(r"\midrule")
        trow = [r"\textbf{Total}"]
        nrow = [r"\textit{n}"]
        for lv in levels:
            allt = [t for c in CONDITION_ORDER for t in term_cells.get((model, lv, c), [])]
            o = _outcome_pcts(allt)
            if o is None:
                trow += ["--", "--", "--"]
                nrow += [r"\multicolumn{3}{c}{--}"]
            else:
                trow += [_pct(o["surv"], 0), _pct(o["anc"], 0), _pct(o["anc2"], 0)]
                nrow.append(rf"\multicolumn{{3}}{{c}}{{{_n(o['n'])}}}")
        out.append(" & ".join(trow) + r" \\")
        out.append(" & ".join(nrow) + r" \\")
        out.append(r"\bottomrule\end{tabular}}")
        out.append(r"\end{table}")
        out.append("")
    return "\n".join(out)


def term_cells_anchor(term_cells, model, level):
    for c in CONDITION_ORDER:
        lst = term_cells.get((model, level, c))
        if lst and lst[0][1] is not None:
            return lst[0][1]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Emit the LaTeX result tables.")
    ap.add_argument("--single-turn", default=SINGLE_TURN)
    ap.add_argument("--iterative", default=ITERATIVE)
    ap.add_argument("--variant-ambiguous", default=VARIANT_AMBIGUOUS)
    ap.add_argument("--variant-nomargin", default=VARIANT_NOMARGIN)
    ap.add_argument("--out", default=str(REPORT_TABLES_TEX))
    args = ap.parse_args()

    ctrl_full, st_cells, st_anchor = load_single_turn(args.single_turn)
    ctrl_rq1 = {}
    for model in MODEL_ORDER:
        sub, _ = subsample_control(ctrl_full.get(model, []))
        ctrl_rq1[model] = sub
    fi_cells, it_anchor, term_cells = load_iterative(args.iterative)
    amb_ctrl, amb_cells, amb_anchor = load_single_turn(args.variant_ambiguous)
    nm_ctrl, nm_cells, nm_anchor = load_single_turn(args.variant_nomargin)

    doc = "\n".join([
        preamble(),
        reading_guide(),
        section_single_shot(ctrl_rq1, st_cells, st_anchor),
        section_first_iter(fi_cells, it_anchor, ctrl_full),
        section_further(term_cells),
        section_prompt_variant(
            r"Prompt sensitivity --- ambiguous (range-based) inputs",
            "Ambiguous", amb_ctrl, amb_cells, amb_anchor, ctrl_full),
        section_prompt_variant(
            r"Prompt sensitivity --- no margin / no CAC",
            "No-margin/no-CAC", nm_ctrl, nm_cells, nm_anchor, ctrl_full),
        r"\end{document}",
        "",
    ])
    with open(args.out, "w") as f:
        f.write(doc)

    n_first = sum(len(v) for v in fi_cells.values())
    n_term = sum(len(v) for v in term_cells.values())
    print(f"Wrote {args.out}")
    print(f"  single-shot control (full pool): " +
          ", ".join(f"{MODEL_PRETTY[m]} n={len(ctrl_full.get(m, []))}" for m in MODEL_ORDER))
    print(f"  RQ1/RQ2 control (subsampled): " +
          ", ".join(f"{MODEL_PRETTY[m]} n={len(ctrl_rq1.get(m, []))}" for m in MODEL_ORDER))
    print(f"  first-iteration trials: {n_first:,}")
    print(f"  further-iteration conversations (round>=2): {n_term:,}")
    print(f"  Compile with:  pdflatex {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
