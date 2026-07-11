"""
Publication-ready bimodality histograms for iterative-design terminal prices.

Reads iterative_all_rounds.csv and writes PDF + PNG figures to:
  results/iterative/per_round_analysis/bimodality_plots/

Plots:
  1. Final price distribution — Very High & Unreasonably High (bimodality check)
  2. Price shift (final − round-1) — same anchors (rigorous bimodality test)
  3. Negative control — Low, Mid, High (unimodal anchor collapse)
  4. Asymmetry — Unreasonably Low vs Unreasonably High
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from statistics import median

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config.paths import BIMODALITY_PLOTS_DIR, ITERATIVE_DIR

HERE = str(_ROOT)
DATA = str(ITERATIVE_DIR / "iterative_all_rounds.csv")
OUT = str(BIMODALITY_PLOTS_DIR)

CLAUDE = "anthropic/claude-haiku-4.5"
CONTROL_MEDIAN = 65.0
BIN_WIDTH = 1.0

LEVEL_PRETTY = {
    "VERY_HIGH": "Very High",
    "UNREASONABLY_HIGH": "Unreasonably High",
    "UNREASONABLY_LOW": "Unreasonably Low",
    "LOW": "Low",
    "MID": "Mid",
    "HIGH": "High",
}

# Muted, colorblind-friendly palette
ANCHOR_COLOR = "#C44E52"
BASELINE_COLOR = "#55A868"
BAR_COLOR = "#4C72B0"
BAR_EDGE = "#FFFFFF"
SHIFT_ZERO_COLOR = "#8172B2"


@dataclass
class Conversation:
    conversation_key: str
    anchor_level: str
    anchor_value: float
    round1_price: float
    final_price: float
    final_round: int


def _setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_conversations(path: str, *, further_only: bool) -> list[Conversation]:
    """Load per-conversation round-1 and terminal prices for Claude, standard assertion."""
    r1: dict[str, dict] = {}
    terminal: dict[str, dict] = {}

    with open(path) as f:
        for r in csv.DictReader(f):
            if r["model"] != CLAUDE or r.get("assertion") != "standard":
                continue
            key = r["conversation_key"]
            rnd = int(r["round"])
            pa = r.get("price_after")
            if pa in (None, ""):
                continue
            pa = float(pa)

            if rnd == 1:
                r1[key] = {
                    "anchor_level": r["anchor_level"],
                    "anchor_value": float(r["anchor_value"]),
                    "round1_price": pa,
                }

            if r.get("is_last_observed_round") != "True":
                continue
            if further_only and rnd < 2:
                continue

            terminal[key] = {
                "anchor_level": r["anchor_level"],
                "anchor_value": float(r["anchor_value"]),
                "final_price": pa,
                "final_round": rnd,
            }

    out: list[Conversation] = []
    for key, t in terminal.items():
        if key not in r1:
            continue
        out.append(Conversation(
            conversation_key=key,
            anchor_level=t["anchor_level"],
            anchor_value=t["anchor_value"],
            round1_price=r1[key]["round1_price"],
            final_price=t["final_price"],
            final_round=t["final_round"],
        ))
    return out


def _dollar_bins(lo: float, hi: float, width: float = BIN_WIDTH) -> np.ndarray:
    lo_bin = np.floor(lo)
    hi_bin = np.ceil(hi)
    return np.arange(lo_bin, hi_bin + width, width)


def _hist_bars(ax, values: np.ndarray, *, lo: float | None = None, hi: float | None = None):
    """Draw $1 histogram bars; returns bin edges used."""
    if len(values) == 0:
        return None
    lo = float(np.min(values)) if lo is None else lo
    hi = float(np.max(values)) if hi is None else hi
    bins = _dollar_bins(lo, hi)
    ax.hist(
        values, bins=bins, color=BAR_COLOR, edgecolor=BAR_EDGE,
        linewidth=0.4, alpha=0.92, rwidth=0.95,
    )
    return bins


def _hist_ax(ax, values: np.ndarray, *, xlabel: str, title: str, panel: str,
             xlim: tuple[float, float] | None = None):
    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    lo, hi = float(np.min(values)), float(np.max(values))
    if xlim:
        lo, hi = xlim
    _hist_bars(ax, values, lo=lo, hi=hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, pad=8)
    ax.text(
        -0.12, 1.06, panel, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="left",
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    if xlim:
        ax.set_xlim(xlim)


def _save(fig: plt.Figure, stem: str):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)


def _add_survivor_inset(ax, prices: np.ndarray, anchor: float, *, xlo: float = 45, xhi: float = 82):
    """Zoom inset for the holdout cluster when the anchor spike dominates the scale."""
    mask = (prices >= xlo) & (prices <= xhi)
    if mask.sum() == 0:
        return
    inset = ax.inset_axes([0.48, 0.48, 0.48, 0.48])
    inset.set_facecolor("#FAFAFA")
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#888888")
    _hist_bars(inset, prices[mask], lo=xlo, hi=xhi)
    inset.axvline(CONTROL_MEDIAN, color=BASELINE_COLOR, ls="--", lw=1.0)
    inset.set_xlim(xlo, xhi)
    inset.set_ylim(0, inset.get_ylim()[1] * 1.15)
    inset.tick_params(labelsize=7, length=2, pad=1)
    inset.set_title("Holdout cluster", fontsize=7.5, pad=3)
    ax.indicate_inset_zoom(inset, edgecolor="#888888", linewidth=0.8, alpha=0.9)


def plot1_final_price(convs: list[Conversation]):
    """Primary bimodality check: final price at Very High and Unreasonably High."""
    levels = ["VERY_HIGH", "UNREASONABLY_HIGH"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    panels = ["A", "B"]

    for level, ax, panel in zip(levels, axes, panels):
        sub = [c for c in convs if c.anchor_level == level]
        prices = np.array([c.final_price for c in sub])
        anchor = sub[0].anchor_value if sub else 0
        n = len(prices)

        _hist_ax(
            ax, prices,
            xlabel="Final recommended price ($)",
            title=f"{LEVEL_PRETTY[level]} anchor (${int(anchor)}),  $n={n:,}$",
            panel=panel,
        )
        ax.axvline(anchor, color=ANCHOR_COLOR, ls="-", lw=1.4, label=f"Anchor (${int(anchor)})")
        ax.axvline(
            CONTROL_MEDIAN, color=BASELINE_COLOR, ls="--", lw=1.2,
            label=f"Control median (${int(CONTROL_MEDIAN)})",
        )
        if level == "VERY_HIGH":
            _add_survivor_inset(ax, prices, anchor)
        if panel == "A":
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1)

    fig.suptitle(
        "Terminal price distribution after further iterations (Claude, pooled authorities)",
        y=1.02, fontsize=11,
    )
    fig.text(
        0.5, -0.02,
        r"$1 bins; terminal cohort with $\geq 2$ rounds (conversations that did not snap on first pushback).",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout()
    _save(fig, "plot1_final_price_bimodality")


def plot2_price_shift(convs: list[Conversation]):
    """Shift from round-1 price: rules out pooling artifact."""
    levels = ["VERY_HIGH", "UNREASONABLY_HIGH"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    panels = ["A", "B"]

    for level, ax, panel in zip(levels, axes, panels):
        sub = [c for c in convs if c.anchor_level == level]
        shifts = np.array([c.final_price - c.round1_price for c in sub])
        anchor = sub[0].anchor_value if sub else 0
        n = len(shifts)
        median_shift_to_anchor = median([anchor - c.round1_price for c in sub]) if sub else 0

        _hist_ax(
            ax, shifts,
            xlabel="Price shift: final $-$ round-1 ($)",
            title=f"{LEVEL_PRETTY[level]} anchor (${int(anchor)}),  $n={n:,}$",
            panel=panel,
        )
        ax.axvline(0, color=SHIFT_ZERO_COLOR, ls="-", lw=1.4, label="Held round-1 price (shift $= 0$)")
        ax.axvline(
            median_shift_to_anchor, color=ANCHOR_COLOR, ls="--", lw=1.2,
            label=f"Median shift to anchor (${median_shift_to_anchor:+.0f})",
        )
        if panel == "A":
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1)

    fig.suptitle(
        "Price shift relative to each conversation's round-1 recommendation",
        y=1.02, fontsize=11,
    )
    fig.text(
        0.5, -0.02,
        r"$1 bins; shift $= 0$ indicates the model held its round-1 price; positive shift toward anchor adoption.",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout()
    _save(fig, "plot2_price_shift_bimodality")


def plot3_negative_control(convs: list[Conversation]):
    """Low / Mid / High anchors: unimodal collapse (negative control)."""
    levels = ["LOW", "MID", "HIGH"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    panels = ["A", "B", "C"]

    for level, ax, panel in zip(levels, axes, panels):
        sub = [c for c in convs if c.anchor_level == level]
        prices = np.array([c.final_price for c in sub])
        anchor = sub[0].anchor_value if sub else 0
        n = len(prices)
        pad = 4

        _hist_ax(
            ax, prices,
            xlabel="Final recommended price ($)",
            title=f"{LEVEL_PRETTY[level]} anchor (${int(anchor)}),  $n={n:,}$",
            panel=panel,
            xlim=(anchor - pad, anchor + pad),
        )
        ax.axvline(anchor, color=ANCHOR_COLOR, ls="-", lw=1.4, label=f"Anchor (${int(anchor)})")

    axes[0].legend(loc="upper right")
    fig.suptitle(
        "Negative control: near-complete anchor adoption (Claude, all terminal conversations)",
        y=1.02, fontsize=11,
    )
    fig.text(
        0.5, -0.02,
        r"$1 bins; x-axis zoomed to $\pm \$4$ around each anchor.",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout()
    _save(fig, "plot3_negative_control_unimodal")


def plot4_asymmetry(convs: list[Conversation]):
    """Unreasonably Low vs Unreasonably High — asymmetry in survival."""
    levels = ["UNREASONABLY_LOW", "UNREASONABLY_HIGH"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    panels = ["A", "B"]

    for level, ax, panel in zip(levels, axes, panels):
        sub = [c for c in convs if c.anchor_level == level]
        prices = np.array([c.final_price for c in sub])
        anchor = sub[0].anchor_value if sub else 0
        n = len(prices)

        at_anchor = sum(1 for p in prices if abs(p - anchor) < 0.5)
        near_r1 = sum(1 for c in sub if abs(c.final_price - c.round1_price) <= 2)
        pct_anchor = 100 * at_anchor / n if n else 0
        pct_survive = 100 * near_r1 / n if n else 0

        _hist_ax(
            ax, prices,
            xlabel="Final recommended price ($)",
            title=(
                f"{LEVEL_PRETTY[level]} anchor (${int(anchor)}),  $n={n:,}$\n"
                f"at anchor: {pct_anchor:.0f}\\%  |  near round-1: {pct_survive:.0f}\\%"
            ),
            panel=panel,
        )
        ax.axvline(anchor, color=ANCHOR_COLOR, ls="-", lw=1.4, label=f"Anchor (${int(anchor)})")
        ax.axvline(
            CONTROL_MEDIAN, color=BASELINE_COLOR, ls="--", lw=1.2,
            label=f"Control median (${int(CONTROL_MEDIAN)})",
        )
        if panel == "A":
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1)

    fig.suptitle(
        "Asymmetric anchor adoption at extreme anchors (Claude, further iterations)",
        y=1.05, fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "plot4_asymmetry_extreme_anchors")


def main():
    _setup_style()
    further = load_conversations(DATA, further_only=True)
    all_terminal = load_conversations(DATA, further_only=False)

    print(f"Further-iteration cohort: {len(further):,} conversations")
    print(f"All-terminal cohort:      {len(all_terminal):,} conversations")

    plot1_final_price(further)
    plot2_price_shift(further)
    plot3_negative_control(all_terminal)
    plot4_asymmetry(further)

    print(f"Wrote figures to {OUT}/")


if __name__ == "__main__":
    main()
