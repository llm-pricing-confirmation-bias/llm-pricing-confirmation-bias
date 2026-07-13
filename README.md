Code, prompts, and results for the paper **Confirmation Bias in LLM Pricing Recommendations**.

Use `python main.py` for everything: regenerating the paper’s tables and figures
from the released data, or running new experiments through OpenRouter.

---

## Setup

```bash
cd <this-directory>
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env
```

For live API runs, set `OPENROUTER_API_KEY` in `.env`. Reproducing tables and
figures from the released archive does not need a key.

```bash
python main.py --help
```

---

## Get the data

Trial-level outputs are large and are **not** shipped in this repository.
Download the curated archive and place it here as `results/`:

1. Download from Google Drive:  
   **[to be added]**
2. Unpack so the directory layout looks like:

```text
results/
├── single_turn/
├── iterative/
├── prompt_variants/
└── analysis/
```

From this package root that means:

```bash
mkdir -p results
# then unzip / move the archive contents into ./results/
```

`results/` is gitignored. Do not commit it.

---

## Reproduce paper outputs

No API calls. Requires `results/` as above.

```bash
python main.py consolidate       # rebuild iterative long-format tables
python main.py dump-report       # → reports/report_tables.tex
python main.py analyze-rounds    # per-round CSVs, plots, wording note
python main.py plot-bimodality   # publication histograms (PDF + PNG)
```

Then, if you have LaTeX installed:

```bash
cd reports && pdflatex report_tables.tex
```

| Command | Output |
|---------|--------|
| `consolidate` | `results/iterative/iterative_all_rounds.{csv,jsonl}` and summary |
| `dump-report` | `reports/report_tables.tex` |
| `analyze-rounds` | `results/iterative/per_round_analysis/` |
| `plot-bimodality` | `.../per_round_analysis/bimodality_plots/` |

---

## Run new experiments

New runs write to **`results_runs/`** (also gitignored) and never overwrite the
curated `results/` archive.

```bash
# Unprompted control
python main.py baseline control --runs 30 --yes

# Anchor in the first prompt
python main.py static anchor \
  --conditions MCKINSEY,INTERN \
  --assertions standard \
  --anchor-levels LOW,MID,HIGH \
  --runs 30 --yes

# Multi-round reconsideration (seed from a baseline JSONL that includes raw_text)
python main.py iterative run \
  --conditions MCKINSEY \
  --assertions standard \
  --anchor-levels MID \
  --max-rounds 5 --yes \
  --baseline-source results_runs/single_turn/single_turn_trials.jsonl
```

Common flags: `--models`, `--runs`, `--concurrency`, `--temperature`,
`--dry-run`, `--yes`, `--include-raw`. Subcommands `summarize` and `show-prompt`
need no API. For full options: `python main.py <command> --help`.

---

## What the code does

Models recommend a retail price for a fixed product profile. Across cells we
vary an **authority-sourced price suggestion** (the anchor) and measure how
recommendations move.

Three pipelines:

| Pipeline | Idea |
|----------|------|
| **Baseline** | Single turn, no anchor (CONTROL). |
| **Static** | Same scaffold, but the authority sentence appears in the first prompt. |
| **Iterative** | CONTROL answer first; later turns push with “Are you sure?” / “Are you really sure?” until snap or `--max-rounds`. |

We also vary authority source (intern → McKinsey, etc.), assertion strength
(weak / standard / strong), and anchor extremity (Mid through extreme levels).
The curated archive includes prompt-sensitivity arms (ambiguous ranges; margin
and CAC removed); those appear in `dump-report` but are not launched by the
current experiment CLIs.

Default models: Claude Haiku 4.5 and GPT-5.4 Mini via OpenRouter.

---

## Package layout

```text
main.py            Unified CLI
config/            Settings, anchors, prompts, paths
utils/             Parse, I/O, stats, CLI helpers
services/          OpenRouter client, prompt builders, batch runner
experiments/       baseline / static / iterative runners
analysis/          consolidate, dump-report, per-round analysis, plots
reports/           Regenerable LaTeX
results/           Curated data (download; gitignored)
results_runs/      Fresh API outputs (gitignored)
```

---

## Data shape (after you install `results/`)

**Single-turn** — `results/single_turn/single_turn_trials.jsonl`  
One row per trial. `condition_id == "CONTROL"` vs authority conditions. Fields
include `price_recommendation`, `anchor_level`, `anchor_value`, `injection_text`.

**Iterative** — `results/iterative/iterative_all_rounds.csv`  
One row per conversation × reconsideration round. Key columns:
`conversation_key`, `round`, `baseline_price`, `price_before`, `price_after`,
`snapped`, `is_last_observed_round`. Rebuild from
`results/iterative/source_per_round/` with `python main.py consolidate`.

---

## Configuration

| File | Role |
|------|------|
| `config/settings.py` | Models, concurrency, API defaults |
| `config/anchors.py` | Dollar values per `(level, model)` |
| `config/prompts.py` | System/user prompts and injection templates (verbatim) |
| `config/paths.py` | Locations of `results/`, `reports/`, `results_runs/` |
| `.env` | Secrets (`OPENROUTER_API_KEY`, optional site headers / timeouts) |

Anchor levels: `UNREASONABLY_LOW` … `RIDICULOUSLY_HIGH`.  
Authorities: `INTERN`, `COLLEAGUE`, `MANAGER`, `CONSULTING`, `MCKINSEY`,
`UNLABELED`, `AI`.
