"""Runtime settings loaded from the environment / ``.env`` file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from config.paths import PACKAGE_ROOT

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_env() -> None:
    """Load ``refactor/.env``, then the repo-root ``.env``, then the process env."""
    if load_dotenv is None:
        print(
            "WARNING: python-dotenv not installed; relying on the shell environment. "
            "pip install python-dotenv",
            file=sys.stderr,
        )
        return
    # Package-local first, then parent repo (legacy location).
    load_dotenv(PACKAGE_ROOT / ".env")
    load_dotenv(PACKAGE_ROOT.parent / ".env")
    load_dotenv()  # cwd / default search


_load_env()

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
SITE_URL = os.environ.get("SITE_URL", "")
SITE_NAME = os.environ.get("SITE_NAME", "Confirmation Bias in LLM Pricing Recommendations")

# Verify exact slugs at openrouter.ai/models before a full run.
DEFAULT_MODELS = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    # "google/gemini-3-flash-preview",
]

DEFAULT_CONCURRENCY = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 2048

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "4"))
REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "180"))

CURRENCY = "USD"
PROFILE_ID = "PRODUCT"
EPS = 1e-9  # "unchanged" / "on anchor" tolerance (to a cent)

VALID_ASSERTIONS = ["weak", "standard", "strong"]
ANCHOR_LEVEL_KEYS = [
    "UNREASONABLY_LOW", "VERY_LOW", "LOW", "MID", "HIGH",
    "VERY_HIGH", "UNREASONABLY_HIGH", "RIDICULOUSLY_HIGH",
]
NONE_TOKEN = "NONE"
CONTROL_TOKEN = "CONTROL"
