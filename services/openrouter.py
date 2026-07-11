"""OpenRouter async chat-completions client with retries."""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time

try:
    import aiohttp
except ImportError:
    sys.exit("ERROR: aiohttp is required.  pip install aiohttp")

from config.settings import (
    API_KEY,
    BASE_URL,
    MAX_RETRIES,
    REQUEST_TIMEOUT_S,
    SITE_NAME,
    SITE_URL,
)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}


async def call_api(session, model, messages, temperature, max_tokens):
    """One OpenRouter chat completion. Returns (content, error, retryable)."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME
    payload = {
        "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "messages": messages,
    }
    try:
        async with session.post(
            BASE_URL, headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
        ) as resp:
            if resp.status in RETRYABLE_STATUS:
                return None, f"HTTP {resp.status}: {(await resp.text())[:200]}", True
            if resp.status >= 400:
                return None, f"HTTP {resp.status}: {(await resp.text())[:300]}", False
            data = await resp.json()
    except asyncio.TimeoutError:
        return None, "Timeout", True
    except aiohttp.ClientError as e:
        return None, f"{type(e).__name__}: {e}", True
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", False

    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message", str(err))[:300] if isinstance(err, dict) else str(err)[:300]
        code = err.get("code") if isinstance(err, dict) else None
        return None, f"API error: {msg}", code in RETRYABLE_STATUS or code == 429
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, f"Malformed response: {json.dumps(data)[:200]}", True
    if not content or not content.strip():
        return None, "Empty completion", True
    return content, None, False


async def call_with_retries(session, model, messages, temperature, max_tokens):
    """Retry wrapper around :func:`call_api`. Returns (content, error, attempts, latency_s)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.monotonic()
        content, api_err, retryable = await call_api(
            session, model, messages, temperature, max_tokens)
        latency = round(time.monotonic() - t0, 2)
        if not api_err:
            return content, None, attempt, latency
        last_err = api_err
        if retryable and attempt < MAX_RETRIES:
            await asyncio.sleep(min(60, 2 ** attempt) + random.uniform(0, 1.5))
            continue
        return None, api_err, attempt, latency
    return None, last_err, MAX_RETRIES, None
