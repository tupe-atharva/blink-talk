"""Gemini reply prediction for BlinkTalk (owner: Yash).

Contract (called by /api/suggest):
    suggest(transcript, history, profile, local_time) -> list[str]
Always returns exactly 6 phrases, or raises so the route can use its fallback.
Hard timeout: 3.5 s.
Prompt tuned in ml/notebooks/04_gemini_prompt.ipynb
(gemini-3.5-flash-lite: 15/16 hits, p50 0.80 s, p95 0.91 s).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

# `or` (not a getenv default) so an empty `GEMINI_MODEL=` in .env still falls back
MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL") or "gemini-3.5-flash"
LOCAL_TZ = os.getenv("LOCAL_TZ") or "America/New_York"   # used when the frontend sends UTC (…Z)
TIMEOUT_S = 3.5
N_PHRASES = 6
MAX_WORDS = 8

SYSTEM = """You suggest spoken replies for a person who cannot speak and picks a reply with their eyes.
Write as that person, in the first person.

Rules:
- Return exactly 6 replies, each at most 6 words, plain words, no emojis, no quotes.
- Replies 1-2 must directly answer what the caregiver just said.
  For a yes/no question: reply 1 starts with "Yes", reply 2 starts with "No".
  For a choice ("A or B?"): replies 1-2 pick each option (the profile's preferred option first).
  For "how are you / how do you feel": lead with feelings (tired, okay, in pain, happy).
  For "can I get you anything?": lead with the profile's needs.
- Replies 3-5 stay on the caregiver's topic first; use profile needs, people and preferences only when they fit.
  Do not mention the same preference in every answer.
- Reply 6 is always a short way to defer, like "Not now, thanks" or "Later, please".
- Never give medical advice, diagnoses or medication instructions.
- If the caregiver said nothing, suggest useful openers for the time of day."""

DEFAULTS = {
    "morning": ["Good morning", "I need water", "I'm hungry", "Open the blinds, please", "I didn't sleep well", "Not now, thanks"],
    "afternoon": ["I need water", "I'm tired", "Can you move me?", "I'm hungry", "Turn on the TV", "Not now, thanks"],
    "evening": ["I'm tired", "I need water", "Can you move me?", "Turn the light off", "Good night", "Not now, thanks"],
}


class Suggestions(BaseModel):
    phrases: list[str]


_client: genai.Client | None = None
_pool = ThreadPoolExecutor(max_workers=4)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                timeout=10_000,  # Gemini's minimum server deadline is 10 s; our 3.5 s limit is enforced in suggest()
                retry_options=types.HttpRetryOptions(attempts=1),  # no hidden retries; route falls back instead
            ),
        )
    return _client


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Works with dicts and pydantic/dataclass objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(LOCAL_TZ)
    except Exception:  # no tz database on the server → assume US Eastern
        return timezone(timedelta(hours=-5))


def _local_dt(local_time: str) -> datetime | None:
    """Parse ISO times like '2026-11-08T02:05:00.000Z' or '...-05:00' and convert to local time."""
    try:
        dt = datetime.fromisoformat((local_time or "").strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(_tz()) if dt.tzinfo else dt


def _hour(local_time: str) -> int:
    dt = _local_dt(local_time)
    if dt:
        return dt.hour
    m = re.search(r"(\d{1,2}):(\d{2})", local_time or "")
    return int(m.group(1)) % 24 if m else 12


def _pretty_time(local_time: str) -> str:
    dt = _local_dt(local_time)
    return dt.strftime("%A %I:%M %p").replace(" 0", " ") if dt else (local_time or "unknown")


def _time_defaults(local_time: str) -> list[str]:
    h = _hour(local_time)
    return DEFAULTS["morning"] if h < 12 else DEFAULTS["afternoon"] if h < 17 else DEFAULTS["evening"]


def _clean(p: Any) -> str:
    return str(p).strip().strip('"').strip()


def _key(p: str) -> str:
    return re.sub(r"\W+", " ", p.lower()).strip()


def finalize(phrases: list[str], profile: Any, local_time: str) -> list[str]:
    """Dedupe, drop long ones, pad to exactly 6 from commonPhrases, then time-of-day defaults."""
    pool = list(phrases) + list(_get(profile, "commonPhrases", []) or []) + _time_defaults(local_time)
    out: list[str] = []
    seen: set[str] = set()
    for p in pool:
        p = _clean(p)
        k = _key(p)
        if p and k and k not in seen and len(p.split()) <= MAX_WORDS:
            seen.add(k)
            out.append(p)
        if len(out) == N_PHRASES:
            break
    return out


def build_prompt(transcript: str | None, history: list[Any], profile: Any, local_time: str) -> str:
    def join(key: str) -> str:
        return ", ".join(_get(profile, key, []) or []) or "none"

    lines = [
        f"Local time: {_pretty_time(local_time)}",
        f"User: {_get(profile, 'name', 'the user')}",
        f"Caregivers: {join('caregivers')}",
        f"Needs: {join('needs')}",
        f"Preferences: {join('preferences')}",
    ]
    if history:
        lines.append("Recent conversation (oldest first):")
        lines += [f"- {_get(m, 'role')}: {_get(m, 'text')}" for m in list(history)[-8:]]
    if transcript and transcript.strip():
        lines.append(f'Caregiver just said: "{transcript.strip()}"')
    else:
        lines.append("The caregiver has not said anything. Suggest useful openers.")
    return "\n".join(lines)


def _config(model: str) -> types.GenerateContentConfig:
    thinking = (types.ThinkingConfig(thinking_level="minimal") if model.startswith("gemini-3")
                else types.ThinkingConfig(thinking_budget=0))
    return types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=Suggestions,
        temperature=0.4,
        max_output_tokens=300,
        thinking_config=thinking,
    )


def _call(prompt: str) -> list[str]:
    """Try MODEL, then FALLBACK_MODEL if the first is overloaded (503) or over quota (429)."""
    last: Exception | None = None
    for model in dict.fromkeys([MODEL, FALLBACK_MODEL]):  # dedupe if both are the same
        try:
            r = _get_client().models.generate_content(model=model, contents=prompt, config=_config(model))
            raw = r.parsed.phrases if r.parsed else json.loads(r.text)["phrases"]
            return [_clean(p) for p in raw if _clean(p)]
        except Exception as e:
            if getattr(e, "code", None) in (429, 503):
                last = e
                continue
            raise
    assert last is not None
    raise last


def suggest(transcript: str | None, history: list[Any], profile: Any, local_time: str) -> list[str]:
    """Exactly 6 reply phrases. Raises on timeout, API error or empty output."""
    prompt = build_prompt(transcript, history, profile, local_time)
    future = _pool.submit(_call, prompt)
    try:
        raw = future.result(timeout=TIMEOUT_S)
    except _FutureTimeout as e:
        raise TimeoutError(f"Gemini took longer than {TIMEOUT_S}s") from e
    if not raw:
        raise ValueError("Gemini returned no phrases")
    out = finalize(raw, profile, local_time)
    if len(out) != N_PHRASES:  # should never happen: defaults always have 6
        raise ValueError(f"expected {N_PHRASES} phrases, got {len(out)}")
    return out


if __name__ == "__main__":  # quick manual test: python api/gemini.py
    import time

    profile = {
        "name": "Maya",
        "caregivers": ["Alex (husband)", "Priya (nurse)"],
        "needs": ["water often", "help changing position"],
        "preferences": ["likes apple juice", "loves the Beatles"],
        "commonPhrases": ["Thank you", "I love you"],
    }
    history = [{"role": "caregiver", "text": "Good evening!", "ts": "2026-11-07T18:29:00"}]
    for q in ["Do you want water or juice?", "How are you feeling?", None]:
        t = time.time()
        try:
            out = suggest(q, history, profile, "2026-11-07T18:30:00-05:00")
            print(f"{time.time() - t:.2f}s  {q!r} -> {out}")
        except Exception as e:
            print(f"{time.time() - t:.2f}s  {q!r} -> FALLBACK ({type(e).__name__}: {str(e)[:80]})")