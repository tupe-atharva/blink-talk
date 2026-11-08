"""BlinkTalk API (FastAPI). Routes follow the shared contract in CLAUDE.md exactly.

Run from inside api/:  uvicorn main:app --reload --port 8000
"""
from pathlib import Path

from dotenv import load_dotenv

# Local dev: read the repo-root .env. In production systemd sets the env (EnvironmentFile), which wins.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import Depends, FastAPI, Query, Request, Response  # noqa: E402

import db  # noqa: E402
import fallbacks  # noqa: E402
import voice  # noqa: E402
from ratelimit import RateLimiter  # noqa: E402
from models import (  # noqa: E402
    AlertIn,
    Message,
    MessageIn,
    Profile,
    SpeakRequest,
    StatusIn,
    SuggestRequest,
    SuggestResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("blinktalk")

GEMINI_TIMEOUT_S = 3.5  # Gemini cap (p95 is ~0.9 s). Contract: /suggest answers within 5 s, which leaves room for a slow first connection

# ---- other people's modules: use them if present, stub them if not ----------
# api/gemini.py is Yash's, api/events.py is Om's. We don't create or edit those files here.
try:
    import gemini  # suggest(transcript, history, profile, local_time) -> list[str]
except ImportError as e:
    gemini = None
    log.warning("gemini.py not available (%s): /suggest will always use fallback phrases", e)

try:
    import events  # publish(event, data) + router for GET /api/events

    _publish = events.publish
    events_router = getattr(events, "router", None)
except ImportError as e:
    events_router = None
    log.warning("events.py not available (%s): SSE publish is a no-op", e)

    def _publish(event: str, data: dict) -> None:
        log.info("[stub publish] %s %s", event, data)


def publish(event: str, data: dict) -> None:
    """Never let a broadcaster problem break a route."""
    try:
        _publish(event, data)
    except Exception:
        log.exception("publish(%s) failed", event)


# ---- app ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    warm = asyncio.create_task(voice.warm_cache(fallbacks.all_phrases(db.get_profile())))
    yield
    warm.cancel()
    await voice.aclose()


app = FastAPI(title="BlinkTalk API", lifespan=lifespan)
if events_router is not None:
    app.include_router(events_router)


@app.get("/api/health")
def health():
    return {"ok": True}


suggest_limiter = RateLimiter("suggest", limit=30)  # per client IP per minute; costs Gemini credits
speak_limiter = RateLimiter("speak", limit=20)      # only counts UNcached text; cached audio is free


@app.post("/api/suggest", response_model=SuggestResponse, dependencies=[Depends(suggest_limiter.check)])
async def suggest(req: SuggestRequest):
    t0 = time.perf_counter()
    profile = db.get_profile()
    # History: use what the client sent (contract); if it sent none, use the last 6 stored messages.
    history = (req.history or db.recent_messages(6))[-6:]

    phrases: list[str] = []
    source = "fallback"
    if gemini is not None:
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(gemini.suggest, req.transcript, history, profile, req.local_time),
                timeout=GEMINI_TIMEOUT_S,
            )
            phrases = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
            source = "gemini" if phrases else "fallback"
        except Exception as e:  # timeout, network, bad JSON, anything
            log.warning("gemini.suggest failed: %s: %s", type(e).__name__, e)

    phrases = fallbacks.pad(phrases, req.local_time, profile) if phrases else fallbacks.phrases(req.local_time, profile)
    # Start generating audio for the 6 tiles now, in the background (returns immediately), so the tile
    # the patient picks is already in the cache by the time /speak asks for it.
    prefetching = voice.prefetch(phrases)
    log.info("/suggest %s in %.0f ms (prefetching %d)", source, (time.perf_counter() - t0) * 1000, prefetching)
    return SuggestResponse(phrases=phrases, source=source)


@app.post("/api/speak")
async def speak(req: SpeakRequest, request: Request):
    t0 = time.perf_counter()
    cached = voice.is_cached(req.text)
    if not cached:
        await speak_limiter.check(request)  # 429 if this client is burning ElevenLabs credits
    try:
        audio = await voice.speak(req.text)
    except voice.VoiceError as e:
        log.warning("/speak failed in %.0f ms: %s", (time.perf_counter() - t0) * 1000, e)
        return Response(status_code=503)  # frontend falls back to browser speech
    log.info("/speak %s in %.0f ms (%d bytes)", "cache" if cached else "elevenlabs",
             (time.perf_counter() - t0) * 1000, len(audio))
    return Response(content=audio, media_type="audio/mpeg")


# The three routes below are `async def` on purpose: publish() touches asyncio queues and must run
# on the event loop thread, not in FastAPI's threadpool. The SQLite writes are sub-millisecond.

@app.post("/api/messages", response_model=Message)
async def post_message(body: MessageIn):
    msg = db.add_message(body.role, body.text)
    publish("message", msg.json_dict())
    return msg


@app.get("/api/messages", response_model=list[Message])
def get_messages(limit: int = Query(20, ge=1, le=200)):
    return db.recent_messages(limit)


@app.post("/api/alert", status_code=204)
async def alert(body: AlertIn):
    data = {"type": body.type, "ts": db.now_iso()}
    db.log_event("alert", data)
    publish("alert", data)
    return Response(status_code=204)


@app.post("/api/status", status_code=204)
async def status(body: StatusIn):
    data = {"state": body.state, "ts": db.now_iso()}
    db.log_event("status", data)
    publish("status", data)
    return Response(status_code=204)


@app.get("/api/profile", response_model=Profile)
def get_profile():
    return db.get_profile()


@app.put("/api/profile", response_model=Profile)
def put_profile(p: Profile):
    return db.put_profile(p)

# DEP-02 drill: deliberate syntax error, reverted right after
def broken(:
