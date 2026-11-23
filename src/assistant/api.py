"""The FastAPI service: `POST /ask` and `GET /health`, with the limits a public demo needs.

The demo runs on a free-tier LLM key, so three guards sit in front of the model:
a per-IP rate limit, a global daily cap and an answer cache. All state lives in memory
(one process), which is enough for a single Hugging Face Space.
"""

import json
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from assistant.config import Settings, get_settings
from assistant.models import Answer

logger = logging.getLogger(__name__)

RATE_LIMIT_WINDOW_SECONDS = 3600
CACHE_SIZE = 256
UNAVAILABLE_MESSAGE = "The demo has reached its usage limit, please try again later."
RATE_LIMIT_MESSAGE = "Too many questions, please try again later."
DAILY_CAP_MESSAGE = "The demo has reached its daily limit, please try again tomorrow."


class AskRequest(BaseModel):
    question: str


class Guard:
    """Per-IP sliding window, global daily cap and LRU answer cache, behind one lock."""

    def __init__(self, settings: Settings, clock: Callable[[], float]):
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._cache: OrderedDict[str, Answer] = OrderedDict()
        self._day = self._today()
        self._used_today = 0

    def _today(self):
        return datetime.fromtimestamp(self._clock(), UTC).date()

    def check_rate(self, ip: str) -> None:
        """Count this request for `ip`, or raise 429 when it already asked too often."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(ip, deque())
            while hits and hits[0] <= now - RATE_LIMIT_WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= self._settings.rate_limit_per_hour:
                raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
            hits.append(now)
            if len(self._hits) > 10_000:  # forget IPs whose window is empty
                stale = [k for k, v in self._hits.items() if not v or v[-1] <= now - 3600]
                for key in stale:
                    del self._hits[key]

    def cached(self, key: str) -> Answer | None:
        with self._lock:
            answer = self._cache.get(key)
            if answer is not None:
                self._cache.move_to_end(key)
            return answer

    def reserve_call(self) -> None:
        """Take one slot of today's cap for an uncached model call, or raise 429."""
        with self._lock:
            today = self._today()
            if today != self._day:  # new UTC day
                self._day, self._used_today = today, 0
            if self._used_today >= self._settings.daily_cap:
                raise HTTPException(status_code=429, detail=DAILY_CAP_MESSAGE)
            self._used_today += 1

    def store(self, key: str, answer: Answer) -> None:
        with self._lock:
            self._cache[key] = answer
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    first_hop = forwarded.split(",")[0].strip()
    if first_hop:
        return first_hop
    return request.client.host if request.client else "unknown"


def _text_as_of(path: Path) -> str:
    """e.g. "AI Act 2026-07-27; GDPR 2016-05-04", from the consolidated article texts."""
    names = {"AI_ACT": "AI Act", "GDPR": "GDPR"}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    dates: dict[str, str] = {}
    for entry in entries:
        if entry.get("role") == "articles":
            dates.setdefault(entry["regulation"], entry["version_date"])
    return "; ".join(f"{names[reg]} {dates[reg]}" for reg in names if reg in dates) or "unknown"


def _count_chunks(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _build_real_answer_fn(settings: Settings) -> Callable[[str], Answer]:
    """Load the index, retriever and LLM once. Heavy imports stay here, out of import time."""
    from assistant.chain import answer_question, make_llm
    from assistant.index import load_index, make_embeddings
    from assistant.parse import read_chunks
    from assistant.retrieve import build_retriever

    llm = make_llm(settings)  # raises ValueError when no API key is configured
    store = load_index(make_embeddings(settings), settings.chroma_dir)
    retrieve = build_retriever(store, read_chunks(settings.chunks_path), settings)
    return lambda question: answer_question(question, retrieve, llm)


def create_app(
    answer_fn: Callable[[str], Answer] | None = None,
    settings: Settings | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    """Build the app. Tests inject `answer_fn` and `clock`; otherwise the chain loads at startup."""
    settings = settings or get_settings()
    guard = Guard(settings, clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if answer_fn is None:
            try:
                app.state.answer_fn = _build_real_answer_fn(settings)
            except Exception as exc:  # no key, no index, ...: stay up, answer 503
                logger.warning("answer chain not available (%s)", type(exc).__name__)
        yield

    app = FastAPI(title="EU AI Act + GDPR assistant", lifespan=lifespan)
    app.state.answer_fn = answer_fn
    health = {
        "status": "ok",
        "text_as_of": _text_as_of(settings.sources_path),
        "chunks": _count_chunks(settings.chunks_path),
    }

    @app.get("/health")
    def get_health() -> dict:
        return health

    # A plain `def` endpoint runs in a worker thread, so a slow LLM call does not block others.
    @app.post("/ask", response_model=Answer)
    def ask(body: AskRequest, request: Request) -> Answer:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="The question must not be empty.")
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The question is too long (limit: {settings.max_question_chars} characters)."
                ),
            )
        guard.check_rate(client_ip(request))
        key = question.lower()
        cached = guard.cached(key)
        if cached is not None:
            return cached
        fn = app.state.answer_fn
        if fn is None:
            raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
        guard.reserve_call()
        try:
            answer = fn(question)
        except Exception as exc:  # quota, network, ...: log the type only, never the message
            logger.warning("answer call failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from None
        guard.store(key, answer)
        return answer

    return app


app = create_app()
