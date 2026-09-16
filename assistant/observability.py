"""Structured events, and one id that ties an answer's lines together.

`CLAUDE.md` asks for structured logs at the significant events — retrieval,
route selection, generation timing, safety-check failure, refusal, database and
Ollama errors — and for a correlation id where it helps trace a single answer.
Here it always helps: one message splits into parts, each part is gated, routed,
retrieved for and checked separately, so a single question can produce a dozen
lines that are only legible if they carry the same id.

Three decisions worth stating, because each is a constraint rather than a
preference.

**The library emits and never configures.** Importing `assistant` must not
attach a handler or print anything; an entry point calls `configure()` when it
wants output. A library that configures logging steals the decision from the
application embedding it, and this engine is meant to be embedded — that is the
whole argument for the channel adapters in the architecture.

**The id travels in a context variable, not in a signature.** Threading a
`correlation_id` parameter through retrieval, routing, generation and checking
would put an observability concern into the signature of every domain function.
A `ContextVar` is also the correct primitive rather than a convenient one: the
web UI serves on a `ThreadingHTTPServer`, and context variables are per-context,
so two concurrent questions cannot pick up each other's id.

**The question text is not logged.** `CLAUDE.md` is explicit about not logging
complete customer conversations. What goes into a log line is the fingerprint —
a short hash and a word count — which is enough to tell whether two lines
concern the same question, to spot a repeated question, and to measure length
distribution, without accumulating what people asked in a file that outlives the
reason for keeping it. The question text is held in one place only, the
`answer_log` table, where it is a deliberate and auditable retention decision.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import time
import uuid
from typing import Any, Iterator

LOGGER_NAME = "assistant"

# A library logger with a null handler: emitting is free and silent until an
# application asks for output.
logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())

_correlation: contextvars.ContextVar[str] = contextvars.ContextVar(
    "assistant_correlation_id", default="")

# The keys a LogRecord already owns. Anything a caller passes under one of these
# names would collide with the record's own attribute and raise inside logging,
# which would turn an observability call into a failed answer.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))


def new_id() -> str:
    """Short enough to read in a terminal, long enough not to collide."""
    return uuid.uuid4().hex[:12]


def correlation_id() -> str:
    """The id of the answer currently being produced, or empty outside one."""
    return _correlation.get()


@contextlib.contextmanager
def correlation(cid: str = "") -> Iterator[str]:
    """Bind an id for the duration of one answer.

    The token is reset in a `finally`, so an exception on the answering path
    cannot leave the id bound to a thread that goes on to serve someone else.
    """
    cid = cid or new_id()
    token = _correlation.set(cid)
    try:
        yield cid
    finally:
        _correlation.reset(token)


def fingerprint(text: str) -> str:
    """Identify a question without recording it."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:12]


def event(name: str, /, **fields: Any) -> None:
    """Emit one structured event, and never fail the caller.

    The event name is positional-only. Without that, a caller logging a
    field that happens to be called `name` — a product name, a document
    name — collides with this parameter and raises `TypeError` from inside
    an observability call, which is precisely the failure this function
    exists to avoid.

    Logging is not allowed to be the reason an answer is lost. Every call site
    here sits on the answering path, so the broad except is deliberate: an
    observability defect degrades observability and nothing else.
    """
    safe = {k: v for k, v in fields.items() if k not in _RESERVED}
    try:
        logger.info(name, extra={"event": name, "fields": safe,
                                 "correlation_id": _correlation.get()})
    except Exception:                                  # pragma: no cover - defensive
        pass


@contextlib.contextmanager
def timed(name: str, /, **fields: Any) -> Iterator[dict]:
    """Emit an event carrying how long the block took.

    The dict yielded is writable, so a block can add what it learned — a hit
    count, a chosen path — to the same event rather than emitting a second one.
    """
    extra: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        yield extra
    finally:
        event(name, seconds=round(time.perf_counter() - started, 4),
              **{**fields, **extra})


class JSONFormatter(logging.Formatter):
    """One event per line, machine-readable, with the correlation id promoted.

    JSON lines rather than prose because these are meant to be aggregated: the
    operational questions are "what is the refusal rate this week" and "which
    checks fire most often", and neither is answerable by grepping sentences.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "at": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
        }
        cid = getattr(record, "correlation_id", "")
        if cid:
            payload["correlation_id"] = cid
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=False)


def configure(stream=None, level: int = logging.INFO) -> logging.Logger:
    """Attach one JSON handler. Called by entry points, never by the library.

    Replaces its own previous handler rather than adding a second, so calling it
    twice — which a test or a reloading server will do — does not double every
    line.
    """
    for existing in list(logger.handlers):
        if getattr(existing, "_assistant_handler", False):
            logger.removeHandler(existing)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    handler._assistant_handler = True                  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(level)
    # The application owns its root configuration; ours must not also reach it.
    logger.propagate = False
    return logger
