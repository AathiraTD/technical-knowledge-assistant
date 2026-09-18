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

---

## Spans

`span()` adds the second half: not only *what happened* but *inside what, and
for how long*. The shape is OpenTelemetry's — a trace holding a tree of spans,
each with an id and a parent — and the SDK is deliberately not. The review in
`docs/conversation-observability-review.md` §2.1 makes that case: the attribute
conventions and the trace/span/parent triple are the valuable part and they are
free, while the SDK is six transitive packages in a project whose decision 4
rests on five inspectable dependencies, and it buys nothing until a collector
exists. So the field names follow OTel where a convention exists —
`gen_ai.request.model`, `gen_ai.usage.output_tokens`, `db.system` — and an OTLP
exporter stays a shim that can be added later without touching a call site.

**Four ids, in a strict containment hierarchy** (review §2.2):

    session_id     the conversation        SessionStore owns it
      turn_id      one message in          new here
        trace_id   one call to ask()       the correlation id, renamed in concept
          span_id  one stage within it     new here, with parent_span_id

`turn_id` and `trace_id` are one-to-one for the single-turn engine and are still
separated, because a re-asked pending question is the same turn and a different
trace; collapsing them would make the ask-back cycle unreadable.

**The span stack is a `ContextVar`, for the same reason the correlation id is.**
The web UI serves on a `ThreadingHTTPServer`. A module-level list would let two
concurrent questions adopt each other's parent and produce one interleaved tree
belonging to nobody. A `ContextVar` is per-context, and the token is reset in a
`finally` so an exception on the answering path cannot leave a dead span as the
parent of whatever that thread serves next.

**A span emits exactly one event, on completion.** Nothing is emitted on entry:
a start event doubles the line count to say something the completion event
already implies, and a span that never completes is visible as a hole in the
tree rather than as a dangling line. The existing JSON-line stream is therefore
the whole transport, and `JSONFormatter` needs only to promote the four ids.

**Spans obey the same privacy rule, with no exception.** No span carries
question text, answer text or passage text — fingerprints, counts, ids, scores
and durations only. `answer_log.question` remains the single deliberate
retention point, and the trace does not duplicate it. This is enforced by a test
(`tests/test_spans.py`) rather than by scrubbing at this boundary, because a
scrubber that silently removed a field would hide the call site that added it.

**Collection is opt-in and separate from emission.** `span()` always emits;
it appends to a persistence buffer only inside `turn()`, which the engine opens
around one answer and drains into `KnowledgeRepository.record_spans`. A caller
that wants the log lines and no database rows — a test, a library embedder —
gets them by not opening a turn, and nothing in `span()` knows about a store.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

from ..knowledge.model import TraceSpan

try:
    from . import otel_export
except ImportError:
    otel_export = None  # type: ignore

LOGGER_NAME = "assistant"

# A library logger with a null handler: emitting is free and silent until an
# application asks for output.
logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())

_correlation: contextvars.ContextVar[str] = contextvars.ContextVar(
    "assistant_correlation_id", default="")

# The current span, the turn it belongs to and the conversation the turn belongs
# to. All three are per-context for the reason the correlation id is: the web UI
# is a `ThreadingHTTPServer`, and a module-level current-span would let two
# concurrent questions parent each other's spans.
_span: contextvars.ContextVar[str] = contextvars.ContextVar(
    "assistant_span_id", default="")
_turn: contextvars.ContextVar[str] = contextvars.ContextVar(
    "assistant_turn_id", default="")
_session: contextvars.ContextVar[str] = contextvars.ContextVar(
    "assistant_session_id", default="")
# Where completed spans accumulate for persistence, or None when nobody is
# collecting. A list rather than a callback: the engine wants the whole turn at
# once so it can write it in one statement, outside the read snapshot.
_collector: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "assistant_span_collector", default=None)
# The surface that asked, carried onto every span for the same reason
# `answer_log.source` exists: evaluation traffic and real traffic in one table
# make any rate computed from it a measurement of the question set.
_source: contextvars.ContextVar[str] = contextvars.ContextVar(
    "assistant_trace_source", default="unknown")

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
                                 "correlation_id": _correlation.get(),
                                 # An ordinary event emitted inside a span is
                                 # part of that span's story — an Ollama error
                                 # during generation, a check that failed. It
                                 # carries the ids so the two can be read
                                 # together without inferring the join from
                                 # timestamps.
                                 "span_id": _span.get(),
                                 "turn_id": _turn.get(),
                                 "session_id": _session.get()})
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


def trace_id() -> str:
    """The id of the answer being produced. The correlation id, named as OTel does."""
    return _correlation.get()


def span_id() -> str:
    """The stage currently running, or empty outside one."""
    return _span.get()


def turn_id() -> str:
    return _turn.get()


def session_id() -> str:
    return _session.get()


@contextlib.contextmanager
def turn(turn: str = "", session: str = "", source: str = "") -> Iterator[list]:
    """Collect one message's spans, and bind the ids they all carry.

    Yields the list the completed spans land in, so the caller can hand it to
    `KnowledgeRepository.record_spans` once the answer is finished. Collection
    is separate from emission on purpose: `span()` writes its log line whether
    or not anyone is collecting, so a library embedder or a test that never
    opens a turn still gets the full stream and writes no database rows.

    The list is drained by the caller rather than written here. This context
    manager knows nothing about a store, which is what keeps observability from
    being able to fail an answer — and it is also why the write happens after
    the read snapshot has closed, exactly as `log_answer`'s does and for the
    same reason.

    Every token is reset in a `finally`. A thread on a `ThreadingHTTPServer`
    that raised halfway through an answer must not go on to serve the next
    caller inside the previous one's turn.
    """
    collected: list[TraceSpan] = []
    tokens = [
        _turn.set(turn or new_id()),
        _session.set(session),
        _source.set(source or "unknown"),
        _collector.set(collected),
        # A turn begins at the root of its own span tree. Without this reset a
        # nested turn — or a thread reusing a context — would inherit a parent
        # from outside the trace and hang the whole tree off a dead span.
        _span.set(""),
    ]
    try:
        yield collected
    finally:
        for var, token in zip((_turn, _session, _source, _collector, _span),
                              tokens):
            var.reset(token)


@contextlib.contextmanager
def span(name: str, /, **attributes: Any) -> Iterator[dict]:
    """One stage of one answer: timed, parented, emitted once on completion.

    The name is positional-only for the reason `event()`'s is: a call site
    logging a field that happens to be called `name` would otherwise collide
    with this parameter and raise `TypeError` from inside an observability call.

    Yields the attribute dict, writable, so a block adds what it learned — a hit
    count, the path taken — to the same span rather than emitting a second
    event. Nothing is emitted on entry: a start line doubles the stream to say
    what the completion line already implies, and a span that never completes
    reads as a hole in the tree, which is more legible than a dangling start.

    On an exception the span still completes, with `status="error"` and the
    exception type as an attribute, and the exception is re-raised untouched.
    Observability records that a stage failed; it does not decide what happens
    next, and it never becomes the reason an answer is lost.

    The parent is whatever span is current in this context, and the token is
    reset in a `finally`, so two questions answered concurrently on the
    threading server cannot adopt each other's parent.
    """
    ident = new_id()
    parent = _span.get()
    token = _span.set(ident)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = "ok"
    try:
        yield attributes
    except BaseException as error:
        status = "error"
        # The type, never the message. An exception string can carry a question
        # or a passage — a database error quoting the statement it failed on is
        # the obvious case — and this table is not allowed to hold either.
        attributes.setdefault("error", type(error).__name__)
        raise
    finally:
        _span.reset(token)
        duration_ms = int(round((time.perf_counter() - started) * 1000))
        safe = {k: v for k, v in attributes.items() if k not in _RESERVED}
        record = TraceSpan(
            trace_id=_correlation.get(), span_id=ident, parent_span_id=parent,
            turn_id=_turn.get(), session_id=_session.get(), name=name,
            started_at=started_at, duration_ms=duration_ms, status=status,
            source=_source.get(), attributes=safe)
        collected = _collector.get()
        if collected is not None:
            collected.append(record)
        try:
            logger.info(name, extra={
                "event": name, "fields": safe,
                "correlation_id": record.trace_id,
                "span_id": ident, "parent_span_id": parent,
                "turn_id": record.turn_id, "session_id": record.session_id,
                "duration_ms": duration_ms, "status": status,
                "span_source": record.source})
        except Exception:                              # pragma: no cover - defensive
            pass
        # Export to OTLP if configured
        if otel_export is not None:
            try:
                otel_export.export_trace(
                    trace_id=record.trace_id, span_id=ident, parent_span_id=parent,
                    name=name, started_at=started_at, duration_ms=duration_ms,
                    status=status, attributes=safe)
            except Exception:  # pragma: no cover - defensive
                pass


class JSONFormatter(logging.Formatter):
    """One event per line, machine-readable, with the correlation id promoted.

    JSON lines rather than prose because these are meant to be aggregated: the
    operational questions are "what is the refusal rate this week" and "which
    checks fire most often", and neither is answerable by grepping sentences.
    """

    def format(self, record: logging.LogRecord) -> str:
        import os
        payload: dict[str, Any] = {
            "at": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
            # OpenTelemetry resource attributes
            "service.name": os.getenv("OTEL_SERVICE_NAME", "technical-knowledge-assistant"),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "development"),
        }
        # The trace ids sit above the event's own fields, so a line is legible
        # left to right: when, what, where in the tree, then the detail. Empty
        # ids are omitted rather than written as "" — a CLI caller genuinely has
        # no session, and a column of empty strings says less than its absence.
        for key, attribute in (("correlation_id", "correlation_id"),
                               ("trace_id", "correlation_id"),
                               ("span_id", "span_id"),
                               ("parent_span_id", "parent_span_id"),
                               ("turn_id", "turn_id"),
                               ("session_id", "session_id")):
            value = getattr(record, attribute, "")
            if value:
                payload[key] = value
        for key in ("duration_ms", "status"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
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
