"""Read back how one answer was produced.

`turn_traces` has been written since the spans landed, and until now nothing
could read it. The gap showed: a product-drift bug was debugged by adding
`print()` calls to the engine, the store and the page, when the span tree
already held the answer — the `retrieval` span records the product the question
named, the score of the top hit and how many documents it spanned, which is
exactly the evidence "why did it say WarmShell" needs.

So this adds a reader and not a second observability model. Every number here
comes from `KnowledgeRepository.traces()`, which both adapters already
implement; nothing new is recorded, no new table is written, and no attribute is
collected that was not being collected before.

**A CLI rather than an HTTP endpoint.** The trace of an answer is operator
evidence, and the surface that serves answers to anonymous callers is the wrong
place to expose it: an endpoint would need an audience gate, and an audience
gate on a debugging aid is a new access-control surface to get wrong. A reader
that runs where the database already is needs no gate at all. A browser test
proving the header maps to a real trace asserts through this reader, against the
same file, rather than through a route the server would have to grow.

    python -m assistant.trace                      the last few turns
    python -m assistant.trace 8fa7...              one answer, as a tree
    python -m assistant.trace --session c41e...    one conversation
    python -m assistant.trace 8fa7... --json       the same, for a test

The correlation id the page returns in `X-Correlation-Id` is the trace id, so
the value copied out of a browser's network tab is the value this takes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from . import use_utf8
from .model import TraceSpan
from .store.factory import open_repository

# Attributes worth putting on the span's own line rather than under it. These
# are the ones a question about a wrong answer actually turns on: which product
# retrieval was told about, how well the top passage scored, which router step
# fired, and which check refused. Everything else still prints, indented below.
HEADLINE = ("product", "top_score", "hits", "documents", "path", "step",
            "route", "refused", "check", "reason", "db.system",
            "gen_ai.request.model")


def _tree(spans: list[TraceSpan]) -> dict[str, list[TraceSpan]]:
    """Spans by parent, in the order they started."""
    children: dict[str, list[TraceSpan]] = defaultdict(list)
    for span in spans:
        children[span.parent_span_id].append(span)
    return children


def _roots(spans: list[TraceSpan]) -> list[TraceSpan]:
    """The spans whose parent is absent from this set.

    Not simply the ones with an empty `parent_span_id`. A conversation read by
    session returns several turns, and a limit can slice a tree so that a
    child's parent was left behind; both cases must still render rather than
    silently drop the spans whose parent cannot be found.
    """
    present = {span.span_id for span in spans}
    return [s for s in spans if not s.parent_span_id or s.parent_span_id not in present]


def _attributes(span: TraceSpan) -> tuple[str, list[str]]:
    """The headline attributes for the span's own line, and the rest."""
    head = [f"{k}={span.attributes[k]!r}" for k in HEADLINE
            if k in span.attributes]
    rest = [f"{k}={v!r}" for k, v in sorted(span.attributes.items())
            if k not in HEADLINE]
    return "  ".join(head), rest


def render(spans: list[TraceSpan], show_all: bool = False) -> str:
    """One turn's span tree, or several turns', as indented text."""
    if not spans:
        return "No spans for that id. Traces are kept for fourteen days."
    children, lines = _tree(spans), []

    def walk(span: TraceSpan, depth: int) -> None:
        pad = "  " * depth
        mark = "" if span.status == "ok" else f" [{span.status}]"
        head, rest = _attributes(span)
        lines.append(f"{pad}{span.name}  {span.duration_ms}ms{mark}"
                     + (f"  {head}" if head else ""))
        if show_all:
            lines.extend(f"{pad}    {line}" for line in rest)
        for child in children.get(span.span_id, []):
            walk(child, depth + 1)

    turn = ""
    for root in _roots(spans):
        # A session read spans several turns, so each one gets a header. A
        # single-trace read has one turn and the header would be noise.
        if root.turn_id != turn:
            turn = root.turn_id
            if len({s.turn_id for s in spans}) > 1:
                lines.append(f"\n--- turn {turn}  ({root.source}, {root.started_at})")
        walk(root, 0)
    return "\n".join(lines)


def recent(spans: list[TraceSpan], limit: int = 20) -> str:
    """The most recent turns, newest first, as ids to look one of up with."""
    seen: dict[str, TraceSpan] = {}
    for span in spans:
        if not span.parent_span_id:
            seen[span.trace_id] = span
    if not seen:
        return "No traces recorded yet."
    rows = list(seen.values())[-limit:]
    width = max(len(s.source) for s in rows)
    return "\n".join(
        f"{s.started_at}  {s.source:<{width}}  {s.duration_ms:>6}ms  {s.trace_id}"
        for s in reversed(rows))


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(
        prog="assistant.trace",
        description="Read back how one answer was produced.")
    parser.add_argument("trace_id", nargs="?", default="",
                        help="the correlation id from X-Correlation-Id")
    parser.add_argument("--session", default="",
                        help="every turn of one conversation")
    parser.add_argument("--db", default="data/index/knowledge.db")
    parser.add_argument("--dsn", default=None,
                        help="PostgreSQL DSN; defaults to ASSISTANT_POSTGRES_DSN")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--all", action="store_true",
                        help="every attribute, not only the headline ones")
    parser.add_argument("--json", action="store_true",
                        help="the spans as JSON, for a test to assert on")
    args = parser.parse_args(argv)

    repo = open_repository(args.db, args.dsn, apply_schema=False)
    spans = repo.traces(trace_id=args.trace_id, session_id=args.session,
                        limit=args.limit)

    if args.json:
        print(json.dumps([vars(s) for s in spans], indent=2))
        return 0 if spans else 1
    if not args.trace_id and not args.session:
        print(recent(spans))
        return 0
    print(render(spans, show_all=args.all))
    return 0 if spans else 1


if __name__ == "__main__":
    raise SystemExit(main())
