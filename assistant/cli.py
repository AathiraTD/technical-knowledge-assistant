"""The canonical interface.

`python -m assistant.cli` for a session, or `-q` for one question. The web page
is a view over this same library and adds no behaviour of its own, which is why
this is the interface the evaluation harness drives.

The audience set is asserted here with a flag rather than authenticated. That
is a real limitation and it is stated in the banner rather than hidden: in
production the audience comes from a signed-in session, and identity is the
first thing a deployment adds.
"""

from __future__ import annotations

import argparse
import sys

from . import ollama, use_utf8
from .engine import Assistant, render
from .repository import IndexMismatch
from .store import EmbeddedRepository

BANNER = """Lime Green technical assistant
Answers only from Lime Green's published material, cites every source, and
refuses when the material does not answer the question.

  index      {docs} documents, {chunks} chunks, built {built}
  models     {embed} for retrieval, {gen} for composition
  audience   {audience}  (asserted at the command line, not authenticated)

Type a question, or 'quit'. Add -v for the routing diagnostics."""


def _open(args) -> EmbeddedRepository:
    return EmbeddedRepository(args.db)


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(
        prog="assistant.cli", description="Ask the Lime Green technical assistant.")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("-a", "--audience", default="public",
                        help="comma-separated: public, trade, staff")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the path taken, the slots and the scores")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="override the abstention threshold")
    parser.add_argument("--db", default="data/index/knowledge.db")
    args = parser.parse_args(argv)

    audiences = tuple(a.strip() for a in args.audience.split(",") if a.strip())

    repo = _open(args)
    try:
        assistant = Assistant(repo, threshold=args.threshold)
    except IndexMismatch as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    except ollama.OllamaUnavailable as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    if args.question:
        return _ask_once(assistant, args.question, audiences, args.verbose)

    snapshot = repo.snapshot()
    print(BANNER.format(
        docs=snapshot.document_count, chunks=snapshot.chunk_count,
        built=snapshot.created_at[:10], embed=snapshot.embedding_model,
        gen=ollama.GENERATION_MODEL, audience=", ".join(audiences)))

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            return 0
        verbose = args.verbose
        if question.endswith(" -v"):
            question, verbose = question[:-3].strip(), True
        _ask_once(assistant, question, audiences, verbose)
    return 0


def _ask_once(assistant, question: str, audiences, verbose: bool) -> int:
    try:
        reply = assistant.ask(question, audiences=audiences)
    except ollama.OllamaUnavailable as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    print()
    print(render(reply, show_diagnostics=verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
