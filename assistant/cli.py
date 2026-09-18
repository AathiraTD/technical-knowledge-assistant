"""The canonical interface.

`python -m assistant.cli` for a session, or `-q` for one question. The web page
is a view over this same library and adds no behaviour of its own, which is why
this is the interface the evaluation harness drives.

The audience set is asserted here with a flag rather than authenticated. That
is a real limitation and it is stated in the banner rather than hidden: in
production the audience comes from a signed-in session, and identity is the
first thing a deployment adds.

Structured logging is behind `--log` and writes to stderr rather than stdout.
Stdout is the transcript: the evaluation harness parses it and the submission
quotes it, so a log line in the middle of an answer is a corrupted artefact
rather than an inconvenience. The web page makes the opposite call, because a
server nobody can see is not operable.
"""

from __future__ import annotations

import os

import argparse
import sys

from . import observability as obs, ollama, use_utf8
from .conversation import TurnInput
from .answering.engine import Assistant, render
from .repository import IndexMismatch
from .store import EmbeddedRepository
from .store.factory import open_repository

BANNER = """Lime Green technical assistant
Answers only from Lime Green's published material, cites every source, and
refuses when the material does not answer the question.

  index      {docs} documents, {chunks} chunks, built {built}
  models     {embed} for retrieval, {gen} for composition
  audience   {audience}  (asserted at the command line, not authenticated)

Type a question, or 'quit'. Add -v for the routing diagnostics."""


def _open(args) -> EmbeddedRepository:
    return open_repository(args.db)


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
    parser.add_argument(
        "--image", action="append", default=[], metavar="PATH",
        help="a photograph to read slots from, repeatable. The vision stage "
             "may fill substrate, location, exposure and symptom, and nothing "
             "else: it never names a product and never decides a cause. A "
             "value it reads is reported as coming from the photograph rather "
             "than as something you said, and a reading it is not confident "
             "about leaves the slot uncued, which asks back exactly as it "
             "would with no photograph at all. Slow: a vision encoder on a "
             "processor with no graphics card is minutes per image.")
    parser.add_argument(
        "--log", action="store_true",
        help="write structured JSON events to stderr. Off by default: stdout "
             "is the transcript and the evaluation harness parses it, so log "
             "lines must not appear in it.")
    args = parser.parse_args(argv)
    if args.log:
        # stderr, so the transcript on stdout is byte-identical either way.
        obs.configure(sys.stderr)

    audiences = tuple(a.strip() for a in args.audience.split(",") if a.strip())

    repo = _open(args)
    try:
        assistant = Assistant(repo, threshold=args.threshold, source="cli")
    except (IndexMismatch, ollama.OllamaUnavailable) as exc:
        # Close the store before giving up. Process exit would release it
        # anyway, but the operator's next move after an index mismatch is to
        # rebuild into this very file, and holding it open is the one thing
        # that would make that fail too.
        repo.close()
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    if args.question:
        return _ask_once(assistant, args.question, audiences, args.verbose,
                         args.image)

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
        # The images belong to the invocation rather than to a turn, so a
        # session re-reads them on every question. That is the honest cost of
        # keeping perception per-turn: the slots a photograph settles are not
        # remembered between turns (they are a model's reading, not something
        # the caller said), so the alternative would be to promote them into
        # memory, which is precisely what must not happen.
        _ask_once(assistant, question, audiences, verbose, args.image)


def _ask_once(assistant, question: str, audiences, verbose: bool,
              images=None) -> int:
    try:
        # Through the graph, like every other surface. The CLI is still
        # single-turn -- each invocation is a new process, so there is nothing
        # for a thread id to continue -- and it goes this way anyway, because
        # what the graph adds is not only memory: structured understanding, the
        # case boundary, the SELECT flow and the evidence gate all live here.
        # A command-line question about which product suits a brick wall should
        # get the same answer the web page gives it.
        turn = TurnInput(raw_question=question, turn_index=1,
                         images=tuple(images or ()), audiences=tuple(audiences),
                         session_id=f"cli-{os.getpid()}")
        reply, _state = assistant.ask_turn(turn)
    except ollama.OllamaUnavailable as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    print()
    print(render(reply, show_diagnostics=verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
