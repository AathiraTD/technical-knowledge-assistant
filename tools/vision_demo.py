"""Play the image scenario against the real stack, one turn at a time.

The evaluation measures perception in isolation. This is the other thing a
panel wants: the **conversation**, run end to end -- real index, real
retrieval, real generation, real vision model -- so what is described can be
shown rather than asserted.

    python tools/vision_demo.py                       # the four-turn scenario
    python tools/vision_demo.py --scenario image-only # photo first, no context
    python tools/vision_demo.py --image render-external-sound
    python tools/vision_demo.py --dry-run             # no model; shape only

**It prints the seams, not just the prose.** Each turn shows the route taken,
the facts the system is holding and where each came from, and -- when a
photograph was sent -- every reading with its certainty and the reason any of
them was not acted on. Those are the things the design claims and they are
invisible in the answer text.

A word on what to expect from the clock: a compose on a question the model has
not seen costs tens of seconds on a processor with no graphics card, and one
photograph through the vision model costs one to three minutes. That is
measured, it is in the record, and it is why the submission's evidence is a
transcript. Run this before the room, not in it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.turn.conversation import ConversationState, TurnInput      # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.store.factory import open_repository                  # noqa: E402

IMAGES = ROOT / "eval" / "fixtures" / "images"

# The recommended demonstration, and the reasoning behind each turn is in the
# `why` line rather than in a comment, because it is what to say out loud.
SCENARIOS = {
    "four-turn": [
        dict(say="I have an old solid brick wall and want to improve its "
                 "insulation. Would Lime Green Ultra be suitable internally?",
             why="Establishes the wall and the product in the customer's own "
                 "words. Everything later has to survive this."),
        dict(say="How much would I need for 30 m2 at 25 mm?",
             why="A quantity on the same wall. Watch the substrate and the "
                 "product carry without the transcript being replayed."),
        dict(say="This is the wall I'm talking about. What can you reliably "
                 "identify from the photo, and does anything here change the "
                 "guidance?",
             image=True,
             why="The turn the demo exists for. The photograph is read, every "
                 "reading is reported with its certainty, and the guidance "
                 "moves only if the evidence moves it. Note that `location` "
                 "is read and not acted on."),
        dict(say="There are also patchy areas after drying. What could be "
                 "causing that?",
             why="A cause is asked, with a photograph already in the "
                 "conversation that shows patchy staining. Nothing may invent "
                 "a cause from either."),
    ],
    "image-only": [
        dict(say="What product should I use here to insulate this?",
             image=True,
             why="A photograph and nothing else. The image settles the "
                 "substrate; it cannot settle inside-or-outside, and for an "
                 "insulation job that decides the product -- so the assistant "
                 "asks for exactly that one thing."),
        dict(say="inside",
             why="One word. The paused turn resumes and answers the question "
                 "originally asked, not the word typed."),
    ],
    "rendered-wall": [
        dict(say="What should I use to insulate this wall inside?",
             image=True,
             why="The substrate gate. The model reports a render and no "
                 "exposed masonry, so by its own two readings the background "
                 "is concealed -- and a substrate claim is refused with the "
                 "reason shown."),
    ],
}

DEFAULT_IMAGE = {"four-turn": "brick-exposed-clear",
                 "image-only": "brick-exposed-clear",
                 "rendered-wall": "render-external-sound"}


class NoVision:
    """A stand-in for `--dry-run`: the shape of a turn without the minutes."""

    @staticmethod
    def observe(images):
        from assistant.answering import vision
        return vision.resolve([vision.Perception(
            error="vision skipped (--dry-run)")])


def rule(text: str = "", char: str = "=") -> None:
    print(f"\n{char * 78}")
    if text:
        print(text)
        print(char * 78)


def show_perception(report: dict) -> None:
    if not report:
        return
    print("\n  FROM THE PHOTOGRAPH")
    if report.get("truncated"):
        print("    (the model's answer was cut short; what it said is shown "
              "and was not acted on)")
    for row in report.get("observations", ()):
        mark = "*" if row["routed"] else " "
        line = f"   {mark} {row['certainty']:<17} {row['attribute']}"
        if row["certainty"] in ("OBSERVED", "LIKELY"):
            line += f" = {row['value']}"
        print(f"{line}   ({row['confidence']})")
        if row["withheld"]:
            print(f"                        -- {row['withheld']}")
    for item in report.get("cannot_determine_from_image", ())[:5]:
        print(f"     CANNOT DETERMINE  {item}")
    if report.get("refused_attributes"):
        print(f"     REFUSED           {report['refused_attributes']}")
    print("\n   (* = this reading actually reached the router)")


def show_turn(index: int, step: dict, reply, state, seconds: float,
              image_name: str) -> None:
    rule(f"TURN {index}")
    print(f"  SAY: {step['say']}")
    if step.get("image"):
        print(f"  ATTACH: {image_name}.jpg")
    print(f"\n  WHY: {step['why']}")

    for part, answer in reply.parts:
        show_perception(answer.diagnostics.get("perception") or {})
        resumed = answer.diagnostics.get("resumed_question")
        rule(f"  ANSWER   [route: {answer.path}"
             + (f", refused" if answer.refused else "")
             + f"]  {seconds:.1f}s", char="-")
        if resumed:
            print(f"  (answering the earlier question: {resumed})")
        print("\n" + (answer.text or answer.body or "(no text)"))
        if answer.sources:
            print("\n  SOURCES")
            for source in answer.sources:
                print(f"    - {source}")
        if answer.facts:
            print("\n  ANSWERED FOR")
            for fact in answer.facts:
                print(f"    - {fact.sentence}")
        if answer.caveats:
            print("\n  CAVEATS")
            for caveat in answer.caveats:
                print(f"    - {caveat}")

    held = state.active()
    origins = state.provenance_of()
    print("\n  CONVERSATION NOW HOLDS")
    for slot in sorted(held):
        print(f"    {slot:<12} = {held[slot]:<14} "
              f"({origins.get(slot).value if origins.get(slot) else '?'})")
    if state.unsettled():
        print(f"    UNSETTLED: {state.unsettled()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="four-turn", choices=SCENARIOS)
    parser.add_argument("--image", default="",
                        help="fixture name, e.g. brick-exposed-clear")
    parser.add_argument("--audience", default="public")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip the vision model; show the turn shape only")
    args = parser.parse_args(argv)

    steps = SCENARIOS[args.scenario]
    image_name = args.image or DEFAULT_IMAGE[args.scenario]
    image_path = IMAGES / f"{image_name}.jpg"
    if not image_path.exists():
        print(f"no such fixture: {image_path}\n"
              f"run: python tools/make_image_fixtures.py")
        return 2

    try:
        repo = open_repository()
    except Exception as exc:                                   # noqa: BLE001
        print(f"could not open the knowledge store: {exc}")
        return 2
    snapshot = repo.snapshot()
    if snapshot is None:
        print("no active index. Run: python -m assistant.indexing.index")
        repo.close()
        return 2

    rule(f"{args.scenario}  --  image: {image_name}")
    print(f"  snapshot {snapshot.snapshot_id} | {snapshot.document_count} "
          f"documents, {snapshot.chunk_count} passages")
    print(f"  embeddings {snapshot.embedding_model} "
          f"({snapshot.embedding_dimensions}d), "
          f"chunking {snapshot.chunking_version}")
    if args.dry_run:
        print("  DRY RUN: the vision model is not called")

    assistant = Assistant(repo, source="demo")
    state = ConversationState()
    session = f"demo-{int(time.time())}"
    total = 0.0
    try:
        for index, step in enumerate(steps, start=1):
            turn = TurnInput(
                raw_question=step["say"], turn_index=index,
                images=(str(image_path),) if step.get("image") else (),
                audiences=(args.audience,), session_id=session)
            started = time.perf_counter()
            reply, state = assistant.ask_turn(
                turn, state, vision=NoVision() if args.dry_run else None)
            seconds = time.perf_counter() - started
            total += seconds
            show_turn(index, step, reply, state, seconds, image_name)
    finally:
        repo.close()

    rule(f"{len(steps)} turns in {total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
