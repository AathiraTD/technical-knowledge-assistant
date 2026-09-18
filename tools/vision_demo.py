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
from assistant.knowledge.store.factory import open_repository                  # noqa: E402

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
    """A separator, because a demonstration is read at a distance.

    Turns run to tens of seconds each and the output scrolls; without hard
    rules between them a room cannot tell where one answer ended and the next
    began. Called with no text it prints a bare line.
    """
    print(f"\n{char * 78}")
    if text:
        print(text)
        print(char * 78)


def show_perception(report: dict) -> None:
    """What the model saw, and — the part that matters — what was acted on.

    Every reading prints with its certainty, and the asterisk marks the ones
    that actually reached the router. The two are deliberately shown together
    because the claim being demonstrated is not "the model can see a brick
    wall"; it is that **a reading below the confidence floor leaves the slot
    uncued and changes nothing**, which is invisible unless the withheld
    readings are printed beside the used ones with the reason attached.

    `cannot_determine_from_image` is printed for the same reason: forcing the
    model to enumerate what it cannot tell is the visual equivalent of refusing
    to answer, and a demonstration that showed only the confident readings
    would be showing the half that flatters it.

    A truncated answer is announced rather than quietly trimmed — a model
    whose output was cut short may have been mid-observation, so what it said
    is shown and was not acted on.
    """
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
    """One turn with its seams exposed: the route, the evidence, the state.

    The answer text is the least interesting thing printed here. Around it go
    the route the deterministic router took, whether the part refused, the
    sources, the caveats appended by code, and — after the reply — every fact
    the conversation is now holding with where each one came from. Those are
    the design's claims, and they are exactly what prose hides.

    The `why` line is printed before the answer rather than kept in a comment,
    because it is what the presenter says out loud: it states what the turn is
    testing before the audience sees whether it held.

    `resumed_question` is surfaced when it is set, because after an ask-back
    the message typed was one word and the question answered was the one from
    two turns earlier — a turn that silently answered something other than what
    was just said would look like a non-sequitur rather than a resume.

    The elapsed seconds are printed on every turn deliberately. The latency is
    the honest constraint on this path and the record says so; hiding it in a
    demonstration would be the one dishonest thing this script could do.
    """
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
    """Play a scripted conversation against the real stack, turn by turn.

    Nothing is stubbed except, under `--dry-run`, the vision model itself. The
    index, retrieval, the router, the checks and generation are the ones that
    serve an answer, and the conversation runs as one session with the state
    carried from turn to turn — which is the thing being demonstrated, since
    the substrate established in turn one has to survive a quantity question, a
    photograph and a symptom without being restated.

    The scenarios are scripted rather than typed live for a reason the clock
    decides: a compose on an unseen question costs tens of seconds on a
    processor with no graphics card and one photograph costs minutes, so this
    is run before the room and read from, not run in it.

    Two preconditions are checked before anything slow starts — the image
    fixture exists, and the store has an active snapshot — each with the
    command that fixes it and exit code 2. Discovering a missing index after a
    three-minute vision call is the failure this ordering exists to avoid.

    The header prints the snapshot id, the counts and the model tags, so the
    output is self-describing: a transcript that cannot say which index and
    which models produced it is not evidence of anything.

    `--dry-run` substitutes `NoVision`, which returns a resolved perception
    carrying an error instead of readings. The turn shape, the routing and the
    ask-back behaviour are all still real — what disappears is the minutes,
    which makes it the right way to check the script itself.

    The store is closed in a `finally`, including on Ctrl-C partway through a
    scenario, because the next thing an operator does after an interrupted
    demonstration is rebuild the index.
    """
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
