"""`ASSISTANT_VISION_DEMO`: one switch, and off is a supported state.

The demo-platform branch reports vision disabled unless the flag is set, so
this branch has to meet that contract rather than assume image reading is
always on. The requirement is narrow and the failure modes are not, which is
what these tests are for.

**Off must not be silent.** The tempting implementation returns nothing when
the flag is clear, and that reproduces the exact bug `analyse_images` already
carries a comment about: every surface passed no provider, every photograph
was ignored, and the page still said it had read one. An upload that is not
looked at has to *say* it was not looked at, or a customer reads a general
answer as a reading of their wall.

**Off must not be a different code path.** A flag that produced a differently
shaped response would mean every renderer needed a second branch, and the
branch nobody exercises is the one that breaks in the room. `disabled_report()`
returns the same keys as `perception_report()`.

**An injected provider is not gated.** Passing one is a deliberate act by a
test, a channel adapter or the demo driver. A flag that overrode it would make
the injection point untestable, and would mean this file could not be written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant.answering import vision
from assistant.infrastructure import ollama
from assistant.turn.conversation import ConversationState, TurnInput      # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)

from test_engine import build_repo, quoting, unit                    # noqa: E402
from test_vision_conversation import Camera, observation             # noqa: E402


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo, source="test")
    finally:
        repo.close()


def turn(question, index=1, images=(), session="flag"):
    made = TurnInput(raw_question=question, turn_index=index,
                     images=tuple(images), session_id=session)
    object.__setattr__(made, "history", "")
    return made


def perception_of(reply) -> dict:
    for _question, answer in reply.parts:
        found = answer.diagnostics.get("perception")
        if found:
            return found
    return {}


# ------------------------------------------------------------- the switch


@pytest.mark.parametrize("value, expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True),
    ("on", True), ("enabled", True), (" 1 ", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("", False), ("maybe", False),
])
def test_the_flag_is_read_the_way_an_operator_would_write_it(value, expected):
    assert vision.enabled({vision.VISION_DEMO_FLAG: value}) is expected


def test_an_unset_flag_means_off():
    """Off by default, and that is the right way round.

    Perception costs one to three minutes per image on a processor with no
    graphics card, so an accidental upload on a shared deployment is a
    denial-of-service with good intentions. The vision model is also a separate
    pull an operator may not have.
    """
    assert vision.enabled({}) is False


def test_the_flag_is_read_at_call_time_not_at_import(monkeypatch):
    """So a process can be started with it set, and a test can toggle it."""
    monkeypatch.delenv(vision.VISION_DEMO_FLAG, raising=False)
    assert vision.enabled() is False
    monkeypatch.setenv(vision.VISION_DEMO_FLAG, "1")
    assert vision.enabled() is True


# ------------------------------------------------------- off, but not silent


def test_switched_off_the_upload_is_acknowledged_rather_than_ignored(
        assistant, monkeypatch):
    """The whole point of the disabled report.

    Returning nothing would leave the turn indistinguishable from one where
    perception ran and found nothing, and a customer would read a general
    answer as a reading of their wall.
    """
    monkeypatch.delenv(vision.VISION_DEMO_FLAG, raising=False)
    called = []
    monkeypatch.setattr(vision, "slots_from_images",
                        lambda *a, **k: called.append(1))

    reply, state = assistant.ask_turn(turn(
        "what should I use on this wall?", images=["wall.jpg"]),
        ConversationState())

    assert called == [], "the model was called although vision is switched off"
    report = perception_of(reply)
    assert report["enabled"] is False
    assert report["summary"], "an upload was dropped without a word"
    assert "not looked at" in " ".join(report["summary"])
    assert report["routed"] == {}
    assert state.observations == ()


def test_the_disabled_report_has_the_same_shape_as_a_real_one():
    """One renderer, one set of keys, no second branch to leave untested."""
    real = vision.perception_report(vision.resolve([vision.Perception(
        observations=(vision.Observation("substrate", "brick", 0.95, "IMG_001",
                                         (0.0, 0.0, 1.0, 1.0)),),
        cannot_determine_from_image=("the moisture source",))]))
    off = vision.disabled_report()

    assert set(off) == set(real)
    assert real["enabled"] is True and off["enabled"] is False
    # And it survives a checkpoint like the real one.
    assert json.loads(json.dumps(off)) == off


def test_switched_off_the_conversation_still_works(assistant, monkeypatch):
    """Coverage drops; nothing else does. Decision 16's published behaviour."""
    monkeypatch.delenv(vision.VISION_DEMO_FLAG, raising=False)

    reply, state = assistant.ask_turn(turn(
        "I have an internal brick wall. How much water does Solo need?",
        images=["wall.jpg"]), ConversationState())

    assert reply.parts, "an upload cost the caller their answer"
    assert state.active()["substrate"] == "brick", (
        "the typed facts were lost because a photograph was not read")


# --------------------------------------------------------------- on, and on


def test_switched_on_the_default_provider_is_the_real_one(assistant, monkeypatch):
    monkeypatch.setenv(vision.VISION_DEMO_FLAG, "1")
    seen = []

    def fake(images, *a, **k):
        seen.append(list(images))
        return vision.resolve([vision.Perception(
            observations=(vision.Observation("substrate", "brickwork", 0.95,
                                             "IMG_001", (0.0, 0.0, 1.0, 1.0)),),
            cannot_determine_from_image=("the moisture source",))])

    monkeypatch.setattr(vision, "slots_from_images", fake)

    reply, state = assistant.ask_turn(turn(
        "what thickness should this be inside?", images=["wall.jpg"]),
        ConversationState())

    assert seen == [["wall.jpg"]]
    report = perception_of(reply)
    assert report["enabled"] is True
    assert report["routed"] == {"substrate": "brick"}


def test_an_injected_provider_runs_whatever_the_flag_says(assistant, monkeypatch):
    """Injection is a deliberate act and is not gated.

    A flag that overrode it would make the injection point untestable -- and
    would mean this file could not be written, since every test in it supplies
    a double.
    """
    monkeypatch.delenv(vision.VISION_DEMO_FLAG, raising=False)
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("exposed_masonry", "exposed", 0.92))

    reply, _state = assistant.ask_turn(turn(
        "what thickness should this be inside?", images=["wall.jpg"]),
        ConversationState(), vision=camera)

    assert camera.calls == 1
    assert perception_of(reply)["routed"] == {"substrate": "brick"}


def test_a_turn_with_no_photograph_is_unaffected_either_way(
        assistant, monkeypatch):
    for value in ("1", ""):
        if value:
            monkeypatch.setenv(vision.VISION_DEMO_FLAG, value)
        else:
            monkeypatch.delenv(vision.VISION_DEMO_FLAG, raising=False)
        reply, _state = assistant.ask_turn(
            turn("how much water does Solo need?", session=f"s{value}"),
            ConversationState())
        assert not perception_of(reply), (
            "a turn with no upload produced a perception report")


# ------------------------------------------------------ the tools are explicit


def test_the_direct_entry_points_are_not_gated(monkeypatch):
    """`observe()` is what the evaluation and the demo driver call.

    Running either *is* the act of asking for vision, so the flag would be
    asking the same question twice. The gate belongs on the request path, where
    an upload arrives without anybody having decided anything.
    """
    monkeypatch.delenv(vision.VISION_DEMO_FLAG, raising=False)
    called = []

    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"response": json.dumps(
                {"observations": [], "cannot_determine_from_image": []})}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            called.append(url)
            return _Response()

    monkeypatch.setattr(vision.httpx, "Client", _Client)

    assert vision.observe(b"\x89PNG\r\n\x1a\nxxxx").ok
    assert called == ["/api/generate"], (
        "the flag gated a direct call that is itself an explicit request")
