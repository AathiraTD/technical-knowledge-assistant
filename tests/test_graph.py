"""The turn as a state machine, end to end, with no network.

`assistant/graph.py` is orchestration and nothing else, so these tests are about
**order and state** rather than about answers: which node ran, what the
conversation believed afterwards, and which edges exist at all.

The last of those is the one worth stating plainly. The requirement that
insufficient evidence must never fall through to Compose is not enforced by a
condition somewhere that could be edited out; it is enforced by there being no
edge from the evidence gate to the composing path. A graph is a structure that
can be asserted against, which is most of the argument for having one.

A stubbed `VisionProvider` supplies known observations, so the image behaviour
is tested deterministically and offline. The live-model equivalent is a separate
opt-in suite; this one must never need Ollama.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.retrieval import candidates as cand
from assistant.turn import graph as g
from assistant.infrastructure import ollama
from assistant.answering import understanding as und
from assistant.answering.answer import (  # noqa: E402
    Answer,
    Provenance,
    SlotFact,
)
from assistant.turn.conversation import (
    ConversationState,
    FactStatus,
    SessionFact,
    TurnInput,
)
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.answering.router import (  # noqa: E402
    Path_,
)

from test_engine import build_repo, quoting, unit                   # noqa: E402


class StubVision:
    """A `VisionProvider` that reports exactly what a test tells it to.

    The structural half of the evaluation: orchestration, state, conflict and
    recommendation behaviour are all reachable without a vision model, a GPU or
    a network. What this cannot test is whether a real VLM reads a wall
    correctly, which is why the live suite exists separately.
    """

    def __init__(self, slots: dict, confidence: float = 0.9):
        self.slots = dict(slots)
        self.confidence = confidence
        self.calls = 0

    def observe(self, images):
        self.calls += 1

        class Attribute:
            def __init__(self, slot, value, confidence):
                self.slot, self.value, self.confidence = slot, value, confidence
                self.sources = ()

        class Resolution:
            pass

        resolution = Resolution()
        resolution.slots = dict(self.slots)
        resolution.attributes = tuple(
            Attribute(s, v, self.confidence) for s, v in self.slots.items())
        resolution.cannot_determine_from_image = ("moisture source",)
        resolution.discarded = ()
        return resolution


@pytest.fixture
def no_ollama(monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)


@pytest.fixture
def assistant(tmp_path, no_ollama):
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo, cache=False, log=False)
    finally:
        repo.close()


def turn(question, index=1, images=(), session="s1", history=""):
    t = TurnInput(raw_question=question, turn_index=index, images=tuple(images),
                  session_id=session)
    object.__setattr__(t, "history", history)
    return t


# ------------------------------------------------------- structure of the graph


def test_there_is_no_edge_from_insufficient_evidence_to_compose():
    """The non-negotiable requirement, asserted as a property of the graph.

    Not "no code path currently takes it" -- no such edge exists, so no edit to
    a condition can create one by accident. `after_evidence` can only return
    three node names, and none of them composes over unapproved passages.
    """
    import inspect

    source = inspect.getsource(g.build)
    # The body of `after_evidence` only, stopping at the graph wiring that
    # follows it. Splitting on "def " instead would run past the end of the
    # function and swallow every node name in the file.
    body = source.split("def after_evidence")[1].split("graph = StateGraph")[0]

    assert "delegate" not in body, (
        "the evidence gate can reach the general composing path")
    for expected in ("recommend", "need_more_information",
                     "no_supported_recommendation"):
        assert expected in body


def test_every_outcome_is_one_of_the_five_named_ones():
    assert {o.value for o in cand.Outcome} == {
        "supported_recommendation", "conditional_recommendation",
        "need_more_information", "no_supported_recommendation", "handoff"}


# ------------------------------------------------------------- the dependency


def test_the_hosted_tracing_client_is_inert():
    """The condition under which importing this dependency is acceptable.

    `langsmith` arrives transitively through `langchain-core` and is a client
    for a hosted tracing service. One environment variable would otherwise ship
    whole customer conversations to a third party. It is switched off in code
    before the import, and that is asserted rather than documented, because a
    posture that depends on nobody setting an env var is not a posture.
    """
    assert g.tracing_disabled()


def test_the_checkpointer_will_not_rehydrate_arbitrary_classes():
    """A checkpoint is data. A deserialiser that constructs anything it is told
    to is a deserialisation bug waiting for a writable checkpoint store."""
    modules = {module for module, _name in g.ALLOWED_TYPES}

    # Only this application's own domain types. Nothing from the standard
    # library, nothing from a dependency, and no wildcard.
    assert modules
    assert all(m == "assistant" or m.startswith("assistant.") for m in modules)


def test_every_allowed_checkpoint_type_actually_resolves():
    """Each entry must name a class that exists, at the path it claims.

    Regression test. The allowlist used to be dotted paths typed out beside
    the class names, and moving `candidates` into `assistant.retrieval` left
    four entries pointing at a module that no longer existed. Nothing failed
    loudly: an entry that matches nothing simply stops permitting anything,
    so the checkpointer would have refused to rehydrate a candidate
    assessment and returned None in its place. The old test asserted the
    literal contents of the list, so it agreed with the stale strings.
    """
    for module, name in g.ALLOWED_TYPES:
        resolved = getattr(importlib.import_module(module), name, None)
        assert resolved is not None, f"{module}.{name} does not exist"
        assert resolved.__module__ == module, (
            f"{name} is listed under {module} but lives in {resolved.__module__}")

    # The types whose absence the checkpointer reported on every turn.
    listed = {(m, n) for m, n in g.ALLOWED_TYPES}
    assert (Answer.__module__, "Answer") in listed
    assert (SlotFact.__module__, "SlotFact") in listed
    assert (SessionFact.__module__, "SessionFact") in listed


# --------------------------------------------------------- existing routes


@pytest.mark.parametrize("question", [
    "How much water does Solo need per bag?",
    "What coverage does Duro give?",
    "How much does Solo cost?",
])
def test_a_non_selection_is_answered_by_the_existing_engine(assistant, question):
    """Seven routes already work. The graph orders the turn; it does not
    re-implement them, and a regression in any would be a cost with no
    matching benefit."""
    reply, _state = assistant.ask_turn(turn(question))

    assert reply.parts, "no answer was produced"
    _q, answer = reply.parts[0]
    assert "delegate" in " ".join(answer.diagnostics.get("graph", []))


def test_the_policy_gate_still_fires_inside_the_graph(assistant):
    reply, _ = assistant.ask_turn(turn("How much does Solo cost?"))

    _q, answer = reply.parts[0]
    assert answer.path == Path_.ROUTE.value


# --------------------------------------------------------- the select flow


def test_a_selection_without_a_substrate_asks_rather_than_guesses(assistant):
    reply, _ = assistant.ask_turn(turn("What product should I use on my wall?"))

    _q, answer = reply.parts[0]
    assert answer.diagnostics["outcome"] == "need_more_information"
    assert answer.path == Path_.ASK_BACK.value


def test_the_ask_names_the_fact_it_needs(assistant):
    reply, _ = assistant.ask_turn(turn("What product should I use on my wall?"))

    _q, answer = reply.parts[0]
    assert "built of" in answer.text, answer.text
    # Substrate only. Decision 10 answers an uncued inside/outside per option
    # rather than asking, so asking for it here would be a question the system
    # already knows how to avoid.
    assert answer.diagnostics["missing"] == ["substrate"]


def test_a_selection_with_no_supporting_evidence_refuses(assistant):
    """The fixture's passages say nothing about cob, so nothing is recommended."""
    reply, _ = assistant.ask_turn(
        turn("What product should I use on my internal cob wall?"))

    _q, answer = reply.parts[0]
    assert answer.diagnostics["outcome"] == "no_supported_recommendation"
    assert answer.refused


# ------------------------------------------------- images and their lifecycle


def test_a_photograph_fills_a_slot_nobody_stated(assistant):
    vision = StubVision({"substrate": "brick"})

    reply, state = assistant.ask_turn(
        turn("What product should I use here?", images=["IMG_1"]), vision=vision)

    assert vision.calls == 1
    assert state.facts["substrate"].current.provenance is Provenance.OBSERVED


def test_a_photograph_does_not_overrule_what_the_person_said(assistant):
    """Turn three of the continuity scenario, through the real graph."""
    vision = StubVision({"substrate": "stone"})
    state = ConversationState()

    _reply, state = assistant.ask_turn(
        turn("I have an internal brick wall.", index=1), state)
    _reply, state = assistant.ask_turn(
        turn("Does the photo change anything?", index=2, images=["IMG_1"]),
        state, vision=vision)

    assert state.facts["substrate"].current.value == "brick"
    assert state.facts["substrate"].current.status is FactStatus.CONFLICTING
    assert state.facts["substrate"].contradicted_by


def test_a_disputed_substrate_makes_a_selection_ask_rather_than_pick(assistant):
    vision = StubVision({"substrate": "stone"})
    state = ConversationState()

    _r, state = assistant.ask_turn(turn("I have an internal brick wall.", 1), state)
    reply, state = assistant.ask_turn(
        turn("What product should I use?", 2, images=["IMG_1"]), state,
        vision=vision)

    _q, answer = reply.parts[0]
    assert answer.diagnostics["outcome"] == "need_more_information"
    assert "do not agree" in answer.text


def test_no_images_means_the_vision_provider_is_never_called(assistant):
    vision = StubVision({"substrate": "stone"})

    assistant.ask_turn(turn("How much water does Solo need per bag?"))

    assert vision.calls == 0


# ----------------------------------------------- the four-turn continuity case


def test_the_four_turn_scenario(assistant):
    """Ultra/brick/internal, then a quantity, then an image, then a correction.

    The scenario the whole state layer exists for. Turn four is the one that
    matters: "actually the wall is stone" must **supersede** brick rather than
    accumulate beside it, and the brick that was believed for three turns must
    still be on file, because an answer given while it was believed has to stay
    explicable.
    """
    vision = StubVision({"substrate": "stone"})
    state = ConversationState()

    _r, state = assistant.ask_turn(
        turn("I want to insulate an internal brick wall with Solo.", 1), state)
    assert state.active()["substrate"] == "brick"

    _r, state = assistant.ask_turn(
        turn("How much would I need for 30 square metres at 25mm?", 2), state)
    assert state.active()["substrate"] == "brick", "the wall did not change"

    _r, state = assistant.ask_turn(
        turn("Here is the wall. Does anything change?", 3, images=["IMG_1"]),
        state, vision=vision)
    assert state.facts["substrate"].current.status is FactStatus.CONFLICTING

    _r, state = assistant.ask_turn(turn("Actually the wall is stone.", 4), state)

    assert state.active()["substrate"] == "stone"
    assert state.facts["substrate"].current.status is FactStatus.ACTIVE
    assert [f.value for f in state.facts["substrate"].superseded] == ["brick"], (
        "the superseded value was discarded instead of retained")


def test_state_survives_across_turns_without_a_transcript(assistant):
    """Continuity comes from typed facts, never from replaying prose."""
    state = ConversationState()

    _r, state = assistant.ask_turn(turn("I have an internal brick wall.", 1), state)
    reply, state = assistant.ask_turn(
        turn("What thickness should it be?", 2), state)

    assert state.active()["substrate"] == "brick"
    _q, answer = reply.parts[0]
    assert answer.diagnostics["slots"].get("substrate") == "brick"


def test_the_transcript_is_not_what_carries_the_conversation(assistant):
    """The same two turns with no history string still inherit everything."""
    state = ConversationState()
    _r, state = assistant.ask_turn(
        turn("I have an internal brick wall.", 1, history=""), state)
    _r, state = assistant.ask_turn(
        turn("What thickness should it be?", 2, history=""), state)

    assert state.active()["substrate"] == "brick"
