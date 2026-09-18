"""One conversation, several walls. What must not travel between them.

The contamination this file exists to prevent was measured, not imagined.
Traced through the real `ask_turn` path before the fix:

    turn 1  "My internal wall is brick."      -> {substrate: brick, location: internal}
    turn 2  "Now I have another wall outside." -> {substrate: brick, location: external}

A wall that exists nowhere, with the invented half printed back as "brick, as
you told me earlier" — which attributes it to the customer. That is the worst
shape this failure can take: a confident, cited recommendation for a building
nobody described, blamed on the person who will act on it.

**Every test here drives the real graph**, because the bug lived in the
interaction between `understanding.resolve`'s inheritance loop and the absence
of any case concept — neither of which is visible from a unit test of either
one. The predicate `opens_a_new_case` is tested directly too, but only the
end-to-end tests prove the behaviour.

**The two errors are not symmetric, and the tests are written to that.** Opening
a case that did not need opening costs one extra question. Failing to open one
recommends a product for the wrong wall. So the suite is strict about missed
boundaries and tolerant about extra ones, and `opens_a_new_case` errs the same
way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.infrastructure import ollama
from assistant.turn.conversation import (
    Case,
    ConversationState,
    FactStatus,
    NewCase,
    SessionFact,
    TurnInput,
    merge_facts,
    opens_a_new_case,
)
from assistant.answering.answer import (  # noqa: E402
    Provenance,
)
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)

from test_engine import build_repo, quoting, unit                  # noqa: E402
from test_graph import StubVision                                  # noqa: E402


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo, cache=False, log=False)
    finally:
        repo.close()


def turn(question, index=1, images=(), session="s1"):
    t = TurnInput(raw_question=question, turn_index=index,
                  images=tuple(images), session_id=session)
    object.__setattr__(t, "history", "")
    return t


def converse(assistant, questions, state=None, vision=None):
    state = state or ConversationState()
    for i, q in enumerate(questions, start=1):
        _reply, state = assistant.ask_turn(turn(q, i), state, vision=vision)
    return state


# ------------------------------------------------- the measured contamination


def test_another_wall_does_not_inherit_the_first_walls_substrate(assistant):
    """The regression. Brick belonged to an internal wall; this one is outside."""
    state = converse(assistant, ["My internal wall is brick.",
                                 "Now I have another wall outside."])

    assert state.active().get("substrate") is None, (
        f"brick leaked onto a second wall: {state.active()}")
    assert state.active().get("location") == "external"


def test_the_leak_stays_closed_on_the_turn_that_would_use_it(assistant):
    """Turn two looked clean once before; turn three is where it was consumed."""
    state = converse(assistant, ["My internal wall is brick.",
                                 "Now I have another wall outside.",
                                 "What should I use on it?"])

    assert "substrate" not in state.active()


def test_a_second_wall_opens_a_case(assistant):
    state = converse(assistant, ["My internal wall is brick.",
                                 "I have a second wall to do."])

    assert "substrate" not in state.active()


def test_the_turn_that_opens_a_case_keeps_what_it_said_itself(assistant):
    """"Another wall outside" establishes `external` about the *new* wall.

    Dropping it with the old case would mean asking immediately about something
    the person had just told us, which is how a safety mechanism becomes an
    annoyance and then gets switched off.
    """
    state = converse(assistant, ["My internal wall is brick.",
                                 "Now I have another wall outside."])

    assert state.active() == {"location": "external"}


# ----------------------------------------------- a correction is not a new case


def test_a_correction_supersedes_rather_than_opening_a_case(assistant):
    """The opposite error, and the more damaging of the two to get wrong.

    Treating "actually it is stone" as a new subject would file the correction
    into a fresh case and leave the original still answering from brick — the
    person corrects themselves and is ignored.
    """
    state = converse(assistant, ["My internal wall is brick.",
                                 "Actually the wall is stone."])

    assert state.active()["substrate"] == "stone"
    assert state.history == (), "a correction retired the case it was correcting"
    assert [f.value for f in state.facts["substrate"].superseded] == ["brick"]


def test_a_correction_keeps_the_rest_of_the_case(assistant):
    state = converse(assistant, ["My internal brick wall, north facing.",
                                 "Actually the wall is stone."])

    assert state.active()["substrate"] == "stone"
    assert state.active().get("location") == "internal", "the case was reset"


def test_an_apology_that_does_introduce_a_wall_still_opens_a_case(assistant):
    """"Actually I have another wall" is both an apology and a new wall.

    Reading only the apology loses the wall, so an explicit introducer plus a
    subject noun outranks the correction marker.
    """
    state = converse(assistant, ["My internal wall is brick.",
                                 "Actually, I have another wall outside."])

    assert "substrate" not in state.active()


# ------------------------------------------------------- the same wall carries


def test_a_follow_up_about_the_same_wall_keeps_everything(assistant):
    """The cost of over-eager boundaries, guarded against."""
    state = converse(assistant, ["My internal wall is brick.",
                                 "What thickness should it be?",
                                 "And how long does it take to dry?"])

    assert state.active()["substrate"] == "brick"
    assert state.active()["location"] == "internal"
    assert state.history == ()


@pytest.mark.parametrize("follow_up", [
    "What thickness should it be?",
    "How many coats?",
    "Would Solo work on it?",
    "What about a different product?",
    "Is there another option?",
])
def test_ordinary_follow_ups_do_not_open_a_case(assistant, follow_up):
    """"Another option" and "a different product" are not another wall.

    The introducer alone is not enough; a subject noun has to be there too, or
    every conversational filler becomes a case boundary.
    """
    state = converse(assistant, ["My internal wall is brick.", follow_up])

    assert state.active().get("substrate") == "brick", (
        f"{follow_up!r} wrongly opened a case")


# -------------------------------------------------------- images and new cases


def test_an_image_observation_does_not_survive_into_a_new_case(assistant):
    """A photograph of one wall must not fill a slot on a different one.

    The sharpest form of the leak: the image has scrolled out of the page, so
    nobody can see what the slot was filled from, and an observation cannot be
    re-checked the way a person can be re-asked.
    """
    vision = StubVision({"substrate": "brick"})
    state = ConversationState()
    _r, state = assistant.ask_turn(
        turn("What should I use here?", 1, images=["IMG_1"]), state, vision=vision)
    assert state.active().get("substrate") == "brick"

    _r, state = assistant.ask_turn(turn("Now I have another wall outside.", 2),
                                   state, vision=vision)

    assert "substrate" not in state.active()
    assert state.observations == ()


# ----------------------------------------------------------- the audit trail


def test_the_retired_case_is_kept_rather_than_discarded(assistant):
    """The earlier answers are still in the transcript and still need explaining.

    Same argument decision 18 makes for superseded document versions: a value
    that is no longer served is not a value that was never true.
    """
    state = converse(assistant, ["My internal wall is brick.",
                                 "Now I have another wall outside."])

    assert len(state.history) == 1
    retired = state.history[0]
    assert retired.facts["substrate"].current.value == "brick"
    assert retired.case_id == "case-1"


def test_the_new_case_records_why_it_opened(assistant):
    """A case boundary is a judgement; an uninspectable judgement is uncorrectable."""
    state = converse(assistant, ["My internal wall is brick.",
                                 "Now I have another wall outside."])

    assert "another" in state.current.opened_because
    assert state.current.opened_at_turn == 2


def test_a_retired_case_is_not_readable_by_routing(assistant):
    """Retained for audit, invisible to the thing that chooses a product."""
    state = converse(assistant, ["My internal wall is brick.",
                                 "Now I have another wall outside."])

    assert "brick" not in state.active().values()
    assert "brick" not in [h.current.value for h in state.facts.values()]


def test_a_new_chat_is_a_new_thread_and_shares_nothing(assistant):
    """The strongest reset, and it is a new *thread* rather than a new object.

    This test used to hand in a fresh `ConversationState` on the same session
    and expect a clean slate. That stopped being the right assertion when the
    checkpointer became the source of continuity, and the change is a
    correction rather than a compromise: continuity must not depend on the
    caller remembering to pass state back, so passing a blank one can no longer
    be what wipes a conversation. "New chat" is a new session id -- which is
    exactly what the web UI's `/new` route already issues.
    """
    _r, _s = assistant.ask_turn(turn("My internal wall is brick.", 1, session="old"))

    reply, fresh = assistant.ask_turn(turn("What should I use?", 1, session="new"))

    assert fresh.active() == {}
    assert fresh.history == ()
    assert fresh.current.case_id == "case-1"
    assert assistant.conversation_state("old").active()["substrate"] == "brick", (
        "the previous conversation should still exist, just not be visible here")


def test_forgetting_a_thread_drops_the_conversation(assistant):
    """What "New chat" does underneath when the session id is reused."""
    assistant.ask_turn(turn("My internal wall is brick.", 1, session="reused"))
    assert assistant.conversation_state("reused").active()

    assistant.forget("reused")

    assert assistant.conversation_state("reused").active() == {}


# ----------------------------------------------------- the predicate directly


@pytest.mark.parametrize("question,expected", [
    ("My internal wall is brick.", False),
    ("Now I have another wall outside.", True),
    ("I have a second wall to do.", True),
    ("And a different wall outside too.", True),
    ("I also have a gable to render.", True),
    ("Actually the wall is stone.", False),
    ("Sorry, not brick - it is stone.", False),
    ("Actually I have another wall.", True),
    ("What thickness should it be?", False),
    ("What about a different product?", False),
])
def test_the_boundary_predicate(question, expected):
    assert opens_a_new_case(question)[0] is expected, question


def test_the_model_may_open_a_case_on_its_own():
    """Deliberate: a wrongly-opened case costs a question, a missed one costs a wall."""
    assert opens_a_new_case("Now the front of the house.", model_says=True)[0]


def test_a_correction_marker_vetoes_the_models_signal():
    """The one case where the model is overruled, and why.

    A model reading "actually it is stone" as a new subject makes exactly the
    mistake that loses a correction into a fresh case.
    """
    assert not opens_a_new_case("Actually it is stone.", model_says=True)[0]


# --------------------------------------------------------- the reducer itself


def test_the_reset_goes_through_the_reducer_like_every_other_write():
    """`NewCase` is an instruction, not an assignment.

    If a node could set the facts channel directly, supersession would be
    bypassable and `merge_facts` would stop being the single place the rules
    are enforced. So resetting is a value the reducer recognises.
    """
    before = merge_facts({}, {
        "substrate": SessionFact("substrate", "brick", Provenance.STATED, 1),
        "location": SessionFact("location", "internal", Provenance.STATED, 1)})

    after = merge_facts(before, NewCase(
        keep={"location": SessionFact("location", "external",
                                      Provenance.STATED, 2)},
        reason="'another' + 'wall'"))

    assert set(after) == {"location"}
    assert after["location"].current.value == "external"


def test_opening_a_case_files_the_old_one_and_starts_a_clean_slate():
    state = ConversationState(facts=merge_facts({}, {
        "substrate": SessionFact("substrate", "brick", Provenance.STATED, 1)}))

    state.open_case(turn_index=2, because="'another' + 'wall'")

    assert state.current.case_id == "case-2"
    assert state.active() == {}
    assert state.history[0].facts["substrate"].current.value == "brick"
