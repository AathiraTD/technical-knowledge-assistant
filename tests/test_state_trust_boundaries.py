"""What a turn says, and what the conversation is then entitled to believe.

Two failures, both traced through the real `ask_turn` path before the fix,
because neither is visible from a unit test of the slot detector -- the detector
is right in both cases and the wrongness is in what was done with its output.

    CASE A  "Would Ultra be suitable on an internal brick wall?"
            -> {substrate: brick, location: internal}, provenance STATED

    CASE B  turn 1  "My wall is brick."      -> {substrate: brick}
            turn 2  "My wall is not brick."  -> {substrate: brick}, source_turn 2

Case A invents a building. Nobody said they had a brick wall; they asked a
catalogue question about one, and every later turn was then answered for that
wall and printed it back as "brick (substrate), as you said" -- which attributes
the invention to the customer. It is the same shape as the cross-case
contamination in `tests/test_case_boundaries.py`, arriving through a different
door: not an old wall leaking forward, but a hypothetical wall being minted.

Case B is worse than ignoring the sentence. "Not brick" reached `merge_facts` as
an assertion of brick, rule 1 read it as a restatement, and the denial therefore
*refreshed* the fact it contradicted -- the system became more confident about a
value the person had just withdrawn.

Both tests drive the graph and read trusted state back through the checkpointer,
which is where the answer to "what does this conversation believe" actually
lives. A helper-level assertion would not have caught either bug: `facts_from`
and `merge_facts` were each behaving exactly as written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import ollama                                       # noqa: E402
from assistant.conversation import (                               # noqa: E402
    ConversationState, FactStatus, TurnInput,
)
from assistant.engine import Assistant                             # noqa: E402

from test_engine import build_repo, quoting, unit                  # noqa: E402


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo, cache=False, log=False)
    finally:
        repo.close()


def _turn(question: str, index: int, session: str) -> TurnInput:
    turn = TurnInput(raw_question=question, turn_index=index,
                     session_id=session)
    object.__setattr__(turn, "history", "")
    return turn


def _converse(assistant, questions, session):
    """Several turns through the public path. Returns (replies, final state)."""
    state = ConversationState()
    replies = []
    for index, question in enumerate(questions, start=1):
        reply, state = assistant.ask_turn(_turn(question, index, session),
                                          state)
        replies.append(reply)
    return replies, state


def test_a_suitability_question_does_not_mint_a_wall(assistant):
    """CASE A. The scenario constrains the request and is not remembered.

    Asserted on both halves, because only together are they the behaviour: the
    turn is still *answered* for an internal brick wall -- dropping the
    constraint would make the question unanswerable -- while the conversation
    afterwards holds no claim about the caller's building at all.
    """
    replies, state = _converse(
        assistant, ["Would Ultra be suitable on an internal brick wall?"],
        "trust-a")

    resolved_slots = replies[0].parts[0][1].diagnostics.get("slots", {})
    assert resolved_slots.get("substrate") == "brick", (
        "the scenario must still constrain this request")
    assert resolved_slots.get("location") == "internal"

    assert state.active() == {}, (
        f"a hypothetical wall became trusted state: {state.active()}")
    assert "substrate" not in state.facts
    assert "location" not in state.facts
    # The checkpoint is what the next turn reads, so it is the one that counts.
    assert assistant.conversation_state("trust-a").active() == {}


def test_a_denial_retires_the_fact_it_contradicts(assistant):
    """CASE B. "Not brick" stops brick being believed, and keeps the record.

    The third assertion is the one the bug was hiding behind: brick has to
    remain in the slot's history, because the answer given while it was believed
    still has to be explicable -- the same argument decision 18 makes for
    superseded document versions. What must not survive is brick as something
    the conversation still asserts.
    """
    _replies, state = _converse(
        assistant, ["My wall is brick.", "My wall is not brick."], "trust-b")

    assert state.active().get("substrate") is None, (
        f"a denied substrate is still believed: {state.active()}")
    assert assistant.conversation_state("trust-b").active().get(
        "substrate") is None
    history = state.facts["substrate"]
    assert history.current.status is FactStatus.SUPERSEDED
    assert [fact.value for fact in history.superseded] == ["brick"], (
        "the retracted value must stay auditable, not vanish")
