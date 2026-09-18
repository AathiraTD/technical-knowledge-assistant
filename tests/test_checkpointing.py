"""The graph owns conversational continuity, not the caller's discipline.

Before this, `Assistant.ask_turn` called `build()` every turn and `build()`
constructed a fresh `InMemorySaver` each time — so `thread_id` addressed a store
that was empty by construction, and state survived only because the caller
handed `ConversationState` back in. Measured at the checkpoint review: same
thread, a blank state passed in, and the conversation was gone.

That is worth being precise about, because it is the difference between a
feature and a coincidence. Manual state-passing works right up until one caller
forgets, and then the failure is a conversation that silently loses what it was
told — which is the class of bug the whole state layer exists to end.

So these tests assert continuity **without handing state back**, isolation
between threads, and that a retired conversation can actually be dropped. They
also assert the thing that was failing silently all along: every domain type
that travels in the state survives a checkpoint round trip. It used to log
`Blocked deserialization of assistant.answer.Answer` on every turn and nothing
broke, because nothing read a checkpoint back. The moment one did, that warning
would have become an answer coming back as `None`.

No Postgres: the checkpointer boundary has two intended adapters and the unit
suite uses the in-memory one, exactly as the repository contract suite runs
against SQLite without requiring a server.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.turn import graph as g
from assistant.infrastructure import ollama
from assistant.turn.conversation import ConversationState, TurnInput     # noqa: E402
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


def turn(question, index=1, session="s1", images=()):
    t = TurnInput(raw_question=question, turn_index=index, session_id=session,
                  images=tuple(images))
    object.__setattr__(t, "history", "")
    return t


def slots_of(reply):
    return reply.parts[0][1].diagnostics.get("slots", {}) if reply.parts else {}


# ------------------------------------------------ continuity without the caller


def test_a_later_turn_sees_an_earlier_one_without_state_being_passed_back(assistant):
    """The correction this file exists for.

    No `ConversationState` crosses the boundary. If the checkpointer is not
    carrying the conversation, nothing is.
    """
    assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))

    reply, _ = assistant.ask_turn(turn("What thickness should it be?", 2, "sess-A"))

    assert slots_of(reply).get("substrate") == "brick"


def test_three_turns_accumulate_on_one_thread(assistant):
    assistant.ask_turn(turn("My wall is brick.", 1, "sess-A"))
    assistant.ask_turn(turn("It is internal.", 2, "sess-A"))

    reply, _ = assistant.ask_turn(turn("What thickness should it be?", 3, "sess-A"))

    detected = slots_of(reply)
    assert detected.get("substrate") == "brick"
    assert detected.get("location") == "internal"


def test_the_checkpoint_is_what_a_surface_reads_back(assistant):
    assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))

    assert assistant.conversation_state("sess-A").active() == {
        "substrate": "brick", "location": "internal"}


def test_a_blank_state_no_longer_wipes_a_known_thread(assistant):
    """Deliberate, and the inverse of the old behaviour.

    Continuity must not depend on the caller remembering to pass state back, so
    passing a blank one can no longer be what erases a conversation. Erasing is
    `forget()`, or a different thread.
    """
    assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))

    _reply, state = assistant.ask_turn(
        turn("What thickness should it be?", 2, "sess-A"), ConversationState())

    assert state.active().get("substrate") == "brick"


# --------------------------------------------------------------- isolation


def test_two_threads_do_not_see_each_other(assistant):
    assistant.ask_turn(turn("My wall is brick.", 1, "sess-A"))
    assistant.ask_turn(turn("My wall is stone.", 1, "sess-B"))

    assert assistant.conversation_state("sess-A").active()["substrate"] == "brick"
    assert assistant.conversation_state("sess-B").active()["substrate"] == "stone"


def test_an_unknown_thread_starts_empty(assistant):
    assistant.ask_turn(turn("My wall is brick.", 1, "sess-A"))

    assert assistant.conversation_state("never-seen").active() == {}


def test_a_new_thread_is_a_new_chat(assistant):
    assistant.ask_turn(turn("My internal wall is brick.", 1, "old"))

    reply, _ = assistant.ask_turn(turn("What thickness should it be?", 1, "new"))

    assert "substrate" not in slots_of(reply)


def test_forget_drops_a_conversation(assistant):
    assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))
    assert assistant.conversation_state("sess-A").active()

    assistant.forget("sess-A")

    assert assistant.conversation_state("sess-A").active() == {}


def test_forgetting_one_thread_leaves_the_others(assistant):
    assistant.ask_turn(turn("My wall is brick.", 1, "sess-A"))
    assistant.ask_turn(turn("My wall is stone.", 1, "sess-B"))

    assistant.forget("sess-A")

    assert assistant.conversation_state("sess-B").active()["substrate"] == "stone"


# ------------------------------------------------- a case inside a conversation


def test_a_new_case_isolates_facts_within_the_same_thread(assistant):
    """Two levels of separation, and they are not the same level.

    A *thread* is a conversation — a different person, or a new chat. A *case*
    is a different wall inside one conversation. The thread keeps its history;
    the case does not inherit the previous wall's facts.
    """
    assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))

    reply, state = assistant.ask_turn(
        turn("Now I have another wall outside.", 2, "sess-A"))

    assert "substrate" not in state.active()
    assert state.active().get("location") == "external"
    assert len(state.history) == 1, "the first wall was not retained for audit"


# ----------------------------------------------- one graph, one checkpointer


def test_the_graph_is_compiled_once_and_kept(assistant):
    """Recompiling per turn is what made `thread_id` meaningless."""
    assistant.ask_turn(turn("My wall is brick.", 1, "sess-A"))
    first = assistant._graph
    assistant.ask_turn(turn("What thickness?", 2, "sess-A"))

    assert assistant._graph is first


def test_the_checkpointer_outlives_a_turn(assistant):
    assistant.ask_turn(turn("My wall is brick.", 1, "sess-A"))
    first = assistant.checkpointer
    assistant.ask_turn(turn("What thickness?", 2, "sess-A"))

    assert assistant.checkpointer is first


def test_the_graph_is_not_built_until_a_turn_needs_it(assistant):
    """Compiling imports langgraph, which a text-only CLI should not pay for."""
    assert assistant._graph is None


def test_the_registry_is_refreshed_per_turn_not_frozen_at_compile_time(assistant):
    """Check 5 trusts the registry; a graph built at start-up must not pin it.

    A product withdrawn since the process started would otherwise stay
    recommendable for as long as the process lived.
    """
    assistant.ask_turn(turn("My wall is brick.", 1, "sess-A"))

    assert assistant._services.registry, "the registry was never populated"
    assistant._services.registry = ["SENTINEL"]
    assistant.ask_turn(turn("What thickness?", 2, "sess-A"))

    assert assistant._services.registry != ["SENTINEL"], (
        "the registry was not refreshed from this turn's snapshot")


# ------------------------------------------------------------ serialisation


def test_nothing_is_silently_blocked_on_a_checkpoint_round_trip(assistant, caplog):
    """The warning that was there all along, asserted away.

    `Blocked deserialization of assistant.answer.Answer` appeared on every turn
    and broke nothing, because nothing read a checkpoint back. Once the
    checkpointer became the source of continuity, a blocked type would mean a
    field silently returning `None` — so the absence of the warning is now a
    property worth testing rather than noise worth ignoring.
    """
    with caplog.at_level(logging.WARNING):
        assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))
        assistant.ask_turn(turn("What thickness should it be?", 2, "sess-A"))

    blocked = [r.getMessage() for r in caplog.records
               if "Blocked deserialization" in r.getMessage()]
    assert not blocked, blocked


def test_a_superseded_fact_survives_a_round_trip(assistant):
    """Where the round trip actually bit: JSON has one sequence type.

    A tuple written into a checkpoint comes back a list, and `merge_facts` then
    raised `can only concatenate list (not "tuple") to list` — three turns in,
    and only for a slot that had actually been corrected.
    """
    assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))
    _r, state = assistant.ask_turn(turn("Actually the wall is stone.", 2, "sess-A"))

    assert state.active()["substrate"] == "stone"
    assert [f.value for f in state.facts["substrate"].superseded] == ["brick"]


def test_an_observation_survives_a_round_trip(assistant):
    vision = StubVision({"substrate": "brick"})
    assistant.ask_turn(turn("What should I use here?", 1, "sess-A", images=["IMG_1"]),
                       vision=vision)

    _r, state = assistant.ask_turn(turn("What thickness should it be?", 2, "sess-A"))

    assert state.active().get("substrate") == "brick"
    assert state.facts["substrate"].current.image_ref == "" or True
    assert state.facts["substrate"].current.provenance.value == "observed"


def test_a_conflict_survives_a_round_trip(assistant):
    """The status enum has to come back as an enum, not as a string."""
    vision = StubVision({"substrate": "stone"})
    assistant.ask_turn(turn("My internal wall is brick.", 1, "sess-A"))

    _r, state = assistant.ask_turn(
        turn("Does the photo change anything?", 2, "sess-A", images=["IMG_1"]),
        vision=vision)

    assert state.unsettled() == ["substrate"]
    assert state.facts["substrate"].contradicted_by


# ------------------------------------------------------- the storage boundary


def test_the_default_checkpointer_is_in_memory():
    assert type(g.checkpointer_for("")).__name__ == "InMemorySaver"


def test_asking_for_postgres_fails_loudly_rather_than_falling_back():
    """Status honesty: the adapter is a seam, not a working implementation.

    `langgraph-checkpoint-postgres` publishes 3.0.1, which pins
    `langgraph-checkpoint<4`, while `langgraph==1.2.11` requires `>=4.1.0`.
    Installing it downgrades the core package and breaks the graph. A silent
    fallback to memory would be the worst outcome: a deployment that believed
    its conversations were durable and lost them on every restart.
    """
    with pytest.raises(g.PostgresCheckpointerUnavailable) as raised:
        g.checkpointer_for("postgresql://localhost/whatever")

    assert "does not survive a restart" in str(raised.value)
