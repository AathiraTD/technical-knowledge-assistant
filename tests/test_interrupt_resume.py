"""The ask-back as a paused graph, not a reconstructed one.

The old shape worked and was fragile in a specific way. The assistant asked a
question, stored the original in a `pending` string, and on the next message
guessed whether that message was an answer — a guess that got "Can I use Ultra
on the same wall?" wrong and silently discarded it. The question the person
actually asked was reconstructed from a string, which meant everything about
the half-finished turn that was not in that string was simply lost.

`interrupt()` keeps the turn instead of describing it. The graph stops inside
`ask_back`, the checkpoint holds the position and the whole state under the
conversation's `thread_id`, and the next message resumes from that point —
through retrieval, evidence sufficiency and recommendation — answering what was
originally asked rather than re-deriving it.

This could not be wired until the checkpointer outlived a turn, which is why it
sat unreachable behind `interruptible=False` until P0-3. These tests exercise
the real pause and the real resume through `Assistant.ask_turn`, not the node
functions in isolation: the bug being guarded against lives in the handover
between two HTTP requests, and a test that called `ask_back()` directly would
never touch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import ollama                                       # noqa: E402
from assistant.conversation import TurnInput                       # noqa: E402
from assistant.engine import Assistant                             # noqa: E402
from assistant.router import Path_                                 # noqa: E402

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


def turn(question, index=1, session="s1"):
    t = TurnInput(raw_question=question, turn_index=index, session_id=session)
    object.__setattr__(t, "history", "")
    return t


ASK = "What product should I use on my wall?"


def answer_of(reply):
    return reply.parts[0][1]


def trace_of(reply):
    return " ".join(answer_of(reply).diagnostics.get("graph") or [])


# ------------------------------------------------------------- the pause


def test_a_missing_fact_pauses_the_graph(assistant):
    reply, _ = assistant.ask_turn(turn(ASK, 1))

    assert answer_of(reply).diagnostics["interrupted"] is True
    assert answer_of(reply).path == Path_.ASK_BACK.value


def test_the_pause_is_visible_in_the_checkpoint(assistant):
    """The conversation is parked, not finished. The graph is the record of it."""
    assistant.ask_turn(turn(ASK, 1))

    app = assistant._graph
    snapshot = app.get_state({"configurable": {"thread_id": "s1"}})

    assert snapshot.next, "the graph did not stop anywhere"
    assert "ask_back" in snapshot.next


def test_the_paused_turn_still_renders_something_to_show(assistant):
    """A paused graph produces no answer, and an HTTP response needs one.

    The rendering uses the same `need_more_information` path a non-interruptible
    run would take, so the wording and the contact line are identical either way
    and only the machinery differs.
    """
    reply, _ = assistant.ask_turn(turn(ASK, 1))

    text = answer_of(reply).text
    assert "built of" in text
    assert "0800" in text, "the hand-off lost its contact line"


def test_a_pause_does_not_finish_the_turn(assistant):
    reply, _ = assistant.ask_turn(turn(ASK, 1))

    assert "retrieve" not in trace_of(reply)
    assert "assess" not in trace_of(reply)


# ------------------------------------------------------------ the resume


def test_the_next_message_resumes_rather_than_starting_a_new_turn(assistant):
    assistant.ask_turn(turn(ASK, 1))

    reply, _ = assistant.ask_turn(turn("brick", 2))

    assert "ask_back:resumed" in trace_of(reply)


def test_resuming_continues_the_original_request_to_the_end(assistant):
    """Through retrieval and the evidence gate, not back to the start.

    This is the property the `pending` string could never have: the turn picks
    up where it stopped, with everything it had already worked out still in
    place.
    """
    assistant.ask_turn(turn(ASK, 1))

    reply, _ = assistant.ask_turn(turn("brick", 2))

    trace = trace_of(reply)
    assert "retrieve" in trace
    assert "assess" in trace


def test_the_supplied_fact_is_recorded_as_the_persons_own(assistant):
    assistant.ask_turn(turn(ASK, 1))

    _reply, state = assistant.ask_turn(turn("brick", 2))

    assert state.active()["substrate"] == "brick"
    assert state.facts["substrate"].current.provenance.value == "stated"


def test_the_conversation_is_no_longer_paused_afterwards(assistant):
    assistant.ask_turn(turn(ASK, 1))
    assistant.ask_turn(turn("brick", 2))

    snapshot = assistant._graph.get_state({"configurable": {"thread_id": "s1"}})

    assert not snapshot.next, "the graph is still parked after being answered"


def test_a_third_turn_is_an_ordinary_turn_again(assistant):
    """An answered ask-back must not keep swallowing later messages."""
    assistant.ask_turn(turn(ASK, 1))
    assistant.ask_turn(turn("brick", 2))

    reply, _ = assistant.ask_turn(turn("What thickness should it be?", 3))

    assert "ask_back:resumed" not in trace_of(reply)
    assert answer_of(reply).diagnostics.get("interrupted") is not True


def test_the_same_question_asked_twice_does_not_pause_the_second_time(assistant):
    """Once the substrate is known, the ask-back has nothing left to ask."""
    assistant.ask_turn(turn(ASK, 1))
    assistant.ask_turn(turn("brick", 2))

    reply, _ = assistant.ask_turn(turn(ASK, 3))

    assert answer_of(reply).diagnostics.get("interrupted") is not True


# ------------------------------------------------------------- isolation


def test_a_pause_on_one_conversation_does_not_pause_another(assistant):
    """Two people, one process. A parked graph belongs to one thread."""
    assistant.ask_turn(turn(ASK, 1, session="paused"))

    reply, _ = assistant.ask_turn(
        turn("How much water does Solo need per bag?", 1, session="other"))

    assert answer_of(reply).diagnostics.get("interrupted") is not True
    assert "ask_back" not in trace_of(reply)


def test_answering_on_the_wrong_thread_does_not_resume_the_paused_one(assistant):
    assistant.ask_turn(turn(ASK, 1, session="paused"))

    assistant.ask_turn(turn("brick", 1, session="other"))

    snapshot = assistant._graph.get_state({"configurable": {"thread_id": "paused"}})
    assert snapshot.next, "another conversation's reply resumed this one"


def test_forgetting_a_paused_conversation_clears_the_pause(assistant):
    """Somebody who abandons an ask-back must not park a graph forever."""
    assistant.ask_turn(turn(ASK, 1))

    assistant.forget("s1")

    snapshot = assistant._graph.get_state({"configurable": {"thread_id": "s1"}})
    assert not snapshot.next


# -------------------------------------------- the non-interruptible shape


def test_a_caller_with_no_way_to_resume_gets_the_question_rendered_instead(assistant):
    """`interruptible=False` still exists, for a one-shot evaluation of a turn.

    It must produce the same *answer*; only the machinery differs. A caller that
    cannot resume should not get a different hand-off from one that can.
    """
    from assistant.graph import Services, build

    services = Services(retriever=assistant.retriever, router=assistant.router,
                        engine=assistant, repo=assistant.repo,
                        registry=["Solo", "Duro"], interruptible=False,
                        checkpointer=assistant.checkpointer)
    app = build(services)

    with assistant.repo.read_snapshot() as snapshot:
        assistant.engine.names = {
            k: snapshot.notes.get(k, d) for k, d in
            (("products", []), ("colours", []), ("merchants", []), ("contact", {}))}
        final = app.invoke(
            {"raw_question": ASK, "turn_index": 1, "audiences": ("public",),
             "session_id": "oneshot", "images": [], "history": ""},
            {"configurable": {"thread_id": "oneshot"}})

    assert not final.get("__interrupt__"), "it paused despite being told not to"
    assert final["answer"].diagnostics["outcome"] == "need_more_information"
    assert "built of" in final["answer"].text
