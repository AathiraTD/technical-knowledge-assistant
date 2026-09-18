"""Where the phrasing stage is wired in: `Assistant._polish`, inside `ask_turn`.

`tests/test_phrasing.py` proves what the stage does with an answer. It says
nothing about the join, and the join is where this could go wrong in the ways
that matter: polishing something unfinished, polishing twice, polishing when
switched off, or claiming in the diagnostics to have polished when it did not.

One test, because the four properties are one flow and separating them would
mean running the same conversation four times to assert quarters of it. The
method is to run the *same question twice* against the same corpus -- once with
the stage off, once with it on -- and compare. That is what makes the central
assertion possible: the text handed to the editor must be exactly the answer the
unpolished run produced, which is the only direct evidence that the stage sees a
*completed* answer rather than an intermediate one.

The composition model is stubbed, as everywhere else in this repository's
offline tests. What is not stubbed is the engine, the graph, the router, the
retriever or the store: the call site is exercised through `ask_turn`, because a
unit test of `_polish` would re-test `phrasing` and prove nothing about the wire.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answering import phrasing
from assistant.infrastructure import ollama
from assistant.turn.conversation import ConversationState, TurnInput     # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)

from test_engine import build_repo, quoting, unit                  # noqa: E402

QUESTION = "How much water does Solo need per bag?"
EDITOR_MARKER = "Rewritten answer:"


def _turn(session: str) -> TurnInput:
    turn = TurnInput(raw_question=QUESTION, turn_index=1, session_id=session)
    object.__setattr__(turn, "history", "")
    return turn


def _run(repo, calls: list, *, polish: bool, monkeypatch):
    """One turn through the real `ask_turn`, recording every model prompt.

    The editor's reply is the verified answer with a direct opener bolted on.
    Deliberately derived from the input rather than written out: the point of
    this test is the wiring, and a hand-written reply would have to be kept in
    step with whatever the composition stub happens to produce.
    """
    def generate(prompt, **kwargs):
        calls.append(prompt)
        if EDITOR_MARKER in prompt:
            body = prompt.split("Verified answer:\n", 1)[1].split(
                f"\n\n{EDITOR_MARKER}")[0]
            return f"Yes. {body.strip()}", 0.25
        return quoting(prompt, **kwargs)

    monkeypatch.setattr(ollama, "generate", generate)
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    if polish:
        monkeypatch.setenv(phrasing.PHRASING_FLAG, "1")
    else:
        monkeypatch.delenv(phrasing.PHRASING_FLAG, raising=False)

    assistant = Assistant(repo, cache=False, log=False)
    reply, _state = assistant.ask_turn(_turn("polish" if polish else "plain"),
                                       ConversationState())
    assert len(reply.parts) == 1, "this question was expected to be one part"
    return reply.parts[0][1]


def test_ask_turn_polishes_the_completed_answer_only_when_switched_on(
        tmp_path, monkeypatch):
    repo = build_repo(tmp_path, two_documents=True)
    try:
        # -- switched off: the default, and it must be a true no-op ---------
        plain_calls: list = []
        plain = _run(repo, plain_calls, polish=False, monkeypatch=monkeypatch)

        assert not any(EDITOR_MARKER in p for p in plain_calls), (
            "the editor was called with the stage switched off")
        assert "phrasing" not in plain.diagnostics, (
            "a skipped stage left a diagnostic claiming otherwise")

        # -- switched on ----------------------------------------------------
        polished_calls: list = []
        polished = _run(repo, polished_calls, polish=True,
                        monkeypatch=monkeypatch)

        editor_prompts = [p for p in polished_calls if EDITOR_MARKER in p]
        assert len(editor_prompts) == 1, (
            f"the editor ran {len(editor_prompts)} times for one answer")

        # Called last, after the answer path finished. Asserted on ordering
        # rather than on a flag, because a flag would only record what the code
        # believes about itself.
        #
        # Not asserted: that something composed first. This question takes
        # Extract, which decision 2 prints by code with no model call at all, so
        # on this path the editor is legitimately the only prompt sent. The
        # ordering claim is carried by the comparison below instead, which is
        # the stronger evidence anyway.
        assert EDITOR_MARKER in polished_calls[-1]

        # The central property: what the editor received is exactly what the
        # unpolished run produced. Anything earlier in the pipeline -- a
        # pre-caveat draft, a passage, a partial compose -- would differ here.
        handed_over = editor_prompts[0].split("Verified answer:\n", 1)[1].split(
            f"\n\n{EDITOR_MARKER}")[0]
        assert handed_over == plain.body, (
            "the stage was handed something other than the finished answer")
        assert QUESTION in editor_prompts[0]

        # -- and the result reaches the caller, recorded honestly ----------
        assert polished.diagnostics["phrasing"] == "applied"
        assert polished.text == f"Yes. {plain.body.strip()}"
        assert polished.path == plain.path, "the polish changed the route"
        assert polished.refused == plain.refused
        # Everything the surface reads beside the prose survives the swap.
        assert polished.sources == plain.sources
        assert polished.caveats == plain.caveats
        assert polished.diagnostics["correlation_id"], (
            "the reply-assembly diagnostics were lost by the swap")
    finally:
        repo.close()
