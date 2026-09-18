"""The transcript reaches the model and nothing else. A regression suite.

`assistant/engine.py` used to open `ask()` with

    if context:
        question = context + "\\n\\n" + question

and everything downstream then read an earlier *answer* as though the person had
typed it. That is not a cosmetic bug. The policy gate, the slot detector, the
product detector and the embedder are the deterministic half of this system —
the half decision 4 exists to keep inspectable — and feeding them generated
prose makes the model the controller through the back door.

Three consequences were reproduced before the fix, and each has a test here:

* a previous answer containing the word "cost" sent an unrelated follow-up to
  the price referral;
* a previous answer mentioning plaster put `substrate=existing_plaster` on a
  question that named no substrate;
* the embedded query became the transcript, so retrieval answered the
  conversation instead of the question.

The positive half matters as much: the transcript still has to *arrive*, or
"does that change your recommendation?" is unanswerable. So the last group
proves it reaches the compose prompt, delimited as non-evidence.

The harness is `tests/test_engine.py`'s — a real SQLite store, no Ollama.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import ollama                                        # noqa: E402
from assistant.cache import AnswerCache                             # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.answering.router import (  # noqa: E402
    Path_,
)

from test_engine import (                                           # noqa: E402
    SOLO, build_repo, quoting, unit,
)

# A previous assistant answer that is entirely ordinary and entirely poisonous
# to a system that reads it as input: it names a price word, a substrate word
# and a second product, none of which the caller ever said.
POISONED_HISTORY = (
    "You asked: How much does Solo cost?\n"
    "You answered: The website does not publish prices. Lime Green does not "
    "sell products online.\n"
    "You asked: What about plaster?\n"
    "You answered: Solo is a one coat lime plaster suitable for existing "
    "plaster and most solid masonry backgrounds. Duro covers 2.5 m2 per bag."
)


@pytest.fixture
def no_ollama(monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)


@pytest.fixture
def assistant(tmp_path, no_ollama):
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo, log=False)
    finally:
        repo.close()


def routes(reply):
    return [answer.path for _part, answer in reply.parts]


def slots(reply):
    merged = {}
    for _part, answer in reply.parts:
        merged.update(answer.diagnostics.get("slots", {}))
    return merged


# ------------------------------------------------- the three reproduced bugs


def test_a_price_word_in_an_earlier_answer_does_not_route_this_question(assistant):
    """The policy gate must read the question, not the conversation.

    "How thick should it be?" is a datasheet question. Answered after a turn
    about price, it used to come back as the price referral -- the caller asking
    a technical question and being told where to find a stockist.
    """
    question = "How thick should it be?"

    clean = assistant.ask(question)
    with_history = assistant.ask(question, context=POISONED_HISTORY)

    assert Path_.ROUTE.value not in routes(with_history), (
        "an earlier answer's word 'cost' reached the policy gate")
    assert routes(with_history) == routes(clean)


def test_a_substrate_in_an_earlier_answer_does_not_become_a_detected_slot(assistant):
    """`detect()` must see the question alone.

    Measured before the fix: `detect("What plaster should I use on my wall")`
    returned `{}`, and `detect(history + question)` returned
    `{'substrate': 'existing_plaster', 'property_asked': 'compatibility'}` --
    a fact about the caller's building, invented by reading the assistant's own
    prose back in.
    """
    question = "What plaster should I use on my wall?"

    clean = slots(assistant.ask(question))
    with_history = slots(assistant.ask(question, context=POISONED_HISTORY))

    assert with_history.get("substrate") == clean.get("substrate")
    assert "existing_plaster" not in with_history.values()


def test_the_embedded_query_is_the_question_not_the_transcript(assistant, monkeypatch):
    """Retrieval must answer what was asked, not what was discussed."""
    embedded: list[str] = []

    def record(text, **_kwargs):
        embedded.append(text)
        return unit(0)

    monkeypatch.setattr(ollama, "embed_one", record)

    assistant.ask("How thick should it be?", context=POISONED_HISTORY)

    assert embedded, "nothing was embedded"
    for text in embedded:
        assert "You answered" not in text, "the transcript was embedded"
        assert "does not publish prices" not in text


def test_history_changes_neither_the_route_nor_the_step(assistant):
    """The strongest form: routing is a pure function of the question.

    Asserted on the router's own step number rather than on the path, because
    two different steps can produce the same path and a drift between them would
    still be the bug coming back.
    """
    for question in ("How thick should it be?",
                     "How much water does Solo need per bag?",
                     "What coverage does Duro give?"):
        clean = assistant.ask(question)
        dirty = assistant.ask(question, context=POISONED_HISTORY)

        steps_clean = [a.diagnostics.get("step") for _p, a in clean.parts]
        steps_dirty = [a.diagnostics.get("step") for _p, a in dirty.parts]
        assert steps_dirty == steps_clean, f"history changed routing for {question!r}"


def test_a_long_transcript_does_not_consume_the_question_word_budget(assistant):
    """`cap()` trims at about 500 words, and it must trim the *question*.

    Concatenating first meant a long conversation pushed the actual question
    past the cap and truncated it away, so the system answered the beginning of
    a transcript. The question is capped alone now, so this cannot happen.
    """
    enormous = "You answered: lime plaster is breathable. " * 400

    reply = assistant.ask("How much water does Solo need per bag?",
                          context=enormous)

    assert "water" in reply.question.lower(), "the question survived the transcript"
    assert "breathable" not in reply.question.lower()


# ------------------------------------------------ the transcript still arrives


def test_the_transcript_reaches_the_compose_prompt(assistant, monkeypatch):
    """It has to arrive, or a follow-up cannot be resolved at all."""
    prompts: list[str] = []

    def capture(prompt, **kwargs):
        prompts.append(prompt)
        return quoting(prompt, **kwargs)

    monkeypatch.setattr(ollama, "generate", capture)

    # A question that actually reaches Compose in this fixture. "Is it
    # suitable for that wall?" refuses at the relevance gate here, because
    # these passages never state suitability -- correct behaviour, and no
    # use for testing what the compose prompt contains.
    assistant.ask("How much water does Solo need per bag?",
                  context=POISONED_HISTORY)

    composed = [p for p in prompts if "Passages:" in p]
    assert composed, "compose never ran"
    assert any("You answered" in p for p in composed), (
        "the model was given no way to resolve 'it'")


def test_the_transcript_is_delimited_as_not_evidence(assistant, monkeypatch):
    """A passage carries a citation marker. The transcript must not look like one.

    The prompt is not the guarantee -- check 1 is -- but a transcript formatted
    like passage `[5]` would invite the model to cite it, and every such answer
    would then be refused. Saying plainly what it is costs nothing and keeps
    the refusal rate honest.
    """
    prompts: list[str] = []
    monkeypatch.setattr(ollama, "generate",
                        lambda p, **k: (prompts.append(p), quoting(p, **k))[1])

    assistant.ask("How much water does Solo need per bag?",
                  context=POISONED_HISTORY)

    composed = next(p for p in prompts if "Passages:" in p)
    before = composed.split("Passages:", 1)[0]
    assert "NOT evidence" in before
    assert "never cite it" in before


def test_no_transcript_means_no_history_block_at_all(assistant, monkeypatch):
    """A first turn must not be given an empty section to reason about."""
    prompts: list[str] = []
    monkeypatch.setattr(ollama, "generate",
                        lambda p, **k: (prompts.append(p), quoting(p, **k))[1])

    assistant.ask("How much water does Solo need per bag?")

    composed = next(p for p in prompts if "Passages:" in p)
    assert "NOT evidence" not in composed


# ------------------------------------------------------------------ the cache


def test_two_conversations_asking_the_same_words_do_not_share_an_answer():
    """"Does that change it?" means nothing without the turns before it.

    Serving one conversation's answer to another is the same class of failure
    as the audience leak the cache key already guards against: a key too loose
    is a claim about somebody that is not true.
    """
    common = ("does that change it?", ("public",), "snap-1", "m", "c1", None, None)

    first = AnswerCache.key(*common, "You answered: use Ultra.")
    second = AnswerCache.key(*common, "You answered: use Duro.")

    assert first != second


def test_the_cache_key_does_not_retain_the_transcript():
    """Keys live for the process's life and are printed in diagnostics.

    The transcript is the one piece of customer text this system is careful not
    to keep where it has not deliberately chosen to, so it goes into the key as
    a digest.
    """
    key = AnswerCache.key("q", ("public",), "snap-1", "m", "c1", None, None,
                          "You answered: the wall is damp near the skirting.")

    assert not any("skirting" in str(part) for part in key)
    assert not any("damp" in str(part) for part in key)


def test_no_transcript_keys_exactly_as_before():
    """Every existing caller passes no history and must keep its entries."""
    common = ("q", ("public",), "snap-1", "m", "c1", None, None)

    assert AnswerCache.key(*common) == AnswerCache.key(*common, "")
