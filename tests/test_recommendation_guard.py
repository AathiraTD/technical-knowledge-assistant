"""A recommendation cannot escape by being misfiled.

The evidence gate is structurally enforced once a request is classified
``SELECT``: `assess_evidence` has three outgoing edges and none of them
composes. That is a real guarantee and it has a real limit, which the checkpoint
review named — it begins *after* classification, and classification is the one
stage in this system that involves a model reading a sentence.

So a selection misread as a lookup goes to `delegate`, composes over whatever
retrieval returned, and can say "use Ultra" having met no evidence assessment at
all. The route is correct for a lookup and the safety property is absent.

This file tests the guard that closes it: an intent-independent check on the
finished text. Whatever route produced the answer, if it *recommends* a product,
that product must have been approved by an evidence assessment in this turn.
The point is defence in depth — intent classification stops being load-bearing
for safety, and the worst a misclassification produces is a refusal.

The tests force the misclassification rather than hoping for one. A stubbed
understanding stage returns ``LOOKUP`` for a question that is plainly a
selection, which is exactly the failure being guarded against and is not
otherwise reachable on demand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import candidates as cand                           # noqa: E402
from assistant import ollama                                       # noqa: E402
from assistant import understanding as und                         # noqa: E402
from assistant.conversation import TurnInput                       # noqa: E402
from assistant.engine import Assistant                             # noqa: E402

from test_engine import build_repo, unit                           # noqa: E402

REGISTRY = ["Ultra", "Lime Green Ultra", "Duro", "Solo"]


def recommending_model(text):
    """A stub that returns exactly this text, whatever the prompt."""
    def generate(prompt, **_kwargs):
        return text, 0.01
    return generate


def recommending_but_well_cited(product: str):
    """A recommendation built out of the passage it cites, so the checks pass.

    This matters more than it looks. A stub returning "I would recommend Ultra
    for this wall [1]" is refused by **check 1** long before the guard under
    test sees it -- the clause shares almost no words with the passage it
    cites. A test written that way would pass while proving nothing about the
    guard, which is the trap the checkpoint review set out: a test only proves
    what it actually exercises.

    So this reads the passages out of the prompt the way the model is asked to,
    quotes one verbatim, and prefixes it with a recommendation. Every one of the
    six checks passes -- the figures are in the cited passage, attached to the
    right product, with a real name -- and the only thing standing between it
    and the page is the guard.
    """
    def generate(prompt, **_kwargs):
        for block in prompt.split("Passages:", 1)[-1].split("\n\n"):
            head, _, body = block.strip().partition("\n")
            head, body = head.strip(), body.strip()
            if not (head.startswith("[") and "]" in head and body):
                continue
            if product.lower() not in f"{head} {body}".lower():
                continue
            marker = head[:head.index("]") + 1]
            sentence = body.split(". ")[0].rstrip(".")
            # "You should use Duro, which covers 2.5 m2 per 25 kg bag [2]."
            lowered = sentence[0].lower() + sentence[1:]
            return f"You should use {lowered} {marker}.", 0.01
        return "Nothing was supplied.", 0.01
    return generate


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo, cache=False, log=False)
    finally:
        repo.close()


def turn(question, index=1, session="s1"):
    t = TurnInput(raw_question=question, turn_index=index, session_id=session)
    object.__setattr__(t, "history", "")
    return t


def misclassify_as(monkeypatch, intent):
    """Force the understanding stage to get the intent wrong."""
    real = und.deterministic

    def wrong(question, detector, gate=None, registry=()):
        reading = real(question, detector, gate, registry)
        if reading.policy_topic:
            # A policy route is rank 1 and is not something a misclassification
            # can reach past, so leave it exactly as the gate decided.
            return reading
        return und.TurnUnderstanding(
            intent=intent, substrate=reading.substrate,
            location=reading.location,
            requested_properties=reading.requested_properties,
            measurements=reading.measurements, source="deterministic")

    monkeypatch.setattr(und, "deterministic", wrong)


SELECTION = "What product should I use on my internal brick wall?"


# ------------------------------------------------------- the guard in isolation


@pytest.mark.parametrize("sentence,expected", [
    ("I would recommend Ultra for this wall [1].", ["ultra"]),
    ("You should use Duro on brick [2].", ["duro"]),
    ("Ultra is the best option here [1].", ["ultra"]),
    ("Go for Solo on this background [1].", ["solo"]),
    # The brand-prefixed form normalises to the same product, which is the
    # whole reason this returns a canonical spelling: comparing "Lime Green
    # Ultra" against an approved "ultra" as plain strings refused a legitimate
    # answer -- evaluation situation S8 and conversation C5.
    ("Lime Green Ultra is suitable for brick [1].", ["ultra"]),
])
def test_a_recommendation_is_recognised_in_one_canonical_spelling(sentence, expected):
    assert cand.recommends_a_product(sentence, REGISTRY) == expected


@pytest.mark.parametrize("sentence", [
    "Ultra covers 1.5 m2 per bag at 10mm [1].",
    "Add between 5 and 6 litres of clean water per 25kg sack of Solo [1].",
    "Duro is a general purpose lime undercoat plaster [1].",
    "Apply Ultra at between 10 and 30mm thickness [2].",
])
def test_an_ordinary_lookup_that_names_a_product_is_not_a_recommendation(sentence):
    """The guard must not refuse every answer that mentions what it was asked about.

    "Ultra covers 1.5 m2 per bag" states a published figure. "Use Ultra" makes a
    commercial recommendation the company stands behind. Failing to separate
    those would turn this guard into a blanket refusal of the datasheet
    questions the system exists to answer.
    """
    assert cand.recommends_a_product(sentence, REGISTRY) == []


def test_a_product_outside_the_registry_is_not_reported():
    """Check 5 owns invented names; this owns unapproved real ones."""
    assert cand.recommends_a_product(
        "I would recommend Lime Green Supreme [1].", REGISTRY) == []


# --------------------------------------------- the adversarial misclassification


def test_the_stub_would_otherwise_print(assistant, monkeypatch):
    """The control. Without it, the tests below could pass for the wrong reason.

    This proves the recommendation the guard is about to refuse is one that
    would genuinely have reached the page: it passes all six checks. A stub that
    check 1 was quietly rejecting would make every assertion below vacuous.
    """
    monkeypatch.setattr(ollama, "generate", recommending_but_well_cited("Duro"))

    # Duro is named in the question, so the guard exempts it: answering about
    # the product somebody asked about is answering their question, not the
    # system choosing one for them. What is left standing between this text and
    # the page is the six checks, and they pass it.
    reply, _ = assistant.ask_turn(turn("What coverage does Duro give?"))
    answer = reply.parts[0][1]

    assert not answer.refused, answer.failed_checks or answer.text[:200]
    assert "You should use" in answer.text, answer.text[:200]
    assert not answer.failed_checks


def test_a_misclassified_selection_cannot_recommend_unapproved(assistant, monkeypatch):
    """The whole point of the file.

    A plain selection is forced to read as ``LOOKUP``, so it never reaches the
    evidence gate and composes instead. The model then recommends a product, in
    words the six checks accept. Without the guard this prints.
    """
    misclassify_as(monkeypatch, und.Intent.LOOKUP)
    monkeypatch.setattr(ollama, "generate", recommending_but_well_cited("Duro"))

    reply, _ = assistant.ask_turn(turn(SELECTION))
    answer = reply.parts[0][1]

    assert answer.refused, (
        "a misclassified selection recommended a product with no evidence "
        f"assessment behind it: {answer.text[:200]}")


def test_the_guard_is_what_refused_and_not_the_six_checks(assistant, monkeypatch):
    """Which mechanism stopped it, asserted rather than assumed.

    Both produce a refusal, and they mean different things: a failed check means
    the prose did not match its evidence, while this means the product was never
    eligible. Reading one as the other would hide the guard being broken.
    """
    misclassify_as(monkeypatch, und.Intent.LOOKUP)
    monkeypatch.setattr(ollama, "generate", recommending_but_well_cited("Duro"))

    reply, _ = assistant.ask_turn(turn(SELECTION))
    answer = reply.parts[0][1]

    assert not answer.failed_checks, (
        f"the six checks refused it first: {answer.failed_checks}")
    assert "duro" in answer.text.lower()
    assert "evidence assessment" in answer.text


@pytest.mark.parametrize("intent", [und.Intent.LOOKUP, und.Intent.UNDERSTAND,
                                    und.Intent.VERIFY, und.Intent.UNKNOWN])
def test_no_intent_label_is_a_way_through(assistant, monkeypatch, intent):
    """The guard is independent of the label, so every label is tried."""
    misclassify_as(monkeypatch, intent)
    monkeypatch.setattr(ollama, "generate", recommending_but_well_cited("Duro"))

    reply, _ = assistant.ask_turn(turn(SELECTION))

    assert reply.parts[0][1].refused, f"{intent.value} was a way through"


def test_the_failure_is_recorded_as_a_failed_check(assistant, monkeypatch, caplog):
    """An over-refusal has to be countable, so the reason is on the record."""
    import logging

    misclassify_as(monkeypatch, und.Intent.LOOKUP)
    monkeypatch.setattr(ollama, "generate", recommending_but_well_cited("Duro"))

    with caplog.at_level(logging.INFO):
        assistant.ask_turn(turn(SELECTION))

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "check_failed" in messages or "unapproved_recommendation" in messages


# ---------------------------------------------------------- no false refusals


def test_an_ordinary_lookup_still_answers(assistant, monkeypatch):
    """The cost of the guard, bounded.

    A question that asks for a published figure and gets one must print, or the
    guard has traded a rare unsafe answer for a common useless one.
    """
    monkeypatch.setattr(ollama, "generate", recommending_model(
        "Add between 5 and 6 litres of clean water per 25kg sack [1]."))

    reply, _ = assistant.ask_turn(
        turn("How much water does Solo need per bag?"))

    assert not reply.parts[0][1].refused, reply.parts[0][1].text[:200]


def test_a_policy_routed_question_is_untouched(assistant, monkeypatch):
    """The referral names no product and must not be caught."""
    monkeypatch.setattr(ollama, "generate", recommending_model("unused"))

    reply, _ = assistant.ask_turn(turn("How much does Solo cost?"))

    assert reply.parts[0][1].path == "route"
    assert not reply.parts[0][1].refused


def test_a_product_the_caller_named_is_not_one_the_system_introduced(assistant,
                                                                    monkeypatch):
    """The exemption, tested for what it is rather than assumed.

    A VERIFY question names its product. Answering it is not a recommendation
    the system originated, and refusing it would be an over-refusal on the
    commonest shape of question the corpus can actually answer.
    """
    monkeypatch.setattr(ollama, "generate", recommending_but_well_cited("Duro"))

    reply, _ = assistant.ask_turn(turn("Would Duro work on my brick wall?"))

    assert not reply.parts[0][1].refused


def test_the_exemption_does_not_cover_a_different_product(assistant, monkeypatch):
    """Naming one product does not license recommending another.

    Somebody asks about Solo; the model answers "you should use Duro". Duro was
    never asked about and never assessed, so it is exactly what the guard is
    for.
    """
    misclassify_as(monkeypatch, und.Intent.LOOKUP)
    monkeypatch.setattr(ollama, "generate", recommending_but_well_cited("Duro"))

    reply, _ = assistant.ask_turn(
        turn("How much water does Solo need per bag?"))

    assert reply.parts[0][1].refused, (
        "a product nobody asked about was recommended without assessment")

