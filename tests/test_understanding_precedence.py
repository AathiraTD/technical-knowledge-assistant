"""Who wins which field: the precedence rule, as a rule.

`assistant/understanding.py` combines two readings of the same sentence — what
the vocabularies detected and what a local model proposed — and the question
"may the model change this?" has to have one answer per field that somebody can
read off a page.

It was not a rule first, and the omission had a measured cost. The live model
read "What plaster should I use?" as something other than a selection, so the
requirement gate stopped running and the ask-back stopped pausing the graph. A
fix for that one field would have left every other field undecided, which is why
`merge_understanding` states the order instead:

    1. a hard policy decision
    2. confident deterministic intent and entity detection
    3. validated model additions and refinements
    4. unknown

The asymmetry underneath it is the reason it is safe. The model may only *add*:
supply an objective, propose candidates, promote an `UNKNOWN`, resolve a phrasing
the vocabularies missed. Everything it may not do — downgrade a confident intent,
remove a detected product, replace a parsed measurement, reopen a policy route —
*removes* a control rather than adding a capability. A model that can only add
has a worst case of a wasted call.

No network: the model's half is constructed directly, which is the only way to
test a precedence rule against readings that genuinely disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import ollama                                       # noqa: E402
from assistant.answering import understanding as u
from assistant.answering.router import (  # noqa: E402
    PolicyGate,
    SlotDetector,
)

REGISTRY = ["Ultra", "Lime Green Ultra", "Duro", "Solo"]


@pytest.fixture
def detector():
    return SlotDetector()


@pytest.fixture
def gate():
    return PolicyGate()


def model_says(**fields) -> u.TurnUnderstanding:
    fields.setdefault("source", "model")
    return u.TurnUnderstanding(**fields)


# ------------------------------------------------ rank 1: a policy decision


@pytest.mark.parametrize("question,topic", [
    ("How much does Solo cost?", "price"),
    ("Where can I buy it?", "where_to_buy"),
    ("Is it in stock?", "stock"),
])
def test_a_policy_question_is_decided_before_anything_else(detector, gate,
                                                           question, topic):
    reading = u.deterministic(question, detector, gate)

    assert reading.policy_topic == topic
    assert reading.intent is u.Intent.ESCALATE


def test_a_policy_question_never_reaches_the_model(detector, gate, monkeypatch):
    """Not "the model is overruled" -- it is not asked.

    Pricing has no slot vocabulary, so before the gate was consulted here it
    read as `UNKNOWN`, which was the trigger for a model call. A question with a
    fixed referral written once and reviewed once was paying a cold generation
    on the fastest path in the system, ahead of the gate that owns it.
    """
    def forbidden(*_a, **_k):
        raise AssertionError("the model was called on a policy question")

    monkeypatch.setattr(ollama, "generate", forbidden)

    reading = u.understand("How much does Solo cost?", detector, gate=gate)

    assert reading.policy_topic == "price"


def test_a_policy_route_survives_a_model_that_disagrees(detector, gate):
    """Even given a reading, the gate's decision is returned untouched."""
    base = u.deterministic("How much does Solo cost?", detector, gate)

    merged = u.merge_understanding(
        base, model_says(intent=u.Intent.SELECT, objective="insulation",
                         candidate_products=("Ultra",)))

    assert merged.policy_topic == "price"
    assert merged.intent is u.Intent.ESCALATE
    assert merged.objective == "", "a model enriched a route it does not own"
    assert merged.candidate_products == ()


# ------------------------------- rank 2 over rank 3: no downgrade, ever


@pytest.mark.parametrize("question,expected", [
    ("What plaster should I use?", u.Intent.SELECT),
    ("My render is crazing and blowing", u.Intent.TROUBLESHOOT),
    ("How many bags for 20 square metres?", u.Intent.CALCULATE),
])
def test_a_confident_intent_is_not_downgraded(detector, gate, question, expected):
    """The general form of the measured failure, across every confident intent."""
    base = u.deterministic(question, detector, gate)
    assert base.confident

    for proposed in (u.Intent.LOOKUP, u.Intent.UNKNOWN, u.Intent.FIND,
                     u.Intent.UNDERSTAND, u.Intent.ESCALATE):
        merged = u.merge_understanding(base, model_says(intent=proposed))
        assert merged.intent is expected, (
            f"the model changed {expected.value} to {proposed.value}")


def test_a_detected_product_cannot_be_removed(detector, gate):
    question = "Would Lime Green Ultra be suitable internally?"
    base = u.deterministic(question, detector, gate)
    merged = u.merge_understanding(base, model_says(intent=u.Intent.LOOKUP,
                                                    explicit_product=""))

    resolved = u.resolve(merged, question, detector, REGISTRY)

    assert resolved.product == "ultra"


def test_a_parsed_measurement_cannot_be_replaced(detector, gate):
    """Decision 2 keeps arithmetic away from the model; so are its inputs.

    A model that could edit a measurement could change the size of somebody's
    job without changing a single word of the answer around it.
    """
    base = u.deterministic("how much for 30 m2 at 25mm", detector, gate)

    merged = u.merge_understanding(
        base, model_says(intent=u.Intent.CALCULATE,
                         measurements={"area_m2": 999, "thickness_mm": 1}))

    assert merged.measurements == {"area_m2": 30.0, "thickness_mm": 25.0}


# ------------------------------------------- rank 3: what the model may add


def test_an_unknown_intent_may_be_promoted(detector, gate):
    """The coverage the model is here for.

    A request phrased in a way the vocabularies do not recognise is the one case
    where the model's reading is better than what we already had.
    """
    base = u.deterministic("I need something for the front of the house",
                           detector, gate)
    assert not base.confident

    merged = u.merge_understanding(base, model_says(intent=u.Intent.SELECT))

    assert merged.intent is u.Intent.SELECT


@pytest.mark.parametrize("field,value", [
    ("objective", "insulation"),
    ("candidate_products", ("Ultra",)),
    ("requested_properties", ("thickness",)),
    ("new_subject", True),
])
def test_the_model_may_fill_what_the_vocabularies_left_empty(detector, gate,
                                                             field, value):
    base = u.deterministic("What plaster should I use?", detector, gate)
    assert not getattr(base, field), f"{field} was not empty to begin with"

    merged = u.merge_understanding(base, model_says(intent=u.Intent.SELECT,
                                                    **{field: value}))

    assert getattr(merged, field) == value


def test_a_deterministic_value_is_preferred_where_both_have_one(detector, gate):
    """Additive means "where there is a gap", not "wherever it likes"."""
    base = u.deterministic("What plaster should I use on my brick wall?",
                           detector, gate)
    assert base.substrate == "brick"

    merged = u.merge_understanding(base, model_says(intent=u.Intent.SELECT,
                                                    substrate="cob"))

    assert merged.substrate == "brick"


# ------------------------------------------------------------ rank 4: unknown


def test_unknown_survives_when_neither_reading_has_anything(detector, gate):
    base = u.deterministic("hello there", detector, gate)

    merged = u.merge_understanding(base, model_says(intent=u.Intent.UNKNOWN))

    assert merged.intent is u.Intent.UNKNOWN


# ------------------------------------------------- the rule, end to end


def test_the_merged_reading_records_that_the_model_ran(detector, gate):
    """A trace has to be able to say which readings produced this."""
    base = u.deterministic("What plaster should I use?", detector, gate)

    merged = u.merge_understanding(base, model_says(intent=u.Intent.LOOKUP))

    assert merged.source == "model"


def test_an_unavailable_model_leaves_the_deterministic_reading_intact(
        detector, gate, monkeypatch):
    def boom(*_a, **_k):
        raise ollama.OllamaUnavailable("connection refused")

    monkeypatch.setattr(ollama, "generate", boom)

    reading = u.understand("What plaster should I use?", detector, gate=gate)

    assert reading.intent is u.Intent.SELECT
    assert reading.source == "deterministic"
