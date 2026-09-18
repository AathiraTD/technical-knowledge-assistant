"""Where model output stops being trusted.

`assistant/answering/understanding.py` is the one place a model is asked to read the
person's question rather than to compose from passages, so it is the one place a
new class of failure can enter: a fluent, well-formed, entirely invented
structure. Constrained decoding guarantees the shape and nothing else — a schema
can say `substrate` is a string, not that the string names a real substrate.

So the tests are almost all adversarial. A stub model returns things a real one
plausibly would — a product that does not exist, a substrate nobody mentioned,
an intent outside the enum, an arithmetic result — and each must be discarded by
deterministic code before it can reach routing or retrieval.

The other half is availability. This stage sits in front of every ambiguous
question, and an unreachable Ollama must cost a better *reading* of the question
and never the answer itself. Decision 5's rule, applied to a new dependency:
failure reduces coverage, not safety.

No network: `ollama.generate` is replaced in every test that reaches it, and
most tests never do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.infrastructure import ollama
from assistant.answering import understanding as u
from assistant.answering.answer import (  # noqa: E402
    Provenance,
)
from assistant.turn.conversation import (
    ConversationState,
    FactStatus,
    SessionFact,
    merge_facts,
)
from assistant.answering.router import (  # noqa: E402
    SlotDetector,
)

# The registry as it is actually harvested: long names and short aliases both.
REGISTRY = ["Ultra: Insulated Lime Plaster Base Coat", "Ultra", "Duro",
            "Solo Onecoat Lime Plaster", "Solo", "Lime Green Ultra",
            "Warmshell Woodfibre Insulation Boards"]


@pytest.fixture
def detector():
    return SlotDetector()


def model_returning(monkeypatch, payload: str):
    """A stubbed Ollama that returns exactly this body."""
    monkeypatch.setattr(ollama, "generate", lambda *_a, **_k: (payload, 0.1))


def unavailable(monkeypatch):
    def boom(*_a, **_k):
        raise ollama.OllamaUnavailable("connection refused")
    monkeypatch.setattr(ollama, "generate", boom)


def forbidden(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("the model was called on a question that did not need it")
    monkeypatch.setattr(ollama, "generate", boom)


# ------------------------------------------------- the deterministic reading


def test_a_plain_lookup_is_understood_without_calling_a_model(detector, monkeypatch):
    """Decision 2 keeps the model off the fast paths, and this keeps it off.

    A cold generation costs tens of seconds on this hardware. Paying that on a
    question the vocabulary has already understood, to understand it the same
    way, is a latency regression bought with nothing.
    """
    forbidden(monkeypatch)

    reading = u.understand("How much water does Solo need per bag?", detector)

    assert reading.source == "deterministic"
    # LOOKUP rather than CALCULATE, and the distinction is the useful one: the
    # sheet prints "5 to 6 litres per 25 kg sack", so this is a published figure
    # to be quoted, not a sum to be done. `CALCULATE` is "how many bags for 20
    # square metres", where router step 6 prints the coverage and refuses the
    # multiplication.
    assert reading.intent is u.Intent.LOOKUP


def test_a_quantity_question_reads_as_calculate(detector):
    """The other side of the LOOKUP/CALCULATE line, so the boundary is pinned."""
    reading = u.deterministic("How many bags for 20 square metres?", detector)

    assert reading.intent is u.Intent.CALCULATE
    assert reading.measurements == {"area_m2": 20.0}


def test_a_product_choice_is_ambiguous_enough_to_be_worth_the_call(detector):
    """SELECT is the intent whose fields decide a recommendation."""
    reading = u.deterministic("What product should I use on my wall?", detector)

    assert reading.intent is u.Intent.SELECT
    assert u.ambiguous(reading, "What product should I use on my wall?")


def test_an_excluded_product_does_not_turn_a_lookup_into_a_selection(detector):
    """Naming a product in order to rule it out is not choosing it.

    The question names three products, so reading the raw sentence found no
    single target and fell through to SELECT -- which sent a question about one
    stated product to the selection gate and had it ask for a substrate rather
    than print two figures the Ultra datasheet publishes.
    """
    question = ("I'm using Lime Green Ultra. What thickness and mixing water "
                "should I use? Please don't give me figures for Solo or Duro.")
    reading = u.deterministic(question, detector, registry=REGISTRY)

    assert reading.intent is u.Intent.VERIFY


@pytest.mark.parametrize("question, intent", [
    # Still a selection: no product is named as the subject.
    ("Which plaster should I use?", "SELECT"),
    ("What product should I use on my wall?", "SELECT"),
    # Still a selection: comparing alternatives names neither as the subject,
    # so two products in the sentence must not read as one chosen product.
    ("Which is best for a brick wall, Solo or Duro?", "SELECT"),
    ("Should I use Solo or Duro on brick?", "SELECT"),
    ("Which plaster should I use, Solo or Duro?", "SELECT"),
    # A named product the question is actually about.
    ("Would Lime Green Ultra be suitable internally?", "VERIFY"),
    ("Is Ultra suitable for my wall?", "VERIFY"),
])
def test_selection_and_verification_stay_separated(detector, question, intent):
    reading = u.deterministic(question, detector, registry=REGISTRY)
    assert reading.intent is getattr(u.Intent, intent), question


def test_a_symptom_reads_as_troubleshoot(detector):
    reading = u.deterministic("My render is crazing and blowing", detector)

    assert reading.intent is u.Intent.TROUBLESHOOT


# ----------------------------------------------------- measurements by code


def test_measurements_are_parsed_by_code_not_taken_from_the_model(detector, monkeypatch):
    """The model may not supply a number that reaches a calculation.

    Decision 2 keeps arithmetic away from the model and router step 6 refuses
    the multiplication outright. A model that returned `area_m2: 750` for a
    question saying 30 would be inventing the job.
    """
    model_returning(monkeypatch, '{"intent": "calculate", "area_m2": 750, '
                                 '"thickness_mm": 999}')

    reading = u.understand("How much for 30 square metres at 25mm?",
                           detector, force=True)

    assert reading.measurements == {"area_m2": 30.0, "thickness_mm": 25.0}


def test_an_absurd_measurement_is_not_carried(detector):
    """A wall is not ten thousand square metres, and a plaster is not a metre thick."""
    assert u.measurements_in("cover 99999 m2 at 900mm") == {}


@pytest.mark.parametrize("text,expected", [
    ("30 square metres", {"area_m2": 30.0}),
    ("30m2", {"area_m2": 30.0}),
    ("30 m²", {"area_m2": 30.0}),
    ("25 mm thick", {"thickness_mm": 25.0}),
    ("no numbers here", {}),
])
def test_measurement_forms_a_person_actually_writes(text, expected):
    assert u.measurements_in(text) == expected


# -------------------------------------------- what the model is not allowed


def test_a_product_the_registry_does_not_hold_is_dropped(detector, monkeypatch):
    """Check 5 refuses an invented name in an answer. There is no reason to
    accept in a query what would be refused in a reply."""
    model_returning(monkeypatch,
                    '{"intent": "select", "explicit_product": "Lime Green Supreme"}')
    reading = u.understand("What should I use?", detector, force=True)

    resolved = u.resolve(reading, "What should I use?", detector, REGISTRY)

    assert resolved.product == ""


def test_a_real_product_is_normalised_the_way_retrieval_needs_it(detector, monkeypatch):
    """Lower case, with the maker's name removed. Both halves are load-bearing.

    Chunks are tagged with the catalogue name -- "Ultra: Insulated Lime Plaster
    Base Coat" -- which neither contains "Lime Green Ultra" nor is contained by
    it. Matching is containment either way, so the brand-prefixed form matches
    no chunk at all and the product boost and the targeted coverage lookup both
    become silent no-ops. "Ultra" finds the coverage passages; "Lime Green
    Ultra" finds none.

    This is `Assistant._named_product`'s normalisation, and it is asserted here
    because reintroducing the longer form was measured: conversation C5 in the
    evaluation harness reported a published coverage figure as unpublished.
    """
    model_returning(monkeypatch,
                    '{"intent": "verify", "explicit_product": "Lime Green Ultra"}')
    question = "Would Lime Green Ultra work?"
    reading = u.understand(question, detector, force=True)

    resolved = u.resolve(reading, question, detector, REGISTRY)

    assert resolved.product == "ultra"


def test_a_substrate_the_question_never_mentioned_is_dropped(detector, monkeypatch):
    """The failure decision 10's ask-back exists to prevent, arriving by model.

    A recommendation made against an invented substrate is the costly error this
    whole design is built around, and it is worse coming from a model than from
    a blank, because it looks like something the person said.
    """
    model_returning(monkeypatch,
                    '{"intent": "select", "substrate": "cob"}')
    question = "What should I use on my wall?"
    reading = u.understand(question, detector, force=True)

    resolved = u.resolve(reading, question, detector, REGISTRY)

    assert resolved.substrate == "", "a substrate nobody stated reached the request"


def test_a_substrate_the_question_did_mention_survives(detector, monkeypatch):
    model_returning(monkeypatch, '{"intent": "select", "substrate": "brick"}')
    question = "What should I use on my brick wall?"
    reading = u.understand(question, detector, force=True)

    resolved = u.resolve(reading, question, detector, REGISTRY)

    assert resolved.substrate == "brick"
    assert resolved.provenance["substrate"] is Provenance.STATED


def test_the_model_word_is_normalised_through_the_vocabulary(detector, monkeypatch):
    """"outside" is the person's word; "external" is the corpus's.

    The model may only *find* what the vocabulary can confirm, which is what
    makes this normalisation rather than acceptance.
    """
    model_returning(monkeypatch, '{"intent": "select", "location": "outside"}')
    question = "What render should I use outside?"
    reading = u.understand(question, detector, force=True)

    resolved = u.resolve(reading, question, detector, REGISTRY)

    assert resolved.location == "external"


def test_an_intent_outside_the_enum_falls_back(detector, monkeypatch):
    model_returning(monkeypatch, '{"intent": "book_me_a_holiday"}')

    reading = u.understand("My render is crazing", detector, force=True)

    assert reading.intent is u.Intent.TROUBLESHOOT, "the deterministic reading stood"


def test_candidate_products_are_hypotheses_normalised_to_real_names(detector, monkeypatch):
    """A model may suggest. It may not thereby approve.

    The invented one is dropped here; the real ones still have to survive
    evidence assessment before either can be recommended.
    """
    model_returning(monkeypatch,
                    '{"intent": "select", "candidate_products": '
                    '["Ultra", "Duro", "Lime Green Supreme"]}')
    question = "What should I use?"
    reading = u.understand(question, detector, force=True)

    resolved = u.resolve(reading, question, detector, REGISTRY)

    assert "Lime Green Supreme" not in resolved.candidate_products
    assert len(resolved.candidate_products) == 2


# --------------------------------------------------------- failure is safe


def test_an_unreachable_model_costs_the_reading_and_not_the_answer(detector, monkeypatch):
    unavailable(monkeypatch)

    reading = u.understand("What product should I use on my brick wall?", detector)

    assert reading.source == "deterministic"
    assert reading.intent is u.Intent.SELECT, "the vocabulary still read it"
    assert reading.substrate == "brick"


def test_an_unparseable_reply_falls_back_rather_than_raising(detector, monkeypatch):
    model_returning(monkeypatch, "I think you should use Ultra!")

    reading = u.understand("What should I use?", detector, force=True)

    assert reading.source == "model+fallback"
    assert reading.error


def test_a_reply_that_is_valid_json_but_not_an_object_falls_back(detector, monkeypatch):
    model_returning(monkeypatch, '["Ultra", "Duro"]')

    reading = u.understand("What should I use?", detector, force=True)

    assert reading.source == "model+fallback"


def test_understanding_can_be_switched_off_entirely(detector, monkeypatch):
    """An operator must be able to run the system exactly as it ran before."""
    forbidden(monkeypatch)

    reading = u.understand("What should I use?", detector, enabled=False)

    assert reading.source == "deterministic"


# -------------------------------------------------- merging with the session


def test_this_turn_beats_what_the_conversation_remembered(detector):
    """"Actually it's stone" must not be answered from brick."""
    state = ConversationState(facts=merge_facts({}, {
        "substrate": SessionFact("substrate", "brick", Provenance.CARRIED, 1)}))
    question = "Actually the wall is stone."
    reading = u.deterministic(question, detector)

    resolved = u.resolve(reading, question, detector, REGISTRY, state=state)

    assert resolved.substrate == "stone"
    assert resolved.provenance["substrate"] is Provenance.STATED


def test_a_turn_that_says_nothing_inherits(detector):
    state = ConversationState(facts=merge_facts({}, {
        "product": SessionFact("product", "Ultra", Provenance.CARRIED, 1),
        "substrate": SessionFact("substrate", "brick", Provenance.CARRIED, 1)}))
    question = "How much would I need for 30 square metres?"
    reading = u.deterministic(question, detector)

    resolved = u.resolve(reading, question, detector, REGISTRY, state=state)

    assert resolved.product == "Ultra"
    assert resolved.substrate == "brick"
    assert resolved.provenance["product"] is Provenance.CARRIED
    assert resolved.measurements == {"area_m2": 30.0}


def test_a_slot_a_photograph_disputes_is_reported_unsettled_not_merged(detector):
    """The gate must be able to ask rather than pick a winner."""
    facts = merge_facts({}, {"substrate": SessionFact(
        "substrate", "brick", Provenance.STATED, 1)})
    facts = merge_facts(facts, {"substrate": SessionFact(
        "substrate", "stone", Provenance.OBSERVED, 3, confidence=0.9)})
    state = ConversationState(facts=facts)
    question = "Does that change anything?"

    resolved = u.resolve(u.deterministic(question, detector), question,
                         detector, REGISTRY, state=state)

    assert resolved.substrate == "", "a disputed slot was used as though settled"
    assert "substrate" in resolved.unsettled


def test_an_observed_slot_keeps_its_provenance_through_the_merge(detector):
    """A recommendation has to be able to say a photograph supplied this."""
    state = ConversationState(facts=merge_facts({}, {
        "substrate": SessionFact("substrate", "brick", Provenance.OBSERVED, 2,
                                 confidence=0.9, image_ref="IMG_1")}))
    question = "What thickness?"

    resolved = u.resolve(u.deterministic(question, detector), question,
                         detector, REGISTRY, state=state)

    assert resolved.substrate == "brick"
    assert resolved.provenance["substrate"] is Provenance.OBSERVED


# ------------------------------------------------------- the retrieval query


def test_the_retrieval_query_is_built_from_the_request_not_the_transcript(detector):
    """Decision 16.1 argues for retrieving on a profile rather than a sentence.

    What must never be in it is an earlier turn. The structured fields are
    added because they carry the job; the transcript is excluded because it
    carries the conversation, and embedding the conversation answers the
    conversation.
    """
    resolved = u.ResolvedRequest(
        intent=u.Intent.SELECT, raw_question="What should I use here?",
        product="Ultra", objective="insulation", substrate="brick",
        location="internal")

    query = resolved.retrieval_query()

    assert "Ultra" in query and "brick" in query and "internal" in query
    assert "What should I use here?" in query


def test_the_retrieval_query_does_not_repeat_itself(detector):
    """A term already in the question is not added twice."""
    resolved = u.ResolvedRequest(
        intent=u.Intent.SELECT, raw_question="brick", substrate="brick")

    assert resolved.retrieval_query().split().count("brick") == 1


def test_slots_are_the_shape_the_router_already_takes(detector):
    resolved = u.ResolvedRequest(
        intent=u.Intent.SELECT, raw_question="q", product="Ultra",
        substrate="brick", location="internal")

    assert resolved.slots() == {"product": "Ultra", "substrate": "brick",
                                "location": "internal"}


# ---------------------------------------------------- what becomes a fact


def test_only_this_turns_statements_become_new_facts(detector):
    """Re-writing an inherited value would reset the turn it came from.

    A substrate stated four turns ago would then look freshly confirmed, and
    the audit trail that makes an earlier answer explicable would be lost one
    turn at a time.
    """
    state = ConversationState(facts=merge_facts({}, {
        "substrate": SessionFact("substrate", "brick", Provenance.CARRIED, 1)}))
    question = "Would Ultra work?"
    resolved = u.resolve(u.deterministic(question, detector), question,
                         detector, REGISTRY, state=state)

    new = u.facts_from(resolved, turn_index=5)

    assert "substrate" not in new, "an inherited value was rewritten as new"
