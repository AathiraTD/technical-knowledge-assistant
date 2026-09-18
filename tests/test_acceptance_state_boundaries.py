"""What the conversation believes is testimony, and testimony is not evidence.

Two kinds of thing can be true inside a turn. "The Solo datasheet says 5 to 6
litres" is evidence — retrieved, cited, checkable. "I am using Ultra on an
internal brick wall" is testimony: nobody can verify it, it is not in the
corpus, and it changes which product gets recommended. Keeping those apart is
what this file exists for, and the acceptance set's *New Chat reset* and
*Conversation-meta routing* rows (T01-T03, T27-T32) are the questions that
catch a system which has confused them.

Three boundaries are asserted throughout, and each was a real defect.

**Recall reads the checkpoint, never the transcript.** `test_recall_reads_checkpoint_not_transcript`
passes a `history` string that says the opposite of what the user said — the
transcript claims Solo on stone, the checkpoint holds Ultra on brick — and the
answer must come from the checkpoint. A conversation-meta question answered
from prose is a system inventing memory, which is precisely what T02 is written
to detect.

**A recall or acknowledgement never reaches a model or retrieval.** The `app`
fixture fails the test on any call to `ollama.generate`, `ollama.embed_one` or
`retriever.search`, so "what substrate did I say my wall was?" cannot be
answered with the nearest passage that happens to mention brick. That is not
hypothetical: a bare retraction used to fall through to retrieval, because
`asserted_text` empties a sentence containing "not" by design, so the
acknowledgement could not see it and the person correcting their wall was
answered with a passage instead.

**Only a real assertion becomes a fact.** The twenty-five parametrised
non-assertion cases are the heart of the file: negation, hypotheticals,
quotation in four different quote styles, hedges, disjunction, exclusion,
questions phrased as statements, and a model reading handed the values
directly. In each, `und.resolve` is given a `TurnUnderstanding` that already
*claims* product, substrate and location — so what is being asserted is that
the deterministic filter overrules the model, not that the model happened not
to guess. `test_facts_from_revalidates_each_value_not_just_nonempty_text`
closes the same door from the other side, forging a `ResolvedRequest` with
`STATED` provenance and asserting it still yields nothing.

Retraction is history rather than deletion: the withdrawn value is
`SUPERSEDED`, no later turn may inherit it, and it stays readable because an
answer given while it was believed still has to be explicable. A vision
observation is retained with its own provenance and never promoted into
something the user said — where the two disagree, the person's word stands and
the conflict is reported rather than resolved.

No model, no retrieval and no store: `services.engine` and `services.retriever`
are mocks, and vision is a stub. This file says nothing about answer quality,
only about what the turn was allowed to believe before answering.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistant.turn import graph
from assistant.infrastructure import observability as obs, ollama
from assistant.answering import understanding as und
from assistant.answering.answer import Answer, Provenance
from assistant.turn.conversation import (
    ConversationState,
    FactStatus,
    SessionFact,
    merge_facts,
)
from assistant.answering.router import PolicyGate, SlotDetector


REGISTRY = ("Ultra", "Lime Green Ultra", "Solo", "Duro")
RECALL = (
    "What product did I say I was using?",
    "What substrate did I say my wall was?",
    "What product am I using?",
    "What substrate is my wall?",
)


@pytest.fixture
def app(monkeypatch):
    """The graph with mocked services, every one of which fails the test if called.

    Yields `(compiled_graph, services)` so a test can both drive a turn and
    assert which boundaries were never crossed.
    """
    def forbidden(*args, **kwargs):
        pytest.fail("state boundary invoked a model or product retrieval")

    monkeypatch.setattr(ollama, "generate", forbidden)
    monkeypatch.setattr(ollama, "embed_one", forbidden)
    retriever = SimpleNamespace(search=Mock(side_effect=forbidden))
    engine = SimpleNamespace(
        answer_part=Mock(return_value=Answer(text="Published answer.", path="extract")),
        need_more_information=Mock(return_value=Answer(
            text="What is the wall built of?", path="ask_back")),
    )
    services = graph.Services(
        retriever=retriever, engine=engine, registry=REGISTRY, repo=None,
        router=SimpleNamespace(slots=SlotDetector(), gate=None),
        interruptible=False, understanding_enabled=True)
    return graph.build(services), services


def invoke(app, question, index=1, thread="acceptance", **extra):
    """One turn through the compiled graph, on a named thread so chats stay separate."""
    return app.invoke({"raw_question": question, "turn_index": index,
                       "images": [], "history": "", **extra},
                      {"configurable": {"thread_id": thread}})


@pytest.mark.parametrize("question", RECALL)
def test_missing_state_never_calls_models_or_retrieval(app, question):
    """All four recall questions answer "not stated in the current chat" with no work done."""
    compiled, services = app
    result = invoke(compiled, question)
    answer = result["answer"]
    assert answer.path == "state"
    assert "not stated" in answer.text and "current chat" in answer.text
    assert not answer.sources and not answer.facts
    assert not result["facts"]
    services.engine.answer_part.assert_not_called()
    services.retriever.search.assert_not_called()


def test_fact_statement_acknowledges_without_unsolicited_specs(app):
    """A statement of facts is acknowledged and recorded, not answered with a datasheet."""
    compiled, services = app
    result = invoke(compiled, "I am using Lime Green Ultra on an internal brick wall.")
    assert result["answer"].path == "acknowledge"
    assert result["answer"].sources == []
    assert {key: history.current.value for key, history in result["facts"].items()} == {
        "product": "ultra", "substrate": "brick", "location": "internal"}
    assert all(history.current.provenance is Provenance.STATED
               for history in result["facts"].values())
    services.engine.answer_part.assert_not_called()
    services.retriever.search.assert_not_called()


@pytest.mark.parametrize("question", RECALL)
def test_recall_reads_checkpoint_not_transcript(app, question):
    """A transcript contradicting the checkpoint changes nothing about what is recalled.

    The history says Solo on stone; the checkpoint says Ultra on brick. A
    conversation-meta answer read from prose would be invented memory.
    """
    compiled, services = app
    invoke(compiled, "I am using Lime Green Ultra on an internal brick wall.")
    result = invoke(compiled, question, 2,
                    history="Assistant: You are using Solo on stone.")
    expected = "ultra" if "product" in question else "brick"
    assert expected in result["answer"].text
    assert "solo" not in result["answer"].text.lower()
    assert result["answer"].facts[0].provenance is Provenance.CARRIED
    assert all(h.current.source_turn == 1 for h in result["facts"].values())
    services.engine.answer_part.assert_not_called()


def test_a_retraction_is_acknowledged_without_retrieving_anything(app):
    """"Actually, my wall is not brick" is a state instruction, not a question.

    It used to fall through to retrieval -- `asserted_text` empties a sentence
    containing "not", by design, so the acknowledgement could not see it -- and
    the person correcting their wall was answered with the nearest passage that
    happened to mention brick. The `app` fixture fails on any model or
    retrieval call, so this asserts the absence of both.
    """
    compiled, services = app
    invoke(compiled, "My wall is brick.")
    result = invoke(compiled, "Actually, my wall is not brick.", 2)

    assert result["answer"].path == "acknowledge"
    assert result["answer"].sources == []
    # A denial asserts nothing: the slot becomes unknown, not known to be
    # something else.
    assert result["answer"].facts == []
    assert "brick" not in result["answer"].text.lower()
    services.engine.answer_part.assert_not_called()
    services.retriever.search.assert_not_called()


def test_a_retracted_substrate_is_no_longer_trusted_but_is_still_history(app):
    """A withdrawn value is superseded and uninheritable, yet retained as history."""
    compiled, _services = app
    invoke(compiled, "My wall is brick.")
    invoke(compiled, "Actually, my wall is not brick.", 2)
    result = invoke(compiled, "What substrate did I say my wall was?", 3)

    history = result["facts"]["substrate"]
    # Retired, so no later turn may inherit it...
    assert history.current.status is FactStatus.SUPERSEDED
    assert ConversationState(facts=result["facts"]).active() == {}
    # ...but kept, because an answer given while brick was believed still has
    # to be explicable.
    assert history.current.value == "brick"
    assert [f.value for f in history.superseded] == ["brick"]


def test_recall_after_a_retraction_does_not_speak_for_the_withdrawn_value(app):
    """The defect: turn three still reported the value turn two took back."""
    compiled, services = app
    invoke(compiled, "My wall is brick.")
    invoke(compiled, "Actually, my wall is not brick.", 2)
    result = invoke(compiled, "What substrate did I say my wall was?", 3)

    answer = result["answer"]
    assert answer.path == "state"
    assert "brick" not in answer.text.lower()
    assert "You said your substrate was" not in answer.text
    assert "not stated" in answer.text and "current chat" in answer.text
    assert answer.facts == []
    services.engine.answer_part.assert_not_called()
    services.retriever.search.assert_not_called()


@pytest.mark.parametrize("question, acknowledged", [
    ("Actually, my wall is not brick.", True),
    ("My wall is not brick.", True),           # a bare retraction, opener or not
    ("My wall is not brick. What should I use?", False),
    ("My wall is not brick. What thickness do I need?", False),
])
def test_only_a_bare_retraction_is_answered_by_an_acknowledgement(question, acknowledged):
    """A correction carrying a question must still reach the answering path.

    Asserted at the boundary rather than through the graph, because a turn that
    is allowed past it goes on to call the model, which is exactly what the
    `app` fixture forbids.
    """
    state = ConversationState(facts=merge_facts(
        {}, {"substrate": SessionFact("substrate", "brick", Provenance.STATED, 1)}))
    answer = und.state_only_answer(
        question, SlotDetector(), REGISTRY, state, 2, gate=PolicyGate())

    if acknowledged:
        assert answer is not None and answer.path == "acknowledge"
        assert "no longer recorded" in answer.text
        assert answer.facts == []
    else:
        assert answer is None or answer.path != "acknowledge"


@pytest.mark.parametrize("question", [
    "I am not using Ultra on a brick wall.",
    "I am never using Ultra on a brick wall.",
    "If I am using Ultra on a brick wall, what happens?",
    "Suppose I am using Ultra on a brick wall.",
    "I might use Ultra on a brick wall.",
    "I would use Ultra on a brick wall.",
    '"I am using Ultra on a brick wall."',
    "'I am using Ultra on a brick wall.'",
    "“I am using Ultra on a brick wall.”",
    "`I am using Ultra on a brick wall.`",
    "The example says I am using Ultra on a brick wall.",
    "I am not applying 12.5 mm of Solo on a stone wall.",
    "If I apply 12.5 mm of Solo on a stone wall, what happens?",
    "If this were my project:\nI am using Ultra on a brick wall.",
])
def test_non_assertions_cannot_populate_facts(question):
    """Fourteen ways to mention a fact without asserting it, all yielding no facts.

    The `TurnUnderstanding` handed in already claims product, substrate and
    location, so this asserts the deterministic filter overrules the model
    rather than that the model declined to guess.
    """
    detector = SlotDetector()
    reading = und.TurnUnderstanding(
        explicit_product="Ultra", substrate="brick", location="internal")
    resolved = und.resolve(reading, question, detector, REGISTRY)
    assert not resolved.slots()
    assert und.facts_from(resolved, 1) == {}
    assert und.state_only_answer(question, detector, REGISTRY) is None


def test_model_cannot_supply_registered_but_unmentioned_product():
    """A product the model names but the question never mentions is not resolved."""
    resolved = und.resolve(und.TurnUnderstanding(explicit_product="Ultra"),
                           "Tell me about plaster.", SlotDetector(), REGISTRY)
    assert not resolved.product


def test_product_alias_requires_word_boundaries():
    """"ultramarine" does not contain the product Ultra."""
    resolved = und.resolve(und.TurnUnderstanding(), "My wall is ultramarine.",
                           SlotDetector(), REGISTRY)
    assert not resolved.product


def test_guarded_statement_does_not_overwrite_prior_facts(app):
    """A hedged or negated second turn leaves the first turn's facts and their source turn intact."""
    compiled, _ = app
    invoke(compiled, "I am using Ultra on a brick wall.")
    result = invoke(compiled, "I am not using Solo on a stone wall.", 2)
    assert result["facts"]["product"].current.value == "ultra"
    assert result["facts"]["substrate"].current.value == "brick"
    assert result["facts"]["product"].current.source_turn == 1


@pytest.mark.parametrize("question", [
    "How thick should I apply it?",
    "What coverage does it give?",
    "What backgrounds is it suitable for?",
])
def test_unknown_product_reference_clarifies_before_retrieval(app, question):
    """"How thick should I apply it?" with no product asks back, naming what is missing.

    This is acceptance row T01: guessing a product here is the failure, and
    the clarification happens before anything is retrieved.
    """
    compiled, services = app
    answer = invoke(compiled, question)["answer"]
    assert answer.path == "ask_back"
    assert answer.diagnostics["missing"] == ["product"]
    services.engine.answer_part.assert_not_called()
    services.retriever.search.assert_not_called()


def test_general_material_question_is_not_blanket_blocked(app):
    """A question about lime plaster in general is not a missing-product question."""
    compiled, services = app
    invoke(compiled, "How thick should lime plaster be?")
    services.engine.answer_part.assert_called_once()


def test_known_product_background_question_delegates_lookup(app):
    """Once a product is established, "it" resolves and the lookup is delegated with it carried."""
    compiled, services = app
    invoke(compiled, "I am using Ultra on an internal brick wall.")
    result = invoke(compiled, "What backgrounds is it suitable for?", 2)
    assert result["resolved"].intent is und.Intent.LOOKUP
    assert result["resolved"].product == "ultra"
    services.engine.answer_part.assert_called_once()
    assert services.engine.answer_part.call_args.kwargs["carried"]["product"] == "ultra"
    assert "delegate:extract" in result["trace"]
    services.retriever.search.assert_not_called()


@pytest.mark.parametrize("question", [
    "What product should I use on my wall?",
    "Which plaster is suitable for a brick wall?",
    "What should I use on this wall?",
])
def test_genuine_selection_remains_selection(question):
    """Asking which product to use stays a selection intent rather than becoming a lookup."""
    reading = und.deterministic(question, SlotDetector(), registry=REGISTRY)
    assert reading.intent is und.Intent.SELECT


@pytest.mark.parametrize("question", RECALL[1::2])
def test_observation_is_retained_without_becoming_user_testimony(app, question):
    """A vision observation answers "not stated", says it came from a photograph, and keeps its provenance."""
    compiled, services = app
    observed = SessionFact("substrate", "stone", Provenance.OBSERVED, 1,
                           image_ref="image-1", confidence=0.9)
    result = invoke(compiled, question, 2,
                    facts=merge_facts({}, {"substrate": observed}))
    assert "not stated" in result["answer"].text
    assert "photograph" in result["answer"].text
    assert result["facts"]["substrate"].current == observed
    assert result["answer"].facts[0].provenance is Provenance.OBSERVED
    services.engine.answer_part.assert_not_called()


def test_conflicting_observation_does_not_erase_what_user_said(app):
    """Where an image disagrees with the person, the person stands and the conflict is reported."""
    compiled, _ = app
    facts = merge_facts({}, {"substrate": SessionFact(
        "substrate", "brick", Provenance.STATED, 1)})
    facts = merge_facts(facts, {"substrate": SessionFact(
        "substrate", "stone", Provenance.OBSERVED, 2)})
    result = invoke(compiled, RECALL[1], 3, facts=facts)
    assert "brick" in result["answer"].text
    assert "disagrees" in result["answer"].text
    assert result["facts"]["substrate"].unsettled
    assert result["facts"]["substrate"].current.value == "brick"
    assert result["facts"]["substrate"].contradicted_by[0].value == "stone"


def test_distinct_threads_and_deleted_checkpoint_do_not_recall_other_chat(app):
    """A new thread and a deleted checkpoint both recall nothing — the New Chat reset rows."""
    compiled, services = app
    invoke(compiled, "I am using Ultra on a brick wall.")
    fresh = invoke(compiled, RECALL[0], thread="new-chat")
    assert "not stated" in fresh["answer"].text
    services.checkpointer.delete_thread("acceptance")
    reset = invoke(compiled, RECALL[0])
    assert "not stated" in reset["answer"].text


def test_delegate_reports_actual_cache_status(app):
    """The span records whether the delegated answer was really cached, not a default."""
    compiled, services = app
    services.engine.answer_part.return_value.diagnostics["cached"] = True
    with obs.turn(turn="1", session="cache", source="test") as spans:
        invoke(compiled, "What coverage does Ultra give?")
    part = next(span for span in spans if span.name == "part")
    assert part.attributes["cached"] is True


def test_recall_does_not_retain_previous_turn_decision(app):
    """Routing state from the previous turn is cleared, so a stale decision cannot be reused."""
    compiled, _ = app
    result = invoke(compiled, RECALL[0], decision="obsolete", hits=["obsolete"],
                    missing=["substrate"], outcome="obsolete")
    assert result["decision"] is None
    assert result["hits"] == [] and result["missing"] == []
    assert result["outcome"] == "state"


def test_technical_question_after_statement_is_not_swallowed():
    """A statement followed by a real question must still reach the answering path."""
    question = "I am using Ultra on a brick wall. What thickness should I use?"
    assert und.state_only_answer(question, SlotDetector(), REGISTRY) is None


def test_decimal_measurements_survive_assertion_filter():
    """Area and thickness with decimal points are read as measurements, not discarded."""
    question = "I have 20.5 m2 at 12.5 mm."
    resolved = und.resolve(und.deterministic(question, SlotDetector()),
                           question, SlotDetector(), REGISTRY)
    assert resolved.measurements == {"area_m2": 20.5, "thickness_mm": 12.5}


def test_policy_precedes_unknown_product_clarification(app):
    """A commercial topic in the message is not answered by an ask-back for a product."""
    compiled, services = app
    services.router.gate = PolicyGate()
    result = invoke(compiled, "What does it cost and what coverage does it give?")
    assert result["boundary_answer"] is None
    services.engine.answer_part.assert_called()


def test_statement_with_image_preserves_explicit_vision_provider(app):
    """With vision explicitly provided, the image is read, recorded as observed, and still loses to what was said."""
    from test_graph import StubVision

    compiled, services = app
    services.vision = StubVision({"substrate": "stone"})
    result = invoke(compiled, "My wall is brick.", images=["image-1"])
    assert services.vision.calls == 1
    assert result["facts"]["substrate"].current.value == "brick"
    assert result["facts"]["substrate"].contradicted_by[0].value == "stone"
    assert result["observations"][0].provenance is Provenance.OBSERVED
    assert result["answer"].path == "acknowledge"


def test_statement_with_image_keeps_default_vision_off(app, monkeypatch):
    """Without the capability flag, the image is not analysed and the turn says so."""
    compiled, _ = app
    monkeypatch.delenv("ASSISTANT_VISION_DEMO", raising=False)
    result = invoke(compiled, "My wall is brick.", images=["image-1"])
    assert "analyse_images:disabled" in result["trace"]
    assert result["perception"]
    assert result["answer"].path == "acknowledge"


@pytest.mark.parametrize("question", [
    "Am I using Ultra on an internal brick wall?",
    "Did I mention Ultra on an internal brick wall?",
    "Do you remember Ultra on an internal brick wall?",
    "Compare Ultra on internal brick with Solo on external stone.",
    "Ultra versus Solo on internal brick.",
    "Imagine I am using Ultra on an internal brick wall.",
    "I am considering Ultra on an internal brick wall.",
    'I am using "Ultra on an internal brick wall.',
    'I am "not" using Ultra on an internal brick wall.',
    "I am using Ultra or Solo on an internal brick wall.",
    "If this were my project; I am using Ultra on an internal brick wall.",
])
def test_nonassertions_reject_all_fact_entrypoints(question):
    """Eleven more non-assertions, closed at both entrypoints.

    `resolve` must produce no slots, and a directly forged `ResolvedRequest`
    with `STATED` provenance must still produce no facts — so the guard cannot
    be bypassed by whatever constructs the request.
    """
    detector = SlotDetector()
    reading = und.TurnUnderstanding(
        explicit_product="Ultra", substrate="brick", location="internal")
    resolved = und.resolve(reading, question, detector, REGISTRY)
    assert not resolved.slots()
    forged = und.ResolvedRequest(
        und.Intent.LOOKUP, question, product="ultra", substrate="brick",
        location="internal", provenance={
            name: Provenance.STATED for name in ("product", "substrate", "location")})
    assert und.facts_from(forged, 1) == {}


def test_model_grounding_requires_whole_words():
    """"cobwebs" is not cob and "internalised" is not internal."""
    resolved = und.resolve(
        und.TurnUnderstanding(substrate="cob", location="internal"),
        "My wall has cobwebs and internalised stains.", SlotDetector(), REGISTRY)
    assert not resolved.substrate and not resolved.location


@pytest.mark.parametrize("question", [
    "Tell me about internal brick.",
    "What is internal brick?",
    "Is the wall brick?",
])
def test_generic_material_questions_are_not_building_testimony(question):
    """Asking about brick is not saying your wall is brick."""
    resolved = und.resolve(
        und.TurnUnderstanding(substrate="brick", location="internal"),
        question, SlotDetector(), REGISTRY)
    assert not resolved.substrate and not resolved.location
    assert not und.facts_from(resolved, 1)


def test_facts_from_revalidates_each_value_not_just_nonempty_text():
    """Each value is revalidated against the sentence, so a forged request yields no facts."""
    forged = und.ResolvedRequest(
        und.Intent.LOOKUP, "I am not using Ultra on brick. My wall is stone.",
        product="ultra", substrate="brick",
        provenance={"product": Provenance.STATED, "substrate": Provenance.STATED})
    assert und.facts_from(forged, 1) == {}


def test_named_lookup_is_topic_not_installation_testimony(app):
    """Naming a product in a question sets the topic for "it", without becoming testimony.

    The follow-up resolves to Ultra, and a recall question still reports that
    no product was stated — the distinction T28 is written to find.
    """
    compiled, services = app
    first = invoke(compiled, "What thickness should Lime Green Ultra be applied at?")
    assert first["resolved"].product == "ultra"
    assert "product" not in first["facts"]
    second = invoke(compiled, "How much water does it need?", 2)
    assert second["resolved"].product == "ultra"
    assert services.engine.answer_part.call_args.kwargs["carried"]["product"] == "ultra"
    recall = invoke(compiled, RECALL[0], 3)
    assert "not stated" in recall["answer"].text


def test_active_topic_preserves_actual_testimony_and_its_source_turn(app):
    """Topic may move with the questions; what the user actually said keeps its turn and its value."""
    compiled, _ = app
    invoke(compiled, "I am using Ultra on a brick wall.")
    result = invoke(compiled, "How much water does it need?", 2)
    assert result["resolved"].provenance["product"] is Provenance.CARRIED
    assert result["facts"]["product"].current.source_turn == 1
    invoke(compiled, "What thickness does Solo need?", 3)
    followup = invoke(compiled, "How much water does it need?", 4)
    assert followup["resolved"].product == "solo"
    recall = invoke(compiled, RECALL[0], 5)
    assert "ultra" in recall["answer"].text


@pytest.mark.parametrize("question", [
    "Can I use Ultra on an internal brick wall?",
    "Could I use Ultra on an internal brick wall?",
    "I am using Ultra on an internal brick wall?",
    "I am using a product other than Ultra on my brick wall.",
    "I am using anything but Ultra on my brick wall.",
])
def test_use_questions_and_exclusions_are_not_installation_assertions(app, question):
    """"Can I use", "could I use" and "anything but" record nothing, and recall stays empty."""
    compiled, _ = app
    result = invoke(compiled, question)
    assert not result["facts"]
    resolved = result["resolved"]
    assert not und.facts_from(resolved, 1)
    assert not resolved.substrate and not resolved.location
    recall = invoke(compiled, RECALL[0], 2)
    assert "not stated" in recall["answer"].text


def test_comparison_never_overwrites_installation_or_active_topic(app):
    """A comparison names products without adopting either, and the established one survives it."""
    compiled, _ = app
    invoke(compiled, "I am using Duro on a brick wall.")
    result = invoke(compiled, "Compare Ultra and Solo thickness.", 2)
    assert result["resolved"].product == ""
    assert result["facts"]["product"].current.value == "duro"
    followup = invoke(compiled, "How much water does it need?", 3)
    assert followup["resolved"].product == "duro"


@pytest.mark.parametrize("question", [RECALL[0], RECALL[1], "My wall is brick."])
def test_boundary_upload_is_visibly_unprocessed_by_default(app, monkeypatch, question):
    """An attachment that was not read is said to be not read, on every state path."""
    compiled, _ = app
    monkeypatch.delenv("ASSISTANT_VISION_DEMO", raising=False)
    result = invoke(compiled, question, images=["image-1"])
    assert result["perception"]["enabled"] is False
    assert "photograph" in result["answer"].text.lower()
    assert "not" in result["answer"].text.lower()


def test_recall_upload_uses_explicit_provider_without_creating_testimony(app):
    """A read photograph fills the slot as observed, and recall still reports nothing stated."""
    from test_graph import StubVision

    compiled, services = app
    services.vision = StubVision({"substrate": "stone"})
    result = invoke(compiled, RECALL[1], images=["image-1"])
    assert services.vision.calls == 1
    assert "not stated" in result["answer"].text
    assert result["facts"]["substrate"].current.provenance is Provenance.OBSERVED


@pytest.mark.parametrize("question", [
    "I am using Ultra. Approve my structural installation.",
    "My wall is brick and structurally safe.",
    "What product am I using? Is my wall structurally safe?",
    "What substrate is my wall? Lime splashed into my eyes.",
])
def test_mandatory_policy_blocks_boundary_even_without_optional_gate(question):
    """The safety check inside the state boundary is not the optional `PolicyGate`.

    A structural or health topic mixed into a statement or a recall must leave
    the boundary rather than be answered there, whether or not a gate was
    configured on the services.
    """
    assert und.state_only_answer(question, SlotDetector(), REGISTRY) is None
