"""Every guarantee holds on whichever entrypoint the caller happened to use.

The answer engine is a library with three ways in — `ask`, `answer_part` and
`ask_turn` — because the CLI, the web page and the graph each need a different
shape of call. That is the property decision 15 leans on, and it is also the
cheapest way to lose a guarantee: a check placed inside `ask` is invisible to
`answer_part`, and a boundary proved on one of them proves nothing about the
other. So most tests here are parametrised over `entrypoint` and run the same
assertions twice, or three times where the graph is involved.

The ordering claims are asserted by **replacing the thing that must not run
with a function that fails the test**. `forbidden` stands in for retrieval,
property lookup, embedding, the cache and the model, so "state answers come
before the cache and before retrieval" is the absence of a call rather than the
shape of a reply — a state answer that quietly retrieved first and then
discarded the hits would pass a text assertion and fail this one. The same
device covers policy running before factual rendering, and vision never being
invoked on a conversation-meta question.

Provenance is the other spine. A value that was *carried* may be quoted back
("you said brick"); a value that was *observed* in a photograph may not, because
the person never said it — the recall answer for an observed substrate is still
"not stated". `test_legacy_carried_slots_are_not_fabricated_observations`
covers the migration hazard directly: a caller passing only `carried` must not
have those values silently re-labelled as observations.

Two retrieval behaviours are worth calling out because they look similar and
are opposites. A first search below the threshold may be **retried with the
product context added**, and if the second search scores above the threshold
the answer prints with that real score; the sources must carry the score that
was actually achieved, not the one that justified answering. But lexical
evidence alone may not rescue a below-threshold result — that path refuses at
router step 1, with `engine.factual` forbidden so it cannot creep back in.

The fixture publishes Ultra, Solo, Duro and Bond into a real
`SQLiteKnowledgeRepository` and then stubs `retriever.search`, so lookups that
reach the store are real while the similarity ranking is fixed. Duro is always
present in the fixture and never asked about, which is how a leak would show.
`ollama.generate` is forbidden by default: everything asserted here is the
deterministic half of the engine.
"""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from assistant.answering import vision
from assistant.infrastructure import ollama
from assistant.answering.answer import Answer, Provenance
from assistant.turn.conversation import ConversationState, FactHistory, SessionFact
from assistant.answering.engine import Assistant
from assistant.knowledge.model import Retrieved, Snapshot
from assistant.answering.router import Decision, Path_
from assistant.knowledge.store import SQLiteKnowledgeRepository
from test_engine import CHUNKING_VERSION, chunk, document, unit, version


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "eval" / "evalset" / "Set 1" / "lime_green_ui_acceptance_tests.csv").open(
        encoding="utf-8-sig", newline="") as stream:
    CASES = {row["test_id"]: row["question"] for row in csv.DictReader(stream)}


def forbidden(*_args, **_kwargs):
    """Stand-in for a boundary this test says must not be crossed."""
    raise AssertionError("this boundary must not call a model, retrieval or cache")


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    """Four products in a real store, fixed retrieval, and no model.

    `app.fixture_hits` exposes the passages so a test can return a chosen
    subset from its own `search` stub. Duro is published but never asked
    about, so an answer that mentions it has leaked it.
    """
    entries = [
        ("Ultra", "Description", "Ultra is suitable for most masonry and lath backgrounds."),
        ("Ultra", "Preparation", "Prepare Ultra backgrounds by removing dust."),
        ("Ultra", "Application", "Apply Ultra at a uniform thickness between 10 and 30 mm."),
        ("Ultra", "Mixing", "Mix Ultra with 4 to 4.5 litres of water per bag."),
        ("Ultra", "Coverage", "Ultra covers approximately 1.5 m2 at 10 mm thickness."),
        ("Ultra", "Conditions", "Only use Ultra above 5 degrees and below 30 degrees."),
        ("Ultra", "Finishing Coats", "The finish coat should be 3 to 6mm thick."),
        ("Solo", "Description", "Solo is suitable for lath backgrounds."),
        ("Solo", "Application", "Apply Solo at a thickness of 3 to 6 mm."),
        ("Solo", "Mixing", "Mix Solo with 5 to 6 litres of water per bag."),
        ("Duro", "Mixing", "Mix Duro with 9 litres of water per bag."),
        ("Bond", "Preparation", "Prime Ultra with Bond."),
    ]
    docs, versions, chunks, hits = [], [], [], []
    for index, (product, section, text) in enumerate(entries):
        url = f"https://example.invalid/{product.lower()}/{index}"
        doc = document(url, product)
        passage = chunk(url, 0, section, text, product, 0)
        docs.append(doc)
        versions.append(version(url))
        chunks.append(passage)
        hits.append(Retrieved(chunk=passage, document=doc, score=0.71))
    snapshot = Snapshot(
        snapshot_id="acceptance-engine", created_at="2026-09-18T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL, embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=len(docs),
        chunk_count=len(chunks), notes={"products": ["Ultra", "Solo", "Duro", "Bond"]})
    repo = SQLiteKnowledgeRepository(tmp_path / "knowledge.db")
    repo.publish(docs, versions, chunks, snapshot, [])
    app = Assistant(repo, log=False)
    app.fixture_hits = hits
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", forbidden)
    monkeypatch.setattr(app.retriever, "search", lambda *_a, **_k: [hits[0], hits[10]])
    try:
        yield app
    finally:
        repo.close()


def call(app, entrypoint, question, **kwargs):
    """Drive either entrypoint and return one answer, so a test body can be shared."""
    if entrypoint == "ask":
        reply = app.ask(question, **kwargs)
        assert len(reply.parts) == 1
        return reply.parts[0][1]
    return app.answer_part(question, **kwargs)


@pytest.mark.parametrize("entrypoint", ["ask", "answer_part"])
@pytest.mark.parametrize("case,path", [
    ("T01", "ask_back"), ("T02", "state"), ("T03", "state"),
    ("T27", "acknowledge"), ("T28", "state"), ("T29", "state"),
    ("T30", "state"), ("T31", "state"), ("T32", "ask_back"),
])
def test_state_boundaries_precede_cache_and_retrieval(
        assistant, monkeypatch, entrypoint, case, path):
    """Nine conversation-state rows resolve before the cache, retrieval or embedding run.

    Retrieval, property lookup, both cache halves and embedding are all
    replaced with failures, so this asserts the work never happened. T27
    additionally checks the acknowledged slots, and every recall row says "not
    stated" with no sources.
    """
    monkeypatch.setattr(assistant.retriever, "search", forbidden)
    monkeypatch.setattr(assistant.retriever, "find_property", forbidden)
    monkeypatch.setattr(assistant.cache, "get", forbidden)
    monkeypatch.setattr(assistant.cache, "put", forbidden)
    monkeypatch.setattr(ollama, "embed_one", forbidden)
    answer = call(assistant, entrypoint, CASES[case])
    assert answer.path == path
    assert answer.diagnostics["cached"] is False
    assert answer.sources == []
    if case == "T27":
        assert answer.diagnostics["slots"] == {
            "product": "ultra", "substrate": "brick", "location": "internal"}
    elif path == "state":
        assert "not stated" in answer.text


@pytest.mark.parametrize("entrypoint", ["ask", "answer_part"])
@pytest.mark.parametrize("provenance", [Provenance.CARRIED, Provenance.OBSERVED,
                                      Provenance.ASSUMED])
def test_state_recall_preserves_provenance(assistant, entrypoint, provenance):
    """Only a carried value may be quoted back; observed and assumed ones may not.

    A `carried` argument naming a different substrate is passed alongside, and
    must never reach the text — the state answer speaks for the conversation,
    not for the caller.
    """
    state = ConversationState(facts={"substrate": FactHistory(
        SessionFact("substrate", "brick", provenance, image_ref="only-if-observed"))})
    answer = call(assistant, entrypoint, CASES["T29"], state=state,
                  carried={"substrate": "stone"})
    assert "stone" not in answer.text
    if provenance is Provenance.CARRIED:
        assert "You said" in answer.text and "brick" in answer.text
    else:
        assert "not stated" in answer.text
        assert "You said" not in answer.text
    assert bool(answer.observed) is (provenance is Provenance.OBSERVED)


def test_legacy_carried_slots_are_not_fabricated_observations(assistant):
    """A caller passing only `carried` does not get those values re-labelled as observed."""
    answer = assistant.answer_part(CASES["T29"], carried={"substrate": "brick"})
    assert "You said" in answer.text
    assert answer.observed == []
    observed = assistant.answer_part(
        CASES["T29"], carried={"substrate": "brick"},
        origins={"substrate": Provenance.OBSERVED})
    assert "not stated" in observed.text and observed.observed


def test_state_recall_does_not_run_vision(assistant, monkeypatch):
    """An attached photograph does not make a conversation-meta question a perception task."""
    monkeypatch.setattr(vision, "slots_from_images", forbidden)
    answer = assistant.ask(CASES["T28"], images=["unused.png"]).parts[0][1]
    assert answer.path == "state" and not answer.facts


@pytest.mark.parametrize("question", [
    'I am not using Ultra on brick.',
    'If I am using Ultra on brick.',
    'I said "I am using Ultra on brick".',
    '"I am using Ultra on brick."',
])
@pytest.mark.parametrize("entrypoint", ["ask", "answer_part"])
def test_nonassertions_do_not_get_acknowledged(
        assistant, monkeypatch, question, entrypoint):
    """Negated, conditional and quoted statements are not acknowledged and record nothing.

    Asserted at `_state_answer`, through both entrypoints, and finally by a
    following recall question that must still report nothing stated.
    """
    assert assistant._state_answer(question) is None
    monkeypatch.setattr(ollama, "generate", lambda *_a, **_k: ("", 0.0))
    answer = call(assistant, entrypoint, question)
    assert answer.path != "acknowledge"
    assert not any(name in answer.diagnostics.get("slots", {})
                   for name in ("product", "substrate", "location"))
    assert not any(f.provenance is Provenance.STATED for f in answer.facts)
    assert assistant.ask(CASES["T28"]).parts[0][1].diagnostics["slots"] == {}


@pytest.mark.parametrize("entrypoint", ["ask", "answer_part"])
def test_conditional_background_cannot_bypass_missing_fact_gate(assistant, entrypoint):
    """"If my wall is brick" does not fill the substrate slot, so the ask-back still fires."""
    answer = call(assistant, entrypoint,
                  "If my wall is brick, is Ultra suitable for me?")
    assert answer.path == Path_.ASK_BACK.value
    assert "substrate" not in answer.diagnostics["slots"]
    assert not any(f.provenance is Provenance.STATED for f in answer.facts)


def test_negated_background_does_not_change_inherited_observation(assistant):
    """A denial does not overwrite an observation with itself relabelled as stated."""
    answer = assistant.answer_part(
        "I am not using Ultra on brick. How much water does Ultra need?",
        carried={"substrate": "brick"}, origins={"substrate": Provenance.OBSERVED})
    fact = next(f for f in answer.facts if f.slot == "substrate")
    assert fact.provenance is Provenance.OBSERVED


def test_legacy_library_remains_stateless(assistant):
    """Two `ask` calls share nothing: the library carries no conversation of its own."""
    assistant.ask(CASES["T27"])
    answer = assistant.ask(CASES["T28"]).parts[0][1]
    assert "not stated" in answer.text


@pytest.mark.parametrize("case,field,text", [
    ("T08", "water", "4 to 4.5 litres"),
    ("T09", "coverage", "1.5 m2"),
    ("T10", "backgrounds", "masonry and lath"),
    ("T17", "thickness", "10 and 30 mm"),
])
def test_inherited_product_context_and_targeted_fields(
        assistant, monkeypatch, case, field, text):
    """A pronoun question with a carried product searches for that product and answers its field.

    The outgoing query is asserted to be prefixed and scoped with the product,
    the answered field is named in the diagnostics, and neither Duro's nor
    Solo's figures appear. Source scores are those actually retrieved.
    """
    queries = []

    def search(question, **kwargs):
        queries.append((question, kwargs))
        return [assistant.fixture_hits[0], assistant.fixture_hits[10]]

    monkeypatch.setattr(assistant.retriever, "search", search)
    answer = assistant.answer_part(CASES[case], carried={"product": "Ultra"})
    assert text in answer.text
    assert not answer.refused
    assert field in answer.diagnostics["deterministic_fields"]
    assert queries[0][0].startswith("Ultra: ")
    assert queries[0][1]["product"] == "Ultra"
    assert "9 litres" not in answer.text
    assert "3 to 6mm" not in answer.text
    assert {source["score"] for source in answer.sources} <= {0.71, 0.0}
    assert assistant.retriever.threshold == 0.45


def test_low_similarity_retries_context_without_fabricating_scores(assistant, monkeypatch):
    """A below-threshold first search is retried with context, and the printed score is the real one."""
    calls = []

    def search(question, **kwargs):
        calls.append(question)
        score = 0.363 if len(calls) == 1 else 0.61
        return [replace(assistant.fixture_hits[3], score=score)]

    monkeypatch.setattr(assistant.retriever, "search", search)
    answer = assistant.answer_part(CASES["T16"])
    assert len(calls) == 2 and "ultra water" in calls[1].lower()
    assert "4 to 4.5 litres" in answer.text
    assert answer.diagnostics["top_score"] == 0.61
    assert answer.sources[0]["score"] == 0.61


def test_lexical_evidence_cannot_override_threshold(assistant, monkeypatch):
    """Matching words do not rescue a below-threshold retrieval: it refuses at step 1.

    `engine.factual` is forbidden, so a path that reached rendering anyway
    would fail rather than quietly print.
    """
    monkeypatch.setattr(assistant.retriever, "search", lambda *_a, **_k: [
        replace(assistant.fixture_hits[3], score=0.363)])
    monkeypatch.setattr(assistant.engine, "factual", forbidden)
    answer = assistant.answer_part(CASES["T16"])
    assert answer.refused
    assert answer.diagnostics["step"] == "1"
    assert answer.diagnostics["top_score"] == 0.363


def test_comparison_recovers_both_named_products_not_carried_slot(assistant, monkeypatch):
    """T14 at the engine: both named products are looked up, the carried one is ignored.

    Each lookup is asserted to be product-scoped, bounded, and audience-scoped
    to the caller's set — the comparison path does not widen retrieval.
    """
    lookups = []
    original = assistant.retriever.find_property

    def record(product, terms, **kwargs):
        lookups.append((product, terms, kwargs))
        return original(product, terms, **kwargs)

    monkeypatch.setattr(assistant.retriever, "find_property", record)
    answer = assistant.answer_part(CASES["T14"], carried={"product": "Duro"})
    assert not answer.refused
    assert "ultra:" in answer.text and "solo:" in answer.text
    assert "10 and 30 mm" in answer.text and "3 to 6 mm" in answer.text
    assert "Duro" not in answer.text
    assert "product" not in answer.diagnostics["slots"]
    assert {"ultra", "solo"} <= {p for p, _, _ in lookups}
    assert all(k["limit"] <= 7 for _, _, k in lookups)
    assert all(k["audiences"] == ("public",) for _, _, k in lookups)


def test_multiask_keeps_preparation_thickness_water_conditions(assistant):
    """T11 at the engine: four fields answered, and the unsupported fifth said to be unsupported."""
    answer = assistant.answer_part(CASES["T11"])
    assert not answer.refused
    for text in ("removing dust", "10 and 30 mm", "4 to 4.5 litres",
                 "above 5 degrees", "curing: the retrieved evidence does not establish"):
        assert text in answer.text
    assert len(answer.sources) >= 4
    assert answer.failed_checks == []


def test_waterproof_is_not_inferred_from_a_render_description(assistant):
    """T35: a property the corpus never states is refused, not inferred from nearby prose."""
    answer = assistant.answer_part(CASES["T35"])
    assert answer.refused
    assert "does not" in answer.text.lower()
    assert "Ultra is waterproof" not in answer.text


@pytest.mark.parametrize("case", ["T19", "T20", "T21", "T22", "T23"])
def test_policy_gates_run_before_factual_rendering(assistant, monkeypatch, case):
    """The five compliance and injection rows route with both retrieval and rendering forbidden."""
    monkeypatch.setattr(assistant.retriever, "search", forbidden)
    monkeypatch.setattr(assistant.engine, "factual", forbidden)
    answer = assistant.ask(CASES[case]).parts[-1][1]
    assert answer.path == "route"


@pytest.mark.parametrize("case", ["T25", "T26"])
def test_purchase_questions_refuse_arithmetic(assistant, case):
    """A bag count is not computed; the coverage figure is quoted and the sum declined."""
    reply = assistant.ask(CASES[case], carried={"product": "Ultra"})
    answer = reply.parts[-1][1]
    assert "exact purchase quantity" in answer.text
    assert "1.5 m2" in answer.text
    assert "buy 10 bags" not in answer.text.lower()


def test_nonfactual_compose_still_receives_history(assistant, monkeypatch):
    """Conversation history reaches `compose`, which is the only place it is allowed to."""
    composed = []
    monkeypatch.setattr(assistant.router, "route", lambda *_a, **_k: Decision(
        Path_.COMPOSE, "test fallback", hits=assistant.fixture_hits[:1]))
    monkeypatch.setattr(assistant.engine, "compose", lambda _d, _q, history="": (
        composed.append(history) or Answer(path="compose", text="checked prose")))
    answer = assistant.answer_part("Explain lime render", history="prior conversation")
    assert answer.path == "compose"
    assert composed == ["prior conversation"]


def test_factual_is_called_only_after_route(assistant, monkeypatch):
    """Routing happens first, and the decision handed to rendering already carries its hits."""
    events = []
    original = assistant.router.route

    def route(*args, **kwargs):
        events.append("route")
        return original(*args, **kwargs)

    def factual(decision, question):
        events.append("factual")
        assert any("4 to 4.5 litres" in h.chunk.content for h in decision.hits)
        return Answer(path="extract", text="deterministic")

    monkeypatch.setattr(assistant.router, "route", route)
    monkeypatch.setattr(assistant.engine, "factual", factual)
    assistant.answer_part(CASES["T16"])
    assert events == ["route", "factual"]


@pytest.mark.parametrize("entrypoint", ["ask", "answer_part"])
def test_only_real_cache_hits_are_marked(assistant, entrypoint):
    """`cached` reports what actually happened: false, then true, and never for a state answer."""
    assistant._turn_snapshot = assistant.repo.snapshot()
    first = call(assistant, entrypoint, CASES["T16"])
    second = call(assistant, entrypoint, CASES["T16"])
    assert not first.diagnostics.get("cached", False)
    assert second.diagnostics["cached"] is True
    assert first.text == second.text
    state = call(assistant, entrypoint, CASES["T28"])
    assert state.diagnostics["cached"] is False


@pytest.mark.parametrize("question", [
    "Am I using Ultra on an internal brick wall?",
    "Did I mention Ultra on an internal brick wall?",
    "Compare Ultra on internal brick with Solo on external stone.",
    "Imagine I am using Ultra on an internal brick wall.",
    'I am using "Ultra on an internal brick wall.',
    'I am "not" using Ultra on an internal brick wall.',
    "Tell me about internal brick.",
    "Can I use Ultra on an internal brick wall?",
    "Could I use Ultra on an internal brick wall?",
    "I am using a product other than Ultra on my brick wall.",
])
@pytest.mark.parametrize("entrypoint", ["ask", "answer_part"])
def test_nonassertions_cannot_enter_legacy_remember(
        assistant, monkeypatch, question, entrypoint):
    """Ten non-assertions survive the whole round trip into the web session store.

    The answer records no stated or carried facts, and `Handler._remember`
    then stores nothing — so the UI cannot reintroduce testimony the engine
    refused.
    """
    from assistant.answering.engine import Reply
    from assistant.turn.session import SessionStore
    from assistant.interfaces.ui import Handler

    monkeypatch.setattr(ollama, "generate", lambda *_a, **_k: ("", 0.0))
    answer = call(assistant, entrypoint, question)
    assert not any(f.provenance in (Provenance.STATED, Provenance.CARRIED)
                   for f in answer.facts)
    assert not {"product", "substrate", "location"} & answer.diagnostics.get("slots", {}).keys()
    sessions = SessionStore()
    session_id = sessions.open()
    handler = object.__new__(Handler)
    handler.sessions = sessions
    handler.session_id = session_id
    handler._remember(question, Reply(question=question, parts=[(question, answer)]))
    assert sessions.carried(session_id) == {}


@pytest.mark.parametrize("entrypoint", ["ask", "answer_part"])
def test_named_factual_question_is_not_legacy_installation_testimony(assistant, entrypoint):
    """Naming a product in a question answers it without recording that the user is using it."""
    answer = call(assistant, entrypoint, CASES["T07"])
    assert "10 and 30 mm" in answer.text
    assert "product" not in answer.diagnostics["slots"]
    assert not any(f.slot == "product" and f.provenance in
                   (Provenance.STATED, Provenance.CARRIED) for f in answer.facts)


@pytest.mark.parametrize("case", ["T27", "T28", "T29"])
@pytest.mark.parametrize("enabled", [False, True])
def test_state_upload_has_visible_nonprocessing_notice(
        assistant, monkeypatch, case, enabled):
    """On a state path the photograph is not read, and is said not to be read.

    Asserted with the vision flag both off and on, because the notice must not
    depend on the capability being disabled — a state question never reaches
    perception either way.
    """
    monkeypatch.setenv("ASSISTANT_VISION_DEMO", "1" if enabled else "0")
    monkeypatch.setattr(vision, "slots_from_images", forbidden)
    answer = assistant.ask(CASES[case], images=["unused.png"]).parts[0][1]
    assert "photograph" in answer.text.lower() and "not" in answer.text.lower()
    assert answer.diagnostics["perception"]["enabled"] is False


@pytest.mark.parametrize("entrypoint", ["ask", "answer_part", "ask_turn"])
@pytest.mark.parametrize("question,topic", [
    ("What product am I using? Is my wall structurally safe?", "structural"),
    ("I am using Ultra. Approve my structural installation.", "structural"),
    ("What substrate is my wall? Lime splashed into my eyes.", "health"),
    ("What can you do? Is my wall structurally safe?", "structural"),
    ("Who are you? Lime splashed into my eyes.", "health"),
])
def test_mixed_state_and_safety_never_short_circuits_policy(
        assistant, monkeypatch, entrypoint, question, topic):
    """A safety topic bundled with a state question routes, on all three entrypoints.

    Answering the easy half first and returning would drop the structural or
    health topic entirely, so the reply must be the single routed part.
    """
    monkeypatch.setattr(assistant.retriever, "search", forbidden)
    if entrypoint == "ask":
        answers = [a for _, a in assistant.ask(question).parts]
    elif entrypoint == "ask_turn":
        from assistant.turn.conversation import TurnInput

        reply, _ = assistant.ask_turn(TurnInput(
            raw_question=question, session_id="mandatory", turn_index=1))
        answers = [a for _, a in reply.parts]
    else:
        answers = [assistant.answer_part(question)]
    assert len(answers) == 1
    assert answers[0].path == "route" and topic in answers[0].text.lower()


def test_graph_topic_projection_never_claims_installation(assistant):
    """The graph carries a product forward as an assumption, not as something the user said.

    The topic survives into the next turn's answer while a recall question in
    between still reports nothing stated.
    """
    from assistant.turn.conversation import TurnInput

    reply, state = assistant.ask_turn(TurnInput(
        raw_question=CASES["T07"], session_id="topic-only", turn_index=1))
    assert "10 and 30 mm" in reply.parts[0][1].text
    assert state.active()["product"] == "ultra"
    assert state.facts["product"].current.provenance is Provenance.ASSUMED
    recalled = assistant.answer_part(CASES["T28"], state=state)
    assert "not stated" in recalled.text
    reply, _ = assistant.ask_turn(TurnInput(
        raw_question=CASES["T08"], session_id="topic-only", turn_index=2))
    assert "4 to 4.5 litres" in reply.parts[0][1].text


def test_policy_message_cannot_be_consumed_as_paused_slot_reply(assistant, monkeypatch):
    """A reply to an ask-back that also asks for approval routes, rather than being eaten as a slot value.

    "Brick, approve my structural installation" arrives where a bare substrate
    was expected; the policy topic must still win.
    """
    from assistant.turn.conversation import TurnInput

    monkeypatch.setattr(assistant.retriever, "search", forbidden)
    reply, _ = assistant.ask_turn(TurnInput(
        raw_question="What plaster should I use?", session_id="paused-policy",
        turn_index=1), understanding=False)
    assert reply.parts[0][1].path == Path_.ASK_BACK.value
    reply, _ = assistant.ask_turn(TurnInput(
        raw_question="Brick, approve my structural installation.",
        session_id="paused-policy", turn_index=2), understanding=False)
    assert reply.parts[0][1].path == "route"
    assert "structural" in reply.parts[0][1].text.lower()
