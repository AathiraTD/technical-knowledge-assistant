"""Regressions for four answer-quality defects found by the gold set.

Each of these was measured against the real 94-document index before it was
fixed, and each is reproduced here against fixtures so it fails in a second
without a model, an index or a network. The gold scenarios in `eval/gold.json`
cover the same three behaviours end to end; these are the fast versions that
run in CI and say which component broke.

The defects, in the order they appear below:

1. **The relevance gate disagreed with itself.** Router step 4 accepts any of
   the properties a question asks about; check 6 re-derived its terms from the
   single best-scoring one. A two-property question therefore passed one half
   of the gate and was refused by the other.

2. **A health question missed the policy gate.** The pattern required the verb
   to follow the word "eye", so "I got lime plaster in my eye" reached
   retrieval and a 142-second generation instead of the instant referral that
   names 111 and the safety data sheet.

3. **A diagnosis hand-off quoted marketing copy.** Product pages outrank
   knowledge-base articles on authority and score highly on similarity, so the
   three passages printed under "what the site does publish on this" were the
   pages selling the product rather than the technical note explaining the
   defect.

4. **The recommendation guard swallowed a refusal.** A refusal quotes the
   closest published passage, and published prose reads as advice; the guard
   cannot tell a quotation from a recommendation, so it replaced a correct
   cited hand-off with a bare one. The two orchestration paths then gave
   different answers to the same question -- and the half of the canonical
   evaluation that would have noticed calls the path no customer uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answering.answer import _diagnostic_passages, run_checks   # noqa: E402
from assistant.model import Chunk, Document, Retrieved          # noqa: E402
from assistant.answering.router import PolicyGate, Router                 # noqa: E402


def passage(content: str, *, product: str = "Forte Render Base Coat",
            section: str = "Finishing Coats", url: str = "https://example/forte",
            document_type: str = "datasheet", authority: int = 1,
            score: float = 0.8) -> Retrieved:
    return Retrieved(
        chunk=Chunk(canonical_url=url, version=1, chunk_index=0, section=section,
                    content=content, product=product,
                    document_type=document_type, authority=authority),
        score=score,
        document=Document(canonical_url=url, title=product,
                          document_type=document_type, authority=authority,
                          product=product, link_text=product),
    )


NAMES = {"products": ["Forte Render Base Coat", "Tradirend Lime Render"],
         "colours": [], "merchants": [],
         "contact": {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}}


# ------------------------------------------------- 1. the gate's two halves

# The Forte datasheet's own words, and the point of the fixture: this passage
# answers a compatibility question without using a single compatibility word.
FORTE_FINISHING = passage(
    "A number of different finish coats and techniques may be used. "
    "Lime Green Tradirend, Natural Finish or Finish WP: key the Forte with a "
    "regular pattern using a render scarifier. Leave to harden for 3 to 5 days "
    "and lightly re-dampen the Forte before applying the finish.")

TWO_PROPERTY_QUESTION = ("Which finish coats are compatible with Forte render, "
                         "including the hardening time required first?")


def test_the_gate_reads_every_property_the_question_asked_for():
    """`detect_properties` finds three; the gate must be satisfied by any."""
    router = Router()
    properties = router.slots.detect_properties(TWO_PROPERTY_QUESTION)
    assert "compatibility" in properties
    # The halves the passage *does* carry lexically. Without these the question
    # has only one readable property and the bug is not reproducible.
    assert {"finish", "coats"} & set(properties)


def test_step_four_admits_a_two_property_question_its_evidence_answers():
    router = Router()
    assert router.unsupported_terms(TWO_PROPERTY_QUESTION, [FORTE_FINISHING]) == []


def test_the_router_carries_the_terms_it_gated_on():
    """The fix itself: one definition, computed once, handed to check 6."""
    router = Router()
    decision = router.route(TWO_PROPERTY_QUESTION, [FORTE_FINISHING],
                            above_threshold=True)
    assert decision.asked_terms, "the gate's word list was not carried"
    # Not merely the winning property's synonyms, which is what check 6 used to
    # re-derive for itself and is exactly the narrowing that caused the refusal.
    compatibility = set(router.slots.terms_for("property_asked", "compatibility"))
    assert set(decision.asked_terms) - compatibility


def test_check_six_accepts_the_passage_step_four_accepted():
    """The regression proper: both halves of the gate now agree.

    Before the fix this returned "check 6: the property asked about is not in a
    cited passage" for a fully published, correctly retrieved answer — a
    reproducible over-refusal on evaluation situation S8.
    """
    router = Router()
    decision = router.route(TWO_PROPERTY_QUESTION, [FORTE_FINISHING],
                            above_threshold=True)
    answer = ("Lime Green Tradirend, Natural Finish or Finish WP need the Forte "
              "keyed and left to harden for 3 to 5 days [1].")
    failures = run_checks(answer, [FORTE_FINISHING], NAMES, decision.asked_terms)
    assert not [f for f in failures if f.startswith("check 6")], failures


def test_check_six_still_refuses_a_genuine_near_miss():
    """The gate must not have been widened into uselessness.

    A property nothing in the passage discusses still fails, which is decision
    9's whole purpose: a confident retrieval of the right product's wrong
    property is the near-miss, and this is the assertion that says the fix
    bought coverage without spending safety.
    """
    router = Router()
    question = "What is the U-value of Forte render?"
    decision = router.route(question, [FORTE_FINISHING], above_threshold=True)
    failures = run_checks("Forte has a U-value of 0.3 [1].", [FORTE_FINISHING],
                          NAMES, decision.asked_terms or
                          router.slots.terms_for("property_asked", "thermal"))
    assert [f for f in failures if f.startswith("check 6")], failures


def test_check_six_runs_when_a_decision_carries_no_terms():
    """A hand-built `Decision` must not switch a safety check off.

    The fallback in `AnswerEngine.compose` exists for exactly this: an empty
    `asked_terms` means "nobody gated", not "nothing to check", and a check that
    quietly stops running is worse than one that is too strict.
    """
    failures = run_checks("Forte has a U-value of 0.3 [1].", [FORTE_FINISHING],
                          NAMES, ["u value", "u-value", "thermal"])
    assert [f for f in failures if f.startswith("check 6")], failures


# ------------------------------------------------------- 2. the health gate

def test_the_health_gate_catches_something_landing_in_an_eye():
    """The phrasing a person actually uses, which the old pattern missed.

    The pattern required the verb *after* the word "eye", so every natural
    ordering — got it in my eye, splashed into my eyes, went in my eye — fell
    through the gate to retrieval and a full generation. The referral this
    restores is the one that names 111 and the safety data sheet.
    """
    gate = PolicyGate()
    for question in ("I got lime plaster in my eye, what should I do?",
                     "Lime splashed into my eyes",
                     "Some render went in my eye",
                     "My eyes got some lime in them"):
        matched = gate.match(question)
        assert matched and matched[0] == "health", question


def test_the_health_gate_leaves_ordinary_technical_questions_alone():
    """A gate that fires on everything has moved the failure, not fixed it."""
    gate = PolicyGate()
    for question in ("What is the coverage of Solo per bag?",
                     "How thick should the render be on an exposed wall?",
                     "What preparation does a brick background need?",
                     "Which finish coats suit Forte?"):
        assert gate.match(question) is None, question


# ------------------------------------------------ 3. the diagnosis hand-off

def test_a_diagnosis_prefers_the_document_that_explains_the_defect():
    """Measured on "patchy colour after drying" against the real index.

    The three highest-scoring passages were two coloured-render product pages
    and a checklist; the technical note publishing the mechanism sat fifth and
    was cut by the `[:3]`. Similarity and authority both push that way — a page
    selling a coloured render repeats the symptom's words, and `product_page`
    outranks `knowledge_base` — so the ordering has to be corrected where the
    hand-off chooses, not by re-scoring retrieval.
    """
    hits = [
        passage("A through-coloured render topcoat available in many colours.",
                product="Finish WP", url="https://example/finish-wp",
                document_type="product_page", authority=3, score=0.746),
        passage("A decorative coating over Forte undercoat render.",
                product="Tradirend", url="https://example/tradirend",
                document_type="product_page", authority=3, score=0.746),
        passage("Render complete elevations in a day. Work with a wet edge.",
                product="Lime Rendering Checklist", section="Application",
                url="https://example/render-checklist",
                document_type="knowledge_base", authority=4, score=0.741),
        passage("A breathable lime render for external walls.",
                product="Natural Finish", url="https://example/natural-finish",
                document_type="product_page", authority=3, score=0.728),
        passage("An even colour is the result of an even drying rate which in "
                "turn is the result of an even application thickness.",
                product="Colour & Colour Consistency", section="Curing:",
                url="https://example/colour-and-colour-consistency",
                document_type="knowledge_base", authority=4, score=0.728),
    ]
    chosen = _diagnostic_passages(hits)
    urls = [h.chunk.canonical_url for h in chosen]
    assert "https://example/colour-and-colour-consistency" in urls, urls
    assert len(chosen) == 3


def test_the_diagnosis_selection_keeps_score_order_inside_each_group():
    """A stable partition, not a re-score. The checklist still leads."""
    hits = [
        passage("marketing", url="https://example/a", document_type="product_page",
                authority=3, score=0.9),
        passage("technical, higher", url="https://example/b",
                document_type="knowledge_base", authority=4, score=0.8),
        passage("technical, lower", url="https://example/c",
                document_type="knowledge_base", authority=4, score=0.7),
    ]
    chosen = _diagnostic_passages(hits)
    assert [h.chunk.canonical_url for h in chosen] == [
        "https://example/b", "https://example/c", "https://example/a"]


def test_a_corpus_of_only_product_pages_still_prints_product_pages():
    """Nothing is invented and nothing is dropped when there is no alternative."""
    hits = [passage(f"page {i}", url=f"https://example/{i}",
                    document_type="product_page", authority=3, score=0.9 - i / 10)
            for i in range(4)]
    chosen = _diagnostic_passages(hits)
    assert [h.chunk.canonical_url for h in chosen] == [
        "https://example/0", "https://example/1", "https://example/2"]


# ------------------------------ 4. the guard must not swallow a refusal

import pytest                                                       # noqa: E402

from assistant.retrieval import candidates as cand
from assistant import ollama                                        # noqa: E402
from assistant.conversation import TurnInput                        # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.indexing.index import (  # noqa: E402
    CHUNKING_VERSION,
)
from assistant.model import DocumentVersion, Snapshot               # noqa: E402
from assistant.store import SQLiteKnowledgeRepository               # noqa: E402

# The three conditions that make the guard fire on a refusal, which is why the
# fixture is built here rather than borrowed. All three hold in the real corpus
# and none of them holds in the small two-product store the other guard tests
# use, which is precisely why the defect survived those tests.
#
#   1. the refusal quotes a passage naming a registry product;
#   2. that passage reads as advice, because datasheets are written to be
#      followed -- "you should use" is a sentence a datasheet contains;
#   3. the question's own wording does not match the registry's spelling, so
#      `resolved.product` is empty and the asked-about exemption does not apply.
#
# Condition 3 is the one that looks like an accident and is not: the site writes
# "Solo Onecoat Lime Plaster" and customers write "Solo Onecoat plaster".
CATALOGUE = "Solo Onecoat Lime Plaster"
SHEET = "https://example.invalid/solo-onecoat"
ADVICE = ("Over old lime plasters you should use Solo Onecoat Lime Plaster "
          "applied as a skim finish approximately 4mm thick.")
DIMS = 1024


def _unit(axis: int) -> list[float]:
    v = [0.0] * DIMS
    v[axis] = 1.0
    return v


@pytest.fixture
def near_miss_assistant(tmp_path, monkeypatch):
    """A store whose only passage is advice, and a question about a property it
    does not publish. The relevance gate must refuse, with the passage shown."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: _unit(0))

    def no_generation(*_a, **_k):
        raise AssertionError("a refusal must not call the model")

    monkeypatch.setattr(ollama, "generate", no_generation)

    document = Document(canonical_url=SHEET, title=f"{CATALOGUE} datasheet",
                        document_type="datasheet", authority=1,
                        product=CATALOGUE, link_text="Datasheet")
    version = DocumentVersion(canonical_url=SHEET, version=1, content_hash="h1",
                              source_path="cache/solo.pdf",
                              first_seen_at="2026-01-01",
                              fetched_at="2026-01-01T00:00:00Z",
                              checked_at="2026-01-01T00:00:00Z")
    passage_ = Chunk(canonical_url=SHEET, version=1, chunk_index=0,
                     section="Preparation & Application", content=ADVICE,
                     product=CATALOGUE, document_type="datasheet", authority=1,
                     source_date="2024-07-01", embedding=_unit(0))
    snapshot = Snapshot(
        snapshot_id="snap-near-miss", created_at="2026-01-01T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=1, chunk_count=1,
        notes={"products": [CATALOGUE], "colours": [], "merchants": [],
               "contact": {"phone": "0800 538 5746",
                           "hours": "Mon - Fri 9:00am - 5:00pm"}})
    repo = SQLiteKnowledgeRepository(tmp_path / "index" / "knowledge.db")
    repo.publish([document], [version], [passage_], snapshot, [])
    try:
        yield Assistant(repo, cache=False, log=False)
    finally:
        repo.close()


def _turn(question: str, session: str = "s1") -> TurnInput:
    turn = TurnInput(raw_question=question, turn_index=1, session_id=session)
    object.__setattr__(turn, "history", "")
    return turn


NEAR_MISS = "What is the U-value of Solo Onecoat plaster?"


def test_the_fixture_reproduces_the_conditions_the_defect_needed():
    """Guard the guard's test: if these stop holding, the test below is empty."""
    assert cand.recommends_a_product(ADVICE, [CATALOGUE]) == ["solo onecoat lime plaster"]
    assert CATALOGUE.lower() not in NEAR_MISS.lower(), (
        "the question must not name the product the registry's way, or the "
        "asked-about exemption applies and nothing is being tested")


def test_a_refusal_keeps_the_passage_it_cited(near_miss_assistant):
    """The defect: the guard discarded a correct, cited hand-off.

    A refusal prints the hand-off template plus the closest published passage,
    quoted and cited -- the value decision 9 says a refusal must carry.
    `recommends_a_product` reads text rather than provenance, so it cannot tell
    that quotation from a recommendation this system made, and the guard then
    substituted a bare refusal citing nothing.

    The trade was strictly negative: a refusal replaced by a worse refusal buys
    no safety, because a refusal is already the fail-closed outcome.
    """
    reply, _state = near_miss_assistant.ask_turn(_turn(NEAR_MISS))
    answers = [a for _q, a in reply.parts]

    assert answers and all(a.refused for a in answers)
    assert any(a.sources for a in answers), (
        "the refusal cited nothing: the guard discarded the hand-off passage")
    assert [a.diagnostics.get("step") for a in answers] == ["4"], (
        [a.diagnostics.get("step") for a in answers])


def test_the_two_orchestration_paths_refuse_the_same_way(near_miss_assistant):
    """`ask` and `ask_turn` must not disagree about the same question.

    The situations half of the canonical evaluation calls `ask`; every surface a
    customer touches -- the page, the CLI conversation -- calls `ask_turn`. A
    divergence means the evaluation is measuring a path nobody uses, which is
    exactly how this survived: situation S2 passed throughout.
    """
    direct = near_miss_assistant.ask(NEAR_MISS)
    graphed, _state = near_miss_assistant.ask_turn(_turn(NEAR_MISS, "s2"))

    assert direct.refused == graphed.refused
    assert ([a.diagnostics.get("step") for _q, a in direct.parts]
            == [a.diagnostics.get("step") for _q, a in graphed.parts])
    assert (bool([s for _q, a in direct.parts for s in a.sources])
            == bool([s for _q, a in graphed.parts for s in a.sources])), (
        "one path cited its hand-off passage and the other did not")


# ------------------------- 5. the same datasheet, indexed under two names

from assistant.retrieval.retrieve import OVERFETCH, _distinct            # noqa: E402

# The shape the real corpus has, because the site publishes two product pages
# for one product -- /products/duro and /products/duro-plaster, /products/ultra
# and /products/ultra-render -- and each links the same datasheet under a
# different filename. The crawl is faithful, so the index holds both with
# identical content hashes under two different product names. Thirty-six of 552
# passages live in a duplicated document, and they belong to the two products a
# demonstration is most likely to ask about.
APPLICATION = ("Use Duro in temperatures of 5C and rising or 30C and falling. "
               "Apply in coats of around 10 to 15mm, or thicker if dubbing out.")


def test_an_identical_passage_under_a_second_product_name_is_dropped():
    """Measured: a fifth of the evidence was a passage the model already had.

    Asked "what temperature range can Duro be applied in", the five retrieved
    passages came back as Duro's Application section at rank 1 and the *same*
    Application section, from the duplicate document, at rank 5. That is what
    `per_document_cap` exists to prevent -- one document crowding out the
    others -- defeated by the document appearing twice under two names.
    """
    hits = [
        passage(APPLICATION, product="Duro Lime Render Base Coat",
                section="Application", url="https://example/duro-tds",
                score=0.752),
        passage("Add approximately 4.5 to 5 litres of water per bag.",
                product="Duro Lime Render Base Coat", section="Mixing",
                url="https://example/duro-tds", score=0.658),
        passage(APPLICATION, product="Lime plaster", section="Application",
                url="https://example/duro-tds-1", score=0.640),
    ]
    kept = _distinct(hits)
    assert [h.chunk.canonical_url for h in kept] == [
        "https://example/duro-tds", "https://example/duro-tds"]


def test_two_sections_of_one_datasheet_both_survive():
    """Identity is the passage, not the document.

    Dropping by document would throw away the Mixing section because the
    Application section came from the same sheet, which is the opposite of what
    is wanted: a datasheet answering two halves of a question with two sections
    is the good case.
    """
    hits = [
        passage("first section", section="Mixing", url="https://example/d"),
        passage("second section", section="Application", url="https://example/d"),
    ]
    assert len(_distinct(hits)) == 2


def test_only_exact_repetition_counts_as_duplication():
    """A near-duplicate is a judgement this layer must not make.

    Silently dropping evidence on a similarity threshold is how a retrieval
    layer starts deciding what an answer may rest on. Whitespace and case are
    normalised because a PDF wraps mid-sentence; nothing else is.
    """
    hits = [
        passage("Apply in coats of around 10 to 15mm.", url="https://example/a"),
        passage("APPLY  in coats\nof around 10 to 15mm.", url="https://example/b"),
        passage("Apply in coats of around 10 to 20mm.", url="https://example/c"),
    ]
    kept = _distinct(hits)
    assert [h.chunk.canonical_url for h in kept] == [
        "https://example/a", "https://example/c"]


def test_the_repository_is_asked_wider_so_the_freed_slot_is_refilled():
    """Dropping a duplicate must not mean returning fewer passages.

    Without the over-fetch this would trade a repeated passage for an empty
    slot, which costs the model evidence rather than giving it better evidence.
    """
    assert OVERFETCH >= 2


# ------------------- 6. the relevance gate's second chance, finally called

COVERAGE_SECTION = ("Store in a dry, draft free area free from any damp. Shelf "
                    "life is 6 months. Use bags within 2 days once open. Each "
                    "bag will cover approximately 1.5m2 at 10mm thick, or 3m2 "
                    "at 5mm thick.")
DESCRIPTION = ("Solo is a one-coat lime plaster for interior use on masonry, "
               "undercoats and boards, designed to be labour saving.")
SOLO_SHEET = "https://example.invalid/solo-tds"


@pytest.fixture
def coverage_assistant(tmp_path, monkeypatch):
    """A store where the passage that answers the question cannot be reached
    by similarity, because its embedding points the other way.

    That is the real corpus's shape, not a contrivance. The Solo datasheet
    publishes "Each bag will cover approximately 1.5m2 at 10mm thick, or 3m2 at
    5mm thick" in a section headed "Storage / Coverage" whose text is mostly
    about storage -- shelf life, keeping bags dry -- so a bare coverage question
    embeds away from it. Measured against the built index it ranked *tenth*,
    below the same datasheet's Disclaimer and Finishing sections, and never
    entered the five passages the answer sees.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: _unit(0))

    def stub(*_a, **_k):
        return ("Each bag will cover approximately 1.5m2 at 10mm thick, "
                "or 3m2 at 5mm thick [2]."), 0.01

    monkeypatch.setattr(ollama, "generate", stub)

    documents_extra: list = []
    versions_extra: list = []
    document = Document(canonical_url=SOLO_SHEET, title="Solo datasheet",
                        document_type="datasheet", authority=1,
                        product="Solo", link_text="Datasheet")
    version = DocumentVersion(canonical_url=SOLO_SHEET, version=1,
                              content_hash="h1", source_path="cache/solo.pdf",
                              first_seen_at="2026-01-01",
                              fetched_at="2026-01-01T00:00:00Z",
                              checked_at="2026-01-01T00:00:00Z")
    chunks = [
        # Reachable: the query embeds onto this one.
        Chunk(canonical_url=SOLO_SHEET, version=1, chunk_index=0,
              section="Description", content=DESCRIPTION, product="Solo",
              document_type="datasheet", authority=1,
              source_date="2024-07-01", embedding=_unit(0)),
        # Unreachable by similarity, and the only passage that answers.
        Chunk(canonical_url=SOLO_SHEET, version=1, chunk_index=1,
              section="Storage / Coverage", content=COVERAGE_SECTION,
              product="Solo", document_type="datasheet", authority=1,
              source_date="2024-07-01", embedding=_unit(7)),
    ]
    # Enough other material to score between the two, so the coverage passage
    # falls outside the window the answer sees. Without this the corpus is
    # small enough that everything is retrieved, the gate is satisfied by the
    # passage it was supposed to have missed, and the test proves nothing --
    # which is what the first version of it did.
    #
    # None of these mentions covering anything, and all belong to other
    # products, so they can neither satisfy the gate nor be reached by a lookup
    # scoped to Solo.
    middling = [0.0] * DIMS
    middling[0], middling[1] = 0.5, 0.8660254
    for i, other in enumerate(("Duro", "Forte", "Ultra", "Tradirend",
                               "Natural Finish"), 2):
        url = f"https://example.invalid/{other.lower().replace(' ', '-')}"
        documents_extra.append(
            Document(canonical_url=url, title=f"{other} datasheet",
                     document_type="datasheet", authority=1, product=other,
                     link_text="Datasheet"))
        versions_extra.append(
            DocumentVersion(canonical_url=url, version=1,
                            content_hash=f"h{i}", source_path=f"cache/{i}.pdf",
                            first_seen_at="2026-01-01",
                            fetched_at="2026-01-01T00:00:00Z",
                            checked_at="2026-01-01T00:00:00Z"))
        chunks.append(
            Chunk(canonical_url=url, version=1, chunk_index=0,
                  section="Description",
                  content=f"{other} is a lime product for masonry backgrounds.",
                  product=other, document_type="datasheet", authority=1,
                  source_date="2024-07-01", embedding=list(middling)))
    snapshot = Snapshot(
        snapshot_id="snap-coverage", created_at="2026-01-01T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=1,
        chunk_count=len(chunks),
        notes={"products": ["Solo", "Duro", "Forte", "Ultra", "Tradirend",
                            "Natural Finish"], "colours": [], "merchants": [],
               "contact": {"phone": "0800 538 5746",
                           "hours": "Mon - Fri 9:00am - 5:00pm"}})
    repo = SQLiteKnowledgeRepository(tmp_path / "index" / "knowledge.db")
    repo.publish([document] + documents_extra, [version] + versions_extra,
                 chunks, snapshot, [])
    try:
        yield Assistant(repo, cache=False, log=False)
    finally:
        repo.close()


def _find_property_calls(assistant, question: str) -> list[tuple]:
    """Every term tuple the targeted lexical lookup was asked for.

    Spying on the seam rather than on the answer, because three different
    passes call `find_property` and the question being pinned is *which one
    ran*. They are distinguishable by shape: `_missed_evidence` asks for one
    rare word at a time, the calculation edge asks for `COVERAGE_TERMS`, and the
    relevance gate's second chance asks for the whole term list the gate refused
    on. Only the last of those carries several terms for a question with no
    calculation words in it.
    """
    calls: list[tuple] = []
    original = assistant.retriever.find_property

    def spy(product, terms, audiences=("public",), limit=3):
        calls.append(tuple(terms))
        return original(product, terms, audiences=audiences, limit=limit)

    assistant.retriever.find_property = spy
    try:
        assistant.ask(question)
    finally:
        assistant.retriever.find_property = original
    return calls


def test_the_gate_second_chance_looks_up_the_terms_it_refused_on(
        coverage_assistant):
    """`Router.unsupported_terms` existed to offer this and nothing called it.

    Its own docstring names the case exactly -- "semantic retrieval having
    failed to surface the term is exactly the moment a lexical lookup is worth
    doing" -- and it was dead code in the product, reachable only from a test.

    Neither existing pass covers it. The targeted coverage lookup fires only on
    calculation words, and "what is the coverage of Solo" has none. And
    `_missed_evidence` declines because its rule is rarity and, scoped to
    "solo", "coverage" comes back over the distinctiveness cap -- it matches
    Solo Primer, Solo Mesh and Solo Filler too. Both rules are right about
    themselves and the answer still went missing between them. Measured on the
    built index, the question refused at step 4 while the Solo datasheet
    publishes "Each bag will cover approximately 1.5m2 at 10mm thick".
    """
    calls = _find_property_calls(coverage_assistant,
                                 "What is the coverage of Solo?")
    gated = [t for t in calls if len(t) > 1 and any("cover" in x for x in t)]
    assert gated, (
        f"the gate's terms were never looked up lexically; calls were {calls}")


def test_the_second_chance_does_not_rescue_a_genuine_absence(
        coverage_assistant):
    """The near-miss must still refuse, or the fix has spent safety for reach.

    Nothing in this corpus states a U-value, so the lexical lookup finds
    nothing and step 4 fires exactly as before. This is the assertion that
    separates "recovered a published fact the embedding missed" from "loosened
    the gate".
    """
    reply = coverage_assistant.ask("What is the U-value of Solo?")
    answers = [a for _q, a in reply.parts]

    assert all(a.refused for a in answers)
    assert [a.diagnostics.get("step") for a in answers] == ["4"], (
        [a.diagnostics.get("step") for a in answers])


def test_the_second_chance_is_scoped_to_the_product_that_was_named(
        coverage_assistant):
    """A question naming no product gets no lexical sweep from this pass.

    An unscoped lookup for "coverage" returns every datasheet's coverage
    section, which is the cross-product contamination this system spends a
    check catching. A question naming no product and retrieving nothing for its
    term is one the gate should refuse.
    """
    calls = _find_property_calls(coverage_assistant, "What is the coverage?")
    gated = [t for t in calls if len(t) > 1 and any("cover" in x for x in t)]
    assert not gated, (
        f"a question naming no product ran a scoped lexical sweep: {calls}")
