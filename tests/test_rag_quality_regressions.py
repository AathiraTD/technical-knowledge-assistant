"""Regressions for three retrieval-quality defects found by the gold set.

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
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answer import _diagnostic_passages, run_checks   # noqa: E402
from assistant.model import Chunk, Document, Retrieved          # noqa: E402
from assistant.router import PolicyGate, Router                 # noqa: E402


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
