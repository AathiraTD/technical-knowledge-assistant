"""The router: the part of the system that is deliberately not a model.

Every branch below is a rule a person can read and disagree with, so every
branch below is tested. The ordering is itself a decision, and the tests assert
the order holds when several conditions are true at once, because that is the
only situation where precedence means anything.

Runs without Ollama and without an index. Passages are constructed by hand, and
a router decision is a pure function of a question, some passages and a
threshold verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.model import Chunk, Document, Retrieved       # noqa: E402
from assistant.router import (                               # noqa: E402
    Decision,
    Path_,
    PolicyGate,
    Router,
    SlotDetector,
    split_by_topic,
)


def hit(content: str, url: str = "https://example/solo", section: str = "Mixing",
        product: str = "Solo Onecoat Lime Plaster", dtype: str = "datasheet",
        authority: int = 1, score: float = 0.8) -> Retrieved:
    return Retrieved(
        chunk=Chunk(canonical_url=url, version=1, chunk_index=0, section=section,
                    content=content, product=product, document_type=dtype,
                    authority=authority),
        score=score,
        document=Document(canonical_url=url, title=product, document_type=dtype,
                          authority=authority, product=product,
                          link_text=f"{product} datasheet"),
    )


SOLO = hit("Add between 5 and 6 litres of clean water per 25kg sack.")
COVERAGE = hit("Coverage is approximately 2 m2 per 25 kg sack at 10 mm.",
               section="Coverage")
OTHER = hit("Duro is a general purpose lime undercoat plaster.",
            url="https://example/duro", product="Duro Lime Plaster",
            section="Description")
DEFERRING = hit("Many MgO boards are not suitable for Solo. Please contact us "
                "for further information in writing before proceeding.")


def route(question: str, hits=None, above: bool = True, audiences=("public",)
          ) -> Decision:
    return Router().route(question, hits if hits is not None else [SOLO],
                          above, audiences)


# ------------------------------------------------------------------ splitting


def test_two_questions_in_one_message_are_two_questions():
    """A price and a coverage must take different paths; answered as one, one is wrong."""
    parts = split_by_topic("How much does Solo cost? And what coverage do I get?")
    assert len(parts) == 2
    assert "cost" in parts[0]
    assert "coverage" in parts[1]


def test_two_properties_of_one_product_stay_in_one_question():
    """This used to split, and splitting was the wrong answer.

    "What is the coverage and how long does it take to dry" is one enquiry
    about one product. Cutting it produced two retrievals against half a
    question each, and the guarantee this test used to assert — that the
    question word "how long" survived the cut — was a repair to damage the cut
    caused. Not cutting is the stronger fix, and both properties are now
    detected on the whole question instead of one being thrown away.
    """
    parts = split_by_topic("What is the coverage and how long does it take to dry?")
    assert len(parts) == 1, parts

    properties = SlotDetector().detect_properties(parts[0])
    assert set(properties) >= {"coverage", "drying"}, properties


def test_a_single_question_is_left_alone():
    assert split_by_topic("Can I use it outside") == ["Can I use it outside"]


def test_a_message_that_splits_into_fragments_falls_back_to_itself():
    """Two-word fragments retrieve nothing; better to route the whole message."""
    assert split_by_topic("Why? Ok.") == ["Why? Ok."]


# --------------------------------------------------------------- policy gate


def test_price_never_reaches_retrieval():
    """Retrieval would find a passage mentioning bags and the model would try to help."""
    matched = PolicyGate().match("How much does a bag of Solo cost?")
    assert matched is not None and matched[0] == "price"


def test_each_policy_topic_matches_something():
    """A topic in the table that nothing can trigger is a topic that does not exist."""
    gate = PolicyGate()
    questions = {
        "price": "What is the price of Duro?",
        "stock": "Do you have Solo in stock?",
        "delivery": "How is delivery arranged?",
        "where_to_buy": "Where can I buy Solo?",
        "colour_matching": "Can you match the colour of my existing render?",
        "warranty": "Is there a warranty on Warmshell?",
        "structural_judgement": "Is my load bearing wall safe?",
        "compliance_signoff": "Can you certify my build?",
        "health": "I swallowed some lime, what do I do?",
        "complaint": "The product was faulty and I want a refund",
        "document_request": "Can you send me the safety data sheet?",
    }
    for topic, question in questions.items():
        matched = gate.match(question)
        assert matched is not None, f"{topic} matched nothing"


def test_a_technical_question_passes_the_gate():
    """The gate must not swallow the questions the system exists to answer."""
    assert PolicyGate().match("How much water does Solo need per bag?") is None


def test_the_document_request_topic_is_answered_from_the_manifest():
    _topic, spec = PolicyGate().match("Which datasheet covers Solo?")
    assert spec.get("from_manifest") is True


# ------------------------------------------------------------ slot detection


def test_the_longer_match_wins_over_the_earlier_one():
    """'How much water ... per bag' read as coverage, because coverage is declared first."""
    assert SlotDetector().detect(
        "How much water does Solo need per bag?")["property_asked"] == "water"


def test_substrate_and_location_are_both_detected():
    slots = SlotDetector().detect("Can I use it on lath inside?")
    assert slots["substrate"] == "lath"
    assert slots["location"] == "internal"


def test_a_question_with_no_vocabulary_match_has_no_slots():
    assert SlotDetector().detect("Hello there please") == {}


def test_the_load_bearing_slots_are_the_two_named_in_the_decision():
    assert set(SlotDetector().load_bearing()) == {"substrate", "location"}


def test_terms_for_returns_the_synonyms_the_gate_compares_against():
    terms = SlotDetector().terms_for("property_asked", "coverage")
    assert "coverage" in terms and "per sack" in terms
    assert SlotDetector().terms_for("property_asked", "nonsense") == []


# ------------------------------------------------------- the asked-for phrase


def test_an_unlisted_property_is_still_recognised():
    """'pot life' is in no vocabulary, and without this no relevance gate ran at all."""
    assert Router().asked_phrase("What is the pot life of Duro?") == "pot life"


def test_the_how_much_form_is_recognised():
    assert Router().asked_phrase("How much shrinkage does Solo have?") == "shrinkage"


def test_the_of_form_captures_the_whole_property_including_its_noun():
    """'fire rating' is the property; 'fire' alone would match any mention of fire."""
    assert Router().asked_phrase(
        "What is the fire rating of Warmshell?") == "fire rating"


def test_the_bare_rating_form_is_recognised():
    """A question with no 'of' still names a property, and still needs a gate."""
    assert Router().asked_phrase("Tell me the fire rating") == "fire"


def test_a_question_with_no_obvious_phrase_yields_none():
    assert Router().asked_phrase("Can I plaster a fireplace?") == ""


def test_the_whole_phrase_is_required_not_its_head_noun():
    """'pot life' reduced to 'life' matches 'shelf life', and the gate stops working."""
    _asked, terms = Router()._asked_terms("What is the pot life of Duro?", {})
    assert "life" not in terms
    assert "pot life" in terms and "pot-life" in terms


def test_a_vocabulary_slot_takes_precedence_over_the_phrase():
    asked, terms = Router()._asked_terms(
        "What is the coverage of Duro?", {"property_asked": "coverage"})
    assert asked == "coverage"
    assert "spread rate" in terms


# ------------------------------------------------- the ordered router, step by step


def test_step_1_below_threshold_refuses():
    """Nothing may generate below the threshold, whatever else the question looks like."""
    d = route("How much water does Solo need?", [SOLO], above=False)
    assert d.path is Path_.REFUSE and d.step == "1"


def test_step_1_fires_on_no_hits_at_all():
    d = route("Anything at all?", [], above=True)
    assert d.path is Path_.REFUSE and d.step == "1"


def test_step_2_a_published_deferral_beats_a_computed_answer():
    """The company has already decided this needs a person; answering around it is worse."""
    d = route("How much Solo for an MgO board?", [DEFERRING])
    assert d.path is Path_.DEFER and d.step == "2"


def test_step_2_outranks_the_calculation_branch():
    """Both conditions hold at once, and precedence is the whole point of an ordered router."""
    d = route("How many bags of Solo do I need for MgO board?", [DEFERRING])
    assert d.path is Path_.DEFER


def test_step_3_a_cause_question_goes_to_diagnosis():
    """Judging why a wall failed is the technical team's call, not a retrieval result."""
    d = route("Why is my render cracking?", [SOLO])
    assert d.path is Path_.DIAGNOSIS and d.step == "3"


def test_step_4_the_near_miss_refuses():
    """Right product, wrong property: a confident retrieval is not an answer."""
    d = route("What is the U-value of Solo?", [SOLO])
    assert d.path is Path_.REFUSE and d.step == "4"
    assert d.missing_term == "thermal"


def test_step_4_passes_when_the_property_is_actually_present():
    d = route("What is the coverage of Solo?", [COVERAGE, SOLO])
    assert d.path is not Path_.REFUSE


def test_step_5_an_uncued_substrate_asks_back():
    """A recommendation on an assumed wall is the costly error the design exists to avoid."""
    d = route("Which plaster should I use?", [OTHER])
    assert d.path is Path_.ASK_BACK and d.step == "5"


def test_step_5_does_not_fire_when_the_substrate_is_given():
    d = route("Which plaster should I use on lath?", [OTHER])
    assert d.path is not Path_.ASK_BACK


def test_step_5_does_not_fire_on_a_factual_lookup():
    """Asking how much water Solo needs does not require knowing the wall."""
    d = route("How much water does Solo need?", [SOLO])
    assert d.path is not Path_.ASK_BACK


def test_step_5_does_not_fire_on_a_property_of_an_already_chosen_product():
    """"What thickness should I use" is not "which product should I use".

    The phrase "should I use" reads both ways. With the product already named
    and a published figure asked for, treating it as a choice asked for a
    substrate before printing two numbers the datasheet states unconditionally.
    """
    d = Router().route(
        "I'm using Ultra. What thickness and mixing water should I use?",
        [SOLO], True, ("public",), {"product": "ultra"})
    assert d.path is not Path_.ASK_BACK
    assert d.step != "5"


@pytest.mark.parametrize("question", [
    "Is Ultra suitable for my wall?",
    "Is Ultra suitable for my wall, and what thickness?",
    "Which plaster should I use?",
    "What product do I need for my wall?",
])
def test_substrate_stays_load_bearing_for_a_choice_or_a_suitability_question(question):
    """The exemption covers a published property, not a judgement about a wall.

    Asserted on the predicate rather than through `route`, because step 4 fires
    first on some of these and step 5 is then never reached -- which is correct
    precedence, and would hide what this is checking.
    """
    assert Router()._needs_substrate(question, {"product": "ultra"}) is True


def test_step_5_still_fires_when_a_choice_is_asked_with_a_product_in_memory():
    """A carried product must not exempt an actual request to choose."""
    for question in ("Which plaster should I use?",
                     "What product do I need for my wall?",
                     "Can you recommend a render for my wall?"):
        d = Router().route(question, [OTHER], True, ("public",),
                           {"product": "ultra"})
        assert d.path is Path_.ASK_BACK and d.step == "5", question


def test_step_6_a_quantity_question_extracts_and_refuses_the_sum():
    """The arithmetic depends on background and thickness, so the system prints, not multiplies."""
    d = route("How many bags do I need for 20 m2 of coverage?", [COVERAGE])
    assert d.path is Path_.EXTRACT and d.step == "6"
    assert d.sum_refused is True


def test_step_7_one_document_and_a_factual_ask_prints_the_passage():
    """On a lookup the model can contribute nothing but paraphrase drift."""
    d = route("What is the coverage of Solo?", [COVERAGE])
    assert d.path is Path_.EXTRACT and d.step == "7"


def test_step_8_several_documents_compose():
    d = route("What is the coverage of Solo?", [COVERAGE, OTHER])
    assert d.path is Path_.COMPOSE and d.step == "8"


def test_staff_see_passages_rather_than_prose():
    """The advisor verifies from the passage text; composing for staff is roadmap."""
    d = route("Tell me about Solo and Duro", [SOLO, OTHER], audiences=("staff",))
    assert d.path is Path_.EXTRACT and d.step == "7s"


# ---------------------------------------------------------- carried conditions


def test_the_photograph_flag_survives_every_path():
    """The cannot-see-photographs line keys on the slot, not the path taken."""
    for hits, above in (([SOLO], False), ([DEFERRING], True), ([SOLO, OTHER], True)):
        d = route("Here is a photo of my wall, what do I do?", hits, above)
        assert d.photograph is True, d.path


def test_an_uncued_location_answers_per_option():
    """Inside and outside split the datasheets anyway, and both fit in five passages."""
    d = route("Which plaster should I use on lath?", [SOLO, OTHER])
    assert d.per_option is True


def test_a_cued_location_does_not_answer_per_option():
    d = route("Which plaster should I use on lath inside?", [SOLO, OTHER])
    assert d.per_option is False


def test_every_decision_carries_its_reason():
    """A path with no stated reason cannot be audited from a transcript."""
    for question, hits, above in (
        ("How much water does Solo need?", [SOLO], False),
        ("How much Solo for MgO?", [DEFERRING], True),
        ("Why is it cracking?", [SOLO], True),
        ("What is the U-value of Solo?", [SOLO], True),
        ("Which plaster should I use?", [OTHER], True),
        ("How many bags for 20 m2 coverage?", [COVERAGE], True),
        ("What is the coverage of Solo?", [COVERAGE], True),
        ("What is the coverage of Solo?", [COVERAGE, OTHER], True),
    ):
        d = route(question, hits, above)
        assert d.reason and d.step, f"{question} produced a bare decision"


def test_the_deferral_matcher_ignores_ordinary_prose():
    """'Contact' in a passage about contact time is not a referral to a person."""
    router = Router()
    assert router._defers("Please contact us for further information")
    assert not router._defers("Allow a contact time of ten minutes.")


def test_single_document_detection():
    router = Router()
    assert router._single_document([SOLO, COVERAGE]) is True
    assert router._single_document([SOLO, OTHER]) is False


# ------------------------------------------------- the remaining edge branches


def test_a_comment_key_in_the_vocabulary_is_not_treated_as_a_slot():
    """The configuration is hand-written JSON; a comment added to it must not become a slot."""
    detector = SlotDetector()
    detector.spec = dict(detector.spec)
    detector.spec["_comment"] = {"values": {"oops": ["brick"]}}
    found = detector.detect("a brick wall inside")
    assert "_comment" not in found
    assert found["substrate"] == "brick"


def test_a_phrase_of_only_noise_words_yields_nothing():
    """'What is the the of it' must not produce an empty gate term that refuses everything."""
    assert Router().asked_phrase("What is the the of Solo?") == ""


def test_a_single_word_phrase_gets_no_hyphen_variants():
    """Hyphenation only matters for multi-word phrases; 'shrinkage-' is not a spelling."""
    _asked, terms = Router()._asked_terms("How much shrinkage does Solo have?", {})
    assert terms == ["shrinkage"]


# ------------------------------------- the substrate rule, narrowed to a wall


def test_a_catalogue_question_is_answered_rather_than_asked_back():
    """The brief's own worked example used to meet a clarifying question."""
    d = route("What products are suitable for lime-based external finishes?",
              [SOLO, OTHER])
    assert d.path is not Path_.ASK_BACK, d.reason


def test_a_question_about_the_asker_s_own_wall_still_asks_back():
    """A recommendation for a specific job without a substrate is a guess."""
    assert route("Which plaster should I use?", [OTHER]).path is Path_.ASK_BACK
    assert route("What do I need for my wall?", [OTHER]).path is Path_.ASK_BACK


def test_a_described_symptom_counts_as_a_question_about_a_wall():
    """Nobody describes crazing about a product range in the abstract."""
    d = route("Which render is suitable where the surface is spalling?", [OTHER])
    assert d.path in (Path_.ASK_BACK, Path_.DIAGNOSIS), d.path


def test_a_photograph_counts_as_a_question_about_a_wall():
    d = route("I attached a photo, which plaster is suitable?", [OTHER])
    assert d.path is Path_.ASK_BACK


def test_a_factual_lookup_never_asks_for_a_substrate():
    assert route("How much water does Solo need?", [SOLO]).path is not Path_.ASK_BACK


# --------------------------------------- the five questions an external review asked

# Five real enquiries, run against the built system by a reviewer who then read
# the code to explain what happened. Four of the five failed before retrieval
# ever ran, for reasons that had nothing to do with the model: the message was
# cut into fragments that had lost the facts making them answerable, and the
# vocabulary could not name what was being asked. These pin the repairs.


FIVE = {
    "compatibility": "Can I use Lime Green Solo directly over old gypsum "
                     "plaster, or do I need Solo Primer first?",
    "insulation": "I have an old solid brick wall and want to improve "
                  "insulation without dry-lining it. Would Lime Green Ultra be "
                  "suitable internally, and what thickness can it be applied at?",
    "quantity": "How much Lime Green Ultra would I need for 30 m\u00b2 at 25 mm "
                "thickness?",
    "patchy": "My external lime render is showing patchy colour after drying. "
              "What could be causing it?",
    "exposed": "I'm rendering an old masonry wall in a very exposed location. "
               "How thick should the lime render be, and what preparation does "
               "the background need?",
}


@pytest.mark.parametrize("name", sorted(FIVE))
def test_a_real_enquiry_is_not_cut_into_fragments(name):
    """Each of these is one job, however many sentences it takes to say.

    The splitter cut after every full stop, so "What could be causing it?"
    arrived at retrieval with no render, no colour and no exposure — a question
    that cannot be answered and cannot even be honestly refused, because
    nothing downstream could tell what had been asked.
    """
    assert len(split_by_topic(FIVE[name])) == 1, split_by_topic(FIVE[name])


def test_dry_lining_is_not_a_question_about_drying():
    """"dry" matched inside "dry-lining", because a hyphen is a word boundary.

    Someone explaining they do *not* want to dry-line a wall was read as asking
    how long something takes to dry, which then steered retrieval.
    """
    slots = SlotDetector().detect(FIVE["insulation"])
    assert slots.get("property_asked") != "drying", slots
    assert slots["substrate"] == "brick"
    assert slots["location"] == "internal"


def test_the_symbol_people_actually_type_is_a_quantity_question():
    """The vocabulary had "m2" and "sq m" but not "m²", which is what a keyboard
    with a UK layout produces and what the reviewer typed. Without it the
    calculation slot never fired and router step 6 was unreachable, so a
    quantity question became an ordinary thickness lookup."""
    assert SlotDetector().detect(FIVE["quantity"])["calculation"] == "quantity"
    assert SlotDetector().detect(
        "How many bags of Duro for 20 square metres")["calculation"] == "quantity"


def test_the_ordinary_way_of_asking_for_a_cause_is_recognised():
    """Step 3 sends a cause question to diagnosis, and depends on this slot.

    The vocabulary knew "what caused" and "what is causing" but not "what could
    be causing", which is how people actually write it.
    """
    slots = SlotDetector().detect(FIVE["patchy"])
    assert slots["cause_asked"] == "cause"
    assert slots["location"] == "external"


def test_masonry_is_not_resolved_to_stone():
    """A masonry wall may be brick, block, stone or mixed.

    Resolving it to stone made the system more certain than the caller had
    been, and a recommendation on an assumed wall is the costly error decision
    10 exists to prevent.
    """
    slots = SlotDetector().detect(FIVE["exposed"])
    assert slots["substrate"] == "masonry", slots
    assert slots["exposure"] == "severe"


def test_both_halves_of_a_two_property_question_survive():
    """Thickness *and* preparation. Keeping only the winner dropped half the job.

    The corpus publishes a whole knowledge-base article on background
    preparation for lime rendering, so the half being discarded was the half
    with the best evidence behind it.
    """
    properties = SlotDetector().detect_properties(FIVE["exposed"])
    assert set(properties) >= {"preparation", "thickness"}, properties


def test_asking_whether_one_product_goes_over_another_is_a_property():
    """There was no way to name compatibility, so the question had no shape.

    It still may be refused — the corpus may genuinely not say whether Solo
    goes over old gypsum — but it should be refused after looking for the right
    thing, not because the enquiry was never understood.
    """
    slots = SlotDetector().detect(FIVE["compatibility"])
    assert slots.get("property_asked") == "compatibility", slots


def test_two_different_policy_topics_are_two_referrals():
    """A price and a delivery question get different fixed replies.

    Merging them would print one referral and silently drop the other, so this
    is the second case where splitting is still right — both halves are gated,
    but to different topics.
    """
    parts = split_by_topic("How much does Solo cost? When will it be delivered?")
    assert len(parts) == 2, parts


def test_two_questions_on_the_same_policy_topic_stay_together():
    """One referral answers both, so cutting gains nothing and costs context."""
    parts = split_by_topic("How much does Solo cost? And how much is Duro?")
    assert len(parts) == 1, parts
