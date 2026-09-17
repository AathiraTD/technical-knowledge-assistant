"""Claim-to-evidence binding, and the substrate word the corpus actually uses.

The defect these pin was measured against the real 94-document index before it
was fixed. Asked

    "I have a solid brick wall internally. Can I use Ultra, and what thickness
     should it be applied at?"

the pipeline named Ultra, resolved substrate=brick and location=internal,
retrieved five passages across three documents at 0.686, routed to compose --
and was refused, because the model wrote

    "Ultra is suitable for most masonry backgrounds including brick [2]."

against the Ultra datasheet's *How to Apply* section, which states a thickness
and says nothing about suitability. Check 1 was right to refuse it. The
suitability evidence was in the evidence set the whole time, at marker [4]: the
Ultra product page's "Suitable for most masonry and lath backgrounds".

Two things were missing and both are deterministic:

1. **Nothing bound a claim to its evidence.** The router already knew the
   question asked about *compatibility* and *thickness* -- `detect_properties`
   returns both -- and then flattened the two vocabularies into one list for the
   relevance gate, discarding which passage states which half.

2. **Nothing related "brick" to "masonry".** No Ultra document contains the word
   "brick". The only thing available to bridge the caller's word and the
   corpus's was the model, which bridged it inside a cited sentence and so
   reported an inference as published text.

A third, smaller gap sat behind both: "Can Ultra be applied internally on solid
brick?" is plainly a suitability question and matched no compatibility term at
all, so no property was detected, no gate ran, and there was nothing to bind.

None of the six checks changed. The binding removes the opportunity to misbind;
it is not what catches a misbinding, and `test_the_binding_cannot_make_an_
unsupported_clause_print` is the test that says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answer import (                                   # noqa: E402
    _binding_guidance, _distinguishes, evidence_binding, promote_bound,
    run_checks,
)
from assistant.model import Chunk, Document, Retrieved           # noqa: E402
from assistant.repository import product_matches                 # noqa: E402
from assistant.router import Path_, Router                       # noqa: E402

COMPOUND = ("I have a solid brick wall internally. Can I use Ultra, and what "
            "thickness should it be applied at?")
NARROW = "Can Ultra be applied internally on solid brick?"
THICKNESS = "What is the application thickness of Ultra?"

ULTRA = "Ultra Insulating Lime Render Base Coat"
ULTRA_PAGE = "Ultra Insulated Lime Render Base Coat"


def passage(content: str, *, section: str, product: str = ULTRA,
            url: str = "https://example/ultra-tds", title: str = "Data Sheet",
            document_type: str = "datasheet", authority: int = 1,
            score: float = 0.6) -> Retrieved:
    return Retrieved(
        chunk=Chunk(canonical_url=url, version=1, chunk_index=0, section=section,
                    content=content, product=product,
                    document_type=document_type, authority=authority),
        score=score,
        document=Document(canonical_url=url, title=title,
                          document_type=document_type, authority=authority,
                          product=product, link_text=title),
    )


# The five passages the live index returns for the compound question, in the
# order it returns them, with the datasheet's own wording.
APPLY = passage(
    "Only use Ultra above 5 o C and below 30 o C. Ultra should be applied to a "
    "straight, flat, dampened surface in a uniform thickness of between 10 and "
    "30mm, make sure this coat is keyed.",
    section="How to Apply", score=0.686)

GENERAL = passage(
    "Lime Green Ultra is a thermally insulating lime basecoat for rendering and "
    "plastering. It can be built up in thick coats where insulation is wanted.",
    section="Product Data Sheet: Ultra / General Information", score=0.634)

INSULATING_PAGE = passage(
    "Lime Green Ultra is a thermally insulating lime basecoat that can be used "
    "for rendering and plastering over masonry, and gives useful depth of "
    "insulation.",
    section="Ultra: Insulating Lime Render Base Coat", product=ULTRA,
    url="https://example/ultra-render", title=ULTRA,
    document_type="product_page", authority=3, score=0.630)

# The passage that actually answers the suitability half.
INSULATED_PAGE = passage(
    "An insulating lime undercoat plaster. Suitable for most masonry and lath "
    "backgrounds. This makes it the perfect plastering base coat for building "
    "renovations, where it gives an additional insulation layer to internal "
    "walls. Masonry and wooden laths. Can be built up thick.",
    section="Ultra: Insulated Lime Plaster Base Coat", product=ULTRA_PAGE,
    url="https://example/ultra-plaster", title=ULTRA_PAGE,
    document_type="product_page", authority=3, score=0.611)

MIX = passage(
    "Ultra can be mixed using a drum mixer, or in a tub using a plasterer's "
    "paddle mixer. Mix with approximately 4 to 4.5 litres of clean water per bag.",
    section="How to Mix", score=0.598)

# Not about Ultra at all. Present so that "no unrelated Warmshell passage is
# offered as evidence" is something a test can observe rather than assume.
WARMSHELL = passage(
    "The Warmshell internal wall insulation system is applied over woodfibre "
    "board at a thickness of 6mm and is suitable for solid walls.",
    section="Warmshell IWI", product="Warmshell system",
    url="https://example/warmshell", title="Natural internal wall insulation",
    document_type="product_page", authority=3, score=0.520)

HITS = [APPLY, GENERAL, INSULATING_PAGE, INSULATED_PAGE, MIX]

NAMES = {"products": [ULTRA, ULTRA_PAGE, "Ultra", "Warmshell system"],
         "colours": [], "merchants": [],
         "contact": {"phone": "0800 538 5746",
                     "hours": "Mon - Fri 9:00am - 5:00pm"}}


def decide(question: str, hits=None, product: str = "ultra"):
    """Route a question the way the engine does, and hand back the decision.

    The product is carried in, because that is what both orchestration paths
    do: the graph passes `ResolvedRequest.slots()`, which holds it, and
    `Assistant._answer_part` writes its own detection onto the decision.
    """
    return Router().route(question, hits if hits is not None else HITS,
                          above_threshold=True,
                          carried={"product": product} if product else None)


def names_of(failures) -> set[str]:
    return {f.split(":", 1)[0] for f in failures}


# ------------------------------------------------- 1. the compound question

def test_the_compound_question_binds_each_half_to_its_own_passage():
    """The defect, stated as the binding it was missing.

    Suitability is stated in one passage and one only; the thickness figure in
    another. Both are named, and neither names the other's.
    """
    decision = decide(COMPOUND)
    assert decision.path is Path_.COMPOSE, decision
    bound = evidence_binding(decision)

    assert bound == {"compatibility": [4], "thickness": [1]}, bound


def test_the_binding_is_the_smallest_supporting_set_not_every_related_mention():
    """Four of the five passages contain a thickness word. One states a thickness.

    "thickness of between 10 and 30mm" states it; "built up in thick coats",
    "useful depth of insulation" and "can be built up thick" mention it. Offering
    all four is property grouping, and it leaves the model exactly as free to
    misbind as it was before.
    """
    decision = decide(COMPOUND)
    thickness_terms = decision.evidence_terms["thickness"]
    mentions = [i for i, hit in enumerate(HITS, 1)
                if any(t in f"{hit.chunk.section} {hit.chunk.content}".lower()
                       for t in thickness_terms)]

    assert len(mentions) >= 4, mentions
    assert evidence_binding(decision)["thickness"] == [1]


def test_no_unrelated_product_is_offered_as_evidence_for_ultra():
    """A Warmshell passage in the evidence set is never bound to an Ultra claim.

    This is the case that proves lexical ranking alone is not enough, and it was
    written before the rule that handles it. The Warmshell fixture says "applied
    over woodfibre board at a thickness of 6mm and is suitable for solid walls":
    it matches both vocabularies, and it states each property beside a figure,
    so directness and specificity both rank it **above** the Ultra product page
    whose "Suitable for most masonry and lath backgrounds" carries no number.
    A binding built on the words alone therefore pointed a suitability claim
    about Ultra at a Warmshell passage -- check 3's cross-product attribution,
    arranged by the prompt rather than caught afterwards.
    """
    hits = HITS + [WARMSHELL]
    decision = decide(COMPOUND, hits)
    bound = evidence_binding(decision)

    warmshell_marker = len(hits)
    assert all(warmshell_marker not in markers for markers in bound.values()), bound
    for markers in bound.values():
        for marker in markers:
            assert "ultra" in hits[marker - 1].chunk.product.lower()


def test_evidence_outside_the_named_product_is_still_reachable():
    """Product scoping prefers; it does not exclude.

    A finish coat's hardening time is published on the base coat's datasheet,
    and the brief asks for exactly those multi-document answers. So when the
    named product publishes nothing for a property, the binding falls back to
    whatever does rather than going silent.
    """
    curing = passage(
        "Leave the Forte to set for 3 to 5 days and lightly re-dampen it "
        "before applying Tradirend.",
        section="Finishing Coats", product="Forte Render Base Coat",
        url="https://example/forte", title="Forte Data Sheet")
    decision = decide("How long does Tradirend need to cure?", [curing],
                      product="tradirend")

    assert evidence_binding(decision) == {"drying": [1]}


def test_the_guidance_names_the_passage_for_each_half():
    guidance = _binding_guidance(decide(COMPOUND))

    assert "compatibility: [4]" in guidance
    assert "thickness: [1]" in guidance
    assert "cited to the passage listed for it" in guidance


def test_the_bound_answer_passes_every_check():
    """What the fix is for: the two-claim answer the corpus supports, printing.

    Each sentence carries the citation of the passage that states it, and the
    substrate is described in the passages' own word.
    """
    decision = decide(COMPOUND)
    text = ("Ultra is an insulating lime undercoat plaster that is suitable for "
            "most masonry and lath backgrounds, and it gives an additional "
            "insulation layer to internal walls [4]. "
            "Ultra should be applied in a uniform thickness of between 10 and "
            "30mm [1].")

    failures = run_checks(text, HITS, NAMES, decision.asked_terms)

    assert failures == [], failures


# ------------------------------------------------ 2. the narrow suitability

def test_a_plain_suitability_question_is_recognised_as_one():
    """"Can Ultra be applied internally on solid brick?" used to detect nothing.

    No compatibility term matched, so no property was detected, no relevance
    gate ran and there was nothing for a binding to bind.
    """
    router = Router()

    assert router.slots.detect_properties(NARROW) == ["compatibility"]


def test_the_ask_only_phrasings_are_not_accepted_as_evidence():
    """Detection widened; what counts as evidence did not.

    "applied" appears in nearly every datasheet. Had it been added to the
    compatibility values to fix detection, the relevance gate and check 6 would
    have been satisfiable by a word that proves nothing about suitability.
    """
    terms = Router().slots.terms_for("property_asked", "compatibility")

    assert "applied" not in terms
    assert "be used" not in terms
    assert "suitable for" in terms


def test_the_user_fact_stays_brick_and_the_published_class_travels_beside_it():
    decision = decide(NARROW)

    assert decision.slots["substrate"] == "brick"
    assert decision.substrate_class == "masonry"


def test_the_source_is_not_made_to_say_brick():
    """The passages say masonry. The answer may not report that as "brick"."""
    guidance = _binding_guidance(decide(NARROW))

    assert '"masonry"' in guidance and '"brick"' in guidance
    assert "never say" in guidance


def test_the_narrow_suitability_answer_is_supported():
    decision = decide(NARROW)
    text = ("Ultra is suitable for most masonry and lath backgrounds [4]. "
            "It gives an additional insulation layer to internal walls [4].")

    failures = run_checks(text, HITS, NAMES, decision.asked_terms)

    assert failures == [], failures


# ------------------------------------------------------- 3. bare thickness

def test_the_thickness_question_binds_to_how_to_apply():
    decision = decide(THICKNESS)

    assert evidence_binding(decision) == {"thickness": [1]}


def test_the_thickness_answer_states_the_published_range():
    decision = decide(THICKNESS)
    text = "Ultra should be applied in a uniform thickness of between 10 and 30mm [1]."

    failures = run_checks(text, HITS, NAMES, decision.asked_terms)

    assert failures == [], failures
    assert "10 and 30mm" in text


def test_a_question_with_no_substrate_gap_gets_no_substrate_line():
    """The wording note is emitted only when there is a gap to close."""
    guidance = _binding_guidance(decide(THICKNESS))

    assert "never say" not in guidance


def test_one_property_is_not_worth_a_binding_line():
    """A single claim cannot be bound to the wrong half of itself.

    Saying so anyway is prompt churn on every lookup in the corpus, and that
    cost was measured rather than reasoned about: the one-property form of this
    guidance turned "25kg sack" into "25 kg sack" on the brief's first test
    question, and the evaluation compares published figures with whitespace
    collapsed and nothing else normalised.
    """
    decision = decide(THICKNESS)
    assert evidence_binding(decision) == {"thickness": [1]}

    assert "thickness: [1]" not in _binding_guidance(decision)


# ------------------------------------- incidental properties are not asks

def test_a_property_a_word_happened_to_match_is_not_bound():
    """"per bag" reads as coverage inside a question about mixing water.

    Listing it told the model to answer it: the live build gave the water
    figure, then a coverage figure, then Solo Primer's coverage — three
    sentences where one thing was asked. The relevance gate still ORs across
    both, because refusing a two-property enquiry whole is the failure *it*
    exists to avoid; only the binding narrows.
    """
    router = Router()
    question = "How much water does Solo Onecoat need per bag?"

    assert router.slots.detect_properties(question) == ["water", "coverage"]
    assert router.slots.primary_properties(question) == ["water"]


def test_an_ask_only_phrasing_does_not_add_a_second_ask():
    """"applied" recognises a suitability question; it does not create one.

    "What thickness should Ultra be applied at?" asks one thing. The ask-only
    phrasing exists so that a question naming *no* property is still
    recognised, and it is dropped wherever a properly matched property
    outscores it.
    """
    router = Router()

    assert router.slots.primary_properties(
        "What thickness should Ultra be applied at?") == ["thickness"]
    assert router.slots.primary_properties(NARROW) == ["compatibility"]


def test_a_genuine_second_ask_survives_the_band():
    """The two-property questions the corpus answers are still bound as two.

    Situation S8's compatibility-and-hardening-time question is the one the
    relevance gate's two halves were reconciled for; narrowing the binding must
    not undo that.
    """
    router = Router()

    assert set(router.slots.primary_properties(COMPOUND)) == {"compatibility",
                                                              "thickness"}
    assert "coats" in router.slots.primary_properties(
        "Which finish coats are compatible with Forte render, including the "
        "hardening time required first?")


def test_an_ask_only_phrasing_cannot_decide_which_property_wins():
    """It is a fallback, consulted only where a property matched none of its own words.

    Letting it add to a compatibility already matched by "can i use" carried the
    compound question from a tie with *thickness* to a compatibility win — and
    `_location_matters` reads the winner, so a question plainly asking a
    thickness stopped being answered for inside and outside separately when the
    caller had not said which. Measured, on the question below.
    """
    router = Router()
    uncued = "Can I use Ultra, and what thickness should it be applied at?"

    assert router.slots.detect(uncued)["property_asked"] == "thickness"
    assert router.route(uncued, HITS, above_threshold=True).per_option


# --------------------------------------------------- 4. the adversarial case

def test_a_suitability_claim_cited_to_the_thickness_passage_is_still_refused():
    """The original defect, as the verifier sees it. Check 1 is unchanged.

    "Ultra is suitable for brick" reduces to {ultra, suitable, brick} against a
    passage carrying only "ultra": 1/3 = 0.33, under the 0.4 line.
    """
    failures = run_checks("Ultra is suitable for brick [1].", HITS, NAMES, [])

    assert "check 1" in names_of(failures), failures


def test_the_binding_cannot_make_an_unsupported_clause_print():
    """Guidance is a generation control, not a safety boundary.

    A model handed the correct binding and writing an unsupported clause anyway
    is refused exactly as it was before the binding existed. If this ever
    passes, the binding has been mistaken for a check.
    """
    decision = decide(COMPOUND)
    assert evidence_binding(decision)["compatibility"] == [4]

    text = ("Ultra is suitable for most masonry and lath backgrounds [4]. "
            "It will not crack on a solid wall [4].")
    failures = run_checks(text, HITS, NAMES, decision.asked_terms)

    assert "check 1" in names_of(failures), failures


def test_a_figure_bound_to_the_wrong_passage_is_still_refused():
    """Check 2 is unchanged: the thickness figure cited to the mixing passage."""
    failures = run_checks(
        "Ultra is applied at between 10 and 30mm [5].", HITS, NAMES, [])

    assert "check 2" in names_of(failures), failures


# ------------------------------------- 5. the product name the index carries

def test_a_question_saying_ultra_matches_the_catalogue_name():
    """Targeted retrieval is not defeated by the catalogue's longer name.

    The index tags Ultra's chunks "Ultra Insulating Lime Render Base Coat" and
    "Ultra Insulated Lime Render Base Coat"; a caller says "Ultra". Containment
    either way is what keeps the two the same product, and a failure here sends
    every targeted lookup to the unscoped fallback.
    """
    assert product_matches("ultra", ULTRA)
    assert product_matches("ultra", ULTRA_PAGE)
    assert product_matches("Lime Green Ultra", "Ultra")
    assert not product_matches("ultra", "Warmshell system")
    assert not product_matches("ultra", "")


# --------------------------------------------------- guarding the generalisation

def test_a_compound_question_is_still_answered_per_option_when_location_is_uncued():
    """The winning property is not the only one that decides this.

    A suitability ask leads the compound sentence and scores above the thickness
    ask, so reading `property_asked` alone made a plainly thickness-shaped
    question stop being answered for both inside and outside.
    """
    decision = decide("Can I use Ultra, and what thickness should it be "
                      "applied at?")

    assert "thickness" in decision.evidence_terms
    assert decision.per_option, decision


def test_a_decision_built_without_the_router_still_composes():
    """Every field is empty-safe. A `Decision` a test builds carries no binding."""
    from assistant.router import Decision

    bare = Decision(Path_.COMPOSE, "built by hand", "8", hits=HITS)

    assert evidence_binding(bare) == {}
    assert _binding_guidance(bare) == ""


def test_a_question_that_names_no_product_still_binds():
    """Product scoping is a preference, not a precondition.

    A question naming no product -- "what preparation does a masonry background
    need" -- has no scope to apply, and the binding falls back to ranking the
    passages on the words alone.
    """
    decision = decide(COMPOUND, product="")

    assert decision.slots.get("product") is None
    assert evidence_binding(decision) == {"compatibility": [4], "thickness": [1]}


def test_a_property_no_passage_states_is_not_bound():
    """The corpus publishes no thermal figure for Ultra, so nothing is offered.

    A binding that named a passage here would be pointing the model at evidence
    for a claim the evidence does not support, which is the near-miss decision 9
    exists to refuse rather than to decorate.
    """
    decision = decide("What is the U-value and the thickness of Ultra?")

    assert "thermal" in decision.evidence_terms
    assert "thermal" not in evidence_binding(decision)


def test_the_substrate_line_is_silent_when_the_passages_use_the_callers_word():
    """No gap, no note. The rule fires on a difference, not on a class existing."""
    says_brick = passage(
        "Suitable for most masonry backgrounds, including brick and stone, at "
        "a thickness of 10 to 30mm.",
        section="Backgrounds", product=ULTRA_PAGE,
        url="https://example/ultra-plaster", title=ULTRA_PAGE,
        document_type="product_page", authority=3)
    decision = decide(NARROW, [says_brick])

    assert decision.substrate_class == "masonry"
    assert "never say" not in _binding_guidance(decision)


def test_a_property_outside_the_vocabulary_is_still_bindable():
    """The noun-phrase fallback, for a property the vocabulary does not hold.

    `primary_properties` reads the property vocabulary, so it has no opinion
    about a phrase lifted out of the question. The binding falls back to
    whatever the gate accepted rather than going silent, which is what keeps
    the fallback a narrowing of the gate and never a second gate of its own.
    """
    pot_life = passage(
        "Ultra has a pot life of 45 minutes once mixed.",
        section="How to Mix")
    decision = decide("What is the pot life of Ultra?", [pot_life])

    assert list(decision.evidence_terms) == ["pot life"]
    assert evidence_binding(decision) == {"pot life": [1]}


# ------------------- a binding that separates nothing is not worth saying

# The GD2 shape, with the rendering checklist's own words. Two properties are
# asked -- a thickness and a preparation -- and the passage that answers the
# thickness half also carries preparation language, so the two bound sets
# overlap on it. Naming them separates nothing the model could not already see.
GD2_QUESTION = ("I'm rendering an old masonry wall in a very exposed location. "
                "How thick should the lime render be, and what preparation "
                "does the background need?")

# The checklist's own words, and the overlap is in them rather than contrived:
# the Design section cites "BS 13914 Design, preparation and application of
# external rendering", so the passage carrying the thickness specification is
# also the best match for *preparation*.
CHECKLIST_DESIGN = passage(
    "Refer to British Standards BS 13914 Design, preparation and application "
    "of external rendering and internal plastering. Specify 16mm minimum "
    "thickness of lime render in moderately exposed locations, or 25mm in "
    "very exposed locations.",
    section="Design", product="Lime Rendering Checklist & Guide",
    url="https://example/rendering-checklist", title="Lime Rendering Checklist",
    document_type="knowledge_base", authority=3, score=0.770)

BACKGROUND_PREP = passage(
    "The background construction should be sufficiently true, in line and "
    "plumb. As a guide the maximum correction is a deviation of 5 mm under a "
    "2 m straight edge on a wall built to the specified thickness.",
    section="1) Construction Issues", product="Background Preparation",
    url="https://example/background-preparation",
    title="Background Preparation For Lime Rendering",
    document_type="knowledge_base", authority=3, score=0.725)

CHECKLIST_APPLICATION = passage(
    "Consult BS13914 Design, preparation and application of external "
    "rendering and internal plastering and the product Technical Datasheets. "
    "Add the same amount of water to each bag and mix for the same time.",
    section="Application", product="Lime Rendering Checklist & Guide",
    url="https://example/rendering-checklist", title="Lime Rendering Checklist",
    document_type="knowledge_base", authority=3, score=0.715)

GD2_HITS = [CHECKLIST_DESIGN, BACKGROUND_PREP, CHECKLIST_APPLICATION]


def test_overlapping_evidence_suppresses_the_binding_block():
    """Two properties, one passage answering both: the block says nothing.

    This is the regression GD2 measured. The block that separated nothing still
    perturbed the generation, and the published "25mm" came back as "25 mm" --
    correct, cited, passing all six checks, and a failed assertion about a
    figure, because the evaluation compares published figures with whitespace
    collapsed and nothing else normalised.
    """
    decision = decide(GD2_QUESTION, GD2_HITS, product="")
    bound = evidence_binding(decision)

    assert len(bound) > 1, bound
    assert not _distinguishes(bound), bound
    assert "Where each thing asked about is stated" not in _binding_guidance(decision)


def test_disjoint_evidence_keeps_the_binding_block():
    """The case the binding exists for: each half stated in its own passage."""
    decision = decide(COMPOUND)
    bound = evidence_binding(decision)

    assert bound == {"compatibility": [4], "thickness": [1]}
    assert _distinguishes(bound)

    guidance = _binding_guidance(decision)
    assert "compatibility: [4]" in guidance
    assert "thickness: [1]" in guidance


def test_distinguishes_is_about_shared_passages_not_set_size():
    """The rule is stated as a property, so it is worth testing as one."""
    assert _distinguishes({})
    assert _distinguishes({"a": [1]})
    assert _distinguishes({"a": [1], "b": [2, 3]})
    assert not _distinguishes({"a": [1, 2], "b": [2]})
    assert not _distinguishes({"a": [1], "b": [1]})


def test_a_suppressed_binding_does_not_take_the_substrate_line_with_it():
    """The two halves of the guidance are independent.

    The substrate line closes a vocabulary gap rather than pointing at a
    passage, so a binding that separates nothing must not silence it.
    """
    one_passage_answers_both = [
        passage("Suitable for most masonry and lath backgrounds. Apply in a "
                "uniform thickness of between 10 and 30mm.",
                section="Backgrounds", product=ULTRA_PAGE,
                url="https://example/ultra-plaster", title=ULTRA_PAGE,
                document_type="product_page", authority=3),
    ]
    decision = decide(COMPOUND, one_passage_answers_both)
    bound = evidence_binding(decision)

    assert len(bound) > 1 and not _distinguishes(bound), bound
    guidance = _binding_guidance(decision)
    assert "Where each thing asked about is stated" not in guidance
    assert 'never say "brick"' in guidance


def test_the_gb1_phrasing_is_untouched_by_the_overlap_rule():
    """GB1 binds one property, so neither the old gate nor the new one applies.

    What answers GB1 is the substrate line, not the binding block -- "would ...
    be suitable" matches no compatibility *value* term, only the ask-only cue,
    so `primary_properties` returns the thickness alone.
    """
    gb1 = ("I have an old solid brick wall and want to improve its insulation. "
           "Would Lime Green Ultra be suitable internally, and what thickness "
           "can it be applied at?")
    decision = decide(gb1)

    assert Router().slots.primary_properties(gb1) == ["thickness"]
    assert list(evidence_binding(decision)) == ["thickness"]
    guidance = _binding_guidance(decision)
    assert "Where each thing asked about is stated" not in guidance
    assert 'never say "brick"' in guidance


# ------------------- ordering: the bound passage read first, not talked about

def test_the_single_bound_passage_is_promoted_to_the_front():
    """The failing image case: one property, one passage, ranked fourth.

    "Would Ultra be suitable internally" binds `compatibility` to the product
    page that says "Suitable for most masonry and lath backgrounds" -- and the
    block that would have said so is suppressed, because one claim cannot be
    bound to the wrong half of itself. The model cited a different passage and
    check 1 refused the answer. The evidence was there; nothing pointed at it.
    """
    ordered = promote_bound(HITS, {"compatibility": [4]})

    assert ordered[0] is INSULATED_PAGE
    assert ordered == [INSULATED_PAGE, APPLY, GENERAL, INSULATING_PAGE, MIX]


def test_promotion_keeps_every_passage_and_their_relative_order():
    """Ordering only: nothing added, nothing dropped, no duplicate."""
    ordered = promote_bound(HITS, {"compatibility": [4]})

    assert len(ordered) == len(HITS)
    assert {id(h) for h in ordered} == {id(h) for h in HITS}
    rest = [h for h in ordered if h is not INSULATED_PAGE]
    assert rest == [h for h in HITS if h is not INSULATED_PAGE]


def test_two_bound_properties_are_left_alone():
    """The model needs both halves; promoting one says the other matters less."""
    assert promote_bound(HITS, {"compatibility": [4], "thickness": [1]}) == HITS


def test_a_property_bound_to_two_passages_is_left_alone():
    """A tie the ranking declined to break is not broken here either."""
    assert promote_bound(HITS, {"thickness": [1, 3]}) == HITS


def test_a_passage_already_first_is_not_moved():
    """No churn where there is nothing to gain: the list is returned as it was."""
    assert promote_bound(HITS, {"thickness": [1]}) is HITS


def test_an_empty_or_impossible_binding_changes_nothing():
    assert promote_bound(HITS, {}) is HITS
    assert promote_bound(HITS, {"thickness": [99]}) is HITS
    assert promote_bound([], {"thickness": [1]}) == []


def test_the_markers_the_checks_count_are_the_markers_the_prompt_used():
    """The one way this change could do real harm, pinned.

    Markers are positional. A list reordered for the prompt and not for
    `run_checks` would renumber the evidence underneath the verification: an
    answer citing [1] would be checked against whatever used to be first. So
    the promoted order has to be the order the checks count, and this asserts
    it by checking a sentence that is true of the promoted passage and false
    of the one it displaced.
    """
    ordered = promote_bound(HITS, {"compatibility": [4]})

    # Cited to [1], which is now the product page, so it must pass.
    passing = run_checks(
        "Ultra is suitable for most masonry and lath backgrounds [1].",
        ordered, NAMES, [])
    assert passing == [], passing

    # The same sentence against the *unpromoted* order cites the datasheet's
    # How to Apply section, which does not say it.
    failing = run_checks(
        "Ultra is suitable for most masonry and lath backgrounds [1].",
        HITS, NAMES, [])
    assert "check 1" in names_of(failing), failing
