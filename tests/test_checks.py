"""The six checks, tested against the failures they exist to catch.

These run without Ollama and without an index: the checks are pure functions
over text and passages, which is deliberate — the part of the system that
decides whether something prints should be testable without a model running.

Each test names the real failure it stands for. A check that passes its own
happy path proves nothing; what matters is that it refuses the specific wrong
answer that motivated it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answer import products_named, run_checks      # noqa: E402
from assistant.model import Chunk, Document, Retrieved       # noqa: E402

NAMES = {
    "products": ["Solo Onecoat Lime Plaster", "Duro Lime Plaster Base Coat"],
    "colours": ["York", "Cotswold", "Purbeck"],
    "merchants": ["Womersley's", "The Lime Centre"],
    "contact": {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"},
}


def passage(content: str, product: str = "Solo Onecoat Lime Plaster",
            section: str = "Mixing", url: str = "https://example/solo") -> Retrieved:
    return Retrieved(
        chunk=Chunk(canonical_url=url, version=1, chunk_index=0, section=section,
                    content=content, product=product, document_type="datasheet",
                    authority=1),
        score=0.8,
        document=Document(canonical_url=url, title=product, document_type="datasheet",
                          authority=1, product=product, link_text=f"{product} datasheet"),
    )


SOLO = passage("Mix Solo with 5-6 litres of clean water per 25 kg sack. "
               "Apply at a minimum thickness of 8 mm.")
DURO = passage("Duro covers approximately 2.5 m2 per 25 kg bag at 11 mm.",
               product="Duro Lime Plaster Base Coat", section="Coverage",
               url="https://example/duro")


def names_of(failures):
    return {f.split(":")[0] for f in failures}


# ---------------------------------------------------------------- check 1

def test_uncited_sentence_is_refused():
    """A sentence with no citation is the plainest hallucination surface."""
    failures = run_checks("Solo needs 5-6 litres per sack.", [SOLO], NAMES, [])
    assert "check 1" in names_of(failures)


def test_citation_to_a_passage_that_was_not_retrieved():
    failures = run_checks("Solo needs 5-6 litres of clean water per 25 kg sack [4].",
                          [SOLO], NAMES, [])
    assert "check 1" in names_of(failures)


def test_sentence_unrelated_to_the_passage_it_cites():
    """Citing a real passage does not make an unrelated claim cited."""
    failures = run_checks(
        "Lime render should be protected from driving rain for six weeks [1].",
        [SOLO], NAMES, [])
    assert "check 1" in names_of(failures)


# ---------------------------------------------------------------- check 2

def test_invented_number_is_refused():
    """The failure this system exists to prevent: a plausible wrong figure."""
    failures = run_checks(
        "Mix Solo with 7 litres of clean water per 25 kg sack [1].",
        [SOLO], NAMES, [])
    assert "check 2" in names_of(failures)


def test_published_number_passes():
    failures = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1].",
        [SOLO], NAMES, [])
    assert "check 2" not in names_of(failures), failures


def test_unit_spelling_is_normalised_for_comparison():
    """'5–6 litres' in the sheet must match '5-6 litres' in the answer."""
    sheet = passage("Mix with 5 – 6 litres of clean water per 25 kg sack.")
    failures = run_checks("Mix with 5-6 litres of clean water per 25 kg sack [1].",
                          [sheet], NAMES, [])
    assert "check 2" not in names_of(failures), failures


def test_rounding_a_published_figure_is_refused():
    """2.5 m2 rendered as 3 m2 is the drift the check is for."""
    failures = run_checks("Duro covers approximately 3 m2 per 25 kg bag [1].",
                          [DURO], NAMES, [])
    assert "check 2" in names_of(failures)


# ---------------------------------------------------------------- check 3

def test_a_figure_attached_to_the_wrong_product():
    """Solo's water figure must not be printed against Duro."""
    failures = run_checks(
        "Duro needs 5-6 litres of clean water per 25 kg sack [1].",
        [SOLO, DURO], NAMES, [])
    assert "check 3" in names_of(failures), failures


# ---------------------------------------------------------------- check 4

def test_qualifier_must_be_in_the_cited_passage():
    """'minimum 8 mm' must not become 'maximum 8 mm'."""
    failures = run_checks("Apply at a maximum thickness of 8 mm [1].",
                          [SOLO], NAMES, [])
    assert "check 4" in names_of(failures), failures


def test_published_qualifier_passes():
    failures = run_checks("Apply at a minimum thickness of 8 mm [1].",
                          [SOLO], NAMES, [])
    assert "check 4" not in names_of(failures), failures


# ---------------------------------------------------------------- check 5

def test_invented_colour_is_refused():
    """'Cotswold Cream' is not a colour Lime Green publishes."""
    failures = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1]. "
        "It is available in Cotswold Cream [1].",
        [SOLO], NAMES, [])
    assert "check 5" in names_of(failures), failures


def test_real_colour_passes_check_five():
    sheet = passage("Solo is available in York and Cotswold.", section="Colours")
    failures = run_checks("Solo is available in York and Cotswold [1].",
                          [sheet], NAMES, [])
    assert "check 5" not in names_of(failures), failures


def test_invented_merchant_is_refused():
    failures = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1]. "
        "Buy it from Barnsley Building Supplies [1].",
        [SOLO], NAMES, [])
    assert "check 5" in names_of(failures), failures


def test_the_makers_name_in_front_of_a_real_product_is_not_an_invention():
    """Regression, from evaluation situation S9.

    The site names its own products both ways — "Lime Green Solo" and "Lime
    Green Duro" are harvested from the pages, plain "Natural Finish" is too —
    so whether a given product appears in the list with or without the brand is
    an accident of how each page happened to be written. Check 5 refused a
    correct, fully cited, two-document answer because the model wrote "Lime
    Green Natural Finish" and only "Natural Finish" was harvested. That is
    over-refusal, and over-refusal on the brief's multi-source test type.
    """
    sheet = passage("Natural Finish is compatible with Duro.", section="Compatibility")
    failures = run_checks(
        "Lime Green Natural Finish is compatible with Duro [1].",
        [sheet], {**NAMES, "products": [*NAMES["products"], "Natural Finish"]}, [])
    assert "check 5" not in names_of(failures), failures


def test_the_makers_name_does_not_launder_an_invented_product():
    """Only the prefix is forgiven; what follows it still has to be published."""
    failures = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1]. "
        "Lime Green Supercoat is the alternative [1].",
        [SOLO], NAMES, [])
    assert "check 5" in names_of(failures), failures


# ---------------------------------------------------------------- check 6

def test_asked_for_property_absent_is_refused():
    """The near-miss: right product, wrong property. Decision 9."""
    failures = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1].",
        [SOLO], NAMES, ["u value", "u-value", "thermal"])
    assert "check 6" in names_of(failures), failures


def test_asked_for_property_present_passes():
    failures = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1].",
        [SOLO], NAMES, ["water", "mixing water"])
    assert "check 6" not in names_of(failures), failures


# ------------------------------------------------------------ a clean answer

def test_a_correct_answer_passes_every_check():
    failures = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1]. "
        "Apply at a minimum thickness of 8 mm [1].",
        [SOLO], NAMES, ["water"])
    assert failures == [], failures


def test_an_ordinary_capitalised_word_is_not_reported_as_an_invented_name():
    """Check 5 must not refuse an answer for capitalising 'Water' mid-sentence."""
    sheet = passage("Add 5 litres per 25 kg sack.")
    failures = run_checks("Add 5 litres of Water per 25 kg sack [1].",
                          [sheet], NAMES, [])
    assert "check 5" not in names_of(failures), failures


def test_a_passage_that_names_another_product_may_be_cited_for_it():
    """The Forte sheet says which finish coats go over Forte, naming them.

    A sentence about Tradirend citing the Forte sheet is correctly attributed,
    because the Forte sheet is what published the claim. Refusing it is a false
    refusal on exactly the multi-document answers the brief asks for.
    """
    forte = passage(
        "Lime Green Tradirend, Natural Finish or Finish WP: key the Forte with "
        "a render scarifier and leave to harden for 3 to 5 days.",
        product="Forte Render Base Coat", section="Finishing Coats",
        url="https://example/forte")
    tradirend = passage("Tradirend is a traditional lime render.",
                        product="Tradirend Lime Render", section="Description",
                        url="https://example/tradirend")

    failures = run_checks(
        "Apply Tradirend after leaving the Forte to harden for 3 to 5 days [1].",
        [forte, tradirend], NAMES, [])
    assert "check 3" not in names_of(failures), failures


def test_a_figure_from_one_sheet_still_cannot_be_moved_to_an_unrelated_product():
    """The guard must stay closed where the cited passage never names the product."""
    solo = passage("Mix with 5-6 litres of clean water per 25 kg sack.",
                   product="Solo Onecoat Lime Plaster", section="Mixing")
    duro = passage("Duro is a general purpose lime undercoat.",
                   product="Duro Lime Plaster Base Coat", section="Description",
                   url="https://example/duro")

    failures = run_checks(
        "Duro needs 5-6 litres of clean water per 25 kg sack [1].",
        [solo, duro], NAMES, [])
    assert "check 3" in names_of(failures), failures


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  pass  {name}")
        except AssertionError:
            failed.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=1)
    print(f"\n{passed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


# ------------------------------------------- what adversarial review found

# Three ways a fabricated claim reached the page while every check reported
# green. None was found by use; all three came out of a review that attacked
# the checks rather than exercising them. Each is pinned from both sides — the
# attack must refuse, and the legitimate answer it resembles must still print,
# because a fix that refuses both is not a fix.


def test_a_figure_that_is_only_the_tail_of_a_published_one_is_refused():
    """A 25 kg sack quoted as 5 kg is a wrong mix on site.

    Check 2 compared with a substring test against a whitespace-stripped
    passage, so any fabricated figure that happened to end a real one passed.
    Dropping a leading digit is also the most likely single-token model error
    there is, which makes this the cheapest possible way to print a dangerous
    number.
    """
    sheet = passage("Solo coverage is 16 to 20 m2 per bag at 10 mm. "
                    "Pack size is 25 kg.", section="Coverage")

    pack = run_checks("Solo coverage is 16 to 20 m2 per bag at 10 mm "
                      "and the pack is 5 kg [1].", [sheet], NAMES, [])
    assert "check 2" in names_of(pack), pack

    coverage = run_checks("Solo coverage is 6 to 20 m2 per bag at 10 mm [1].",
                          [sheet], NAMES, [])
    assert "check 2" in names_of(coverage), coverage


def test_a_figure_inside_a_published_range_still_prints():
    """The other side of the same fix, and the brief's first test question.

    The Solo sheet prints "5-6 litres". An answer saying "between 5 and 6
    litres" is correct and must not be refused — requiring equality with a
    whole published figure would have refused it, which is why the rule is a
    digit boundary rather than equality.
    """
    ok = run_checks("Add between 5 and 6 litres of clean water per 25 kg sack [1]. "
                    "Apply at a minimum thickness of 8 mm [1].",
                    [SOLO], NAMES, ["water"])
    assert ok == [], ok


def test_an_unsupported_clause_cannot_shelter_behind_a_cited_one():
    """`DECISIONS.md` says check 1 kills "Ultra is probably fine on cob".

    It killed the standalone sentence only. Overlap was measured across a whole
    sentence, so a grounded opening diluted a fabricated tail below the
    threshold and carried it onto the page under the opening's citation. The
    tail is where a model puts an inference.
    """
    sheet = passage("Solo is a one coat lime plaster suitable for most solid "
                    "masonry backgrounds including brick and stone.",
                    section="Description")

    for tail in ["; it is probably fine on cob too.",
                 ", and cob is fine.",
                 ": it will not crack.",
                 "; it is interchangeable with Duro."]:
        answer = ("Solo is a one coat lime plaster suitable for most solid "
                  "masonry backgrounds [1]" + tail)
        failures = run_checks(answer, [sheet], NAMES, [])
        assert "check 1" in names_of(failures), f"{tail!r} printed: {failures}"


def test_an_ordinary_second_clause_still_prints():
    """Clause checking must not refuse a correct answer that uses punctuation."""
    ok = run_checks("Add 5-6 litres of clean water per 25 kg sack [1]; "
                    "apply at a minimum thickness of 8 mm [1].",
                    [SOLO], NAMES, ["water"])
    assert ok == [], ok

    listed = run_checks("Mix Solo with 5-6 litres of clean water per 25 kg sack [1], "
                        "and apply at a minimum thickness of 8 mm [1].",
                        [SOLO], NAMES, ["water"])
    assert listed == [], listed


def test_an_invented_product_is_refused_wherever_it_sits_in_the_sentence():
    """Check 5 exempted sentence-opening words, which is where a name goes.

    The system prompt tells the model to answer directly, so a name-first
    sentence is the common shape. The same invention passed or failed on word
    order alone.
    """
    sheet = passage("Suitable for lath and plasterboard backgrounds.",
                    section="Backgrounds")

    first = run_checks("Supercoat is suitable for lath and plasterboard "
                       "backgrounds [1].", [sheet], NAMES, [])
    assert "check 5" in names_of(first), first

    last = run_checks("For lath and plasterboard backgrounds use Supercoat [1].",
                      [sheet], NAMES, [])
    assert "check 5" in names_of(last), last


def test_a_real_product_opening_a_sentence_is_not_flagged():
    """And ordinary words opening one are still not mistaken for names."""
    sheet = passage("Solo is applied in one coat onto lath and plasterboard.",
                    section="Backgrounds")

    for answer in ["Solo is applied in one coat [1].",
                   "Apply Solo in one coat [1].",
                   "Use Solo onto lath and plasterboard [1]."]:
        failures = run_checks(answer, [sheet], NAMES, [])
        assert "check 5" not in names_of(failures), f"{answer!r}: {failures}"


def test_a_qualifier_must_sit_with_its_figure_not_merely_in_the_passage():
    """The architecture promises qualifiers "travel with their figure".

    Requiring only that the word appear somewhere in the passage let a sheet
    reading "Maximum coverage is achieved on a well prepared background. Apply
    at 10 mm per coat." license "apply at a maximum of 10 mm per coat" — a
    maximum thickness invented out of a sentence about coverage. The word was
    published; the claim was not.
    """
    detached = passage("Maximum coverage is achieved on a well prepared "
                       "background. Apply at 10 mm per coat.", section="Coverage")
    failures = run_checks("Apply at a maximum of 10 mm per coat [1].",
                          [detached], NAMES, [])
    assert "check 4" in names_of(failures), failures


def test_a_qualifier_published_beside_its_figure_still_prints():
    attached = passage("Apply at a maximum thickness of 10 mm per coat.",
                       section="Application")
    ok = run_checks("Apply at a maximum thickness of 10 mm per coat [1].",
                    [attached], NAMES, [])
    assert ok == [], ok


def test_the_asked_for_term_must_be_in_a_passage_the_answer_cites():
    """Decision 9's gate is about the evidence used, not the evidence retrieved.

    Scanning every retrieved passage let an answer citing only [1] satisfy a
    question about a property that appeared only in uncited [2] — which is the
    near-miss the gate exists to catch, arriving through the gate itself.
    """
    mixing = passage("Mix Solo with 5-6 litres of clean water per 25 kg sack.")
    coverage = passage("Solo covers approximately 2.5 m2 per 25 kg bag.",
                       section="Coverage", url="https://example/solo-coverage")

    cited_elsewhere = run_checks(
        "Mix Solo with 5-6 litres of clean water per 25 kg sack [1].",
        [mixing, coverage], NAMES, ["coverage"])
    assert "check 6" in names_of(cited_elsewhere), cited_elsewhere

    cited_properly = run_checks(
        "Solo covers approximately 2.5 m2 per 25 kg bag [2].",
        [mixing, coverage], NAMES, ["coverage"])
    assert "check 6" not in names_of(cited_properly), cited_properly


def test_the_overlap_threshold_is_pinned_from_below_as_well_as_above():
    """A constant nothing constrains is a constant that drifts.

    Every passing case in this file scores overlap 1.000 and every refusing one
    0.000, so the 0.4 threshold could be moved anywhere between 0.05 and about
    0.9 without a single test objecting — confirmed by mutation. That matters
    because 0.4 is precisely what decides whether a half-grounded clause prints.
    This pins a sentence either side of it.
    """
    sheet = passage("Solo is a one coat lime plaster for solid masonry "
                    "backgrounds.", section="Description")

    # Three content words, one of them published: 1/3 = 0.33, under the line.
    thin = run_checks("Solo needs careful priming beforehand [1].",
                      [sheet], NAMES, [])
    assert "check 1" in names_of(thin), thin

    # Three content words, two published: 2/3 = 0.67, over it.
    grounded = run_checks("Solo is a lime plaster for masonry [1].",
                          [sheet], NAMES, [])
    assert "check 1" not in names_of(grounded), grounded


def test_a_clause_of_nothing_but_stop_words_is_not_a_claim():
    """A clause asserting nothing has nothing to check, and must not divide by zero.

    Note how little it takes to be a claim: "it is for the wall" reduces to
    {wall} and is correctly refused against a passage that never says so. What
    passes here is a clause that reduces to no content words at all, which is
    the only case the guard is for.
    """
    sheet = passage("Solo is a one coat lime plaster for solid masonry.",
                    section="Description")
    ok = run_checks("Solo is a one coat lime plaster for solid masonry [1]; "
                    "it is up to you.", [sheet], NAMES, [])
    assert "check 1" not in names_of(ok), ok


def test_a_qualifier_with_no_figure_after_it_qualifies_nothing():
    """"Apply the minimum number of coats" claims no measurement, so check 4 stops.

    The adjacency rule needs a figure to be adjacent to. A qualifier used as an
    ordinary English word is not a numeric claim and must not be refused as one.
    """
    sheet = passage("Apply the minimum number of coats needed to cover.",
                    section="Application")
    ok = run_checks("Apply the minimum number of coats needed to cover [1].",
                    [sheet], NAMES, [])
    assert "check 4" not in names_of(ok), ok


def test_the_makers_name_is_forgiven_against_the_passage_too():
    """The brand prefix is stripped before both lists, not only the name list.

    A product named in a retrieved passage but absent from the harvested name
    list — which happens, because the list is built from product pages and a
    datasheet may mention a product that has none — must behave the same way
    with the maker's name in front of it as without.
    """
    # The passage names the product *without* the brand, and the answer adds it.
    # Written the other way round the earlier plain-substring test catches it and
    # this branch is never reached — which is what the first draft of this test
    # did, passing while proving nothing.
    # Mid-sentence, so the brand survives to the debranding test. Opening the
    # sentence with it instead ("Use Lime Green Silguard…") strips "Lime" and
    # "Green" as grammar words before this branch is ever reached, which is how
    # the first draft of this test passed without exercising it.
    sheet = passage("Use Silguard as the finishing treatment.", section="Finishing")
    ok = run_checks("The finishing treatment is Lime Green Silguard [1].",
                    [sheet], NAMES, [])
    assert "check 5" not in names_of(ok), ok


# ---------------------------------------------------------------- check 7
#
# The answer is still about the product that was asked about.
#
# Checks 1 to 6 ask whether a claim is *supported*. None asks whether it is
# about the right thing, and the gap between those two is a real answer this
# system gave: asked "and what thickness should I apply it at?" with Ultra
# carried from the previous turn, it replied "for the general purpose Duro lime
# base coat, the first coat should be applied between 9 to 12 mm thick" and
# passed every check. It passed honestly -- the sentence was supported by the
# passage it cited, the figure was verbatim in it, and Duro is a real published
# product, so check 5 allowed it. Check 3, the one that exists to keep a figure
# with its product, could not fire either: it compares against the products of
# the *retrieved* passages, and no Duro passage was retrieved. Duro was named
# inside the prose of a general FAQ answer about lime basecoats, and the model
# copied it out. Citation correctness held; product correctness did not.

ULTRA_REGISTRY = {
    "products": ["Ultra", "Ultra Insulating Lime Render Base Coat", "Duro",
                 "Solo", "Natural Finish"],
    "colours": [], "merchants": [],
    "contact": {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"},
}

ULTRA_PASSAGE = passage(
    "Ultra should be applied in a uniform thickness of between 10 and 30mm.",
    product="Ultra Insulating Lime Render Base Coat", section="How to Apply",
    url="https://example/ultra")

# The shape that caused the defect: a general answer carrying no product tag of
# its own, naming another product in its prose.
FAQ_PASSAGE = passage(
    "The recommended thickness depends on the product being applied. In the "
    "case of our general purpose Duro lime base coat, the first coat should be "
    "applied between 9 to 12 mm thick.",
    product="Frequently Asked Questions about Lime Green products",
    section="Plasters", url="https://example/faq")


def test_an_answer_about_another_product_is_refused():
    """The measured failure, reduced to its fixture.

    Everything about this sentence is honest except its subject: it cites the
    passage it came from, the figure is verbatim in that passage, and Duro is a
    published product. Only check 7 can see that the question was about Ultra.
    """
    failures = run_checks(
        "For the general purpose Duro lime base coat, the first coat should be "
        "applied between 9 to 12 mm thick [1].",
        [FAQ_PASSAGE], ULTRA_REGISTRY, [], product="ultra")

    assert "check 7" in names_of(failures), failures


def test_an_answer_about_the_resolved_product_passes():
    """The ordinary case, which must not become a refusal."""
    failures = run_checks(
        "Ultra should be applied in a uniform thickness of between 10 and 30mm [1].",
        [ULTRA_PASSAGE], ULTRA_REGISTRY, [], product="ultra")

    assert failures == [], failures


def test_a_product_the_caller_named_themselves_is_allowed():
    """Situation S9: is Natural Finish compatible with Duro?

    Naming Duro in the answer is answering the question that was asked. The
    allowed set takes it from the question rather than from the evidence,
    because a caller who names a product has authorised it whether or not
    retrieval happened to rank one of its passages.
    """
    both = passage(
        "Natural Finish is applied as a finishing coat to Duro lime render.",
        product="Natural Finish", section="Finishing Coats",
        url="https://example/natural-finish")

    failures = run_checks(
        "Natural Finish is applied as a finishing coat to Duro lime render [1].",
        [both], ULTRA_REGISTRY, [], product="natural finish",
        asked_products=("duro", "natural finish"))

    assert failures == [], failures


def test_a_product_the_cited_evidence_introduces_is_allowed():
    """Situation S8's shape: the resolved product's own sheet names another.

    The Forte datasheet's Finishing Coats section names Tradirend outright, so
    naming it back is not drift -- that evidence authorised it. Here the cited
    Ultra passage names Solo.
    """
    ultra_names_solo = passage(
        "Ultra may be finished with Solo where a smooth surface is wanted.",
        product="Ultra Insulating Lime Render Base Coat",
        section="Finishing Coats", url="https://example/ultra")

    failures = run_checks(
        "Ultra may be finished with Solo where a smooth surface is wanted [1].",
        [ultra_names_solo], ULTRA_REGISTRY, [], product="ultra")

    assert failures == [], failures


def test_an_uncited_passage_cannot_widen_the_allowed_set():
    """Cited evidence only, which is the same tightening check 6 already had.

    The Ultra passage naming Solo is retrieved but never cited, so it
    authorises nothing. Expanding from everything retrieval returned was
    measured and is too wide: on the failing turn it would have admitted Solo,
    Fine Stuff and Natural Finish, and caught Duro only by the accident that no
    Ultra passage mentions it.
    """
    ultra_names_solo = passage(
        "Ultra may be finished with Solo where a smooth surface is wanted.",
        product="Ultra Insulating Lime Render Base Coat",
        section="Finishing Coats", url="https://example/ultra-finishing")

    failures = run_checks(
        "Ultra is applied in a uniform thickness of between 10 and 30mm, and "
        "Solo goes over it [1].",
        [ULTRA_PASSAGE, ultra_names_solo], ULTRA_REGISTRY, [], product="ultra")

    assert "check 7" in names_of(failures), failures


def test_check_seven_does_not_run_when_no_product_was_resolved():
    """Most questions resolve no product, and GD2 and S8 are two of them.

    A question about rendering an exposed wall names no product and must not
    acquire a product-scope rule it was never in scope for.
    """
    failures = run_checks(
        "For the general purpose Duro lime base coat, the first coat should be "
        "applied between 9 to 12 mm thick [1].",
        [FAQ_PASSAGE], ULTRA_REGISTRY, [])

    assert "check 7" not in names_of(failures), failures


def test_the_catalogue_name_and_the_callers_name_are_one_product():
    """Ultra and Ultra Insulating Lime Render Base Coat are not two products.

    Chunks carry the catalogue name and callers say "Ultra", so comparing the
    allowed set by equality would refuse an answer for naming the very product
    it is about. The shared containment rule is what makes them one.
    """
    failures = run_checks(
        "Ultra Insulating Lime Render Base Coat should be applied in a uniform "
        "thickness of between 10 and 30mm [1].",
        [ULTRA_PASSAGE], ULTRA_REGISTRY, [], product="ultra")

    assert "check 7" not in names_of(failures), failures


def test_a_product_name_inside_another_word_is_not_a_mention():
    """Word boundaries, not containment.

    Testing membership with `in` matches "solo" inside "isolation", and what
    this feeds refuses an answer, so a false positive here is a refusal of
    something correct.
    """
    named = products_named(
        "Ultra provides good isolation and consolidates the surface.",
        ULTRA_REGISTRY["products"])

    assert named == {"ultra"}

    failures = run_checks(
        "Ultra provides good isolation of the wall at between 10 and 30mm [1].",
        [ULTRA_PASSAGE], ULTRA_REGISTRY, [], product="ultra")

    assert "check 7" not in names_of(failures), failures
