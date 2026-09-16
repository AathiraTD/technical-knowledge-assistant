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

from assistant.answer import run_checks                      # noqa: E402
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
