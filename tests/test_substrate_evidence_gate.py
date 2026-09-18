"""The substrate half of the relevance gate.

Decision 9 and check 6 both say "the requested property **or substrate**, or a
synonym, must appear in a cited passage". The property half was implemented;
the substrate half was not, and the gap is not cosmetic. Asked "Can I use Solo
on cob walls?" the gate took its terms from `property_asked=compatibility`,
found "Solo is suitable for many backgrounds" in the evidence, and printed a
confident list of background guidance. No passage in the corpus says Solo suits
cob; the one place cob is given a material recommendation names a lime putty
mortar instead. That is the costly error decision 10 exists to prevent, arriving
silently.

The asymmetry is what made it invisible: the same question about plasterboard or
masonry answers correctly, because those substrates *are* named in Solo's
evidence. Only a substrate the corpus does not cover exposes it.

Runs without Ollama and without an index, like `test_router.py`: passages are
constructed by hand and a decision is a pure function of question and evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answering.router import Path_, Router          # noqa: E402
from assistant.knowledge.model import Chunk, Document, Retrieved   # noqa: E402


SOLO = "Solo Onecoat Lime Plaster"
FORTE = "Forte Lime Render"


def hit(content: str, section: str, product: str = SOLO,
        url: str | None = None, score: float = 0.72) -> Retrieved:
    url = url or f"https://example.invalid/{product}"
    return Retrieved(
        chunk=Chunk(canonical_url=url, version=1, chunk_index=0, section=section,
                    content=content, product=product, document_type="datasheet",
                    authority=1),
        score=score,
        document=Document(canonical_url=url, title=product, product=product,
                          document_type="datasheet", authority=1,
                          link_text=f"{product} datasheet"),
    )


# The three passages the live index actually returns for a Solo background
# question, transcribed from the shipped corpus. Between them they name
# masonry, woodfibre, plasterboard and undercoats -- and no earth substrate.
SOLO_BACKGROUNDS = [
    hit("4/7/24 Solo is highly breathable yet simple and convenient plaster "
        "for multiple different indoor backgrounds. Solo is suitable for many "
        "backgrounds and helps control and manage moisture in your walls.",
        section="Solo Onecoat Lime Plaster"),
    hit("For use on various internal walls and backgrounds such as masonry, "
        "woodfibre boards, plasterboard and suitable undercoats, Solo is "
        "designed to be labour saving and quick to use.",
        section="Description"),
    hit("Backgrounds should be porous, clean and dust free before starting. "
        "Do not wet down, apply directly to the board. Plasterboard: standard "
        "paper faced plasterboard do not need to be primed unless they have "
        "been installed for some time.",
        section="Preparation & Application"),
]


def route(question: str, hits=None):
    return Router().route(question, hits if hits is not None else SOLO_BACKGROUNDS,
                          True, ("public",))


# ------------------------------------------------- the substrate is unsupported


def test_a_substrate_the_corpus_never_names_is_refused():
    """"Can I use Solo on cob walls?" -- no Solo passage mentions cob.

    The evidence talks about backgrounds in general and satisfies every
    compatibility synonym in the gate's list. That is precisely the near-miss:
    right product, right topic, and silent about the one thing that was asked.
    """
    decision = route("Can I use Solo on cob walls?")
    assert decision.path is Path_.REFUSE, (
        f"answered on {decision.path.name} with asked_terms="
        f"{decision.asked_terms} -- no passage names cob")


def test_an_unsupported_substrate_is_not_rescued_by_a_general_claim():
    """"suitable for many backgrounds" must not stand in for the substrate.

    Separate from the test above because it is the specific sentence that
    carried the old defect: a general suitability claim is not evidence about
    any particular wall, and a gate that accepts it has stopped being a gate.
    """
    general_only = [SOLO_BACKGROUNDS[0]]
    assert route("Can I use Solo on cob walls?", general_only).path is Path_.REFUSE


def test_a_substrate_outside_the_vocabulary_is_still_refused():
    """"straw bale" is not a value in the substrate vocabulary.

    An unknown substrate must fail closed. Failing open here would mean the
    gate protects exactly the substrates somebody remembered to enumerate,
    which is the weakest possible reading of decision 9 -- and every substrate
    the partnership has not yet captured is one of these.
    """
    decision = route("Can I use Solo on straw bale walls?")
    assert decision.path is Path_.REFUSE, (
        f"answered on {decision.path.name} -- no passage names straw bale")


# --------------------------------------------------- the substrate is supported


def test_a_substrate_the_evidence_names_is_answered():
    """plasterboard is named twice in Solo's own evidence, so it must answer.

    The guard against over-correcting. A gate that refused this would refuse
    every compatibility question in the corpus and trade one failure mode for a
    worse one.
    """
    decision = route("Can I use Solo on plasterboard?")
    assert decision.path is not Path_.REFUSE, (
        "refused a substrate the evidence names outright")


def test_a_substrate_synonym_counts_as_the_substrate():
    """The vocabulary exists so the customer need not use the sheet's word.

    "dot and dab" is a plasterboard synonym; the evidence says "plasterboard".
    Refusing this would be the vocabulary gap the gate is explicitly not
    supposed to police.
    """
    decision = route("Can I use Solo over dot and dab?")
    assert decision.path is not Path_.REFUSE, (
        "refused a published substrate because the caller used a synonym")


def test_the_published_class_discharges_the_substrate():
    """brick is masonry, and masonry is the word the sheets print.

    Caught as a live regression: the first version of this gate required the
    caller's own word, and "Can I use Ultra on a brick wall?" began refusing a
    question the corpus answers -- no Ultra document contains "brick", and the
    product page says "Suitable for most masonry and lath backgrounds".

    The bridge is `substrate_classes`, the same map the evidence binding reads.
    It lets the question through the gate; it does not let the answer say
    "brick", which is check 1's job and is asserted in `test_evidence_binding`.
    """
    masonry = [hit("Suitable for most masonry and lath backgrounds.",
                   section="Description", product="Ultra",
                   url="https://example.invalid/ultra")]
    assert route("Can I use Ultra on a brick wall?", masonry).path is not Path_.REFUSE


def test_a_substrate_with_no_published_class_gets_no_bridge():
    """cob is deliberately absent from `substrate_classes`, and must stay absent.

    The class map is what lets brick reach a masonry passage. If cob were added
    to it -- to quieten this gate, or by someone reading earth and masonry as
    near enough -- then cob would reach the same passage and this whole defect
    would return through the fix for it. Asserted here so that edit fails a test
    rather than passing review.
    """
    from assistant.answering.router import SlotDetector
    assert SlotDetector().class_of("cob") == "", (
        "cob has been given a published class; it now bridges to masonry "
        "evidence and 'Can I use Solo on cob walls?' answers again")


# ------------------------------------------- the multi-source half must not move


def test_multi_source_support_may_be_spread_across_passages():
    """S8/S9 regression: one cited passage need not carry every asked term.

    Support is per claim and aggregated across citations. Requiring each
    passage to contain every requested property is what threw away two fully
    published two-document answers, and a fix to the substrate half that
    tightened this as a side effect would be a net loss.
    """
    evidence = [
        hit("Suitable finishing coats for Forte include Tradirend and Natural "
            "Finish.", section="Finishing Coats", product=FORTE,
            url="https://example.invalid/forte"),
        hit("Leave to harden for 3 to 5 days before applying the finish.",
            section="Application", product=FORTE,
            url="https://example.invalid/forte"),
    ]
    decision = route("Which finish coats are compatible with Forte render, "
                     "including the hardening time required first?", evidence)
    assert decision.path is not Path_.REFUSE, (
        "refused a question both halves of which are published, because no "
        "single passage carries both")
