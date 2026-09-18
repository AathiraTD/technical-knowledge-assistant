"""Eligibility, evidence sufficiency and containment.

The module under test is the gate `DECISIONS.md` records as missing: *"Nothing
deterministic decides which products are eligible for a substrate before
retrieval runs."* These tests are written against the failures that gate exists
to stop, not against its implementation, so the assertions are mostly about
**which document supported which claim** rather than about what the prose says.

That distinction is the point of the file. An answer can name Ultra, cite a
passage, pass all six checks and still be wrong, if the figure it cited was
published for Warmshell. `answer_contains: ["Ultra"]` cannot tell the
difference; `assessment.evidence["thickness"]` can.

A fake repository stands in for the store. It is a dozen lines and returns
exactly what `find_passages` is contracted to return, which keeps the tests
about eligibility rather than about SQL — the real adapters have their own
contract suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.retrieval import candidates as c
from assistant.model import Chunk, Document, Retrieved             # noqa: E402
from assistant.router import SlotDetector                          # noqa: E402
from assistant.understanding import Intent, ResolvedRequest        # noqa: E402

REGISTRY = ["Ultra", "Duro", "Solo", "Warmshell Woodfibre Insulation Boards",
            "Forte"]


def chunk(product: str, section: str, content: str, index: int = 0) -> Chunk:
    return Chunk(canonical_url=f"https://example.invalid/{product.lower()}",
                 version=1, chunk_index=index, section=section, content=content,
                 product=product, document_type="datasheet", authority=1)


def hit(product: str, section: str = "Description",
        content: str = "text") -> Retrieved:
    return Retrieved(chunk=chunk(product, section, content), score=0.8,
                     document=Document(
                         canonical_url=f"https://example.invalid/{product.lower()}",
                         title=product, document_type="datasheet", authority=1,
                         product=product))


class FakeRepo:
    """A store whose `find_passages` returns what it was told to.

    `passages` maps product -> list of chunks. `find_passages` filters by the
    terms the way the real adapters do — the asked-for word must be present in
    the section or the text — because that lexical guarantee is precisely what
    the sufficiency stage relies on.
    """

    def __init__(self, passages: dict, caveats: dict | None = None):
        self.passages = passages
        self._caveats = caveats or {}
        self.calls: list[tuple] = []

    def find_passages(self, product, terms, audiences=("public",), limit=3):
        self.calls.append((product, tuple(terms)))
        out = []
        for ch in self.passages.get(product, []):
            haystack = f"{ch.section} {ch.content}".lower()
            if any(t.lower() in haystack for t in terms):
                out.append(Retrieved(chunk=ch, score=0.0, document=Document(
                    canonical_url=ch.canonical_url, title=ch.product,
                    document_type="datasheet", authority=1, product=ch.product)))
        return out[:limit]

    def manifest(self, audiences=("public",)):
        return [Document(canonical_url=f"https://example.invalid/{p.lower()}",
                         title=p, document_type="datasheet", authority=1,
                         product=p) for p in self.passages]

    def caveats(self, url):
        return self._caveats.get(url, [])


@pytest.fixture
def detector():
    return SlotDetector()


def request_for(**kwargs) -> ResolvedRequest:
    base = dict(intent=Intent.SELECT, raw_question="What should I use?",
                objective="plaster", substrate="brick", location="internal")
    base.update(kwargs)
    return ResolvedRequest(**base)


# ------------------------------------------------- the missing-information gate


def test_requirements_are_not_a_universal_checklist():
    """A repair does not need to know inside from outside; a render does.

    A single global list asks people questions their job does not need, which
    turns an ask-back into an interrogation.
    """
    assert c.requirement_for(request_for(objective="repair")).always == ("substrate",)
    assert "location" in c.requirement_for(request_for(objective="render")).always


def test_an_external_job_additionally_needs_the_exposure():
    """A conditional requirement, triggered by the answer to another."""
    missing = c.missing_facts(request_for(objective="render", location="external"))

    assert missing == ["exposure"]


def test_the_same_job_inside_does_not_need_the_exposure():
    assert c.missing_facts(request_for(objective="render",
                                       location="internal")) == []


def test_a_missing_substrate_is_asked_for_before_anything_is_assessed(detector):
    """Assessing eight products against an unknown wall is meaningless work.

    And the approved set it produced would change the moment the person
    answered, so it would also be misleading.
    """
    repo = FakeRepo({"Ultra": [chunk("Ultra", "Backgrounds", "suits brick")]})

    decision = c.decide(request_for(substrate=""), [hit("Ultra")],
                        REGISTRY, repo, detector)

    assert decision.outcome is c.Outcome.NEED_MORE_INFORMATION
    assert "substrate" in decision.missing
    assert repo.calls == [], "products were assessed against an unknown wall"


def test_a_slot_a_photograph_disputes_counts_as_missing(detector):
    """Neither source may be silently preferred, so the gate asks."""
    repo = FakeRepo({"Ultra": [chunk("Ultra", "Backgrounds", "suits brick")]})

    decision = c.decide(request_for(unsettled=("substrate",)), [hit("Ultra")],
                        REGISTRY, repo, detector)

    assert decision.outcome is c.Outcome.NEED_MORE_INFORMATION
    assert "substrate" in decision.missing


# --------------------------------------------------------------- discovery


def test_a_model_hypothesis_is_assessed_like_any_other_name(detector):
    """It may propose. It may not thereby approve."""
    repo = FakeRepo({"Duro": [chunk("Duro", "Backgrounds", "suits brick")]})

    decision = c.decide(request_for(candidate_products=("Duro",)),
                        [], REGISTRY, repo, detector)

    assert decision.approved_names == {"Duro"}, "the hypothesis had real evidence"


def test_a_model_hypothesis_with_no_evidence_is_rejected(detector):
    """The containment property, at the discovery end."""
    repo = FakeRepo({"Duro": [chunk("Duro", "Colours", "twenty four colours")]})

    decision = c.decide(request_for(candidate_products=("Duro",)),
                        [], REGISTRY, repo, detector)

    assert decision.outcome is c.Outcome.NO_SUPPORTED_RECOMMENDATION
    assert decision.approved_names == frozenset()


def test_a_product_outside_the_registry_never_becomes_a_candidate(detector):
    repo = FakeRepo({})

    found = c.discover(request_for(candidate_products=("Lime Green Supreme",)),
                       [], REGISTRY)

    assert found == []


def test_the_named_product_is_considered_first(detector):
    """The person asked about it, so it is the first thing looked at."""
    found = c.discover(request_for(product="Duro"),
                       [hit("Ultra"), hit("Solo")], REGISTRY)

    assert found[0] == "Duro"


# ------------------------------------------- evidence sufficiency per property


def test_retrieval_returning_something_is_not_sufficiency(detector):
    """The distinction this stage exists to draw.

    Ultra's datasheet is retrieved, is genuinely about Ultra, and says nothing
    whatever about brick. That is a retrieval hit and not a reason to recommend.
    """
    repo = FakeRepo({"Ultra": [chunk("Ultra", "Colours", "available in 24 colours")]})

    assessment = c.assess("Ultra", request_for(), repo, detector)

    assert assessment.status is c.Sufficiency.INSUFFICIENT
    assert "substrate" in assessment.blocking_unknowns


def test_a_supported_substrate_makes_a_candidate_eligible(detector):
    repo = FakeRepo({"Ultra": [
        chunk("Ultra", "Backgrounds", "suitable for brick and solid masonry")]})

    assessment = c.assess("Ultra", request_for(), repo, detector)

    assert assessment.eligible
    assert "substrate" in assessment.supported_properties
    assert assessment.evidence["substrate"], "no chunk id recorded"


def test_properties_are_evaluated_independently(detector):
    """The half that is published still gets answered.

    An all-or-nothing verdict refuses a candidate whose thickness is documented
    because its preparation is not, and the person is told nothing instead of
    being told the half the sheets actually carry.
    """
    repo = FakeRepo({"Ultra": [
        chunk("Ultra", "Backgrounds", "suitable for brick", 0)]})

    assessment = c.assess("Ultra", request_for(objective="insulation"),
                          repo, detector)

    assert "substrate" in assessment.supported_properties
    assert "thickness" in assessment.unsupported_properties
    assert assessment.eligible, "a documented substrate is still a usable answer"
    assert assessment.status is c.Sufficiency.CONDITIONAL


# ------------------------------- case D: cross-product evidence must not count


def test_a_figure_published_for_another_product_does_not_support_this_one(detector):
    """The cross-product protection, asserted where it actually happens.

    Retrieval for an Ultra question returns the Warmshell system guide happily.
    A thickness lifted from it would be attributed to Ultra in the answer, and
    the prose would look perfect. This is the stage that refuses it.
    """
    repo = FakeRepo({
        "Ultra": [chunk("Ultra", "Backgrounds", "suitable for brick")],
        "Warmshell Woodfibre Insulation Boards": [
            chunk("Warmshell Woodfibre Insulation Boards", "Thickness",
                  "apply at a thickness of 60mm")],
    })

    assessment = c.assess("Ultra", request_for(objective="insulation"),
                          repo, detector)

    assert "thickness" in assessment.unsupported_properties
    supporting = [cid for ids in assessment.evidence.values() for cid in ids]
    assert not any("warmshell" in cid.lower() for cid in supporting), (
        "an Ultra claim was supported by a Warmshell passage")


def test_evidence_ids_name_the_product_they_came_from(detector):
    """What makes the assertion above possible, stated as its own property."""
    repo = FakeRepo({"Duro": [chunk("Duro", "Backgrounds", "suitable for brick")]})

    assessment = c.assess("Duro", request_for(), repo, detector)

    assert all("duro" in cid.lower()
               for ids in assessment.evidence.values() for cid in ids)


# ------------------------------------------------------------- the outcomes


def test_insufficient_evidence_never_produces_a_compose_outcome(detector):
    """Non-negotiable: Compose is not an escape hatch from the evidence gate.

    Asserted structurally rather than by example -- no member of `Outcome` is
    Compose at all, so there is no value this function could return that would
    route there.
    """
    assert not any("compose" in o.value for o in c.Outcome)

    repo = FakeRepo({"Ultra": [chunk("Ultra", "Colours", "24 colours")]})
    decision = c.decide(request_for(), [hit("Ultra")], REGISTRY, repo, detector)

    assert decision.outcome is c.Outcome.NO_SUPPORTED_RECOMMENDATION


def test_a_documented_exclusion_is_reported_differently_from_silence(detector):
    """"The sheet says no" and "the sheets say nothing" are different answers."""
    class Matrix:
        def is_compatible(self, product, substrate, location=None):
            return False

    repo = FakeRepo({"Ultra": [chunk("Ultra", "Backgrounds", "suits brick")]})

    assessment = c.assess("Ultra", request_for(), repo, detector, matrix=Matrix())

    assert assessment.status is c.Sufficiency.INCOMPATIBLE
    assert assessment.status is not c.Sufficiency.INSUFFICIENT


def test_an_absent_rule_reads_as_unknown_and_not_as_permission(detector):
    """The trap in a half-populated matrix: silence approving everything."""
    class EmptyMatrix:
        def is_compatible(self, product, substrate, location=None):
            return None

    repo = FakeRepo({"Ultra": [chunk("Ultra", "Colours", "24 colours")]})

    assessment = c.assess("Ultra", request_for(), repo, detector,
                          matrix=EmptyMatrix())

    assert assessment.status is c.Sufficiency.INSUFFICIENT, (
        "a missing compatibility rule was read as approval")


def test_a_conditional_recommendation_when_caveats_are_published(detector):
    from assistant.model import Caveat
    url = "https://example.invalid/ultra"
    repo = FakeRepo(
        {"Ultra": [chunk("Ultra", "Backgrounds", "suitable for brick")]},
        caveats={url: [Caveat(url, "temperature",
                              "Do not apply below 5 degrees C.", "Mixing")]})

    decision = c.decide(request_for(), [hit("Ultra")], REGISTRY, repo, detector)

    assert decision.outcome is c.Outcome.CONDITIONAL_RECOMMENDATION
    assert decision.approved[0].caveats


# ------------------------------------- case H: the answer may not widen the set


def test_the_verifier_rejects_a_product_that_was_never_approved(detector):
    """Bounded ranking is only safe because this runs afterwards.

    A model given two approved candidates and asked to explain the runner-up is
    doing something useful. Nothing in a prompt stops it naming a third, so a
    prompt is not what stops it.
    """
    decision = c.RecommendationDecision(
        outcome=c.Outcome.SUPPORTED_RECOMMENDATION,
        approved=(c.CandidateAssessment("Ultra", c.Sufficiency.SUPPORTED),))

    ok, intruders = c.contained("Use Ultra, or Duro if you prefer.",
                                decision, REGISTRY)

    assert not ok
    assert intruders == ["Duro"]


def test_the_verifier_accepts_an_answer_drawn_from_the_approved_set(detector):
    decision = c.RecommendationDecision(
        outcome=c.Outcome.SUPPORTED_RECOMMENDATION,
        approved=(c.CandidateAssessment("Ultra", c.Sufficiency.SUPPORTED),
                  c.CandidateAssessment("Duro", c.Sufficiency.CONDITIONAL)))

    ok, intruders = c.contained("Ultra suits this; Duro is the alternative.",
                                decision, REGISTRY)

    assert ok and intruders == []


def test_a_longer_approved_name_containing_a_shorter_one_is_not_an_intruder():
    """"Ultra" inside "Lime Green Ultra" is the same product, not a second one."""
    registry = ["Ultra", "Lime Green Ultra", "Duro"]
    decision = c.RecommendationDecision(
        outcome=c.Outcome.SUPPORTED_RECOMMENDATION,
        approved=(c.CandidateAssessment("Lime Green Ultra",
                                        c.Sufficiency.SUPPORTED),))

    ok, intruders = c.contained("Lime Green Ultra is suitable.", decision, registry)

    assert ok, intruders
