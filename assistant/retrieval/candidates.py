"""Which products may be recommended, and what the corpus actually supports.

`DECISIONS.md` lists this as a known weakness in as many words: *"Compatibility
is enforced by citation, not by a rules gate. Nothing deterministic decides
which products are eligible for a substrate before retrieval runs."* This module
is that gate, built the only way the data allows.

**There is no compatibility matrix, and this module does not invent one.**
`assistant/compatibility.py` can load `config/compatibility_matrix.json`; that
file does not exist, because the product-to-substrate matrix exists nowhere on
the site and lives in datasheets and advisors' heads. Writing one from model
knowledge would be fabricating safety data in a liability-sensitive domain and
dressing it as a deterministic control. So eligibility here is decided by
**published evidence**: a product is a candidate because the corpus says
something about it for this job, and it survives because the corpus supports the
claims the recommendation would rest on. If the matrix is supplied later, it
becomes an additional and stricter gate, and `_documented_rule` is where it
lands.

Three stages, and the separation between them is the point:

**Discovery** proposes. It reads the registry harvested at ingestion and the
passages retrieved for the *resolved request*, never the transcript. A model may
also propose, through `ResolvedRequest.candidate_products`, and its proposals
enter here on exactly the same footing as any other name — normalised to the
registry upstream, and assessed identically. A model hypothesis is not an
eligible product.

**Assessment** decides, per candidate and **per property independently**. This
is the stage that makes "retrieval returned something" stop being a synonym for
"we can recommend this". Each required property is looked up by metadata against
that product's own passages, so a thickness figure published for Warmshell
cannot support a claim about Ultra — the failure the whole citation discipline
exists to prevent, arriving one level earlier.

**Containment** verifies. After composition, every product named in the answer
must be one that survived assessment. A model asked to choose between two
approved candidates and explain the runner-up may not introduce a third.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .. import observability as obs
from ..understanding import Intent, ResolvedRequest

# How many candidates are worth assessing. Each costs a metadata lookup per
# required property, so this bounds the work a single question can cause; and a
# recommendation naming more than a handful of products is not a recommendation.
MAX_CANDIDATES = 8

# Products whose passages may support a claim about a *system* they belong to.
# Kept empty rather than populated by guesswork: a system guide naming a product
# is caught by `_belongs_to` through the chunk's own product metadata.
SYSTEM_DOCUMENT_TYPES = ("system_guide",)


class Sufficiency(str, Enum):
    """Whether the corpus supports recommending this product for this job.

    ``INSUFFICIENT`` and ``INCOMPATIBLE`` are deliberately different. "The
    datasheets do not say whether Ultra suits cob" and "the datasheet says Ultra
    does not suit cob" are different answers to the customer and different
    admissions by the company, and collapsing them into one refusal would throw
    away the more useful half.
    """

    SUPPORTED = "supported"
    CONDITIONAL = "conditional"      # supported, with published conditions
    INSUFFICIENT = "insufficient"    # the corpus does not establish it
    INCOMPATIBLE = "incompatible"    # the corpus establishes the opposite


class Outcome(str, Enum):
    """How a selection ends. Note what is absent: ``COMPOSE``.

    Compose is a way of *printing* an answer that evidence already supports. It
    is not a destination for a selection whose evidence did not hold up, and
    routing there on insufficient evidence is how a system that refuses becomes
    a system that improvises -- the model is handed whatever retrieval returned
    and asked to be helpful about a product nothing established.
    """

    SUPPORTED_RECOMMENDATION = "supported_recommendation"
    CONDITIONAL_RECOMMENDATION = "conditional_recommendation"
    NEED_MORE_INFORMATION = "need_more_information"
    NO_SUPPORTED_RECOMMENDATION = "no_supported_recommendation"
    HANDOFF = "handoff"


@dataclass(frozen=True)
class Requirement:
    """What must be known before this kind of job can be asked about.

    Decision 10 privileges two slots globally and says so in its own "where it
    breaks": *"Two globally privileged slots is also a simplification: the facts
    a recommendation actually requires vary by intent."* This is that
    generalisation -- required facts per objective, with conditional ones that
    depend on the answers.

    It is authored policy, not derived compatibility data, and the difference
    matters. "You cannot choose an external render without knowing the exposure"
    is a statement about what makes a question answerable. "Ultra suits brick"
    is a statement about a product, and this module will not make one up.
    """

    always: tuple[str, ...] = ("substrate", "location")
    # slot value -> further slots it makes necessary
    conditional: dict = field(default_factory=dict)
    # Properties the corpus must support for this objective before a
    # recommendation may be made.
    properties: tuple[str, ...] = ("substrate",)


# Keyed by objective. `None` is the fallback and is deliberately the narrowest
# useful set rather than a union of everything -- a universal checklist asks
# people questions their job does not need, which is how an ask-back becomes an
# interrogation and the tool stops being used.
REQUIREMENTS: dict = {
    "insulation": Requirement(
        always=("substrate", "location"),
        conditional={"external": ("exposure",)},
        properties=("substrate", "thickness")),
    "render": Requirement(
        always=("substrate", "location"),
        conditional={"external": ("exposure",)},
        properties=("substrate", "coats")),
    "plaster": Requirement(
        always=("substrate", "location"),
        properties=("substrate",)),
    "repair": Requirement(
        always=("substrate",),
        properties=("substrate",)),
    # The fallback asks for the substrate and **not** the location, which is
    # decision 10's rule rather than a shortcut. Substrate is the one fact that
    # triggers an ask-back there; an uncued inside/outside is answered per
    # option instead, because the datasheets split that way and both answers fit
    # in the same five passages. Requiring it here would ask a question the rest
    # of the system already knows how to avoid asking.
    #
    # The named objectives above do require it, and that is the point of having
    # a table: choosing a render without knowing whether it faces the weather is
    # not a question that can be answered per option -- the exposure rules
    # differ, and one of the two answers would be wrong rather than merely
    # unnecessary.
    None: Requirement(always=("substrate",)),
}


def requirement_for(resolved: ResolvedRequest) -> Requirement:
    """The requirement set for this job, not for every job."""
    return REQUIREMENTS.get(resolved.objective or None, REQUIREMENTS[None])


def missing_facts(resolved: ResolvedRequest) -> list[str]:
    """What has to be asked before this question can be answered honestly.

    A slot the conversation holds as `CONFLICTING` -- a photograph disagreeing
    with something the person said -- counts as missing. That is the whole
    reason `FactStatus.CONFLICTING` exists: the alternative is silently backing
    one source over the other, and neither choice is defensible without asking.
    """
    requirement = requirement_for(resolved)
    have = resolved.slots()

    needed = list(requirement.always)
    for slot, value in have.items():
        needed += list(requirement.conditional.get(value, ()))

    missing = [slot for slot in dict.fromkeys(needed) if slot not in have]
    missing += [slot for slot in resolved.unsettled if slot not in missing]
    return missing


# ------------------------------------------------------------ the evidence

@dataclass(frozen=True)
class CandidateAssessment:
    """One product, and exactly what the corpus does and does not establish."""

    product: str
    status: Sufficiency
    supported_properties: tuple[str, ...] = ()
    unsupported_properties: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    blocking_unknowns: tuple[str, ...] = ()
    # Chunk ids, per property. This is what makes the cross-product test
    # possible: an assertion can ask which document supported which claim
    # rather than trusting that the right product appeared in the prose.
    evidence: dict = field(default_factory=dict)
    rejected_because: str = ""

    @property
    def eligible(self) -> bool:
        return self.status in (Sufficiency.SUPPORTED, Sufficiency.CONDITIONAL)


@dataclass(frozen=True)
class RecommendationDecision:
    """The outcome, the set it is allowed to draw on, and why."""

    outcome: Outcome
    approved: tuple[CandidateAssessment, ...] = ()
    rejected: tuple[CandidateAssessment, ...] = ()
    missing: tuple[str, ...] = ()
    reason: str = ""

    @property
    def approved_names(self) -> frozenset[str]:
        return frozenset(a.product for a in self.approved)


def _belongs_to(hit, product: str) -> bool:
    """Is this passage published *for this product*?

    The single most important predicate in the module. Retrieval for an Ultra
    question happily returns a Warmshell system guide, and a coverage figure
    lifted from it would be attributed to Ultra in the answer -- check 3 catches
    that after generation, and this catches it before, which is better because
    a refusal the customer never sees costs nothing.

    Matching is on the chunk's own `product` metadata, written at ingestion from
    the page the document came from, rather than on the text. A datasheet
    mentioning a neighbouring product in one sentence does not become evidence
    about it.
    """
    owner = (getattr(hit.chunk, "product", "") or "").lower()
    wanted = product.lower()
    if not owner:
        return False
    return owner == wanted or wanted in owner or owner in wanted


def _terms_for(prop: str, resolved: ResolvedRequest, detector) -> tuple[str, ...]:
    """The words a property is printed under, from the existing vocabulary."""
    if prop == "substrate" and resolved.substrate:
        terms = detector.terms_for("substrate", resolved.substrate)
        return tuple(terms) or (resolved.substrate,)
    terms = detector.terms_for("property_asked", prop)
    return tuple(terms) or (prop,)


def assess(product: str, resolved: ResolvedRequest, repo, detector,
           audiences: tuple[str, ...] = ("public",),
           matrix=None) -> CandidateAssessment:
    """Does the corpus support recommending this product for this job?

    Every required property is evaluated **independently**, and that is the
    design decision worth defending. The tempting shape is a single boolean --
    "is this eligible" -- and it produces all-or-nothing refusals: a candidate
    whose thickness is documented and whose preparation is not gets refused
    whole, and the person is told nothing rather than told the half that is
    published. Splitting the verdict lets the answer say "supported to 25 mm;
    the sheets do not state the preparation for this background", which is both
    more useful and more honest.

    `blocking_unknowns` is the subset that must not be papered over. Substrate
    suitability is one: a product recommended for a wall the sheets never
    mention it for is the costly error, whatever else is documented.
    """
    requirement = requirement_for(resolved)
    supported: list[str] = []
    unsupported: list[str] = []
    evidence: dict = {}
    caveats: list[str] = []

    documented = _documented_rule(product, resolved, matrix)
    if documented is False:
        return CandidateAssessment(
            product=product, status=Sufficiency.INCOMPATIBLE,
            rejected_because="a documented rule excludes this product here")

    for prop in requirement.properties:
        terms = _terms_for(prop, resolved, detector)
        hits = repo.find_passages(product, terms, audiences=audiences, limit=3)
        own = [h for h in hits if _belongs_to(h, product)]
        if own:
            supported.append(prop)
            evidence[prop] = [h.chunk.chunk_id for h in own]
        else:
            unsupported.append(prop)

    caveats.extend(_caveats_for(product, repo))

    blocking = tuple(p for p in unsupported if p == "substrate")
    if blocking:
        status = Sufficiency.INSUFFICIENT
    elif unsupported or caveats:
        status = Sufficiency.CONDITIONAL
    else:
        status = Sufficiency.SUPPORTED

    return CandidateAssessment(
        product=product, status=status,
        supported_properties=tuple(supported),
        unsupported_properties=tuple(unsupported),
        caveats=tuple(caveats[:3]),
        blocking_unknowns=blocking,
        evidence=evidence,
        rejected_because=("the published material does not establish that this "
                          "product suits this background" if blocking else ""))


def _documented_rule(product: str, resolved: ResolvedRequest, matrix):
    """A documented compatibility decision, if one exists at all.

    Returns `True`, `False` or `None`, and **`None` means unknown rather than
    allowed**. The distinction is the whole reason this returns three values: a
    missing rule must not read as permission, or an absent matrix would silently
    approve everything it failed to mention.
    """
    if matrix is None or not resolved.substrate:
        return None
    return matrix.is_compatible(product, resolved.substrate,
                                resolved.location or None)


def _caveats_for(product: str, repo) -> list[str]:
    """Published conditions attached to this product's documents.

    Decision 11 holds caveats at document level and appends them by code. A
    conditional recommendation is one whose conditions are these, so they are
    read here rather than re-derived.
    """
    try:
        manifest = repo.manifest()
    except Exception:                                  # noqa: BLE001
        return []
    found: list[str] = []
    for document in manifest:
        if (document.product or "").lower() != product.lower():
            continue
        try:
            found += [c.sentence for c in repo.caveats(document.canonical_url)]
        except Exception:                              # noqa: BLE001
            continue
    return found


# ----------------------------------------------------------- the three stages

def discover(resolved: ResolvedRequest, hits, registry) -> list[str]:
    """Products worth assessing, from the corpus and the registry.

    Order matters and is not alphabetical. An explicitly named product comes
    first because the person asked about it; products whose own passages were
    retrieved for this request come next, because the corpus put them there;
    model hypotheses come last, because they are the weakest signal in the list.
    Every one of them is then assessed identically -- the order decides what is
    looked at, never what is approved.
    """
    approved_names = {p.lower(): p for p in registry}
    found: list[str] = []

    def add(name: str) -> None:
        canonical = approved_names.get((name or "").lower())
        if canonical and canonical not in found:
            found.append(canonical)

    if resolved.product:
        add(resolved.product)
    for hit in hits:
        add(getattr(hit.chunk, "product", "") or "")
    for hypothesis in resolved.candidate_products:
        add(hypothesis)

    return found[:MAX_CANDIDATES]


def decide(resolved: ResolvedRequest, hits, registry, repo, detector,
           audiences: tuple[str, ...] = ("public",),
           matrix=None) -> RecommendationDecision:
    """Discovery, then assessment, then an outcome that is never ``COMPOSE``.

    The missing-information gate runs **first**, before any retrieval is
    assessed. Asking what the wall is made of is cheap and correct; assessing
    eight products against an unknown substrate is expensive and meaningless,
    and would produce an approved set that changes the moment the person
    answers.
    """
    with obs.span("candidate_discovery", intent=resolved.intent.value,
                  objective=resolved.objective) as span:
        missing = missing_facts(resolved)
        span["missing"] = missing
        if missing:
            obs.event("missing_fact_decision", missing=missing,
                      unsettled=list(resolved.unsettled))
            return RecommendationDecision(
                outcome=Outcome.NEED_MORE_INFORMATION,
                missing=tuple(missing),
                reason=f"cannot choose a product without: {', '.join(missing)}")

        names = discover(resolved, hits, registry)
        span["candidates"] = names
        span["from_model"] = list(resolved.candidate_products)

    assessments = [assess(name, resolved, repo, detector, audiences, matrix)
                   for name in names]
    approved = tuple(a for a in assessments if a.eligible)
    rejected = tuple(a for a in assessments if not a.eligible)

    for assessment in assessments:
        obs.event("evidence_sufficiency", product=assessment.product,
                  status=assessment.status.value,
                  supported=list(assessment.supported_properties),
                  unsupported=list(assessment.unsupported_properties),
                  blocking=list(assessment.blocking_unknowns))
    for assessment in rejected:
        obs.event("candidate_rejected", product=assessment.product,
                  status=assessment.status.value,
                  reason=assessment.rejected_because)

    if not approved:
        incompatible = [a for a in rejected if a.status is Sufficiency.INCOMPATIBLE]
        return RecommendationDecision(
            outcome=Outcome.NO_SUPPORTED_RECOMMENDATION,
            rejected=rejected,
            reason=("the published material excludes the products considered"
                    if incompatible else
                    "the published material does not establish that any product "
                    "suits this background"))

    outcome = (Outcome.SUPPORTED_RECOMMENDATION
               if any(a.status is Sufficiency.SUPPORTED for a in approved)
               else Outcome.CONDITIONAL_RECOMMENDATION)
    obs.event("recommendation_selected", outcome=outcome.value,
              approved=[a.product for a in approved])
    return RecommendationDecision(outcome=outcome, approved=approved,
                                  rejected=rejected)


# Phrases that turn a mention of a product into a recommendation of it. A
# datasheet answer says "Ultra covers 1.5 m2 per bag"; a recommendation says
# "use Ultra". The distinction is what stops this guard refusing every ordinary
# lookup that happens to name the product it was asked about.
_RECOMMENDING = re.compile(
    r"\b(?:i (?:would |'d )?recommend|we recommend|recommended|"
    r"you (?:should|could|can) use|use\s+(?=[A-Z])|"
    r"(?:is|are|would be) (?:the )?(?:best|ideal|suitable|right|appropriate)|"
    r"go (?:for|with)|opt for|choose|the right product)\b", re.I)


def recommends_a_product(text: str, registry) -> list[str]:
    """Which approved-sounding products this answer actually recommends.

    The last line of defence, and deliberately independent of the intent label.
    Everything upstream -- the evidence gate, the approved candidate set, the
    containment check -- only engages once a request has been classified
    ``SELECT``. That makes intent classification a single point of failure for
    the whole recommendation safety story, and intent classification is the one
    stage in this system that involves a language model reading a sentence.

    So this asks a different question, of the finished text, whatever route
    produced it: does this answer tell somebody to use a product? If it does,
    every product it names has to have been approved by an evidence assessment
    in this turn. A lookup that merely mentions a product is untouched, which is
    why the phrasing matters -- "Ultra covers 1.5 m2 per bag" states a published
    figure and "use Ultra" makes a commercial recommendation the company stands
    behind.
    """
    from ..understanding import normalise_product

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    named: list[str] = []
    for sentence in sentences:
        if not _RECOMMENDING.search(sentence):
            continue
        lowered = sentence.lower()
        for product in registry:
            if not product or product.lower() not in lowered:
                continue
            # Returned in the one normalised spelling, so a caller comparing
            # against an approved set is comparing like with like. Returning the
            # registry's own form meant "Lime Green Ultra" failed to match an
            # approved "ultra" and an ordinary VERIFY answer was refused as an
            # unapproved recommendation -- evaluation situation S8 and
            # conversation C5, both of which had been passing.
            canonical = normalise_product(product)
            if canonical not in named:
                named.append(canonical)
    return named


def contained(text: str, decision: RecommendationDecision,
              registry) -> tuple[bool, list[str]]:
    """Does the answer name only products that survived assessment?

    The last gate, and the one that makes bounded ranking safe. Once there is an
    approved set, a model may be asked to pick from it and explain the
    runner-up -- a genuinely useful thing for it to do, and the kind of
    comparison the deterministic layer cannot write. What it must not be able to
    do is widen the set on the way, and nothing in a prompt can guarantee that.

    Checked against the **registry**, not against a word list, so a product the
    model invented outright is caught by check 5 and a real product it was never
    approved to mention is caught here.
    """
    lowered = text.lower()
    approved = {p.lower() for p in decision.approved_names}
    intruders = sorted({
        product for product in registry
        if product.lower() in lowered and product.lower() not in approved
        # A longer approved name containing this one is not an intrusion:
        # "Ultra" inside "Lime Green Ultra" is the same product.
        and not any(product.lower() in name for name in approved)
    })
    return (not intruders), intruders
