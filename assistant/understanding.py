"""What this turn is asking for, as a structure rather than as a sentence.

The router detects slots with hand-written vocabularies, and that is the right
instrument for "does this question name a substrate". It is the wrong instrument
for "is this person choosing a product, checking one, or costing a job", and
scaling it to answer that means a growing list of patterns like
`what.*product.*should.*use` — each of which is a guess about phrasing, none of
which generalises, and all of which have to be maintained by whoever notices the
next miss.

So the *language* understanding moves to a local model with a constrained JSON
schema, and the *authority* stays here. The split is the same one the rest of
the system makes: a model may propose, deterministic code disposes.

**Nothing this module receives from the model is trusted.** `understand()`
returns a `TurnUnderstanding`, which is raw output and is named so that no
caller mistakes it for a decision. `resolve()` is what produces a
`ResolvedRequest`, and it throws away everything the model invented on the way:

* a product not in the snapshot's harvested registry is dropped, not queried —
  the same list check 5 already uses to refuse an invented name in an answer;
* a substrate or location outside `config/vocabularies.json` is dropped;
* an intent outside the enum falls back to the deterministic reading;
* measurements are re-parsed from the question by `re`, because the model's
  arithmetic is never used for anything.

**The deterministic path remains the fallback, and it is a real fallback.** If
Ollama is unreachable, if the reply does not parse, if the schema comes back
empty — `understand()` returns a `TurnUnderstanding` built by `SlotDetector` and
the question is answered exactly as it is answered today. Coverage drops,
safety does not, which is the rule the whole design follows.

**Why the model call is conditional.** Decision 2 keeps the model on Compose
only, and the measured cost of a cold generation on this hardware is tens of
seconds. Putting a second call in front of every question would be a latency
regression on every lookup in the system to improve the classification of a
minority of them. So `understand()` runs the model when the deterministic
reading is *ambiguous* — no confident intent, or a product-choice shape, which
is exactly the SELECT case this exists for — and skips it when the vocabulary
has already answered. `force=True` overrides that for evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from . import observability as obs
from . import ollama
from .answer import Provenance
from .conversation import ConversationState, SessionFact

# The model is asked for one small object and nothing else, so the budget is a
# fraction of an answer's. A reply longer than this is a malfunction, not a
# richer understanding.
MAX_UNDERSTANDING_TOKENS = 200

# Bounds on what may be lifted out of a question. A measurement past these is
# not a wall, and carrying it into a calculation would produce a confident
# absurdity.
MAX_AREA_M2 = 10_000.0
MAX_THICKNESS_MM = 500.0


class Intent(Enum):
    """What the person wants done, as distinct from how the answer is printed.

    `Path_` in `assistant/router.py` is the *output* shape — extract, compose,
    refuse. This is the *request* shape, and the two are deliberately different
    vocabularies: several intents print through Compose, and one intent can
    print through three different paths depending on what the evidence turns out
    to support. Collapsing them is what makes a router grow special cases.

    ``SELECT`` is the member this module was written for. It is a first-class
    intent and not an image feature: "what should I use on this internal brick
    wall?" is the same request with or without a photograph attached, and an
    image is one more source of evidence about it rather than what defines it.
    """

    SELECT = "select"              # which product for this job
    LOOKUP = "lookup"              # a published figure or fact
    VERIFY = "verify"              # is this specific product right here
    CALCULATE = "calculate"        # how much, how many
    UNDERSTAND = "understand"      # how does this work, why
    TROUBLESHOOT = "troubleshoot"  # something has gone wrong
    FIND = "find"                  # which document, where
    ESCALATE = "escalate"          # a person is needed
    UNKNOWN = "unknown"


# What the model is allowed to return. `product` is a free string on purpose:
# constraining it to the registry would let the schema pick the nearest
# allowed name for something the person never said, and a silent substitution
# is worse than a dropped field. It is checked against the registry afterwards.
SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "explicit_product": {"type": "string"},
        "candidate_products": {"type": "array", "items": {"type": "string"},
                               "maxItems": 5},
        "objective": {"type": "string"},
        "substrate": {"type": "string"},
        "location": {"type": "string"},
        "requested_properties": {"type": "array", "items": {"type": "string"},
                                 "maxItems": 6},
        # Whether this turn moves to a different wall, room or job. The model is
        # a second opinion here rather than the mechanism -- deterministic
        # phrase detection runs alongside it and either is enough, because a
        # wrongly-opened case costs one extra question and a wrongly-missed one
        # recommends a product for a wall the customer never described.
        "new_subject": {"type": "boolean"},
        "area_m2": {"type": "number"},
        "thickness_mm": {"type": "number"},
    },
    "required": ["intent"],
}

SYSTEM = (
    "You classify a single question about building products into a small JSON "
    "object. You do not answer it and you do not give advice.\n"
    "\n"
    "new_subject — true only if this turn moves to a DIFFERENT physical wall, "
    "room, elevation or job from the one being discussed. Correcting a detail "
    "about the same wall is not a new subject.\n"
    "\n"
    "intent — select: they are choosing a product for a job. verify: they name "
    "one product and ask whether it suits. calculate: how much or how many. "
    "lookup: a published figure or fact. understand: how or why something "
    "works. troubleshoot: something has gone wrong. find: which document. "
    "escalate: a person is needed.\n"
    "\n"
    "Copy values out of the question. Do not infer a substrate that is not "
    "mentioned, do not guess a product that is not named, and leave a field out "
    "rather than filling it with a plausible value. candidate_products may "
    "suggest products worth checking; they are checked elsewhere and are not "
    "recommendations."
)

PROMPT = """Classify this question.

Question: {question}

JSON:"""


@dataclass(frozen=True)
class TurnUnderstanding:
    """Raw output. Named to make it obvious that nothing here is decided yet.

    Every field is what a model said, or what the vocabulary said, and none of
    it has been checked against a registry. `resolve()` turns this into a
    `ResolvedRequest`, which is the thing a router may read.
    """

    intent: Intent = Intent.UNKNOWN
    explicit_product: str = ""
    candidate_products: tuple[str, ...] = ()
    objective: str = ""
    substrate: str = ""
    location: str = ""
    requested_properties: tuple[str, ...] = ()
    measurements: dict = field(default_factory=dict)
    # The model's reading of whether this turn changes subject. Corroborating
    # evidence for `conversation.opens_a_new_case`, never the sole authority.
    new_subject: bool = False
    # The policy topic this question matches, if any. Rank 1 -- a hard decision
    # that nothing downstream may reopen.
    policy_topic: str = ""
    # How this was arrived at, so a trace can say whether the model ran.
    source: str = "deterministic"     # deterministic | model | model+fallback
    error: str = ""

    @property
    def confident(self) -> bool:
        """Did deterministic detection actually recognise this request?

        Rank 2 of the precedence order. `UNKNOWN` means the vocabularies found
        nothing to go on, which is the one case where the model's reading is
        better than what we already had.
        """
        return self.intent is not Intent.UNKNOWN


@dataclass(frozen=True)
class ResolvedRequest:
    """The validated request, merged with what the conversation already knows.

    **This is the only thing routing and retrieval are allowed to read.** Not
    the transcript, not the raw model output, not a previous answer. Everything
    in it has either been said by the person, observed from an image with its
    provenance attached, or derived by deterministic code.
    """

    intent: Intent
    raw_question: str
    product: str = ""
    objective: str = ""
    substrate: str = ""
    location: str = ""
    exposure: str = ""
    requested_properties: tuple[str, ...] = ()
    measurements: dict = field(default_factory=dict)
    candidate_products: tuple[str, ...] = ()
    # The policy topic this question matched, if any. Carried so a later stage
    # can see that the route was decided before any of this ran.
    policy_topic: str = ""
    # Where each field came from. A recommendation has to be able to say why it
    # believed something, and "the person told me two turns ago" and "a model
    # read it off a photograph" are different answers to that.
    provenance: dict = field(default_factory=dict)
    unsettled: tuple[str, ...] = ()

    def slots(self) -> dict[str, str]:
        """The shape `Router.route(carried=...)` already takes."""
        return {name: value for name, value in (
            ("product", self.product), ("substrate", self.substrate),
            ("location", self.location), ("exposure", self.exposure),
        ) if value}

    def retrieval_query(self) -> str:
        """What to embed: the structured request, never the transcript.

        Decision 16.1 argues for retrieving on a building profile rather than on
        the sentence, and this is that, generalised past images. The question
        text is included because it carries the property being asked about in
        words the corpus uses; what is *excluded* is every earlier turn.
        """
        parts = [self.raw_question]
        parts += [v for v in (self.product, self.objective, self.substrate,
                              self.location, self.exposure) if v]
        parts += list(self.requested_properties)
        return " ".join(dict.fromkeys(parts))


# ------------------------------------------------------------- deterministic

_SELECT_SHAPE = re.compile(
    r"\b(which|what)\s+(product|plaster|render|mortar|system|one)\b"
    r"|\brecommend\b|\bshould i use\b|\bbest for\b|\bwhat do i (use|need)\b"
    r"|\bsuitable\b", re.I)

_AREA = re.compile(r"(\d+(?:\.\d+)?)\s*(?:square\s*met(?:re|er)s?|m2|m²|sqm)", re.I)
_THICKNESS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm|millimet(?:re|er)s?)", re.I)


def measurements_in(question: str) -> dict:
    """Numbers lifted from the question by code, never by the model.

    Decision 2 keeps arithmetic away from the model and router step 6 refuses
    the multiplication outright. This does not compute anything either -- it
    only records what was written, so a later stage can say "30 m² at 25 mm"
    was asked about without re-reading the sentence.
    """
    found: dict = {}
    area = _AREA.search(question)
    if area and 0 < float(area.group(1)) <= MAX_AREA_M2:
        found["area_m2"] = float(area.group(1))
    thickness = _THICKNESS.search(question)
    if thickness and 0 < float(thickness.group(1)) <= MAX_THICKNESS_MM:
        found["thickness_mm"] = float(thickness.group(1))
    return found


def deterministic(question: str, detector, gate=None,
                  registry=()) -> TurnUnderstanding:
    """The reading the existing vocabularies give. Always computed.

    It is the fallback when the model is unavailable, the validator when the
    model runs, and the thing that decides whether the model is worth calling at
    all.

    `gate` is the policy gate, and passing it here rather than consulting it
    later is what makes rank 1 of the precedence order structural. A price
    question has a fixed referral written once and reviewed once; nothing a
    model reads in the sentence can change that, so the model is not asked.
    That is also why "How much does Solo cost?" no longer pays a generation
    before the gate sees it -- it used to, because pricing has no slot
    vocabulary and therefore read as `UNKNOWN`.
    """
    slots = detector.detect(question)
    measured = measurements_in(question)

    matched = gate.match(question) if gate is not None else None
    if matched:
        return TurnUnderstanding(
            intent=Intent.ESCALATE, policy_topic=matched[0],
            measurements=measured, source="deterministic")

    if "cause_asked" in slots or "symptom" in slots:
        intent = Intent.TROUBLESHOOT
    elif "calculation" in slots or measured:
        intent = Intent.CALCULATE
    elif _SELECT_SHAPE.search(question):
        # Naming a product turns a selection into a verification, and the
        # difference is not cosmetic. "What plaster should I use?" asks the
        # system to *choose*, which is what the evidence gate and the approved
        # candidate set exist for. "Would Lime Green Ultra be suitable
        # internally, and what thickness can it be applied at?" asks about one
        # product the person has already chosen, and the honest answer is what
        # the datasheet says about it -- which the ordinary pipeline retrieves,
        # cites and checks.
        #
        # Treating the second as a selection was measured and wrong: evaluation
        # conversation C5 went to the evidence gate, found no candidate whose
        # substrate suitability was independently established, and refused a
        # question the corpus answers with citations. The gate was working; it
        # was being asked the wrong question.
        intent = (Intent.VERIFY if _named_in(question, registry)
                  else Intent.SELECT)
    elif "property_asked" in slots:
        intent = Intent.LOOKUP
    else:
        intent = Intent.UNKNOWN

    return TurnUnderstanding(
        intent=intent,
        # The objective decides *which* facts a recommendation requires
        # (`candidates.REQUIREMENTS`), and until this line it came from the
        # model and from nowhere else. That is the wrong place for it twice
        # over. It makes a control the model supplies -- principle 3 says
        # deterministic code owns routing -- and it makes the control vanish
        # whenever the model is unavailable, which is precisely when the system
        # should be *more* careful rather than less: with no objective the
        # requirement table falls back to substrate alone, so an insulation job
        # whose substrate a photograph had supplied asked nothing at all and
        # recommended a product without ever establishing whether the wall was
        # inside or out.
        #
        # The vocabulary is deliberately narrow -- see its own `_comment` --
        # and the model may still supply what it does not cover, because
        # `merge_understanding` takes the model's objective when this one is
        # empty. So this can only add a control, never remove one.
        objective=slots.get("objective", ""),
        substrate=slots.get("substrate", ""),
        location=slots.get("location", ""),
        requested_properties=tuple(detector.detect_properties(question)),
        measurements=measured,
        source="deterministic",
    )


def ambiguous(reading: TurnUnderstanding, question: str) -> bool:
    """Is the model worth a call on this turn? Only for a product choice.

    `UNKNOWN` used to be here too, on the reasoning that a request the
    vocabularies did not recognise is exactly the one worth reading properly.
    That was wrong, and the browser tests caught it: **"How much does Solo
    cost?" reads as `UNKNOWN`**, because pricing has no slot vocabulary -- it
    has a policy gate. So a question the gate is supposed to refuse instantly,
    with no model anywhere near it, was paying a cold generation first. On this
    hardware that is tens of seconds added to the fastest path in the system,
    and it inverts decision 8's whole point about the gate running before
    anything expensive.

    `SELECT` is the one intent whose fields -- objective, substrate, what is
    being asked for -- decide a recommendation rather than decorate an answer,
    and the one this module exists to read properly. Everything else keeps the
    deterministic reading, which is what it had before this module existed.

    The cost of the narrower rule is a selection phrased so unusually that
    `_SELECT_SHAPE` misses it: it reads as `UNKNOWN`, takes the ordinary
    pipeline, and is answered rather than recommended. That is a coverage loss
    and not a safety one -- the recommendation guard in `assistant/graph.py`
    catches an unapproved recommendation whatever the intent label says.
    """
    if reading.policy_topic:
        return False        # rank 1: the route is already decided
    return reading.intent is Intent.SELECT


# --------------------------------------------------------------- precedence

def merge_understanding(base: TurnUnderstanding,
                        model: TurnUnderstanding) -> TurnUnderstanding:
    """Combine the two readings under one stated order of authority.

    Written as a rule rather than as a set of conditions scattered through
    `understand()`, because the question "may the model change this?" has to
    have one answer per field that somebody can read off a page. It was not a
    rule first, and the omission had a measured cost: the live model read "What
    plaster should I use?" as something other than a selection, the requirement
    gate stopped running, and the ask-back stopped pausing the graph. A fix for
    that one field would have left every other field undecided.

    **The order, highest first.**

    1. **A hard policy decision.** A matched policy topic has a fixed referral
       written once and reviewed once. Nothing reopens it -- the model is not
       even asked, so there is no reading to overrule.
    2. **Confident deterministic detection.** An intent the vocabularies
       actually recognised, a product found in the registry by the question's
       own words, and measurements parsed by `re`. These are positive evidence
       from what the person wrote.
    3. **Validated model additions.** Fields deterministic detection left empty,
       after normalisation against the registry and the vocabularies.
    4. **Unknown.** The honest result when none of the above produced anything.

    **What the model may do:** supply an objective, add requested properties,
    propose candidate products, flag a change of subject, resolve a phrasing the
    vocabularies missed, and promote an `UNKNOWN` turn to a recognised intent.
    All of that is coverage the deterministic layer does not have.

    **What it may not do:** downgrade a confident intent, remove a product found
    deterministically, replace a parsed measurement, or touch a policy route.
    Each of those *removes* a control rather than adding a capability, and the
    asymmetry is the whole point -- a model that can only add is a model whose
    worst case is a wasted call.
    """
    if base.policy_topic:
        # Rank 1. Returned unchanged, including the intent, so nothing
        # downstream can be fooled into re-routing a question the gate owns.
        return base

    # Rank 2 over rank 3 for the intent: a recognised one stands.
    intent = base.intent if base.confident else model.intent

    return TurnUnderstanding(
        intent=intent,
        # Deterministic detection does not name products -- `resolve()` does
        # that against the registry -- so the model's suggestion is carried and
        # validated there, never used raw.
        explicit_product=base.explicit_product or model.explicit_product,
        # Additive: hypotheses to be assessed, not approvals.
        candidate_products=model.candidate_products or base.candidate_products,
        # Fields the vocabularies have no opinion about at all.
        objective=base.objective or model.objective,
        substrate=base.substrate or model.substrate,
        location=base.location or model.location,
        requested_properties=(base.requested_properties
                              or model.requested_properties),
        # Never the model's. Arithmetic and the numbers feeding it stay in code,
        # per decision 2 -- a model that could edit a measurement could change
        # the size of somebody's job.
        measurements=base.measurements,
        new_subject=model.new_subject,
        policy_topic=base.policy_topic,
        source=model.source,
        error=model.error,
    )


# -------------------------------------------------------------- the model

def _decode(body: str) -> dict:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def understand(question: str, detector, *, model: str = "",
               force: bool = False, enabled: bool = True,
               gate=None, registry=()) -> TurnUnderstanding:
    """Read one turn. The model assists; it never decides.

    Returns a `TurnUnderstanding` that is still raw -- `resolve()` is what
    validates it and makes it usable. The deterministic reading is computed first and returned
    unchanged whenever the model is skipped, unavailable or unusable, so every
    path out of this function produces something the rest of the system can
    route on.
    """
    base = deterministic(question, detector, gate, registry)
    if not enabled or (not force and not ambiguous(base, question)):
        return base

    with obs.span("query_understanding", intent_before=base.intent.value,
                  forced=force) as span:
        try:
            body, seconds = ollama.generate(
                PROMPT.format(question=question),
                model=model or ollama.GENERATION_MODEL,
                system=SYSTEM, schema=SCHEMA,
                num_predict=MAX_UNDERSTANDING_TOKENS)
        except ollama.OllamaUnavailable as error:
            # Not raised. An unreachable model on this stage costs a better
            # reading of the question, not the answer -- the deterministic one
            # is a complete reading and the system shipped on it.
            obs.event("ollama_error", stage="understanding",
                      error=type(error).__name__, detail=str(error))
            span["status"] = "unavailable"
            return TurnUnderstanding(**{**base.__dict__,
                                        "source": "deterministic",
                                        "error": str(error)})

        raw = _decode(body)
        span["seconds"] = round(seconds, 2)
        span["parsed"] = bool(raw)
        if not raw:
            span["status"] = "unparseable"
            return TurnUnderstanding(**{**base.__dict__, "source": "model+fallback",
                                        "error": "unparseable reply"})

        try:
            intent = Intent(raw.get("intent", ""))
        except ValueError:
            intent = Intent.UNKNOWN

        # What the model said, as its own reading. Nothing here is decided:
        # `merge_understanding` applies the precedence order, and it is the only
        # place that decides which of the two wins a field.
        proposed = TurnUnderstanding(
            intent=intent,
            explicit_product=str(raw.get("explicit_product", ""))[:80],
            candidate_products=tuple(
                str(p)[:80] for p in (raw.get("candidate_products") or [])[:5]),
            objective=str(raw.get("objective", ""))[:60],
            substrate=str(raw.get("substrate", ""))[:60],
            location=str(raw.get("location", ""))[:60],
            requested_properties=tuple(
                str(p)[:40] for p in (raw.get("requested_properties") or [])[:6]),
            new_subject=bool(raw.get("new_subject", False)),
            source="model",
        )
        merged = merge_understanding(base, proposed)
        if base.confident and proposed.intent is not base.intent:
            obs.event("intent_downgrade_refused", model_said=proposed.intent.value,
                      kept=base.intent.value)
        span["intent_after"] = merged.intent.value
        span["model_intent"] = proposed.intent.value
        return merged


# ----------------------------------------------------------- normalisation

# The manufacturer, as the website writes it in front of its own product names.
_BRAND = "lime green"


def normalise_product(name: str) -> str:
    """One spelling for one product, used everywhere two names are compared.

    Lower case with the maker's name removed, which is the form retrieval
    matches chunks on. Exposed rather than kept private because the alternative
    was measured: `assistant/candidates.py` compared full registry names against
    this form, so an answer saying "Lime Green Ultra" did not match an approved
    "ultra" and was refused as an unapproved recommendation. Two normalisations
    of the same thing is one too many.
    """
    lowered = (name or "").strip().lower()
    return lowered.removeprefix(_BRAND).strip() or lowered


def _named_in(question: str, registry) -> str:
    """The registry product this question names, by its own words.

    Deterministic and independent of the model, so a question saying "Duro" is
    about Duro whether or not the understanding stage ran.

    The normalisation is `Assistant._named_product`'s, character for character,
    and it is not cosmetic. Chunks are tagged with the catalogue name -- "Ultra
    Insulated Lime Render Base Coat" -- which neither contains "Lime Green Ultra"
    nor is contained by it. Matching is containment either way, so returning the
    brand-prefixed form makes the product boost and the targeted coverage lookup
    silent no-ops for every question that names a product the way the website
    writes it. "Ultra" finds three coverage passages; "Lime Green Ultra" finds
    none, which is how a quantity question comes to report a coverage figure as
    unpublished while it sits in the datasheet.

    Reintroducing that was measured rather than theorised: conversation C5 in
    the evaluation harness went from pass to "answer does not contain '0.6' as
    published" the moment this function returned the longer name.
    """
    lowered = question.lower()
    found = [p for p in registry if p and p.lower() in lowered]
    if not found:
        return ""
    return normalise_product(max(found, key=len))


def _registered(name: str, registry) -> str:
    """The registry's spelling of a product, or nothing at all.

    Longest match wins, the same rule `Assistant._named_product` uses, because
    "Lime Green Ultra" and "Ultra" are both harvested names and the longer one
    is the more specific claim. A name the registry does not hold returns "" --
    dropped rather than corrected, because the nearest allowed name to something
    a person never said is a silent substitution.
    """
    if not name:
        return ""
    wanted = name.strip().lower()
    matches = [p for p in registry
               if p.lower() == wanted or wanted in p.lower() or p.lower() in wanted]
    return max(matches, key=len) if matches else ""


def resolve(reading: TurnUnderstanding, question: str, detector, registry,
            state: ConversationState | None = None,
            turn_index: int = 0) -> ResolvedRequest:
    """Validate this turn's reading, then merge what the conversation knows.

    Merge order is the same one the router has always used and the same one
    `assistant/conversation.py` enforces: **this turn over history**. A value
    stated now beats a value remembered, because a person correcting themselves
    must not be answered from the thing they just corrected.

    A slot the conversation holds as `CONFLICTING` -- a photograph disagreeing
    with something the person said -- is deliberately *not* merged in. It is
    reported in `unsettled` instead, so the requirement gate can ask about it
    rather than pick a winner.
    """
    state = state or ConversationState()
    detected = detector.detect(question)
    provenance: dict = {}

    def take(slot: str, model_value: str) -> str:
        """This turn's value for a slot. Two independent checks, both required.

        The vocabulary check alone is not enough, and the gap is worth naming
        because the first version of this function had it. Running
        `detector.detect("cob")` over a model's answer returns
        ``{"substrate": "cob"}`` -- because cob *is* a substrate. It says
        nothing whatever about whether the person mentioned one. A model
        answering "cob" to "what should I use on my wall?" therefore passed a
        check designed to stop exactly that, and an invented substrate reached
        the resolved request looking like something the caller had said.

        So the model's word must also be **present in the question**. The
        vocabulary normalises; the question grounds. What the model still buys
        is the case the detector's patterns miss -- a substrate named in a
        phrasing the regexes do not match -- and what it can no longer do is
        supply a fact from nowhere.
        """
        if slot in detected:
            provenance[slot] = Provenance.STATED
            return detected[slot]
        if model_value:
            word = model_value.strip().lower()
            if word and word in question.lower():
                confirmed = detector.detect(model_value).get(slot, "")
                if confirmed:
                    provenance[slot] = Provenance.STATED
                    return confirmed
        return ""

    substrate = take("substrate", reading.substrate)
    location = take("location", reading.location)
    exposure = take("exposure", "")

    # The product this turn names, from the question itself first and from the
    # model only as a fallback. That order matters and the reverse was a real
    # gap: `deterministic()` never sets `explicit_product`, so on every turn the
    # model did not run -- which is most turns, by design -- `ResolvedRequest.
    # product` came back empty even when the question said "Duro" in plain
    # words. Retrieval lost its product constraint and the recommendation guard
    # lost the exemption that keeps it from refusing ordinary VERIFY questions.
    #
    # Longest match wins, the same rule `Assistant._named_product` uses, because
    # "Lime Green Ultra" and "Ultra" are both harvested names and the longer one
    # is the more specific claim.
    product = _named_in(question, registry)
    if not product and reading.explicit_product:
        product = _registered(reading.explicit_product, registry)
    if product:
        provenance["product"] = Provenance.STATED

    def origin_of(slot: str) -> Provenance:
        """How to describe a fact this turn inherited rather than heard.

        Derived from the turn it was stated in, compared against the turn being
        answered -- not from a conversion applied when state changes hands.

        The difference stopped being academic when the checkpointer became the
        source of continuity. `ConversationState.inherit()` did the conversion,
        and it is now applied only to seed a thread that has none, so from turn
        two onwards a fact read back out of the checkpoint was still marked
        ``STATED``. The answer then said "brick (substrate), as you said" about
        something said three messages ago -- a small wrongness in the one
        sentence whose entire job is to be exact about who said what.

        An observation does not age into testimony, so only ``STATED`` moves.
        """
        fact = state.facts[slot].current
        if (fact.provenance is Provenance.STATED
                and fact.source_turn and fact.source_turn < turn_index):
            return Provenance.CARRIED
        return fact.provenance

    inherited = state.active()
    for slot, value in inherited.items():
        if slot == "product" and not product:
            product = value
            provenance["product"] = origin_of(slot)
        elif slot == "substrate" and not substrate:
            substrate = value
            provenance["substrate"] = origin_of(slot)
        elif slot == "location" and not location:
            location = value
            provenance["location"] = origin_of(slot)
        elif slot == "exposure" and not exposure:
            exposure = value
            provenance["exposure"] = origin_of(slot)

    measurements = dict(reading.measurements)
    if measurements:
        provenance["measurements"] = Provenance.STATED

    resolved = ResolvedRequest(
        intent=reading.intent,
        policy_topic=reading.policy_topic,
        raw_question=question,
        product=product,
        objective=reading.objective.strip().lower()[:60],
        substrate=substrate,
        location=location,
        exposure=exposure,
        requested_properties=reading.requested_properties,
        measurements=measurements,
        # Hypotheses only. Each is normalised to a real name here and still has
        # to survive evidence assessment before it can be recommended.
        candidate_products=tuple(
            dict.fromkeys(filter(None, (_registered(p, registry)
                                        for p in reading.candidate_products)))),
        provenance=provenance,
        unsettled=tuple(state.unsettled()),
    )

    obs.event("state_resolution", intent=resolved.intent.value,
              slots=sorted(resolved.slots()),
              inherited=sorted(s for s in resolved.slots()
                               if provenance.get(s) in (Provenance.CARRIED,
                                                        Provenance.OBSERVED)),
              unsettled=list(resolved.unsettled),
              candidates=len(resolved.candidate_products))
    return resolved


def facts_from(resolved: ResolvedRequest, turn_index: int) -> dict[str, SessionFact]:
    """This turn's contribution to conversation state.

    Only what the person supplied in *this* turn becomes a new fact. Inherited
    values are already in the state and re-writing them would reset their
    source turn, making a value stated four turns ago look freshly confirmed.
    """
    return {slot: SessionFact(slot, value, Provenance.STATED, turn_index)
            for slot, value in resolved.slots().items()
            if resolved.provenance.get(slot) is Provenance.STATED}
