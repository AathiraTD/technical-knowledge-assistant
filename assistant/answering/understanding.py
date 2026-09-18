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

from ..infrastructure import observability as obs
from ..infrastructure import ollama
from .answer import (Answer, Provenance, SlotFact, _named_aliases,
                     _product_aliases, _without_brand)
from ..turn.conversation import ConversationState, Denial, FactStatus, SessionFact

# The model is asked for one small object and nothing else, so the budget is a
# fraction of an answer's. A reply longer than this is a malfunction, not a
# richer understanding.
MAX_UNDERSTANDING_TOKENS = 200

# Bounds on what may be lifted out of a question. A measurement past these is
# not a wall, and carrying it into a calculation would produce a confident
# absurdity.
MAX_AREA_M2 = 10_000.0
MAX_THICKNESS_MM = 500.0
MANDATORY_TOPICS = frozenset({"structural_judgement", "compliance_signoff", "health"})


class Intent(Enum):
    """What the person wants done, as distinct from how the answer is printed.

    `Path_` in `assistant/answering/router.py` is the *output* shape — extract, compose,
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
    # Building slots this turn supplied as part of a *scenario* rather than as
    # a description of the caller's own wall. They constrain this request
    # exactly as a stated slot does -- retrieval and routing cannot tell the
    # difference and must not -- and they are the one thing that does not
    # become a remembered fact. See `scenario_only`.
    transient: tuple[str, ...] = ()
    # Building slots this turn explicitly took back, mapped to the value being
    # withdrawn. Excluded from the request as well as from what is remembered:
    # a value the person has just denied must not shape the answer they are
    # about to read, and must not be inherited on the turns after it. See
    # `denials_in`.
    denied: dict = field(default_factory=dict)

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

# Words that ask the system to choose. "Suitable" is kept apart because it
# reads both ways: "which is suitable" chooses, "compare X and Y for suitable
# backgrounds" asks what each sheet publishes.
_CHOICE_SHAPE = re.compile(
    r"\b(which|what)\s+(product|plaster|render|mortar|system|one)\b"
    r"|\brecommend\b|\bshould i use\b|\bbest for\b|\bwhat do i (use|need)\b",
    re.I)
_SELECT_SHAPE = re.compile(_CHOICE_SHAPE.pattern + r"|\bsuitable\b", re.I)

_AREA = re.compile(r"(\d+(?:\.\d+)?)\s*(?:square\s*met(?:re|er)s?|m2|m²|sqm)", re.I)
_THICKNESS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm|millimet(?:re|er)s?)", re.I)

_NON_ASSERTION = re.compile(
    r"\b(?:not|never|no|if|unless|suppose|supposing|assuming|hypothetically|"
    r"might|maybe|perhaps|said|says|quote|example|imagine|imagining|"
    r"considering|consider|remember|recall|mention|mentioned|told)\b"
    r"|\b(?:other than|except|excluding|without|anything but)\b"
    r"|\b(?:i|we)\s+(?:would|could|may)\b|\b\w+n['’]t\b", re.I)
_QUOTED = re.compile(
    r'"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)\'[^\']*\'(?!\w)|`[^`]*`',
    re.S)
_BACKGROUND_LOOKUP = re.compile(
    r"\b(?:what|which)\s+(?:backgrounds?|substrates?|surfaces?)\b"
    r".*\b(?:suitable|compatible|appl(?:y|ied)|use[ds]?)\b", re.I)
_PRODUCT_REFERENCE = re.compile(
    r"\b(?:it|its|this product|that product|this plaster|that plaster)\b", re.I)
_COMPARISON = re.compile(
    r"\b(?:compare|comparison|versus|vs|difference|different|differ|"
    r"rather than|better than|instead of|or)\b", re.I)
_FACT_QUESTION = re.compile(
    r"^\s*(?:am|are|is|was|were|did|do|does|have|has|had)\s+(?:i|we|my|our)\b"
    r"|^\s*(?:is|was)\s+(?:the|this|that)\s+(?:wall|substrate|material|product)\b"
    r"|\b(?:what|which)\s+(?:product|substrate|material)\s+"
    r"(?:am|are|is|was|were|did)\b", re.I)


def reference_text(question: str) -> str:
    """Unquoted, nonhypothetical text that may name a lookup target.

    A semicolon or newline cannot end a conditional's scope. An independent
    sentence can, allowing a subsequent explicit correction to stand.
    """
    text = _QUOTED.sub(" QUOTED_CONTENT ", question)
    # An unmatched opening quote is still quoted input, not testimony.
    text = re.sub(r'(["“‘`]|(?<!\w)\').*', " QUOTED_CONTENT ", text, flags=re.S)
    return " ".join(part for part in re.split(r"(?<=[.!?])\s+", text)
                    if "QUOTED_CONTENT" not in part
                    and not _NON_ASSERTION.search(part)
                    and not _FACT_QUESTION.search(part))


def asserted_text(question: str) -> str:
    """Eligible fact context; comparing alternatives asserts neither."""
    return " ".join(part for part in re.split(
        r"(?<=[.!?])\s+", reference_text(question))
        if not _COMPARISON.search(part)
        and not re.match(r"\s*(?:can|could|would|should|may|might|will)\s+"
                         r"(?:i|we)\b", part, re.I)
        and not (part.rstrip().endswith("?")
                 and re.match(r"\s*(?:i|we|my|our)\b", part, re.I)
                 and not re.search(r"\b(?:what|which|how|why|where|when)\b",
                                   part, re.I))
        and not re.match(r"\s*(?:tell me about|explain|describe|"
                         r"what (?:is|are))\b", part, re.I))


_ALTERNATIVE = re.compile(r"\bor\b", re.I)
_EXPLICIT_COMPARISON = re.compile(
    r"\b(?:compare|comparison|versus|vs|difference|different|differ|"
    r"rather than|better than|instead of)\b", re.I)
# How far, in words, a product name may sit from an "or" and still be one of
# the things it weighs: exactly a brand-prefixed name ("Lime Green Duro"). Any
# wider and "Can Ultra go on brick or stone?" reads the substrates as products.
_ALTERNATIVE_WINDOW = 3


def compares_products(text: str, registry) -> bool:
    """Does this sentence weigh one product against another?

    `_COMPARISON` is the right filter for building facts, where "brick or
    stone" asserts neither substrate. It is the wrong one for the discussion
    target. "Curing or application conditions" lists two things asked about one
    product, and reading its "or" as a comparison left `topic_product` empty --
    so the recommendation guard in `assistant/turn/graph.py`, which exempts only the
    product the caller named, refused the answer's own mention of that product
    as an unapproved recommendation. A bare "or" therefore counts only when a
    product name sits within a few words of it; the explicit comparison words
    count anywhere.
    """
    if _EXPLICIT_COMPARISON.search(text):
        return True
    aliases = _product_aliases(registry)
    words = text.split()
    for index, word in enumerate(words):
        if not _ALTERNATIVE.fullmatch(word.strip(",.;:?!")):
            continue
        beside = words[max(0, index - _ALTERNATIVE_WINDOW):index] \
            + words[index + 1:index + 1 + _ALTERNATIVE_WINDOW]
        if _named_aliases(" ".join(beside), aliases):
            return True
    return False


def topic_product(question: str, registry) -> str:
    """One named discussion target; alternatives and recall establish none."""
    text = reference_text(question)
    return "" if compares_products(text, registry) else _named_in(text, registry)


def compared_products(question: str, registry) -> set[str]:
    """The products an explicit comparison sets side by side, or nothing.

    "Compare Ultra and Solo for suitable backgrounds" asks what each sheet
    publishes about each, not which to put on a wall. The router already holds
    that substrate is not load-bearing in a comparison; reading the same
    sentence as a selection here sent acceptance case T14 to an ask-back for a
    substrate nobody needed. A choice word keeps it a selection: "compare them
    and tell me which plaster I should use" still has to survive the evidence
    gate.
    """
    if _CHOICE_SHAPE.search(question) or not _EXPLICIT_COMPARISON.search(question):
        return set()
    named = _named_aliases(reference_text(question), _product_aliases(registry))
    return named if len(named) >= 2 else set()


def named_products(question: str, registry) -> set[str]:
    """Every product the question is about, in each spelling the registry holds.

    Spelling-complete on purpose. The recommendation guard compares these
    against `candidates.recommends_a_product`, which reports whichever registry
    spelling the answer used, and "ultra" must excuse "Ultra: Insulated Lime
    Plaster Base Coat" as readily as the reverse. Read through `reference_text`
    so a product named in order to be excluded is not one the caller asked
    about.
    """
    aliases = _product_aliases(registry)
    found = _named_aliases(reference_text(question), aliases)
    return {normalise_product(p) for p in registry
            if aliases.get(_without_brand((p or "").lower()).strip()) in found}


def stated_product(question: str, registry) -> str:
    """An actual use statement, not merely the product a question names."""
    for part in re.split(r"(?<=[.!?])\s+", asserted_text(question)):
        if re.search(
                r"^\s*(?:(?:actually|now|also)[, ]+)?"
                r"(?:(?:i|we)\s+(?:(?:am|are)\s+)?(?:use|using|apply|applying)"
                r"|i['’]m\s+(?:using|applying)|my product is)\b", part, re.I):
            return _named_in(part, registry)
    return ""


_ASKS_SOMETHING = re.compile(
    r"\?|\b(?:what|which|how|why|when|where|should|can|could|recommend|"
    r"suitable|tell|explain|give)\b", re.I)


def state_request_slot(question: str) -> str:
    """Recognise personal-state questions, not product selection or properties."""
    text = re.sub(r"\s+", " ", question.strip().rstrip("?.!")).lower()
    for slot, pattern in (
        ("product", r"(?:what|which) product (?:am i using|"
         r"did i (?:say|tell you)(?: that)? i (?:was|am) using)"),
        ("substrate", r"(?:what|which) (?:substrate|material) "
         r"(?:is my wall|did i (?:say|tell you)(?: that)? my wall was)"
         r"(?: (?:made|built) (?:of|from))?"),
        ("substrate", r"what (?:is my wall|did i say my wall was) "
         r"(?:made|built) (?:of|from)"),
    ):
        if re.fullmatch(pattern, text):
            return slot
    return ""


def needs_product_clarification(question: str, resolved: ResolvedRequest) -> bool:
    """Only unresolved product references block a property lookup, not materials."""
    return (not resolved.policy_topic and not resolved.product
            and bool(_PRODUCT_REFERENCE.search(question))
            and resolved.intent in (Intent.LOOKUP, Intent.CALCULATE, Intent.VERIFY)
            and bool(resolved.requested_properties
                     or _BACKGROUND_LOOKUP.search(question)
                     or re.search(r"\b(?:thick|thickness|apply|coverage|mix)\b",
                                  question, re.I)))


def state_only_answer(question: str, detector, registry,
                      state: ConversationState | None = None,
                      turn_index: int = 0, *, gate=None,
                      active_product: str = "") -> Answer | None:
    """Pure pre-model/pre-retrieval boundary shared by graph and library callers.

    Returns an uncached Answer for recall, personal fact acknowledgement, or an
    unresolved product reference; otherwise None. Does not mutate state. Callers
    persist ``facts_from(resolve(...), turn_index)`` through their state reducer.
    """
    state = state or ConversationState()
    from .router import PolicyGate

    if (gate or PolicyGate()).match(question):
        return None
    slot = state_request_slot(question)
    if slot:
        history = state.facts.get(slot)
        fact = history.current if history else None
        # A retracted value is history, not testimony. `merge_facts` retires a
        # denied fact by marking `current` superseded and asserting nothing in
        # its place -- which is why `ConversationState.active()` stops
        # inheriting it -- but reading `current` without its status reported
        # the withdrawn value straight back as "you said your substrate was
        # brick", one turn after the person had said it was not. The history
        # is deliberately still there; what changes is that recall no longer
        # speaks for it.
        #
        # `SUPERSEDED` rather than "not `ACTIVE`" on purpose: `CONFLICTING` is
        # also not active, and it has its own wording below -- a photograph
        # disagreeing with the person is something to confirm, not something
        # the person took back.
        if fact is not None and fact.status is FactStatus.SUPERSEDED:
            fact = None
        facts = []
        if fact is None or fact.provenance is Provenance.ASSUMED:
            text = f"You have not stated a {slot} in this current chat."
        elif fact.from_person:
            text = f"You said your {slot} was {fact.value.replace('_', ' ')}."
            facts = [SlotFact(slot, fact.value, Provenance.CARRIED)]
            if history.unsettled:
                text += " An image observation disagrees; please confirm it."
        else:
            text = (f"You have not stated a {slot} in this current chat. "
                    f"The photograph suggested {fact.value.replace('_', ' ')}; "
                    "that is an observation, not something you said.")
            facts = [SlotFact(slot, fact.value, fact.provenance)]
        return Answer(path="state", text=text, facts=facts,
                      observed=[f.sentence for f in facts
                                if f.provenance is Provenance.OBSERVED],
                      diagnostics={"step": "conversation state", "state_slot": slot,
                                   "slots": {f.slot: f.value for f in facts},
                                   "cached": False})

    reading = deterministic(question, detector, registry=registry)
    resolved = resolve(reading, question, detector, registry, state, turn_index,
                       active_product=active_product)
    if needs_product_clarification(question, resolved):
        return Answer(path="ask_back",
                      text="Which product are you using? Please give its name.",
                      facts=[SlotFact(name, value, resolved.provenance[name])
                             for name, value in resolved.slots().items()],
                      diagnostics={"step": "product reference", "missing": ["product"],
                                   "slots": resolved.slots(),
                                   "cached": False})

    # A retraction is a state instruction, and it has to be acknowledged here
    # or not at all. The acknowledgement below cannot see it: a sentence
    # containing "not" is emptied by `asserted_text`, by design, so `clean`
    # comes back blank and the turn falls through to retrieval -- which is how
    # "actually, my wall is not brick" came to be answered with the nearest
    # passage about external wall insulation. What this returns asserts
    # nothing, exactly as `Denial` does: the slot becomes unknown rather than
    # known to be something else, and the caller still persists the `Denial`
    # through `merge_facts`, which is the only thing that retires the fact.
    retractions = {slot: instruction
                   for slot, instruction in facts_from(resolved, turn_index).items()
                   if isinstance(instruction, Denial)}
    withdrawn = [
        slot for slot, instruction in retractions.items()
        if (held := state.facts.get(slot)) is not None
        and (not instruction.value or held.current.value == instruction.value)]
    if withdrawn and not _ASKS_SOMETHING.search(question):
        text = ("Noted for this chat: " + "; ".join(
            f"{slot} is no longer recorded" for slot in sorted(withdrawn)) + ".")
        return Answer(path="acknowledge", text=text, facts=[],
                      diagnostics={"step": "conversation state",
                                   "retracted": sorted(withdrawn),
                                   "slots": {},
                                   "cached": False})

    # An acknowledgement must not swallow an accompanying technical question.
    clean = asserted_text(question)
    personal = re.match(
        r"^\s*(?:(?:actually|now)[, ]+)?(?:i (?:am|have|use)|i['’]m|my\b)",
        clean, re.I)
    if (personal and clean.strip() == question.strip()
            and not re.search(r"\?|\b(?:what|which|how|why|when|where|"
                              r"should|can|please|recommend|tell|explain)\b",
                              question, re.I)):
        new = {slot: fact
               for slot, fact in facts_from(resolved, turn_index).items()
               if isinstance(fact, SessionFact)}
        if new:
            text = "Noted for this chat: " + "; ".join(
                f"{name}: {fact.value.replace('_', ' ')}"
                for name, fact in new.items()) + "."
            return Answer(path="acknowledge", text=text,
                          facts=[SlotFact(f.slot, f.value, f.provenance)
                                 for f in new.values()],
                          diagnostics={"step": "conversation state",
                                       "slots": {name: f.value for name, f in new.items()},
                                       "cached": False})
    return None


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

    from .router import PolicyGate

    matched = (gate or PolicyGate()).match(question)
    if matched:
        return TurnUnderstanding(
            intent=Intent.ESCALATE, policy_topic=matched[0],
            measurements=measured, source="deterministic")

    if state_request_slot(question):
        intent = Intent.LOOKUP
    elif "cause_asked" in slots or "symptom" in slots:
        intent = Intent.TROUBLESHOOT
    elif "calculation" in slots or measured:
        intent = Intent.CALCULATE
    elif _BACKGROUND_LOOKUP.search(question):
        intent = Intent.LOOKUP
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
        #
        # `topic_product` rather than `_named_in` on the raw question, because
        # a named product the sentence *excludes* is not a product the person
        # has chosen. "I'm using Ultra ... please don't give me figures for
        # Solo or Duro" names three, so `_named_in` found no single target,
        # returned "" and sent a question about one stated product to the
        # selection gate -- which asked for a substrate before it would choose
        # anything, and so refused two figures the Ultra datasheet publishes.
        # `topic_product` reads the same question through `reference_text`,
        # where a non-assertion clause establishes nothing, and it keeps the
        # comparison rule: "is Solo or Duro better" still names no target and
        # is still a selection.
        intent = (Intent.VERIFY
                  if topic_product(question, registry)
                  or compared_products(question, registry)
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
    and not a safety one -- the recommendation guard in `assistant/turn/graph.py`
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
    was measured: `assistant/retrieval/candidates.py` compared full registry names against
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
    found = _named_aliases(question, _product_aliases(registry))
    if len(found) != 1:
        return ""
    return normalise_product(next(iter(found)))


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


# The slots that describe the building rather than the shape of the question.
# Only these can be scenario-transient or denied: `product` is the subject under
# discussion rather than a claim about a wall, so "would Ultra be suitable..."
# legitimately leaves Ultra as the topic while leaving the wall unknown.
BUILDING_SLOTS = ("substrate", "location", "exposure")

# A question that opens in the conditional or interrogative *and* asks whether
# something is suitable. Both halves are required, and the pairing is what keeps
# this from being a general discourse parser: it recognises one very common
# shape -- "would X be suitable on a Y wall" -- and nothing else.
#
# It is deliberately narrower than `_NON_ASSERTION` and does a different job.
# That filter asks whether a *sentence* is testimony at all; this asks whether a
# sentence which is plainly the caller's own question is describing their
# building or naming the conditions of a catalogue enquiry. "Would Ultra be
# suitable on an internal brick wall?" passes `asserted_text` untouched -- it
# hedges nothing and quotes nobody -- and is still not a statement that the
# person has a brick wall.
_SCENARIO_ASK = re.compile(
    r"^\s*(?:would|could|can|is|are|does|do|will|shall|should|what\s+if|"
    r"suppose|hypothetically)\b", re.I)
_SUITABILITY = re.compile(
    r"\b(?:suitable|suited|appropriate|compatible|recommended|ok|okay|fine|"
    r"work|works|be\s+used|be\s+applied|go\s+on)\b", re.I)

# Anything that makes the sentence a claim about the caller's own building. Its
# presence is a veto: "is Ultra suitable on my internal brick wall" describes a
# real wall and the substrate is theirs to have remembered.
_OWN_BUILDING = re.compile(
    r"\b(?:my|our|mine)\b"
    r"|\bi\s+(?:have|am|was|ve|m)\b|\bi\s*[’']\s*(?:ve|m)\b"
    r"|\we\s+(?:have|are|ve)\b|\bi\s+got\b"
    r"|\bthe\s+wall\s+is\b|\bit\s+is\b|\bit\s*[’']s\b"
    r"|\bthey\s+are\b|\working\s+on\b", re.I)


def scenario_only(question: str) -> bool:
    """Is this turn asking about a hypothetical wall rather than describing one?

    The distinction exists because the two are indistinguishable to a slot
    detector and must not be to conversation state. "Would Ultra be suitable on
    an internal brick wall?" mentions brick the way a catalogue question
    mentions it -- as the condition being asked about. Recording it as a fact
    means every later turn is answered for a brick wall the person never said
    they had, and `Provenance.STATED` makes the answer print it back as "brick,
    as you said" -- attributing an invention to the customer, which is the same
    shape of failure the case boundaries in `assistant/turn/conversation.py` exist to
    prevent.

    **The two errors are not symmetric and this errs the safe way.** Treating a
    real description as a scenario costs the person one extra question later.
    Treating a scenario as a description answers the next four turns about the
    wrong wall. So the rule is deliberately narrow -- a conditional opener, a
    suitability word, and no first-person claim anywhere in the sentence -- and
    anything it does not recognise is remembered exactly as before.
    """
    return bool(_SCENARIO_ASK.match(question or "")
                and _SUITABILITY.search(question or "")
                and not _OWN_BUILDING.search(question or ""))


_NEGATOR = (r"(?:not|isn\s*[’']?\s*t|aren\s*[’']?\s*t|"
            r"wasn\s*[’']?\s*t|weren\s*[’']?\s*t|"
            r"no\s+longer|never)")


def _negation_count(question: str, term: str) -> tuple[int, int]:
    """How many occurrences of `term` are negated, and how many there are.

    Positional rather than a whole-sentence search, because a sentence can carry
    a negation and an assertion at once: "it is not render, it is brick" negates
    one term and asserts the other, and a search that only asked "does this
    sentence contain 'not'" would throw away the half that matters.

    The window is short and word-only on purpose. It spans "not a", "isn't the"
    and "is no longer", and stops at punctuation or a longer intervening
    phrase. That is what keeps this from firing on the sentences
    `_NON_ASSERTION` already refuses for a different reason: in "I am not
    applying 12.5 mm of Solo on a stone wall" the negation governs *applying*,
    eight words away from the substrate, so stone is not read as denied -- it is
    simply never asserted, which that filter had already established.
    """
    negated = total = 0
    for match in re.finditer(rf"\b{re.escape(term)}\b", question, re.I):
        total += 1
        before = question[max(0, match.start() - 30):match.start()]
        if re.search(rf"\b{_NEGATOR}\b[\s\w]{{0,7}}$", before, re.I):
            negated += 1
    return negated, total


def denials_in(question: str, detected: dict, detector) -> dict[str, str]:
    """Building slots this turn takes back rather than states.

    Read from the **raw** question rather than from `asserted_text`, which is
    the whole reason this is a separate stage. `_NON_ASSERTION` matches "not",
    so a denial is exactly the kind of sentence that filter removes -- correctly,
    because "my wall is not brick" asserts nothing. But removing it also made it
    inert: the fact stated three turns ago stayed active and went on being
    inherited into every later request. Silence and retraction are different
    inputs and only one of them leaves a belief standing.

    A denial is recognised only when *every* occurrence of the matched value's
    vocabulary is negated. One un-negated mention means the person is
    contrasting rather than retracting -- "it is not brick, it is stone" detects
    stone, and brick is then simply not the winning value -- and a partial
    reading would unsettle a slot the same sentence had just filled.

    What this returns is the value being withdrawn, not a new value: "not brick"
    says nothing about what the wall *is*. `Denial` in
    `assistant/turn/conversation.py` turns that into a state change, so the
    supersession rule stays in the one place that owns it.
    """
    denied: dict[str, str] = {}
    for slot in BUILDING_SLOTS:
        value = detected.get(slot)
        if not value:
            continue
        negated = present = 0
        for term in detector.terms_for(slot, value):
            found, total = _negation_count(question, term)
            negated += found
            present += total
        if present and negated == present:
            denied[slot] = value
    return denied


def resolve(reading: TurnUnderstanding, question: str, detector, registry,
            state: ConversationState | None = None,
            turn_index: int = 0, *, active_product: str = "") -> ResolvedRequest:
    """Validate this turn's reading, then merge what the conversation knows.

    Merge order is the same one the router has always used and the same one
    `assistant/turn/conversation.py` enforces: **this turn over history**. A value
    stated now beats a value remembered, because a person correcting themselves
    must not be answered from the thing they just corrected.

    A slot the conversation holds as `CONFLICTING` -- a photograph disagreeing
    with something the person said -- is deliberately *not* merged in. It is
    reported in `unsettled` instead, so the requirement gate can ask about it
    rather than pick a winner.
    """
    state = state or ConversationState()
    asserted = asserted_text(question)
    detected = detector.detect(asserted)
    # Denials are read from the raw question, because `asserted_text` removes
    # the sentence that carries one. See `denials_in`.
    denied = denials_in(question, detector.detect(question), detector)
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
        if slot in denied:
            # Named in this sentence, and negated in it. Neither the detector
            # nor the model's reading can tell those apart; this is where the
            # difference is made.
            return ""
        if slot in detected:
            provenance[slot] = Provenance.STATED
            return detected[slot]
        if model_value:
            word = model_value.strip().lower()
            if word and re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)",
                                  asserted, re.I):
                confirmed = detector.detect(model_value).get(slot, "")
                if confirmed:
                    provenance[slot] = Provenance.STATED
                    return confirmed
        return ""

    substrate = take("substrate", reading.substrate)
    location = take("location", reading.location)
    exposure = take("exposure", "")

    # Building slots this turn supplied, captured here rather than after the
    # inheritance loop below so an inherited value can never be mistaken for
    # one the sentence carried. If the sentence is a scenario rather than a
    # description, these constrain the request and nothing more.
    transient = (tuple(slot for slot in BUILDING_SLOTS
                       if provenance.get(slot) is Provenance.STATED)
                 if scenario_only(question) else ())

    # The product this turn names must occur in eligible question text; a model
    # suggestion alone cannot establish it. That order matters and the reverse was a real
    # gap: `deterministic()` never sets `explicit_product`, so on every turn the
    # model did not run -- which is most turns, by design -- `ResolvedRequest.
    # product` came back empty even when the question said "Duro" in plain
    # words. Retrieval lost its product constraint and the recommendation guard
    # lost the exemption that keeps it from refusing ordinary VERIFY questions.
    #
    # Longest match wins, the same rule `Assistant._named_product` uses, because
    # "Lime Green Ultra" and "Ultra" are both harvested names and the longer one
    # is the more specific claim.
    product = topic_product(question, registry)
    if product:
        provenance["product"] = (
            Provenance.STATED if stated_product(question, registry) == product
            else Provenance.ASSUMED)
    elif active_product and not compares_products(question, registry):
        product = active_product
        provenance["product"] = Provenance.ASSUMED

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
        if slot in denied:
            # Retracted in this very sentence. The reducer is about to retire
            # it, and inheriting it here would answer this turn from the value
            # the person has just withdrawn -- which is what made the denial
            # inert rather than merely unrecorded.
            continue
        if (slot == "product" and not compares_products(question, registry)
                and (not product or (product == value
                     and provenance.get("product") is Provenance.ASSUMED))):
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

    measurements = measurements_in(asserted)
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
        transient=transient,
        denied=dict(denied),
    )

    obs.event("state_resolution", intent=resolved.intent.value,
              slots=sorted(resolved.slots()),
              inherited=sorted(s for s in resolved.slots()
                               if provenance.get(s) in (Provenance.CARRIED,
                                                        Provenance.OBSERVED)),
              unsettled=list(resolved.unsettled),
              transient=list(resolved.transient),
              denied=sorted(resolved.denied),
              candidates=len(resolved.candidate_products))
    return resolved


def facts_from(resolved: ResolvedRequest,
               turn_index: int) -> dict[str, SessionFact | Denial]:
    """This turn's contribution to conversation state.

    Only what the person supplied in *this* turn becomes a new fact. Inherited
    values are already in the state and re-writing them would reset their
    source turn, making a value stated four turns ago look freshly confirmed.

    Two things this turn said are deliberately not facts about the building.

    A **scenario constraint** shaped the request and stops there. "Would Ultra
    be suitable on an internal brick wall?" is a question about a hypothetical
    wall, and `resolved.substrate` is set so retrieval and routing can use it --
    but writing it here would make every later turn an answer about a brick wall
    the person never said they had, printed back as "brick, as you said".

    A **denial** produces a `Denial` rather than a fact, and is emitted before
    the eligibility checks below rather than after. That order is the point: a
    retraction contains "not", so `asserted_text` empties it and every one of
    those checks would refuse it -- correctly reading it as no testimony, and
    thereby leaving the belief it contradicts standing. Silence and retraction
    are different inputs.

    Denials are revalidated here rather than trusted from `resolved`, for the
    same reason every value below is: a caller can construct a
    `ResolvedRequest` directly, and a forged field must not be able to retire a
    fact any more than it can create one.
    """
    from .router import SlotDetector

    detector = SlotDetector()
    out: dict[str, SessionFact | Denial] = {
        slot: Denial(slot, value) for slot, value in denials_in(
            resolved.raw_question,
            detector.detect(resolved.raw_question), detector).items()}

    asserted = asserted_text(resolved.raw_question)
    if state_request_slot(resolved.raw_question) or not asserted.strip():
        return out
    # Revalidate every value: callers can construct ResolvedRequest directly,
    # and a harmless second sentence must not legitimise a rejected first one.
    grounded = detector.detect(asserted)
    product = stated_product(resolved.raw_question, (resolved.product,))
    if product:
        grounded["product"] = product
    for slot, value in resolved.slots().items():
        if slot in resolved.transient or slot in out:
            continue
        if (resolved.provenance.get(slot) is Provenance.STATED
                and grounded.get(slot) == value):
            out[slot] = SessionFact(slot, value, Provenance.STATED, turn_index)
    return out
