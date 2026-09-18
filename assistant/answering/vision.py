"""Perception only: what can be seen in a photograph, and nothing about what to do.

DECISIONS 16.1 splits the image path into three stages — perception, then
interpretation, then recommendation — and names the rule that makes the split
worth having: *the vision model never names a product*. This module is the
first stage and only the first stage. It reads pixels into structured
observations, resolves those observations onto slots the existing router
already understands, and stops. It does not retrieve, does not compose, does
not diagnose and does not recommend.

Three properties carry that guarantee, and each one is a line of code rather
than a line of prose:

1. **The attribute is an enum, not free text.** The model is constrained to a
   JSON schema whose `attribute` field may only be one of the four slots vision
   is allowed to fill (`substrate`, `location`, `exposure`, `symptom`). There
   is no `product` attribute for it to fill in, and no `advice` field to write
   into.

2. **The value must already exist in the vocabulary.** `resolve()` compares an
   observed value against `config/vocabularies.json` — the same file the
   router's own `SlotDetector` reads — and discards anything that does not map
   onto a value the vocabulary already defines. A model that answers
   `"substrate": "Lime Green Solo"` produces no slot at all, because "solo" is
   not a substrate the vocabulary knows. The vision model therefore cannot
   invent a substrate, and cannot smuggle a product name in as one.

3. **The free-text fields are never read by the resolver.** `observation` and
   `possible_interpretations` are carried for a human reading the hand-off.
   `resolve()` looks at `attribute`, `value`, `confidence` and `region` and at
   nothing else, so no sentence the model writes can reach a slot.

What this module does *not* change is decision 16 itself. Filling a substrate
slot from a photograph is perception. Deciding what is wrong with the wall is
not, and the router's step 3 still sends a cause question to the hand-off. That
is why `cause_asked` is absent from `VISION_SLOTS`: the model may report that
it can see white crystalline deposits (a symptom), and may not report that the
wall has rising damp (a cause).

**Failure reduces coverage, not safety.** Nothing here raises. An unreachable
Ollama, a malformed response, an unreadable file or a response that does not
match the schema all produce an empty `Perception` carrying an `error`, which
resolves to an empty slot dict, which leaves the router exactly where it would
have been had no photograph been sent — asking back for the substrate.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from dataclasses import dataclass, field, replace as field_replace
from enum import Enum
from functools import lru_cache
from pathlib import Path

import httpx

from ..infrastructure.ollama import HOST, KEEP_ALIVE, GENERATION_MODEL, NUM_CTX
from .router import SlotDetector

# --------------------------------------------------- what may be reported
#
# Three tiers, and the tier an attribute sits in is a safety decision rather
# than a taxonomy. Everything in all three tiers is *reported* to the caller;
# only the first tier is allowed to change what the system does.

# **Tier 1 — may fill a router slot.** An attribute here can change which
# product is recommended, so it is the shortest list that is defensible.
#
# `substrate` is here because a photograph of genuinely exposed masonry does
# settle it, and it is the slot the whole selection turns on. It is also
# gated: see `_covered_without_exposure`.
#
# `symptom` is here because a symptom is a *visible condition* -- white
# deposits, cracking, a blown patch. That is perception. What it emphatically
# is not is a cause, which is why `cause_asked` is absent from every tier: a
# cause is a judgement the technical team makes and stands behind (decision
# 16), and a model able to fill it would route a diagnosis question away from
# the hand-off it must take.
ROUTER_SLOTS: tuple[str, ...] = ("substrate", "symptom")

# **Tier 2 — reported, never routed.** These were tier 1 until the first real
# photograph went through the real model, and moving them is the single most
# important change this module has had.
#
# Asked to describe a flat elevation of brickwork, `qwen3.5:4b` returned
# `{"attribute": "exposure", "value": "visible", "confidence": 1.0}` -- twelve
# times. Not one of those strings is a value the vocabulary defines, so nothing
# reached a slot and the safety property held. But it makes the point that no
# amount of prompt work removes: **a photograph does not carry these facts, so
# a model reporting them is producing text, not evidence.**
#
# `location` is the clearest case and the one the spec names first. Whether a
# wall is inside or outside is decided by things that are usually out of frame
# -- a skirting board, a sky, a ground line -- and getting it wrong means
# recommending an external render for a bedroom. It is also a load-bearing slot
# under decision 10, so leaving it uncued produces an ask-back, which is a
# known-good outcome. `exposure` is worse still: severe/sheltered is a fact
# about a site's weather, and no single frame contains it.
#
# So both are read, both are shown to the caller as context, and neither
# reaches the router. The person is asked instead.
CONTEXT_SLOTS: tuple[str, ...] = ("location", "exposure")

# **Tier 3 — directly observable wall condition.** The things a photograph is
# actually good for, and which `conversation.OBSERVABLE_SLOTS` already declared
# an intention to carry before anything produced them. These never route
# either; they are what the assistant reports back when somebody asks "what can
# you reliably identify from the photo", and two of them (`exposed_masonry`,
# `existing_finish`) additionally gate the substrate claim.
CONDITION_ATTRIBUTES: tuple[str, ...] = (
    "exposed_masonry", "existing_finish", "damaged_finish",
    "cracks", "staining", "texture",
)

# Everything the model may put in the `attribute` field, and therefore the
# whole of the schema's enum. Note what is still absent, and permanently:
# `product`, `cause_asked`, and anything naming structure, compliance or
# chemistry. The enum is the first half of the never-names-a-product guarantee;
# `resolve()` is the second.
VISION_SLOTS: tuple[str, ...] = ROUTER_SLOTS + CONTEXT_SLOTS + CONDITION_ATTRIBUTES

# Claims a photograph may never carry, whatever the model writes. The schema's
# enum already makes these unreachable, so this list is the belt to that
# braces: it is checked in the decoder, counted, and surfaced in `refused`, so
# a model that ignores its own schema is *visible* rather than merely
# unsuccessful. Silence about an attack that failed is how the next one
# succeeds unnoticed.
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
    "product", "products", "recommendation", "remedy", "treatment",
    "cause", "cause_asked", "diagnosis", "damp_cause", "moisture_source",
    "structural", "structure", "structural_safety", "load_bearing",
    "compliance", "building_regulations", "regulations", "certification",
    "mortar_chemistry", "mortar_mix", "chemistry", "composition",
    "hidden_substrate", "wall_construction", "compatibility",
})

# A photograph from a phone is a few megabytes. Anything much larger is either
# a mistake or an attempt to make the machine do arithmetic on a gigabyte, and
# both are refused at the boundary rather than sent to a model.
MAX_IMAGE_BYTES = int(os.environ.get("VISION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))

VISION_MODEL = os.environ.get("VISION_MODEL", GENERATION_MODEL)

# **Must equal `assistant.infrastructure.ollama.generate`'s `num_ctx`.** Not "be large
# enough" — equal. Two different context sizes for the same model are two
# resident instances, and the failure that produces is described in full beside
# the request body in `observe`.
#
# So it is now the same constant rather than a second copy of the number.
# It was previously read from its own `VISION_NUM_CTX` environment variable
# while the text path kept a literal, which left the invariant true only by
# default: one `export` moved this size and not the other, and the hang came
# back. `OLLAMA_NUM_CTX` moves both or neither. The module still owns its own
# HTTP client; it already imported the host, keep-alive and model tag from
# `ollama.py`, so this adds no dependency that was not there.
VISION_NUM_CTX = NUM_CTX

# --------------------------------------------------- the demonstration flag
#
# `ASSISTANT_VISION_DEMO=1` turns image reading on. **Off is the default, and
# off is a supported state rather than a broken one** -- it is exactly decision
# 16's published behaviour: the photograph is detected, the assistant says it
# cannot see it, and the enquiry goes to a person. Nothing degrades except
# coverage.
#
# Defaulting to off is the right way round for three reasons that are worth
# stating, because "new capability, therefore on" is the obvious instinct.
# Perception costs one to three minutes per image on a processor with no
# graphics card, so an accidental upload on a shared deployment is a
# denial-of-service with good intentions. The vision model is a separate pull
# that an operator may simply not have. And decision 16's argument -- that a
# confident wrong visual reading is the worst failure this system can produce
# -- has not been retired by making perception work; it is why every reading
# is hedged and why `location` still cannot route.
#
# Read through `enabled()` rather than captured here, so a process can be
# started with it set and a test can turn it on and off without reimporting
# the module.
VISION_DEMO_FLAG = "ASSISTANT_VISION_DEMO"

_TRUE = {"1", "true", "yes", "on", "enabled"}


def enabled(environ=None) -> bool:
    """Is image reading switched on for this process?

    The single answer to that question. A surface, the graph and a readiness
    check must not each decide it from their own reading of the environment --
    that is how a page comes to say it read a photograph that nothing looked
    at, which is the specific bug the graph's `analyse_images` already carries
    a comment about.
    """
    raw = (environ if environ is not None else os.environ).get(
        VISION_DEMO_FLAG, "")
    return str(raw).strip().lower() in _TRUE


# What a surface says when a photograph arrives and vision is switched off.
# Phrased as decision 16's hand-off rather than as an error, because that is
# what it is: the system is doing the published thing, not failing to do a
# different one.
DISABLED_NOTE = (
    "Image reading is switched off on this deployment, so the photograph was "
    "not looked at. Tell me what the wall is built of and whether it is "
    "inside or outside and I can answer from the published material."
)

# Long, because a vision encoder on a processor with no graphics card is slow.
# Measured, not estimated: one 512x512 image through `qwen3.5:4b` on the build
# machine took **191.9 seconds**. That is the same order as the 115-199 s the
# text path costs on an unseen question, and it is the reason this stage is
# roadmap rather than something to put in front of a customer on this hardware.
VISION_TIMEOUT = float(os.environ.get("VISION_TIMEOUT", "600"))

# A ceiling on generated tokens, and a ceiling on observations kept.
#
# Both exist because of what the first real call actually did. Asked to
# describe a 512x512 image, `qwen3.5:4b` fell into a repetition loop and
# emitted 37 near-identical observations — "fine horizontal lines", "fine
# vertical lines", "fine diagonal lines", on and on — each one reporting a
# confidence of 0.95. Nothing reached a slot, because none of those strings is
# a value the vocabulary defines, so the safety property held. What did not
# hold was the cost: an unbounded response is unbounded latency on a stage
# already measured in minutes.
#
# The loop is also the best argument this module has for Band. A model
# repeating itself thirty-seven times and stamping 0.95 on every repetition is
# not right 95 per cent of the time, and any design that read that float as a
# probability would have been reading noise.
# Both were raised and lowered respectively after the first real photograph, and
# the pair of changes is one fix rather than two.
#
# 512 tokens was not enough to *finish*. Ollama pretty-prints a constrained
# response, so one observation costs about forty tokens -- a region alone is six
# lines -- and twelve of them cannot fit. The response was therefore cut off
# mid-object on every real image, `json.loads` refused the fragment, and the
# whole 204-second call returned nothing at all. Raising the ceiling alone would
# have bought a complete answer by paying for the repetition; lowering the cap
# alone would have truncated sooner. Together they leave headroom: six
# observations at forty tokens is under three hundred, inside a budget of nine
# hundred, with room for a model that indents more than expected.
#
# Six is also the right number on its own terms. There are fourteen attributes
# and no photograph honestly settles fourteen things; a model reporting all of
# them is padding, and padding is what the cap is for.
MAX_VISION_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "900"))
MAX_OBSERVATIONS = int(os.environ.get("VISION_MAX_OBSERVATIONS", "6"))

# Against the repetition loop, at the sampler rather than in the prompt.
# Temperature zero makes the model deterministic, not sensible: with no
# penalty, the highest-probability continuation after an observation is often
# the same observation again, and greedy decoding takes it every time. This is
# the one option that acts on that directly. Kept mild -- a large penalty would
# start discouraging the legitimately repeated *structure* of a JSON array.
VISION_REPEAT_PENALTY = float(os.environ.get("VISION_REPEAT_PENALTY", "1.15"))


class Band(str, Enum):
    """Confidence as a routing band, not as a probability.

    DECISIONS 16.1 is explicit about why, and it is worth restating because the
    number looks so much like a probability that it invites being used as one:
    *a vision model reporting 0.91 is not correct 91 per cent of the time*. The
    figure is an uncalibrated self-report. Treating it as a probability would
    let a false-precision threshold — 0.85 rather than 0.83 — carry a decision
    that the measurement cannot support.

    So the number is collapsed into three bands the moment it arrives, and only
    the band is allowed to decide anything. `HIGH` may fill a slot. `UNCERTAIN`
    and `NOT_DETERMINABLE` leave the slot uncued, which the router's existing
    load-bearing-slot rule already handles by asking back — the behaviour the
    system has today with no photograph at all.

    The cut points below are a declared starting position, not a measurement.
    16.1 says the real thresholds come from labelled examples through the
    threshold sweep the harness already does, and no such labels exist: the
    failure library is recorded in the data inventory as not existing. When it
    does, the bands are what get re-cut, and nothing downstream changes,
    because nothing downstream reads the float.
    """

    HIGH = "HIGH"
    UNCERTAIN = "UNCERTAIN"
    NOT_DETERMINABLE = "NOT_DETERMINABLE"


HIGH_FLOOR = 0.80
UNCERTAIN_FLOOR = 0.50
# Splits the middle band for reporting only. Nothing routes on it: a reading at
# 0.7 and a reading at 0.55 are both refused a slot, and this only decides
# whether the caller reads "likely" or "uncertain" beside it.
LIKELY_FLOOR = 0.65


def band_for(confidence: float) -> Band:
    """The band a self-reported confidence falls in. The only reader of the float."""
    if confidence >= HIGH_FLOOR:
        return Band.HIGH
    if confidence >= UNCERTAIN_FLOOR:
        return Band.UNCERTAIN
    return Band.NOT_DETERMINABLE


# The float a named band is carried as, once the model names one directly.
#
# `Observation.confidence` stays a float because two things still read it that
# a band cannot answer: `resolve()` orders competing readings of the same
# attribute by it, and `perception_report` prints it. Using the band's own
# floor keeps the mapping honest in both directions -- `band_for` returns the
# band it came from, exactly, and what the reader sees is "at least this band"
# rather than a precision the model never offered.
BAND_FLOOR = {
    Band.HIGH: HIGH_FLOOR,
    Band.UNCERTAIN: UNCERTAIN_FLOOR,
    Band.NOT_DETERMINABLE: 0.0,
}

# The range read as a percentage rather than refused. `qwen3.5:4b` answered
# `100` where the contract asked for 0 to 1, and a reading of 100 is not a
# malformed 1.0 -- it is the same claim on the scale a person would write.
#
# **Whole numbers only, and from 2 up.** The bound is not tidiness; it is the
# difference between reading an answer and inventing one. A percentage is
# written as a whole number -- 90, 95, 100 -- whereas a value like 1.4 is a
# probability that has overshot, and rescaling *that* would turn a malformed
# 1.4 into a confident-looking 0.014 and file it as NOT_DETERMINABLE. Refusing
# it is what the old guard did and it was right to. Starting at 2 leaves 1.0
# unambiguously a probability, which is what it has always been.
PERCENT_FLOOR = 2
PERCENT_CEILING = 100


class Certainty(str, Enum):
    """How firmly the system holds one attribute, in the four words it reports.

    `Band` collapses the model's self-reported float; `Status` records whether
    several images agreed. This is the *reader's* view, and it exists because
    neither of those answers the question a customer actually asks -- "what can
    you reliably identify from the photo" -- in language anybody would use.

    The four levels are ordered, and the ordering is the safety property: only
    ``OBSERVED`` may change what the system does. Everything below it is
    information for a person to weigh, printed with its own hedge, and left out
    of the router entirely.

    ``OBSERVED``          directly visible, auditable to a region, top band,
                          and not contradicted by another image. The only level
                          that may fill a router slot.
    ``LIKELY``            visible, but not firmly enough to act on alone: the
                          middle band, or a top-band reading with no usable
                          region to check it against.
    ``UNCERTAIN``         a weak reading, or two confident readings that
                          disagree. Reported with the disagreement showing.
    ``CANNOT_DETERMINE``  the image does not settle it. Reached by a reading
                          below the floor, and by the model naming the
                          attribute in `cannot_determine_from_image`.

    ``CANNOT_DETERMINE`` is a *result*, not a failure, and that is the whole
    argument of decision 16 in one enum member. A wall that has been rendered
    hides its own construction; the honest output is that sentence, not a
    guess with a number on it.
    """

    OBSERVED = "OBSERVED"
    LIKELY = "LIKELY"
    UNCERTAIN = "UNCERTAIN"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"


class Status(str, Enum):
    """What the resolver concluded about one attribute across several images.

    16.1's first named refinement: three photographs may disagree — brick 0.82,
    stone 0.61, brick 0.91 — and the later image must not simply overwrite the
    earlier. Disagreement at the top band is `CONFLICTING`, and a conflicting
    attribute fills no slot, because two confident contradictory readings are
    less informative than one, not more.
    """

    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTING = "CONFLICTING"
    NOT_DETERMINABLE = "NOT_DETERMINABLE"


@dataclass(frozen=True)
class Observation:
    """One auditable visual claim.

    The contract 16.1 sets out: a value, a confidence, the image it came from,
    and the region within it. The region is to a visual claim what a cited
    passage is to a textual one — it is what lets a technical advisor see which
    pixels produced the claim — so an observation without a usable region is
    kept for the human to read and is not allowed to fill a slot.

    `image` is set by this module from the caller's own identifier, never read
    out of the model's response. Provenance that the model could write is not
    provenance.
    """

    attribute: str
    value: str
    confidence: float
    image: str
    region: tuple[float, float, float, float] | None = None
    observation: str = ""
    possible_interpretations: tuple[dict, ...] = ()

    @property
    def band(self) -> Band:
        return band_for(self.confidence)

    @property
    def auditable(self) -> bool:
        return self.region is not None


@dataclass(frozen=True)
class Perception:
    """Everything one call to the vision model produced, including its failures.

    `cannot_determine_from_image` is required rather than optional, and that is
    the point of it. Forcing the model to enumerate what it cannot tell from a
    photograph is the visual equivalent of refusing: a wall that has been
    rendered hides its own substrate, and a model that never says so will
    invent one. An empty list from a real response is accepted — the model may
    genuinely believe it saw everything — but the field must be present, and a
    response missing it is a schema failure, not a silent default.
    """

    observations: tuple[Observation, ...] = ()
    cannot_determine_from_image: tuple[str, ...] = ()
    image: str = ""
    model: str = ""
    seconds: float = 0.0
    error: str = ""
    # The response stopped at the token ceiling. What was recovered is real and
    # is reported; none of it may fill a slot, because a model that runs to the
    # ceiling is looping and a looping model's readings are not evidence.
    truncated: bool = False
    # Attributes the model tried to report that no photograph may carry.
    refused: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass(frozen=True)
class ResolvedAttribute:
    """One attribute after aggregation, with the observations that produced it."""

    slot: str
    value: str
    status: Status
    confidence: float
    sources: tuple[Observation, ...] = field(default=())
    # Why this attribute did not reach a slot, when it did not. Carried so a
    # hand-off can say "the render hides the background" rather than simply
    # omitting the substrate and leaving the caller to wonder whether the
    # photograph was looked at.
    withheld: str = ""

    @property
    def certainty(self) -> Certainty:
        """The four-level reading, derived. See `Certainty`."""
        if self.status is Status.CONFIRMED and not self.withheld:
            return Certainty.OBSERVED
        if self.status is Status.CONFLICTING:
            return Certainty.UNCERTAIN
        if self.status is Status.NOT_DETERMINABLE:
            return Certainty.CANNOT_DETERMINE
        # Everything left is a reading the system declined to act on. A
        # top-band reading that lost its slot for a mechanical reason -- no
        # usable region, or a covering finish over the substrate -- is still a
        # strong reading, and calling it merely "uncertain" would understate
        # what the model saw.
        if self.confidence >= LIKELY_FLOOR:
            return Certainty.LIKELY
        return Certainty.UNCERTAIN

    @property
    def routed(self) -> bool:
        """Did this attribute actually change what the system does?"""
        return self.certainty is Certainty.OBSERVED and not self.withheld


@dataclass(frozen=True)
class Resolution:
    """The profile resolver's output: slots to hand over, and why.

    Four collections rather than one, and the split is the safety boundary made
    visible. `slots` is the only one anything downstream routes on; the rest
    exist so the assistant can answer "what can you reliably identify from the
    photo" without that answer widening what it acts on.

    `slots`       slot to vocabulary value, ready for
                  `Assistant.ask(question, carried=...)`. Only ``OBSERVED``
                  tier-1 attributes appear.
    `context`     tier-2 readings -- location, exposure. Reported, never
                  routed, however confident. See `CONTEXT_SLOTS`.
    `conditions`  tier-3 readings -- what the surface actually looks like.
    `attributes`  every resolved attribute across all three tiers, including
                  the ones that were withheld and why.
    """

    slots: dict[str, str]
    attributes: tuple[ResolvedAttribute, ...] = ()
    cannot_determine_from_image: tuple[str, ...] = ()
    discarded: tuple[str, ...] = ()
    context: tuple[ResolvedAttribute, ...] = ()
    conditions: tuple[ResolvedAttribute, ...] = ()
    # Attributes the model tried to report that no photograph may carry. Empty
    # in the ordinary case; non-empty is worth an operator's attention, because
    # it means a model ignored its own schema.
    refused: tuple[str, ...] = ()
    # Did any perception run out of tokens mid-answer? A truncated response is
    # a looping model, and a looping model's readings do not fill slots.
    truncated: bool = False

    # There is deliberately no `summary()` or `reportable()` here.
    #
    # Both existed and both were replaced by `perception_report`, which has to
    # tolerate an injected provider that is not this class. Leaving them would
    # have left **two** copies of the certainty phrase table -- one keyed on
    # the enum, one on its value -- and two renderings of the same reading that
    # nothing forced to agree. A drift between them would show up as a
    # photograph described one way on the page and another in the log, which is
    # the sort of discrepancy that costs an afternoon to chase and is free to
    # prevent by having one renderer.


# --------------------------------------------------------------- the schema

# Ollama constrains a response to a JSON schema through `format`. 16.1's last
# refinement asks for exactly this rather than parsing prose out of a vision
# model, and it is the first half of the never-names-a-product guarantee: there
# is no field in this schema a product name belongs in. The second half is
# `resolve()`, which checks the values anyway, because a schema constrains
# shape and not meaning — nothing here stops a model writing "Solo" into
# `value`, and the vocabulary check is what makes that harmless.
def observation_schema(slots: tuple[str, ...] = VISION_SLOTS) -> dict:
    """The JSON schema the model is constrained to.

    `confidence` is the band by name and not a number, and that is the one
    field in here whose type was chosen by measurement rather than by taste.

    Asked to read a photograph of an exposed brick wall, `qwen3.5:4b` returned
    two correct observations -- `exposed_masonry: yes` and
    `existing_finish: render` -- and stamped **`"confidence": 100`** on both. A
    schema typing that field as `number` accepts 100 happily: constrained
    decoding enforces *types*, not ranges, so `minimum`/`maximum` would have
    bought nothing. `_decode_observation` then dropped both readings on its
    `0.0 <= c <= 1.0` guard, and a perception step that had in fact read the
    wall correctly reported nothing at all.

    An enum is the fix that makes the failure structurally impossible rather
    than caught after the fact, because an enum is exactly what a grammar *can*
    enforce. It is also what `Band` has always said the contract should be:
    "nothing downstream reads the float". If nothing reads it, asking for it is
    risk with no buyer.

    The float remains accepted on the way in -- see `_decode_observation` --
    because a model that ignores the enum and answers 0.9 is still telling us
    something usable, and because every fixture written against the old shape
    is still a valid response.
    """
    return {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string"},
                        "attribute": {"type": "string", "enum": list(slots)},
                        "value": {"type": "string"},
                        "confidence": {"type": "string",
                                       "enum": [b.value for b in Band]},
                        "region": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["observation", "attribute", "value",
                                 "confidence", "region"],
                },
                "maxItems": MAX_OBSERVATIONS,
            },
            "cannot_determine_from_image": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["observations", "cannot_determine_from_image"],
    }


# The tier-3 vocabulary. Held here rather than in `config/vocabularies.json`
# because these attributes describe a *surface*, and the router has no slot for
# any of them -- adding them to the router's file would imply a routing
# behaviour that deliberately does not exist. The same rule still applies as
# for the router slots: a value that matches nothing here is discarded, so the
# model cannot invent a condition any more than it can invent a substrate.
CONDITION_VALUES: dict[str, dict[str, list[str]]] = {
    "exposed_masonry": {
        "yes": ["exposed", "bare", "uncovered", "open masonry", "visible masonry",
                "exposed masonry", "bare brick", "bare stone", "fully exposed"],
        "partial": ["partial", "partially", "in places", "patches", "some areas",
                    "partly exposed", "patchy"],
        "no": ["none", "not exposed", "covered", "concealed", "hidden",
               "no exposed masonry", "fully covered"],
    },
    "existing_finish": {
        "render": ["render", "rendered", "rendering", "external render",
                   "cement render", "pebbledash", "roughcast"],
        "plaster": ["plaster", "plastered", "skim", "skimmed", "plasterwork"],
        "paint": ["paint", "painted", "emulsion", "masonry paint", "limewash",
                  "coating", "coated"],
        "tile": ["tile", "tiled", "tiling", "tiles"],
        "none": ["none", "bare", "unfinished", "no finish", "no coating",
                 "uncoated", "untreated"],
    },
    "damaged_finish": {
        "yes": ["damaged", "loose", "blown", "failing", "failed", "missing",
                "spalling", "delaminating", "flaking", "peeling", "hollow",
                "coming away", "fallen away", "broken away"],
        "no": ["sound", "intact", "undamaged", "good condition", "no damage",
               "well adhered", "firm"],
    },
    "cracks": {
        "crazing": ["crazing", "crazed", "map cracking", "map cracks",
                    "hairline", "fine cracks", "network of cracks",
                    "interlinked cracks", "surface cracking"],
        "linear": ["crack", "cracks", "cracked", "linear crack", "single crack",
                   "vertical crack", "horizontal crack", "diagonal crack",
                   "step crack"],
        "none": ["none", "no cracks", "uncracked", "no cracking"],
    },
    "staining": {
        "white": ["white", "white deposit", "white deposits", "bloom",
                  "efflorescence", "salt", "salts", "crystalline",
                  "pale deposits", "whitish"],
        "dark": ["dark", "dark patch", "dark patches", "darker", "black",
                 "discoloured", "discolored", "shadowed area", "grey patch"],
        "patchy": ["patchy", "patches", "uneven colour", "uneven color",
                   "mottled", "blotchy", "variegated", "shading"],
        "none": ["none", "no staining", "clean", "unstained",
                 "no discolouration", "no discoloration"],
    },
    "texture": {
        "smooth": ["smooth", "flat", "even", "fine", "polished", "sleek"],
        "textured": ["textured", "rough", "coarse", "rustic", "float finish",
                     "sponged", "uneven surface", "granular"],
        "open_jointed": ["open joint", "open joints", "open jointed", "recessed",
                         "raked", "eroded joints", "missing mortar",
                         "perished mortar"],
    },
}

SYSTEM_PROMPT = (
    "You are a perception stage. You report only what is visible in the "
    "photograph. You never name a product, never name a brand, never "
    "recommend anything, and never state a cause or a diagnosis. Reporting "
    "what you cannot determine from the image is part of the task, not a "
    "failure of it."
)


@lru_cache(maxsize=1)
def _router_spec() -> dict:
    """The router's own vocabulary file, read once.

    Cached because it is now read on the way *in* as well as on the way out --
    the prompt lists the permitted values, and building that list on every call
    would re-parse the file for every photograph.
    """
    return SlotDetector().spec


def _value_menu(slots: tuple[str, ...]) -> str:
    """The permitted values for each attribute, as a line the model can follow.

    **Why telling the model the answers is not cheating.** `resolve()` checks
    every value against the same vocabulary regardless of what the prompt said,
    so this changes the *hit rate* and not the *guarantee* -- a model that
    ignores the menu is discarded exactly as before. Without it the first real
    call answered `{"attribute": "exposure", "value": "visible"}` twelve times:
    the schema's enum forced a legal attribute, nothing suggested a legal
    value, and every reading was thrown away at the vocabulary check. Perfect
    safety, zero coverage.
    """
    router = _router_spec()
    lines = []
    for slot in slots:
        if slot in CONDITION_VALUES:
            values = list(CONDITION_VALUES[slot])
        elif slot in router and not slot.startswith("_"):
            values = list(router[slot]["values"])
        else:
            continue
        lines.append(f"  {slot}: one of {', '.join(values)}")
    return "\n".join(lines)


def prompt_for(slots: tuple[str, ...] = VISION_SLOTS) -> str:
    """The instruction, built around the attributes actually on offer.

    Three things it now does that the fixed string did not, each traceable to
    what the real model actually did with the fixed string:

    * **It lists the permitted values.** See `_value_menu`.
    * **It asks for one observation per attribute.** The model emitted the same
      reading twelve times over; a cap in the schema bounds that but does not
      stop it wasting the whole token budget on repetition, which is what
      truncated the JSON and lost the answer entirely.
    * **It says which attributes a photograph usually cannot settle.** Naming
      `location` and `exposure` as things to put in
      `cannot_determine_from_image` rather than to guess is cheaper than
      discarding the guess afterwards, and it produces a better hand-off.
    """
    return (
        "Describe only what is visible in this photograph of a building "
        "surface.\n\n"
        "Report at most ONE observation per attribute. Do not repeat an "
        "attribute. If you cannot see something, leave it out rather than "
        "guessing.\n\n"
        "Attributes, and the only values accepted for each:\n"
        f"{_value_menu(slots)}\n\n"
        "For each observation give the attribute, the value from the list "
        "above, a short note of what you actually see, your confidence as one "
        "of HIGH, UNCERTAIN or NOT_DETERMINABLE, and the region as "
        "[x0, y0, x1, y1] in fractions of the width and height between 0 and "
        "1 -- not in pixels.\n\n"
        "Here is the exact shape of one observation. Follow it:\n"
        '{"observation": "orange-red bricks in lime mortar, plaster removed", '
        '"attribute": "substrate", "value": "brick", "confidence": "HIGH", '
        '"region": [0.18, 0.02, 0.95, 0.60]}\n\n'
        "Note what that example does: `attribute` is the attribute name from "
        "the list, `value` is one of that attribute's permitted values, and "
        "the free text goes in `observation`. Do not put the attribute name "
        "in `observation`.\n\n"
        "Then list, in cannot_determine_from_image, everything this photograph "
        "cannot settle. A photograph usually cannot settle whether a wall is "
        "internal or external, how weather-exposed the site is, what is behind "
        "a render or plaster, the moisture source, the mortar mix, or whether "
        "any movement is structural. Say so there rather than guessing.\n\n"
        "Do not name any product, brand or remedy. Do not state a cause."
    )


def __getattr__(name: str):
    """`vision.PROMPT` still works, without a file read at import time.

    The prompt is built from the vocabulary file now, and building it eagerly
    would mean importing this module opened and parsed `config/vocabularies.
    json`. An import that can fail on a missing config file is an import that
    can take the whole application down for a stage that is meant to degrade
    rather than raise.
    """
    if name == "PROMPT":
        return prompt_for()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ------------------------------------------------------------------ perceive


def _read_image(image: bytes | bytearray | str | Path) -> tuple[bytes, str]:
    """The image bytes and a default identifier, or an error in the second slot."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image), ""
    try:
        return Path(image).read_bytes(), ""
    except OSError as exc:
        return b"", f"could not read the image: {exc}"


def _decode_region(raw) -> tuple[float, float, float, float] | None:
    """A region, or None if the model did not give a usable one.

    Four numbers, inside the image, and the second corner past the first.
    A degenerate or inverted box is not a region a person could check, and an
    unauditable claim must not fill a slot.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
        return None
    x0, y0, x1, y1 = (float(v) for v in raw)
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _decode_confidence(raw) -> float | None:
    """How sure the model said it was, as a float, or None if it did not say.

    Three shapes are read, and the order is the order of trust.

    A **band name** is the contract `observation_schema` now asks for, and the
    only one a grammar can enforce. It is carried as the band's floor.

    A **float in 0 to 1** is the old contract, kept because every fixture
    written against it is a valid response and because a model that answers
    0.9 has told us something usable.

    A **whole number from 2 to 100** is read as a percentage. This is the case
    that made the function necessary. Asked for 0 to 1, `qwen3.5:4b` answered
    `100` on two observations that were otherwise correct and
    vocabulary-valid -- `exposed_masonry: yes`, `existing_finish: render` --
    and the old guard dropped both, so a perception step that had read the
    wall correctly reported nothing at all. Rescaling reads the model's answer
    on the scale it plainly used; it does not invent one.

    Anything else is refused, and refusing is still the common case worth
    protecting: a missing confidence, a string that is not a band, a bool
    (which `isinstance(True, int)` would otherwise let through as 1.0), a
    number too large to be a percentage of anything, and -- the one worth
    naming -- a value like `1.4`, which is a probability that overshot rather
    than one and a bit per cent. Reading that as a percentage would turn a
    malformed figure into a confident-looking 0.014, which is worse than
    dropping it.

    **This widens what is read, never what is trusted.** The band still decides
    whether a slot may be filled, `HIGH` is still the only band that fills one,
    and a model stamping its top confidence on everything is exactly why
    `Band` exists. What changes is that a correct reading is no longer thrown
    away for writing 100 where the schema said 1.
    """
    if isinstance(raw, str):
        try:
            return BAND_FLOOR[Band(raw.strip().upper())]
        except (KeyError, ValueError):
            return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return None
    value = float(raw)
    if 0.0 <= value <= 1.0:
        return value
    if PERCENT_FLOOR <= value <= PERCENT_CEILING and value == int(value):
        return value / PERCENT_CEILING
    return None


def _decode_observation(raw, image: str, slots: tuple[str, ...]) -> Observation | None:
    """One observation from the model's JSON, or None if it is not one."""
    if not isinstance(raw, dict):
        return None
    attribute = raw.get("attribute")
    value = raw.get("value")
    if attribute not in slots:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    confidence = _decode_confidence(raw.get("confidence"))
    if confidence is None:
        return None
    interpretations = raw.get("possible_interpretations")
    if not isinstance(interpretations, list):
        interpretations = []
    return Observation(
        attribute=attribute,
        value=value.strip(),
        confidence=float(confidence),
        image=image,
        region=_decode_region(raw.get("region")),
        observation=(raw.get("observation")
                     if isinstance(raw.get("observation"), str) else ""),
        possible_interpretations=tuple(i for i in interpretations if isinstance(i, dict)),
    )


def salvage(body: str) -> tuple[dict | None, bool]:
    """A truncated response into whatever *completed*, and a flag saying so.

    **This exists because of what happened the first time a real photograph met
    the real model.** `qwen3.5:4b` looped -- the same reading, over and over --
    exhausted `num_predict` part-way through the twelfth copy, and the response
    ended `"confidence`. `json.loads` refused it, correctly, and the whole call
    came back empty after 204 seconds. Every real photograph did that. The
    module was safe and completely useless, which is a failure mode worth
    naming as loudly as an unsafe one: nothing prints, nothing warns, and the
    page still says a photograph was read.

    What is recovered is **only complete objects**. The scanner walks the text
    tracking string state and nesting depth, remembers the end of every
    observation that closed, and rebuilds the array up to the last one. A
    half-written object is dropped whole; no field is ever defaulted, guessed
    or repaired. So this cannot invent an observation -- it can only fail to
    recover one.

    `cannot_determine_from_image` comes *after* `observations` in the schema,
    so a truncation inside the array means it never arrived. It stays absent
    rather than being defaulted to empty: the caller is told the response was
    truncated, and `resolve()` refuses to fill a slot from a truncated
    perception. Recovering the readings for a person to look at is worth doing;
    letting a looping model steer a recommendation is not.
    """
    start = body.find('"observations"')
    if start == -1:
        return None, False
    open_bracket = body.find("[", start)
    if open_bracket == -1:
        return None, False

    depth = 0
    in_string = False
    escaped = False
    ends: list[int] = []
    for index in range(open_bracket + 1, len(body)):
        char = body[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            if depth == 0:          # the observations array closed normally
                break
            depth -= 1
            if depth == 0 and char == "}":
                ends.append(index)
    if not ends:
        return None, True

    fragment = body[open_bracket:ends[-1] + 1] + "]"
    try:
        observations = json.loads(fragment)
    except (json.JSONDecodeError, TypeError):
        return None, True
    if not isinstance(observations, list):
        return None, True
    return {"observations": observations}, True


def _decode(body: str, image: str, model: str, seconds: float,
            slots: tuple[str, ...]) -> Perception:
    """The model's text into a Perception, degrading rather than raising."""
    truncated = False
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        parsed, truncated = salvage(body if isinstance(body, str) else "")
        if parsed is None:
            return Perception(
                image=image, model=model, seconds=seconds, truncated=truncated,
                error=("the vision model ran out of tokens before it finished "
                       "and nothing complete could be recovered"
                       if truncated else
                       "the vision model did not return JSON"))
    if not isinstance(parsed, dict):
        return Perception(image=image, model=model, seconds=seconds,
                          error="the vision model did not return an object")

    raw_observations = parsed.get("observations")
    cannot = parsed.get("cannot_determine_from_image")
    if not isinstance(raw_observations, list):
        return Perception(image=image, model=model, seconds=seconds,
                          truncated=truncated,
                          error="the response did not match the observation contract")
    if not isinstance(cannot, list):
        # The required field is still required on a *complete* response: a
        # model that answered in full and skipped the one field that exists to
        # stop it inventing has not answered the contract, and defaulting it
        # would invent that field on the model's behalf.
        #
        # A truncated response is a different situation and gets a different
        # answer. The field is missing because the sentence stopped, not
        # because the model declined it, so the observations recovered above
        # are kept for a person to read -- and `resolve()` will not let any of
        # them fill a slot.
        if not truncated:
            return Perception(
                image=image, model=model, seconds=seconds,
                error="the response did not match the observation contract")
        cannot = []

    refused: list[str] = []
    decoded: list[Observation] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_observations[:MAX_OBSERVATIONS]:
        if isinstance(raw, dict):
            attribute = raw.get("attribute")
            if isinstance(attribute, str) and \
                    attribute.strip().lower() in FORBIDDEN_ATTRIBUTES:
                # Unreachable through the schema's enum, which is the point of
                # checking it anyway: if it ever becomes reachable, this is
                # what makes it visible instead of silent.
                refused.append(attribute.strip().lower())
                continue
        observation = _decode_observation(raw, image, slots)
        if observation is None:
            continue
        # One reading per attribute-and-value. The loop that truncated the
        # first real response repeated a single observation twelve times; the
        # token ceiling bounds the damage and this removes it from the
        # aggregation, where twelve copies of one guess would otherwise look
        # like twelve images agreeing.
        key = (observation.attribute, observation.value.lower())
        if key in seen:
            continue
        seen.add(key)
        decoded.append(observation)

    return Perception(
        observations=tuple(decoded),
        cannot_determine_from_image=tuple(c for c in cannot if isinstance(c, str)),
        image=image,
        model=model,
        seconds=seconds,
        truncated=truncated,
        refused=tuple(dict.fromkeys(refused)),
    )


def observe(
    image: bytes | bytearray | str | Path,
    image_id: str = "",
    model: str = "",
    slots: tuple[str, ...] = VISION_SLOTS,
    timeout: float = VISION_TIMEOUT,
    host: str = "",
) -> Perception:
    """One photograph into structured observations. Never raises.

    The call is `/api/generate` with an `images` field and a `format` schema —
    the same endpoint the text path uses, at temperature zero with a fixed seed
    for the same reason: a reviewer re-running this should get it back.

    It is a local HTTP call rather than a call into `assistant.infrastructure.ollama` because
    that module's `generate()` takes neither images nor a response schema, and
    widening it for a roadmap stage is not this module's to do.

    A truncated response is a malformed one and degrades to no slots, which is
    the correct outcome: a model that has run to the token ceiling is looping,
    and a looping model's observations are not evidence.
    """
    model = model or VISION_MODEL
    if image_id:
        identifier = str(image_id)
    elif isinstance(image, (str, Path)):
        # A filename is provenance the caller already has; bytes are not.
        identifier = str(image)
    else:
        identifier = "image"

    data, read_error = _read_image(image)
    if read_error:
        return Perception(image=identifier, model=model, error=read_error)
    if not data:
        return Perception(image=identifier, model=model, error="the image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        return Perception(
            image=identifier, model=model,
            error=f"the image is {len(data)} bytes, over the "
                  f"{MAX_IMAGE_BYTES} byte limit",
        )

    body = {
        "model": model,
        "prompt": prompt_for(slots),
        "system": SYSTEM_PROMPT,
        "images": [base64.b64encode(data).decode("ascii")],
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "format": observation_schema(slots),
        # The same context size `assistant/ollama.py` asks for, and it has to
        # be the same rather than merely large enough.
        #
        # Omitting it let Ollama size this instance at the model default, 4096,
        # while every text answer asks for 8192 -- two different sizes of the
        # same model. Serving both needs two resident instances, and on a
        # machine with no room for the second, Ollama does not evict, error or
        # resize: it **blocks, indefinitely**. Measured on this build, a five
        # token completion returned in 1.08 s with no `num_ctx` against a
        # resident instance and never returned at all with `num_ctx: 8192`
        # against a 4096 one; unloading the model first, the same call took
        # 39 s and succeeded.
        #
        # The effect was a photograph question that hung forever -- perception
        # loads 4096, its own compose then asks 8192 -- with no error and
        # nothing in the log, which is the worst shape a failure can take in
        # front of a customer. It survived every test because the vision suite
        # fakes the HTTP client, so no test had ever run perception and
        # generation against one real Ollama.
        #
        # 8192 is also what this path needs on its own terms: prompt and image
        # measured 1,958 tokens against a `MAX_VISION_TOKENS` budget of 900,
        # which fits 4096 only while the photograph stays small.
        "options": {"temperature": 0, "top_p": 1, "seed": 0,
                    "num_ctx": VISION_NUM_CTX,
                    "num_predict": MAX_VISION_TOKENS,
                    "repeat_penalty": VISION_REPEAT_PENALTY},
    }

    started = time.perf_counter()
    try:
        with httpx.Client(base_url=host or HOST, timeout=timeout) as client:
            response = client.post("/api/generate", json=body)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        return Perception(image=identifier, model=model,
                          seconds=time.perf_counter() - started,
                          error=f"the vision call failed: {exc}")
    except ValueError as exc:
        return Perception(image=identifier, model=model,
                          seconds=time.perf_counter() - started,
                          error=f"the vision call returned unparseable JSON: {exc}")
    seconds = time.perf_counter() - started

    text = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        return Perception(image=identifier, model=model, seconds=seconds,
                          error="the vision call returned no response field")
    return _decode(text, identifier, model, seconds, slots)


def observe_many(images, model: str = "", **kwargs) -> list[Perception]:
    """Several photographs, in order. A failed one does not stop the others.

    The guided visual survey 16.1 describes collects more than one image — a
    wider elevation, an exposed section, the ground line — and a resolver that
    aggregates with provenance needs them together.
    """
    out: list[Perception] = []
    for index, image in enumerate(images, start=1):
        identifier = image if isinstance(image, str) else f"IMG_{index:03d}"
        out.append(observe(image, image_id=str(identifier), model=model, **kwargs))
    return out


# ------------------------------------------------------------- the resolver


_WORD = re.compile(r"[a-z0-9]+")


def _vocabulary(slots: tuple[str, ...]) -> dict[str, dict[str, list[str]]]:
    """The slot vocabularies, read from the one file the router reads.

    Read through `SlotDetector` rather than re-parsed here, so that a value
    this module will accept and a value the router would have detected from
    typed text are the same list by construction.

    Router slots only. Tier-3 conditions have no router slot and are defined in
    `CONDITION_VALUES`; `_values_for` is the reader that spans both, and this
    function stays narrow so that "the vocabulary is the router's own" remains
    a checkable claim about the slots that actually route.
    """
    spec = _router_spec()
    return {
        slot: spec[slot]["values"]
        for slot in slots
        if slot in spec and not slot.startswith("_")
    }


def _values_for(attribute: str) -> dict[str, list[str]] | None:
    """The permitted values for any attribute in any tier, or None if unknown."""
    if attribute in CONDITION_VALUES:
        return CONDITION_VALUES[attribute]
    spec = _router_spec()
    if attribute in spec and not attribute.startswith("_"):
        return spec[attribute]["values"]
    return None


def _match_value(observed: str, values: dict[str, list[str]]) -> tuple[str, str]:
    """The vocabulary value an observed string means, and why it means nothing.

    Returns the value and an empty reason, or an empty value and the reason it
    was rejected.

    This is the line that stops a product reaching a slot. `"value": "Solo"`
    matches no substrate term, so it is discarded — and because the only thing
    that can ever come out of here is a key of the vocabulary, no string the
    model invents can become a slot value.

    Two vocabulary values matching at once is an ambiguity, and an ambiguity is
    refused rather than scored. "old brick masonry" matches both `brick` and
    `stone`, and the router's own detector would resolve that by longest term
    and land on `stone`, which is the wrong answer to a question nobody asked
    it. The difference is trust: the router is scoring a customer's own words,
    where picking the most specific reading is a reasonable service, whereas
    this is scoring a model's uncalibrated guess about a photograph, into the
    slot that decides the product. An uncued substrate becomes an ask-back,
    which is a known-good outcome; a confidently wrong one is the error the
    whole design exists to avoid.
    """
    lowered = " ".join(_WORD.findall(observed.lower()))
    if not lowered:
        return "", "not a value the vocabulary defines"
    matched = [
        value for value, terms in values.items()
        if any(re.search(rf"\b{re.escape(term.lower())}\b", lowered)
               for term in terms)
    ]
    if not matched:
        return "", "not a value the vocabulary defines"
    if len(matched) > 1:
        return "", ("ambiguous between " + ", ".join(sorted(matched)))
    return matched[0], ""


def _covered_without_exposure(resolved: dict[str, ResolvedAttribute]) -> str:
    """Is the wall's construction hidden behind a finish? Returns why, or "".

    **The substrate gate, and the reason it is a gate rather than a caveat.**
    "Exact hidden substrate" is on the list of things a photograph may not
    settle, and the case where a model is most tempted to settle it is exactly
    the case where it cannot: a sound rendered elevation, where the only honest
    reading of the background is that there isn't one. A model asked what the
    wall is made of will happily answer "brick", because most walls are, and
    that answer is a prior dressed as a perception.

    Two things make this checkable rather than a matter of judgement. The model
    is asked separately whether masonry is exposed, and separately what the
    finish is; those are both genuinely visible. If it reports a covering
    finish and does not report exposed masonry, then by its own two readings
    the background is concealed, and a substrate claim contradicts them.

    Stated as a *contradiction* rather than as a requirement on purpose. A
    photograph of bare brickwork carries no finish observation at all, and
    demanding a positive "yes, masonry is exposed" before believing the obvious
    would refuse the one case vision is unambiguously good at. The gate fires
    only when the model has actually said the wall is covered.
    """
    finish = resolved.get("existing_finish")
    if finish is None or finish.value in ("", "none"):
        return ""
    if finish.certainty not in (Certainty.OBSERVED, Certainty.LIKELY):
        return ""
    exposed = resolved.get("exposed_masonry")
    if exposed is not None and exposed.value in ("yes", "partial"):
        return ""
    return (f"the {finish.value} covers the background, so what the wall is "
            f"built of cannot be read from this photograph")


def resolve(
    perceptions,
    slots: tuple[str, ...] = ROUTER_SLOTS,
    context: tuple[str, ...] = CONTEXT_SLOTS,
    conditions: tuple[str, ...] = CONDITION_ATTRIBUTES,
) -> Resolution:
    """Observations into slots, aggregating across images with provenance.

    Four rules, and each one is a refusal rather than a guess:

    * **Not in the vocabulary, not a slot.** An observed value must map onto
      exactly one value `config/vocabularies.json` already defines. This is the
      guarantee that a vision model cannot invent a substrate or name a
      product, and an observed value matching two of them is an ambiguity that
      resolves to nothing.
    * **Not auditable, not a slot.** An observation with no usable region is
      kept in `attributes` for a person to read and cannot fill a slot, because
      a claim nobody can check against the pixels is not evidence.
    * **Not `HIGH`, not a slot.** Below the top band the slot stays uncued and
      the router asks back, exactly as it does when no photograph was sent.
    * **Confident and contradictory, not a slot.** Two `HIGH` observations
      naming different substrates are `CONFLICTING`, and a conflict resolves to
      nothing rather than to whichever image happened to arrive last.

    Note what is never read: `observation`, `possible_interpretations` and any
    other free text the model wrote. Only the enum, the vocabulary-checked
    value, the band and the region decide anything here.
    """
    known: dict[str, dict[str, list[str]]] = {}
    for attribute in tuple(slots) + tuple(context) + tuple(conditions):
        values = _values_for(attribute)
        if values is not None:
            known[attribute] = values

    accepted: dict[str, list[tuple[str, Observation]]] = {a: [] for a in known}
    seen: dict[str, list[tuple[str, Observation]]] = {a: [] for a in known}
    discarded: list[str] = []
    refused: list[str] = []
    truncated = False

    for perception in perceptions:
        refused.extend(getattr(perception, "refused", ()))
        # A truncated response is a looping model. What it managed to say is
        # kept and reported; none of it is allowed to decide anything.
        cut = bool(getattr(perception, "truncated", False))
        truncated = truncated or cut
        for observation in perception.observations:
            if observation.attribute not in known:
                discarded.append(
                    f"{observation.attribute}={observation.value!r}: "
                    "not a slot a photograph may fill")
                continue
            value, reason = _match_value(observation.value,
                                         known[observation.attribute])
            if not value:
                discarded.append(
                    f"{observation.attribute}={observation.value!r}: {reason}")
                continue
            seen[observation.attribute].append((value, observation))
            if not observation.auditable:
                discarded.append(
                    f"{observation.attribute}={observation.value!r}: "
                    "no usable region, so the claim is not auditable")
                continue
            if cut:
                discarded.append(
                    f"{observation.attribute}={observation.value!r}: the "
                    "response was truncated, so it is reported but not acted on")
                continue
            if observation.band is not Band.HIGH:
                continue
            accepted[observation.attribute].append((value, observation))

    by_attribute: dict[str, ResolvedAttribute] = {}
    for attribute in known:
        candidates = accepted[attribute]
        if not candidates:
            observed = seen[attribute]
            if not observed:
                continue
            best_value, best_observation = max(
                observed, key=lambda pair: pair[1].confidence)

            # A top-band reading reaching this branch was strong enough to act
            # on and lost its place for a *mechanical* reason -- the response
            # was truncated, or no region was given to check it against.
            # Calling that "uncertain" would understate what the model saw, so
            # the reason is recorded and `Certainty` reads it back as LIKELY.
            if best_observation.band is Band.HIGH:
                status = Status.UNCERTAIN
                withheld = ("no region was given, so the claim cannot be checked"
                            if not best_observation.auditable
                            else "the response was truncated")
            else:
                status = (Status.UNCERTAIN if best_observation.band is Band.UNCERTAIN
                          else Status.NOT_DETERMINABLE)
                withheld = ""

            by_attribute[attribute] = ResolvedAttribute(
                attribute, best_value, status, best_observation.confidence,
                tuple(o for _, o in observed), withheld=withheld)
            continue

        distinct = {value for value, _ in candidates}
        sources = tuple(o for _, o in candidates)
        if len(distinct) > 1:
            by_attribute[attribute] = ResolvedAttribute(
                attribute, "", Status.CONFLICTING,
                max(o.confidence for o in sources), sources,
                withheld="two images disagree")
            discarded.append(
                f"{attribute}: confident observations disagree "
                f"({', '.join(sorted(distinct))}), so the slot stays uncued")
            continue

        value, best = max(candidates, key=lambda pair: pair[1].confidence)
        by_attribute[attribute] = ResolvedAttribute(
            attribute, value, Status.CONFIRMED, best.confidence, sources)

    # The substrate gate runs after aggregation, because it reads one resolved
    # attribute against another. See `_covered_without_exposure`.
    hidden = _covered_without_exposure(by_attribute)
    if hidden and "substrate" in by_attribute:
        by_attribute["substrate"] = field_replace(
            by_attribute["substrate"], withheld=hidden)
        discarded.append(f"substrate: {hidden}")

    # An attribute the model itself listed as undeterminable is reported as
    # such even when it also guessed at it. The model's own refusal outranks
    # the model's own guess -- it is the only place in the contract where the
    # thing being measured gets to say "I cannot", and honouring it is what
    # makes asking worth anything.
    cannot: list[str] = []
    for perception in perceptions:
        for item in perception.cannot_determine_from_image:
            if item not in cannot:
                cannot.append(item)
    for attribute in list(by_attribute):
        if _named_in(attribute, cannot) and not by_attribute[attribute].routed:
            by_attribute[attribute] = field_replace(
                by_attribute[attribute], status=Status.NOT_DETERMINABLE,
                withheld="the model reported that the photograph cannot "
                         "settle this")

    resolved = {a: r.value for a, r in by_attribute.items()
                if a in slots and r.routed}

    return Resolution(
        slots=resolved,
        attributes=tuple(by_attribute.values()),
        cannot_determine_from_image=tuple(cannot),
        discarded=tuple(discarded),
        context=tuple(r for a, r in by_attribute.items() if a in context),
        conditions=tuple(r for a, r in by_attribute.items() if a in conditions),
        refused=tuple(dict.fromkeys(refused)),
        truncated=truncated,
    )


def _named_in(attribute: str, cannot: list[str]) -> bool:
    """Did the model's own `cannot_determine_from_image` name this attribute?

    Word-boundary matching on the attribute's own words, so "the wall
    construction behind the render" matches `substrate` only if the vocabulary
    says those words mean substrate -- it does not, and that is deliberate.
    This looks for the attribute *name*, which is what the prompt asks the
    model to use.
    """
    words = attribute.replace("_", " ")
    return any(re.search(rf"\b{re.escape(words)}\b", item.lower())
               for item in cannot)


def slots_from_images(images, model: str = "", **kwargs) -> Resolution:
    """Photographs straight to a `carried` dict. The whole module in one call.

    The result's `.slots` is what `Assistant.ask(question, carried=...)` takes.
    A total failure of the vision path yields `{}`, which is what the caller
    would have passed had there been no photograph — coverage reduced, safety
    unchanged.
    """
    return resolve(observe_many(images, model=model, **kwargs))


def perception_report(resolution) -> dict:
    """A `Resolution` as plain data, for a checkpoint, a surface or a log.

    Primitives only, deliberately. The turn state is checkpointed, and a
    checkpointer that has to rehydrate `ResolvedAttribute`, `Observation`,
    `Certainty` and `Status` to show somebody what a photograph showed is four
    more types in the serialisation allow-list for no gain. A channel adapter
    rendering this its own way wants strings anyway.

    **Every claim travels with its certainty and, where there is one, the
    reason it was not acted on.** They are in one record rather than two
    because a caller able to read the value without the hedge would turn a
    reading into a fact, which is the failure this module is arranged to
    prevent. `summary` is the pre-rendered English for a surface that wants to
    print it without deciding the wording itself.

    Every field is read with a fallback, and that is deliberate rather than
    defensive habit. `Services.vision` is an injected provider -- a test double
    today, a channel adapter's own perception service tomorrow -- and the
    contract it satisfies is "returns something with slots and attributes". A
    report that raised on a provider without `certainty` would mean **a
    reporting function could fail an answer**, which inverts this module's one
    rule: failure reduces coverage, not safety. A minimal provider gets a
    thinner report and a working answer.
    """
    routed_slots = dict(getattr(resolution, "slots", {}) or {})

    def certainty_of(attribute) -> Certainty:
        stated = getattr(attribute, "certainty", None)
        if isinstance(stated, Certainty):
            return stated
        # A provider that does not compute one: derive it from what it did
        # give, the same way `ResolvedAttribute` would have.
        if getattr(attribute, "slot", "") in routed_slots:
            return Certainty.OBSERVED
        confidence = float(getattr(attribute, "confidence", 0.0) or 0.0)
        if confidence >= HIGH_FLOOR:
            return Certainty.LIKELY
        if confidence >= LIKELY_FLOOR:
            return Certainty.LIKELY
        if confidence >= UNCERTAIN_FLOOR:
            return Certainty.UNCERTAIN
        return Certainty.CANNOT_DETERMINE

    def one(attribute) -> dict:
        sources = tuple(getattr(attribute, "sources", ()) or ())
        status = getattr(attribute, "status", None)
        return {
            "attribute": getattr(attribute, "slot", ""),
            "value": getattr(attribute, "value", ""),
            "certainty": certainty_of(attribute).value,
            "status": getattr(status, "value", "") if status else "",
            "confidence": round(float(
                getattr(attribute, "confidence", 0.0) or 0.0), 3),
            "routed": bool(getattr(attribute, "routed",
                                   getattr(attribute, "slot", "") in routed_slots)),
            "withheld": getattr(attribute, "withheld", "") or "",
            "images": sorted({o.image for o in sources
                              if getattr(o, "image", "")}),
            "regions": [list(o.region) for o in sources
                        if getattr(o, "region", None)],
            # The model's own words, carried for a person to read and never
            # read by anything that decides. Capped: this is shown on a page.
            "notes": [o.observation for o in sources
                      if getattr(o, "observation", "")][:2],
        }

    order = {Certainty.OBSERVED.value: 0, Certainty.LIKELY.value: 1,
             Certainty.UNCERTAIN.value: 2, Certainty.CANNOT_DETERMINE.value: 3}
    rows = sorted((one(a) for a in getattr(resolution, "attributes", ()) or ()),
                  key=lambda r: (order.get(r["certainty"], 9), r["attribute"]))

    phrase = {
        Certainty.OBSERVED.value: "{value} ({slot}) -- clearly visible",
        Certainty.LIKELY.value: "{value} ({slot}) -- likely, but not certain "
                                "from the photograph",
        Certainty.UNCERTAIN.value: "{slot} -- uncertain; the photograph is not "
                                   "clear enough to rely on",
        Certainty.CANNOT_DETERMINE.value: "{slot} -- cannot be determined from "
                                          "the photograph",
    }
    summary = []
    for row in rows:
        summary.append(phrase[row["certainty"]].format(
            slot=row["attribute"].replace("_", " "),
            value=str(row["value"]).replace("_", " ")))
        if row["withheld"]:
            summary.append(f"    ({row['withheld']})")

    return {
        "routed": routed_slots,
        "observations": rows,
        "cannot_determine_from_image": list(
            getattr(resolution, "cannot_determine_from_image", ()) or ()),
        "summary": summary,
        "discarded": list(getattr(resolution, "discarded", ()) or ()),
        "refused_attributes": list(getattr(resolution, "refused", ()) or ()),
        "truncated": bool(getattr(resolution, "truncated", False)),
        # True whenever perception actually ran, which is what lets a surface
        # tell "looked and saw nothing" apart from "never looked". The two
        # deserve different sentences and only one of them is a limitation of
        # the photograph.
        "enabled": True,
    }


def disabled_report() -> dict:
    """The same shape, for a turn where image reading is switched off.

    Same keys as `perception_report`, so no caller needs a second branch and
    no renderer can be surprised by a missing field. What differs is `enabled`,
    and it differs *loudly*: `summary` carries the hand-off sentence, so a
    surface that prints the summary and nothing else still tells the truth.

    This exists because the alternative -- returning nothing at all -- is the
    failure the graph already carries a comment about: every surface passed no
    provider, every photograph was ignored, and the page still said it had read
    one. Silence about an upload is indistinguishable from having looked.
    """
    return {
        "routed": {},
        "observations": [],
        "cannot_determine_from_image": [],
        "summary": [DISABLED_NOTE],
        "discarded": [],
        "refused_attributes": [],
        "truncated": False,
        "enabled": False,
    }


def encode_image(data: bytes) -> str:
    """Base64 as Ollama's `images` field wants it. Exposed for callers and tests."""
    return base64.b64encode(data).decode("ascii")


def decode_image(text: str) -> bytes:
    """The inverse, used by tests asserting what was actually sent."""
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return b""
