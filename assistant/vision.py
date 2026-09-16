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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

from .ollama import HOST, KEEP_ALIVE, GENERATION_MODEL
from .router import SlotDetector

# The slots a photograph is allowed to fill. Deliberately four, and
# deliberately not five: `cause_asked` is missing because a cause is a
# judgement the technical team makes and stands behind (decision 16), and a
# model that could fill it would route a diagnosis question away from the
# hand-off it must take.
VISION_SLOTS: tuple[str, ...] = ("substrate", "location", "exposure", "symptom")

# A photograph from a phone is a few megabytes. Anything much larger is either
# a mistake or an attempt to make the machine do arithmetic on a gigabyte, and
# both are refused at the boundary rather than sent to a model.
MAX_IMAGE_BYTES = int(os.environ.get("VISION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))

VISION_MODEL = os.environ.get("VISION_MODEL", GENERATION_MODEL)

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
MAX_VISION_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "512"))
MAX_OBSERVATIONS = int(os.environ.get("VISION_MAX_OBSERVATIONS", "12"))


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


def band_for(confidence: float) -> Band:
    """The band a self-reported confidence falls in. The only reader of the float."""
    if confidence >= HIGH_FLOOR:
        return Band.HIGH
    if confidence >= UNCERTAIN_FLOOR:
        return Band.UNCERTAIN
    return Band.NOT_DETERMINABLE


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


@dataclass(frozen=True)
class Resolution:
    """The profile resolver's output: slots to hand over, and why.

    `slots` is a plain `dict` of slot to vocabulary value, suitable to pass
    straight to `Assistant.ask(question, carried=...)`. Only `CONFIRMED`
    attributes appear in it. `attributes` carries everything, including what
    was seen and rejected, so a hand-off can say what the photograph did and
    did not settle.
    """

    slots: dict[str, str]
    attributes: tuple[ResolvedAttribute, ...] = ()
    cannot_determine_from_image: tuple[str, ...] = ()
    discarded: tuple[str, ...] = ()


# --------------------------------------------------------------- the schema

# Ollama constrains a response to a JSON schema through `format`. 16.1's last
# refinement asks for exactly this rather than parsing prose out of a vision
# model, and it is the first half of the never-names-a-product guarantee: there
# is no field in this schema a product name belongs in. The second half is
# `resolve()`, which checks the values anyway, because a schema constrains
# shape and not meaning — nothing here stops a model writing "Solo" into
# `value`, and the vocabulary check is what makes that harmless.
def observation_schema(slots: tuple[str, ...] = VISION_SLOTS) -> dict:
    """The JSON schema the model is constrained to."""
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
                        "confidence": {"type": "number"},
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


SYSTEM_PROMPT = (
    "You are a perception stage. You report only what is visible in the "
    "photograph. You never name a product, never name a brand, never "
    "recommend anything, and never state a cause or a diagnosis. Reporting "
    "what you cannot determine from the image is part of the task, not a "
    "failure of it."
)

PROMPT = (
    "Describe only what is visible in this photograph of a building surface.\n"
    "For each attribute you can observe, give the attribute, a short plain "
    "value, your confidence from 0 to 1, and the region of the image as "
    "[x0, y0, x1, y1] in fractions of the width and height.\n"
    "Then list, in cannot_determine_from_image, everything a photograph "
    "cannot settle here — for example the existing plaster composition, the "
    "moisture source, the wall construction depth, the substrate suction or "
    "structural movement.\n"
    "Do not name any product, brand or remedy. Do not state a cause."
)


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


def _decode_observation(raw, image: str, slots: tuple[str, ...]) -> Observation | None:
    """One observation from the model's JSON, or None if it is not one."""
    if not isinstance(raw, dict):
        return None
    attribute = raw.get("attribute")
    value = raw.get("value")
    confidence = raw.get("confidence")
    if attribute not in slots:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
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


def _decode(body: str, image: str, model: str, seconds: float,
            slots: tuple[str, ...]) -> Perception:
    """The model's text into a Perception, degrading rather than raising."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return Perception(image=image, model=model, seconds=seconds,
                          error="the vision model did not return JSON")
    if not isinstance(parsed, dict):
        return Perception(image=image, model=model, seconds=seconds,
                          error="the vision model did not return an object")

    raw_observations = parsed.get("observations")
    cannot = parsed.get("cannot_determine_from_image")
    if not isinstance(raw_observations, list) or not isinstance(cannot, list):
        # The required fields are required. A response without them has not
        # answered the contract, and guessing a default for the list of things
        # the model could not determine would invent the one field that exists
        # to stop the model inventing things.
        return Perception(image=image, model=model, seconds=seconds,
                          error="the response did not match the observation contract")

    decoded = [_decode_observation(r, image, slots)
               for r in raw_observations[:MAX_OBSERVATIONS]]
    return Perception(
        observations=tuple(o for o in decoded if o is not None),
        cannot_determine_from_image=tuple(c for c in cannot if isinstance(c, str)),
        image=image,
        model=model,
        seconds=seconds,
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

    It is a local HTTP call rather than a call into `assistant.ollama` because
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
        "prompt": PROMPT,
        "system": SYSTEM_PROMPT,
        "images": [base64.b64encode(data).decode("ascii")],
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "format": observation_schema(slots),
        "options": {"temperature": 0, "top_p": 1, "seed": 0,
                    "num_predict": MAX_VISION_TOKENS},
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
    """
    spec = SlotDetector().spec
    return {
        slot: spec[slot]["values"]
        for slot in slots
        if slot in spec and not slot.startswith("_")
    }


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


def resolve(
    perceptions,
    slots: tuple[str, ...] = VISION_SLOTS,
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
    vocabulary = _vocabulary(slots)
    accepted: dict[str, list[tuple[str, Observation]]] = {s: [] for s in vocabulary}
    seen: dict[str, list[tuple[str, Observation]]] = {s: [] for s in vocabulary}
    discarded: list[str] = []

    for perception in perceptions:
        for observation in perception.observations:
            if observation.attribute not in vocabulary:
                discarded.append(
                    f"{observation.attribute}={observation.value!r}: "
                    "not a slot a photograph may fill")
                continue
            value, reason = _match_value(observation.value,
                                         vocabulary[observation.attribute])
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
            if observation.band is not Band.HIGH:
                continue
            accepted[observation.attribute].append((value, observation))

    attributes: list[ResolvedAttribute] = []
    resolved: dict[str, str] = {}
    for slot in vocabulary:
        candidates = accepted[slot]
        if not candidates:
            observed = seen[slot]
            if not observed:
                continue
            best = max(observed, key=lambda pair: pair[1].confidence)
            status = (Status.UNCERTAIN if best[1].band is Band.UNCERTAIN
                      else Status.NOT_DETERMINABLE)
            attributes.append(ResolvedAttribute(
                slot, best[0], status, best[1].confidence,
                tuple(o for _, o in observed)))
            continue

        distinct = {value for value, _ in candidates}
        sources = tuple(o for _, o in candidates)
        if len(distinct) > 1:
            attributes.append(ResolvedAttribute(
                slot, "", Status.CONFLICTING,
                max(o.confidence for o in sources), sources))
            discarded.append(
                f"{slot}: confident observations disagree "
                f"({', '.join(sorted(distinct))}), so the slot stays uncued")
            continue

        value, best = max(candidates, key=lambda pair: pair[1].confidence)
        attributes.append(ResolvedAttribute(
            slot, value, Status.CONFIRMED, best.confidence, sources))
        resolved[slot] = value

    cannot: list[str] = []
    for perception in perceptions:
        for item in perception.cannot_determine_from_image:
            if item not in cannot:
                cannot.append(item)

    return Resolution(
        slots=resolved,
        attributes=tuple(attributes),
        cannot_determine_from_image=tuple(cannot),
        discarded=tuple(discarded),
    )


def slots_from_images(images, model: str = "", **kwargs) -> Resolution:
    """Photographs straight to a `carried` dict. The whole module in one call.

    The result's `.slots` is what `Assistant.ask(question, carried=...)` takes.
    A total failure of the vision path yields `{}`, which is what the caller
    would have passed had there been no photograph — coverage reduced, safety
    unchanged.
    """
    return resolve(observe_many(images, model=model, **kwargs))


def encode_image(data: bytes) -> str:
    """Base64 as Ollama's `images` field wants it. Exposed for callers and tests."""
    return base64.b64encode(data).decode("ascii")


def decode_image(text: str) -> bytes:
    """The inverse, used by tests asserting what was actually sent."""
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return b""
