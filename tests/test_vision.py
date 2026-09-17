"""The perception stage: what it accepts, what it discards, and what it refuses.

The single most important test in this file is
`test_a_product_name_never_reaches_a_slot`. Everything else here supports it.
DECISIONS 16.1 makes one rule load-bearing — the vision model never names a
product — and the way this module honours it is structural rather than
promissory: the resolver only ever emits values `config/vocabularies.json`
already defines. So the adversarial test feeds a model response that names
products, recommends one, and tries to smuggle a brand in through every field
the contract has, and asserts that the resulting slot dict contains nothing but
vocabulary values.

No test here touches a real model. The Ollama call is replaced by a fake
client, so the whole file runs offline in milliseconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import vision                                    # noqa: E402
from assistant.vision import (                                  # noqa: E402
    Band,
    Observation,
    Perception,
    Status,
    VISION_SLOTS,
    band_for,
    decode_image,
    encode_image,
    observation_schema,
    observe,
    observe_many,
    resolve,
    slots_from_images,
)

PIXEL = b"\x89PNG\r\n\x1a\n-not-really-a-png-but-bytes-are-bytes"


# ------------------------------------------------------------------ fakes


class _Response:
    def __init__(self, payload, error: Exception | None = None,
                 json_error: Exception | None = None):
        self._payload = payload
        self._error = error
        self._json_error = json_error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


class _Client:
    """Stands in for httpx.Client, recording the one request it is given."""

    sent: dict = {}

    def __init__(self, response: _Response, post_error: Exception | None = None,
                 **kwargs):
        self._response = response
        self._post_error = post_error
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):
        _Client.sent = {"url": url, "body": json, "kwargs": self.kwargs}
        if self._post_error:
            raise self._post_error
        return self._response


def fake_ollama(monkeypatch, *, response_text=None, payload=None,
                post_error=None, status_error=None, json_error=None):
    """Replace the HTTP client with one that returns exactly what a test wants."""
    if payload is None:
        payload = {"response": response_text}
    response = _Response(payload, error=status_error, json_error=json_error)

    def factory(**kwargs):
        return _Client(response, post_error=post_error, **kwargs)

    monkeypatch.setattr(vision.httpx, "Client", factory)
    return response


def body(observations, cannot=("the moisture source",)) -> str:
    return json.dumps({
        "observations": list(observations),
        "cannot_determine_from_image": list(cannot),
    })


def obs(attribute="substrate", value="brick", confidence=0.92,
        region=(0.1, 0.1, 0.9, 0.9), **extra) -> dict:
    out = {"attribute": attribute, "value": value, "confidence": confidence,
           "observation": f"visible {value}",
           "region": list(region) if region else region}
    out.update(extra)
    return out


# ------------------------------------------------- the adversarial test


def test_a_product_name_never_reaches_a_slot(monkeypatch):
    """A model that names, recommends and brands its way through every field.

    This is the failure DECISIONS 16.1 exists to prevent: perception fused to
    recommendation — "I see rising damp, therefore use Product X". The response
    below does exactly that, in prose, in the value field, in the
    interpretations, and by inventing a `product` attribute outright.

    Not one of them reaches the slots. The four substrate readings are
    discarded because no product is a substrate the vocabulary defines, and the
    invented `product` attribute never survives decoding because the schema has
    no such attribute. What does survive is the one part of that response that
    was actually perception: "rising damp, so apply Solo" contains a symptom
    the vocabulary defines, and it resolves to the bare vocabulary key `damp`.

    That is the shape of the guarantee, and it is stronger than rejection. The
    slot dict cannot contain a product name because the only strings that can
    ever be written into it are keys of `config/vocabularies.json` — the
    model's own wording is discarded whether or not it was safe.
    """
    text = body([
        obs(value="Lime Green Solo Onecoat"),
        obs(value="Solo"),
        obs(value="Warmshell Internal system board"),
        obs(attribute="substrate", value="use Duro here"),
        obs(attribute="product", value="Solo"),
        obs(attribute="symptom", value="rising damp, so apply Solo",
            observation="I see rising damp, therefore use Solo",
            possible_interpretations=[{"cause": "use Solo", "confidence": 0.9}]),
    ])
    fake_ollama(monkeypatch, response_text=text)

    resolution = resolve([observe(PIXEL, image_id="IMG_001")])

    assert resolution.slots == {"symptom": "damp"}
    blob = json.dumps(resolution.slots).lower()
    for brand in ("solo", "duro", "warmshell", "lime green", "apply", "use"):
        assert brand not in blob
    # The invented attribute never decoded; the four products were discarded
    # by name, rather than dropped silently.
    assert len(resolution.discarded) == 4
    assert all("not a value the vocabulary defines" in d
               for d in resolution.discarded)


def test_every_resolved_value_is_a_vocabulary_value(monkeypatch):
    """The general form of the rule above, asserted against the vocabulary itself."""
    text = body([
        obs(value="old brickwork"),
        obs(attribute="location", value="exterior elevation", confidence=0.95),
        obs(attribute="symptom", value="white crystalline deposits and salts",
            confidence=0.9),
    ])
    fake_ollama(monkeypatch, response_text=text)

    resolution = resolve([observe(PIXEL)])
    vocabulary = vision._vocabulary(VISION_SLOTS)
    for slot, value in resolution.slots.items():
        assert value in vocabulary[slot]
    assert resolution.slots["substrate"] == "brick"
    assert resolution.slots["location"] == "external"
    assert resolution.slots["symptom"] == "salts"


def test_a_cause_is_not_a_slot_a_photograph_may_fill(monkeypatch):
    """Decision 16 is unchanged: diagnosis is a human judgement.

    `cause_asked` is a real slot in the vocabulary and the router's step 3
    routes on it, so a vision model able to set it would steer a diagnosis
    question away from the hand-off it must take. It is absent from
    `VISION_SLOTS`, and an attempt to set it is discarded.
    """
    assert "cause_asked" not in VISION_SLOTS
    fake_ollama(monkeypatch,
                response_text=body([obs(attribute="cause_asked", value="cause")]))
    perception = observe(PIXEL)
    # The schema has no such attribute, so it does not survive decoding: the
    # resolver is never even offered the chance to reject it.
    assert perception.observations == ()
    assert resolve([perception]).slots == {}


# --------------------------------------------------------- the contract


def test_the_schema_offers_no_field_a_product_belongs_in():
    """Decision 16.1's first rule: the vision model never names a product.

    The enum comparison below only proves the schema and the constant agree, so
    it would go on passing if `product` were added to both. The explicit
    exclusion is the one that matters, and it carries more weight than it used
    to: `product` is a carried slot now, and the session keeps a carried slot
    unless the answer marks it OBSERVED. A product the model could see would
    therefore be a model's guess persisted across turns as something the caller
    said -- which is the leak the whole vision seam exists to prevent, arriving
    through the one slot the seam does not filter.
    """
    assert "product" not in VISION_SLOTS

    schema = observation_schema()
    item = schema["properties"]["observations"]["items"]
    assert item["properties"]["attribute"]["enum"] == list(VISION_SLOTS)
    assert set(item["required"]) == {"observation", "attribute", "value",
                                     "confidence", "region"}
    assert schema["required"] == ["observations", "cannot_determine_from_image"]
    assert "product" not in json.dumps(schema)


def test_the_call_carries_the_image_the_schema_and_temperature_zero(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs()]))
    perception = observe(PIXEL, image_id="IMG_007", model="qwen3.5:4b")

    sent = _Client.sent["body"]
    assert _Client.sent["url"] == "/api/generate"
    assert decode_image(sent["images"][0]) == PIXEL
    assert sent["format"] == observation_schema()
    assert sent["options"]["temperature"] == 0
    assert sent["options"]["seed"] == 0
    assert sent["think"] is False
    system = sent["system"].lower()
    assert "never name" in system
    assert perception.image == "IMG_007"
    assert perception.model == "qwen3.5:4b"
    assert perception.ok


def test_cannot_determine_is_required_not_defaulted(monkeypatch):
    """A response without it has not answered the contract.

    Forcing the model to enumerate what it cannot tell is the visual
    equivalent of refusing. Defaulting the field to an empty list would let a
    model skip the one part of the contract that exists to stop it inventing.
    """
    fake_ollama(monkeypatch,
                response_text=json.dumps({"observations": [obs()]}))
    perception = observe(PIXEL)
    assert not perception.ok
    assert "observation contract" in perception.error
    assert resolve([perception]).slots == {}


def test_an_empty_cannot_determine_list_is_accepted(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs()], cannot=[]))
    perception = observe(PIXEL)
    assert perception.ok
    assert perception.cannot_determine_from_image == ()


def test_cannot_determine_is_carried_through_and_deduplicated(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([], cannot=["the moisture source",
                                                           "the moisture source",
                                                           "the wall depth"]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.cannot_determine_from_image == ("the moisture source",
                                                      "the wall depth")


def test_non_string_entries_in_cannot_determine_are_dropped(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([], cannot=["the wall depth", 7]))
    assert observe(PIXEL).cannot_determine_from_image == ("the wall depth",)


# ------------------------------------------------------------- the bands


@pytest.mark.parametrize("confidence, expected", [
    (1.0, Band.HIGH),
    (0.80, Band.HIGH),
    (0.799, Band.UNCERTAIN),
    (0.50, Band.UNCERTAIN),
    (0.499, Band.NOT_DETERMINABLE),
    (0.0, Band.NOT_DETERMINABLE),
])
def test_confidence_is_collapsed_into_three_bands(confidence, expected):
    assert band_for(confidence) is expected


def test_below_the_top_band_the_slot_stays_uncued(monkeypatch):
    """Which is what the router's load-bearing-slot rule already handles.

    A 0.61 reading of the substrate leaves the caller exactly where they would
    have been with no photograph: asked back for the substrate.
    """
    fake_ollama(monkeypatch, response_text=body([obs(confidence=0.61)]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.slots == {}
    assert resolution.attributes[0].status is Status.UNCERTAIN
    assert resolution.attributes[0].value == "brick"


def test_a_very_low_reading_is_not_determinable(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs(confidence=0.2)]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.slots == {}
    assert resolution.attributes[0].status is Status.NOT_DETERMINABLE


# ------------------------------------------------------- the aggregation


def test_agreeing_images_confirm_and_the_strongest_reading_is_reported(monkeypatch):
    """16.1's worked example, with the disagreement removed: brick 0.82, brick 0.91."""
    fake_ollama(monkeypatch, response_text=body([obs(confidence=0.82)]))
    first = observe(PIXEL, image_id="IMG_001")
    fake_ollama(monkeypatch, response_text=body([obs(confidence=0.91)]))
    second = observe(PIXEL, image_id="IMG_002")

    resolution = resolve([first, second])
    assert resolution.slots == {"substrate": "brick"}
    attribute = resolution.attributes[0]
    assert attribute.status is Status.CONFIRMED
    assert attribute.confidence == 0.91
    assert [o.image for o in attribute.sources] == ["IMG_001", "IMG_002"]


def test_confident_disagreement_resolves_to_nothing(monkeypatch):
    """16.1's actual example: brick 0.82, stone 0.61, brick 0.91 — plus a rival.

    The later image must not simply overwrite the earlier. Two confident
    contradictory readings are less informative than one, so the slot stays
    uncued and the router asks.
    """
    fake_ollama(monkeypatch, response_text=body([
        obs(value="brick", confidence=0.91),
        obs(value="cob wall", confidence=0.88),
    ]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.slots == {}
    assert resolution.attributes[0].status is Status.CONFLICTING
    assert resolution.attributes[0].value == ""
    assert any("disagree" in d for d in resolution.discarded)


def test_an_uncertain_rival_does_not_create_a_conflict(monkeypatch):
    """Only top-band observations can contradict each other."""
    fake_ollama(monkeypatch, response_text=body([
        obs(value="brick", confidence=0.91),
        obs(value="stone", confidence=0.55),
    ]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.slots == {"substrate": "brick"}
    assert resolution.attributes[0].status is Status.CONFIRMED


# --------------------------------------------------------- the auditable


def test_an_observation_without_a_region_cannot_fill_a_slot(monkeypatch):
    """A region is to a visual claim what a cited passage is to a textual one."""
    fake_ollama(monkeypatch, response_text=body([obs(region=None)]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.slots == {}
    assert any("not auditable" in d for d in resolution.discarded)
    assert resolution.attributes[0].status is Status.NOT_DETERMINABLE


@pytest.mark.parametrize("region", [
    [0.1, 0.1, 0.9],                 # three numbers
    "0.1,0.1,0.9,0.9",               # not a list
    [0.1, 0.1, "0.9", 0.9],          # not all numbers
    [0.1, 0.1, True, 0.9],           # a bool is not a coordinate
    [-0.1, 0.1, 0.9, 0.9],           # outside the image
    [0.1, 0.1, 1.4, 0.9],            # outside the image
    [0.9, 0.1, 0.1, 0.9],            # inverted in x
    [0.1, 0.9, 0.9, 0.1],            # inverted in y
    [0.5, 0.1, 0.5, 0.9],            # degenerate
])
def test_an_unusable_region_is_no_region(monkeypatch, region):
    fake_ollama(monkeypatch, response_text=body([obs(region=region)]))
    perception = observe(PIXEL)
    assert perception.observations[0].region is None
    assert perception.observations[0].auditable is False


def test_a_usable_region_is_kept_as_fractions(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs(region=[0, 0, 1, 1])]))
    assert observe(PIXEL).observations[0].region == (0.0, 0.0, 1.0, 1.0)


# ------------------------------------------- malformed responses degrade


@pytest.mark.parametrize("text, fragment", [
    ("I can see a brick wall.", "did not return JSON"),
    ("[1, 2, 3]", "did not return an object"),
    ('{"observations": "brick", "cannot_determine_from_image": []}',
     "observation contract"),
    ('{"observations": [], "cannot_determine_from_image": "none"}',
     "observation contract"),
])
def test_a_malformed_response_degrades_to_no_slots(monkeypatch, text, fragment):
    """CLAUDE.md principle 5: failure reduces coverage, not safety."""
    fake_ollama(monkeypatch, response_text=text)
    perception = observe(PIXEL)
    assert not perception.ok
    assert fragment in perception.error
    assert resolve([perception]).slots == {}


def test_a_response_with_no_response_field_degrades(monkeypatch):
    fake_ollama(monkeypatch, payload={"error": "model not found"})
    perception = observe(PIXEL)
    assert "no response field" in perception.error


def test_a_non_object_payload_degrades(monkeypatch):
    fake_ollama(monkeypatch, payload=["not", "a", "dict"])
    assert "no response field" in observe(PIXEL).error


def test_unreachable_ollama_degrades_rather_than_raising(monkeypatch):
    fake_ollama(monkeypatch, post_error=httpx.ConnectError("connection refused"))
    perception = observe(PIXEL)
    assert "the vision call failed" in perception.error
    assert perception.seconds >= 0
    assert resolve([perception]).slots == {}


def test_an_http_error_status_degrades(monkeypatch):
    fake_ollama(monkeypatch, status_error=httpx.HTTPStatusError(
        "500", request=None, response=None))
    assert "the vision call failed" in observe(PIXEL).error


def test_an_unparseable_http_body_degrades(monkeypatch):
    fake_ollama(monkeypatch, json_error=ValueError("no JSON"))
    assert "unparseable JSON" in observe(PIXEL).error


@pytest.mark.parametrize("raw", [
    "not a dict",
    {"attribute": "substrate", "value": "brick"},                    # no confidence
    {"attribute": "nonsense", "value": "brick", "confidence": 0.9},  # not a slot
    {"attribute": "substrate", "value": "", "confidence": 0.9},      # empty value
    {"attribute": "substrate", "value": 7, "confidence": 0.9},       # not a string
    {"attribute": "substrate", "value": "brick", "confidence": "high"},
    {"attribute": "substrate", "value": "brick", "confidence": True},
    {"attribute": "substrate", "value": "brick", "confidence": 1.4},
    {"attribute": "substrate", "value": "brick", "confidence": -0.1},
])
def test_a_malformed_observation_is_dropped_not_guessed(monkeypatch, raw):
    fake_ollama(monkeypatch, response_text=body([raw]))
    assert observe(PIXEL).observations == ()


def test_free_text_fields_are_optional_and_sanitised(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([{
        "attribute": "substrate", "value": "brick", "confidence": 0.9,
        "observation": 7, "possible_interpretations": "nonsense",
    }]))
    observation = observe(PIXEL).observations[0]
    assert observation.observation == ""
    assert observation.possible_interpretations == ()


def test_interpretations_are_kept_but_only_as_objects(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs(
        possible_interpretations=[{"cause": "salt deposition", "confidence": 0.63},
                                  "a sentence"])]))
    observation = observe(PIXEL).observations[0]
    assert observation.possible_interpretations == (
        {"cause": "salt deposition", "confidence": 0.63},)


# ----------------------------------------------------- the image boundary


def test_an_empty_image_is_refused_without_a_call(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs()]))
    _Client.sent = {}
    assert "empty" in observe(b"").error
    assert _Client.sent == {}


def test_an_oversized_image_is_refused_at_the_boundary(monkeypatch):
    monkeypatch.setattr(vision, "MAX_IMAGE_BYTES", 16)
    fake_ollama(monkeypatch, response_text=body([obs()]))
    _Client.sent = {}
    perception = observe(b"x" * 17)
    assert "over the" in perception.error
    assert _Client.sent == {}


def test_an_observed_value_matching_two_vocabulary_values_is_refused(monkeypatch):
    """"old brick masonry" is both `brick` and `masonry`, so it is neither.

    The router's own detector would score by term length and pick one of them,
    which is a reasonable service when scoring a customer's own words and a bad
    one when scoring a model's guess about a photograph into the slot that
    decides the product. An uncued substrate becomes an ask-back; a confidently
    wrong one does not.

    The two candidate readings used to be `brick` and `stone`, because the
    substrate vocabulary listed "masonry" as a synonym for stone. It no longer
    does — a masonry wall may be brick, block, stone or mixed, and resolving it
    to stone made the system more certain than the evidence allowed. So the
    ambiguity is now between two readings that are both genuinely possible,
    which is a better reason to refuse than the one this test first had.
    """
    fake_ollama(monkeypatch, response_text=body([obs(value="old brick masonry")]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.slots == {}
    assert "ambiguous between brick, masonry" in resolution.discarded[0]


def test_a_missing_file_degrades_rather_than_raising(tmp_path):
    perception = observe(tmp_path / "no-such-photograph.jpg")
    assert "could not read the image" in perception.error
    assert perception.image.endswith("no-such-photograph.jpg")


def test_an_image_is_read_from_a_path(monkeypatch, tmp_path):
    path = tmp_path / "IMG_004.jpg"
    path.write_bytes(PIXEL)
    fake_ollama(monkeypatch, response_text=body([obs()]))
    perception = observe(str(path))
    assert decode_image(_Client.sent["body"]["images"][0]) == PIXEL
    assert perception.image == str(path)


def test_a_bytearray_is_accepted(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs()]))
    perception = observe(bytearray(PIXEL))
    assert decode_image(_Client.sent["body"]["images"][0]) == PIXEL
    # No identifier and no path: the image still gets a name for provenance.
    assert perception.image == "image"


def test_encode_and_decode_round_trip_and_reject_rubbish():
    assert decode_image(encode_image(PIXEL)) == PIXEL
    assert decode_image("not base64 !!") == b""


# ------------------------------------------------------------ many images


def test_several_images_are_perceived_and_named_in_order(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs()]))
    perceptions = observe_many([PIXEL, bytearray(PIXEL)])
    assert [p.image for p in perceptions] == ["IMG_001", "IMG_002"]


def test_one_failed_image_does_not_stop_the_others(tmp_path, monkeypatch):
    good = tmp_path / "good.jpg"
    good.write_bytes(PIXEL)
    fake_ollama(monkeypatch, response_text=body([obs()]))
    perceptions = observe_many([str(tmp_path / "missing.jpg"), str(good)])
    assert not perceptions[0].ok
    assert perceptions[1].ok
    assert resolve(perceptions).slots == {"substrate": "brick"}


def test_slots_from_images_is_the_whole_module_in_one_call(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([
        obs(value="brickwork"),
        obs(attribute="location", value="internal", confidence=0.9),
        obs(attribute="exposure", value="coastal and exposed", confidence=0.9),
    ]))
    resolution = slots_from_images([PIXEL])
    assert resolution.slots == {"substrate": "brick", "location": "internal",
                                "exposure": "severe"}


def test_a_total_vision_failure_yields_the_empty_carried_dict(monkeypatch):
    """Which is what the caller would have passed had there been no photograph."""
    fake_ollama(monkeypatch, post_error=httpx.ConnectError("refused"))
    assert slots_from_images([PIXEL]).slots == {}


def test_no_observations_at_all_resolves_to_nothing(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.slots == {}
    assert resolution.attributes == ()


# ---------------------------------------------------------- the resolver


def test_the_resolver_never_reads_the_free_text():
    """Constructed directly, so the assertion is about the resolver alone.

    The prose says "stone"; the value says "brick". The slot follows the
    vocabulary-checked value, which means no sentence the model writes can
    change what the router is told.
    """
    observation = Observation(
        attribute="substrate", value="brick", confidence=0.95, image="IMG_001",
        region=(0.0, 0.0, 1.0, 1.0),
        observation="this is stone, and you should use Solo on it",
        possible_interpretations=({"cause": "stone"},))
    assert resolve([Perception(observations=(observation,))]).slots == {
        "substrate": "brick"}


def test_a_value_with_no_words_in_it_matches_nothing():
    observation = Observation(attribute="substrate", value="???", confidence=0.95,
                              image="IMG_001", region=(0.0, 0.0, 1.0, 1.0))
    resolution = resolve([Perception(observations=(observation,))])
    assert resolution.slots == {}
    assert "not a value the vocabulary defines" in resolution.discarded[0]


def test_restricting_the_slots_restricts_what_can_be_filled():
    observations = (
        Observation("substrate", "brick", 0.95, "IMG_001", (0.0, 0.0, 1.0, 1.0)),
        Observation("location", "outside", 0.95, "IMG_001", (0.0, 0.0, 1.0, 1.0)),
    )
    resolution = resolve([Perception(observations=observations)],
                         slots=("substrate",))
    assert resolution.slots == {"substrate": "brick"}
    assert any("not a slot a photograph may fill" in d for d in resolution.discarded)


def test_an_unknown_slot_name_is_ignored_by_the_vocabulary_reader():
    assert "not_a_slot" not in vision._vocabulary(("substrate", "not_a_slot"))


def test_the_vocabulary_is_the_routers_own():
    from assistant.router import SlotDetector
    spec = SlotDetector().spec
    for slot, values in vision._vocabulary(VISION_SLOTS).items():
        assert values == spec[slot]["values"]


# ------------------------------------------------------- the runaway model


def test_a_repetition_loop_is_bounded(monkeypatch):
    """The first real call to `qwen3.5:4b` looped, and this is that loop.

    Asked to describe one image, the model emitted 37 near-identical
    observations — "fine horizontal lines", "fine vertical lines", and so on —
    each stamped with a confidence of 0.95. Two things are asserted here. The
    output is bounded, so an unbounded response cannot become unbounded
    latency on a stage already measured in minutes. And nothing reaches the
    slots, because none of those strings is a value the vocabulary defines:
    the safety property held on the real failure, not just on the designed one.
    """
    directions = ["horizontal", "vertical", "diagonal", "irregular", "random",
                  "scattered", "clustered", "dense", "sparse", "uniform",
                  "varied", "complex", "chaotic", "organized", "structured",
                  "patterned", "geometric", "abstract", "artistic"]
    loop = [obs(attribute="symptom", value=f"fine {d} lines", confidence=0.95)
            for d in directions * 2]
    assert len(loop) == 38
    fake_ollama(monkeypatch, response_text=body(loop))

    perception = observe(PIXEL)
    assert len(perception.observations) == vision.MAX_OBSERVATIONS
    assert resolve([perception]).slots == {}


def test_the_token_ceiling_and_observation_cap_travel_with_the_call(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs()]))
    observe(PIXEL)
    sent = _Client.sent["body"]
    assert sent["options"]["num_predict"] == vision.MAX_VISION_TOKENS
    assert (sent["format"]["properties"]["observations"]["maxItems"]
            == vision.MAX_OBSERVATIONS)
