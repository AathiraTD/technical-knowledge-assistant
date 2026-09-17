"""What the perception contract promises, and the real failure that shaped it.

`tests/test_vision.py` was written against a contract that had never met a
photograph. Every test in it feeds `_decode` a complete, well-formed string,
and on that diet the module looked finished. The first real image through the
real model produced **nothing at all**: Ollama pretty-prints a constrained
response, `qwen3.5:4b` looped, `num_predict` ran out mid-object, `json.loads`
refused the fragment, and 204 seconds returned an empty `Perception`. Every
real photograph did that. The module was safe and completely useless -- a
failure worth naming as loudly as an unsafe one, because nothing printed,
nothing warned, and the page still said a photograph had been read.

So this file covers the things that only exist once a real model is on the
other end: a response that stops mid-sentence, a model repeating itself, the
three tiers of attribute, the four words a reading is reported in, and the
gate that stops a substrate being claimed through a render.

The helpers come from `test_vision.py` rather than being rebuilt, so both
files drive the decoder the same way and a change to the fake HTTP client
cannot leave one of them testing something else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant import vision                                        # noqa: E402
from assistant.vision import (                                      # noqa: E402
    Observation, Perception, VISION_SLOTS, observe, resolve,
)

from test_vision import PIXEL, _Client, body, fake_ollama, obs       # noqa: E402


# =====================================================================
#  The truncated response
# =====================================================================

TRUNCATED = (
    '{\n  "observations": [\n'
    '    {\n      "observation": "coursed brickwork",\n'
    '      "attribute": "substrate",\n      "value": "brick",\n'
    '      "confidence": 0.95,\n      "region": [0.0, 0.0, 1.0, 1.0]\n    },\n'
    '    {\n      "observation": "mortar joints",\n'
    '      "attribute": "texture",\n      "value": "open joints",\n'
    '      "confidence": 0.9,\n      "region": [0.1, 0.1, 0.9, 0.9]\n    },\n'
    '    {\n      "observation": "brick pattern",\n'
    '      "attribute": "staining",\n      "value": "none",\n'
    '      "confidence'
)


def test_a_truncated_response_keeps_what_completed(monkeypatch):
    """The recovery, and its exact limit: complete objects only.

    Two readings finished and a third did not. The two are recovered whole and
    the half-written one is dropped whole -- no field defaulted, guessed or
    repaired -- so this can fail to recover a reading and cannot invent one.
    """
    fake_ollama(monkeypatch, response_text=TRUNCATED)
    perception = observe(PIXEL)

    assert perception.truncated is True
    assert perception.ok, "a recoverable response was thrown away entirely"
    assert [(o.attribute, o.value) for o in perception.observations] == [
        ("substrate", "brick"), ("texture", "open joints")]


def test_a_truncated_reading_is_reported_but_never_fills_a_slot(monkeypatch):
    """A model at the token ceiling is looping, and a loop is not evidence.

    The readings are still shown -- a person can weigh them, and silently
    dropping them is how "the assistant ignored my photo" happens. What they
    may not do is decide anything.
    """
    fake_ollama(monkeypatch, response_text=TRUNCATED)
    resolution = resolve([observe(PIXEL)])

    assert resolution.truncated is True
    assert resolution.slots == {}
    substrate = next(a for a in resolution.attributes if a.slot == "substrate")
    assert substrate.certainty is vision.Certainty.LIKELY
    assert substrate.withheld == "the response was truncated"
    assert not substrate.routed
    assert any("truncated" in d for d in resolution.discarded)


def test_cannot_determine_may_be_absent_only_because_it_was_cut_off(monkeypatch):
    """The required field stays required on a *complete* response.

    It comes after `observations` in the schema, so a truncation means it never
    arrived -- missing because the sentence stopped, not because the model
    declined it. A complete response that skips it has not answered the
    contract and is still refused: the field exists to stop the model
    inventing, and defaulting it on a whole response would invent it on the
    model's behalf.
    """
    fake_ollama(monkeypatch, response_text=TRUNCATED)
    assert observe(PIXEL).ok

    fake_ollama(monkeypatch,
                response_text=json.dumps({"observations": [obs()]}))
    complete = observe(PIXEL)
    assert not complete.ok
    assert "observation contract" in complete.error


@pytest.mark.parametrize("text", [
    '{"observations": [',                       # nothing closed at all
    '{"observations": [{"attribute": "subs',    # cut inside the first object
    '{"observations": ',                        # cut before the array
    "not json at all",                          # never started
])
def test_an_unrecoverable_truncation_degrades_to_nothing(monkeypatch, text):
    fake_ollama(monkeypatch, response_text=text)
    perception = observe(PIXEL)
    assert not perception.ok
    assert resolve([perception]).slots == {}


def test_a_brace_inside_a_string_does_not_fool_the_scanner():
    """The scanner tracks string state, so punctuation in prose is not depth."""
    text = ('{"observations": [{"observation": "a note with } and ] and \\" '
            'in it", "attribute": "substrate", "value": "brick", '
            '"confidence": 0.95, "region": [0.0, 0.0, 1.0, 1.0]}, '
            '{"observation": "cut')
    recovered, truncated = vision.salvage(text)

    assert truncated is True
    assert len(recovered["observations"]) == 1
    assert recovered["observations"][0]["value"] == "brick"


def test_a_complete_response_is_not_reported_as_truncated(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([obs()]))
    perception = observe(PIXEL)
    assert perception.truncated is False
    assert resolve([perception]).truncated is False


# =====================================================================
#  The repetition loop
# =====================================================================


def test_the_call_asks_the_model_not_to_repeat_itself(monkeypatch):
    """Temperature zero makes a model deterministic, not sensible.

    With no penalty the highest-probability continuation after an observation
    is often the same observation again, and greedy decoding takes it every
    time. This is the one option that acts on that directly.
    """
    fake_ollama(monkeypatch, response_text=body([obs()]))
    observe(PIXEL)
    assert (_Client.sent["body"]["options"]["repeat_penalty"]
            == vision.VISION_REPEAT_PENALTY)


def test_the_same_reading_twice_is_one_reading(monkeypatch):
    """Twelve copies of one guess must not look like twelve images agreeing.

    The aggregation counts corroboration, so a loop surviving the decoder would
    manufacture it -- the opposite of what a repeated token is evidence of.
    """
    fake_ollama(monkeypatch, response_text=body([obs(), obs(), obs()]))
    assert len(observe(PIXEL).observations) == 1


def test_the_token_budget_leaves_room_for_the_observation_cap():
    """The two constants are one fix and have to stay consistent.

    Six observations at roughly forty tokens each -- a pretty-printed region is
    six lines on its own -- sits under three hundred, inside a budget of nine
    hundred. An edit raising the cap without raising the budget would silently
    restore the truncation the pair exists to prevent.
    """
    assert vision.MAX_VISION_TOKENS >= 60 * vision.MAX_OBSERVATIONS


# =====================================================================
#  Three tiers, four words
# =====================================================================


def test_the_tiers_are_disjoint_and_cover_the_schema():
    assert set(VISION_SLOTS) == (set(vision.ROUTER_SLOTS)
                                 | set(vision.CONTEXT_SLOTS)
                                 | set(vision.CONDITION_ATTRIBUTES))
    assert not set(vision.ROUTER_SLOTS) & set(vision.CONTEXT_SLOTS)
    assert not set(vision.ROUTER_SLOTS) & set(vision.CONDITION_ATTRIBUTES)
    assert not set(vision.CONTEXT_SLOTS) & set(vision.CONDITION_ATTRIBUTES)
    # The two that may decide a product, and nothing else.
    assert set(vision.ROUTER_SLOTS) == {"substrate", "symptom"}
    # The two the first real call proved a photograph does not carry.
    assert set(vision.CONTEXT_SLOTS) == {"location", "exposure"}


def test_no_tier_contains_a_claim_a_photograph_may_not_carry():
    for attribute in VISION_SLOTS:
        assert attribute not in vision.FORBIDDEN_ATTRIBUTES
    for forbidden in ("product", "cause_asked", "structural", "compliance",
                      "mortar_chemistry", "hidden_substrate"):
        assert forbidden in vision.FORBIDDEN_ATTRIBUTES


def test_a_forbidden_attribute_is_refused_and_counted(monkeypatch):
    """Unreachable through the enum, which is exactly why it is checked anyway.

    If it ever becomes reachable -- a widened schema, a provider that does not
    constrain the response -- this makes the attempt *visible* rather than
    merely unsuccessful. Silence about an attack that failed is how the next
    one succeeds unnoticed.
    """
    text = body([obs(attribute="product", value="Solo"),
                 obs(attribute="structural", value="movement"),
                 obs()])
    fake_ollama(monkeypatch, response_text=text)
    perception = observe(PIXEL, slots=VISION_SLOTS + ("product", "structural"))

    assert set(perception.refused) == {"product", "structural"}
    assert [o.attribute for o in perception.observations] == ["substrate"]
    assert set(resolve([perception]).refused) == {"product", "structural"}


def test_the_condition_attributes_have_their_own_vocabulary():
    """Tier 3 is constrained exactly as tightly as the router slots are.

    Held outside `config/vocabularies.json` because the router has no slot for
    any of them, and putting them in the router's file would imply a routing
    behaviour that deliberately does not exist.
    """
    for attribute in vision.CONDITION_ATTRIBUTES:
        assert attribute in vision.CONDITION_VALUES
        assert vision.CONDITION_VALUES[attribute], attribute
    assert vision._vocabulary(VISION_SLOTS).keys() <= set(
        vision.ROUTER_SLOTS + vision.CONTEXT_SLOTS)


def test_an_invented_condition_value_reaches_nothing(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([
        obs(attribute="existing_finish", value="Lime Green Solo topcoat"),
        obs(attribute="cracks", value="catastrophic structural failure"),
    ]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.conditions == ()
    assert len(resolution.discarded) == 2


def test_a_condition_is_reported_and_never_routed(monkeypatch):
    fake_ollama(monkeypatch, response_text=body([
        obs(attribute="staining", value="white deposits", confidence=0.95)]))
    resolution = resolve([observe(PIXEL)])

    assert resolution.slots == {}
    assert [(c.slot, c.value) for c in resolution.conditions] == [
        ("staining", "white")]
    assert resolution.conditions[0].certainty is vision.Certainty.OBSERVED


@pytest.mark.parametrize("confidence, expected", [
    (0.95, vision.Certainty.OBSERVED),
    (0.72, vision.Certainty.LIKELY),
    (0.55, vision.Certainty.UNCERTAIN),
    (0.20, vision.Certainty.CANNOT_DETERMINE),
])
def test_the_four_words_a_reading_is_reported_in(monkeypatch, confidence,
                                                 expected):
    fake_ollama(monkeypatch, response_text=body([obs(confidence=confidence)]))
    resolution = resolve([observe(PIXEL)])
    assert resolution.attributes[0].certainty is expected
    assert bool(resolution.slots) is (expected is vision.Certainty.OBSERVED)


def test_only_observed_ever_routes(monkeypatch):
    """The ordering of the four words is the safety property, asserted as one."""
    for confidence in (0.0, 0.3, 0.49, 0.5, 0.64, 0.65, 0.79, 0.8, 1.0):
        fake_ollama(monkeypatch,
                    response_text=body([obs(confidence=confidence)]))
        resolution = resolve([observe(PIXEL)])
        for attribute in resolution.attributes:
            assert attribute.routed == (attribute.slot in resolution.slots)
            if attribute.routed:
                assert attribute.certainty is vision.Certainty.OBSERVED


def test_the_model_saying_it_cannot_tell_outranks_the_model_guessing(monkeypatch):
    """The one place the thing being measured gets to say "I cannot".

    Honouring it is what makes asking worth anything. A model that lists
    `location` as undeterminable and then reports one has contradicted itself,
    and the refusal is the half to believe.
    """
    fake_ollama(monkeypatch, response_text=body(
        [obs(attribute="location", value="exterior", confidence=0.6)],
        cannot=["location", "the moisture source"]))
    resolution = resolve([observe(PIXEL)])

    location = next(a for a in resolution.attributes if a.slot == "location")
    assert location.certainty is vision.Certainty.CANNOT_DETERMINE
    assert "cannot settle this" in location.withheld


# =====================================================================
#  The substrate gate
# =====================================================================


def _seen(*pairs, confidence=0.93):
    return Perception(
        observations=tuple(
            Observation(attribute=a, value=v, confidence=confidence,
                        image="IMG_001", region=(0.0, 0.0, 1.0, 1.0))
            for a, v in pairs),
        cannot_determine_from_image=("the moisture source",))


def test_a_substrate_is_refused_through_a_covering_finish():
    """"Exact hidden substrate" is on the list a photograph may not carry.

    A model asked what a rendered wall is made of will answer "brick", because
    most walls are -- a prior dressed as a perception. It has already said the
    wall is rendered and has not said the masonry is exposed, so by its own two
    readings the background is concealed.
    """
    resolution = resolve([_seen(("substrate", "brick"),
                                ("existing_finish", "render"))])

    assert resolution.slots == {}
    substrate = next(a for a in resolution.attributes if a.slot == "substrate")
    assert "covers the background" in substrate.withheld
    assert substrate.certainty is vision.Certainty.LIKELY


def test_exposed_masonry_reopens_the_gate():
    """A blown patch showing the background is the case vision is good at."""
    resolution = resolve([_seen(("substrate", "brick"),
                                ("existing_finish", "plaster"),
                                ("exposed_masonry", "in places"))])
    assert resolution.slots["substrate"] == "brick"


def test_bare_masonry_is_not_asked_to_prove_a_negative():
    """The gate is a contradiction, not a requirement, and this is why.

    A photograph of bare brickwork carries no finish observation at all.
    Demanding a positive "yes, masonry is exposed" before believing the obvious
    would refuse the one case a photograph settles unambiguously.
    """
    assert resolve([_seen(("substrate", "brick"))]).slots == {
        "substrate": "brick"}


def test_a_finish_of_none_is_not_a_covering():
    assert resolve([_seen(("substrate", "stone"),
                          ("existing_finish", "bare"))]).slots == {
        "substrate": "stone"}


def test_an_uncertain_finish_does_not_gate_a_confident_substrate():
    """A weak reading may not veto a strong one; it is not evidence either."""
    perception = Perception(
        observations=(
            Observation("substrate", "brick", 0.95, "IMG_001", (0, 0, 1, 1)),
            Observation("existing_finish", "render", 0.35, "IMG_001",
                        (0, 0, 1, 1)),
        ), cannot_determine_from_image=("the moisture source",))
    assert resolve([perception]).slots["substrate"] == "brick"


# =====================================================================
#  The report a surface prints
# =====================================================================


def test_the_report_carries_the_hedge_with_every_claim():
    report = vision.perception_report(
        resolve([_seen(("substrate", "brick"), ("existing_finish", "render"))]))

    assert report["routed"] == {}
    rows = {r["attribute"]: r for r in report["observations"]}
    assert rows["substrate"]["certainty"] == "LIKELY"
    assert rows["substrate"]["withheld"]
    assert rows["substrate"]["routed"] is False
    # Every row, without exception, says how firmly it is held.
    assert all(r["certainty"] for r in report["observations"])


def test_the_report_is_plain_data_a_checkpoint_can_hold():
    """No enum, no dataclass, no tuple: it round-trips through JSON."""
    report = vision.perception_report(resolve([_seen(("substrate", "brick"))]))
    assert json.loads(json.dumps(report)) == report


def test_the_report_survives_a_provider_that_is_not_the_real_resolver():
    """A reporting function must never be able to fail an answer.

    `Services.vision` is injected, and a channel adapter's provider satisfies
    "returns something with slots and attributes". Raising on a missing
    `certainty` would invert this module's one rule: failure reduces coverage,
    not safety.
    """
    class Minimal:
        slots = {"substrate": "brick"}

        class _Attribute:
            slot, value, confidence, sources = "substrate", "brick", 0.9, ()

        attributes = (_Attribute(),)
        cannot_determine_from_image = ()
        discarded = ()

    report = vision.perception_report(Minimal())

    assert report["routed"] == {"substrate": "brick"}
    assert report["observations"][0]["certainty"] == "OBSERVED"
    assert report["truncated"] is False


def test_the_prompt_lists_the_values_and_names_what_cannot_be_seen():
    """Telling the model the answers changes the hit rate, not the guarantee.

    `resolve()` checks every value against the same vocabulary regardless of
    what the prompt said. Without the menu the first real call answered
    `{"attribute": "exposure", "value": "visible"}` twelve times: the enum
    forced a legal attribute, nothing suggested a legal value, and every
    reading was discarded. Perfect safety, zero coverage.
    """
    text = vision.prompt_for()

    assert "brick" in text and "crazing" in text
    assert "ONE observation per attribute" in text
    assert "internal or external" in text
    assert "Do not name any product" in text
