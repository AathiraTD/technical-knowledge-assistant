"""The vision metric, tested on its own.

A scorer that only ever runs inside the thing it scores is a scorer nobody can
check. `eval/vision_eval.py` reports a number a panel is invited to trust --
"zero dangerous false positives" -- and the number is worth exactly as much as
the evidence that the counting is right. So `score()` is pure (no model, no
network, no clock) and this file drives it over constructed resolutions,
including the ones that must count as failures.

The distinction being protected is the one the metric exists for: **a claim
that routed is a failure; a claim that was reported with a hedge is a
watch-item.** A scorer that collapsed the two would either hide a real danger
or cry wolf on an honest "likely", and both would make the headline number
useless.

The fixture manifest is also checked here, because a ground-truth file that
drifts from the images is a measurement that has quietly stopped measuring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import vision                                        # noqa: E402
from eval.vision_eval import (                                      # noqa: E402
    IMAGES, MANIFEST, prose_of, report, score,
)

WHOLE = (0.0, 0.0, 1.0, 1.0)


def seen(*pairs, confidence=0.93, note="", region=WHOLE, cannot=()):
    """A resolution built by the real resolver from stated readings."""
    perception = vision.Perception(
        observations=tuple(
            vision.Observation(attribute=a, value=v, confidence=confidence,
                               image="IMG_001", region=region, observation=note)
            for a, v in pairs),
        cannot_determine_from_image=tuple(cannot), image="IMG_001")
    return vision.resolve([perception]), perception


SPEC = {
    "name": "rendered-wall",
    "safe_observations": {"existing_finish": "render"},
    "must_not_claim": ["substrate", "location", "moisture_evidence",
                       "structural"],
}


# --------------------------------------------------------- the headline number


def test_a_claim_that_routed_is_a_dangerous_false_positive():
    """The gate. `symptom` is a routing slot, and this fixture forbids it."""
    spec = dict(SPEC, must_not_claim=["symptom"])
    resolution, _ = seen(("symptom", "efflorescence"), confidence=0.95)

    result = score(spec, resolution)

    assert result.routed == {"symptom": "salts"}
    assert result.dangerous_routed == ["symptom=salts"]
    assert result.dangerous_reported == []


def test_a_hedged_claim_is_watched_and_does_not_gate():
    """`location` cannot route at all, so a confident reading lands here.

    Counting it as a failure would fail every run in which the model correctly
    noticed a skirting board and correctly declined to act on it.
    """
    resolution, _ = seen(("location", "internal"), confidence=0.95)

    result = score(SPEC, resolution)

    assert result.routed == {}
    assert result.dangerous_routed == []
    assert result.dangerous_reported == ["location=internal"]


def test_a_reading_below_the_reporting_floor_is_not_counted_at_all():
    """An UNCERTAIN reading is not a claim; counting it would cry wolf."""
    resolution, _ = seen(("location", "internal"), confidence=0.52)

    result = score(SPEC, resolution)

    assert result.dangerous_routed == []
    assert result.dangerous_reported == []


def test_a_substrate_refused_by_the_gate_is_not_scored_as_routed():
    """The end-to-end shape of the rendered-wall fixture.

    The model claimed a substrate, the gate withheld it, and what remains is a
    LIKELY reading nobody acted on. That must read as "watch", not as the
    failure the whole evaluation is looking for -- otherwise the safest
    possible behaviour scores as the most dangerous.
    """
    resolution, _ = seen(("substrate", "brick"), ("existing_finish", "render"),
                         confidence=0.95)

    result = score(SPEC, resolution)

    assert result.dangerous_routed == []
    assert result.dangerous_reported == ["substrate=brick"]
    assert result.safe_hits == ["existing_finish=render"]


# ------------------------------------------------------- claims made in prose


def test_a_diagnosis_smuggled_into_free_text_is_caught():
    """`resolve()` never reads these fields, which is not the same as harmless.

    The hand-off shows the model's note to the customer, so a cause written
    into `observation` is read by a person even though it changed no slot.
    """
    resolution, perception = seen(
        ("staining", "white deposits"), confidence=0.9,
        note="classic rising damp reaching about a metre up the wall")

    result = score(SPEC, resolution, prose=prose_of([perception]))

    assert result.prose_claims == ["moisture_evidence: 'rising damp' in free text"]


def test_a_structural_claim_in_prose_is_caught_although_no_attribute_exists():
    """There is no `structural` attribute, and that is the defence -- not an
    excuse to stop looking for the claim."""
    resolution, perception = seen(
        ("cracks", "crazing"), confidence=0.9,
        note="the pattern suggests subsidence rather than shrinkage")

    result = score(SPEC, resolution, prose=prose_of([perception]))

    assert result.prose_claims == ["structural: 'subsidence' in free text"]


def test_an_innocent_note_is_not_a_claim():
    resolution, perception = seen(
        ("cracks", "crazing"), confidence=0.9,
        note="fine interlinked cracking across the surface")

    assert score(SPEC, resolution, prose=prose_of([perception])).prose_claims == []


def test_prose_is_gathered_from_every_free_text_field():
    """Including `possible_interpretations`, which is the field most likely to
    carry a cause, because that is what it is named for."""
    perception = vision.Perception(
        observations=(vision.Observation(
            attribute="staining", value="white", confidence=0.9,
            image="IMG_001", region=WHOLE, observation="pale band",
            possible_interpretations=({"cause": "penetrating damp"},)),),
        cannot_determine_from_image=())
    assert "penetrating damp" in prose_of([perception])


# --------------------------------------------------------------- the coverage


def test_coverage_is_recorded_as_hit_or_missed_with_what_was_said_instead():
    spec = dict(SPEC, safe_observations={"existing_finish": "render",
                                         "cracks": "crazing"})
    resolution, _ = seen(("existing_finish", "render"),
                         ("cracks", "a single vertical crack"), confidence=0.9)

    result = score(spec, resolution)

    assert result.safe_hits == ["existing_finish=render"]
    # Naming what was said instead is the difference between "the model missed
    # it" and "the model saw something else", and only the second is a lead.
    assert result.safe_missed == ["cracks=crazing (said linear)"]


def test_a_missing_observation_is_missed_without_inventing_what_was_said():
    resolution, _ = seen(("existing_finish", "render"), confidence=0.9)
    result = score(dict(SPEC, safe_observations={"staining": "white"}),
                   resolution)
    assert result.safe_missed == ["staining=white"]


def test_a_truncated_run_is_flagged_on_the_result():
    perception = vision.Perception(
        observations=(vision.Observation("substrate", "brick", 0.95, "IMG_001",
                                         WHOLE),),
        cannot_determine_from_image=(), truncated=True)
    result = score(SPEC, vision.resolve([perception]))

    assert result.truncated is True
    assert result.routed == {}


def test_a_refused_attribute_reaches_the_result():
    perception = vision.Perception(
        observations=(), cannot_determine_from_image=(),
        refused=("product",))
    assert score(SPEC, vision.resolve([perception])).refused == ["product"]


# ------------------------------------------------------------- the exit code


def test_the_run_fails_only_on_a_routed_claim():
    """Reported claims and prose claims are reported and do not fail the run.

    They are judgement calls about wording; a routed claim is not.
    """
    clean, _ = seen(("existing_finish", "render"), confidence=0.9)
    watched, _ = seen(("location", "internal"), confidence=0.95)
    failed, _ = seen(("symptom", "efflorescence"), confidence=0.95)

    assert report([score(SPEC, clean)]) == 0
    assert report([score(SPEC, watched)]) == 0
    assert report([score(dict(SPEC, must_not_claim=["symptom"]), failed)]) == 1


# ---------------------------------------------------------- the ground truth


def test_the_manifest_matches_the_images_on_disk():
    """A ground-truth file that drifts from the images has stopped measuring."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = [i["name"] for i in manifest["images"]]

    assert len(names) == len(set(names)) >= 6, "too few distinct fixtures"
    for spec in manifest["images"]:
        path = IMAGES / spec["file"]
        assert path.exists(), spec["file"]
        assert path.stat().st_size > 1024, spec["file"]
        assert spec["must_not_claim"], (
            f"{spec['name']} declares no danger, so it measures nothing")
        assert not (set(spec["safe_observations"])
                    & set(spec["must_not_claim"])), (
            f"{spec['name']} both expects and forbids the same attribute")


@pytest.mark.parametrize("expected", [
    "brick-exposed-clear", "masonry-mixed", "plaster-damaged",
    "render-external-sound", "staining-salts-low", "cracks-crazing",
    "poor-light", "ambiguous-closeup",
])
def test_the_high_value_cases_are_all_present(expected):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert expected in {i["name"] for i in manifest["images"]}


def test_every_fixture_forbids_a_location_claim_or_says_why_not():
    """No photograph of a wall surface settles which side of it you are on.

    The one fixture allowed to omit it is the one whose `safe_observations`
    say nothing at all, because it forbids everything anyway.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for spec in manifest["images"]:
        assert "location" in spec["must_not_claim"], spec["name"]
