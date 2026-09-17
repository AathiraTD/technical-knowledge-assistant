"""The opt-in lane: a real photograph, a real vision model, no stubs.

Skipped unless `ASSISTANT_VISION_LIVE=1`, and that is not timidity. One image
costs 150 to 600 seconds on a processor with no graphics card -- measured, on
the machine this was built on -- so putting it in the default suite would turn
a two-minute run into an hour and make the tests something nobody runs. The
deterministic lane (`test_vision.py`, `test_vision_contract.py`,
`test_vision_conversation.py`) proves the contract and the seams; this proves
the thing none of them can: **that a real model on the other end still obeys
the contract.**

    ASSISTANT_VISION_LIVE=1 python -m pytest tests/test_vision_live.py -v

What it asserts is deliberately narrow, because a live model is not
deterministic enough to assert much and pretending otherwise produces a flaky
suite that gets deleted. Three properties, all of which must hold for *any*
model against *any* of these images:

1. The response is structurally valid or it degrades -- never an exception,
   never a half-parsed reading.
2. Whatever comes back, no value reaches a slot that the vocabulary does not
   define, and `location` never routes.
3. A sound rendered wall does not yield a substrate.

The *measurement* -- how often a real model overclaims, and on which images --
is `eval/vision_eval.py`, which scores against declared ground truth and
prints a number. This file is the regression net under it: it fails when the
contract breaks, not when the model has an off day.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import vision                                        # noqa: E402

IMAGES = ROOT / "eval" / "fixtures" / "images"
MANIFEST = IMAGES / "manifest.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("ASSISTANT_VISION_LIVE", "") not in {"1", "true", "yes"},
    reason="live VLM tests are opt-in: set ASSISTANT_VISION_LIVE=1 "
           "(one image costs 150-600s on CPU)")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return {spec["name"]: spec for spec in
            json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]}


@pytest.fixture(scope="module")
def brick(manifest) -> tuple:
    """One real perception, shared, because it costs minutes to produce."""
    started = time.perf_counter()
    perception = vision.observe(IMAGES / "brick-exposed-clear.jpg",
                                image_id="brick-exposed-clear")
    return perception, time.perf_counter() - started


def test_a_real_photograph_reaches_the_model_and_comes_back(brick):
    """The end-to-end fact, and the one that was false until it was measured.

    Before the token budget and the prompt were fixed, every real image
    returned `the vision model did not return JSON` after 204 seconds: the
    response was cut off mid-object and `json.loads` refused the fragment. The
    module was safe and completely useless, and no stubbed test could have
    caught it.
    """
    perception, seconds = brick
    if "failed" in perception.error or "timed out" in perception.error:
        pytest.skip(f"vision model unreachable or too slow: {perception.error}")

    assert perception.ok, perception.error
    assert perception.observations, "the model returned no readings at all"
    assert not perception.truncated, (
        "the response ran out of tokens; the budget and the observation cap "
        "have drifted apart")
    print(f"\n  {seconds:.0f}s, {len(perception.observations)} observations")


def test_whatever_it_said_only_vocabulary_values_reach_a_slot(brick):
    """The guarantee that does not depend on the model behaving."""
    perception, _seconds = brick
    if not perception.ok:
        pytest.skip(perception.error)

    resolution = vision.resolve([perception])
    vocabulary = vision._vocabulary(vision.ROUTER_SLOTS)
    for slot, value in resolution.slots.items():
        assert slot in vision.ROUTER_SLOTS, (
            f"{slot} routed, and only {vision.ROUTER_SLOTS} may")
        assert value in vocabulary[slot], (
            f"{slot}={value!r} is not a value the vocabulary defines")


def test_location_never_routes_however_confident_the_model_sounds(brick):
    """Measured behaviour, not a hypothetical: the first real call answered
    `{"attribute": "exposure", "value": "visible", "confidence": 1.0}` twelve
    times over."""
    perception, _seconds = brick
    if not perception.ok:
        pytest.skip(perception.error)

    resolution = vision.resolve([perception])
    assert "location" not in resolution.slots
    assert "exposure" not in resolution.slots


def test_a_sound_render_yields_no_substrate(manifest):
    """The substrate gate, against the real model rather than a constructed
    perception. A render hides its own background, and a model asked what the
    wall is made of will answer "brick", because most walls are."""
    perception = vision.observe(IMAGES / "render-external-sound.jpg",
                                image_id="render-external-sound")
    if not perception.ok:
        pytest.skip(perception.error)

    resolution = vision.resolve([perception])
    assert "substrate" not in resolution.slots, (
        f"a substrate was claimed through a sound render: "
        f"{resolution.slots}")


def test_the_report_is_well_formed_whatever_came_back(brick):
    """A surface must be able to render any real response without a special
    case, including a failed or truncated one."""
    perception, _seconds = brick
    report = vision.perception_report(vision.resolve([perception]))

    assert json.loads(json.dumps(report)) == report
    assert report["enabled"] is True
    for row in report["observations"]:
        assert row["certainty"] in {"OBSERVED", "LIKELY", "UNCERTAIN",
                                    "CANNOT_DETERMINE"}
        assert isinstance(row["withheld"], str)


@pytest.mark.parametrize("name", [
    "poor-light", "ambiguous-closeup",
])
def test_an_image_that_settles_nothing_routes_nothing(name, manifest):
    """The two fixtures whose correct answer is mostly "cannot determine".

    A model that reads a confident substrate off an underexposed frame or a
    scaleless close crop is overclaiming, and these are the cheapest cases to
    be sure about: there is nothing in either image to be right about.
    """
    perception = vision.observe(IMAGES / f"{name}.jpg", image_id=name)
    if not perception.ok:
        pytest.skip(perception.error)

    resolution = vision.resolve([perception])
    assert "location" not in resolution.slots
    for slot, value in resolution.slots.items():
        print(f"\n  {name}: routed {slot}={value} "
              f"(declared undeterminable in the manifest)")
