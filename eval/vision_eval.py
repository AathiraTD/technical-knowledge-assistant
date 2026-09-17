"""Score the perception stage on the thing that can actually hurt somebody.

**Not recognition accuracy.** A vision model that fails to notice brickwork
costs coverage: the substrate stays uncued, the router asks back, and the
person types "brick". That is the system working. A vision model that reports
`location = external` for an interior wall, or reads a salt bloom as rising
damp, costs a building -- it routes a confident, cited recommendation at a wall
that was never described, and decision 16's whole argument is that this is the
worst failure available here.

So the headline number is **dangerous false positives**: claims on a fixture's
`must_not_claim` list, which `eval/fixtures/images/manifest.json` declares per
image. The accuracy figures are reported beside it and are explicitly the
secondary measurement. A run that recognised nothing and claimed nothing scores
zero dangerous false positives and says so honestly; a run that recognised
everything and claimed one substrate through a sound render has failed.

Two claims are counted, and the difference between them is the point:

* **Routed** -- the claim reached `Resolution.slots` and would have changed a
  recommendation. This is the number that gates.
* **Reported** -- the claim was shown to the caller at OBSERVED or LIKELY
  certainty but did not route. Worth watching and not a gate: a hedged sentence
  a person reads is not the same as a fact the router acted on.

Run against the real model:

    python -m eval.vision_eval                    # every fixture
    python -m eval.vision_eval --image poor-light
    python -m eval.vision_eval --model qwen3-vl:4b

Each image costs minutes on a processor with no graphics card, so the whole set
is a deliberate, opt-in run. It is not part of the unit suite and must not
become part of it -- `tests/test_vision_eval.py` runs this same scoring against
recorded perceptions, with no model and no network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import vision                                        # noqa: E402

IMAGES = ROOT / "eval" / "fixtures" / "images"
MANIFEST = IMAGES / "manifest.json"

# Attributes named in a `must_not_claim` list that no attribute in
# `VISION_SLOTS` corresponds to. They cannot be claimed through the schema at
# all, so a claim can only arrive as free text, and that is where it is looked
# for. Kept explicit so the manifest can name a danger the contract has no
# field for -- "structural" is the example, and the fact that there is no
# structural attribute is the defence, not an excuse to stop checking.
PROSE_ONLY = {
    "structural": ("structural", "movement", "subsidence", "settlement",
                   "unsafe", "load bearing", "load-bearing"),
    "moisture_evidence": ("rising damp", "penetrating damp", "damp proof",
                          "dpc", "moisture ingress", "water ingress",
                          "condensation", "leak"),
}


@dataclass
class ImageResult:
    """What one photograph produced, and what it cost."""

    name: str
    seconds: float = 0.0
    error: str = ""
    truncated: bool = False
    routed: dict = field(default_factory=dict)
    reported: dict = field(default_factory=dict)
    cannot_determine: list = field(default_factory=list)
    refused: list = field(default_factory=list)
    # The findings.
    dangerous_routed: list = field(default_factory=list)
    dangerous_reported: list = field(default_factory=list)
    prose_claims: list = field(default_factory=list)
    safe_hits: list = field(default_factory=list)
    safe_missed: list = field(default_factory=list)


def score(spec: dict, resolution, seconds: float = 0.0,
          prose: str = "") -> ImageResult:
    """One resolution against one fixture's declared ground truth.

    Pure: no model, no network, no clock. That is what lets the unit suite run
    exactly this function over recorded perceptions and assert the scoring
    itself is right, rather than trusting a number produced by the thing being
    measured.
    """
    result = ImageResult(name=spec["name"], seconds=seconds,
                         truncated=bool(getattr(resolution, "truncated", False)),
                         refused=list(getattr(resolution, "refused", ())),
                         cannot_determine=list(
                             resolution.cannot_determine_from_image))

    result.routed = dict(resolution.slots)
    for attribute in resolution.attributes:
        if attribute.certainty in (vision.Certainty.OBSERVED,
                                   vision.Certainty.LIKELY):
            result.reported[attribute.slot] = attribute.value

    forbidden = set(spec.get("must_not_claim", ()))
    for slot, value in result.routed.items():
        if slot in forbidden:
            result.dangerous_routed.append(f"{slot}={value}")
    for slot, value in result.reported.items():
        if slot in forbidden and slot not in resolution.slots:
            result.dangerous_reported.append(f"{slot}={value}")

    # Dangers with no attribute to arrive through can still arrive as prose.
    lowered = prose.lower()
    for danger in forbidden:
        for term in PROSE_ONLY.get(danger, ()):
            if term in lowered:
                result.prose_claims.append(f"{danger}: {term!r} in free text")
                break

    for slot, value in spec.get("safe_observations", {}).items():
        seen = result.reported.get(slot)
        if seen == value:
            result.safe_hits.append(f"{slot}={value}")
        else:
            result.safe_missed.append(
                f"{slot}={value}" + (f" (said {seen})" if seen else ""))
    return result


def prose_of(perceptions) -> str:
    """Every free-text field the model wrote, for the prose check.

    `resolve()` never reads these, which is the guarantee. This function exists
    because "the resolver ignores it" is not the same as "it is harmless" --
    the hand-off shows the model's note to a person, so a diagnosis smuggled
    into `observation` would be read by the customer even though it changed no
    slot.
    """
    parts = []
    for perception in perceptions:
        for observation in perception.observations:
            parts.append(observation.observation)
            for interpretation in observation.possible_interpretations:
                parts.extend(str(v) for v in interpretation.values())
    return " ".join(parts)


def run(names=None, model: str = "") -> list[ImageResult]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    for spec in manifest["images"]:
        if names and spec["name"] not in names:
            continue
        path = IMAGES / spec["file"]
        started = time.perf_counter()
        perception = vision.observe(path, image_id=spec["name"], model=model)
        seconds = time.perf_counter() - started
        resolution = vision.resolve([perception])
        result = score(spec, resolution, seconds, prose_of([perception]))
        result.error = perception.error
        results.append(result)
        _print_one(spec, result)
    return results


def _print_one(spec: dict, r: ImageResult) -> None:
    print(f"\n{'=' * 72}\n{r.name}  --  {spec['title']}\n{'=' * 72}")
    print(f"  {r.seconds:.1f}s"
          + ("  TRUNCATED" if r.truncated else "")
          + (f"  ERROR: {r.error}" if r.error else ""))
    print(f"  routed   : {r.routed or '{}'}")
    print(f"  reported : {r.reported or '{}'}")
    if r.cannot_determine:
        print(f"  cannot   : {'; '.join(r.cannot_determine[:4])}")
    if r.refused:
        print(f"  REFUSED ATTRIBUTES: {r.refused}")
    if r.dangerous_routed:
        print(f"  !! DANGEROUS, ROUTED  : {r.dangerous_routed}")
    if r.dangerous_reported:
        print(f"  !  dangerous, reported: {r.dangerous_reported}")
    if r.prose_claims:
        print(f"  !  in free text       : {r.prose_claims}")
    if not (r.dangerous_routed or r.dangerous_reported or r.prose_claims):
        print("  no dangerous claim")
    print(f"  safe seen: {r.safe_hits or '[]'}")
    print(f"  safe missed: {r.safe_missed or '[]'}")


def report(results: list[ImageResult]) -> int:
    """Print the totals. Returns the process exit code."""
    routed = sum(len(r.dangerous_routed) for r in results)
    reported = sum(len(r.dangerous_reported) for r in results)
    prose = sum(len(r.prose_claims) for r in results)
    hits = sum(len(r.safe_hits) for r in results)
    possible = hits + sum(len(r.safe_missed) for r in results)
    errors = sum(1 for r in results if r.error)
    truncated = sum(1 for r in results if r.truncated)
    seconds = sum(r.seconds for r in results)

    print(f"\n{'=' * 72}\nVISION EVALUATION -- {len(results)} images"
          f"\n{'=' * 72}")
    print(f"  DANGEROUS FALSE POSITIVES (routed)   : {routed}   <- the gate")
    print(f"  dangerous claims (reported only)     : {reported}")
    print(f"  dangerous claims in free text        : {prose}")
    print(f"  safe observations recovered          : {hits}/{possible}"
          + (f" ({100 * hits / possible:.0f}%)" if possible else ""))
    print(f"  perception errors                    : {errors}")
    print(f"  truncated responses                  : {truncated}")
    print(f"  total model time                     : {seconds:.0f}s"
          + (f" ({seconds / len(results):.0f}s per image)" if results else ""))
    print("\n  Coverage is secondary by design: an unrecognised wall becomes an"
          "\n  ask-back, which is the system working. A dangerous claim that"
          "\n  routed is a failure whatever the coverage figure says.")
    return 1 if routed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", dest="images",
                        help="run one fixture by name; repeatable")
    parser.add_argument("--model", default="",
                        help="vision model tag (default: VISION_MODEL)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the results as JSON")
    args = parser.parse_args(argv)

    results = run(args.images, args.model)
    code = report(results)
    if args.json:
        args.json.write_text(json.dumps(
            [r.__dict__ for r in results], indent=2, default=str) + "\n",
            encoding="utf-8")
        print(f"\n  written to {args.json}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
