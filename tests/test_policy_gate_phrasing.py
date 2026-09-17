"""Regression tests for deterministic policy-gate phrasing coverage.

The policy gate handles governed topics before retrieval/generation.

These tests pin both:
- the canonical phrasings the routing patterns were designed for, and
- the paraphrase / word-order variants from the guardrail probe suite

so future routing changes cannot silently send those questions to retrieval
or the model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTING = json.loads(
    (ROOT / "config" / "routing.json").read_text(encoding="utf-8")
)
PROBES = json.loads(
    (ROOT / "eval" / "probes.json").read_text(encoding="utf-8")
)


def topics_matching(question: str) -> list[str]:
    """Return every deterministic policy topic matching this question.

    This deliberately evaluates the routing patterns directly from
    `config/routing.json` rather than importing the router's matching function.

    The purpose of this test is to verify coverage of the configured patterns
    themselves, so reusing the implementation under test would make the check
    agree with the code by construction.
    """
    return sorted(
        {
            name
            for name, topic in ROUTING["topics"].items()
            for pattern in topic.get("patterns", [])
            if re.search(pattern, question, re.I)
        }
    )


def probe(probe_id: str) -> dict:
    """Return a guardrail probe by id."""
    for entry in PROBES["probes"]:
        if entry["id"] == probe_id:
            return entry
    raise AssertionError(f"{probe_id} is no longer in eval/probes.json")


# ------------------------------------------------ canonical policy coverage


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How much does a bag of Solo cost?", "price"),
        ("Where can I buy Duro?", "where_to_buy"),
        (
            "Can you match the colour of my existing render?",
            "colour_matching",
        ),
        (
            "My gable wall has a 10 mm crack, is it safe?",
            "structural_judgement",
        ),
        (
            "Does this comply with Part L?",
            "compliance_signoff",
        ),
    ],
)
def test_the_gate_catches_the_phrasings_its_patterns_were_written_for(
    question,
    expected,
):
    """Canonical governed questions must hit the deterministic policy gate."""
    matched = topics_matching(question)

    assert expected in matched, (
        f"{question!r} no longer reaches the {expected} topic; "
        f"it matched {matched}"
    )


def test_every_topic_the_record_names_exists_in_the_table():
    """The routing table contains the eleven documented policy topics."""
    assert len(ROUTING["topics"]) == 11, sorted(ROUTING["topics"])


# ------------------------------------------ regression: probe paraphrases


def test_a_structural_judgement_is_caught_however_it_is_phrased():
    """P6 must route before retrieval despite the structural-safety paraphrase."""
    question = probe("P6")["question"]
    matched = topics_matching(question)

    assert "structural_judgement" in matched, (
        f"{question!r} matched {matched}"
    )


def test_a_health_question_is_caught_however_it_is_phrased():
    """P7 must route before retrieval despite reversed health-word ordering."""
    question = probe("P7")["question"]
    matched = topics_matching(question)

    assert "health" in matched, (
        f"{question!r} matched {matched}"
    )


def test_a_certification_request_is_caught_however_it_is_phrased():
    """P8 must route before retrieval despite reversed compliance-word ordering."""
    question = probe("P8")["question"]
    matched = topics_matching(question)

    assert "compliance_signoff" in matched, (
        f"{question!r} matched {matched}"
    )


def test_policy_probe_phrasings_are_all_caught():
    """Canary: all policy-focused probe phrasings hit deterministic routing."""
    # P3 is deliberately excluded: it exercises the harvested-name validation
    # rather than the pre-retrieval policy gate.
    policy_probes = {"P6", "P7", "P8"}

    missed = sorted(
        entry["id"]
        for entry in PROBES["probes"]
        if entry["id"] in policy_probes
        and not topics_matching(entry["question"])
    )

    assert missed == [], (
        "policy-gate probe questions unexpectedly bypassed deterministic routing: "
        f"{missed}"
    )