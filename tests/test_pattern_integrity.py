"""No control character may hide inside a pattern.

This exists because the bug has now happened twice in this repository, in two
different files, and it is invisible in every way a reviewer normally looks.

The mechanism: `\\b` is a word boundary to a regex engine and a **backspace** to
anything that resolves escapes first. Write `\\b` into `config/routing.json`
through a tool that unescapes once, and the file still parses as valid JSON,
still renders as `\\b` in most terminals, still reviews clean in a diff — and
the pattern it belongs to now matches nothing, because it is looking for
`\\x08`. The policy gate that was supposed to keep price questions away from
retrieval silently stops firing. The same slip in `assistant/router.py` turned
the new ask-back guard's `any)\\b` into `any)\\x08`.

Nothing else in the suite catches it. A pattern that matches nothing throws no
error; it just quietly answers a question it was built to refuse. So the check
is a property of the source rather than of any one behaviour: outside a string
that means to contain them, a Python or JSON source file in this project holds
no C0 control characters but tab and newline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Tab and newline are the two that legitimately appear in source.
FORBIDDEN = {chr(c) for c in range(0x20)} - {"\t", "\n", "\r"}

SEARCHED = sorted(
    [p for p in (ROOT / "assistant").rglob("*.py") if "__pycache__" not in p.parts]
    + [p for p in (ROOT / "config").glob("*.json")]
    + [p for p in (ROOT / "eval").glob("*.json")]
)


def name(path: Path) -> str:
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("path", SEARCHED, ids=name)
def test_no_stray_control_character_in_source(path: Path):
    """The whole point: a backspace that reads as a word boundary."""
    text = path.read_text(encoding="utf-8")

    found = sorted({c for c in text if c in FORBIDDEN})

    assert not found, (
        f"{name(path)} contains control character(s) "
        f"{[hex(ord(c)) for c in found]} — almost certainly a regex escape "
        f"that was resolved one time too many. A pattern holding one matches "
        f"nothing and fails open."
    )


def test_every_policy_gate_pattern_compiles_and_can_match_something():
    """A gate topic whose patterns cannot fire is a gate that is not there.

    Compiling is not enough — `\\x08` compiles perfectly well. What is asserted
    is that each pattern contains no control character, which is the only way
    this particular failure shows up before it reaches production.
    """
    routing = json.loads((ROOT / "config" / "routing.json").read_text(encoding="utf-8"))

    for topic, spec in routing["topics"].items():
        assert spec["patterns"], f"{topic} has no patterns"
        for pattern in spec["patterns"]:
            assert not any(c in FORBIDDEN for c in pattern), (
                f"{topic}: pattern {pattern!r} holds a control character")
            re.compile(pattern, re.I)       # raises if malformed


def test_the_price_gate_still_catches_the_question_it_exists_for():
    """The regression that motivated this file, kept as a behaviour too.

    "How much is Duro?" escaped the gate once, when the pattern was narrowed to
    `how much (is|are) (it|this|the)`, and reached retrieval — where a price
    question has no honest answer.
    """
    from assistant.router import PolicyGate

    gate = PolicyGate()

    for question in ("How much is Duro?",
                     "What does Solo cost?",
                     "How much are your renders?",
                     "Can I get a discount?"):
        assert gate.match(question), f"the price gate let through {question!r}"

    # And still does not eat the questions that merely look similar.
    for question in ("How much water is needed per bag?",
                     "How much coverage does Duro give?"):
        matched = gate.match(question)
        assert matched is None or matched[0] != "price", (
            f"the price gate swallowed {question!r}")
