"""Phrasing polish: what it may change, and what it must never change.

The stage exists because a supported statement is not necessarily a relevant
answer. "The wall can be made of masonry or wooden laths…" is cited, published
and true, and it is a poor reply to "would Ultra be suitable on my internal
brick wall" because it leads with background instead of the answer.

So these tests are almost all about the *guard* rather than about the prose.
The one directness test asserts an ordering property and nothing about wording,
because wording is the model's and decision 7's G4 note is explicit that this
build reproduces evidence rather than sentences. Everything else asserts that a
rewrite which moved a figure, a marker, a name or a meaning is discarded and the
verified answer is returned untouched — which is the only property a customer
depends on.

The model is a stub throughout. A real editor call would make these tests a
measurement of qwen3.5 rather than of the boundary, and the boundary is what
can break silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answering import phrasing
from assistant.answering.answer import (  # noqa: E402
    Answer,
)
from assistant.infrastructure.ollama import (  # noqa: E402
    OllamaUnavailable,
)


ON = {phrasing.PHRASING_FLAG: "1"}
REGISTRY = ["Lime Green Ultra", "Ultra", "Solo", "Duro"]

INDIRECT = ("The wall can be made of masonry or wooden laths [1]. "
            "Lime Green Ultra is suitable for most masonry backgrounds [2]. "
            "Apply at a uniform thickness of 10-30 mm [2].")

DIRECT = ("Yes. Lime Green Ultra is suitable for most masonry backgrounds [2]. "
          "Apply it at a uniform thickness of 10-30 mm [2]. "
          "Backgrounds may be masonry or wooden laths [1].")

QUESTION = "Would Lime Green Ultra be suitable on my internal brick wall?"


def answer(text: str, **extra) -> Answer:
    return Answer(text=text, path=extra.pop("path", "compose"), **extra)


def stub(reply: str, *, raises: Exception | None = None, seen: list | None = None):
    """A model that returns `reply`, recording exactly what it was handed."""
    def generate(prompt, **kwargs):
        if seen is not None:
            seen.append({"prompt": prompt, **kwargs})
        if raises is not None:
            raise raises
        return reply, 0.4
    return generate


def polish(original: Answer, reply: str, **kwargs) -> Answer:
    return phrasing.polish_verified_answer(
        QUESTION, original, registry=REGISTRY,
        generate=stub(reply, **kwargs.pop("stub", {})),
        environ=ON, **kwargs)


# ------------------------------------------------------------- 1 directness


def test_an_indirect_answer_is_reordered_to_answer_first():
    """The property is ordering, not wording.

    Asserted as "the answer precedes the background it used to open with",
    which is what the stage is for, rather than on a sentence the model chose.
    """
    polished = polish(answer(INDIRECT), DIRECT)

    assert polished.text == DIRECT
    assert polished.diagnostics["phrasing"] == "applied"
    suitability = polished.text.index("suitable for most masonry")
    background = polished.text.index("wooden laths")
    assert suitability < background, (
        "the rewrite still opens with background rather than with the answer")


# ------------------------------------------------- 2-4 material preservation


@pytest.mark.parametrize("rewrite, why", [
    ("Yes. Ultra suits masonry [3]. Apply at 10-30 mm [2].",
     "a citation marker changed"),
    ("Yes. Ultra suits masonry [2]. Apply at 10-30 mm [2]. Laths too [1][1].",
     "a citation marker was added"),
    ("Yes. Lime Green Ultra suits masonry [2]. Apply at 10-40 mm [2]. Laths [1].",
     "a figure changed"),
    ("Yes. Lime Green Ultra suits masonry [2]. Apply at 10-30 cm [2]. Laths [1].",
     "a unit changed"),
    ("Yes. Lime Green Solo suits masonry [2]. Apply at 10-30 mm [2]. Laths [1].",
     "a product name changed"),
])
def test_a_rewrite_that_moves_something_material_is_discarded(rewrite, why):
    original = answer(INDIRECT)

    polished = polish(original, rewrite)

    assert polished is original, f"{why} and the rewrite was still applied"
    assert polished.text == INDIRECT
    assert "phrasing" not in polished.diagnostics


def test_citation_markers_survive_an_accepted_rewrite_byte_for_byte():
    polished = polish(answer(INDIRECT), DIRECT)

    assert phrasing._citations(polished.text) == phrasing._citations(DIRECT)
    assert "[1]" in polished.text and "[2]" in polished.text
    assert "10-30 mm" in polished.text
    assert "Lime Green Ultra" in polished.text


# --------------------------------------------------------- 5 refusal meaning


REFUSAL = ("I cannot confirm that from the published Lime Green guidance. "
           "The Ultra documentation does not state a U-value [1].")


@pytest.mark.parametrize("rewrite", [
    "Yes. Lime Green Ultra states a U-value [1].",
    "Lime Green Ultra documentation covers this and confirms the figure [1].",
])
def test_a_refusal_cannot_be_rewritten_into_an_approval(rewrite):
    """The worst thing this stage could do, so it is tested directly.

    The markers are `answer.py`'s own `_NEGATED` and `_UNCERTAIN`, so "what
    counts as a refusal" is not re-decided here.
    """
    original = answer(REFUSAL, refused=True, path="refuse")

    polished = polish(original, rewrite)

    assert polished is original
    assert polished.text == REFUSAL


def test_a_conditional_statement_cannot_become_unconditional():
    original = answer("Lime Green Ultra is suitable only if the background is "
                      "sound [1].")

    polished = polish(original, "Lime Green Ultra is suitable [1].")

    assert polished is original


def test_a_refusal_may_be_reworded_while_it_stays_a_refusal():
    """The stage is not disabled on refusals -- a refusal is worth phrasing well."""
    better = ("I can't confirm that from the published Lime Green guidance. "
              "The Ultra documentation does not state a U-value [1].")

    polished = polish(answer(REFUSAL, refused=True, path="refuse"), better)

    assert polished.text == better
    assert polished.diagnostics["phrasing"] == "applied"


# ---------------------------------------------------------------- 6 fallback


@pytest.mark.parametrize("stub_kwargs, reply", [
    ({"raises": OllamaUnavailable("model is not loaded")}, ""),
    ({"raises": TimeoutError("read timed out")}, ""),
    ({"raises": RuntimeError("anything at all")}, ""),
    ({}, ""),
    ({}, "   \n  "),
])
def test_any_failure_returns_the_verified_answer_unchanged(stub_kwargs, reply):
    original = answer(INDIRECT)

    polished = polish(original, reply, stub=stub_kwargs)

    assert polished is original
    assert polished.text == INDIRECT


def test_a_rewrite_failing_the_project_verifier_is_discarded():
    """`run_checks`-shaped verifier, judged against the original's own result.

    The original here already carries one failure, the way a refusal quoting
    published prose does. The rewrite is rejected for the failure it *adds*,
    not for the one it inherited -- which is the difference between reusing the
    verifier and demanding that a rewrite clear a bar the original never did.
    """
    original = answer(INDIRECT)
    calls = {}

    def verify(text):
        calls[text] = calls.get(text, 0) + 1
        inherited = ["check 1: a sentence carries no citation"]
        return inherited + (["check 6: the asked-for term is absent"]
                            if text == DIRECT else [])

    polished = phrasing.polish_verified_answer(
        QUESTION, original, registry=REGISTRY, generate=stub(DIRECT),
        verify=verify, environ=ON)

    assert polished is original
    assert calls[INDIRECT] == 1, "the original must be judged on the same checks"


def test_a_rewrite_inheriting_only_the_originals_failures_is_accepted():
    original = answer(INDIRECT)

    polished = phrasing.polish_verified_answer(
        QUESTION, original, registry=REGISTRY, generate=stub(DIRECT),
        verify=lambda _text: ["check 1: a sentence carries no citation"],
        environ=ON)

    assert polished.text == DIRECT


def test_the_stage_is_off_unless_switched_on():
    original = answer(INDIRECT)

    assert phrasing.polish_verified_answer(
        QUESTION, original, registry=REGISTRY,
        generate=stub(DIRECT), environ={}) is original


# ------------------------------------------------------------ 7 no evidence


def test_the_editor_receives_only_the_question_and_the_verified_answer():
    """The argument that this cannot introduce a fact is that it sees none.

    Asserted on the prompt actually sent, because the claim is about the bytes
    the model receives and not about what the caller meant to send.
    """
    seen: list = []
    passage = "Ultra is applied over woodfibre board at 6mm and suits solid walls."
    original = answer(INDIRECT, diagnostics={
        "chunk_ids": ["solo-0", "ultra-2"], "top_score": 0.71,
        "step": "8", "reason": "several passages bore on it",
        "passages": [passage]},
        sources=[{"name": "Ultra datasheet", "url": "https://example.invalid"}])

    phrasing.polish_verified_answer(
        QUESTION, original, registry=REGISTRY,
        generate=stub(DIRECT, seen=seen), environ=ON)

    prompt = seen[0]["prompt"]
    assert prompt == phrasing.build_prompt(QUESTION, INDIRECT)
    assert QUESTION in prompt and INDIRECT in prompt
    for leak in (passage, "solo-0", "ultra-2", "0.71",
                 "several passages bore on it", "https://example.invalid"):
        assert leak not in prompt, f"the editor was handed {leak!r}"
    # The system prompt is the editor's charter and carries no evidence either.
    assert "You are a response editor" in seen[0]["system"]
    assert seen[0]["timeout"] == phrasing.POLISH_TIMEOUT


def test_a_rewrite_naming_internal_machinery_is_discarded():
    original = answer(INDIRECT)

    polished = polish(original,
                      "Yes. The retrieved evidence shows Lime Green Ultra "
                      "suits masonry [2]. Apply at 10-30 mm [2]. Laths [1].")

    assert polished is original


# ------------------------------------------------- the quoted evidence block


def test_a_refusals_quoted_disclosure_is_never_paraphrased():
    """`Answer.body` finds the disclosure by suffix, and it is a quotation.

    Held out of the rewrite and re-attached verbatim, so a surface that shows
    it behind a disclosure keeps working and the quoted datasheet text stays
    byte-identical.
    """
    disclosure = "\n\nPublished: Ultra suits masonry backgrounds [1]."
    original = answer(REFUSAL + disclosure, refused=True, path="refuse",
                      disclosure=disclosure)
    better = ("I can't confirm that from the published Lime Green guidance. "
              "The Ultra documentation does not state a U-value [1].")
    seen: list = []

    polished = phrasing.polish_verified_answer(
        QUESTION, original, registry=REGISTRY,
        generate=stub(better, seen=seen), environ=ON)

    assert disclosure not in seen[0]["prompt"], (
        "the quoted evidence block was sent to the editor")
    assert polished.text.endswith(disclosure)
    assert polished.body == better


# ------------------------------------------------- one UI-shaped end-to-end


def test_a_polished_answer_renders_through_the_page_unchanged_in_structure(
        monkeypatch):
    """The polish is presentation, so the presentation layer must not notice it.

    Deliberately one test and not an acceptance suite: what is being checked is
    that a rewritten `Answer` still renders with its path hidden, its citation
    markers intact and its sources attached -- the properties
    `tests/test_acceptance_ui_boundaries.py` owns -- rather than re-testing the
    page.
    """
    from bs4 import BeautifulSoup

    from assistant.interfaces import ui
    from assistant.answering.engine import Reply

    monkeypatch.setenv(phrasing.PHRASING_FLAG, "1")
    source = {"name": "Ultra datasheet", "url": "https://example.invalid/ultra"}
    original = answer(INDIRECT, sources=[source])

    polished = phrasing.polish_verified_answer(
        QUESTION, original, registry=REGISTRY, generate=stub(DIRECT))

    soup = BeautifulSoup(
        ui.render_html(Reply(QUESTION, [(QUESTION, polished)]), False, "abc123def456"),
        "html.parser")
    rendered = soup.select_one(".answer-text").get_text()
    assert rendered == DIRECT
    assert "[1]" in rendered and "[2]" in rendered
    assert "phrasing" not in rendered and "compose" not in rendered
    assert not soup.select(".answer-text .tag")
