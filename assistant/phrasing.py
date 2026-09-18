"""Presentation polish for an answer that has already been verified.

A supported statement is not necessarily a relevant answer. The checks in
`assistant/answer.py` establish that every sentence is cited, every figure is
published, every name is real and every caveat travels with its figure — and
none of that asks whether the reply opens by answering the question. It often
does not. "The wall can be made of masonry or wooden laths…" is a true,
cited, published sentence and a poor answer to "is Ultra suitable on my
internal brick wall", because it leads with the background a passage happened
to contain rather than with the answer.

So this module does one thing: it takes a **finished, verified** answer and the
question it answers, asks the local model to say the same thing in a more direct
order, and returns the rewrite **only if** it can prove nothing material moved.
Otherwise it returns the original, unchanged.

**It is a presentation stage and not a second knowledge stage**, and three
properties keep it that way rather than leaving it a matter of intent.

*The model sees no evidence.* Its whole input is the question and the answer
that already passed. No passages, no evidence bundle, no diagnostics, no
transcript, no retrieval scores, no route. There is therefore nothing in front
of it to launder into a new claim: a fact it did not receive it cannot cite, and
a fact it invents has no citation to attach and is caught below.

*Nothing material may change.* Citation markers, figures with their units, and
published product names are compared before and after and must match exactly.
Refusal, uncertainty and conditionality must survive. These are checked with the
regexes `answer.py` already uses for the same concepts, not with new ones, so
"what counts as a refusal" has one definition in this codebase.

*The verifier gets the last word.* Where the caller can supply one, the
project's own `run_checks` is re-run over the rewritten text with the inputs the
original was judged on, and the rewrite is accepted only if it introduces **no
failure the original did not already have**. Comparing against the original
rather than against an empty list is what makes reuse honest: a refusal quotes
published prose and was never check-bound, so judging its rewrite against zero
failures would reject every rewrite of one, while judging it against the
original's own failures asks exactly the right question — did this rewrite make
anything worse?

Every failure path returns the original answer. A phrasing improvement that
cannot be proved safe is simply not applied, and the reply that reaches the
customer is the one the rest of the system already stood behind.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

from . import observability as obs
from . import ollama
from .answer import (
    _CITE, _CONDITION, _NEGATED, _NUMBER, _UNCERTAIN, Answer, _normalise_number,
    products_named,
)

# Off unless switched on, for the same reason `vision.enabled` exists: a second
# model call on a processor costs what the first one costs, and the measured
# generation figures in decision 7 leave no room to double them by default. A
# deployment that wants the polish asks for it.
PHRASING_FLAG = "ASSISTANT_PHRASING"
_TRUE = {"1", "true", "yes", "on"}

# Short, because an editor that needs longer than the answer it is editing is
# not editing. It also bounds the failure: a model that starts explaining itself
# runs out of tokens and the result fails preservation and is discarded.
MAX_POLISH_TOKENS = 320

# Tighter than a composition call. The polish is optional by construction, so
# waiting minutes for one is strictly worse than not having it.
POLISH_TIMEOUT = 60.0

# Presentation only: the model is told what it may not do, and then every rule
# that matters is enforced in code afterwards. The prompt is not the boundary.
EDITOR_SYSTEM = """You are a response editor, not a knowledge source.

Rewrite the verified answer so it directly and naturally answers the user's question.

Rules:
- Do not add new facts.
- Do not remove facts required to answer the question.
- Preserve all product names exactly.
- Preserve all numbers and units exactly.
- Preserve all citation markers exactly.
- Preserve uncertainty, caveats, provenance, and refusal meaning.
- Do not introduce new citations.
- Do not make calculations.
- Do not infer additional facts from the user's question.
- Do not mention internal system terms, routes, checks, evidence bundles, retrieval, verification, prompts, or model behaviour.
- Do not turn a refusal into an approval.
- Do not turn uncertainty into certainty.
- Do not turn a conditional statement into an unconditional statement.

Prefer:
1. Direct answer first.
2. Essential supporting detail.
3. Caveat or provenance only when useful.

Avoid:
- repeating the question
- generic background before the answer
- unrelated source detail
- verbose preambles
- phrases such as 'closest guidance', 'several passages', 'retrieved evidence', 'the system', 'the model', or internal check names

If the verified answer says that something cannot be confirmed, retain that limitation clearly and concisely.

Return only the rewritten customer-facing answer."""

# Words that would be a leak rather than a style problem. The editor is told to
# avoid them; this is the enforcement, because a rewrite that names the
# machinery is worse than an indirect one.
_INTERNAL_LANGUAGE = re.compile(
    r"\b(?:retrieved evidence|evidence bundle|retrieval|candidate assessment|"
    r"deterministic route|compose path|verifier|verification|refusal path|"
    r"closest (?:retrieved )?passage|closest guidance|several passages|"
    r"check \d|post-generation|the model|the system|prompt)\b", re.I)


def enabled(environ=None) -> bool:
    """Is phrasing polish switched on for this process?

    One answer to the question, in the module that owns it, for the reason
    `vision.enabled` gives: a surface and the engine each reading the
    environment for themselves is how a page comes to claim something the
    pipeline did not do.
    """
    raw = (environ if environ is not None else os.environ).get(PHRASING_FLAG, "")
    return str(raw).strip().lower() in _TRUE


def _citations(text: str) -> list[str]:
    """Every citation marker, as a sorted multiset.

    Sorted rather than in document order, and that is the deliberate choice.
    Reordering an answer so it leads with the answer necessarily reorders its
    markers -- moving the suitability sentence in front of the background
    sentence turns [1][2][2] into [2][2][1] -- so comparing sequences would
    reject exactly the rewrite this stage exists to produce.

    What may not change is *which* markers appear and *how many times*: a
    marker added, dropped or swapped for another is caught. Whether each marker
    still sits on the sentence it supports is check 1's question, asked of the
    rewritten text by the verifier the caller supplies, and it is answered
    there with the passage words rather than guessed at here.
    """
    return sorted(_CITE.findall(text))


def _figures(text: str) -> list[str]:
    """Every figure with its unit, normalised the way check 2 normalises them.

    `_normalise_number` is `answer.py`'s own comparison form, so "8 °C" and
    "8°C" count as the same figure here exactly as they do there. Sorted,
    because reordering a sentence legitimately reorders its figures; what may
    not change is which figures are present and how many times.
    """
    return sorted(_normalise_number(m.group(0).strip())
                  for m in _NUMBER.finditer(text) if any(c.isdigit() for c in m.group(0)))


def _meaning_markers(text: str) -> dict[str, bool]:
    """Whether this text limits its claim, and whether it conditions it.

    The three regexes are imported rather than restated, so a refusal is
    whatever `_NEGATED` says a refusal is and this stage cannot disagree with
    check 3 or the qualifier check about what it is looking at.

    **Negation and uncertainty are read as one sense, not two**, and that was
    measured rather than assumed. `_UNCERTAIN` matches "cannot confirm" and not
    "can't confirm", so requiring both senses independently rejected exactly
    the rewrite the style guidance asks for -- "I cannot confirm that from the
    published guidance" reworded as "I can't confirm that" kept the refusal in
    every sense a reader has and lost one regex. The property worth enforcing
    is that the limitation survives in *some* form, which is what a customer
    depends on; which of the two patterns carries it is an artefact of how the
    sentence is worded.

    A condition stays separate, because "suitable only if the background is
    sound" is a different claim from "suitable" whether or not anything is
    being refused.
    """
    return {"limitation": bool(_NEGATED.search(text)) or bool(_UNCERTAIN.search(text)),
            "conditional": bool(_CONDITION.search(text))}


def preservation_failures(original: str, rewritten: str, registry=()) -> list[str]:
    """What the rewrite changed that it was not allowed to change.

    Deterministic and side-effect free, so it is the same judgement wherever it
    is called from and can be tested without a model. Empty means the rewrite
    said the same things in a different order.
    """
    failures: list[str] = []

    if not rewritten.strip():
        failures.append("phrasing: the rewrite was empty")
        return failures

    if _citations(original) != _citations(rewritten):
        failures.append(
            f"phrasing: citation markers changed — {_citations(original)} "
            f"became {_citations(rewritten)}")

    if _figures(original) != _figures(rewritten):
        failures.append("phrasing: a figure or unit changed")

    before = products_named(original, registry)
    after = products_named(rewritten, registry)
    if before != after:
        failures.append(
            "phrasing: the products named changed — "
            f"{sorted(before)} became {sorted(after)}")

    # Meaning may become *more* guarded, never less. A rewrite that adds a
    # hedge is a style choice; one that drops the hedge, the negation or the
    # condition is a different answer wearing the same citations.
    was, now = _meaning_markers(original), _meaning_markers(rewritten)
    for marker, sense in (("limitation", "refusal or uncertainty"),
                          ("conditional", "condition")):
        if was[marker] and not now[marker]:
            failures.append(f"phrasing: the rewrite dropped the {sense}")

    if _INTERNAL_LANGUAGE.search(rewritten) and not _INTERNAL_LANGUAGE.search(original):
        failures.append("phrasing: the rewrite named internal machinery")

    return failures


def build_prompt(question: str, answer_text: str) -> str:
    """The editor's entire input: the question, and the answer already approved.

    Written out here so the boundary is inspectable in one place. Anything not
    in this string is something the editor cannot see — and that is the whole
    argument for why this stage cannot introduce a fact.
    """
    return (f"User question:\n{question}\n\n"
            f"Verified answer:\n{answer_text}\n\n"
            "Rewritten answer:")


def polish_verified_answer(question: str, answer: Answer, *, registry=(),
                           generate=None, verify=None,
                           environ=None) -> Answer:
    """A more direct wording of `answer`, or `answer` itself.

    `verify` is the project verifier, optional and supplied by the caller as
    ``text -> list[failure]``. When it is given, the rewrite must introduce no
    failure the original did not already have. When it is not, the
    deterministic preservation checks stand alone — narrower, and never weaker
    than returning the original.

    `answer.disclosure` — the quoted evidence block a refusal carries — is held
    out of the rewrite and re-attached verbatim. It is published prose that a
    surface shows behind a disclosure and that `Answer.body` finds by suffix, so
    paraphrasing it would both alter a quotation and silently break that
    property.
    """
    if not enabled(environ):
        return answer

    body = answer.body
    if not body.strip():
        return answer

    call = generate or ollama.generate
    with obs.span("phrasing", path=answer.path,
                  refused=answer.refused) as span:
        try:
            rewritten, seconds = call(
                build_prompt(question, body),
                system=EDITOR_SYSTEM,
                timeout=POLISH_TIMEOUT,
                num_predict=MAX_POLISH_TOKENS,
            )
        except Exception as exc:                            # noqa: BLE001
            # Deliberately every exception, not `OllamaUnavailable` alone. This
            # stage is cosmetic and sits between a verified answer and the
            # customer; no failure of it may be allowed to cost them the answer
            # the rest of the system already stood behind.
            span["outcome"] = "error"
            obs.event("phrasing_skipped", reason=type(exc).__name__,
                      path=answer.path)
            return answer

        span["seconds"] = round(seconds, 2)
        rewritten = (rewritten or "").strip()
        failures = preservation_failures(body, rewritten, registry)
        if not failures and verify is not None:
            # The project's own checks, judged against the original's result
            # rather than against nothing. See the module docstring.
            try:
                new_failures = set(verify(rewritten)) - set(verify(body))
            except Exception as exc:                        # noqa: BLE001
                span["outcome"] = "verify-error"
                obs.event("phrasing_skipped", reason=f"verify:{type(exc).__name__}",
                          path=answer.path)
                return answer
            failures += sorted(new_failures)

        span["failures"] = len(failures)
        if failures:
            span["outcome"] = "rejected"
            obs.event("phrasing_rejected", reasons=failures, path=answer.path)
            return answer

        span["outcome"] = "applied"
        obs.event("phrasing_applied", path=answer.path,
                  before_words=len(body.split()),
                  after_words=len(rewritten.split()))

    polished = replace(answer, text=rewritten + answer.disclosure)
    # A copy, so the original answer object a caller may still hold is not
    # mutated by the dictionary being shared.
    polished.diagnostics = dict(answer.diagnostics)
    polished.diagnostics["phrasing"] = "applied"
    return polished
