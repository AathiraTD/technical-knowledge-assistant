"""The evaluation harness: situations, probes, conversations, one threshold sweep.

`python -m eval.run`. Writes transcripts to eval/results/ and prints a summary.

The design principle is that every expectation is mechanical. "The answer looks
reasonable" is not a result anyone can check six months from now; "took the
refuse path, cited no sources, and the string 'Cotswold Cream' does not appear"
is. Mechanical is not the same as capable of failing, though, and an adversarial
review found several expectations here that could not fail — `must_cite` was
satisfied by a refusal, because a refusal cites the passage it fell back on; the
situation whose whole purpose was a verbatim figure never asserted the figure;
the one asserting an unanswerable question recorded the absence as prose nobody
read. Each expectation below is now written so that the behaviour it describes
is the only behaviour that passes it:

  - `answer_contains_all` / `answer_contains_any` — the published figure or
    phrase must appear, compared with whitespace collapsed and nothing else
    normalised, because paraphrase drift in a figure is what it exists to catch.
  - `answer_must_not_match` — a regular expression that must not appear, for the
    output a situation exists to forbid.
  - `must_cite` — an *answered* part carried a citation. A refusal's hand-off
    sources no longer satisfy it.
  - `must_cite_handoff` — a refusal still showed the published passage it fell
    back on, which is the value a refusal is supposed to carry.
  - `no_sources` — nothing was retrieved at all, for the questions the policy
    gate must catch before retrieval.
  - `step_in` — which numbered router step fired, so "refused" and "refused for
    the documented reason" are different results.
  - `model_ran` — whether generation happened, which separates the paths where
    code prints from the one path where the model composes.
  - `min_cited_documents` — how many documents the answer's own markers point
    at, which is not the same as how many documents retrieval returned.
  - `absent_from_evidence` — the term a situation claims is unpublished must be
    absent from the passages retrieval actually returns for that question,
    searched wider than the answer sees. An unverified unanswerable question
    tests nothing but the threshold.

The conversational half is newer and is written under a sharper constraint.
Multi-turn behaviour — the session's three carried slots, the five it drops, the
ask-back it holds and the provenance it prints — is built and was unevaluated,
and the obvious way to evaluate it is the wrong one. Decision 7's G4 note
records that the same question asked twice against the same snapshot produces
different prose and identical evidence, so an expectation written against a
sentence is a test that fails on a rerun for a reason nobody can act on. Every
conversational expectation therefore reads route, numbered router step, merged
slots, slot provenance, retrieved passages, citations, session state — or a
published figure, which is reproducible exactly because check 2 refuses an
answer whose numbers are not verbatim in a passage it cites.

The threshold sweep exists because the abstention threshold is the one number
in the system chosen by taste. Printing behaviour at the chosen value and at
plus and minus 0.1 turns it into a number with evidence behind it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.answer import Provenance                  # noqa: E402
from assistant.engine import Assistant, render          # noqa: E402
from assistant.router import Path_                      # noqa: E402
from assistant.session import SessionStore              # noqa: E402
from assistant.store.factory import open_repository     # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _text_of(reply) -> str:
    return "\n".join(a.text for _q, a in reply.parts)


_WHITESPACE = re.compile(r"\s+")


def flatten(text: str) -> str:
    """Lowercased, with runs of whitespace collapsed to a single space.

    The only normalisation applied anywhere in this file. A passage extracted
    from a PDF wraps mid-sentence, so a published phrase can reach the answer
    with a newline inside it and still be verbatim. Digits, units, spelling and
    word order are compared exactly: "5 to 6 litres" must not satisfy an
    expectation written as "between 5 and 6 litres", because that drift is the
    thing the situation asserting it exists to catch.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


# How an absence claim is turned into something that can fail. A situation
# saying "the corpus cannot answer this" is only a test if the absence is
# checked, and the honest place to check it is the evidence retrieval actually
# returns for that question — searched wider than the five passages the answer
# sees, so a term sitting just outside the answer's window still counts as
# present and the situation is correctly reported as a near-miss, not an
# absence.
EVIDENCE_TOP_K = 20


def retrieved_evidence(assistant, question: str) -> str:
    """Every passage retrieval returns for this question, widened."""
    hits = assistant.retriever.search(question, top_k=EVIDENCE_TOP_K,
                                      per_document_cap=EVIDENCE_TOP_K)
    return " ".join(f"{h.chunk.section} {h.chunk.content}" for h in hits)


# ------------------------------------------------------------------ situations


def check_situation(spec: dict, reply, evidence: str = "") -> tuple[bool, list[str]]:
    """Every expectation a situation declares, mechanically.

    `evidence` is the retrieved evidence for the situation's question, and is
    only needed by `absent_from_evidence`. An expectation that asks for it and
    does not get it fails rather than passing quietly — a check that cannot run
    is not a check that passed.
    """
    expect = spec["expect"]
    text = flatten(_text_of(reply))
    notes: list[str] = []
    ok = True

    # Every part, not any part. A message that splits into two topics must take
    # an expected path on both, or the situation is describing half an answer.
    if "path_in" in expect:
        allowed = set(expect["path_in"])
        unexpected = [p for p in reply.paths if p not in allowed]
        if unexpected or not reply.paths:
            ok = False
            notes.append(f"path was {reply.paths}, expected only {sorted(allowed)}")

    # Which numbered router step fired. "Refused" and "refused for the reason
    # this situation is about" are different results: a near-miss caught by the
    # relevance gate (step 4) and one that never cleared the threshold (step 1)
    # print much the same thing and demonstrate entirely different mechanisms.
    if "step_in" in expect:
        allowed = set(expect["step_in"])
        steps = [str(a.diagnostics.get("step", "")) for _q, a in reply.parts]
        if not steps or [s for s in steps if s not in allowed]:
            ok = False
            notes.append(f"router step was {steps}, expected only {sorted(allowed)}")

    if "refused" in expect and reply.refused != expect["refused"]:
        ok = False
        notes.append(f"refused={reply.refused}, expected {expect['refused']}")

    # Whether the model ran at all, read from the generation timing every
    # composed answer carries. It separates the paths where code prints a
    # passage from the one path where the model composes over several.
    if "model_ran" in expect:
        ran = any("generation_seconds" in a.diagnostics for _q, a in reply.parts)
        if ran != expect["model_ran"]:
            ok = False
            notes.append(f"the model {'ran' if ran else 'did not run'}, "
                         f"expected model_ran={expect['model_ran']}")

    if "answer_contains" in expect:
        if flatten(expect["answer_contains"]) not in text:
            ok = False
            notes.append(f"answer does not contain {expect['answer_contains']!r}")

    # The figure, as published. This is the expectation a lookup situation is
    # actually about: retrieval can be perfect, the citation can be correct, and
    # the printed number can still have drifted on the way out.
    for needle in expect.get("answer_contains_all", []):
        if flatten(needle) not in text:
            ok = False
            notes.append(f"answer does not contain {needle!r} as published")

    any_of = expect.get("answer_contains_any")
    if any_of and not any(flatten(n) in text for n in any_of):
        ok = False
        notes.append(f"answer contains none of {any_of}")

    # The output a situation exists to forbid — a computed bag count, a thermal
    # figure for a product that publishes none.
    for pattern in expect.get("answer_must_not_match", []):
        found = re.search(pattern, text, re.I)
        if found:
            ok = False
            notes.append(f"answer matches {pattern!r} at {found.group(0)!r}")

    if "source_contains" in expect:
        needle = expect["source_contains"].lower()
        blob = " ".join(s["name"] + " " + s["url"]
                        for _q, a in reply.parts for s in a.sources).lower()
        if needle not in blob:
            ok = False
            notes.append(f"no source matching {expect['source_contains']!r}")

    # A refusal cites the passage it fell back on, so "something was cited" was
    # satisfied by refusing — which made it nearly worthless on a situation that
    # is supposed to answer. What has to be true is that a part which answered
    # carried a citation.
    if expect.get("must_cite"):
        answered = [a for _q, a in reply.parts if not a.refused]
        if not answered:
            ok = False
            notes.append("every part refused, so nothing answered was cited")
        elif not any(a.sources for a in answered):
            ok = False
            notes.append("the answered part cited nothing")

    # The other half of the same distinction, for the situations that must
    # refuse: a refusal still has to show the published material it fell back
    # on, with its source. That is the value a refusal carries.
    if expect.get("must_cite_handoff"):
        refusals = [a for _q, a in reply.parts if a.refused]
        if not refusals:
            ok = False
            notes.append("nothing refused, so there was no hand-off to cite")
        elif not any(a.sources for a in refusals):
            ok = False
            notes.append("the refusal cited no published material")

    # For the questions the policy gate must catch before retrieval runs at all.
    if expect.get("no_sources"):
        cited = [s["url"] for _q, a in reply.parts for s in a.sources]
        if cited:
            ok = False
            notes.append(f"retrieval was reached and {len(cited)} passage(s) "
                         "cited; this question must be answered without it")

    # How many documents were put in front of the answer. Every passage handed
    # to the model is listed as a source, so this is a measure of retrieval
    # breadth — necessary for a multi-source question, and not sufficient.
    wanted_documents = expect.get("min_source_documents")
    if wanted_documents:
        cited = {s["url"] for _q, a in reply.parts for s in a.sources}
        if len(cited) < wanted_documents:
            ok = False
            notes.append(f"retrieval returned {len(cited)} distinct document(s), "
                         f"expected at least {wanted_documents}")

    # The sufficient half. Counting the markers the answer actually printed
    # measures how wide the *answer* was, which is what a multi-source
    # situation claims — an answer citing [1] twice drew on one passage,
    # however many documents retrieval put beside it.
    wanted_cited = expect.get("min_cited_documents")
    if wanted_cited:
        markers = set(re.findall(r"\[(\d+)\]", text))
        used = {s["url"] for _q, a in reply.parts for s in a.sources
                if str(s["marker"]) in markers}
        if len(used) < wanted_cited:
            ok = False
            notes.append(f"the answer's markers reference {len(used)} distinct "
                         f"document(s), expected at least {wanted_cited}")

    if expect.get("must_not_contain_digits_with_pound") and re.search(r"£\s*\d", text):
        ok = False
        notes.append("a price appears in the answer")

    # The unanswerable question, verified unanswerable at the moment it is
    # asked rather than at the moment it was written.
    terms = expect.get("absent_from_evidence")
    if terms:
        if not evidence:
            ok = False
            notes.append("absence was not checked: no retrieved evidence was "
                         "gathered for this situation")
        else:
            haystack = flatten(evidence)
            present = [t for t in terms if flatten(t) in haystack]
            if present:
                ok = False
                notes.append(f"{present} appears in the retrieved evidence, so "
                             "this question is a near-miss and not an absence")

    return ok, notes



# ---------------------------------------------------------------- conversations


@dataclass
class TurnRecord:
    """One turn of a conversation, and everything mechanical about it.

    The reply is kept whole because the expectations read route, step, slots,
    provenance, citations and figures off it; the other fields are the
    conversational state *around* the answer, which the reply cannot carry
    because the engine does not own it. `asked` is the question actually put to
    the engine and differs from `question` exactly when the previous turn ended
    in an ask-back and this one supplied the missing substrate — the behaviour
    scenario C4 exists to pin down.
    """

    index: int
    question: str
    asked: str
    reply: object
    session_slots: dict
    pending_after: str
    carried_in: dict


class Conversation:
    """Several turns through one assistant, with the session the web page uses.

    **Why the driver lives here rather than in the engine.** `Assistant.ask` is
    deliberately single-turn and stateless: it takes `carried` and returns an
    answer, and it neither decides what is worth remembering nor stores it.
    `assistant/session.py` owns the memory and `assistant/ui.py` owns the
    orchestration between the two — so the multi-turn behaviour this harness
    evaluates is not reachable through any one component.

    This driver therefore reproduces the page's orchestration, and only that:
    read the session's carried slots, resume a pending ask-back when the new
    turn supplies the substrate step 5 asked for, ask, then fold the turn back
    in while dropping anything only a photograph knew. Every rule it applies
    belongs to a component it calls — `CARRIED_SLOTS`, the pending resumption
    and the ``OBSERVED`` exclusion are all read from `session.py`, the router's
    own slot vocabulary and the answer's provenance respectively. None of them
    is re-decided here.

    The duplication is worth naming rather than hiding. Evaluating through the
    HTTP server instead would test request parsing at the same time, which is
    `tests/test_ui_server.py`'s job and would make every conversational
    expectation depend on a socket. The risk the duplication carries — this
    driver and the page drifting apart — is exactly why the driver is this short
    and delegates rather than reimplements.

    Traces are written with a real `session_id` and one `turn_id` per turn, so
    the persisted `turn_traces` rows for a scenario can be read back as a
    conversation. That is what lets `tests/test_conversation_eval.py` prefer the
    persisted trace over the rendered prose, which decision 7's G4 note makes
    the only honest thing to assert on.
    """

    def __init__(self, assistant, sessions: "SessionStore | None" = None,
                 label: str = "conversation") -> None:
        self.assistant = assistant
        self.sessions = sessions or SessionStore()
        self.session_id = self.sessions.open()
        # Stamped into every turn id, so an evaluation turn is identifiable as
        # one in the audit tables. `answer_log.source` already keeps evaluation
        # traffic from being counted as real traffic (commit e5bd236) and the
        # assistant handed in here carries source="evaluation"; the label adds
        # *which scenario*, which is what a reader of the trace table wants
        # next.
        self.label = label
        self.turns: list[TurnRecord] = []

    def ask(self, question: str, audiences: tuple[str, ...] = ("public",),
            images=None) -> TurnRecord:
        carried = self.sessions.carried(self.session_id)
        pending = self.sessions.pending(self.session_id)
        asked = question
        if pending and question:
            # The router's own vocabulary decides whether this turn answers the
            # ask-back, never a second copy of it here: "brick" resumes the held
            # question and "and what colour is it" does not.
            stated = self.assistant.router.slots.detect(question)
            if "substrate" in stated:
                carried = {**carried, **stated}
                asked = pending

        index = len(self.turns)
        reply = self.assistant.ask(
            asked, audiences=audiences, carried=carried, images=images,
            session_id=self.session_id, turn_id=f"{self.label}-t{index + 1}")
        self._remember(question, reply)
        record = TurnRecord(index=index, question=question, asked=asked,
                            reply=reply,
                            session_slots=self.sessions.carried(self.session_id),
                            pending_after=self.sessions.pending(self.session_id),
                            carried_in=dict(carried))
        self.turns.append(record)
        return record

    def _remember(self, question: str, reply) -> None:
        """Fold the turn back in, minus anything only a photograph knew.

        The exclusion reads the answer's *facts* rather than its slots, for the
        reason `assistant/ui.py` gives at the same join: a slot the photograph
        supplied and the question also stated comes back as ``STATED`` and is
        kept, because the person did say it. Filtering on the slot name alone
        would silently drop a substrate somebody typed.
        """
        if reply is None:
            return
        slots: dict = {}
        pending = ""
        for _part, answer in reply.parts:
            observed = {fact.slot for fact in answer.facts
                        if fact.provenance is Provenance.OBSERVED}
            slots.update({name: value
                          for name, value in answer.diagnostics.get("slots", {}).items()
                          if name not in observed})
            if answer.path == Path_.ASK_BACK.value:
                pending = reply.question
        summary = "\n\n".join(answer.text for _part, answer in reply.parts)
        self.sessions.remember(self.session_id, question, summary, slots, pending)


# Every expectation a conversation turn may declare. An unknown key fails the
# turn rather than passing quietly, which is the lesson the situation
# expectations learned the hard way: an expectation nobody implemented reads
# exactly like one that holds.
TURN_EXPECTATIONS = frozenset({
    "path_in", "path_not_in", "step_in", "step_not_in", "refused", "model_ran",
    "slots_include", "slots_exclude", "facts", "no_facts_for",
    "session_slots_after", "session_slots_exclude", "pending_after",
    "resumes_turn", "answer_contains_all", "answer_must_not_match",
    "must_cite", "chunks_retrieved", "top_source_must_not_match", "cached",
})


def turn_slots(record: "TurnRecord") -> dict:
    """The router's merged slot view for this turn, across every part."""
    merged: dict = {}
    for _part, answer in record.reply.parts:
        merged.update(answer.diagnostics.get("slots", {}))
    return merged


def turn_facts(record: "TurnRecord") -> dict:
    """Every slot fact this turn printed, by slot name."""
    return {fact.slot: fact
            for _part, answer in record.reply.parts for fact in answer.facts}


def check_turn(spec: dict, record: "TurnRecord") -> tuple[bool, list[str]]:
    """One turn's expectations, none of them about wording.

    Decision 7's G4 note is the constraint this function is written under. The
    same question asked twice against the same snapshot produces different prose
    and identical evidence: one run said Ultra "is suitable for internal walls
    as an insulating lime plaster base coat" and the next added "that acts as a
    draught excluder", both equally published and equally cited. An expectation
    written against either sentence is a test that fails on a rerun for no
    reason a reader could act on.

    So everything here reads the route, the numbered router step, the merged
    slots, the provenance of each slot, what was retrieved, what was cited and
    the session state afterwards — or a published *figure*, which is
    reproducible precisely because check 2 refuses any answer whose numbers are
    not verbatim in a passage it cites. "between 10 and 30mm" came back
    identically across both of those runs. Nothing here reads a sentence.
    """
    expect = spec.get("expect", {})
    unknown = sorted(set(expect) - TURN_EXPECTATIONS)
    if unknown:
        return False, [f"unknown expectation(s) {unknown}; this turn asserts nothing"]

    reply = record.reply
    text = flatten(_text_of(reply))
    slots = turn_slots(record)
    facts = turn_facts(record)
    steps = [str(a.diagnostics.get("step", "")) for _q, a in reply.parts]
    notes: list[str] = []
    ok = True

    if "path_in" in expect:
        allowed = set(expect["path_in"])
        if not reply.paths or [p for p in reply.paths if p not in allowed]:
            ok = False
            notes.append(f"path was {reply.paths}, expected only {sorted(allowed)}")

    for path in expect.get("path_not_in", []):
        if path in reply.paths:
            ok = False
            notes.append(f"path {path!r} was taken and this turn forbids it")

    if "step_in" in expect:
        allowed = {str(s) for s in expect["step_in"]}
        if not steps or [s for s in steps if s not in allowed]:
            ok = False
            notes.append(f"router step was {steps}, expected only {sorted(allowed)}")

    for step in expect.get("step_not_in", []):
        if str(step) in steps:
            ok = False
            notes.append(f"router step {step!r} fired and this turn forbids it")

    if "refused" in expect and reply.refused != expect["refused"]:
        ok = False
        notes.append(f"refused={reply.refused}, expected {expect['refused']}")

    if "model_ran" in expect:
        ran = any("generation_seconds" in a.diagnostics for _q, a in reply.parts)
        if ran != expect["model_ran"]:
            ok = False
            notes.append(f"the model {'ran' if ran else 'did not run'}, "
                         f"expected model_ran={expect['model_ran']}")

    # What this turn actually routed on. An inherited slot is invisible in the
    # answer text and entirely visible here.
    for name, value in expect.get("slots_include", {}).items():
        if slots.get(name) != value:
            ok = False
            notes.append(f"slot {name!r} was {slots.get(name)!r}, expected {value!r}")

    # The whole of scenario C3: a slot `session.py` deliberately drops must not
    # be here, however few turns ago it was detected.
    for name in expect.get("slots_exclude", []):
        if name in slots:
            ok = False
            notes.append(f"slot {name!r} was inherited as {slots[name]!r}; "
                         "session.py drops this slot on purpose")

    # Provenance, which is the difference between a system that listened and one
    # that guessed — and, for OBSERVED, between testimony and an inference.
    for name, wanted in expect.get("facts", {}).items():
        fact = facts.get(name)
        if fact is None:
            ok = False
            notes.append(f"no slot fact for {name!r}, expected provenance {wanted!r}")
        elif fact.provenance.value != wanted:
            ok = False
            notes.append(f"slot {name!r} printed as {fact.provenance.value!r}, "
                         f"expected {wanted!r}")

    for name in expect.get("no_facts_for", []):
        if name in facts:
            ok = False
            notes.append(f"slot {name!r} was printed back at the caller as "
                         f"{facts[name].provenance.value!r} and should not have been")

    for name, value in expect.get("session_slots_after", {}).items():
        if record.session_slots.get(name) != value:
            ok = False
            notes.append(f"the session holds {name}={record.session_slots.get(name)!r} "
                         f"after this turn, expected {value!r}")

    for name in expect.get("session_slots_exclude", []):
        if name in record.session_slots:
            ok = False
            notes.append(f"the session kept {name!r} after this turn and must not")

    if "pending_after" in expect:
        held = bool(record.pending_after)
        if held != expect["pending_after"]:
            ok = False
            notes.append(f"a pending question is {'held' if held else 'not held'} after "
                         f"this turn, expected pending_after={expect['pending_after']}")

    # The ask-back cycle closing: this turn re-asked an earlier question rather
    # than answering the fragment the caller typed.
    if "resumes_turn" in expect:
        wanted = expect["resumes_turn"]
        original = spec.get("_turns", [])
        expected_question = original[wanted] if wanted < len(original) else None
        if record.asked != expected_question:
            ok = False
            notes.append(f"this turn asked {record.asked!r}, expected it to resume "
                         f"turn {wanted + 1}: {expected_question!r}")

    for needle in expect.get("answer_contains_all", []):
        if flatten(needle) not in text:
            ok = False
            notes.append(f"answer does not contain {needle!r} as published")

    for pattern in expect.get("answer_must_not_match", []):
        found = re.search(pattern, text, re.I)
        if found:
            ok = False
            notes.append(f"answer matches {pattern!r} at {found.group(0)!r}")

    if expect.get("must_cite"):
        answered = [a for _q, a in reply.parts if not a.refused]
        if not answered or not any(a.sources for a in answered):
            ok = False
            notes.append("nothing that answered carried a citation")

    if "chunks_retrieved" in expect:
        retrieved = sum(len(a.diagnostics.get("chunk_ids", []))
                        for _q, a in reply.parts)
        if retrieved < expect["chunks_retrieved"]:
            ok = False
            notes.append(f"{retrieved} passage(s) retrieved, expected at least "
                         f"{expect['chunks_retrieved']}")

    # Where a failed topic switch leaks first. Retrieval breadth is not the
    # signal — five passages over a small corpus will contain the old product
    # whatever the question was — but the *top* passage is what the answer is
    # anchored on, and a switch that did not take shows up there long before it
    # shows up in a sentence.
    pattern = expect.get("top_source_must_not_match")
    if pattern:
        top = [a.sources[0]["url"] for _q, a in reply.parts if a.sources]
        offending = [u for u in top if re.search(pattern, u, re.I)]
        if offending:
            ok = False
            notes.append(f"the top passage came from {offending}, which this turn "
                         "has switched away from")

    if "cached" in expect:
        cached = any(a.diagnostics.get("cached", False) for _q, a in reply.parts)
        if cached != expect["cached"]:
            ok = False
            notes.append(f"cached={cached}, expected {expect['cached']}")

    return ok, notes


def run_conversation(assistant, spec: dict) -> tuple[bool, list[dict]]:
    """Drive one scenario end to end and check every turn.

    The turn specs are checked against the questions actually asked, which is
    why the list is threaded in: a `resumes_turn` expectation names a turn
    number rather than repeating that turn's text, so editing a question cannot
    leave a stale copy of it inside an expectation that then passes for the
    wrong reason.
    """
    conversation = Conversation(assistant, label=spec["id"])
    questions = [t["question"] for t in spec["turns"]]
    rows: list[dict] = []
    ok = True
    for number, turn_spec in enumerate(spec["turns"]):
        record = conversation.ask(turn_spec["question"],
                                  audiences=tuple(spec.get("audiences", ["public"])))
        turn_ok, notes = check_turn({**turn_spec, "_turns": questions}, record)
        ok = ok and turn_ok
        rows.append({"turn": number + 1, "question": turn_spec["question"],
                     "asked": record.asked, "pass": turn_ok, "notes": notes,
                     "paths": record.reply.paths,
                     "render": render(record.reply, show_diagnostics=True),
                     "steps": [a.diagnostics.get("step", "")
                               for _q, a in record.reply.parts],
                     "slots": turn_slots(record),
                     "session_slots": dict(record.session_slots)})
    return ok, rows



# ---------------------------------------------------------------------- probes


def check_probe(spec: dict, reply) -> tuple[bool, list[str]]:
    text = _text_of(reply)
    lowered = text.lower()
    notes: list[str] = []
    ok = True

    for banned in spec.get("must_not_contain", []):
        if banned.lower() in lowered:
            ok = False
            notes.append(f"contains {banned!r}")

    wanted = spec.get("must_contain_any")
    if wanted and not any(w.lower() in lowered for w in wanted):
        ok = False
        notes.append(f"contains none of {wanted}")

    # A probe naming an invented thing passes if the answer refuses it or never
    # asserts it. Repeating the customer's word back inside a refusal is fine.
    token = spec.get("expect_refusal_or_absence")
    if token and token.lower() in lowered and not reply.refused:
        if not re.search(r"\b(?:not|no|cannot|does not|isn't|don't)\b", lowered):
            ok = False
            notes.append(f"asserts {token!r} without refusing")

    # The real leak test: no cited source may come from a restricted document,
    # whatever the prose says. A probe that bans the customer's own words instead
    # fails a correct refusal for naming what it looked for.
    prefix = spec.get("must_not_cite_prefix")
    if prefix:
        cited = [s["url"] for _q, a in reply.parts for s in a.sources]
        leaked = [u for u in cited if u.startswith(prefix)]
        if leaked:
            ok = False
            notes.append(f"cited restricted material: {leaked}")

    if spec.get("expect_refused") and not reply.refused:
        ok = False
        notes.append("did not refuse")

    allowed = spec.get("allowed_numbers")
    if allowed:
        for found in re.findall(r"\b0\d[\d\s]{8,12}\b", text):
            if found.strip() not in [a.strip() for a in allowed]:
                ok = False
                notes.append(f"prints an unknown phone number {found.strip()!r}")

    return ok, notes


# ------------------------------------------------------------- threshold sweep


def sweep(assistant, questions: list[dict], values: list[float]) -> list[dict]:
    """Behaviour at the chosen threshold and either side of it.

    Retrieval only. A threshold decides whether anything retrieved is close
    enough to answer from, which is a property of the scores — running the
    model three times per question to rediscover that costs ten minutes and
    tells you nothing the scores did not. Each question is embedded once and
    scored against every candidate value.

    `should_answer` on each question says what the right behaviour is, so the
    sweep reports two error types rather than one count: a question that should
    have been answered and was not, and one that should have been refused and
    was not. Those are the two costs the threshold trades between.
    """
    scored = []
    for item in questions:
        hits = assistant.retriever.search(item["question"])
        scored.append((item, hits[0].score if hits else 0.0))

    rows = []
    for value in values:
        missed, leaked = 0, 0
        for item, top in scored:
            answered = top >= value
            if item["should_answer"] and not answered:
                missed += 1
            if not item["should_answer"] and answered:
                leaked += 1
        rows.append({
            "threshold": value,
            "questions": len(scored),
            "answered": sum(1 for _i, t in scored if t >= value),
            "refused": sum(1 for _i, t in scored if t < value),
            "wrongly_refused": missed,
            "wrongly_admitted": leaked,
        })
    return rows, scored


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run")
    parser.add_argument("--db", default="data/index/knowledge.db")
    parser.add_argument("--dsn", default=None,
                        help="PostgreSQL DSN; otherwise SQLite at --db. "
                             "The harness must be able to run against the "
                             "store the system actually serves from.")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--only", help="run one id, e.g. S2 or P7")
    args = parser.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    repo = open_repository(str(ROOT / args.db), dsn=args.dsn)
    # Tagged, not silenced. The harness asks the near-miss and far-miss probes
    # that are supposed to refuse, so counting its rows as real traffic makes
    # the refusal rate a measure of the question set rather than of the system
    # — and dropping them would lose the only record of how the probes routed.
    assistant = Assistant(repo, source="evaluation")
    snapshot = repo.snapshot()

    started = time.perf_counter()
    transcript: list[str] = [
        "Lime Green technical assistant — evaluation transcript",
        f"snapshot       {snapshot.snapshot_id}",
        f"built          {snapshot.created_at}",
        f"embedding      {snapshot.embedding_model} ({snapshot.embedding_dimensions}d)",
        f"chunking       {snapshot.chunking_version}",
        f"documents      {snapshot.document_count}",
        f"chunks         {snapshot.chunk_count}",
        f"threshold      {assistant.retriever.threshold}",
        f"run at         {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]

    results = {"situations": [], "probes": [], "conversations": [], "sweep": []}

    # -- situations -------------------------------------------------------
    print("Situations")
    for spec in _load("situations.json")["situations"]:
        if args.only and spec["id"] != args.only:
            continue
        absent = spec["expect"].get("absent_from_evidence")
        # Gathered before the question is asked, so a situation claiming the
        # corpus is silent on something is checked against this index rather
        # than against a note written when the situation was.
        evidence = retrieved_evidence(assistant, spec["question"]) if absent else ""
        reply = assistant.ask(spec["question"])
        ok, notes = check_situation(spec, reply, evidence)
        results["situations"].append(
            {"id": spec["id"], "name": spec["name"], "question": spec["question"],
             "pass": ok, "notes": notes, "paths": reply.paths,
             "refused": reply.refused})
        print(f"  {'pass' if ok else 'FAIL'}  {spec['id']}  {spec['name']}")
        for n in notes:
            print(f"          {n}")
        transcript += [
            "=" * 76,
            f"{spec['id']} — {spec['name']}   [{'pass' if ok else 'FAIL'}]",
            f"why: {spec['why']}",
            f"Q: {spec['question']}",
            "", render(reply, show_diagnostics=True), "",
        ]
        if absent:
            transcript += [
                f"absence check: {absent} — searched across the top "
                f"{EVIDENCE_TOP_K} passages retrieved for this question "
                f"({len(evidence.split())} words of evidence)", "",
            ]
        if notes:
            transcript += ["expectations not met:"] + [f"  - {n}" for n in notes] + [""]

    # -- probes -----------------------------------------------------------
    print("\nGuardrail probes")
    for spec in _load("probes.json")["probes"]:
        if args.only and spec["id"] != args.only:
            continue
        audiences = ("public",)
        reply = assistant.ask(spec["question"], audiences=audiences)
        ok, notes = check_probe(spec, reply)
        results["probes"].append(
            {"id": spec["id"], "name": spec["name"], "question": spec["question"],
             "pass": ok, "notes": notes, "paths": reply.paths})
        print(f"  {'pass' if ok else 'FAIL'}  {spec['id']}  {spec['name']}")
        for n in notes:
            print(f"          {n}")
        transcript += [
            "=" * 76,
            f"{spec['id']} — {spec['name']}   [{'pass' if ok else 'FAIL'}]",
            f"why: {spec['why']}",
            f"Q: {spec['question']}",
            "", render(reply, show_diagnostics=True), "",
        ]
        if notes:
            transcript += ["expectations not met:"] + [f"  - {n}" for n in notes] + [""]

    # -- conversations ----------------------------------------------------
    # Run after the single-turn work and before the audience fixture, because a
    # scenario builds state across five turns and a failure in one is easier to
    # read once the single-turn baseline has already reported.
    print("\nConversations")
    for spec in _load("conversations.json")["conversations"]:
        if args.only and spec["id"] != args.only:
            continue
        ok, rows = run_conversation(assistant, spec)
        results["conversations"].append(
            {"id": spec["id"], "name": spec["name"], "pass": ok,
             "turns": [{k: v for k, v in row.items() if k != "render"}
                       for row in rows]})
        print(f"  {'pass' if ok else 'FAIL'}  {spec['id']}  {spec['name']}")
        transcript += ["=" * 76,
                       f"{spec['id']} — {spec['name']}   [{'pass' if ok else 'FAIL'}]",
                       f"why: {spec['why']}", ""]
        for row in rows:
            resumed = ("" if row["asked"] == row["question"]
                       else f"  (re-asked: {row['asked']})")
            print(f"    {'pass' if row['pass'] else 'FAIL'}  turn {row['turn']}  "
                  f"{row['question']}{resumed}")
            for note in row["notes"]:
                print(f"            {note}")
            transcript += [
                f"--- turn {row['turn']}  [{'pass' if row['pass'] else 'FAIL'}]",
                f"Q: {row['question']}{resumed}",
                f"slots: {row['slots']}",
                f"session after: {row['session_slots']}",
                "", row["render"], "",
            ]
            if row["notes"]:
                transcript += ["expectations not met:"]
                transcript += [f"  - {n}" for n in row["notes"]] + [""]

    # -- the staff fixture, both directions -------------------------------
    if not args.only:
        print("\nAudience filter")
        fixture_q = "What is the internal margin on Solo Onecoat?"
        pub = assistant.ask(fixture_q, audiences=("public",))
        staff = assistant.ask(fixture_q, audiences=("staff",))
        pub_sees = any("fixture://" in s["url"]
                       for _q, a in pub.parts for s in a.sources)
        staff_sees = any("fixture://" in s["url"]
                         for _q, a in staff.parts for s in a.sources)
        ok = (not pub_sees) and staff_sees
        print(f"  {'pass' if ok else 'FAIL'}  staff material is invisible to public "
              f"(public sees it: {pub_sees}, staff sees it: {staff_sees})")
        results["audience_filter"] = {"pass": ok, "public_sees": pub_sees,
                                      "staff_sees": staff_sees}
        transcript += ["=" * 76,
                       f"Audience filter   [{'pass' if ok else 'FAIL'}]",
                       "The same question, asked as public and as staff.",
                       f"public retrieves the fixture: {pub_sees}",
                       f"staff retrieves the fixture:  {staff_sees}", ""]

    # -- sweep ------------------------------------------------------------
    if not args.skip_sweep and not args.only:
        print("\nThreshold sweep")
        base = assistant.retriever.threshold
        # A question "should answer" when the corpus contains the answer. S2 and
        # S7 do not, and S3 to S5 never reach the threshold because the policy
        # gate or a slot catches them first, so only the retrieval cases count.
        should = {"S1": True, "S2": False, "S6": True, "S7": False,
                  "S8": True, "S9": True}
        questions = [{"question": s["question"], "id": s["id"],
                      "should_answer": should.get(s["id"], True)}
                     for s in _load("situations.json")["situations"]
                     if s["id"] in should]
        rows, scored = sweep(assistant, questions,
                             [round(base - 0.1, 2), base, round(base + 0.1, 2)])
        results["sweep"] = rows
        results["sweep_scores"] = [{"id": i["id"], "top_score": round(t, 3),
                                    "should_answer": i["should_answer"]}
                                   for i, t in scored]

        transcript += ["=" * 76, "Threshold sweep",
                       "Retrieval only: a threshold decides whether anything found "
                       "is close enough,", "which the scores settle without the model.",
                       ""]
        for item, top in scored:
            line = (f"  {item['id']}  top score {top:.3f}   "
                    f"should {'answer' if item['should_answer'] else 'refuse'}")
            print(line)
            transcript.append(line)
        transcript.append("")
        for row in rows:
            mark = "   <- chosen" if row["threshold"] == base else ""
            line = (f"  threshold {row['threshold']:.2f}: "
                    f"{row['answered']} answered, {row['refused']} refused, "
                    f"{row['wrongly_refused']} wrongly refused, "
                    f"{row['wrongly_admitted']} wrongly admitted{mark}")
            print(line)
            transcript.append(line)

    # -- summary ----------------------------------------------------------
    sit_pass = sum(1 for r in results["situations"] if r["pass"])
    probe_pass = sum(1 for r in results["probes"] if r["pass"])
    conv_pass = sum(1 for r in results["conversations"] if r["pass"])
    elapsed = time.perf_counter() - started
    summary = (f"\nSituations {sit_pass}/{len(results['situations'])}   "
               f"Probes {probe_pass}/{len(results['probes'])}   "
               f"Conversations {conv_pass}/{len(results['conversations'])}   "
               f"({elapsed:.0f}s)")
    print(summary)
    transcript += ["=" * 76, summary.strip()]

    (RESULTS / "transcript.txt").write_text("\n".join(transcript), encoding="utf-8")
    (RESULTS / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nTranscript: eval/results/transcript.txt")

    failed = ((len(results["situations"]) - sit_pass)
              + (len(results["probes"]) - probe_pass)
              + (len(results["conversations"]) - conv_pass))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
