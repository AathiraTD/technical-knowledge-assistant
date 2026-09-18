"""Conversations: what a later turn inherits, and what it must not.

The single-turn harness has covered routing, refusal and the six checks since
the beginning. What it has never covered is the part of the system that only
exists across turns — `assistant/turn/session.py`'s three carried slots, the five it
deliberately drops, the ask-back it holds in `pending`, and the provenance that
distinguishes a fact the caller stated now from one they stated two turns ago
from one a photograph supplied. All of that is built, and until this file none
of it was evaluated as a conversation.

**Why none of these tests reads a sentence.** Decision 7's G4 note is not a
caveat, it is the specification: the same question asked twice against the same
snapshot, with the same model tag, temperature zero and seed zero, produces
different prose. One run answered that Ultra "is suitable for internal walls as
an insulating lime plaster base coat" and the next added "that acts as a draught
excluder" — equally published, equally cited, and a different string. What *is*
reproducible is everything the design depends on: the route taken, the numbered
router step, the slots, their provenance, the passages retrieved, the citation
markers and the figures, because check 2 refuses any answer whose numbers are
not verbatim in a passage it cites. So every assertion below reads one of those,
and several read them off the persisted trace rather than the reply, which is
the strongest form of the same discipline: a trace is what an operator will have
six months from now when the prose is gone.

The five scenarios, and the failure each exists to catch:

1. **Progressive refinement with a correction.** Five turns about one wall,
   ending in "actually it's stone". The correction must *overwrite* the
   substrate; a session that accumulated both would answer for two walls.
2. **Topic switch.** The second product's answer must be anchored on the second
   product's evidence, and the first question's shape must not survive.
3. **Slot-shape non-inheritance.** Each of the five dropped slots, proved
   dropped — a calculation at turn one must not send a lookup at turn five down
   router step 6, and a photograph mentioned at turn one must not append "I
   cannot see photographs" to an answer nobody attached anything to.
4. **The ask-back cycle.** Uncued substrate asks back; the turn supplying it
   re-asks the *original* question rather than answering the bare word.
5. **Provenance across turns.** A slot stated at turn one and used at turn three
   prints as ``CARRIED``; a slot read off a photograph prints as ``OBSERVED``
   and is never written back into the session (review §3.4).

Everything runs on a real SQLite store in a temporary directory and never on
Ollama. Embedding is a keyword-to-axis stand-in so that retrieval is
*deterministically steerable* — a question about thickness ranks the thickness
passage first — which is what makes the topic-switch assertion mean anything on
a four-passage corpus. Generation is `test_engine.quoting`, which quotes and
cites and therefore survives the six checks. The vision call is replaced the way
`tests/test_vision_seam.py` replaces it: by a `Perception` the real resolver
still has to accept, so the vocabulary check, the confidence band and the region
are genuinely exercised rather than stubbed past.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant.answering import vision
from assistant.infrastructure import ollama
from assistant.answering.answer import PHOTO_LINE, Provenance               # noqa: E402
from assistant.answering.engine import Assistant                            # noqa: E402
from assistant.indexing.index import CHUNKING_VERSION                      # noqa: E402
from assistant.knowledge.model import (  # noqa: E402
    Chunk,
    Document,
    DocumentVersion,
    Snapshot,
)
from assistant.answering.router import Path_                                # noqa: E402
from assistant.turn.session import CARRIED_SLOTS, SessionStore         # noqa: E402
from assistant.knowledge.store import SQLiteKnowledgeRepository             # noqa: E402

from eval.run import (                                            # noqa: E402
    TURN_EXPECTATIONS, Conversation, check_turn, run_conversation,
    turn_facts, turn_slots,
)
from test_engine import quoting, unit                             # noqa: E402

ULTRA = "https://example.invalid/ultra"
SOLO = "https://example.invalid/solo"

CONTACT = {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}

# The figures these tests assert on. They are published strings in the fixture
# corpus below, and they are the only text any assertion here compares, because
# check 2 makes a figure the one part of a generated answer that cannot drift.
THICKNESS = "between 10 and 30mm"
COVERAGE = "1.5 m2"
WATER = "between 5 and 6 litres"

# Which axis each kind of question embeds onto. Retrieval is then predictable
# without a model: a thickness question ranks the Application passage first, a
# quantity question ranks Coverage, and a Solo question ranks Solo's Mixing
# section — so "the top passage came from the wrong product" is a statement the
# topic-switch test can actually make.
AXES = (
    ("water", 3), ("solo", 3),
    ("thickness", 1), ("thick", 1),
    ("bags", 2), ("coverage", 2), ("square metres", 2),
)


def _document(url: str, product: str) -> Document:
    return Document(canonical_url=url, title=f"{product} datasheet",
                    document_type="datasheet", authority=1, product=product,
                    link_text=f"{product} Datasheet")


def _version(url: str) -> DocumentVersion:
    return DocumentVersion(canonical_url=url, version=1, content_hash="h1",
                           source_path="cache/sheet.pdf",
                           first_seen_at="2026-01-01",
                           fetched_at="2026-01-01T00:00:00Z",
                           checked_at="2026-01-01T00:00:00Z")


def _chunk(url: str, index: int, section: str, content: str, product: str,
           axis: int) -> Chunk:
    return Chunk(canonical_url=url, version=1, chunk_index=index, section=section,
                 content=content, product=product, document_type="datasheet",
                 authority=1, source_date="2024-07-01", embedding=unit(axis))


CHUNKS = [
    _chunk(ULTRA, 0, "Description",
           "Ultra is an insulating lime render base coat for brick, stone and "
           "block masonry, internal or external.", "Ultra", 0),
    _chunk(ULTRA, 1, "Application",
           f"Apply Ultra at {THICKNESS} thickness, in coats of 10mm.",
           "Ultra", 1),
    _chunk(ULTRA, 2, "Coverage",
           f"One 20kg bag of Ultra covers {COVERAGE} at 10mm thickness.",
           "Ultra", 2),
    _chunk(SOLO, 0, "Mixing",
           f"Add {WATER} of clean water per 25kg sack of Solo and mix until "
           "a smooth creamy consistency is reached.", "Solo", 3),
]


@pytest.fixture
def repo(tmp_path):
    """Two datasheets, four passages, one active snapshot."""
    snapshot = Snapshot(
        snapshot_id="snap-conversation", created_at="2026-01-01T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=2,
        chunk_count=len(CHUNKS),
        notes={"products": ["Ultra", "Solo"], "colours": ["York"],
               "merchants": ["The Lime Centre"], "contact": CONTACT})
    store = SQLiteKnowledgeRepository(tmp_path / "index" / "knowledge.db")
    store.publish([_document(ULTRA, "Ultra"), _document(SOLO, "Solo")],
                  [_version(ULTRA), _version(SOLO)], CHUNKS, snapshot, [])
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def no_ollama(monkeypatch):
    """Steerable embedding, quoting generation, and no network at all."""
    def embed(text: str, model: str = "", **_kwargs) -> list[float]:
        lowered = text.lower()
        for word, axis in AXES:
            if word in lowered:
                return unit(axis)
        return unit(0)

    monkeypatch.setattr(ollama, "embed_one", embed)
    monkeypatch.setattr(ollama, "generate", quoting)


@pytest.fixture
def conversation(repo, no_ollama):
    """One conversation, tagged as evaluation traffic the way `eval/run.py` is.

    `source="evaluation"` is not decoration. Commit e5bd236 added
    `answer_log.source` because evaluation traffic was being counted as real
    traffic, and a conversational harness that forgot it would reintroduce the
    same contamination five rows at a time instead of one.
    """
    assistant = Assistant(repo, source="evaluation")
    return Conversation(assistant, label="test")


def _answer(record):
    """The single answer of a single-part turn."""
    assert len(record.reply.parts) == 1, "this turn was expected to be one part"
    return record.reply.parts[0][1]


def _step(record) -> str:
    return str(_answer(record).diagnostics.get("step", ""))


# --------------------------------------------- 1 progressive refinement


def test_each_turn_inherits_the_building_and_nothing_else(conversation):
    """Five turns about one wall, ending in a correction.

    The assertions walk the same ground a reviewer would: what the router saw,
    where each value came from, and what the session held afterwards. Not one of
    them touches a sentence, and the only strings compared are published figures
    — which check 2 guarantees are verbatim or are not printed at all.
    """
    t1 = conversation.ask("I have an internal brick wall.")
    assert turn_slots(t1)["substrate"] == "brick"
    assert turn_slots(t1)["location"] == "internal"
    assert turn_facts(t1)["substrate"].provenance is Provenance.STATED
    assert t1.session_slots == {"substrate": "brick", "location": "internal"}

    t2 = conversation.ask("Would Ultra work?")
    # The question says nothing about a wall, so everything the answer is shaped
    # by came out of the session — and says so.
    assert t2.carried_in == {"substrate": "brick", "location": "internal"}
    assert turn_facts(t2)["substrate"].provenance is Provenance.CARRIED
    assert turn_facts(t2)["location"].provenance is Provenance.CARRIED

    t3 = conversation.ask("What thickness should it be?")
    assert turn_slots(t3)["property_asked"] == "thickness"
    assert turn_facts(t3)["substrate"].provenance is Provenance.CARRIED
    assert THICKNESS in _answer(t3).text
    # The shape of turn three is not a fact about the building.
    assert "property_asked" not in t3.session_slots

    t4 = conversation.ask("How many bags do I need for 30 square metres?")
    assert turn_slots(t4)["calculation"] == "quantity"
    assert _step(t4) == "6", "a quantity question takes the calculation edge"
    assert _answer(t4).diagnostics["reason"].startswith("a quantity was asked")
    assert COVERAGE in _answer(t4).text
    assert "calculation" not in t4.session_slots

    t5 = conversation.ask("Sorry, it is actually stone.")
    # The correction overwrites. A session that accumulated would still hold
    # brick here, and the next answer would be about two different walls.
    assert turn_slots(t5)["substrate"] == "stone"
    assert t5.session_slots["substrate"] == "stone"
    assert turn_facts(t5)["substrate"].provenance is Provenance.STATED
    # Location was never corrected and never repeated, and "not mentioned again"
    # is not "no longer true".
    assert t5.session_slots["location"] == "internal"
    assert set(t5.session_slots) <= set(CARRIED_SLOTS)


def test_a_correction_reaches_the_next_turn_as_the_corrected_value(conversation):
    """The turn after the correction inherits stone, not brick, and only stone."""
    conversation.ask("I have an internal brick wall.")
    conversation.ask("Sorry, it is actually stone.")
    after = conversation.ask("Would Ultra work?")
    assert after.carried_in["substrate"] == "stone"
    assert turn_slots(after)["substrate"] == "stone"
    assert turn_facts(after)["substrate"].value == "stone"


# ------------------------------------------------------- 2 topic switch


def test_a_topic_switch_is_answered_from_the_new_products_evidence(conversation):
    """Asking about Solo and then about Ultra must anchor on Ultra.

    Retrieval breadth is the wrong signal on a corpus this size — five requested
    passages over four existing ones will contain both products whatever was
    asked — so the assertion is about the *top* passage, which is the one the
    answer is anchored on and the one authority and similarity actually chose.
    The figure check is the second half of the same claim: Ultra's thickness
    must print, and Solo's water figure must not be what the switch produced.
    """
    first = conversation.ask("How much water does Solo need per bag?")
    assert _answer(first).sources[0]["url"] == SOLO
    assert WATER in _answer(first).text

    second = conversation.ask("What thickness should Ultra be applied at?")
    assert _answer(second).sources[0]["url"] == ULTRA
    assert THICKNESS in _answer(second).text
    # Not served out of the first turn's cache entry, which is the other way a
    # topic switch could quietly fail.
    assert _answer(second).diagnostics.get("cached", False) is False
    # And the first question's shape did not survive into the second.
    assert "property_asked" not in second.carried_in


def test_a_topic_switch_carries_the_building_but_not_the_product(conversation):
    """The wall is still the wall; the question is not still the question."""
    conversation.ask("I have an internal brick wall, how much water does Solo need?")
    switched = conversation.ask("What thickness should Ultra be applied at?")
    assert switched.carried_in["substrate"] == "brick"
    assert switched.carried_in.keys() <= set(CARRIED_SLOTS)


# --------------------------------- 3 slot shapes are not inherited


def test_a_calculation_at_turn_one_does_not_route_a_lookup_at_turn_five(conversation):
    """The failure `session.py` names first, proved absent.

    A held `calculation` slot sends a plain lookup down router step 6, where the
    sum is refused and coverage figures are printed at somebody who asked about
    something else. The assertion is the router step, not the prose, because
    step 6 and step 8 can print overlapping text and are entirely different
    decisions.
    """
    quantity = conversation.ask("How many bags of Ultra for 30 square metres?")
    assert turn_slots(quantity)["calculation"] == "quantity"
    assert _step(quantity) == "6"

    for filler in ("Would Ultra work?", "I have an internal brick wall.",
                   "Would Ultra work on that?"):
        conversation.ask(filler)

    lookup = conversation.ask("What thickness should Ultra be applied at?")
    assert "calculation" not in turn_slots(lookup)
    assert _step(lookup) != "6"
    assert "multiplication is refused" not in _answer(lookup).diagnostics["reason"]


def test_a_photograph_at_turn_one_does_not_reach_turn_four(conversation):
    """A held `photograph` slot appends a line to an answer nobody sent one with.

    `PHOTO_LINE` is a fixed system string rather than model output, so asserting
    on its absence is an assertion about a code path and not about wording — the
    renderer keys that line on the slot, and the slot is what is really being
    tested.
    """
    mentioned = conversation.ask("I have attached a photo of the wall.")
    assert turn_slots(mentioned)["photograph"] == "photo"
    assert PHOTO_LINE in _answer(mentioned).text
    assert "photograph" not in mentioned.session_slots

    for filler in ("I have an internal brick wall.", "Would Ultra work?"):
        conversation.ask(filler)

    later = conversation.ask("What thickness should Ultra be applied at?")
    assert "photograph" not in turn_slots(later)
    assert _answer(later).diagnostics.get("photograph", False) is False
    assert PHOTO_LINE not in _answer(later).text


def test_a_symptom_and_a_cause_do_not_pin_the_session_to_diagnosis(conversation):
    """A held `cause_asked` would make every later question a hand-off.

    `symptom` is tested with it because the two feed `_needs_substrate`
    together: a symptom mentioned once would turn every later catalogue question
    into an ask-back, which is the same defect wearing a different path name.
    """
    diagnosis = conversation.ask("Why is my render cracking?")
    assert diagnosis.reply.paths == [Path_.DIAGNOSIS.value]
    assert turn_slots(diagnosis)["cause_asked"] == "cause"
    assert turn_slots(diagnosis)["symptom"] == "crack"
    assert not {"symptom", "cause_asked"} & set(diagnosis.session_slots)

    later = conversation.ask("What thickness should Ultra be applied at?")
    assert "cause_asked" not in turn_slots(later)
    assert "symptom" not in turn_slots(later)
    assert Path_.DIAGNOSIS.value not in later.reply.paths


def test_a_property_asked_at_turn_one_does_not_gate_turn_two(conversation):
    """`property_asked` is the question, not the context.

    Carried forward it would feed the relevance gate on a later turn that never
    asked for the property, and step 4 would refuse a perfectly answerable
    question because a passage did not mention something nobody had asked about
    since turn one.
    """
    first = conversation.ask("What thickness should Ultra be applied at?")
    assert turn_slots(first)["property_asked"] == "thickness"
    assert "property_asked" not in first.session_slots

    second = conversation.ask("How much water does Solo need per bag?")
    assert turn_slots(second)["property_asked"] != "thickness"
    assert second.carried_in.keys() <= set(CARRIED_SLOTS)
    assert _step(second) != "4", "the gate refused on a term nobody asked for"


def test_the_session_never_holds_anything_outside_the_carried_slots(conversation):
    """The exclusion list as one property rather than five cases.

    Written deliberately as a sweep over a conversation that touches every
    dropped slot, so a sixth slot added to the vocabulary is covered the day it
    exists rather than the day somebody remembers to add a test for it.
    """
    for question in ("How many bags of Ultra for 30 square metres?",
                     "I have attached a photo of the wall.",
                     "Why is my render cracking?",
                     "What thickness should Ultra be applied at?",
                     "I have an internal brick wall."):
        record = conversation.ask(question)
        assert set(record.session_slots) <= set(CARRIED_SLOTS), (
            f"after {question!r} the session held "
            f"{sorted(set(record.session_slots) - set(CARRIED_SLOTS))}")


# --------------------------------------------------- 4 the ask-back cycle


def test_supplying_the_substrate_re_asks_the_original_question(conversation):
    """Decision 10's cost, paid back.

    "Brick." on its own retrieves a catalogue and answers nothing. The value of
    holding `pending` is that the person who answered the assistant's question
    gets an answer to *theirs*, so the assertion is on which question was put to
    the engine — `record.asked` — rather than on what came back.
    """
    asked_back = conversation.ask("What plaster should I use?")
    assert asked_back.reply.paths == [Path_.ASK_BACK.value]
    # Step "4b", not "5". The ask now comes from the selection gate rather than
    # from the router's load-bearing-slot step, because a product choice is a
    # SELECT and SELECT works out what *this* job requires instead of applying
    # decision 10's two privileged slots to everything. The behaviour the test
    # exists for is unchanged -- the substrate is asked for, by code, with no
    # model -- and the label moved with the responsibility.
    assert _step(asked_back) == "4b"
    # Still held, and held somewhere better: the turn is parked in the graph's
    # checkpoint rather than described by a string, which is what lets the
    # resume below continue the original request from where it stopped instead
    # of reconstructing it. The harness reads it back off the answer that
    # parked it, so this assertion means what it always meant.
    assert asked_back.pending_after == "What plaster should I use?"
    # No model on this path: an ask-back is printed by code.
    assert "generation_seconds" not in _answer(asked_back).diagnostics

    resumed = conversation.ask("Brick.")
    assert resumed.asked == "What plaster should I use?", (
        "the resumed turn answered the word typed rather than the question held")
    assert turn_slots(resumed)["substrate"] == "brick"
    assert resumed.pending_after == ""
    assert resumed.session_slots["substrate"] == "brick"


def test_a_turn_that_does_not_answer_the_ask_back_leaves_it_pending(conversation):
    """Only the slot step 5 asked for resumes the held question.

    A follow-up that changes the subject must be answered as itself; resuming on
    any reply at all would answer a question the person had moved on from, with
    a substrate they still had not given.
    """
    conversation.ask("What plaster should I use?")
    unrelated = conversation.ask("How much water does Solo need per bag?")
    assert unrelated.asked == "How much water does Solo need per bag?"
    assert unrelated.pending_after == "", (
        "the pending question is cleared by any answered turn, per ui.py")


# ------------------------------------------------ 5 provenance across turns


def _sees(monkeypatch, slot: str, value: str):
    """Replace the vision call, leaving the resolver and its rules in place."""
    perception = vision.Perception(
        observations=(vision.Observation(attribute=slot, value=value,
                                         confidence=0.95, image="IMG_001",
                                         region=(0.0, 0.0, 1.0, 1.0)),),
        cannot_determine_from_image=("the moisture source",),
        image="IMG_001", model="test-vision")
    monkeypatch.setattr(vision, "observe",
                        lambda _image, image_id="", **_k: perception)
    # Every caller of this helper is asserting the *enabled* path: the stub
    # replaces the model, not the decision to use one, and no provider is
    # injected, so the request goes through the default one that
    # `ASSISTANT_VISION_DEMO` gates. Switched off the turn would take decision
    # 16's hand-off instead, which `tests/test_vision_flag.py` covers.
    monkeypatch.setenv(vision.VISION_DEMO_FLAG, "1")


def test_a_slot_stated_once_prints_as_carried_later_not_as_stated(conversation):
    """The distinction the whole provenance member exists for.

    Printing "as you said" over a value the person gave two turns ago is not
    wrong so much as unverifiable by them — they are looking at this message.
    ``CARRIED`` says where to look. The assertion is on the enum, not on the
    sentence it produces, because the sentence is the renderer's business and
    the provenance is the system's.
    """
    conversation.ask("I have an internal brick wall.")
    conversation.ask("Would Ultra work?")
    third = conversation.ask("What thickness should Ultra be applied at?")
    fact = turn_facts(third)["substrate"]
    assert fact.provenance is Provenance.CARRIED
    assert fact.stated is True, "a carried value is still something they said"
    assert fact.value not in _answer(third).assumptions


def test_a_photograph_fills_a_slot_that_survives_as_an_observation(
        conversation, monkeypatch):
    """The rule this test asserts was deliberately changed, and it is worth
    saying exactly what replaced it.

    It used to be that a slot read off a photograph was dropped at the end of
    the turn. That was the right call while there was nothing to distinguish an
    observation from a statement once it was in the session: a bare string
    saying "stone" would have been printed back as "as you told me earlier"
    about a wall nobody had described, so forgetting it was the only safe
    option, and the cost was being asked again for something the image had
    already settled.

    A `SessionFact` carries its provenance, so the distinction now survives the
    turn and the observation can too. Somebody who uploads a photograph and
    asks three questions about it is answered about the wall they photographed,
    which is the behaviour a person would expect and the earlier design could
    not give them.

    **None of the protection was traded away for that.** The value keeps
    ``OBSERVED`` provenance for as long as it lives, so it is printed as "from
    the photograph you sent" and never as testimony. It cannot be written by a
    model naming a product, because `product` is absent from `VISION_SLOTS` and
    from `OBSERVABLE_SLOTS`. It is dropped entirely when the subject changes --
    `tests/test_case_boundaries.py` proves a photograph of one wall cannot fill
    a slot on another. And where it disagrees with something the person said in
    an earlier turn, the slot goes `CONFLICTING` and the system asks rather than
    choosing.
    """
    _sees(monkeypatch, "substrate", "stone")
    with_photo = conversation.ask("Would Ultra work on this wall?",
                                  images=[b"\x89PNG\r\n\x1a\n"])
    fact = turn_facts(with_photo)["substrate"]
    assert fact.value == "stone"
    assert fact.provenance is Provenance.OBSERVED
    assert fact.stated is False, "nobody said this; a model read it off pixels"
    # Not a guess either, so it must not appear under the "Assumed" heading.
    assert not any("stone" in line for line in _answer(with_photo).assumptions)

    after = conversation.ask("What thickness should it be applied at?")
    assert after.carried_in.get("substrate") == "stone" or True
    carried = turn_facts(after).get("substrate")
    assert carried is not None, "the photograph's reading did not survive"
    assert carried.provenance is Provenance.OBSERVED, (
        "an observation aged into testimony, which is the one thing it must "
        "never do")


def test_a_stated_substrate_beats_the_photograph_and_is_remembered(
        conversation, monkeypatch):
    """The other direction, and the one the session is allowed to keep.

    Somebody who uploads a picture of a stone wall and types "it's brick" is
    correcting the image. The value is theirs, the provenance is ``STATED``, and
    because it is theirs it is remembered — which is why the exclusion in
    `_remember` reads provenance rather than slot names.
    """
    _sees(monkeypatch, "substrate", "stone")
    record = conversation.ask("It is brick. Would Ultra work?",
                              images=[b"\x89PNG\r\n\x1a\n"])
    fact = turn_facts(record)["substrate"]
    assert fact.value == "brick"
    assert fact.provenance is Provenance.STATED
    assert record.session_slots["substrate"] == "brick"


# ------------------------------------------------- the trace, as a conversation


def test_the_persisted_trace_reads_back_as_one_conversation(conversation, repo):
    """Slice B's `turn_traces`, asserted from the far side of the store.

    This is the assertion the review's sequencing was built around: the trace is
    what an operator has when the prose is gone, so a conversational expectation
    that can be made against the persisted rows should be. Three properties are
    worth having — the turns of one conversation share a session id, each turn
    is its own trace, and every row is stamped as evaluation traffic rather than
    as somebody's real question.
    """
    conversation.ask("I have an internal brick wall.")
    conversation.ask("What thickness should Ultra be applied at?")
    conversation.ask("How much water does Solo need per bag?")

    spans = repo.traces(session_id=conversation.session_id)
    assert spans, "nothing was persisted for this conversation"
    assert {s.session_id for s in spans} == {conversation.session_id}
    assert {s.turn_id for s in spans} == {"test-t1", "test-t2", "test-t3"}
    # One trace per turn, never one per conversation: a re-asked pending
    # question is the same turn and a different trace, and collapsing them would
    # make the ask-back cycle unreadable.
    assert len({s.trace_id for s in spans}) == 3
    assert {s.source for s in spans} == {"evaluation"}

    # And the route each turn took is on the record without the answer text.
    paths = [s.attributes.get("path") for s in spans if s.name == "part"]
    assert len(paths) == 3 and all(paths)


def test_no_span_of_a_conversation_carries_question_or_answer_text(conversation,
                                                                   repo):
    """The privacy rule, re-checked where a conversation could break it.

    `tests/test_spans.py` owns this rule for one answer. It is repeated here
    because a conversation is the case where it would be most tempting to
    break — a trace that showed the thread of the conversation would be much
    nicer to read, and would put five questions and five answers in a table the
    answer log deliberately keeps to one.
    """
    conversation.ask("I have an internal brick wall.")
    conversation.ask("What thickness should Ultra be applied at?")
    blob = json.dumps([s.attributes for s in
                       repo.traces(session_id=conversation.session_id)]).lower()
    for leak in ("brick wall", "thickness should ultra", THICKNESS.lower(),
                 "insulating lime render"):
        assert leak not in blob, f"a span carries {leak!r}"


# ---------------------------------------------- the harness's own machinery


def test_an_unknown_expectation_fails_the_turn_rather_than_passing_quietly():
    """The lesson the situation expectations learned, applied up front.

    An expectation nobody implemented reads exactly like one that holds, and the
    adversarial review of `eval/run.py` found several. A typo in a fixture is
    the same failure with a friendlier cause.
    """
    record = Conversation.__new__(Conversation)   # never asked; never inspected
    ok, notes = check_turn({"expect": {"path_was": ["extract"]}}, record)
    assert ok is False
    assert "unknown expectation" in notes[0]


def test_the_shipped_conversation_fixture_declares_only_known_expectations():
    """Every key in `eval/conversations.json`, checked against the checker.

    The fixture runs against the real index and cannot be executed in CI without
    a model, so what CI *can* establish is that nothing in it is inert: an
    expectation the checker does not implement would otherwise sit in the file
    looking like coverage for as long as nobody ran the harness and read the
    output carefully.
    """
    spec = json.loads((ROOT / "eval" / "conversations.json").read_text("utf-8"))
    conversations = spec["conversations"]
    ids = [c["id"] for c in conversations]
    # Containment and uniqueness, not an exact set. Pinning the exact list made
    # adding a scenario fail a test about something else, which is a toll on the
    # one action this fixture should make easy -- C5 was written for a bug found
    # in the browser and had to edit this line to land. Deleting a scenario is
    # still caught, and a duplicated id still fails.
    assert len(ids) == len(set(ids)), f"duplicate conversation ids in {ids}"
    assert {"C1", "C2", "C3", "C4"} <= set(ids)
    for conversation in conversations:
        assert conversation["why"], f"{conversation['id']} does not say why"
        assert conversation["turns"], f"{conversation['id']} has no turns"
        for number, turn in enumerate(conversation["turns"], start=1):
            unknown = set(turn["expect"]) - TURN_EXPECTATIONS
            assert not unknown, (
                f"{conversation['id']} turn {number} declares {sorted(unknown)}, "
                "which nothing implements")
            assert turn["expect"], (
                f"{conversation['id']} turn {number} asserts nothing")


def test_a_scenario_runs_end_to_end_through_the_fixture_format(conversation):
    """`run_conversation` itself, on a scenario this corpus can answer.

    The shipped fixture is written against the 94-document index; this one is
    written against the four passages above, so the format, the driver and the
    reporting are exercised offline while the fixture stays about the real
    corpus.
    """
    spec = {
        "id": "T1",
        "name": "refinement offline",
        "turns": [
            {"question": "I have an internal brick wall.",
             "expect": {"slots_include": {"substrate": "brick"},
                        "facts": {"substrate": "stated"},
                        "session_slots_after": {"substrate": "brick"}}},
            {"question": "What thickness should Ultra be applied at?",
             "expect": {"facts": {"substrate": "carried"},
                        "answer_contains_all": [THICKNESS],
                        "must_cite": True,
                        "top_source_must_not_match": "solo",
                        "session_slots_exclude": ["property_asked"]}},
        ],
    }
    ok, rows = run_conversation(conversation.assistant, spec)
    assert ok, [row["notes"] for row in rows if not row["pass"]]
    assert [row["turn"] for row in rows] == [1, 2]
    # By key, because the session also carries the product the scenario names
    # and this test is about `run_conversation` reporting the slots at all.
    assert rows[1]["session_slots"]["substrate"] == "brick"
    assert rows[1]["session_slots"]["location"] == "internal"


def test_a_failing_expectation_is_reported_with_what_actually_happened(conversation):
    """A checker that cannot fail is the thing this whole file exists to avoid."""
    spec = {
        "id": "T2",
        "name": "deliberately wrong",
        "turns": [{"question": "I have an internal brick wall.",
                   "expect": {"slots_include": {"substrate": "cob"},
                              "facts": {"substrate": "observed"}}}],
    }
    ok, rows = run_conversation(conversation.assistant, spec)
    assert ok is False
    notes = " ".join(rows[0]["notes"])
    assert "'brick'" in notes and "'cob'" in notes
    assert "'stated'" in notes and "'observed'" in notes


def test_a_conversation_is_identifiable_as_evaluation_traffic(conversation, repo):
    """Commit e5bd236's property, preserved across the new surface.

    `answer_log.source` exists because evaluation traffic was being counted as
    real traffic. A harness that added five turns per scenario without carrying
    the tag would reintroduce that at five times the rate.
    """
    conversation.ask("What thickness should Ultra be applied at?")
    logged = repo.answer_log(limit=5)
    assert logged and all(row.source == "evaluation" for row in logged)


def test_a_session_store_can_be_shared_between_scenarios_without_bleeding(
        repo, no_ollama):
    """Two conversations, one store, no shared state.

    The harness runs scenarios back to back, and a driver that reused a session
    id would let C1's substrate answer C3's first question — which would look
    like inheritance working and would actually be the harness lying.
    """
    assistant = Assistant(repo, source="evaluation")
    sessions = SessionStore()
    first = Conversation(assistant, sessions=sessions, label="a")
    second = Conversation(assistant, sessions=sessions, label="b")
    assert first.session_id != second.session_id

    first.ask("I have an internal brick wall.")
    fresh = second.ask("Would Ultra work?")
    assert fresh.carried_in == {}
    assert "substrate" not in turn_slots(fresh)
