"""Answering an ask-back has to be worth doing more than once.

Decision 10 makes substrate load-bearing: asked for a plaster for "my wall"
with no substrate stated, the system asks which wall rather than guessing. The
session layer exists so the reply can be answered in place — the person types
"brick", and the question they actually asked gets answered.

That worked exactly once. The value they typed was detected from their reply,
merged into a **local** dict to re-ask the pending question, and then dropped:
`assistant/ui.py`'s `_remember` re-read the session store for the
auto-answered branch, and the store had never been told. So turn two was right
and turn three asked for the substrate again, and again, and the feature
defeated itself in the least visible way available — by working the first time.

The second bug in this file is the other half of the same seam and is worse.
`is_answer_to_askback` classed any short message carrying any detected slot as
an answer, so **"Can I use Ultra on the same wall?"** — nine words, and `detect`
finds `property_asked=compatibility` in it — was treated as a reply to the
pending question. The person's actual question was discarded and the earlier
one re-answered in its place. A false positive here does not cost an ask-back;
it silently loses the question.

The harness is `tests/test_ui_server.py`'s, because this behaviour lives in the
orchestration between the engine and the session store and is not reachable
through either alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.router import SlotDetector                          # noqa: E402

from test_ui_server import (                                       # noqa: E402
    ASK, Browser, chat,                                            # noqa: F401
)


# ----------------------------------------------- the value has to be kept


def test_the_answer_to_an_ask_back_is_remembered_beyond_the_turn(chat):
    """The regression this file exists for.

    Turn two was always right. Turn three is the one that was broken, and it is
    the one a real conversation always reaches.
    """
    browser = Browser(chat)

    first = browser.json(ASK)
    assert first["parts"][0]["path"] == "ask back", first["parts"]

    browser.json("brick")

    third = browser.json("What thickness should it be?")
    assert third["session_slots"].get("substrate") == "brick", (
        "the substrate was used to answer the pending question and then dropped")


def test_answering_an_ask_back_does_not_ask_the_same_thing_again(chat):
    """The behavioural form of the same fact, stated the way a user would."""
    browser = Browser(chat)
    browser.json(ASK)
    browser.json("brick")

    again = browser.json(ASK)

    paths = [part["path"] for part in again["parts"]]
    assert "ask back" not in paths, "asked for a substrate already given"


def test_the_pending_question_is_answered_not_the_bare_word(chat):
    """"brick" is an answer, so the reply is about plaster, not about brick."""
    browser = Browser(chat)
    browser.json(ASK)

    second = browser.json("brick")

    assert second["answered"] == ASK, (
        "the reply answered the word rather than the question it was an answer to")


def test_the_answered_substrate_is_attributed_to_the_person(chat):
    """It must never be presented as something the system decided.

    ``CARRIED`` is what arrives here, and it is correct rather than a
    near-miss: the question being answered is the *pending* one from the turn
    before, and relative to that question "brick" genuinely was said earlier in
    the conversation. ``STATED`` would read equally true. The distinction that
    actually matters is the one asserted below.

    ``ASSUMED`` would print "substrate: brick -- assumed, since you did not
    say" at somebody who had just been asked for it and had just answered, and
    ``OBSERVED`` would credit a photograph that was never sent. Either would
    invite them to correct a fact they had personally supplied one message ago,
    which is the failure decision 10's ask-back exists to end rather than to
    restart.
    """
    browser = Browser(chat)
    browser.json(ASK)

    second = browser.json("brick")

    facts = {f["slot"]: f["provenance"]
             for part in second["parts"] for f in part.get("facts", [])}
    assert facts.get("substrate") in ("stated", "carried"), (
        f"the substrate was not attributed to the caller: {facts}")


def test_an_answered_question_does_not_resume_on_a_later_turn(chat):
    """An answered ask-back must not re-fire and hijack a third question.

    `pending` is not in the JSON payload, so this asserts the consequence the
    payload does expose: turn three is answered as itself.
    """
    browser = Browser(chat)
    browser.json(ASK)
    browser.json("brick")

    third = browser.json("What thickness should it be?")

    assert third["answered"] == "What thickness should it be?"


def test_a_correction_after_an_ask_back_still_wins(chat):
    """The ask-back path must not become a way to pin a slot."""
    browser = Browser(chat)
    browser.json(ASK)
    browser.json("brick")

    corrected = browser.json("actually it is stone")

    assert corrected["session_slots"].get("substrate") == "stone"


# ------------------------------------ a question is a question, however short


def test_a_short_follow_up_question_is_not_swallowed_as_an_ask_back_answer(chat):
    """The expensive false positive: the real question disappears.

    Before the fix this returned the answer to `ASK` and the person never found
    out that what they asked had not been read.
    """
    browser = Browser(chat)
    browser.json(ASK)

    follow_up = browser.json("Can I use Ultra on the same wall?")

    assert follow_up["answered"] == "Can I use Ultra on the same wall?", (
        "a new question was discarded and the pending one re-answered")


def test_only_a_building_fact_counts_as_an_answer_to_an_ask_back():
    """`property_asked`, `calculation` and `symptom` describe a question.

    An ask-back is only ever raised for a building fact, so only a building
    fact can answer one. Tested directly on the detector because this is the
    predicate, and the orchestration above is its consequence.
    """
    detector = SlotDetector()

    assert detector.is_answer_to_askback("brick")
    assert detector.is_answer_to_askback("outside")
    assert detector.is_answer_to_askback("exposed, north-facing")
    assert detector.is_answer_to_askback("brick, outside")

    assert not detector.is_answer_to_askback("Can I use Ultra on the same wall?")
    assert not detector.is_answer_to_askback("What about Forte?")
    assert not detector.is_answer_to_askback("How many bags?")
    assert not detector.is_answer_to_askback("yes")


def test_a_question_mark_makes_it_a_question_even_with_a_substrate_in_it():
    """"Is it brick?" is asking us, not telling us."""
    detector = SlotDetector()

    assert not detector.is_answer_to_askback("is it brick?")
    assert detector.is_answer_to_askback("it is brick")


def test_the_length_bound_still_holds():
    """Unchanged behaviour, pinned so the new predicate did not replace it."""
    detector = SlotDetector()

    assert detector.is_answer_to_askback(
        "brick on the outside of my house in the north")
    assert not detector.is_answer_to_askback(
        "the wall is brick on the outside of my house in the north")
