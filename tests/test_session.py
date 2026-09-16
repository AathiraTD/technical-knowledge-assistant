"""Conversation state: what it carries, what it refuses to carry, and its bounds.

The store is three lines of dictionary and one lock, and every interesting
property of it is a decision rather than a mechanism. Which slots survive a turn
is the load-bearing one — decision 10 makes substrate and inside/outside the two
facts a recommendation cannot be made without, and carrying the wrong slot does
not lose an answer, it silently reroutes a later question that never asked for
it. So the drop list is tested as hard as the carry list.

The rest is what an in-process dictionary reachable by an anonymous caller has
to prove: that it is bounded, that it forgets, that two callers cannot see each
other, and that an id the caller invented names nothing.

`assistant/session.py` holds no repository and no model, so nothing here needs a
server. The end-to-end proof that an ask-back can actually be answered lives in
`tests/test_ui_server.py`, where it runs against a real one.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.session import (                                      # noqa: E402
    CARRIED_SLOTS, MAX_TURNS, TURN_TEXT_CAP, SessionStore,
)


class FakeClock:
    """A clock a test can move, because waiting half an hour is not a test."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------------ identity


def test_an_id_the_caller_invented_names_nothing_and_gets_a_fresh_one():
    """A cookie must not be a request parameter for somebody else's session."""
    store = SessionStore()

    issued = store.open("guessed-id-belonging-to-nobody")

    assert issued != "guessed-id-belonging-to-nobody"
    assert store.carried("guessed-id-belonging-to-nobody") == {}


def test_an_id_this_store_issued_is_honoured_on_the_next_request():
    store = SessionStore()
    first = store.open("")

    assert store.open(first) == first


def test_two_ids_are_never_the_same():
    """A collision would hand one caller another caller's wall."""
    store = SessionStore()

    assert len({store.open("") for _ in range(50)}) == 50


# ------------------------------------------------------- what is carried


def test_the_carried_slots_are_the_facts_about_a_building():
    """Decision 10's two load-bearing slots, plus the one of the same kind."""
    assert CARRIED_SLOTS == ("substrate", "location", "exposure")


def test_a_substrate_established_in_one_turn_is_offered_to_the_next():
    store = SessionStore()
    session = store.open("")

    store.remember(session, "brick", "an answer", {"substrate": "brick"})

    assert store.carried(session) == {"substrate": "brick"}


def test_a_later_turn_that_names_a_different_substrate_replaces_the_first():
    """"Actually it's stone" has to win, or the session is a trap."""
    store = SessionStore()
    session = store.open("")
    store.remember(session, "brick", "an answer", {"substrate": "brick"})

    store.remember(session, "actually stone", "an answer", {"substrate": "stone"})

    assert store.carried(session)["substrate"] == "stone"


def test_a_turn_that_says_nothing_about_the_wall_leaves_the_wall_alone():
    """Not mentioned again is not no longer true."""
    store = SessionStore()
    session = store.open("")
    store.remember(session, "brick", "an answer", {"substrate": "brick"})

    store.remember(session, "how long does it take to dry", "an answer",
                   {"property_asked": "drying"})

    assert store.carried(session) == {"substrate": "brick"}


def test_the_question_shaped_slots_are_dropped_rather_than_carried():
    """The slots that would reroute a later question nobody asked that way.

    A held `calculation` sends a plain lookup down router step 6; a held
    `photograph` appends "I cannot see photographs" to an answer with no image
    attached; `symptom` and `cause_asked` pin the session to ask-back and
    diagnosis respectively. Each of these is a fact about one question, not
    about the building, so none of them survives the turn.
    """
    store = SessionStore()
    session = store.open("")

    store.remember(session, "how many bags for my crazed brick wall", "an answer",
                   {"substrate": "brick", "location": "external",
                    "exposure": "severe", "calculation": "quantity",
                    "photograph": "photo", "symptom": "crack",
                    "cause_asked": "cause", "property_asked": "coverage"})

    assert store.carried(session) == {"substrate": "brick", "location": "external",
                                      "exposure": "severe"}


def test_the_carried_slots_are_a_copy_that_a_caller_cannot_write_through():
    """A handed-out dict is merged into by the request thread; it must not alias."""
    store = SessionStore()
    session = store.open("")
    store.remember(session, "brick", "an answer", {"substrate": "brick"})

    store.carried(session)["substrate"] = "stone"

    assert store.carried(session)["substrate"] == "brick"


# --------------------------------------------------------------- the pending ask


def test_an_ask_back_leaves_the_question_pending_and_an_answer_clears_it():
    store = SessionStore()
    session = store.open("")

    store.remember(session, "what plaster", "which wall?", {},
                   pending="what plaster")
    assert store.pending(session) == "what plaster"

    store.remember(session, "brick", "an answer", {"substrate": "brick"})
    assert store.pending(session) == "", "an answered question resumed later"


def test_an_unknown_session_has_no_slots_no_pending_and_no_turns():
    """Every read has to survive a cookie for a session that has expired."""
    store = SessionStore()

    assert store.carried("gone") == {}
    assert store.pending("gone") == ""
    assert store.turns("gone") == []


def test_writing_to_an_unknown_session_creates_nothing():
    """A forged cookie must not seed state for a later caller to read."""
    store = SessionStore()

    store.remember("forged", "brick", "an answer", {"substrate": "brick"})

    assert len(store) == 0
    assert store.carried("forged") == {}


# ---------------------------------------------------------------- the transcript


def test_the_conversation_is_kept_oldest_first_and_bounded():
    store = SessionStore()
    session = store.open("")

    for i in range(MAX_TURNS + 3):
        store.remember(session, f"question {i}", f"answer {i}", {})

    turns = store.turns(session)
    assert len(turns) == MAX_TURNS
    assert turns[0] == ("question 3", "answer 3"), "the oldest turns went first"
    assert turns[-1] == (f"question {MAX_TURNS + 2}", f"answer {MAX_TURNS + 2}")


def test_a_long_answer_is_clipped_so_a_session_cannot_be_inflated():
    store = SessionStore()
    session = store.open("")

    store.remember(session, "q", "x" * (TURN_TEXT_CAP * 3), {})

    assert len(store.turns(session)[0][1]) == TURN_TEXT_CAP


def test_the_turns_list_is_a_copy():
    store = SessionStore()
    session = store.open("")
    store.remember(session, "q", "a", {})

    store.turns(session).append(("injected", "injected"))

    assert len(store.turns(session)) == 1


# ------------------------------------------------------------------- the bounds


def test_the_store_refuses_to_grow_past_its_bound_oldest_first():
    """An anonymous caller can mint sessions, so unbounded is a denial of service."""
    store = SessionStore(max_sessions=3)
    first = store.open("")
    for _ in range(3):
        store.open("")

    assert len(store) == 3
    assert store.open(first) != first, "the least recently used session survived"


def test_using_a_session_keeps_it_from_being_evicted():
    """Least recently *used*, not least recently created."""
    store = SessionStore(max_sessions=3)
    first = store.open("")
    second = store.open("")
    store.open("")
    store.open(first)                       # touch it, so `second` is now oldest
    store.open("")                          # over the bound: something must go

    assert store.open(first) == first
    assert store.open(second) != second


def test_a_session_left_idle_is_forgotten():
    """A laptop left open must not answer tomorrow's question from today's wall."""
    clock = FakeClock()
    store = SessionStore(idle_seconds=60, clock=clock)
    session = store.open("")
    store.remember(session, "brick", "an answer", {"substrate": "brick"})

    clock.now += 61
    store.open("")                           # any write sweeps the expired prefix

    assert len(store) == 1
    assert store.carried(session) == {}


def test_a_session_still_inside_the_idle_period_survives_a_sweep():
    clock = FakeClock()
    store = SessionStore(idle_seconds=60, clock=clock)
    session = store.open("")
    store.remember(session, "brick", "an answer", {"substrate": "brick"})

    clock.now += 59
    store.open("")

    assert store.carried(session) == {"substrate": "brick"}


def test_a_turn_restarts_the_idle_period():
    """Idle means idle. A conversation in progress is not idle."""
    clock = FakeClock()
    store = SessionStore(idle_seconds=60, clock=clock)
    session = store.open("")

    clock.now += 50
    store.remember(session, "brick", "an answer", {"substrate": "brick"})
    clock.now += 50
    store.open("")

    assert store.carried(session) == {"substrate": "brick"}


def test_the_sweep_walks_only_the_expired_prefix():
    """Cheap by construction: it stops at the first live session.

    Ordering is what makes that correct, so the property is asserted on the
    outcome — an old session goes and a newer one stays, in one sweep.
    """
    clock = FakeClock()
    store = SessionStore(idle_seconds=60, clock=clock)
    old = store.open("")
    clock.now += 40
    newer = store.open("")

    clock.now += 30                          # old is 70s idle, newer is 30s
    store.open("")

    assert store.open(old) != old
    assert store.open(newer) == newer


# ------------------------------------------------------------------ concurrency


def test_concurrent_conversations_never_see_each_others_slots():
    """The page is a ThreadingHTTPServer, so this is the real access pattern."""
    store = SessionStore()
    sessions = [store.open("") for _ in range(12)]
    substrates = ["brick", "stone", "cob", "lath"]
    seen: dict[str, str] = {}
    barrier = threading.Barrier(len(sessions))

    def converse(index: int) -> None:
        session = sessions[index]
        mine = substrates[index % len(substrates)]
        barrier.wait()
        for _ in range(25):
            store.remember(session, "my wall", "an answer", {"substrate": mine})
            assert store.carried(session)["substrate"] == mine
        seen[session] = store.carried(session)["substrate"]

    threads = [threading.Thread(target=converse, args=(i,))
               for i in range(len(sessions))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert seen == {s: substrates[i % len(substrates)]
                    for i, s in enumerate(sessions)}


def test_concurrent_opens_all_get_distinct_sessions():
    """Two browsers arriving at once must not be handed one conversation."""
    store = SessionStore()
    issued: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(16)

    def arrive() -> None:
        barrier.wait()
        session = store.open("")
        with lock:
            issued.append(session)

    threads = [threading.Thread(target=arrive) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(set(issued)) == 16
    assert len(store) == 16
