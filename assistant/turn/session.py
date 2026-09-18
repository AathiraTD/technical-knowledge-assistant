"""What a conversation remembers between turns, and what it deliberately forgets.

Decision 10 records the cost of answering every question alone: router step 5
asks "what is the wall built of?" when the substrate is uncued, and in a
single-turn tool the person's reply "brick" arrives as a brand new question
about brick. The ask-back is a dead end. This module is the state that closes
it — a session id, the facts that session has established, and the question that
is still waiting on one of them.

It is memory, not permission. Nothing about the caller's rights is stored here:
the audience set is resolved per request from what the server was started to
allow, and a session that could widen it would be an access-control bug wearing
a cookie. `carried()` returns slots and only slots, and the engine merges them
*under* whatever the current question says, so a value stated now always beats a
value remembered.

**What is worth carrying, and why the list is short.**

Carried: `product`, `substrate`, `location`, `exposure`.

- **Product**: The current subject of discussion. "I have a brick wall — should I
  use Ultra?" sets `product=Ultra`. A follow-up "How much would I need?" stays
  about Ultra. A later "what about Forte instead?" re-detects and overwrites.
- **Substrate, location, exposure**: Facts about the building itself. They do not
  expire between turns — a wall does not stop being brick because the next
  question is about drying times. Substrate and location are the two
  load-bearing slots of decision 10, which is the whole reason this file exists;
  exposure joins them because it is the same kind of fact and is only ever a
  stated assumption.

All four are printed back as stated assumptions, so all four are worth carrying.

Dropped, on purpose: `calculation`, `photograph`, `symptom`, `cause_asked`,
`property_asked`. Every one of these describes *this question's shape* rather
than the building or its active topic, and carrying a shape forward changes the
route of a later question that never asked for it. A `calculation` slot held from
two turns ago sends a plain lookup down router step 6, where the sum is refused.
A held `photograph` slot appends "I cannot see photographs" to an answer nobody
attached. `cause_asked` pins the session to the diagnosis path. The rule is:
carry what the person told us about their wall and what they are talking about,
never what the last question happened to be asking for.

Bounded and idle-expiring, because this is a process-local dictionary on a
`ThreadingHTTPServer` and an unbounded one reachable by an anonymous caller is a
denial-of-service path, not a cache. Least-recently-used out first, and anything
untouched for the idle period is dropped on the next write.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

# The facts about a building and its active topic, carried across turns.
# Substrate, location, exposure describe the wall itself; product is the current
# subject of discussion. All four are stated back as assumptions, so all four are
# worth carrying. A change of product in a follow-up ("what about Forte instead?")
# re-detects it; an unmentioned product stays the prior one.
CARRIED_SLOTS = ("product", "substrate", "location", "exposure")

# A demonstration and a day of visitors. Each session is a handful of short
# strings, so the memory is unremarkable and the bound is about refusing to grow
# without limit rather than about saving bytes.
DEFAULT_MAX_SESSIONS = 512

# Long enough that somebody can read an ask-back, go and look at their wall, and
# come back; short enough that a laptop left open does not answer tomorrow's
# question from today's substrate.
DEFAULT_IDLE_SECONDS = 30 * 60

# How much of the conversation the page shows back. Eight turns is more than any
# observed enquiry and keeps a session small enough to be uninteresting.
MAX_TURNS = 8

# A transcript line is a reminder, not a second copy of the answer. Slicing
# rather than summarising, so a session cannot be inflated by a long reply.
TURN_TEXT_CAP = 400


@dataclass
class Session:
    """One conversation: what it established, what it is waiting on, what was said."""

    touched: float
    slots: dict[str, str] = field(default_factory=dict)
    # The question that produced an ask-back and has not been answered yet. The
    # next turn that supplies the missing fact re-asks it, which is the whole
    # point: the person answered the question the assistant asked, so the
    # assistant owes them the answer to the question they asked.
    pending: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)


class SessionStore:
    """In-process conversation state: thread-safe, bounded, idle-expiring."""

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS,
                 idle_seconds: float = DEFAULT_IDLE_SECONDS,
                 clock=time.monotonic) -> None:
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        # Monotonic, not wall clock: an idle period measured against a clock the
        # operating system can step backwards is not a bound at all.
        self._clock = clock
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        # Reentrant to match the rest of the shared state in this codebase
        # (`assistant/knowledge/store/locking.py`, `assistant/cache.py`), so a
        # method may call another.
        self._lock = threading.RLock()

    # -- identity ----------------------------------------------------------

    def open(self, session_id: str = "") -> str:
        """The id this request belongs to: the one presented, or a fresh one.

        A presented id is honoured only if this store minted it and it has not
        expired. That is what keeps a cookie from being a request parameter: an
        id invented by the caller names no session, so it gets a new one rather
        than a guess at somebody else's. The token is 128 bits of `secrets`
        randomness for the same reason.
        """
        with self._lock:
            self._sweep()
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touched = self._clock()
                self._sessions.move_to_end(session_id)
                return session_id
            fresh = secrets.token_urlsafe(16)
            self._sessions[fresh] = Session(touched=self._clock())
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)     # least recently used out
            return fresh

    # -- reading -----------------------------------------------------------

    def carried(self, session_id: str) -> dict[str, str]:
        """The building facts this session has established. A copy, never the dict."""
        with self._lock:
            session = self._sessions.get(session_id)
            return dict(session.slots) if session else {}

    def pending(self, session_id: str) -> str:
        """The question still waiting on a missing fact, or an empty string."""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.pending if session else ""

    def turns(self, session_id: str) -> list[tuple[str, str]]:
        """The conversation so far, oldest first, for the page to show back."""
        with self._lock:
            session = self._sessions.get(session_id)
            return list(session.turns) if session else []

    # -- writing -----------------------------------------------------------

    def remember(self, session_id: str, question: str, answer: str,
                 slots: dict, pending: str = "") -> None:
        """Fold one finished turn into the session.

        Only the carried slots are taken, and only when the turn actually
        detected them — a slot absent from this turn leaves the remembered value
        alone, because "not mentioned again" is not "no longer true". An unknown
        session id writes nothing rather than creating one, so a forged cookie
        cannot seed state for a later caller to read.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            for name in CARRIED_SLOTS:
                if name in slots:
                    session.slots[name] = slots[name]
            session.pending = pending
            session.turns.append((question, answer[:TURN_TEXT_CAP]))
            del session.turns[:-MAX_TURNS]
            session.touched = self._clock()

    # -- housekeeping ------------------------------------------------------

    def _sweep(self) -> None:
        """Drop everything untouched for the idle period.

        Cheap because the order is least-recently-touched first, so the expired
        sessions are a prefix: stop at the first live one rather than walking
        the whole store on every request.
        """
        cutoff = self._clock() - self.idle_seconds
        while self._sessions:
            _id, session = next(iter(self._sessions.items()))
            if session.touched > cutoff:
                return
            self._sessions.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
