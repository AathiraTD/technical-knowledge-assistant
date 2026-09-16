"""Reuse of a finished answer, keyed so it cannot leak across audiences.

Decision 14 designs a cache keyed on the question template, the detected slot
values, the caller's audience set and the index version. Templates do not exist
in this code, so what is built here is the weaker **exact-key** form that
decision 14 itself compares against and rates lower: it hits only when the same
question is asked in the same words. That distinction is worth stating rather
than letting the decision record imply more than exists.

Weaker on hit rate, identical on safety. Decision 14's warning is explicit —
"the audience set must be in the key or a staff answer reaches a public caller,
a leak, not a performance bug" — so the audience set is in the key, and so is
the snapshot, because an answer is only true of the index that produced it.

It is worth having at all because a composed answer costs 35 to 55 seconds on a
processor and a repeat costs nothing. Refusals are cached too: a refusal is a
full retrieval's work to discover, and refusing quickly is as valuable as
answering quickly.

Deliberately not a semantic cache. Serving a near neighbour's answer is exactly
the approximation this system refuses to make everywhere else.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any

# Enough for a demonstration and a day's questions, small enough that the
# memory is unremarkable. An answer is a few kilobytes.
DEFAULT_MAX_ENTRIES = 256

_WHITESPACE = re.compile(r"\s+")


def normalise(question: str) -> str:
    """Fold the differences that are certainly not meaning.

    Case and spacing only. Nothing here touches words, because two questions
    that differ by a word are two questions — treating them as one is the
    semantic caching this design rejects.
    """
    return _WHITESPACE.sub(" ", question.strip().lower())


class AnswerCache:
    """An exact-match, audience-scoped, snapshot-scoped store of finished answers."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: "OrderedDict[tuple, Any]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(question: str, audiences: tuple[str, ...], snapshot_id: str,
            generation_model: str, chunking_version: str,
            carried: dict | None = None) -> tuple:
        """Everything that can change the right answer to the same words.

        The audience set is sorted rather than taken as given, so ("public",
        "trade") and ("trade", "public") are one key. Two callers with the same
        rights should share a cache entry; the order they listed their rights in
        is not a fact about the answer.

        `carried` is in the key for the same reason the audience set is, and the
        failure it prevents is worse. Slots held from an earlier turn change the
        answer to identical words — "what plaster should I use" answered for a
        brick wall is a different answer from the same question answered for
        cob. Leaving them out served one conversation's answer to another
        caller who had never said brick, which is exactly the silent assumption
        about somebody's wall that decision 10 exists to prevent.

        It also broke multi-turn outright. Resuming a pending question re-asks
        the same words, so without the slots in the key the cache returned the
        stored *ask-back* instead of the answer the new information had finally
        made possible — the feature defeating itself through its own cache.
        """
        return (normalise(question), tuple(sorted(audiences)), snapshot_id,
                generation_model, chunking_version,
                tuple(sorted((carried or {}).items())))

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
            self.misses += 1
            return None

    def put(self, key: tuple, answer: Any) -> None:
        with self._lock:
            self._entries[key] = answer
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)     # oldest out first

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
