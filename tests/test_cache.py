"""The answer cache, tested against the leak it could cause.

Decision 14 names the failure mode outright: leave the audience set out of the
key and a staff answer reaches a public caller, which is a leak rather than a
performance bug. So the first test here is that leak, and the speed properties
come after it.

Nothing in this file needs Ollama, an index or a store. A cache is a dictionary
with rules, and the rules are the whole point.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.cache import AnswerCache, normalise           # noqa: E402

QUESTION = "How much water does Solo Onecoat need per bag?"
SNAP = "snap-1"
MODEL = "qwen3.5:4b"
CHUNKING = "structure-aware/1.0"


def key(cache: AnswerCache, question: str = QUESTION,
        audiences: tuple[str, ...] = ("public",), snapshot: str = SNAP,
        model: str = MODEL, chunking: str = CHUNKING) -> tuple:
    return cache.key(question, audiences, snapshot, model, chunking)


# --------------------------------------------------------------- the leak


def test_a_staff_answer_is_not_served_to_a_public_caller():
    """Decision 14's named failure: a leak, not a performance bug."""
    cache = AnswerCache()
    cache.put(key(cache, audiences=("staff",)), "the internal margin is 42%")

    assert cache.get(key(cache, audiences=("public",))) is None


def test_a_public_answer_is_not_served_to_a_staff_caller_either():
    """The filter is a boundary, not a one-way gate; staff may see more."""
    cache = AnswerCache()
    cache.put(key(cache, audiences=("public",)), "public answer")

    assert cache.get(key(cache, audiences=("staff",))) is None


def test_the_order_rights_were_listed_in_is_not_part_of_the_key():
    """Two callers with the same rights should share an entry."""
    cache = AnswerCache()
    cache.put(key(cache, audiences=("public", "trade")), "answer")

    assert cache.get(key(cache, audiences=("trade", "public"))) == "answer"


# --------------------------------------------------- what invalidates an entry


def test_a_new_snapshot_invalidates_the_answer():
    """An answer is only true of the index that produced it."""
    cache = AnswerCache()
    cache.put(key(cache, snapshot="snap-1"), "old answer")

    assert cache.get(key(cache, snapshot="snap-2")) is None


def test_a_different_generation_model_invalidates_the_answer():
    cache = AnswerCache()
    cache.put(key(cache), "answer from qwen")

    assert cache.get(key(cache, model="granite4.2:3b")) is None


def test_a_different_chunking_version_invalidates_the_answer():
    """Chunk boundaries move without any content changing; the answer can too."""
    cache = AnswerCache()
    cache.put(key(cache), "answer")

    assert cache.get(key(cache, chunking="structure-aware/2.0")) is None


# ----------------------------------------------------------- what it does fold


def test_case_and_spacing_do_not_make_a_new_question():
    cache = AnswerCache()
    cache.put(key(cache), "answer")

    assert cache.get(key(cache, question="  HOW much   WATER does Solo "
                                         "Onecoat need per bag?  ")) == "answer"


def test_a_different_wording_is_a_different_question():
    """Serving a near neighbour is the approximation this design refuses."""
    cache = AnswerCache()
    cache.put(key(cache), "answer")

    assert cache.get(key(cache, question="How much water for Solo?")) is None


def test_normalise_folds_only_case_and_whitespace():
    assert normalise("  A   B  ") == "a b"
    assert normalise("A b") == normalise("a  B")


# ------------------------------------------------------------ what it stores


def test_a_refusal_is_cached_too():
    """A refusal costs a full retrieval to discover; refusing fast is worth as much."""
    cache = AnswerCache()
    cache.put(key(cache), "I could not find this in the published material.")

    assert "could not find" in cache.get(key(cache))


def test_hits_and_misses_are_counted_so_the_effect_is_measurable():
    cache = AnswerCache()
    assert cache.get(key(cache)) is None
    cache.put(key(cache), "answer")
    assert cache.get(key(cache)) == "answer"

    assert (cache.hits, cache.misses) == (1, 1)


def test_the_oldest_entry_is_dropped_when_the_cache_is_full():
    cache = AnswerCache(max_entries=2)
    cache.put(key(cache, question="one"), 1)
    cache.put(key(cache, question="two"), 2)
    cache.put(key(cache, question="three"), 3)

    assert len(cache) == 2
    assert cache.get(key(cache, question="one")) is None
    assert cache.get(key(cache, question="three")) == 3


def test_reading_an_entry_keeps_it_alive():
    """A question asked repeatedly is the one worth keeping."""
    cache = AnswerCache(max_entries=2)
    cache.put(key(cache, question="one"), 1)
    cache.put(key(cache, question="two"), 2)
    cache.get(key(cache, question="one"))          # one is now the most recent
    cache.put(key(cache, question="three"), 3)

    assert cache.get(key(cache, question="one")) == 1
    assert cache.get(key(cache, question="two")) is None


def test_clearing_empties_it_and_resets_the_counters():
    cache = AnswerCache()
    cache.put(key(cache), "answer")
    cache.get(key(cache))
    cache.clear()

    assert len(cache) == 0
    assert (cache.hits, cache.misses) == (0, 0)


# --------------------------------------------------------------- concurrency


def test_concurrent_callers_do_not_corrupt_it():
    """The web server answers on many threads; one shared cache sits behind them."""
    cache = AnswerCache(max_entries=64)

    def work(n: int) -> None:
        k = key(cache, question=f"question {n % 16}")
        cache.put(k, n)
        cache.get(k)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(200)))

    assert len(cache) == 16
