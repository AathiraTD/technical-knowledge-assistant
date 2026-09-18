"""The embedding cache, tested on the property that makes it safe to ship.

The cache exists to save about forty minutes per rebuild, but the reason it can
be committed to the repository at all is its key: the SHA-256 of the passage
text together with the model tag and the dimension count. A vector comes back
only for the exact text that produced it, computed by the exact model that
produced it. Everything else about the cache is an optimisation; that is the
correctness argument, so most of this file is about the key.

The rest covers the mechanics that would fail quietly. A miss has to stay
missing, so the caller recomputes it rather than embedding a hole. And a read
of more than five hundred passages has to page, because SQLite will not take
more parameters than that in one statement and 644 is the real corpus size.

No Ollama, no index, and every cache lives in a temporary directory: the
shipped `data/embeddings.db` is never touched by these tests.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.indexing.embedcache import EmbeddingCache, key_for          # noqa: E402

MODEL = "qwen3-embedding:0.6b"
DIMS = 4


def vec(*values: float) -> list[float]:
    return list(values) + [0.0] * (DIMS - len(values))


@pytest.fixture
def cache(tmp_path):
    """A cache in a temporary directory, never the shipped one."""
    made = EmbeddingCache(tmp_path / "nested" / "embeddings.db")
    try:
        yield made
    finally:
        try:
            made.close()
        except sqlite3.ProgrammingError:
            pass          # the test closed it deliberately


# ------------------------------------------------------------------- the key


def test_a_changed_passage_gets_a_different_key():
    """An edited datasheet must not be answered from the old text's vector."""
    assert (key_for("Mix with 5-6 litres per sack.", MODEL, DIMS)
            != key_for("Mix with 4-5 litres per sack.", MODEL, DIMS))


def test_a_changed_model_gets_a_different_key():
    """A vector from another model is the confident nonsense the header prevents."""
    assert (key_for("Mix with 5-6 litres per sack.", MODEL, DIMS)
            != key_for("Mix with 5-6 litres per sack.", "nomic-embed-text", DIMS))


def test_a_changed_dimension_count_gets_a_different_key():
    """Same model tag, different configured width, is still a different vector."""
    assert (key_for("Mix with 5-6 litres per sack.", MODEL, 1024)
            != key_for("Mix with 5-6 litres per sack.", MODEL, 768))


def test_the_same_passage_and_model_get_the_same_key():
    """Without this the cache never hits and the forty minutes are paid anyway."""
    assert (key_for("Coverage is 2.5 m2 per bag.", MODEL, DIMS)
            == key_for("Coverage is 2.5 m2 per bag.", MODEL, DIMS))


# ---------------------------------------------------------------- round trip


def test_a_stored_vector_comes_back_by_input_position(cache):
    """Positions are how the caller reassembles the batch; a shift misfiles every one."""
    texts = ["mixing", "coverage", "curing"]
    cache.put_many(list(zip(texts, [vec(1, 0), vec(0, 1), vec(0, 0, 1)])),
                   MODEL, DIMS)

    found = cache.get_many(texts, MODEL, DIMS)

    assert sorted(found) == [0, 1, 2]
    assert found[0] == pytest.approx(vec(1, 0))
    assert found[1] == pytest.approx(vec(0, 1))
    assert found[2] == pytest.approx(vec(0, 0, 1))


def test_only_the_found_positions_are_returned_so_the_rest_recompute(cache):
    """A miss returned as an empty vector would index a hole and retrieve nothing."""
    cache.put_many([("mixing", vec(1, 0))], MODEL, DIMS)

    found = cache.get_many(["mixing", "never seen before"], MODEL, DIMS)

    assert list(found) == [0]
    assert 1 not in found


def test_another_models_vectors_are_never_served(cache):
    """Pointing the indexer at a new model must miss every lookup, not reuse them."""
    cache.put_many([("mixing", vec(1, 0))], MODEL, DIMS)

    assert cache.get_many(["mixing"], "nomic-embed-text", DIMS) == {}
    assert cache.get_many(["mixing"], MODEL, 1024) == {}


def test_recomputing_a_passage_replaces_rather_than_duplicates(cache):
    """A primary key collision on a rerun must not fail the build."""
    cache.put_many([("mixing", vec(1, 0))], MODEL, DIMS)
    cache.put_many([("mixing", vec(0, 1))], MODEL, DIMS)

    assert cache.count() == 1
    assert cache.get_many(["mixing"], MODEL, DIMS)[0] == pytest.approx(vec(0, 1))


# ------------------------------------------------------------------- paging


def test_more_passages_than_sqlite_takes_parameters_are_read_in_pages(cache):
    """The corpus is 644 passages and SQLite refuses more than 500 parameters."""
    texts = [f"passage number {i}" for i in range(620)]
    cache.put_many([(t, vec(float(i))) for i, t in enumerate(texts)], MODEL, DIMS)

    found = cache.get_many(texts, MODEL, DIMS)

    assert len(found) == 620
    assert found[0][0] == pytest.approx(0.0)
    assert found[500][0] == pytest.approx(500.0), "the second page lost its offset"
    assert found[619][0] == pytest.approx(619.0)


def test_a_miss_in_a_later_page_stays_missing(cache):
    """Paging must not renumber positions, or the wrong passage gets recomputed."""
    texts = [f"passage number {i}" for i in range(620)]
    stored = [t for i, t in enumerate(texts) if i != 517]
    cache.put_many([(t, vec(1.0)) for t in stored], MODEL, DIMS)

    found = cache.get_many(texts, MODEL, DIMS)

    assert 517 not in found
    assert len(found) == 619


# ------------------------------------------------- counters and housekeeping


def test_hits_and_misses_are_counted_so_the_report_cannot_lie(cache):
    """A run that quietly recomputed everything must not read like a cached one."""
    cache.put_many([("mixing", vec(1, 0))], MODEL, DIMS)

    cache.get_many(["mixing", "unknown", "also unknown"], MODEL, DIMS)

    assert cache.hits == 1
    assert cache.misses == 2


def test_count_reports_what_is_actually_stored(cache):
    assert cache.count() == 0
    cache.put_many([("a", vec(1)), ("b", vec(2))], MODEL, DIMS)
    assert cache.count() == 2


def test_the_context_manager_closes_the_database(tmp_path):
    """A cache left open holds a write lock the next build needs."""
    path = tmp_path / "embeddings.db"
    with EmbeddingCache(path) as opened:
        opened.put_many([("mixing", vec(1, 0))], MODEL, DIMS)

    with pytest.raises(sqlite3.ProgrammingError):
        opened.count()

    # Committed on the way out, so a second process sees the row.
    with EmbeddingCache(path) as reopened:
        assert reopened.count() == 1
