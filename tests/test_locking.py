"""The lock that lets one store serve a threaded server.

This wrapper is the reason the web page works, so it is tested for the
properties that matter rather than for its happy path: that concurrent callers
are serialised, that a read snapshot holds the lock for its whole lifetime
rather than only its opening, and that a failure inside the snapshot releases
the lock instead of wedging every later request.

The last one is the quiet danger. A lock leaked on an exception does not raise;
it hangs, and the page simply stops answering.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.knowledge.store.locking import LockedRepository          # noqa: E402


class Inner:
    """A stand-in store that records overlap, so serialisation is observable."""

    def __init__(self, snapshot_fails: bool = False):
        self.name = "inner"                      # a non-callable attribute
        self.closed = False
        self.snapshot_fails = snapshot_fails
        self.exited = False
        self.concurrent = 0
        self.max_concurrent = 0
        self._counter = threading.Lock()

    def _enter(self):
        with self._counter:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)

    def _leave(self):
        with self._counter:
            self.concurrent -= 1

    def retrieve(self, *_a, **_k):
        self._enter()
        time.sleep(0.01)                         # long enough for overlap to show
        self._leave()
        return ["passage"]

    def snapshot(self):
        return "snap-1"

    def close(self):
        self.closed = True

    def read_snapshot(self):
        outer = self

        class Ctx:
            def __enter__(self):
                if outer.snapshot_fails:
                    raise RuntimeError("the store could not open a read snapshot")
                return outer

            def __exit__(self, *exc):
                outer.exited = True
                return False

        return Ctx()


def test_a_non_callable_attribute_passes_straight_through():
    """Delegation must not wrap data as though it were a method."""
    repo = LockedRepository(Inner())
    assert repo.name == "inner"


def test_a_delegated_call_returns_what_the_store_returned():
    assert LockedRepository(Inner()).retrieve([0.0]) == ["passage"]


def test_the_delegated_method_keeps_its_name_and_docstring():
    """A wrapper that erases identity makes every traceback harder to read."""
    repo = LockedRepository(Inner())
    assert repo.snapshot.__name__ == "snapshot"


def test_concurrent_callers_are_serialised_rather_than_overlapping(  ):
    """The whole point: one caller at a time against one connection."""
    inner = Inner()
    repo = LockedRepository(inner)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: repo.retrieve([0.0]), range(16)))

    assert inner.max_concurrent == 1, (
        f"{inner.max_concurrent} callers were inside the store at once")


def test_a_read_snapshot_holds_the_lock_for_its_whole_life():
    """Releasing after opening would let a publish land mid-read."""
    repo = LockedRepository(Inner())
    held = []

    with repo.read_snapshot():
        def other():
            held.append(repo._lock.acquire(blocking=False))
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()

    assert held == [False], "another thread entered the store mid-snapshot"


def test_the_snapshot_exits_the_inner_context():
    inner = Inner()
    with LockedRepository(inner).read_snapshot():
        pass
    assert inner.exited is True


def test_a_snapshot_that_fails_to_open_releases_the_lock():
    """A leaked lock does not raise. It hangs, and the page stops answering."""
    repo = LockedRepository(Inner(snapshot_fails=True))

    with pytest.raises(RuntimeError, match="read snapshot"):
        with repo.read_snapshot():
            pass

    assert repo._lock.acquire(blocking=False) is True, "the lock was leaked"
    repo._lock.release()


def test_closing_closes_the_store_underneath():
    inner = Inner()
    LockedRepository(inner).close()
    assert inner.closed is True


def test_it_works_as_a_context_manager():
    inner = Inner()
    with LockedRepository(inner) as repo:
        assert repo.snapshot() == "snap-1"
    assert inner.closed is True
