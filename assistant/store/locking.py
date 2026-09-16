"""Serialise access to a repository that is shared across threads.

`ThreadingHTTPServer` hands every request to a new thread. A SQLite connection
is bound to the thread that opened it, and a single `psycopg` connection cannot
carry two transactions at once — so a store opened at startup and used from
request threads is wrong in both adapters, in different ways. The web page
raised `ProgrammingError` on every question that touched the store.

A lock rather than a connection per thread, deliberately. Retrieval is one
matrix multiply over a few hundred vectors and measures around ten milliseconds,
so serialising reads costs nothing anyone can perceive, while a connection per
thread would hold a separate copy of the loaded vector matrix for each one. It
is also honest about what it is: concurrent requests are served *correctly* and
*serially*. This is not a connection pool and must not be described as one.

The wrapper lives here rather than in `ui.py` because it is a property of the
store, not of the page: the CLI, the harness and any future channel adapter get
it from the same factory. It was first written when `ui.py` was excluded from
coverage measurement, which is no longer true of either file.
"""

from __future__ import annotations

import threading
from typing import Any


class LockedRepository:
    """A `KnowledgeRepository` that admits one caller at a time.

    Every Protocol method is delegated under one reentrant lock. Reentrant
    because `read_snapshot()` is a context manager that holds the lock while the
    caller makes further repository calls inside it — a plain lock would
    deadlock on the first `retrieve` inside a snapshot.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._lock = threading.RLock()

    # -- the repository surface -------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate anything not named here, still under the lock.

        Written as a fallback rather than a long list of hand-written wrappers
        so that a method added to the Protocol later is covered by default. The
        alternative fails open: a new method would bypass the lock silently and
        the bug would look exactly like the one this class exists to fix.
        """
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def guarded(*args, **kwargs):
            with self._lock:
                return attribute(*args, **kwargs)

        guarded.__name__ = getattr(attribute, "__name__", name)
        guarded.__doc__ = getattr(attribute, "__doc__", None)
        return guarded

    # -- the context managers ----------------------------------------------

    def read_snapshot(self):
        """Hold the lock for the whole consistent read, not just its opening.

        The point of a read snapshot is that everything inside it sees one
        state. Releasing the lock after opening it would let another thread
        publish midway through, which is the inconsistency the snapshot exists
        to prevent.
        """
        return _LockedSnapshot(self._lock, self._inner)

    def close(self) -> None:
        with self._lock:
            self._inner.close()

    def __enter__(self) -> "LockedRepository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _LockedSnapshot:
    """Holds the repository lock for the lifetime of one consistent read."""

    def __init__(self, lock: threading.RLock, inner: Any) -> None:
        self._lock = lock
        self._inner = inner
        self._snapshot = None

    def __enter__(self):
        self._lock.acquire()
        try:
            self._snapshot = self._inner.read_snapshot()
            return self._snapshot.__enter__()
        except BaseException:
            self._lock.release()
            raise

    def __exit__(self, *exc) -> bool | None:
        try:
            return self._snapshot.__exit__(*exc)
        finally:
            self._lock.release()
