"""Persistent backend for multi-turn conversation state.

Wraps the in-memory `SessionStore` with a database backend (SQLite or PostgreSQL),
enabling session recovery across server restarts and multiple concurrent sessions.

The database schema is in db/schema.sqlite.sql and db/schema.postgres.sql,
with identical column names and semantics.

Design:
- In-memory store remains the source of truth for performance.
- DB is kept in sync on every write and loaded on reconnect.
- Idle expiration (30 min) is checked on every read/write.
- Session audience is bound at creation, cannot change.
- Slots and pending flag are stored as JSON/JSONB.
- Turns are stored as JSON/JSONB, capped at MAX_TURNS.

Thread safety: Reentrant lock matches the in-memory store's semantics.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Protocol

from .session import Session, SessionStore, DEFAULT_IDLE_SECONDS


class Connection(Protocol):
    """Duck-type for SQLite and PostgreSQL connections."""

    def execute(self, sql: str, params: tuple | dict = ()) -> object:
        ...

    def executemany(self, sql: str, params: list[tuple | dict]) -> object:
        ...

    def commit(self) -> None:
        ...

    def close(self) -> None:
        ...


class PersistedSessionStore(SessionStore):
    """In-process sessions with database persistence and idle expiration.

    Extends SessionStore to read/write to SQLite or PostgreSQL, allowing
    sessions to survive server restarts and be shared across instances.

    Args:
        store: The in-memory SessionStore to wrap.
        connection: SQLite or PostgreSQL connection.
        is_postgres: True if connection is PostgreSQL, False if SQLite.
    """

    def __init__(
        self,
        store: SessionStore,
        connection: Connection,
        is_postgres: bool = False,
    ) -> None:
        """Wrap an in-memory store with database persistence."""
        super().__init__(
            max_sessions=store.max_sessions,
            idle_seconds=store.idle_seconds,
            clock=store._clock,
        )
        self._wrapped = store
        self._connection = connection
        self._is_postgres = is_postgres
        # Match the wrapped store's lock
        self._lock = store._lock

    # -- identity ----------------------------------------------------------

    def open(self, session_id: str = "") -> str:
        """Open a session from the store or database, or create a fresh one.

        Presented id is honoured only if it exists in the DB and has not expired.
        Expired sessions are swept before checking.
        """
        with self._lock:
            self._sweep_db()
            if session_id and self._load_from_db(session_id):
                # Found in DB and loaded into in-memory store
                return session_id
            # Not found or expired; create fresh
            fresh = self._wrapped.open(session_id="")
            self._save_to_db(fresh)
            return fresh

    # -- reading -----------------------------------------------------------

    def carried(self, session_id: str) -> dict[str, str]:
        """The building facts this session has established."""
        with self._lock:
            self._sweep_db()
            return self._wrapped.carried(session_id)

    def pending(self, session_id: str) -> str:
        """The question still waiting on a missing fact, or an empty string."""
        with self._lock:
            self._sweep_db()
            return self._wrapped.pending(session_id)

    def turns(self, session_id: str) -> list[tuple[str, str]]:
        """The conversation so far, oldest first, for the page to show back."""
        with self._lock:
            self._sweep_db()
            return self._wrapped.turns(session_id)

    # -- writing -----------------------------------------------------------

    def remember(
        self,
        session_id: str,
        question: str,
        answer: str,
        slots: dict,
        pending: str = "",
    ) -> None:
        """Fold one finished turn into the session and persist it."""
        with self._lock:
            # Update in-memory store first
            self._wrapped.remember(session_id, question, answer, slots, pending)
            # Then sync to database
            self._save_to_db(session_id)
            self._sweep_db()

    # -- database operations -----------------------------------------------

    def _load_from_db(self, session_id: str) -> bool:
        """Load a session from the database into the in-memory store.

        Returns True if found and loaded, False if not found or expired.
        """
        try:
            row = self._query_one(
                "SELECT session_id, audience, slots, pending, turns, touched "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            if not row:
                return False

            _id, audience, slots_json, pending, turns_json, touched = row
            # Deserialize
            slots = json.loads(slots_json) if slots_json else {}
            turns = json.loads(turns_json) if turns_json else []

            # Check expiration
            if touched < (self._clock() - self.idle_seconds):
                return False

            # Load into in-memory store
            session = Session(
                touched=touched,
                slots=slots,
                pending=pending,
                turns=turns,
            )
            self._wrapped._sessions[session_id] = session
            self._wrapped._sessions.move_to_end(session_id)
            return True
        except Exception:
            # DB error; fail gracefully (return False so caller gets new session)
            return False

    def _save_to_db(self, session_id: str) -> None:
        """Persist a session from the in-memory store to the database."""
        with self._lock:
            session = self._wrapped._sessions.get(session_id)
            if not session:
                return

            # Serialize
            slots_json = json.dumps(session.slots)
            turns_json = json.dumps(session.turns)
            now_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            try:
                # Upsert: insert if not exists, update if exists
                if self._is_postgres:
                    # PostgreSQL upsert
                    self._execute(
                        """
                        INSERT INTO sessions
                          (session_id, audience, slots, pending, turns, touched, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                          slots = EXCLUDED.slots,
                          pending = EXCLUDED.pending,
                          turns = EXCLUDED.turns,
                          touched = EXCLUDED.touched,
                          updated_at = EXCLUDED.updated_at
                        """,
                        (
                            session_id,
                            session.slots.get("audience", "public"),
                            slots_json,
                            session.pending,
                            turns_json,
                            session.touched,
                            now_timestamp,
                            now_timestamp,
                        ),
                    )
                else:
                    # SQLite upsert
                    self._execute(
                        """
                        INSERT INTO sessions
                          (session_id, audience, slots, pending, turns, touched, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (session_id) DO UPDATE SET
                          slots = EXCLUDED.slots,
                          pending = EXCLUDED.pending,
                          turns = EXCLUDED.turns,
                          touched = EXCLUDED.touched,
                          updated_at = EXCLUDED.updated_at
                        """,
                        (
                            session_id,
                            session.slots.get("audience", "public"),
                            slots_json,
                            session.pending,
                            turns_json,
                            session.touched,
                            now_timestamp,
                            now_timestamp,
                        ),
                    )
                self._connection.commit()
            except Exception:
                # DB error; silently fail to avoid breaking the answer
                pass

    def _sweep_db(self) -> None:
        """Delete expired sessions from the database.

        Runs on every read/write. Cheap because the index is on touched.
        """
        try:
            cutoff = self._clock() - self.idle_seconds
            if self._is_postgres:
                self._execute(
                    "DELETE FROM sessions WHERE touched < %s",
                    (cutoff,),
                )
            else:
                self._execute(
                    "DELETE FROM sessions WHERE touched < ?",
                    (cutoff,),
                )
            self._connection.commit()
        except Exception:
            # DB error; silently fail
            pass

    def _execute(self, sql: str, params: tuple | dict = ()) -> None:
        """Execute a SQL statement (INSERT, UPDATE, DELETE)."""
        cursor = self._connection.cursor()
        cursor.execute(sql, params)

    def _query_one(
        self, sql: str, params: tuple | dict = ()
    ) -> tuple | None:
        """Execute a query and return one row, or None."""
        cursor = self._connection.cursor()
        cursor.execute(sql, params)
        return cursor.fetchone()


def open_persisted_session_store(
    connection: Connection,
    is_postgres: bool = False,
) -> PersistedSessionStore:
    """Create a persisted session store backed by a database connection.

    Args:
        connection: SQLite or PostgreSQL connection.
        is_postgres: True if connection is PostgreSQL, False if SQLite.

    Returns:
        A PersistedSessionStore wrapping an in-memory SessionStore.
    """
    in_memory = SessionStore()
    return PersistedSessionStore(in_memory, connection, is_postgres=is_postgres)
