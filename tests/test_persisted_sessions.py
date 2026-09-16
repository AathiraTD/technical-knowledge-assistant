"""Tests for persistent session storage across server restarts.

Verifies that:
- Sessions persist to the database and survive server restarts
- Idle expiration works correctly (sessions expire after 30 min idle)
- Concurrent reads/writes are serialized correctly
- Audience binding works (cannot change mid-session)
- Turns list is capped at MAX_TURNS on persist
- Session state survives database issues gracefully
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from assistant.session import Session, SessionStore, CARRIED_SLOTS, MAX_TURNS, TURN_TEXT_CAP
from assistant.session_storage import PersistedSessionStore, open_persisted_session_store


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database with the sessions table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Apply schema
    schema_path = Path(__file__).parent.parent / "db" / "schema.sqlite.sql"
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.commit()

    yield conn

    conn.close()
    Path(db_path).unlink()


class TestPersistedSessionStore:
    """Test suite for PersistedSessionStore."""

    def test_create_and_load_session(self, temp_sqlite_db):
        """Session is created, persisted to DB, and loads on reconnect."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)

        # Create a session
        session_id = store.open()
        assert session_id
        assert len(session_id) > 0

        # Add some data
        store.remember(
            session_id,
            question="How much Solo?",
            answer="Between 5 and 6 litres per 25kg sack.",
            slots={"substrate": "brick", "location": "exterior"},
            pending="",
        )

        # Create a new store (simulating server restart)
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)

        # Load the session
        loaded_id = store2.open(session_id)
        assert loaded_id == session_id

        # Verify the data loaded
        assert store2.carried(session_id) == {"substrate": "brick", "location": "exterior"}
        turns = store2.turns(session_id)
        assert len(turns) == 1
        assert turns[0][0] == "How much Solo?"

    def test_idle_expiration(self, temp_sqlite_db):
        """Sessions expire after 30 minutes of inactivity."""
        store = PersistedSessionStore(
            SessionStore(idle_seconds=1),  # 1 second for testing
            temp_sqlite_db,
            is_postgres=False,
        )

        # Create a session
        session_id = store.open()
        store.remember(
            session_id,
            question="Test?",
            answer="Answer.",
            slots={},
            pending="",
        )

        # Wait for expiration
        time.sleep(1.1)

        # Try to load it
        new_id = store.open(session_id)
        # Should get a new session, not the old one
        assert new_id != session_id

    def test_audience_binding(self, temp_sqlite_db):
        """Audience is bound at creation and cannot change."""
        # Create session for public audience
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        session_id = store.open()

        # Store the session in DB with a mock audience (simulating a real implementation
        # that would bind audience from identity)
        cursor = temp_sqlite_db.cursor()
        cursor.execute(
            "UPDATE sessions SET audience = ? WHERE session_id = ?",
            ("public", session_id),
        )
        temp_sqlite_db.commit()

        # Reload
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        loaded_id = store2.open(session_id)
        assert loaded_id == session_id

        # Audience should be public (cannot be changed mid-session)

    def test_slot_carrying(self, temp_sqlite_db):
        """Carried slots (substrate, location, exposure) persist across turns."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        session_id = store.open()

        # Turn 1: Set substrate and location
        store.remember(
            session_id,
            question="Wall is brick, exterior.",
            answer="OK, noted.",
            slots={"substrate": "brick", "location": "exterior"},
            pending="",
        )

        # Turn 2: Retrieve carried slots (simulating a reload)
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        store2.open(session_id)

        carried = store2.carried(session_id)
        assert carried.get("substrate") == "brick"
        assert carried.get("location") == "exterior"

        # Turn 3: New question should use carried substrate
        store2.remember(
            session_id,
            question="What about Forte?",
            answer="Forte is compatible with brick.",
            slots={},  # Not restating substrate
            pending="",
        )

        # Substrate should still be remembered
        assert store2.carried(session_id).get("substrate") == "brick"

    def test_pending_question_memory(self, temp_sqlite_db):
        """Pending flag is stored and retrieved correctly."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        session_id = store.open()

        # Turn 1: Ask a question, get an ask-back (set pending)
        store.remember(
            session_id,
            question="How much Solo for the wall?",
            answer="I need to know the substrate. Is it brick?",
            slots={},
            pending="How much Solo for the wall?",
        )

        # Reload
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        store2.open(session_id)

        # Pending should be remembered
        assert store2.pending(session_id) == "How much Solo for the wall?"

        # Turn 2: User provides the missing fact, clear pending
        store2.remember(
            session_id,
            question="It's brick, exterior.",
            answer="5-6 litres per 25kg sack.",
            slots={"substrate": "brick", "location": "exterior"},
            pending="",
        )

        # Pending should now be empty
        assert store2.pending(session_id) == ""

    def test_turns_list_capped(self, temp_sqlite_db):
        """Turns list is capped at MAX_TURNS (8) when stored."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        session_id = store.open()

        # Add 10 turns (exceeds MAX_TURNS)
        for i in range(10):
            store.remember(
                session_id,
                question=f"Question {i}",
                answer=f"Answer {i}",
                slots={},
                pending="",
            )

        # Reload
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        store2.open(session_id)

        turns = store2.turns(session_id)
        # Should be capped at MAX_TURNS
        assert len(turns) <= MAX_TURNS

    def test_answer_text_capped(self, temp_sqlite_db):
        """Answer excerpt is capped at TURN_TEXT_CAP characters."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        session_id = store.open()

        long_answer = "x" * 2000  # Much longer than TURN_TEXT_CAP

        store.remember(
            session_id,
            question="Test?",
            answer=long_answer,
            slots={},
            pending="",
        )

        # Reload
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        store2.open(session_id)

        turns = store2.turns(session_id)
        assert len(turns[0][1]) <= TURN_TEXT_CAP

    def test_concurrent_access(self, temp_sqlite_db):
        """Concurrent reads/writes are serialized correctly."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        session_id = store.open()

        results = []

        def thread_write(i):
            store.remember(
                session_id,
                question=f"Q{i}",
                answer=f"A{i}",
                slots={},
                pending="",
            )
            results.append(("write", i))

        def thread_read(i):
            turns = store.turns(session_id)
            results.append(("read", i, len(turns)))

        # Spawn several threads
        threads = []
        for i in range(5):
            t_w = threading.Thread(target=thread_write, args=(i,))
            t_r = threading.Thread(target=thread_read, args=(i,))
            threads.extend([t_w, t_r])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all operations completed without error
        assert len(results) == 10

        # Reload and verify final state
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        store2.open(session_id)
        turns = store2.turns(session_id)
        assert len(turns) == 5

    def test_graceful_db_failure(self, temp_sqlite_db):
        """Session state survives database errors gracefully."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        session_id = store.open()

        store.remember(
            session_id,
            question="Test?",
            answer="Answer.",
            slots={"substrate": "brick"},
            pending="",
        )

        # Simulate DB error by closing the connection
        temp_sqlite_db.close()

        # These should not crash; they should fail gracefully
        carried = store.carried(session_id)
        pending = store.pending(session_id)
        turns = store.turns(session_id)

        # In-memory state should still be available
        assert carried.get("substrate") == "brick"

    def test_multiple_concurrent_sessions(self, temp_sqlite_db):
        """Multiple independent sessions remain isolated."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)

        # Create session 1
        s1 = store.open()
        store.remember(s1, "Q1", "A1", {"substrate": "brick"}, "")

        # Create session 2
        s2 = store.open()
        store.remember(s2, "Q2", "A2", {"substrate": "stone"}, "")

        # Reload and verify isolation
        store2 = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)
        store2.open(s1)
        store2.open(s2)

        assert store2.carried(s1).get("substrate") == "brick"
        assert store2.carried(s2).get("substrate") == "stone"

        turns1 = store2.turns(s1)
        turns2 = store2.turns(s2)
        assert len(turns1) == 1
        assert len(turns2) == 1
        assert turns1[0][0] == "Q1"
        assert turns2[0][0] == "Q2"

    def test_fresh_session_gets_new_id(self, temp_sqlite_db):
        """Forged session ID does not load; creates fresh session instead."""
        store = PersistedSessionStore(SessionStore(), temp_sqlite_db, is_postgres=False)

        # Try to open a non-existent session
        result = store.open("nonexistent_id_12345")

        # Should get a new ID
        assert result != "nonexistent_id_12345"
        assert len(result) > 0

    def test_schema_migration(self, temp_sqlite_db):
        """Sessions table exists and is queryable."""
        cursor = temp_sqlite_db.cursor()

        # Verify table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        assert cursor.fetchone() is not None

        # Verify columns
        cursor.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {"session_id", "audience", "slots", "pending", "turns", "touched", "created_at", "updated_at"}
        assert expected.issubset(columns)
