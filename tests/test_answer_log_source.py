"""The defect: evaluation traffic was logged as though a person had asked.

`eval/run.py` builds an `Assistant` the same way every other surface does, and
`log` defaults to `True`, so the harness's questions went into `answer_log`
indistinguishable from real ones. That would be merely untidy if the harness
asked ordinary questions. It does not: the probe set exists to exercise the
near-miss and the far-miss, and those are *supposed* to refuse. A refusal rate
counted from that table therefore measured the question set rather than the
system, and the higher the guardrail's success the worse the number looked.

The fix records who asked rather than stopping the harness logging, because the
harness's rows are the only evidence of how the probes actually routed. These
tests pin the fix at the place the defect lived — the engine, not the store —
and pin the migration, because the column has to arrive in databases that
already exist without failing their next insert.
"""

from __future__ import annotations

import inspect
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.answering.engine import Assistant                       # noqa: E402
from assistant.model import AnswerLogEntry                   # noqa: E402
from assistant.store import SQLiteKnowledgeRepository        # noqa: E402


@pytest.fixture
def repo(tmp_path):
    store = SQLiteKnowledgeRepository(tmp_path / "log.db")
    yield store
    store.close()


# ------------------------------------------------- the engine tags its rows


def test_the_engine_carries_the_surface_onto_every_audit_row(repo, monkeypatch):
    """`_log` writes `self.source`, so tagging the instance tags its rows.

    Built without touching the retriever, which verifies a published snapshot
    on construction: this test is about what `_log` writes, and standing up an
    index to assert it would be testing the indexer.
    """
    engine = Assistant.__new__(Assistant)
    engine.repo, engine.log, engine.source = repo, True, "evaluation"

    answer = SimpleNamespace(path="refuse", diagnostics={}, failed_checks=[])
    reply = SimpleNamespace(parts=[("Asked by the harness.", answer)],
                            audiences=("public",))
    engine._log(reply)

    assert repo.answer_log()[0].source == "evaluation"


def test_a_caller_that_does_not_say_is_unknown_not_cli():
    """`unknown` is a real state and must not be quietly promoted to a surface.

    The default is asserted at the signature rather than on an instance,
    because constructing one requires a published index and the claim under
    test is about what an unstated caller becomes.
    """
    assert inspect.signature(Assistant).parameters["source"].default == "unknown"


def test_every_surface_names_itself():
    """The three entry points tag themselves, which is where the defect was.

    Read as source rather than executed, because starting a server, a REPL and
    the harness to assert three string literals would test the test harness.
    The assertion that matters is that no entry point constructs an untagged
    `Assistant` — an untagged one is exactly what produced the contaminated
    table.
    """
    surfaces = {
        "assistant/cli.py": 'source="cli"',
        "assistant/ui.py": 'source="web"',
        "eval/run.py": 'source="evaluation"',
    }
    for path, expected in surfaces.items():
        body = (ROOT / path).read_text(encoding="utf-8")
        assert expected in body, f"{path} builds an Assistant without naming itself"
        assert "Assistant(repo)" not in body, f"{path} still builds an untagged Assistant"


# ---------------------------------------------------------- the round trip


def test_a_logged_answer_keeps_the_surface_that_asked(repo):
    repo.log_answer(AnswerLogEntry(question="From the harness.",
                                   path_taken="refuse", source="evaluation"))
    assert repo.answer_log()[0].source == "evaluation"


def test_an_empty_source_is_written_as_unknown(repo):
    """`NOT NULL DEFAULT 'unknown'` is a floor, and the adapter respects it.

    An entry constructed with `source=""` is a caller that said nothing, not a
    caller that said "". Writing the empty string would put a fourth value into
    a column with three meanings and one honest absence.
    """
    repo.log_answer(AnswerLogEntry(question="Blank.", path_taken="refuse", source=""))
    assert repo.answer_log()[0].source == "unknown"


def test_refusals_are_countable_by_surface(repo):
    """The whole point: the rate can now be computed over real traffic alone."""
    for surface, path in (("evaluation", "refuse"), ("evaluation", "refuse"),
                          ("web", "compose"), ("cli", "extract")):
        repo.log_answer(AnswerLogEntry(question=f"{surface} {path}",
                                       path_taken=path, source=surface))

    rows = repo.answer_log()
    real = [e for e in rows if e.source in ("cli", "web")]
    assert len(real) == 2
    assert not [e for e in real if e.path_taken == "refuse"]
    assert len([e for e in rows if e.source == "evaluation"]) == 2


# ------------------------------------------------------------- the migration


def test_a_database_written_before_the_column_existed_gains_it(tmp_path):
    """Additive, because `CREATE TABLE IF NOT EXISTS` reshapes nothing.

    A store built before `source` would otherwise fail on its next insert. The
    existing rows take the default rather than being guessed at: nobody
    recorded which surface asked, and `unknown` is what that means.
    """
    path = tmp_path / "old.db"
    store = SQLiteKnowledgeRepository(path)
    store.close()

    # Reproduce the old shape: drop the column the migration is supposed to add.
    legacy = sqlite3.connect(path)
    legacy.execute("ALTER TABLE answer_log DROP COLUMN source")
    legacy.execute(
        """INSERT INTO answer_log (asked_at, question, audiences, path_taken,
                                   chunk_ids, generation_model, check_failed)
           VALUES ('2026-01-01T00:00:00+00:00','Asked before the column.',
                   '[]','extract','[]','','')""")
    legacy.commit()
    legacy.close()

    migrated = SQLiteKnowledgeRepository(path)
    try:
        # The old row survives, and says what is true of it.
        assert migrated.answer_log()[0].source == "unknown"
        # And the next write no longer fails.
        migrated.log_answer(AnswerLogEntry(question="After.", path_taken="refuse",
                                           source="web"))
        assert migrated.answer_log()[0].source == "web"
    finally:
        migrated.close()
