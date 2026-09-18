"""The span stack: parenting, isolation, privacy, and never failing an answer.

`assistant/observability.py` could already say *what* happened. What it could
not say was *inside what, and for how long* — durations existed only where
somebody had remembered to wrap a block, and nothing tied the dozen lines a
two-topic question emits into a tree. `span()` adds that, in OpenTelemetry's
shape and without its SDK.

Four properties are worth a test rather than a docstring, and each of them has a
way of being quietly wrong:

* **Parenting.** A span tree that parents its stages incorrectly is worse than
  no tree, because it reads as a system that did something it did not do.
* **Isolation.** The web UI is a `ThreadingHTTPServer`. Two questions answered
  at once must not adopt each other's parent, and the failure would only appear
  under concurrency — the condition least likely to be reproduced by hand.
* **Reset on the exception path.** A thread that raised halfway through an
  answer must not serve the next caller from inside the dead span, and a
  `finally` that is missing looks exactly like one that is present until
  something throws.
* **Privacy.** No span may carry question text, answer text or passage text.
  That rule is enforced here rather than by scrubbing at the boundary, because
  a scrubber would silently delete the evidence of the call site that broke it.

The persistence half — `record_spans` and `traces` on both adapters, the
additive migration and retention — is below, and the adapter-parity rows are in
`tests/test_repository_contract.py`, where every adapter runs them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant import observability as obs                         # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.model import TraceSpan                              # noqa: E402
from assistant.repository import (                                 # noqa: E402
    TRACE_PRUNE_STRIDE, TRACE_RETENTION_DAYS, TRACE_ROW_CAP,
)
from assistant.store import SQLiteKnowledgeRepository              # noqa: E402

from test_engine import build_repo, no_ollama, quoting             # noqa: E402,F401
from assistant import ollama                                       # noqa: E402


def now(offset_days: int = 0) -> str:
    return (datetime.now(timezone.utc)
            + timedelta(days=offset_days)).isoformat(timespec="seconds")


def span(name: str, trace: str = "t", **kwargs) -> TraceSpan:
    fields = {"trace_id": trace, "span_id": name, "name": name,
              "started_at": now(), "duration_ms": 1, "turn_id": "u"}
    fields.update(kwargs)
    return TraceSpan(**fields)


# ------------------------------------------------------------------ the stack


def test_a_span_emits_once_on_completion_carrying_its_ids():
    """One event per span, and nothing on entry.

    A start line would double the stream to say what the completion line
    already implies, and an incomplete span is more legible as a hole in the
    tree than as a line with no end.
    """
    with obs.correlation("trace-1"), obs.turn(turn="turn-1", session="sess-1",
                                              source="web") as collected:
        with obs.span("answer", parts=1):
            assert collected == [], "a span must emit nothing on entry"

    assert len(collected) == 1
    record = collected[0]
    assert record.name == "answer"
    assert record.trace_id == "trace-1"
    assert record.turn_id == "turn-1"
    assert record.session_id == "sess-1"
    assert record.source == "web"
    assert record.parent_span_id == ""
    assert record.status == "ok"
    assert record.attributes == {"parts": 1}
    assert isinstance(record.duration_ms, int)


def test_a_nested_span_is_parented_by_the_one_it_runs_inside():
    with obs.correlation("t"), obs.turn() as collected:
        with obs.span("answer"):
            with obs.span("part"):
                with obs.span("retrieval"):
                    pass
            with obs.span("render"):
                pass

    by_name = {s.name: s for s in collected}
    assert by_name["answer"].parent_span_id == ""
    assert by_name["part"].parent_span_id == by_name["answer"].span_id
    assert by_name["retrieval"].parent_span_id == by_name["part"].span_id
    # `render` is a sibling of `part`, not a child of it: the stack has to pop.
    assert by_name["render"].parent_span_id == by_name["answer"].span_id
    # Completion order, so a tree is rebuilt from parents and never from order.
    assert [s.name for s in collected] == ["retrieval", "part", "render", "answer"]


def test_a_sibling_after_a_failed_span_is_not_parented_by_it():
    """The reset is in a `finally`, so an exception cannot leave a dead parent.

    Without it the next stage in the same answer — and, on the threading
    server, the next *caller* served by that thread — would hang off a span
    that had already ended.
    """
    with obs.correlation("t"), obs.turn() as collected:
        with obs.span("answer"):
            with pytest.raises(ValueError):
                with obs.span("retrieval"):
                    raise ValueError("the store was unreachable")
            with obs.span("render"):
                assert obs.span_id() != ""

    by_name = {s.name: s for s in collected}
    assert by_name["retrieval"].status == "error"
    # The type, never the message: an exception string can quote the statement
    # it failed on, and that statement can contain a question.
    assert by_name["retrieval"].attributes["error"] == "ValueError"
    assert "unreachable" not in json.dumps(by_name["retrieval"].attributes)
    assert by_name["render"].parent_span_id == by_name["answer"].span_id
    # And the stack is empty again outside the turn.
    assert obs.span_id() == ""


def test_an_exception_is_re_raised_untouched():
    """Observability records that a stage failed; it does not decide anything."""
    marker = KeyError("substrate")
    with pytest.raises(KeyError) as raised:
        with obs.span("slot_detection"):
            raise marker
    assert raised.value is marker


def test_two_concurrent_threads_cannot_adopt_each_other_s_parent():
    """The reason the stack is a `ContextVar` and not a module-level list.

    The web UI is a `ThreadingHTTPServer`, so two questions are answered in two
    threads at once. A shared stack would interleave them into one tree
    belonging to nobody, and the defect would appear only under concurrency —
    the condition least likely to be reproduced by hand.

    Both threads are held at the same point by a barrier before either opens
    its child span, so the interleaving this test is about is forced rather
    than hoped for.
    """
    barrier = threading.Barrier(2)
    results: dict[str, list] = {}

    def answer(label: str) -> None:
        with obs.correlation(f"trace-{label}"), obs.turn(session=label) as got:
            with obs.span("answer", who=label):
                barrier.wait(timeout=5)
                with obs.span("part", who=label):
                    barrier.wait(timeout=5)
        results[label] = got

    threads = [threading.Thread(target=answer, args=(label,))
               for label in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert set(results) == {"a", "b"}
    for label, collected in results.items():
        by_name = {s.name: s for s in collected}
        assert by_name["part"].parent_span_id == by_name["answer"].span_id
        assert by_name["answer"].parent_span_id == ""
        # And nothing leaked across: every span in this thread's turn carries
        # this thread's trace, turn and session.
        assert {s.trace_id for s in collected} == {f"trace-{label}"}
        assert {s.session_id for s in collected} == {label}
        assert len({s.turn_id for s in collected}) == 1


def test_a_turn_inside_a_turn_starts_its_own_root():
    """A nested turn must not hang off the outer turn's current span.

    Nothing in the engine nests turns today. The reset is there because a
    context reused by a thread pool would otherwise inherit a parent from an
    answer that had already finished, which is a tree hanging off a dead span.
    """
    with obs.correlation("t"), obs.turn() as outer:
        with obs.span("answer"):
            with obs.turn(turn="inner") as inner:
                with obs.span("answer"):
                    pass
    assert inner[0].parent_span_id == ""
    assert inner[0].turn_id == "inner"
    assert outer[0].turn_id != "inner"


def test_collection_is_opt_in_and_emission_is_not():
    """A caller that never opens a turn still gets the log line and no rows.

    That split is what keeps `span()` ignorant of the store: a library embedder
    or a test wants the stream without a database, and the engine wants both.
    """
    stream = StringIO()
    obs.configure(stream, level=logging.INFO)
    try:
        with obs.correlation("t"):
            with obs.span("retrieval", returned=5):
                pass
    finally:
        for handler in list(obs.logger.handlers):
            if getattr(handler, "_assistant_handler", False):
                obs.logger.removeHandler(handler)

    line = json.loads(stream.getvalue().strip())
    assert line["event"] == "retrieval"
    assert line["returned"] == 5
    assert line["trace_id"] == "t"
    assert line["span_id"] and "parent_span_id" not in line
    assert line["status"] == "ok"
    assert isinstance(line["duration_ms"], int)


def test_the_existing_api_still_works_unchanged():
    """`event`, `timed` and `correlation` are untouched; this slice is additive.

    Every call site in the repository that did not become a span still emits
    what it always emitted, and a `timed()` block outside a turn writes no rows.
    """
    with obs.correlation("t") as cid:
        assert obs.correlation_id() == cid == obs.trace_id()
        with obs.timed("indexing", documents=3) as record:
            record["chunks"] = 9
        obs.event("snapshot_published", chunks=553)   # must not raise


# ------------------------------------------------------------------- privacy


def test_no_span_carries_question_answer_or_passage_text(tmp_path, monkeypatch,
                                                         no_ollama):
    """The privacy rule of the review §2.5, enforced over a real answer.

    Not a review of the call sites by eye: the whole engine is run against a
    real store with a model that quotes its passages back, and every attribute
    of every span produced is searched for the distinctive words of the
    question, of the passage and of the generated answer. A future call site
    that adds `question=part` rather than a fingerprint fails here.

    `answer_log.question` remains the single deliberate retention point, and
    the last assertion is that it still holds the text this one forbids — so
    the rule reads as "the trace does not duplicate it" rather than as "nothing
    is kept anywhere".
    """
    monkeypatch.setattr(ollama, "generate", quoting)
    repo = build_repo(tmp_path, two_documents=True)
    try:
        assistant = Assistant(repo, source="test")
        question = "How much water does Solo need per sack, and what coverage?"
        reply = assistant.ask(question, session_id="sess-privacy")
        spans = repo.traces(session_id="sess-privacy")
        assert spans, "the answer produced no trace"

        blob = json.dumps([s.attributes for s in spans]).lower()
        # `gypsum` is the shape that caught a real leak: `_missed_evidence`
        # lifts a rare word straight out of the question and hands it to the
        # targeted retrieval, and the span there named its terms until this
        # test was written. A word from the question is question text.
        gypsum = assistant.ask("Can I use Solo over old gypsum plaster?",
                               session_id="sess-privacy")
        blob += json.dumps([s.attributes for s in
                            repo.traces(trace_id=gypsum.correlation_id)]).lower()
        forbidden = [
            "gypsum",
            # The question, word by word, in the words a caller would search.
            "water", "sack", "coverage",
            # The passage text, from the fixture datasheets.
            "litres", "stir for three minutes", "25 kg",
            # The generated answer.
            *[w for w in reply.parts[0][1].text.lower().split() if len(w) > 6],
        ]
        leaked = [term for term in forbidden if term in blob]
        assert not leaked, f"spans carried text they must not: {leaked}"

        # Names, ids, counts and durations are all that is left, and the span
        # names themselves are a fixed vocabulary rather than caller input.
        assert {s.name for s in spans} >= {"answer", "part", "route", "render"}

        # The deliberate retention point is untouched: the question is still in
        # the audit table, which is the one place it was ever meant to be. It
        # is stored per part, which is why this compares words rather than the
        # whole sentence — the audit row records the topic that was answered.
        logged = " ".join(e.question for e in repo.answer_log()).lower()
        assert "water" in logged, "the audit table lost the question it retains"
        assert "water" not in json.dumps([s.attributes for s in spans]).lower()
    finally:
        repo.close()


# -------------------------------------------------------- the engine's tree


def test_one_question_produces_the_span_tree_of_the_review(tmp_path, monkeypatch,
                                                           no_ollama):
    """The shape of §2.3, asserted against a real answer rather than drawn."""
    monkeypatch.setattr(ollama, "generate", quoting)
    repo = build_repo(tmp_path, two_documents=True)
    try:
        assistant = Assistant(repo, source="test")
        reply = assistant.ask("What coverage does Duro give per bag?",
                              session_id="sess-tree")
        spans = repo.traces(trace_id=reply.correlation_id)
        by_id = {s.span_id: s for s in spans}
        names = {s.name for s in spans}

        assert {"answer", "split_by_topic", "part", "cache_lookup",
                "slot_detection", "retrieval", "embed_question", "search",
                "route", "render"} <= names

        def parent_of(name: str) -> str:
            span_ = next(s for s in spans if s.name == name)
            return by_id[span_.parent_span_id].name if span_.parent_span_id else ""

        assert parent_of("answer") == ""
        assert parent_of("split_by_topic") == "answer"
        assert parent_of("part") == "answer"
        for child in ("cache_lookup", "slot_detection", "retrieval", "route",
                      "render"):
            assert parent_of(child) == "part", child
        for child in ("embed_question", "search"):
            assert parent_of(child) == "retrieval", child

        # Every span in the turn shares one trace, one turn and one session, and
        # names the surface that asked.
        assert {s.trace_id for s in spans} == {reply.correlation_id}
        assert {s.session_id for s in spans} == {"sess-tree"}
        assert {s.source for s in spans} == {"test"}
        assert len({s.turn_id for s in spans}) == 1

        # OTel's names where a convention exists, so an exporter is a shim.
        embedding = next(s for s in spans if s.name == "embed_question")
        assert embedding.attributes["gen_ai.request.model"] == ollama.EMBED_MODEL
        search = next(s for s in spans if s.name == "search")
        assert search.attributes["db.system"] == "sqlite"
    finally:
        repo.close()


def test_a_cli_caller_writes_an_empty_session_id(tmp_path, monkeypatch, no_ollama):
    """Empty is a fact about the CLI, not a gap in the trace (review §8.3).

    The CLI constructs no session and carries nothing between questions, so
    consecutive CLI turns are unrelated by construction. A synthetic id would
    group them into a "conversation" sharing nothing but a terminal, which is a
    worse answer than none because it is wrong rather than empty.
    """
    monkeypatch.setattr(ollama, "generate", quoting)
    repo = build_repo(tmp_path)
    try:
        assistant = Assistant(repo, source="cli")
        reply = assistant.ask("What coverage does Solo give?")
        spans = repo.traces(trace_id=reply.correlation_id)
        assert spans and {s.session_id for s in spans} == {""}
        # And the empty rows stay out of the conversation index, which is why
        # it is partial: a session replay never asks for them.
        assert repo.traces(session_id="") == repo.traces()
    finally:
        repo.close()


def test_a_failing_record_spans_does_not_fail_the_answer(tmp_path):
    """Observability degrades observability and nothing else.

    Asserted at `_record_spans` rather than through a whole answer, because the
    claim is about what the engine does with a store that raises, and standing
    up an index to reach it would be testing the indexer. Built with
    `__new__` for the same reason `tests/test_answer_log_source.py` does.
    """
    events: list[tuple] = []
    engine = Assistant.__new__(Assistant)
    engine.log = True

    class Hostile:
        def record_spans(self, spans):
            raise RuntimeError("disk full")

    engine.repo = Hostile()
    obs_event = obs.event
    try:
        obs.event = lambda name, /, **f: events.append((name, f))
        engine._record_spans([span("answer")])          # must not raise
    finally:
        obs.event = obs_event

    assert events and events[0][0] == "store_error"
    assert events[0][1]["operation"] == "record_spans"
    assert events[0][1]["error"] == "RuntimeError"


def test_a_repository_without_record_spans_still_answers(tmp_path):
    """The engine is built against a Protocol, and doubles predate this method."""
    engine = Assistant.__new__(Assistant)
    engine.log = True
    engine.repo = SimpleNamespace()
    engine._record_spans([span("answer")])              # must not raise


def test_recording_follows_the_logging_switch(tmp_path):
    """One switch, not two: a caller that silenced recording silenced it."""
    written: list = []
    engine = Assistant.__new__(Assistant)
    engine.log = False
    engine.repo = SimpleNamespace(record_spans=written.append)
    engine._record_spans([span("answer")])
    assert written == []


# --------------------------------------------------------------- persistence


@pytest.fixture
def repo(tmp_path):
    store = SQLiteKnowledgeRepository(tmp_path / "traces.db")
    yield store
    store.close()


def test_spans_round_trip_oldest_first(repo):
    repo.record_spans([
        span("answer", span_id="s1", started_at=now(-1)),
        span("part", span_id="s2", parent_span_id="s1", started_at=now()),
    ])
    got = repo.traces(trace_id="t")
    assert [s.name for s in got] == ["answer", "part"]
    assert got[1].parent_span_id == "s1"


def test_an_empty_batch_writes_nothing(repo):
    """A route or a policy hit still opens a turn; not every turn has spans."""
    repo.record_spans([])
    assert repo.traces() == []


def test_the_two_access_paths_are_the_two_filters(repo):
    repo.record_spans([span("answer", trace="t1", session_id="s", turn_id="u1"),
                       span("answer", span_id="b", trace="t2", session_id="s",
                            turn_id="u2"),
                       span("answer", span_id="c", trace="t3")])
    assert len(repo.traces(trace_id="t1")) == 1
    assert len(repo.traces(session_id="s")) == 2
    assert len(repo.traces()) == 3


def test_a_store_written_before_the_table_existed_gains_it(tmp_path):
    """Additive on a database that predates `turn_traces`.

    The table itself arrives with the schema script, which runs on every open —
    `CREATE TABLE IF NOT EXISTS` creates what is missing. What it does *not* do
    is reshape a table that already exists, which is the `answer_log.source`
    lesson, so the column migration is checked by dropping a column too. The
    `DROP COLUMN` is also what pins the comment placement: SQLite reparses the
    stored DDL to do it, and a comment trailing the last column would leave it
    with "incomplete input".
    """
    path = tmp_path / "old.db"
    SQLiteKnowledgeRepository(path).close()

    legacy = sqlite3.connect(path)
    legacy.execute("ALTER TABLE turn_traces DROP COLUMN source")
    legacy.execute(
        """INSERT INTO turn_traces (session_id, turn_id, trace_id, span_id,
                                    parent_span_id, name, started_at,
                                    duration_ms, status, attributes)
           VALUES ('','u','t','s','','answer','2026-01-01T00:00:00+00:00',
                   3,'ok','{}')""")
    legacy.commit()
    legacy.close()

    migrated = SQLiteKnowledgeRepository(path)
    try:
        # The old row survives and says what is true of it: nobody recorded
        # which surface asked.
        assert migrated.traces()[0].source == "unknown"
        # And the next write no longer fails.
        migrated.record_spans([span("part", span_id="s2", source="web")])
        assert migrated.traces()[-1].source == "web"
    finally:
        migrated.close()


def test_a_table_dropped_entirely_is_recreated(tmp_path):
    path = tmp_path / "gone.db"
    first = SQLiteKnowledgeRepository(path)
    first.close()
    legacy = sqlite3.connect(path)
    legacy.execute("DROP TABLE turn_traces")
    legacy.commit()
    legacy.close()

    store = SQLiteKnowledgeRepository(path)
    try:
        store.record_spans([span("answer")])
        assert len(store.traces()) == 1
    finally:
        store.close()


# ---------------------------------------------------------------- retention


def test_the_window_prunes(repo):
    """Fourteen days is the policy, and the policy has to actually delete."""
    repo.trace_prune_stride = 1
    repo.record_spans([span("answer", span_id="old",
                            started_at=now(-TRACE_RETENTION_DAYS - 1))])
    # The first batch is written before the sweep runs, so the stale row is
    # still there; the second write is what sweeps it.
    repo.record_spans([span("part", span_id="new")])
    kept = [s.span_id for s in repo.traces()]
    assert "old" not in kept
    assert "new" in kept


def test_a_row_inside_the_window_survives(repo):
    repo.trace_prune_stride = 1
    repo.record_spans([span("answer", span_id="recent",
                            started_at=now(-TRACE_RETENTION_DAYS + 1))])
    repo.record_spans([span("part", span_id="new")])
    assert {s.span_id for s in repo.traces()} == {"recent", "new"}


def test_the_cap_prunes_a_burst_the_window_cannot_see(repo):
    """The backstop: everything is inside the window and there is still too much.

    A load test or a retry loop is exactly this shape, and it is the case a
    window alone cannot bound — which is why the review settles on both.
    """
    repo.trace_prune_stride = 1
    repo.trace_row_cap = 5
    for i in range(10):
        repo.record_spans([span("part", span_id=f"s{i}")])
    kept = [s.span_id for s in repo.traces()]
    assert len(kept) == 5
    # The newest survive, which is the half of the table worth keeping.
    assert kept == ["s5", "s6", "s7", "s8", "s9"]


def test_the_prune_is_amortised_over_the_stride(repo):
    """The delete stays off the answering path.

    A `DELETE ... WHERE started_at < ?` over an indexed column costs nothing
    when it matches nothing, and is still a write per answered question if it
    runs every time. This is the cost the review accepts in exchange: a burst
    can overshoot the cap between strides.
    """
    assert repo.trace_prune_stride == TRACE_PRUNE_STRIDE > 1
    repo.trace_row_cap = 2
    for i in range(5):
        repo.record_spans([span("part", span_id=f"s{i}",
                                started_at=now(-TRACE_RETENTION_DAYS - 1))])
    # Nothing has been swept, because the stride has not come round.
    assert len(repo.traces()) == 5


def test_the_retention_policy_is_one_definition_for_both_adapters(repo):
    """Shared constants, so the window cannot differ between the two paths."""
    assert (repo.trace_retention_days, repo.trace_row_cap) == (
        TRACE_RETENTION_DAYS, TRACE_ROW_CAP)
    assert (TRACE_RETENTION_DAYS, TRACE_ROW_CAP) == (14, 200_000)


def test_answer_log_is_deliberately_not_pruned(repo):
    """The departure, stated as a test so it reads as a decision.

    `answer_log` retains question text, so deleting from it is a data-retention
    decision with a privacy argument attached that nobody has taken. Bounding
    the trace and not the audit trail is the honest split; the audit table's own
    unbounded growth stays a stated open item.
    """
    from assistant.model import AnswerLogEntry
    repo.trace_prune_stride = 1
    repo.log_answer(AnswerLogEntry(question="Asked long ago.", path_taken="extract",
                                   asked_at="2020-01-01T00:00:00+00:00"))
    repo.record_spans([span("answer")])
    repo.record_spans([span("part", span_id="b")])
    assert len(repo.answer_log()) == 1


def test_the_ids_are_readable_from_inside_a_turn():
    """Accessors, so a caller can stamp an id onto something else.

    The engine uses them to put the trace and turn ids into an answer's
    diagnostics, so a diagnostics dump and a trace can be joined without the
    store — the same reason the correlation id was already carried there.
    """
    assert (obs.trace_id(), obs.turn_id(), obs.session_id()) == ("", "", "")
    with obs.correlation("t"), obs.turn(turn="u", session="s"):
        assert obs.trace_id() == "t"
        assert obs.turn_id() == "u"
        assert obs.session_id() == "s"
    assert obs.session_id() == ""


def test_a_store_that_cannot_write_a_span_degrades_observability_only(tmp_path,
                                                                      monkeypatch):
    """The adapter swallows, and says so, rather than raising into an answer.

    This is where the guarantee actually lives: the engine's own guard covers a
    repository that raises, and the adapter's covers the ordinary case of a
    database that has gone away mid-answer. A store that cannot write a timing
    has cost the operator a debugging aid and the caller nothing, which is the
    opposite of `log_answer`, where the audit row is a promise to somebody and
    a failure is reported by raising.
    """
    store = SQLiteKnowledgeRepository(tmp_path / "broken.db")
    events: list[tuple] = []
    monkeypatch.setattr(obs, "event",
                        lambda name, /, **f: events.append((name, f)))
    # Make the write fail the way a real one does — the file is gone.
    monkeypatch.setattr(store, "path", tmp_path / "no" / "such" / "dir" / "x.db")
    try:
        store.record_spans([span("answer")])            # must not raise
    finally:
        store.close()

    assert events and events[0][0] == "store_error"
    assert events[0][1]["operation"] == "record_spans"
