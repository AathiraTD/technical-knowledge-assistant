"""Logging, held to the same standard as everything else it watches.

Observability code is the easiest thing in a system to leave untested, because
nothing visibly breaks when it is wrong. The failures are real, though, and all
three are tested here: a log line that carries the customer's question is a data
retention problem, a correlation id that leaks between threads makes concurrent
traces unreadable, and a logging call that raises turns an answer that was ready
to print into an exception.

The last one is why `event()` swallows. That swallow is load-bearing rather than
lazy, so it is pinned by a test that makes the logger itself fail.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.infrastructure import observability as obs


@pytest.fixture
def captured():
    """A configured logger writing JSON lines into a buffer."""
    stream = io.StringIO()
    obs.configure(stream)
    try:
        yield stream
    finally:
        for handler in list(obs.logger.handlers):
            if getattr(handler, "_assistant_handler", False):
                obs.logger.removeHandler(handler)


def lines(stream) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


# ------------------------------------------------------------ the correlation id


def test_an_id_is_generated_when_none_is_supplied():
    with obs.correlation() as cid:
        assert cid and cid == obs.correlation_id()


def test_a_caller_with_its_own_request_id_keeps_it():
    """A web request already has a trace; starting a second one loses the join."""
    with obs.correlation("req-abc123") as cid:
        assert cid == "req-abc123"
        assert obs.correlation_id() == "req-abc123"


def test_the_id_is_unbound_afterwards():
    with obs.correlation("x"):
        pass
    assert obs.correlation_id() == ""


def test_the_id_is_unbound_even_when_the_answer_raises():
    """Otherwise a failed answer leaves its id on a thread that serves the next."""
    with pytest.raises(ValueError):
        with obs.correlation("doomed"):
            raise ValueError("generation failed")

    assert obs.correlation_id() == ""


def test_nested_ids_restore_the_outer_one_rather_than_clearing_it():
    with obs.correlation("outer"):
        with obs.correlation("inner"):
            assert obs.correlation_id() == "inner"
        assert obs.correlation_id() == "outer"


def test_two_threads_do_not_see_each_others_id():
    """The web UI serves on a ThreadingHTTPServer; this is why it is a ContextVar."""
    seen: dict[str, str] = {}
    started = threading.Barrier(2)

    def answer(name: str) -> None:
        with obs.correlation(name):
            started.wait(timeout=5)        # both ids bound at the same moment
            seen[name] = obs.correlation_id()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(answer, ["first", "second"]))

    assert seen == {"first": "first", "second": "second"}


def test_generated_ids_are_distinct():
    assert len({obs.new_id() for _ in range(200)}) == 200


# ------------------------------------------------------------------- privacy


def test_a_question_is_fingerprinted_rather_than_recorded():
    question = "My bedroom wall at 14 Example Street is damp, what do I use?"
    printed = obs.fingerprint(question)

    assert "Example Street" not in printed
    assert len(printed) == 12


def test_the_same_question_fingerprints_the_same_way():
    """Otherwise a repeated question cannot be recognised as one."""
    assert obs.fingerprint("How much water?") == obs.fingerprint(
        "  how much WATER?  ")


def test_different_questions_fingerprint_differently():
    assert obs.fingerprint("How much water") != obs.fingerprint("How much sand")


# -------------------------------------------------------------- what is emitted


def test_an_event_carries_its_fields_and_its_id(captured):
    with obs.correlation("trace-1"):
        obs.event("retrieval", hits=5, top_score=0.81)

    record = lines(captured)[0]
    assert record["event"] == "retrieval"
    assert record["correlation_id"] == "trace-1"
    assert record["hits"] == 5
    assert record["top_score"] == 0.81
    assert record["level"] == "info"
    assert record["at"]


def test_an_event_outside_an_answer_omits_the_id_rather_than_inventing_one(captured):
    obs.event("snapshot_published", chunks=553)

    record = lines(captured)[0]
    assert "correlation_id" not in record
    assert record["chunks"] == 553


def test_a_field_named_like_a_log_record_attribute_does_not_raise(captured):
    """`logging` refuses to overwrite its own attributes, and raises to do it.

    A product name, a document name, a module name — `name=` is an obvious
    field to want. Passing it through `extra` raises `KeyError: "Attempt to
    overwrite 'name' in LogRecord"`, from inside a logging call on the
    answering path. So the event name is positional-only and reserved keys are
    filtered out rather than allowed to take the answer down with them.
    """
    with obs.correlation("t"):
        obs.event("route", name="Solo", module="answer", path="compose")

    record = lines(captured)[0]
    assert record["event"] == "route"
    assert record["path"] == "compose"
    assert record.get("name") != "Solo", "a reserved key reached the record"


def test_a_reserved_key_is_the_only_thing_dropped(captured):
    obs.event("route", message="not reserved", path="compose")

    record = lines(captured)[0]
    assert record["message"] == "not reserved"


def test_an_unserialisable_value_does_not_lose_the_line(captured):
    obs.event("route", slots={"substrate"}, hits=object())

    record = lines(captured)[0]
    assert record["event"] == "route"


def test_every_line_is_one_json_object(captured):
    obs.event("a", x=1)
    obs.event("b", y=2)

    assert [r["event"] for r in lines(captured)] == ["a", "b"]


# --------------------------------------------------------------- timing


def test_a_timed_block_reports_how_long_it_took(captured):
    with obs.timed("retrieval", top_k=5):
        pass

    record = lines(captured)[0]
    assert record["event"] == "retrieval"
    assert record["top_k"] == 5
    assert isinstance(record["seconds"], float)


def test_a_timed_block_can_add_what_it_learned_to_the_same_event(captured):
    with obs.timed("retrieval") as extra:
        extra["hits"] = 3

    assert lines(captured)[0]["hits"] == 3


def test_a_timed_block_that_raises_still_reports(captured):
    """A retrieval that failed is exactly the one worth having a timing for."""
    with pytest.raises(RuntimeError):
        with obs.timed("retrieval"):
            raise RuntimeError("store is gone")

    assert lines(captured)[0]["event"] == "retrieval"


# ------------------------------------------------- logging must never break an answer


def test_a_broken_logger_does_not_break_the_caller(monkeypatch):
    """The swallow in `event` is load-bearing, so it is pinned rather than trusted."""
    def explode(*_a, **_k):
        raise RuntimeError("the logging subsystem is misconfigured")

    monkeypatch.setattr(obs.logger, "info", explode)
    obs.event("route", path="compose")            # must not raise


def test_an_exception_is_formatted_into_the_line(captured):
    try:
        raise ValueError("connection refused")
    except ValueError:
        obs.logger.info("store_error", exc_info=True,
                        extra={"event": "store_error", "fields": {},
                               "correlation_id": ""})

    assert "connection refused" in lines(captured)[0]["error"]


# ---------------------------------------------------------------- configuration


def test_importing_the_library_prints_nothing(capsys):
    """A library that configures logging steals the decision from its host."""
    for handler in list(obs.logger.handlers):
        if getattr(handler, "_assistant_handler", False):
            obs.logger.removeHandler(handler)

    obs.event("route", path="compose")

    assert capsys.readouterr().err == ""
    assert any(isinstance(h, logging.NullHandler) for h in obs.logger.handlers)


def test_configuring_twice_does_not_double_every_line():
    """A reloading server calls this more than once."""
    first, second = io.StringIO(), io.StringIO()
    obs.configure(first)
    obs.configure(second)
    try:
        obs.event("route", path="compose")

        assert lines(first) == []
        assert len(lines(second)) == 1
    finally:
        for handler in list(obs.logger.handlers):
            if getattr(handler, "_assistant_handler", False):
                obs.logger.removeHandler(handler)


def test_configure_returns_the_logger_and_does_not_reach_the_root(captured):
    assert obs.logger.propagate is False
    assert obs.logger.level == logging.INFO
