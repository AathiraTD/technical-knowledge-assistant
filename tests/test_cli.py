"""The canonical interface, driven the way a person and the harness drive it.

Every other test in this suite calls the library directly, which is the right
shape for the guardrails and the exact reason this file has to exist. The
command line is what `README.md` marks built and verified, what the transcript
is produced with, and what the evaluation harness runs through — so it carries
risks that belong to no other module: an index that cannot be queried has to be
reported rather than traced back, a one-shot question has to carry its outcome
out in an exit code, and the session loop has to stop when it is told to.

`main()` is called with real argv lists against a real SQLite store in a
temporary directory, because the store is the thing the CLI is wiring up and a
mocked one would prove nothing about the wiring. It is never called against
Ollama: the embedding call is answered locally and generation is replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import cli, observability as obs, ollama           # noqa: E402
from assistant.indexing.index import CHUNKING_VERSION                      # noqa: E402
from assistant.model import (                                     # noqa: E402
    Caveat, Chunk, Document, DocumentVersion, Snapshot,
)
from assistant.store import SQLiteKnowledgeRepository             # noqa: E402

DIMS = 1024
SOLO = "https://example.invalid/solo"
SOLO_TEXT = ("Mix Solo with 5-6 litres of clean water per 25 kg sack and stir "
             "for three minutes.")
CONTACT = {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}


def unit(axis: int) -> list[float]:
    """A unit vector along one axis, so cosine similarity is predictable."""
    v = [0.0] * DIMS
    v[axis] = 1.0
    return v


def build_index(tmp_path, embedding_model: str = "") -> str:
    """One public datasheet with an active snapshot. Returns the --db path.

    `embedding_model` is a parameter because the index header is a safety
    control rather than bookkeeping: an index built by one model and queried by
    another returns confident nonsense, and the CLI has to stop on it.
    """
    documents = [Document(canonical_url=SOLO, title="Solo datasheet",
                          document_type="datasheet", authority=1, product="Solo",
                          link_text="Solo Datasheet")]
    versions = [DocumentVersion(canonical_url=SOLO, version=1, content_hash="h1",
                                source_path="cache/solo.pdf",
                                fetched_at="2026-01-01T00:00:00Z")]
    chunks = [Chunk(canonical_url=SOLO, version=1, chunk_index=0, section="Mixing",
                    content=SOLO_TEXT, product="Solo", document_type="datasheet",
                    authority=1, source_date="2024-07-01", embedding=unit(0))]
    snapshot = Snapshot(
        snapshot_id="snap-cli", created_at="2026-01-01T00:00:00Z",
        embedding_model=embedding_model or ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=1, chunk_count=1,
        notes={"products": ["Solo"], "colours": ["York"],
               "merchants": ["The Lime Centre"], "contact": CONTACT})

    path = tmp_path / "index" / "knowledge.db"
    repo = SQLiteKnowledgeRepository(path)
    repo.publish(documents, versions, chunks, snapshot,
                 [Caveat(SOLO, "temperature", "Do not apply below 5 degrees C.",
                         "Mixing")])
    repo.close()          # the CLI opens its own connection, as it would live
    return str(path)


@pytest.fixture
def no_ollama(monkeypatch):
    """Answer the embedding call locally; generation returns a citing sentence."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(
        ollama, "generate",
        lambda *_a, **_k: (f"{SOLO_TEXT} [1]", 0.01))


@pytest.fixture
def db(tmp_path, no_ollama) -> str:
    return build_index(tmp_path)


@pytest.fixture(autouse=True)
def detach_the_log():
    """Take the handler back off after every test.

    `configure()` binds whichever stream it was handed, and under capsys that
    stream belongs to one test. A handler left attached writes into a buffer
    that no longer exists, which turns this file's own logging into noise in
    somebody else's failure.
    """
    yield
    for handler in list(obs.logger.handlers):
        if getattr(handler, "_assistant_handler", False):
            obs.logger.removeHandler(handler)


def log_lines(text: str) -> list[dict]:
    """The JSON events in a captured stderr stream."""
    import json
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def session(monkeypatch, *lines: str) -> None:
    """Type these lines at the prompt, then end the input as a closed pipe does.

    Ending with EOF rather than a sentinel is deliberate: `python -m
    assistant.cli < questions.txt` is how the session is scripted, and it must
    exit rather than spin on an exhausted stdin.
    """
    queue = list(lines)

    def typed(_prompt: str = "") -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", typed)


# ---------------------------------------------------------- one question in


def test_one_question_prints_the_passage_its_source_and_exits_zero(db, capsys):
    """The brief's first test type, through the surface the transcript uses."""
    code = cli.main(["--db", db, "-q", "How much water does Solo need per bag"])
    out = capsys.readouterr().out

    assert code == 0
    assert "5-6 litres" in out, out
    assert "Sources:" in out
    assert SOLO in out, "the citation has to carry a resolvable link"


def test_a_question_the_material_does_not_answer_refuses_and_still_hands_over(
        db, capsys):
    """A refusal is a successful run: the exit code reports the CLI, not the answer."""
    code = cli.main(["--db", db, "-q", "What is the pot life of Duro"])
    out = capsys.readouterr().out

    assert code == 0, "a refusal is not a CLI failure"
    assert "does not state pot life" in out, out
    assert CONTACT["phone"] in out, "a refusal that hands over has to say where to"


def test_the_verbose_flag_prints_the_route_that_produced_the_answer(db, capsys):
    """Diagnostics are the audit surface; -v is the only way to see them."""
    code = cli.main(["--db", db, "-v", "-q", "How much water does Solo need per bag"])
    out = capsys.readouterr().out

    assert code == 0
    assert "path: extract" in out, out
    assert "step 7" in out, "the router step is what makes the path explicable"


def test_the_threshold_override_reaches_retrieval_rather_than_being_parsed_and_lost(
        db, capsys):
    """-t is a sweep control. A flag that parses and does nothing is worse than none."""
    # The fixture's only passage scores exactly 1.0 against the query vector, so
    # a threshold above 1.0 is the one value that can prove the flag arrived.
    code = cli.main(["--db", db, "-t", "1.01", "-v",
                     "-q", "How much water does Solo need per bag"])
    out = capsys.readouterr().out

    assert code == 0
    assert "step 1" in out, out
    assert CONTACT["phone"] in out, "a below-threshold refusal still hands over"


# ------------------------------------------------------------- the session


def test_the_banner_states_the_index_the_models_and_that_audience_is_asserted(
        db, monkeypatch, capsys):
    """The honesty line is load-bearing: the audience flag is not authentication."""
    session(monkeypatch)                      # no input at all: straight to EOF
    code = cli.main(["--db", db])
    out = capsys.readouterr().out

    assert code == 0
    assert "1 documents, 1 chunks, built 2026-01-01" in out, out
    assert ollama.EMBED_MODEL in out and ollama.GENERATION_MODEL in out
    assert "asserted at the command line, not authenticated" in out


def test_an_empty_audience_entry_is_dropped_rather_than_passed_to_retrieval(
        db, monkeypatch, capsys):
    """`-a public,` is a plausible typo; an empty audience must not reach the query."""
    session(monkeypatch)
    cli.main(["--db", db, "-a", "public,"])
    out = capsys.readouterr().out

    # The banner prints the parsed set, so a surviving empty entry shows up as a
    # trailing separator rather than having to be reached for through a mock.
    assert "audience   public  (asserted" in out, out


def test_a_blank_line_is_ignored_rather_than_asked(db, monkeypatch, capsys):
    """Return on an empty prompt is a slip, not a question; it must cost nothing."""
    session(monkeypatch, "", "   ")
    code = cli.main(["--db", db])
    out = capsys.readouterr().out

    assert code == 0
    assert "Sources:" not in out, "a blank line reached retrieval"


@pytest.mark.parametrize("word", ["quit", "exit", "q", "QUIT", " Exit "])
def test_the_words_that_end_a_session_all_end_it(db, monkeypatch, capsys, word):
    """Advertised in the banner as 'quit'; the other two and any case must work too."""
    session(monkeypatch, word, "How much water does Solo need per bag")
    code = cli.main(["--db", db])
    out = capsys.readouterr().out

    assert code == 0
    assert "5-6 litres" not in out, "the session answered a question after quit"


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_ctrl_c_and_a_closed_stdin_both_end_the_session_cleanly(
        db, monkeypatch, capsys, interruption):
    """Ctrl-C at a prompt is how a demonstration ends. It must not print a traceback."""
    def interrupted(_prompt: str = "") -> str:
        raise interruption

    monkeypatch.setattr("builtins.input", interrupted)
    code = cli.main(["--db", db])

    assert code == 0
    assert "Traceback" not in capsys.readouterr().err


def test_a_question_answered_in_the_session_prints_the_same_answer_as_one_shot(
        db, monkeypatch, capsys):
    """The loop and the -q path must not drift into two different behaviours."""
    session(monkeypatch, "How much water does Solo need per bag")
    code = cli.main(["--db", db])
    out = capsys.readouterr().out

    assert code == 0
    assert "5-6 litres" in out and "Sources:" in out


def test_a_trailing_dash_v_turns_on_diagnostics_for_that_question_only(
        db, monkeypatch, capsys):
    """The banner advertises it. The suffix must be stripped before it is asked."""
    session(monkeypatch,
            "How much water does Solo need per bag -v",
            "How much water does Solo need per bag")
    cli.main(["--db", db])
    out = capsys.readouterr().out

    assert out.count("path: extract") == 1, "verbose leaked into the next question"
    # The flag must not travel into the question: a question ending in "-v"
    # would take a different retrieval path from the same question without it.
    assert out.count("5-6 litres") == 2, out


# --------------------------------------------------------- the failure paths


def test_a_missing_index_is_reported_with_the_command_that_builds_one(
        tmp_path, no_ollama, capsys):
    """The first thing a clean clone hits. It must say what to run, not traceback."""
    code = cli.main(["--db", str(tmp_path / "nothing" / "knowledge.db"),
                     "-q", "How much water does Solo need per bag"])
    err = capsys.readouterr().err

    assert code == 1
    assert "No active index" in err, err
    assert "python -m assistant.indexing.index" in err
    assert "Traceback" not in err


def test_an_index_built_by_another_embedding_model_is_refused_by_name(
        tmp_path, no_ollama, capsys):
    """Querying an index with the wrong model returns confident nonsense, silently."""
    path = build_index(tmp_path, embedding_model="some-other-embedding:1b")
    code = cli.main(["--db", path, "-q", "How much water does Solo need per bag"])
    err = capsys.readouterr().err

    assert code == 1
    assert "some-other-embedding:1b" in err, err
    assert ollama.EMBED_MODEL in err, "the operator needs both names to act on this"


def test_an_unreachable_model_is_reported_rather_than_traced_back(
        db, monkeypatch, capsys):
    """`ollama serve` not running is the likeliest demonstration failure of all."""
    def unavailable(*_a, **_k):
        raise ollama.OllamaUnavailable("Ollama is not answering. Run `ollama serve`.")

    monkeypatch.setattr(ollama, "embed_one", unavailable)
    code = cli.main(["--db", db, "-q", "How much water does Solo need per bag"])
    captured = capsys.readouterr()

    assert code == 1
    assert "ollama serve" in captured.err
    assert "Traceback" not in captured.err


def test_an_unreachable_model_does_not_end_the_session(db, monkeypatch, capsys):
    """A model restarted mid-session should cost the question, not the session."""
    def unavailable(*_a, **_k):
        raise ollama.OllamaUnavailable("Ollama is not answering.")

    monkeypatch.setattr(ollama, "embed_one", unavailable)
    session(monkeypatch, "How much water does Solo need per bag")
    code = cli.main(["--db", db])

    assert code == 0, "the session ended on a recoverable error"
    assert "Ollama is not answering." in capsys.readouterr().err


def test_an_unparseable_argument_exits_rather_than_answering(db):
    """argparse owns this; the test pins that the CLI does not swallow it."""
    with pytest.raises(SystemExit) as exit_code:
        cli.main(["--db", db, "--not-an-option"])
    assert exit_code.value.code == 2


# ------------------------------------------------------------ the log stream


def test_nothing_is_logged_unless_the_log_is_asked_for(db, capsys):
    """Off by default. A library that logs on import steals the decision."""
    cli.main(["--db", db, "-q", "How much water does Solo need per bag"])
    err = capsys.readouterr().err

    assert err == "", err


def test_the_log_goes_to_stderr_and_leaves_the_transcript_byte_identical(db, capsys):
    """The harness parses stdout. A log line inside an answer corrupts the artefact."""
    question = "How much water does Solo need per bag"

    cli.main(["--db", db, "-q", question])
    quiet = capsys.readouterr().out

    cli.main(["--db", db, "--log", "-q", question])
    logged = capsys.readouterr()

    assert logged.out == quiet, "the transcript changed when logging was turned on"
    events = log_lines(logged.err)
    assert events, logged.err
    assert {"retrieval", "answer"} <= {e["event"] for e in events}, events


def test_every_line_of_one_answer_carries_the_same_correlation_id(db, capsys):
    """A question splits into parts; the id is the only thing that rejoins them."""
    cli.main(["--db", db, "--log",
              "-q", "How much does Solo cost and how much water does it need"])
    events = log_lines(capsys.readouterr().err)

    ids = {e.get("correlation_id") for e in events}
    assert len(events) > 2, events
    assert len(ids) == 1 and "" not in ids, ids


def test_the_question_itself_is_never_written_to_the_log(db, capsys):
    """CLAUDE.md forbids logging complete customer conversations."""
    question = "How much water does Solo need per bag for Mrs Attwood at number 14"
    cli.main(["--db", db, "--log", "-q", question])
    err = capsys.readouterr().err

    assert "Attwood" not in err, err
    assert "number 14" not in err
    # The fingerprint is what stands in for it, so the line is still useful.
    assert any("question" in e for e in log_lines(err))
