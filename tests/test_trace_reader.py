"""Reading a trace back: the tree, the headline evidence, and the awkward reads.

`turn_traces` was written for a fortnight before anything could read it, and the
cost of that gap was a product-drift bug debugged with `print()` calls in the
engine, the store and the page while the span tree already held the answer. So
the reader's job is narrow and worth pinning: turn a flat list of spans into the
tree it encodes, and put the attributes a wrong answer actually turns on —
which product retrieval was told about, what the top passage scored, which
router step fired — where somebody reading a terminal will see them.

Four things here have a way of being quietly wrong.

* **Parenting.** The same property `tests/test_spans.py` asserts on the writing
  side, restated on the reading side, because a reader that renders a correct
  tree incorrectly is indistinguishable from a system that did something else.
* **Orphans.** `traces()` takes a limit, and a limit slices a tree. A child
  whose parent was left behind must still print; dropping it would hide the
  slowest stage of a long turn precisely when somebody is looking for it.
* **The headline.** `product=` on the retrieval span is the whole reason this
  exists. If it renders below the fold with thirty other attributes, the reader
  has not solved the problem it was written for.
* **Nothing to show.** An id that has aged out of the fourteen-day window is the
  normal case for a bug reported late, and it must say so rather than print an
  empty string that reads as "the answer had no stages".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.infrastructure import trace as reader
from assistant.model import TraceSpan
from assistant.store.embedded import SQLiteKnowledgeRepository


def span(name, span_id, parent="", *, turn="t1", trace="tr1", ms=1,
         session="", source="web", **attributes) -> TraceSpan:
    return TraceSpan(
        trace_id=trace, span_id=span_id, parent_span_id=parent, name=name,
        started_at="2026-09-17T10:00:00+00:00", duration_ms=ms, turn_id=turn,
        session_id=session, source=source, attributes=attributes)


def tree_lines(rendered: str) -> list[str]:
    """The span tree, with the identity header dropped.

    A single-turn read now opens with the correlation, session and turn ids --
    the three a person has to hand from a page header and could not previously
    turn into a conversation. The tree is what these tests are about, so they
    find it rather than assuming it starts at line zero.
    """
    lines = rendered.splitlines()
    for index, line in enumerate(lines):
        if line and not line.startswith(("correlation", "session", "turn",
                                         "source", "failed", "---")):
            return lines[index:]
    return lines


def test_the_identity_of_the_turn_comes_before_the_tree():
    """A correlation id is the thing someone has; the session is what they want."""
    out = reader.render([span("answer", "a", session="sess-9", trace="tr-7")])

    assert "correlation  tr-7" in out
    assert "session      sess-9" in out
    assert "turn         t1" in out
    # And the command that widens one answer into the whole conversation, so
    # the next step does not have to be remembered.
    assert "--session sess-9" in out


def test_a_child_renders_indented_under_its_parent():
    out = tree_lines(reader.render([
        span("answer", "a"),
        span("part", "b", "a"),
        span("retrieval", "c", "b"),
    ]))

    assert out[0].startswith("answer")
    assert out[1].startswith("  part")
    assert out[2].startswith("    retrieval")


def test_a_span_whose_parent_was_sliced_away_still_renders():
    """A limit cuts a tree, and the cut branch must not vanish silently."""
    out = reader.render([span("generation", "g", parent="missing", ms=2751)])

    assert "generation" in out
    assert "2751ms" in out


def test_the_evidence_a_wrong_answer_turns_on_is_on_the_span_s_own_line():
    """The drift case: which product retrieval was told about, and what scored."""
    line = tree_lines(reader.render([
        span("retrieval", "r", product="ultra", top_score=0.79, hits=5,
             documents=4, expanded=True),
    ]))[0]

    assert "product='ultra'" in line
    assert "top_score=0.79" in line
    assert "expanded" not in line          # real, but not what you look at first


def test_what_the_question_was_understood_to_be_asking_is_on_the_line():
    """The other drift case, and the one a conversation makes possible.

    A recommendation that surprises someone usually turns on the intent and the
    facts still missing, not on the score. Both are recorded already; until now
    neither printed without `--all`, so the headline view answered "which
    passage" and not "which question did it think this was".
    """
    line = tree_lines(reader.render([
        span("candidate_discovery", "c", intent="select_product",
             objective="replaster a solid brick wall",
             missing=["substrate"], considered=11),
    ]))[0]

    assert "intent='select_product'" in line
    assert "missing=['substrate']" in line
    assert "considered" not in line


def test_the_rest_of_the_attributes_are_available_but_not_in_the_way():
    spans = [span("retrieval", "r", product="ultra", expanded=True)]

    assert "expanded" not in reader.render(spans)
    assert "expanded=True" in reader.render(spans, show_all=True)


def test_a_failed_span_says_so():
    assert "[error]" in reader.render([
        TraceSpan(trace_id="t", span_id="s", name="generation",
                  started_at="2026-09-17T10:00:00+00:00", duration_ms=9,
                  status="error")])


def test_one_turn_gets_no_header_and_a_conversation_gets_one_per_turn():
    """A session read replays several turns; a trace read is one and needs no label."""
    one = reader.render([span("answer", "a"), span("part", "b", "a")])
    assert "--- turn" not in one

    many = reader.render([
        span("answer", "a", turn="t1", trace="tr1"),
        span("answer", "b", turn="t2", trace="tr2"),
    ])
    assert many.count("--- turn") == 2


def test_an_id_that_has_aged_out_explains_itself():
    out = reader.render([])

    assert "No spans" in out and "fourteen days" in out


def test_recent_lists_one_row_per_turn_newest_first():
    out = reader.recent([
        span("answer", "a", trace="first", ms=10),
        span("part", "b", "a", trace="first"),        # not a root, not listed
        span("answer", "c", trace="second", ms=20),
    ]).splitlines()

    assert len(out) == 2
    assert "second" in out[0] and "first" in out[1]


def test_recent_on_an_empty_table_explains_itself():
    assert "No traces" in reader.recent([])


def test_the_correlation_id_a_browser_returns_reads_the_turn_back(tmp_path,
                                                                 capsys):
    """The whole chain: X-Correlation-Id is the trace id, and this reads it.

    Through `main` against a real store rather than through `render`, because
    the claim being tested is that the value copied out of a browser's network
    tab is the value the CLI takes -- which is a fact about the id, the query
    and the adapter together, and not about the formatting.
    """
    db = tmp_path / "knowledge.db"
    store = SQLiteKnowledgeRepository(db)
    store.record_spans([
        span("answer", "a", trace="8fa7c001", session="s1", ms=3379),
        span("retrieval", "b", "a", trace="8fa7c001", session="s1",
             product="ultra", top_score=0.79),
    ])
    store.close()

    assert reader.main(["8fa7c001", "--db", str(db)]) == 0
    printed = capsys.readouterr().out
    assert "answer" in printed
    assert "  retrieval" in printed
    assert "product='ultra'" in printed


def test_an_unknown_id_exits_non_zero_so_a_script_can_tell(tmp_path):
    db = tmp_path / "knowledge.db"
    SQLiteKnowledgeRepository(db).close()

    assert reader.main(["nosuchtrace", "--db", str(db)]) == 1


def test_json_mode_emits_the_spans_for_a_test_to_assert_on(tmp_path, capsys):
    db = tmp_path / "knowledge.db"
    store = SQLiteKnowledgeRepository(db)
    store.record_spans([span("retrieval", "b", trace="tr9", product="ultra")])
    store.close()

    reader.main(["tr9", "--db", str(db), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload[0]["name"] == "retrieval"
    assert payload[0]["attributes"]["product"] == "ultra"
