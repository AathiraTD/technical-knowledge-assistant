"""Citations must name what the answer used, and nothing else.

Found by looking at the page rather than at the suite. Asked

    "What application thickness can Lime Green Ultra be applied at?"

the assistant printed five quoted sentences, every one of them marked ``[1]``,
above a *Sources* disclosure listing **eight** documents. Nothing was wrong with
the quoting -- all five sentences really are in source [1], and the checks that
verify a figure against its cited passage all passed -- but a reader counting
eight sources against one marker reads the numbering as broken, and seven of the
eight were never used for anything.

The cause was one line. `AnswerEngine._finish` already worked out which hits the
text cites, and used that set for the caveats:

    markers = {int(m) for m in _CITE.findall(text)}
    cited = [h for i, h in enumerate(hits, 1) if i in markers] if markers else hits
    caveats = _caveat_lines(decision, self.repo, question, cited_hits=cited)
    ...
    sources=_source_rows(hits),          # <- everything retrieved, not `cited`

So the filter existed and was applied to the caveats and not to the source list.

Filtering the list is half of it. Markers are *positions* in whatever list the
reader is shown, so dropping an uncited passage shifts every marker after it --
an answer citing [1] and [4] of five passages must print [1] and [2] over a
two-entry list, or the second citation now points at the wrong document. That is
a worse defect than the one being fixed, which is why the renumbering has its
own tests here and why `test_a_marker_that_no_passage_backs_changes_nothing`
pins the fail-closed case.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answer import AnswerEngine                        # noqa: E402
from assistant.model import Chunk, Document, Retrieved           # noqa: E402
from assistant.router import Decision, Path_, Router             # noqa: E402


def passage(n: int, *, score: float = 0.7) -> Retrieved:
    """One retrieved passage, distinguishable by its number in every field."""
    url = f"https://example.invalid/doc-{n}"
    product = f"Product {n}"
    return Retrieved(
        chunk=Chunk(canonical_url=url, version=1, chunk_index=0,
                    section=f"Section {n}", content=f"Published sentence {n}.",
                    product=product, document_type="datasheet", authority=1),
        score=score,
        document=Document(canonical_url=url, title=f"Data Sheet {n}",
                          document_type="datasheet", authority=1,
                          product=product, link_text=f"Data Sheet {n}"),
    )


HITS = [passage(n) for n in range(1, 6)]            # five passages, [1]..[5]


class _NoCaveats:
    """The only thing `_finish` asks a repository for is a document's caveats."""

    def caveats(self, url):                                  # noqa: ARG002
        return []


def engine() -> AnswerEngine:
    """An engine with a stub store; `_finish` only reads caveats from it."""
    made = AnswerEngine.__new__(AnswerEngine)
    made.repo = _NoCaveats()
    made.retriever = Router()
    made.model = None
    made.names = {"products": [], "colours": [], "merchants": [], "contact": {}}
    return made


def finish(text: str, hits=None):
    decision = Decision(Path_.EXTRACT, "because", "7", hits=hits or HITS)
    return engine()._finish(decision, text, hits or HITS, "a question")


# ------------------------------------------------------- the list that prints

def test_the_source_list_holds_only_the_passages_the_text_cites():
    """The reported defect: five quotes on [1], eight documents listed."""
    answer = finish("“Published sentence 1.” [1]. “And again.” [1].")

    assert len(answer.sources) == 1
    assert answer.sources[0]["name"] == "Data Sheet 1"
    assert answer.sources[0]["marker"] == 1


def test_every_listed_source_is_reachable_from_a_marker_in_the_text():
    """The invariant behind the fix, stated over the list rather than an example."""
    answer = finish("One [1]. Three [3]. Five [5].")
    printed = {int(m) for m in __import__("re").findall(r"\[(\d+)\]", answer.text)}

    assert printed == {row["marker"] for row in answer.sources}


# ------------------------------------------------------------ the renumbering

def test_markers_are_renumbered_onto_the_filtered_list():
    """[1] and [4] of five become [1] and [2] over a two-entry list."""
    answer = finish("First [1]. Fourth [4].")

    assert len(answer.sources) == 2
    assert "[4]" not in answer.text
    assert answer.text == "First [1]. Fourth [2]."
    # The renumbering is only correct if [2] now names the document [4] named.
    assert [row["name"] for row in answer.sources] == ["Data Sheet 1", "Data Sheet 4"]


def test_a_citation_still_points_at_the_document_it_pointed_at_before():
    """The property that makes renumbering safe rather than merely tidy."""
    answer = finish("Only the fourth [4].")

    assert answer.text == "Only the fourth [1]."
    assert answer.sources[0]["url"] == "https://example.invalid/doc-4"
    assert answer.sources[0]["section"] == "Section 4"


def test_order_follows_the_evidence_not_the_order_the_markers_appear():
    """[4] cited before [1] still lists document 1 first, as [1]."""
    answer = finish("Fourth first [4]. Then first [1].")

    assert [row["name"] for row in answer.sources] == ["Data Sheet 1", "Data Sheet 4"]
    assert answer.text == "Fourth first [2]. Then first [1]."


# --------------------------------------------------------------- the edges

def test_an_answer_that_cites_nothing_keeps_every_passage():
    """Unchanged behaviour: a refusal prints what was found, uncited."""
    answer = finish("Nothing published states this.")

    assert len(answer.sources) == len(HITS)


def test_a_marker_that_no_passage_backs_changes_nothing():
    """Fail closed. A marker out of range is a defect, not a renumbering input.

    The checks refuse a generated answer citing a passage it was not given, so
    this should be unreachable from Compose. If it is ever reached, the answer
    keeps the full list and the untouched text, because silently renumbering
    around a marker nothing backs would turn a visible fault into a citation
    that looks correct and is not.
    """
    answer = finish("Real [1]. Invented [9].")

    assert answer.text == "Real [1]. Invented [9]."
    assert len(answer.sources) == len(HITS)


def test_the_disclosure_suffix_still_strips_after_renumbering():
    """`Answer.body` is `text` minus `disclosure`; renumbering must not break it."""
    answer = finish("Fourth [4].")
    answer.disclosure = "Source passage — Section 4:\n“Published sentence 4.”"
    answer.text = f"{answer.text}\n\n{answer.disclosure}"

    assert answer.body == "Fourth [1]."


def test_chunk_ids_still_record_everything_retrieved():
    """Diagnostics answer "what did retrieval find", not "what printed"."""
    answer = finish("First [1].")

    assert len(answer.diagnostics["chunk_ids"]) == len(HITS)
