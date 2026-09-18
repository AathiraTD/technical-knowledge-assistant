"""A PDF line that survives the emptiness filter can still normalise to nothing.

`_pdf_lines` in `assistant/indexing/extract.py` drops empty spans with
`sp["text"].strip()` and then normalises whatever is left with `clean()`. Those
are two different notions of empty, and the gap between them is the defect this
file pins. A zero-width space is not whitespace to `str.strip()`, so a span
containing only `\u200b` passes the first filter with its font size and bold
flag intact — and then `clean()` reduces it to the empty string.

The second guard, `if not text: continue`, is therefore **load-bearing rather
than defensive**. Without it the extractor emits a line with no characters but
a real font size, which then feeds the body-size weighting that decides what
counts as a heading, and is offered to `_is_heading_text` as a candidate. An
invisible character would be shaping the chunk boundaries of a datasheet, and
nothing downstream could see why.

This is a unit test against one function with a hand-built page object, not an
extraction-quality test: it says nothing about whether real PDFs chunk well.
That evidence lives in the ingestion report and in the corpus-wide probe
recorded under decision 5.
"""
from assistant.indexing.extract import _pdf_lines


def test_pdf_line_containing_only_invisible_unicode_is_not_evidence():
    """A span of only `​` passes `strip()` and must still be dropped by `clean()`."""
    class Page:
        def get_text(self, mode):
            return {'blocks': [{'lines': [{'spans': [
                {'text': '\u200b', 'size': 11, 'flags': 0, 'font': 'Regular'}
            ]}]}]}
    assert _pdf_lines([Page()]) == []
