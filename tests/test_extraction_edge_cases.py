"""Text normalization may remove a nonempty PDF span completely."""
from assistant.extract import _pdf_lines


def test_pdf_line_containing_only_invisible_unicode_is_not_evidence():
    class Page:
        def get_text(self, mode):
            return {'blocks': [{'lines': [{'spans': [
                {'text': '\u200b', 'size': 11, 'flags': 0, 'font': 'Regular'}
            ]}]}]}
    assert _pdf_lines([Page()]) == []
