"""Lime Green Technical Assistant.

A local-LLM assistant over Lime Green's published material: it answers only from
retrieved passages, cites every source by document name, and refuses — with a
hand-off — when the published material does not answer the question.

Design and rationale: DECISIONS.md and docs/architecture.md.
"""

__version__ = "0.1.0"


def use_utf8() -> None:
    """Make the console print the characters the datasheets actually use.

    Windows consoles still default to a legacy code page, which turns °C into a
    replacement character and curly quotes into noise. That matters more here
    than it would elsewhere: the promise is that figures print exactly as
    published, and "5-6 litres between 5°C and 25°C" visibly breaks it.
    Errors are replaced rather than raised, because a console that cannot show
    a character should not end the answer.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
