"""Lime Green Technical Assistant.

A local-LLM assistant over Lime Green's published material: it answers only from
retrieved passages, cites every source by document name, and refuses — with a
hand-off — when the published material does not answer the question.

Design and rationale: DECISIONS.md and docs/architecture.md.
"""

__version__ = "0.1.0"

# --------------------------------------------------------------- privacy

# Hosted tracing is closed here, at the top of the package, so that importing
# *anything* from this application closes it -- not only the one module that
# happens to pull in `langgraph`.
#
# `langgraph` depends on `langchain-core`, which depends on `langsmith`, which
# is a client for a hosted tracing service. With credentials present and a
# single environment variable set, it exports whole runs -- including the
# conversation state this system is otherwise careful never to send anywhere.
# That is incompatible with the privacy posture in CLAUDE.md ("avoid logging
# complete customer conversations") and `assistant/infrastructure/observability.py` is the only
# telemetry this project sanctions.
#
# Assignment, not `setdefault`. The earlier version used `setdefault` so as not
# to override a deliberate operator choice, and that left a measured hole: a
# parent environment carrying `LANGCHAIN_TRACING_V2=true` re-enabled export,
# because not overriding is precisely what `setdefault` does. For this variable,
# in this application, there is no operator choice to respect.
#
# Only environment variables are touched here, and deliberately: it costs
# nothing and imports nothing, so a CLI start does not pay for loading
# `langsmith`. `assistant/turn/graph.py` closes the higher-precedence global
# fallback as well, once that library is being loaded anyway.
_TRACING_VARS = (
    "LANGSMITH_TRACING", "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2",
    "LANGSMITH_OTEL_ENABLED", "LANGCHAIN_OTEL_ENABLED",
)


def _close_hosted_tracing() -> None:
    import os

    for _var in _TRACING_VARS:
        os.environ[_var] = "false"


_close_hosted_tracing()


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
