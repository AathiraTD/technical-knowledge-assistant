"""Hosted tracing stays off, however hostile the environment.

`langgraph` brings `langchain-core`, which brings `langsmith` — a client for a
hosted tracing service. With credentials present and one environment variable
set, it exports whole runs to a third party, and a "run" here contains the
customer's question, the retrieved passages and the conversation state. CLAUDE.md
says not to log complete customer conversations; `assistant/infrastructure/observability.py` is
the only telemetry this project sanctions. So the dependency is acceptable only
while this file passes.

**Why every test here spawns a subprocess.** The variables are read at import
time, and the protection is applied at import time, so the only honest way to
test it is to start a fresh interpreter with the hostile values already in the
environment. Setting them with `monkeypatch` inside an interpreter that has
already imported `assistant` would test nothing at all — it would assert against
a module whose guard had run before the test existed.

Each hostile case is paired with a **control**: the same environment, without
importing this application, asserting that tracing really would have been on.
Without the control the suite could pass because the variables were misspelled,
or because the library changed its mind, and it would look like protection.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Everything that can switch hosted tracing on, in both of the prefixes
# `langsmith` resolves, plus credentials — so a pass cannot be explained away by
# "there was no API key anyway".
HOSTILE = {
    "LANGCHAIN_TRACING_V2": "true",
    "LANGSMITH_TRACING": "true",
    "LANGCHAIN_TRACING": "true",
    "LANGSMITH_TRACING_V2": "true",
    "LANGSMITH_OTEL_ENABLED": "true",
    "LANGSMITH_API_KEY": "ls__fake_key_for_the_test",
    "LANGCHAIN_API_KEY": "ls__fake_key_for_the_test",
    "LANGSMITH_PROJECT": "should-never-be-used",
}

ASK = ("from langsmith.utils import tracing_is_enabled;"
       "print('TRACING=' + str(tracing_is_enabled()))")


def run(code: str, env_extra: dict) -> str:
    """A fresh interpreter, with the environment set before anything imports."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LANGSMITH_", "LANGCHAIN_"))}
    env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, r'%s');\n" % ROOT + code],
        env=env, capture_output=True, text=True, cwd=str(ROOT), timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def traced(code: str, env_extra: dict) -> bool:
    out = run(code, env_extra)
    assert "TRACING=" in out, out
    return "TRACING=True" in out


# ------------------------------------------------------------- the control


def test_the_hostile_environment_really_would_enable_tracing():
    """Without this application, the same variables switch tracing on.

    The control that gives every other test in this file its meaning. If this
    ever starts failing, the protection below is no longer being *tested* — the
    variables have been renamed, or the library has changed, and the suite would
    otherwise keep passing while guarding nothing.
    """
    assert traced(ASK, HOSTILE), (
        "the hostile environment no longer enables tracing on its own, so the "
        "tests below prove nothing until they are updated")


# --------------------------------------------------- importing the application


def test_importing_the_package_closes_it():
    """`import assistant` is enough. Not only the module that pulls in langgraph."""
    assert not traced("import assistant;" + ASK, HOSTILE)


def test_importing_the_graph_closes_it():
    assert not traced("import assistant.turn.graph;" + ASK, HOSTILE)


@pytest.mark.parametrize("module", ["assistant.interfaces.ui", "assistant.interfaces.cli",
                                    "assistant.answering.engine", "assistant.turn.graph"])
def test_importing_any_surface_closes_it(module):
    """Whichever entry point a deployment happens to load first."""
    assert not traced(f"import {module};" + ASK, HOSTILE)


def test_the_whole_application_together_closes_it():
    assert not traced(
        "import assistant.interfaces.ui, assistant.interfaces.cli, assistant.answering.engine, assistant.turn.graph;"
        + ASK, HOSTILE)


# ------------------------------------------------- the individual switches


@pytest.mark.parametrize("variable", sorted(
    v for v in HOSTILE if "TRACING" in v or "OTEL" in v))
def test_no_single_switch_can_reopen_it(variable):
    """Each variable on its own, with credentials. This is the measured hole.

    `LANGCHAIN_TRACING_V2=true` was the one that got through: the guard used
    `os.environ.setdefault`, which by definition does not override a value the
    parent already set.
    """
    env = {variable: "true",
           "LANGSMITH_API_KEY": "ls__fake_key_for_the_test"}

    assert not traced("import assistant.turn.graph;" + ASK, env)


def test_the_guard_survives_something_rewriting_the_environment_afterwards():
    """The env is only the weakest of four inputs, so it is not the only guard.

    `langsmith` consults a context variable, an open run tree, a process-global
    fallback and then the environment. `assistant/turn/graph.py` closes the global
    fallback too, which outranks the environment — so re-setting a variable
    after import does not reopen it.
    """
    code = ("import assistant.turn.graph;"
            "import os;"
            "os.environ['LANGCHAIN_TRACING_V2'] = 'true';"
            "os.environ['LANGSMITH_TRACING'] = 'true';"
            + ASK)

    assert not traced(code, HOSTILE)


# ----------------------------------------------------------- what stays ours


def test_our_own_observability_is_untouched():
    """The point is not "no telemetry". It is "our telemetry, not theirs"."""
    out = run(
        "import assistant.turn.graph;"
        "from assistant.infrastructure import observability as obs;"
        "obs.event('probe', n=1);"
        "print('OURS=ok')", HOSTILE)

    assert "OURS=ok" in out


def test_no_langsmith_client_is_constructed_at_import():
    """A configured client is a socket waiting for a reason to be used."""
    out = run(
        "import assistant.turn.graph;"
        "from langsmith._internal import _context as c;"
        "print('CLIENT=' + str(c._GLOBAL_CLIENT is not None));"
        + ASK, HOSTILE)

    assert "CLIENT=False" in out, out


def test_the_module_reports_its_own_state_truthfully():
    """`tracing_disabled()` asks the library rather than reading the env back.

    The earlier version of this check inspected environment variables, which is
    how a hole that lived in a different one of the four inputs went unnoticed.
    """
    out = run("import assistant.turn.graph as g;"
              "print('SELFREPORT=' + str(g.tracing_disabled()));" + ASK, HOSTILE)

    assert "SELFREPORT=True" in out
    assert "TRACING=False" in out
