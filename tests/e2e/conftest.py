"""The browser fixture, and the reasons it is opt-in.

These tests drive a real Chromium against a real server, because the things
they prove cannot be proved any other way: an HttpOnly cookie is invisible to
the page's own script, a multipart upload is assembled by the browser and not by
the test, and whether a citation is a working link is a question about rendered
DOM. `eval/run.py` deliberately drives the library instead, and decision 15
keeps the CLI canonical so that a UI failure never costs the evidence -- so this
suite is a thin proof of the surface, not a second evaluation of answer quality.
Nothing here asserts whether an answer is *right*; the harness owns that.

**Opt-in twice, and for two different reasons.** `ASSISTANT_E2E` gates the run
because a browser download is exactly the heavyweight dependency `CLAUDE.md`
keeps out of deterministic CI. The browser-present check gates it again because
a developer who installed the package but not the binary should get a skip that
says so rather than a failure that reads like a broken test.

**Questions that route, not questions that compose.** Every test here picks a
question the policy gate answers -- price, stockist -- so no generation runs.
That is not a shortcut around slowness: an uncached compose costs tens of
seconds to minutes on this hardware (DECISIONS.md, "Still open"), and a browser
suite whose every case waits on the model would be abandoned within a week. The
session, the cookie and the upload boundary behave identically either way,
because none of them is downstream of the model.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# A question the policy gate answers from the routing table, with no retrieval
# and no model. Used wherever a test needs a turn to have happened rather than a
# particular answer to come back.
ROUTED_QUESTION = "How much does a bag cost?"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:                                  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    """Skip the whole directory rather than fail it, with the reason said once."""
    if not os.environ.get("ASSISTANT_E2E"):
        reason = "ASSISTANT_E2E is not set; browser tests not run"
    elif not _browser_available():
        reason = ("no Chromium for Playwright; install it with "
                  "python -m playwright install chromium")
    else:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if Path(str(item.fspath)).parent.name == "e2e":
            item.add_marker(skip)


@pytest.fixture(scope="session")
def server() -> str:
    """A real `python -m assistant.ui`, on its own port, polled until ready.

    A subprocess rather than a thread, because the thing under test includes
    `main` -- the audience the server was started with, the store it opened, the
    session store it installed. A thread sharing this process's imports would
    prove a handler class works and leave the wiring untested.
    """
    import httpx

    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "assistant.ui", "--port", str(port),
         "--no-browser", "--host", "127.0.0.1"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):                           # the index and Ollama check
            if process.poll() is not None:
                out, err = process.communicate()
                pytest.skip("the server would not start: "
                            + err.decode("utf-8", "replace")[-400:])
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except Exception:                          # noqa: BLE001
                time.sleep(0.5)
        else:
            pytest.skip("the server did not become ready within a minute")
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as play:
        instance = play.chromium.launch()
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser, server):
    """A fresh context per test, so no cookie outlives the test that made it."""
    context = browser.new_context(base_url=server)
    try:
        yield context.new_page()
    finally:
        context.close()


def ask(page, question: str, timeout: int = 60_000) -> None:
    """Type a question, send it, and wait for the answer itself to land.

    Waits on `.tag` rather than on the message count. An upload note is also a
    `.message.assistant`, so counting messages would return as soon as the
    warning appeared and leave the assertions racing the answer behind it; the
    route tag is written once, by the answer, and only by the answer.
    """
    before = page.locator(".message.assistant .tag").count()
    page.fill("#question-input", question)
    page.click("#send-btn")
    page.wait_for_function(
        "n => document.querySelectorAll('.message.assistant .tag').length > n",
        arg=before, timeout=timeout)
