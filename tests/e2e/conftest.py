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
def server(tmp_path_factory) -> str:
    """A real `python -m assistant.ui`, on its own port, polled until ready.

    A subprocess rather than a thread, because the thing under test includes
    `main` -- the audience the server was started with, the store it opened, the
    session store it installed. A thread sharing this process's imports would
    prove a handler class works and leave the wiring untested.

    **Its output goes to a file, never to a pipe.** `assistant/ui.py` turns
    structured logging on by default and writes a JSON line per event to stderr,
    which is several lines per answer. A `subprocess.PIPE` nobody drains holds
    about 64KB before the writing process blocks on it forever, so the server
    answered the first test, filled the buffer, and then hung -- presenting as
    every later test timing out against a server that looked alive and was
    simply stuck mid-write. A file has no such limit, and it is also the thing
    worth printing when something here fails.
    """
    import httpx

    port = _free_port()
    log = tmp_path_factory.mktemp("server") / "ui.log"
    handle = log.open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "assistant.ui", "--port", str(port),
         "--no-browser", "--host", "127.0.0.1"],
        cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):                           # the index and Ollama check
            if process.poll() is not None:
                pytest.skip("the server would not start: "
                            + log.read_text("utf-8", "replace")[-400:])
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
        handle.close()


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


# --------------------------------------------------------------- demo journeys
#
# Two suites share this fixture set, and they are deliberately different kinds
# of test. `test_web_ui.py` proves the surface -- a cookie is HttpOnly, a
# multipart body arrives, a citation is a working link. `test_demo_journeys.py`
# walks the journeys the demonstration actually walks, in the order it walks
# them, so that "it worked when I rehearsed it" is a thing CI can say rather
# than a thing a person remembers.
#
# The split that keeps them usable is speed, and speed here means one thing:
# whether the model runs. An uncached compose costs tens of seconds to minutes
# on a processor with no graphics card (DECISIONS.md, "Still open"), so a suite
# whose every case composes is a suite nobody runs twice. Every journey that is
# really about browser behaviour therefore picks a question the policy gate or
# the router answers without generation, and the handful that genuinely need a
# composed answer are marked `slow` and excluded from the smoke run.
#
#   python -m pytest tests/e2e -m smoke      # ~3 min, pre-flight
#   python -m pytest tests/e2e               # everything, pre-demo
#
# Measured: smoke is 11 tests in 2m45s, most of which is one server start and
# one cold embedding load. The `slow` half composes or perceives and runs in
# tens of minutes, which is why it is a pre-demo check and not a pre-commit one.

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: fast pre-flight journeys; no model generation")
    config.addinivalue_line(
        "markers", "slow: needs a real composed answer; minutes, not seconds")


# A question the policy gate answers from the routing table: no retrieval, no
# model, and a referral printed verbatim. Journey E's safety case and the
# fixtures that only need "a turn happened" both use this shape.
CERTIFICATION_QUESTION = "Can you confirm my Warmshell build complies with Part L?"

# Router step 5: the substrate slot is load-bearing and uncued, so this asks
# back rather than guessing a wall. No generation.
SELECTION_QUESTION = "Which plaster should I use?"

# Router step 6: calculation words take the extract path over the coverage and
# pack-size passages, printed by code with the sum refused. A real retrieval and
# a real citation, without a generation.
QUANTITY_QUESTION = "How many bags of Duro do I need for 20 square metres?"

# The smallest thing that is really a PNG: an 8-byte signature and a valid IHDR.
ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")


def ask_json(page, question: str, timeout: int = 60_000) -> dict:
    """Send a question and return the `/ask` payload the page received.

    The rendered DOM is what a person sees and the payload is what the server
    decided; a journey usually wants to assert on both, and intercepting the
    response is the only way to read the second without asking the page to
    display things it has no reason to display -- the route taken, the slots
    carried, how many images were read.
    """
    with page.expect_response(lambda r: "/ask" in r.url, timeout=timeout) as caught:
        page.fill("#question-input", question)
        page.click("#send-btn")
    return caught.value.json()


def route_of(payload: dict) -> str:
    """The path the deterministic router took, or '' if nothing was answered."""
    parts = payload.get("parts") or []
    return parts[0]["path"] if parts else ""


def answer_text(payload: dict) -> str:
    parts = payload.get("parts") or []
    return " ".join((p.get("text") or p.get("body") or "") for p in parts)


def expand(page, label: str):
    """Open the Sources or 'Why this answer?' disclosure and return its panel.

    Both are a `button.disclosure-btn` beside a sibling panel, toggled by adding
    `.open` to each -- not a `<details>`, so there is no `open` attribute to
    wait on and visibility is decided by a class.
    """
    button = page.locator(".disclosure-btn", has_text=label).first
    button.click()
    panel = button.locator("xpath=following-sibling::*[1]")
    panel.wait_for(state="visible", timeout=5_000)
    return panel


@pytest.fixture(scope="session", autouse=True)
def warm(server):
    """One retrieval before any browser test, so the first one is not the slow one.

    The embedding model is loaded lazily by Ollama, so whichever test asks the
    first question that reaches retrieval pays several tens of seconds that have
    nothing to do with it -- and, being first, it is usually the one with the
    tightest timeout. That is exactly how the ask-back journey failed at sixty
    seconds while measuring 21.5 s against an already-warm server.

    This is not a trick to make the suite look fast. It is the same step
    `docs/DEMO-SCRIPT.md` puts at the top of its pre-demo checklist, for the
    same reason, and doing it here means a timeout in this suite means something
    is actually wrong rather than that Ollama was cold.

    Deliberately not warming the *generation* model: the tests that compose are
    marked `slow` and carry timeouts that expect to pay for it, and warming it
    would hide a real regression in generation latency behind a fixture.
    """
    import httpx

    try:
        httpx.get(f"{server}/ask", params={"q": QUANTITY_QUESTION}, timeout=300)
    except Exception:                                  # noqa: BLE001
        pass          # a cold model is a slow suite, not a failed one
