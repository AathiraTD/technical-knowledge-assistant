"""The Ollama client, tested against a fake server rather than a real one.

Nothing in this file contacts Ollama. Every call goes through a stand-in client
installed over `ollama._client`, which is the whole point: the behaviours that
matter here are what happens when the server misbehaves, and a healthy local
server cannot demonstrate any of them.

Two of these tests stand for wall-clock time that was actually lost. A single
transient 400 ended a forty-minute embedding run at 97 per cent, which is why
the retry exists; a batch the server would not accept whole took the good
passages down with it, which is why the split exists. The rest guard the
quieter failure: a vector of the wrong width written into a column sized for a
different one is silent corruption, so it is refused loudly instead.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.infrastructure import ollama
from assistant.infrastructure.ollama import (  # noqa: E402
    OllamaUnavailable,
)


# ----------------------------------------------------------- the fake server


class FakeResponse:
    """Just enough of `httpx.Response` for the two calls this client makes."""

    def __init__(self, payload=None, status: int = 200, text: str = "",
                 unparseable: bool = False) -> None:
        self.payload = payload
        self.status = status
        self.text = text
        self.unparseable = unparseable

    def raise_for_status(self) -> None:
        if self.status >= 400:
            request = httpx.Request("POST", ollama.HOST)
            response = httpx.Response(self.status, text=self.text, request=request)
            raise httpx.HTTPStatusError(self.text, request=request, response=response)

    def json(self):
        if self.unparseable:
            raise json.JSONDecodeError("Expecting value", "<not json>", 0)
        return self.payload


class FakeClient:
    """Records every call, answers from a handler, never opens a socket."""

    def __init__(self, handler, calls: list) -> None:
        self.handler = handler
        self.calls = calls

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def post(self, path: str, json=None):
        self.calls.append((path, json))
        return self.handler(path, json)

    def get(self, path: str):
        self.calls.append((path, None))
        return self.handler(path, None)


@pytest.fixture
def server(monkeypatch):
    """Install a fake Ollama, and make the retry backoff cost nothing."""
    monkeypatch.setattr(ollama.time, "sleep", lambda _seconds: None)

    state: dict = {"calls": [], "timeouts": []}

    def install(handler):
        def factory(timeout):
            state["timeouts"].append(timeout)
            return FakeClient(handler, state["calls"])

        monkeypatch.setattr(ollama, "_client", factory)
        return state

    return install


def vector(seed: float = 0.0) -> list[float]:
    """A vector of the declared width, carrying an identifying first value."""
    return [seed] + [0.0] * (ollama.EMBED_DIMENSIONS - 1)


def vectors_for(window: list[str]) -> dict:
    """What a healthy server returns: one vector per input, in input order."""
    return {"embeddings": [vector(float(len(t))) for t in window]}


# -------------------------------------------------------- retry and splitting


def test_a_transient_failure_does_not_lose_the_run(server):
    """One transient 400 once ended a forty minute embedding run at 97 per cent."""
    attempts = {"n": 0}

    def handler(_path, body):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResponse(status=400, text="unexpected server error")
        return FakeResponse(vectors_for(body["input"]))

    state = server(handler)
    out = ollama.embed(["mixing water", "coverage"], batch=16)

    assert len(out) == 2
    assert attempts["n"] == 2, "the batch was abandoned rather than retried"
    assert len(state["calls"]) == 2


def test_a_batch_the_server_will_not_take_whole_is_halved_not_abandoned(server):
    """One unacceptable batch must not take the 643 good passages down with it."""
    def handler(_path, body):
        window = body["input"]
        if len(window) > 2:
            return FakeResponse(status=400, text="input exceeds the context window")
        return FakeResponse(vectors_for(window))

    state = server(handler)
    out = ollama.embed(["a", "bb", "ccc", "dddd"], batch=4)

    assert len(out) == 4
    # Three exhausted attempts on the whole window, then one call per half.
    assert len(state["calls"]) == 5
    # The halves must come back in input order, not in the order they finished.
    assert [v[0] for v in out] == [1.0, 2.0, 3.0, 4.0]


def test_a_single_unembeddable_passage_names_itself_and_quotes_the_server(server):
    """The one passage that cannot be embedded has to be identifiable afterwards."""
    server(lambda _p, _b: FakeResponse(status=400, text="unexpected EOF in input"))

    with pytest.raises(OllamaUnavailable) as caught:
        ollama.embed(["x" * 41], batch=1)

    message = str(caught.value)
    assert "41 characters" in message
    assert "unexpected EOF in input" in message
    assert "3 attempts" in message


def test_a_transport_error_without_a_response_is_still_reported(server):
    """A connection that drops carries no body, and must not mask its own cause."""
    def handler(_path, _body):
        raise httpx.ConnectError("connection reset by peer")

    server(handler)
    with pytest.raises(OllamaUnavailable) as caught:
        ollama.embed(["one passage"], batch=1)
    assert "connection reset by peer" in str(caught.value)


# ----------------------------------------------------------- the shape check


def test_a_missing_vector_is_refused_rather_than_written():
    """Fewer vectors than passages would silently misalign every citation."""
    with pytest.raises(OllamaUnavailable) as caught:
        ollama._check([vector()], expected=2, model=ollama.EMBED_MODEL)
    assert "asked for 2 embeddings, got 1" in str(caught.value)


def test_a_vector_of_the_wrong_width_is_refused():
    """A 512d vector in a column sized for 1024d is corruption, not a warning."""
    with pytest.raises(OllamaUnavailable) as caught:
        ollama._check([[0.0] * 512], expected=1, model="some-other-model")
    message = str(caught.value)
    assert "512 dimensions" in message
    assert str(ollama.EMBED_DIMENSIONS) in message
    assert "some-other-model" in message


def test_a_wrong_width_is_not_retried_into_a_split(server):
    """A dimension mismatch is a configuration fault, so retrying only wastes time."""
    state = server(lambda _p, body: FakeResponse(
        {"embeddings": [[0.0] * 8 for _ in body["input"]]}))

    with pytest.raises(OllamaUnavailable):
        ollama.embed(["one", "two"], batch=2)
    assert len(state["calls"]) == 1


# ----------------------------------------------------------------- batching


def test_embedding_is_batched_and_reports_progress(server):
    """644 separate round trips cost more in overhead than the inference does."""
    state = server(lambda _p, body: FakeResponse(vectors_for(body["input"])))
    seen: list[tuple[int, int]] = []

    out = ollama.embed([f"passage {i}" for i in range(35)], batch=16,
                       progress=lambda done, total: seen.append((done, total)))

    assert len(out) == 35
    assert [len(body["input"]) for _path, body in state["calls"]] == [16, 16, 3]
    assert seen == [(16, 35), (32, 35), (35, 35)]


def test_a_question_is_embedded_on_its_own(server):
    """Query time embeds one string; handing back a list of one would break callers."""
    server(lambda _p, body: FakeResponse(vectors_for(body["input"])))
    got = ollama.embed_one("how much water does Solo need")
    assert len(got) == ollama.EMBED_DIMENSIONS


# ---------------------------------------------------------- tags and models


def test_a_server_that_is_not_answering_names_the_command_that_fixes_it(server):
    """The first reader of this error is an assessor on a clean machine."""
    def handler(_path, _body):
        raise httpx.ConnectError("all connection attempts failed")

    server(handler)
    with pytest.raises(OllamaUnavailable) as caught:
        ollama.available()

    message = str(caught.value)
    assert "ollama serve" in message
    assert ollama.HOST in message


def test_available_lists_the_tags_the_server_holds(server):
    server(lambda _p, _b: FakeResponse(
        {"models": [{"model": "qwen3-embedding:0.6b"}, {"model": "qwen3.5:4b"}]}))
    assert ollama.available() == ["qwen3-embedding:0.6b", "qwen3.5:4b"]


def test_an_exactly_matching_tag_satisfies_require(server):
    server(lambda _p, _b: FakeResponse({"models": [{"model": "qwen3.5:4b"}]}))
    ollama.require("qwen3.5:4b")


def test_a_bare_name_match_satisfies_require(server):
    """A pull that landed a different quantisation is still the same model."""
    server(lambda _p, _b: FakeResponse({"models": [{"model": "qwen3.5:4b-q8"}]}))
    ollama.require("qwen3.5:4b")


def test_a_model_that_is_not_pulled_names_the_pull_command(server):
    """Failing before a long run beats failing in the middle of one."""
    server(lambda _p, _b: FakeResponse({"models": [{"model": "qwen3.5:4b"}]}))

    with pytest.raises(OllamaUnavailable) as caught:
        ollama.require("nomic-embed-text")

    message = str(caught.value)
    assert "ollama pull nomic-embed-text" in message
    assert "qwen3.5:4b" in message


def test_a_server_holding_nothing_says_so_rather_than_printing_an_empty_list(server):
    server(lambda _p, _b: FakeResponse({"models": []}))
    with pytest.raises(OllamaUnavailable) as caught:
        ollama.require("qwen3.5:4b")
    assert "(none)" in str(caught.value)


# --------------------------------------------------------------- generation


def test_generation_is_pinned_so_a_transcript_reproduces(server):
    """A reviewer re-running the transcript has to get the transcript back."""
    state = server(lambda _p, _b: FakeResponse({"response": "  Mix with water [1].  "}))

    text, seconds = ollama.generate("Question: how much water?")

    (path, body), = state["calls"]
    assert path == "/api/generate"
    assert body["options"]["temperature"] == 0
    assert body["options"]["top_p"] == 1
    assert body["options"]["seed"] == 0
    assert body["options"]["num_ctx"] == 8192
    assert body["think"] is False, "reasoning tokens are the whole latency budget"
    assert body["stream"] is False
    assert "system" not in body
    assert text == "Mix with water [1]."
    assert seconds >= 0


def test_a_system_prompt_is_sent_when_one_is_given(server):
    state = server(lambda _p, _b: FakeResponse({"response": "ok"}))
    ollama.generate("prompt", system="answer only from the passages", seed=7)
    (_path, body), = state["calls"]
    assert body["system"] == "answer only from the passages"
    assert body["options"]["seed"] == 7


def test_a_failed_generation_call_is_reported_as_unavailable(server):
    """A dead model server must not surface as a bare httpx traceback."""
    def handler(_path, _body):
        raise httpx.ReadTimeout("timed out waiting for the model")

    server(handler)
    with pytest.raises(OllamaUnavailable) as caught:
        ollama.generate("prompt")
    assert "generation call failed" in str(caught.value)


def test_an_unparseable_reply_is_refused_rather_than_printed(server):
    """Half a JSON body must not become half an answer."""
    server(lambda _p, _b: FakeResponse(unparseable=True))
    with pytest.raises(OllamaUnavailable) as caught:
        ollama.generate("prompt", model="qwen3.5:4b")
    assert "unparseable response from qwen3.5:4b" in str(caught.value)


def test_the_client_only_ever_talks_to_the_local_host():
    """Nothing in this module should be able to reach off the machine."""
    with ollama._client(5.0) as client:
        assert str(client.base_url).rstrip("/") == ollama.HOST
        assert client.timeout.read == 5.0


# ------------------------------------------------------------------ latency


def test_generation_asks_ollama_to_keep_the_model_resident(server):
    """An evicted model costs about 7.4 seconds to read back off disk."""
    state = server(lambda _p, _b: FakeResponse({"response": "ok [1]."}))
    ollama.generate("prompt")

    (_path, body), = state["calls"]
    assert body["keep_alive"] == ollama.KEEP_ALIVE
    assert ollama.KEEP_ALIVE, "an empty value would unload immediately"


def test_the_answer_length_is_bounded(server):
    """The architecture claims a cap of about 200 tokens; it has to be real."""
    state = server(lambda _p, _b: FakeResponse({"response": "ok [1]."}))
    ollama.generate("prompt")

    (_path, body), = state["calls"]
    assert body["options"]["num_predict"] == ollama.MAX_ANSWER_TOKENS
    assert ollama.MAX_ANSWER_TOKENS == 256


def test_a_caller_may_bound_the_answer_further(server):
    """Warming wants one token, not two hundred and fifty six."""
    state = server(lambda _p, _b: FakeResponse({"response": "ok"}))
    ollama.generate("prompt", num_predict=1)

    (_path, body), = state["calls"]
    assert body["options"]["num_predict"] == 1


def test_embedding_also_keeps_the_model_resident(server):
    """A question embedded cold costs 3033 ms against 679 ms warm."""
    state = server(lambda _p, body: FakeResponse(vectors_for(body["input"])))
    ollama.embed_one("how much water does Solo need")

    (_path, body), = state["calls"]
    assert body["keep_alive"] == ollama.KEEP_ALIVE


def test_the_keep_alive_window_can_be_turned_off():
    """Holding 3 GB resident is the right trade on the build machine, not everywhere."""
    assert ollama.KEEP_ALIVE == os.environ.get("OLLAMA_KEEP_ALIVE", "30m")


def test_warming_loads_the_model_with_a_single_token(server):
    """The first real question should not be the one that pays for the load."""
    state = server(lambda _p, _b: FakeResponse({"response": "o"}))

    assert ollama.warm() is True

    (path, body), = state["calls"]
    assert path == "/api/generate"
    assert body["model"] == ollama.GENERATION_MODEL
    assert body["options"]["num_predict"] == 1, "warming must not generate an answer"
    assert body["keep_alive"] == ollama.KEEP_ALIVE


def test_warming_warms_the_model_it_is_given(server):
    state = server(lambda _p, _b: FakeResponse({"response": "o"}))
    assert ollama.warm(model="qwen3:4b-instruct") is True
    (_path, body), = state["calls"]
    assert body["model"] == "qwen3:4b-instruct"


def test_a_refused_warm_up_reports_failure_rather_than_raising(server):
    """Warming is an optimisation, so it must never stop the program starting."""
    def handler(_path, _body):
        raise httpx.ConnectError("all connection attempts failed")

    server(handler)
    assert ollama.warm() is False


def test_an_unparseable_warm_up_reply_is_also_survived(server):
    """The other way generate fails must not escape either."""
    server(lambda _p, _b: FakeResponse(unparseable=True))
    assert ollama.warm() is False
