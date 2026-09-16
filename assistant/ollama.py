"""The Ollama client: two endpoints over HTTP, no client library.

Decision recorded among the five of no consequence — a dependency that wraps
`/api/embed` and `/api/generate` earns its place only if it does something
those two calls do not, and it does not.

What this module does carry is the temperature and the refusal to guess. The
generation call is pinned to temperature zero with a fixed seed, because a
reviewer re-running a transcript should get the transcript back. And the
embedding call fails loudly on a dimension it did not expect, rather than
writing a shorter vector into a column sized for a longer one.

It also carries the two settings that decide how long an answer takes: how long
Ollama keeps the model resident, and how many tokens it is allowed to generate.
Both are here rather than at the call sites because they are properties of the
model server, not of any one question.
"""

from __future__ import annotations

import json
import os
import time

import httpx

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# DEVELOPMENT DEFAULT, not a final selection. Decision 6 says the embedding
# model is chosen by measurement on the same corpus and questions, and
# `eval/embedding_choice.py` is that measurement. Whatever is set here is
# recorded in the index header, and the engine refuses to query an index built
# with a different one — so changing it is safe, and forgetting to rebuild is
# not silently possible.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "1024"))

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "qwen3.5:4b")

# How long Ollama holds the model in memory after a call. Send nothing and the
# server applies its own default, which is five minutes: measured by asking
# /api/ps when the deadline was set. Any gap longer than that and the next
# question pays to read the weights back off disk before it processes a single
# token. Measured with a one token generate, so the figure is load and nothing
# else: 8.11 seconds cold against 0.72 warm, a reload cost of 7.39 seconds.
# That lands on whoever asks the first question after a quiet spell, which in a
# demonstration is always the assessor.
#
# The trade is roughly 3 GB of resident memory for a predictable first answer,
# and it is configurable because that trade is not always the right one. On the
# 24 GB build machine, holding the generation model alongside the embedding
# model is affordable; on a smaller box, or on a server running other work, it
# is not. `OLLAMA_KEEP_ALIVE=5m` restores the server default, and `0` unloads
# the moment the call returns, which is stricter than the old behaviour rather
# than the same as it.
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# The ceiling on a generated answer, in tokens. The architecture already says
# answers are capped at about 200 tokens; until now nothing enforced it, so a
# model that decided to narrate ran until it stopped itself, and the slowest
# observed compose was 55.4 seconds.
#
# A cap is safe here in a way it would not be elsewhere, because of what the
# checks then do with a truncated answer. Cut mid sentence, the tail has no
# citation marker, so check 1 fails it and the whole answer becomes a refusal
# carrying the published passage. Cut exactly on a sentence boundary, every
# sentence that survives is still cited, so what prints is shorter but no less
# supported. Neither outcome is a new failure mode: the system already prefers
# a refusal to an answer it cannot verify.
#
# Measured on a prompt that invites a long answer: 939 tokens in 93.3 seconds
# uncapped, against 256 tokens in 23.7 seconds. It does not touch a normal
# answer, which runs to a few dozen tokens; it bounds the tail.
MAX_ANSWER_TOKENS = 256


class OllamaUnavailable(RuntimeError):
    """Ollama is not running, or the model is not pulled.

    Raised with the command that fixes it, because the most likely reader of
    this error is an assessor running the project for the first time.
    """


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(base_url=HOST, timeout=timeout)


def available() -> list[str]:
    """Model tags Ollama currently has, or a usable error explaining it does not."""
    try:
        with _client(10) as c:
            return [m["model"] for m in c.get("/api/tags").json().get("models", [])]
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(
            f"Ollama is not answering on {HOST} ({exc}). Start the Ollama "
            "application, or run `ollama serve`."
        ) from exc


def require(*models: str) -> None:
    """Fail before a long run rather than in the middle of one."""
    have = available()
    bare = {m.split(":")[0] for m in have}
    for model in models:
        if model not in have and model.split(":")[0] not in bare:
            raise OllamaUnavailable(
                f"model {model!r} is not pulled. Run: ollama pull {model}\n"
                f"Available: {', '.join(have) or '(none)'}"
            )


def _check(vectors: list, expected: int, model: str) -> None:
    if len(vectors) != expected:
        raise OllamaUnavailable(f"asked for {expected} embeddings, got {len(vectors)}")
    for v in vectors:
        if len(v) != EMBED_DIMENSIONS:
            raise OllamaUnavailable(
                f"{model} returned {len(v)} dimensions, expected {EMBED_DIMENSIONS}. "
                "The schema and the index header both assume the declared size; "
                "fix the model tag rather than the assertion."
            )


def _embed_batch(c: httpx.Client, window: list[str], model: str,
                 attempts: int = 3) -> list[list[float]]:
    """One batch, retried, then split rather than abandoned.

    A forty-minute embedding run ended on a single transient 400 from a batch
    that succeeded on the next attempt. Retrying costs a second; not retrying
    costs the whole run, which is the difference between an assessor seeing the
    system work and seeing a stack trace.

    If retries are exhausted the batch is halved and each half tried on its own,
    so one genuinely unembeddable passage is isolated and named instead of
    taking 643 good ones down with it.
    """
    last = ""
    for attempt in range(attempts):
        try:
            r = c.post("/api/embed", json={"model": model, "input": window,
                                           "keep_alive": KEEP_ALIVE})
            r.raise_for_status()
            vectors = r.json().get("embeddings") or []
            _check(vectors, len(window), model)
            return vectors
        except httpx.HTTPError as exc:
            last = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))

    if len(window) > 1:
        half = len(window) // 2
        return (_embed_batch(c, window[:half], model, attempts)
                + _embed_batch(c, window[half:], model, attempts))

    raise OllamaUnavailable(
        f"embedding failed after {attempts} attempts on a single passage "
        f"({len(window[0])} characters). Ollama said: {last[:300]}"
    )


def embed(
    texts: list[str],
    model: str = EMBED_MODEL,
    batch: int = 16,
    timeout: float = 600,
    progress=None,
) -> list[list[float]]:
    """Embed a list of texts, in batches, returning one vector per input.

    Batched because 644 separate HTTP round trips cost more in overhead than in
    inference, and because Ollama holds the model loaded across a batch.
    """
    out: list[list[float]] = []
    with _client(timeout) as c:
        for start in range(0, len(texts), batch):
            window = texts[start:start + batch]
            out.extend(_embed_batch(c, window, model))
            if progress:
                progress(min(start + batch, len(texts)), len(texts))
    return out


def embed_one(text: str, model: str = EMBED_MODEL) -> list[float]:
    """One vector, for a question at query time.

    This is the call that most obviously pays for an unloaded model: 3033
    milliseconds when the embedding model has been evicted against 679 when it
    has not, for the same short question. `KEEP_ALIVE` travels with the batch
    request, so a question asked after a quiet spell is embedded at the warm
    price rather than the cold one.
    """
    return embed([text], model=model, batch=1, timeout=120)[0]


def generate(
    prompt: str,
    model: str = GENERATION_MODEL,
    system: str = "",
    timeout: float = 300,
    num_ctx: int = 8192,
    seed: int = 0,
    num_predict: int = MAX_ANSWER_TOKENS,
) -> tuple[str, float]:
    """One completion at temperature zero. Returns the text and the seconds it took.

    Latency is returned rather than logged because it decides something: the
    trade's tolerance is about ten seconds, and whether the demonstration runs
    live or from a transcript depends on the measured number.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,          # qwen3.5 emits reasoning blocks unless told not to
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if system:
        body["system"] = system

    started = time.perf_counter()
    try:
        with _client(timeout) as c:
            r = c.post("/api/generate", json=body)
            r.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"generation call failed: {exc}") from exc
    elapsed = time.perf_counter() - started

    try:
        return r.json().get("response", "").strip(), elapsed
    except json.JSONDecodeError as exc:
        raise OllamaUnavailable(f"unparseable response from {model}: {exc}") from exc


def warm(model: str = GENERATION_MODEL) -> bool:
    """Load the model before it is needed. Returns whether it worked.

    One token generated for nothing, so that the weights are resident and
    `KEEP_ALIVE` has started counting before the first real question arrives.
    Without it the load cost, about 7.4 seconds, is paid by whoever asks first,
    and in a seven minute demonstration that is the assessor watching a blank
    terminal while nothing appears to happen.

    It returns a bool and swallows `OllamaUnavailable` rather than raising,
    because warming is an optimisation and must never be the reason a program
    fails to start. A machine with no Ollama running should reach the point
    where it can print a usable error about the question it was asked, not die
    here on a call nobody requested.
    """
    try:
        generate("ok", model=model, timeout=120, num_predict=1)
    except OllamaUnavailable:
        return False
    return True
