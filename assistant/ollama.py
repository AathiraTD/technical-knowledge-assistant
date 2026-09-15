"""The Ollama client: two endpoints over HTTP, no client library.

Decision recorded among the five of no consequence — a dependency that wraps
`/api/embed` and `/api/generate` earns its place only if it does something
those two calls do not, and it does not.

What this module does carry is the temperature and the refusal to guess. The
generation call is pinned to temperature zero with a fixed seed, because a
reviewer re-running a transcript should get the transcript back. And the
embedding call fails loudly on a dimension it did not expect, rather than
writing a shorter vector into a column sized for a longer one.
"""

from __future__ import annotations

import json
import time

import httpx

HOST = "http://127.0.0.1:11434"

# DEVELOPMENT DEFAULT, not a final selection. Decision 6 says the embedding
# model is chosen by measurement on the same corpus and questions, and
# `eval/embedding_choice.py` is that measurement. Whatever is set here is
# recorded in the index header, and the engine refuses to query an index built
# with a different one — so changing it is safe, and forgetting to rebuild is
# not silently possible.
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_DIMENSIONS = 1024

GENERATION_MODEL = "qwen3.5:4b"


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
            r = c.post("/api/embed", json={"model": model, "input": window})
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
    """One vector, for a question at query time."""
    return embed([text], model=model, batch=1, timeout=120)[0]


def generate(
    prompt: str,
    model: str = GENERATION_MODEL,
    system: str = "",
    timeout: float = 300,
    num_ctx: int = 8192,
    seed: int = 0,
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
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "num_ctx": num_ctx,
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
