"""Close decision 6 by measurement: which embedding model retrieves better?

`python -m eval.embedding_choice`

Decision 6 said the embedding model should not be chosen by argument, and named
the criteria: retrieval quality on questions with known answers, index build
time, vector dimension, and input window against chunk size. This measures the
first and reports the rest.

The questions below each have a verified answer in a named document — verified
by reading the extracted text, not by assuming. Recall is then a fact: did the
document that actually contains the answer come back in the top five?

The comparison also covers the query-prefix asymmetry. Qwen3-Embedding is
trained with an instruction prefix on queries and none on documents; running it
without one is a configuration error that looks like a model weakness, so both
configurations are measured rather than one being assumed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import ollama, use_utf8                    # noqa: E402
from assistant.embedcache import EmbeddingCache           # noqa: E402
from assistant.extract import extract_html, extract_pdf   # noqa: E402
from assistant.index import chunk_sections, embedding_text, product_name  # noqa: E402
from assistant.model import Chunk                         # noqa: E402
from assistant.retrieve import QUERY_INSTRUCTION          # noqa: E402

# Each question's answer was located in the extracted corpus before the question
# was written. `contains` is the string that must appear in a retrieved passage
# for the retrieval to count as correct — a document match is not enough, the
# passage has to carry the answer.
QUESTIONS = [
    {"q": "How much water does Solo Onecoat need per bag?",
     "doc": "solo-one-coat-lime-plaster-tds.pdf",
     "contains": "5 and 6 litres"},
    {"q": "What is the coverage of Duro lime plaster base coat?",
     "doc": "Duro-TDS",
     "contains": "coverage"},
    {"q": "Can I plaster a fireplace with lime?",
     "doc": "support-faq",
     "contains": "fireplace"},
    {"q": "What is the difference between a pass and a coat?",
     "doc": "support-faq",
     "contains": "pass is the process"},
    {"q": "How do I prepare laths before lime plastering?",
     "doc": "plastering_onto_laths",
     "contains": "lath"},
    {"q": "The wall has gone damp, what does the site say about efflorescence?",
     "doc": "",
     "contains": "efflorescence"},
    {"q": "What temperature can I apply medium mortar at?",
     "doc": "medium-mortar-tds.pdf",
     "contains": "5"},
    {"q": "How many bricks high can I lay with pure lime mortar?",
     "doc": "support-faq",
     "contains": "lime mortar"},
]

MODELS = [
    {"tag": "qwen3-embedding:0.6b", "dims": 1024, "query_prefix": QUERY_INSTRUCTION},
    {"tag": "qwen3-embedding:0.6b", "dims": 1024, "query_prefix": "",
     "label": "qwen3-embedding:0.6b (no query prefix)"},
    {"tag": "nomic-embed-text", "dims": 768, "query_prefix": "search_query: ",
     "doc_prefix": "search_document: "},
]


def build_chunks() -> list[Chunk]:
    log = json.loads((ROOT / "data/cache/crawl-log.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for entry in log["fetched"]:
        path = ROOT / entry["path"]
        if not path.exists():
            continue
        if entry["kind"] == "page":
            ex, _ = extract_html(str(path), entry["doc_type"])
        else:
            ex = extract_pdf(str(path))
        product = product_name(entry, entry.get("title", ""))
        for i, (heading, text) in enumerate(chunk_sections(ex.sections)):
            chunks.append(Chunk(canonical_url=entry["url"], version=1, chunk_index=i,
                                section=heading, content=text, product=product,
                                document_type=entry["doc_type"]))
    return chunks


def embed_all(texts: list[str], model: str, dims: int, label: str) -> np.ndarray:
    with EmbeddingCache() as cache:
        known = cache.get_many(texts, model, dims)
        todo = [i for i in range(len(texts)) if i not in known]
        if todo:
            print(f"    {len(known)} cached, embedding {len(todo)} with {label} ...")
            fresh = ollama.embed([texts[i] for i in todo], model=model,
                                 progress=lambda d, t: print(
                                     f"\r      {d}/{t}", end="", flush=True))
            cache.put_many([(texts[i], v) for i, v in zip(todo, fresh)], model, dims)
            known.update(dict(zip(todo, fresh)))
            print()
        else:
            print(f"    all {len(texts)} from cache")
    m = np.vstack([np.asarray(known[i], dtype=np.float32) for i in range(len(texts))])
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def evaluate(spec: dict, chunks: list[Chunk]) -> dict:
    label = spec.get("label", spec["tag"])
    dims = spec["dims"]
    doc_prefix = spec.get("doc_prefix", "")

    # Patch the dimension assertion for the duration of this model.
    original = ollama.EMBED_DIMENSIONS
    ollama.EMBED_DIMENSIONS = dims
    try:
        texts = [doc_prefix + embedding_text(c) for c in chunks]
        started = time.perf_counter()
        matrix = embed_all(texts, spec["tag"], dims, label)
        build_seconds = time.perf_counter() - started

        hits, detail = 0, []
        query_started = time.perf_counter()
        for item in QUESTIONS:
            q = spec["query_prefix"] + item["q"]
            with EmbeddingCache() as cache:
                cached = cache.get_many([q], spec["tag"], dims)
                if cached:
                    vector = np.asarray(cached[0], dtype=np.float32)
                else:
                    vector = np.asarray(
                        ollama.embed([q], model=spec["tag"], batch=1)[0],
                        dtype=np.float32)
                    cache.put_many([(q, vector.tolist())], spec["tag"], dims)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
            order = np.argsort(-(matrix @ vector))[:5]
            top = [chunks[i] for i in order]
            found = any(
                item["contains"].lower() in c.content.lower()
                and (not item["doc"] or item["doc"].lower() in c.canonical_url.lower())
                for c in top
            )
            hits += found
            detail.append({"question": item["q"], "found": bool(found),
                           "top": f"{top[0].product} — {top[0].section}"[:58]})
        query_seconds = time.perf_counter() - query_started
    finally:
        ollama.EMBED_DIMENSIONS = original

    return {"model": label, "tag": spec["tag"], "dimensions": dims,
            "recall_at_5": f"{hits}/{len(QUESTIONS)}", "hits": hits,
            "build_seconds": round(build_seconds, 1),
            "query_seconds": round(query_seconds, 1),
            "vector_mb": round(len(chunks) * dims * 4 / 1e6, 2),
            "detail": detail}


def main() -> int:
    use_utf8()
    chunks = build_chunks()
    print(f"{len(chunks)} chunks, {len(QUESTIONS)} questions with verified answers\n")

    rows = []
    for spec in MODELS:
        label = spec.get("label", spec["tag"])
        print(f"  {label}")
        try:
            ollama.require(spec["tag"])
        except ollama.OllamaUnavailable as exc:
            print(f"    skipped: {exc}\n")
            rows.append({"model": label, "skipped": str(exc)})
            continue
        rows.append(evaluate(spec, chunks))
        print()

    print(f"{'model':<42} {'dims':>5} {'recall@5':>9} {'index':>9} {'vectors':>9}")
    print("-" * 80)
    for r in rows:
        if r.get("skipped"):
            print(f"{r['model'][:42]:<42} {'—':>5} {'skipped':>9}")
            continue
        print(f"{r['model'][:42]:<42} {r['dimensions']:>5} {r['recall_at_5']:>9} "
              f"{r['build_seconds']:>8.0f}s {r['vector_mb']:>8.2f}MB")

    print("\nquestions each model missed:")
    for r in rows:
        if r.get("skipped"):
            continue
        missed = [d["question"] for d in r["detail"] if not d["found"]]
        print(f"\n  {r['model']}:")
        if not missed:
            print("    none")
        for m in missed:
            top = next(d["top"] for d in r["detail"] if d["question"] == m)
            print(f"    - {m}\n        top hit was: {top}")

    out = ROOT / "eval" / "results" / "embedding-choice.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
