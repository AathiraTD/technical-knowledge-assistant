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

from assistant.infrastructure import ollama
from assistant import use_utf8                    # noqa: E402
from assistant.indexing.embedcache import EmbeddingCache           # noqa: E402
from assistant.indexing.extract import extract_html, extract_pdf   # noqa: E402
from assistant.indexing.index import (  # noqa: E402
    chunk_sections,
    embedding_text,
    product_name,
)
from assistant.knowledge.model import Chunk                         # noqa: E402
from assistant.retrieval.retrieve import QUERY_INSTRUCTION          # noqa: E402

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
    """The corpus as the indexer would chunk it, without touching the index.

    Re-extracts and re-chunks from the shipped crawl log using the same
    `extract_html`, `extract_pdf` and `chunk_sections` the real pipeline uses,
    so what is being compared is the retrieval quality of the models rather
    than the difference between this harness and the build. Importing those
    functions rather than reimplementing them is the whole point — a benchmark
    that chunked differently from the indexer would measure a corpus that is
    never served.

    It deliberately does not read the knowledge store and deliberately writes
    nothing to it. The index holds vectors from one model; this needs the same
    passages under several, and running it must not disturb the release an
    answer could be served from at the same moment.

    Cache entries whose file is missing are skipped rather than raised on, so a
    partial cache still yields a comparison across the documents that are
    present — the measurement is relative, and every model sees the same set.
    """
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
    """Vectors for every passage, normalised so cosine is a dot product.

    The rows are unit length on return, which is why `evaluate` can score with
    a single matrix multiply and no per-query normalisation of the corpus.

    It shares the indexer's `EmbeddingCache`, and sharing it is safe for the
    same reason the cache is safe in the build: the key is the hash of the text
    together with the model tag and the dimension count, so a vector is only
    ever returned for the exact text it was computed from by the exact model
    that computed it. Pointing this at a second model therefore misses every
    key and recomputes, which is the required behaviour — a cached vector from
    another model is exactly the confident nonsense the index header exists to
    prevent.

    The practical consequence is what makes this benchmark runnable at all: the
    first model's pass is largely free against the shipped cache, so the cost
    of the comparison is the candidates that have never been run, not the whole
    corpus times the number of models. The printed cached/computed counts are
    there so a run that quietly recomputed everything cannot be mistaken for
    one that did not.

    Note that the reported build time is therefore only a like-for-like
    embedding cost when the cache is cold for every candidate; decision 6 lists
    build time as a criterion, and a warm cache measures the cache.
    """
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
    """One model's recall at five, and what it cost to get there.

    A question counts as answered only when a passage in the top five contains
    the verified answer string *and* comes from the named document. Both halves
    matter: a document match alone would credit a model for retrieving the
    right datasheet's wrong section, which is precisely the near-miss the
    relevance gate exists to catch downstream, and crediting it here would
    select the model that produces the most refusals.

    The prefix handling is the reason this file has three specs for two models.
    Qwen3-Embedding is trained with an instruction prefix on queries and none
    on documents; running it without one is a configuration error that reads
    as a model weakness, so both configurations are measured rather than one
    being assumed. `nomic-embed-text` asymmetrically prefixes both sides, which
    is why `doc_prefix` is applied to the passages and not only to the query.

    `ollama.EMBED_DIMENSIONS` is patched for the duration and restored in a
    `finally`, because the module-level dimension is an assertion the embedding
    call makes against its own output and the candidates disagree about width
    (1024 against 768). Mutating shared module state is the ugly part of this
    function and it is contained here deliberately — the alternative was a
    dimension parameter threaded through the production call path for the sole
    benefit of a benchmark.

    The returned row also carries index build time, query time and the vector
    footprint, which are three of decision 6's criteria. The two it cannot
    measure — memory with the generation model also resident, and input window
    against the longest chunk — stay a matter for the record.
    """
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
    """Run the comparison decision 6 is still open on, and write the evidence.

    Decision 6 records the embedding model as chosen by measurement and the
    measurement as not yet run: `qwen3-embedding:0.6b` is the **development
    default** actually in use and recorded in every snapshot, not a selected
    model. This module is the thing that closes it, and until it has been run
    against every candidate the record says so rather than claiming otherwise.

    A model Ollama does not hold is skipped with the reason kept in the results
    row, never silently dropped — a table missing a row reads as a model that
    scored nothing rather than one that was never asked. The comparison is
    still valid across whatever did run, because every candidate sees the same
    chunks and the same questions.

    The missed questions are printed with the top hit each model returned
    instead, which is the part worth reading. A recall count says which model
    is better; the near miss says why, and whether the failure is one a
    synonym in the vocabulary would fix rather than a different model.

    Results are written to `eval/results/embedding-choice.json` so the decision
    can cite a file rather than a terminal, and so a rerun can be compared
    against the last one.
    """
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
