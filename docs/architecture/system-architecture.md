# System architecture — Lime Green Technical Assistant

High-level architecture derived from the mental-model record (`docs/Lime Green Technical Assistant - Mental Model and Working Record (1).docx`), sections 9 (How), 10 (Bound) and 11 (Build), cross-checked against the live site inventory in `docs/architecture/decision-0-notes.md`.

Two pieces are intentionally left open — **embedding model**, **vector storage** and **local generation model** are marked "yet to decide" pending the latency test in decision 1 (agent prompt, Appendix A) and are not blocking the rest of the pipeline design.

```mermaid
flowchart TD
    subgraph SRC["Data source — lime-green.co.uk (live site, decision 0)"]
        WEB["Product & category pages (~35)<br/>FAQ page (~41 Q&A)<br/>Knowledge-base articles (~15)<br/>Technical datasheets — PDF (~30)<br/>Contact / find-a-supplier pages"]
    end

    subgraph INGEST["Ingest & Know — Stage 5 (Build)"]
        CRAWL["Crawler + disk cache<br/>(sitemap-driven, rate-limited, robots.txt respected)"]
        EXTRACT["Text extraction<br/>boilerplate stripped<br/>classified by link text, not filename"]
        CHUNK["Chunking per document shape<br/>FAQ = 1 Q&A per chunk<br/>TDS = split by section<br/>KB = split by heading<br/>(caveats kept with their figures)"]
        MANIFEST["Source manifest<br/>authority rank · audience tag · date · link<br/>+ excluded-document list (SDS/DoP/Carbon/EPD)"]
        CRAWL --> EXTRACT --> CHUNK --> MANIFEST
    end

    subgraph STORE["Knowledge store"]
        EMB["Embedding model<br/>— YET TO DECIDE —<br/>(candidates: qwen3-embedding:0.6b, nomic-embed-text)"]
        VEC["Vector storage<br/>— YET TO DECIDE —<br/>(candidates: numpy array, Chroma, FAISS)"]
        CHUNK -.-> EMB --> VEC
    end

    subgraph QUERY["Query time"]
        Q["User question<br/>(CLI: 'Ask a question:')"]
        CLARIFY["Clarify gate<br/>underspecified? -> state assumptions"]
        POLICY["Policy gate<br/>price / stock / competitor / compliance / health?"]
        Q --> CLARIFY --> POLICY
    end

    subgraph RETRIEVE["Retrieval"]
        SIM["Similarity search vs VEC<br/>per-document cap in top-k"]
        THRESH{"Best score<br/>vs threshold?"}
        POLICY -->|not routed| SIM --> THRESH
    end

    subgraph ROUTER["Deterministic router — 5 paths"]
        direction TB
        ROUTE["Route<br/>fixed referral text<br/>(no retrieval)"]
        EXTR["Extract<br/>one dominant doc<br/>quote verbatim + cite, LLM bypassed/minimal"]
        COMPOSE["Compose<br/>>=2 docs above threshold<br/>LLM synthesises with numbered markers"]
        DEFER["Cited hand-off<br/>top passage itself says 'contact us'"]
        REFUSE["Refuse<br/>nothing above threshold,<br/>or a guardrail check fails"]
    end

    POLICY -->|policy pattern matched| ROUTE
    THRESH -->|one doc dominates, factual| EXTR
    THRESH -->|multi-doc, synthesis-shaped| COMPOSE
    THRESH -->|top passage defers| DEFER
    THRESH -->|below threshold| REFUSE

    subgraph LLM["Local LLM — YET TO DECIDE"]
        MODEL["Generation model via Ollama<br/>(candidates: qwen3:4b-instruct, granite4.2:3b)<br/>context-only prompt, temperature zero"]
    end
    COMPOSE --> MODEL

    subgraph GUARD["Guardrails — post-generation checks"]
        CITE["Citation-marker check<br/>marker must point at a retrieved passage"]
        NUM["Numbers-verbatim check<br/>figure must appear word-for-word in cited passage"]
        QUAL["Qualifier/caveat travels with its number"]
        MODEL --> CITE --> NUM --> QUAL
    end

    subgraph MODE["Audience filter"]
        FLAG{"Staff or public mode?"}
        FLAG -->|staff, on refuse| CAND["Show nearest candidates + scores"]
        FLAG -->|public, on refuse| HANDOFF["Hand off to technical team<br/>(contact info from crawled contact page)"]
    end
    SIM -. audience-tagged retrieval filter .-> FLAG
    REFUSE --> FLAG

    subgraph OUT["Output — CLI"]
        REPLY["Composite reply template<br/>Answer / Sources / Refusal-or-Handoff<br/>(brief's exact format, doc name + URL)"]
    end
    ROUTE --> REPLY
    EXTR --> REPLY
    QUAL --> REPLY
    DEFER --> REPLY
    CAND --> REPLY
    HANDOFF --> REPLY

    subgraph EVAL["Evaluation harness — offline, separate entry point"]
        SITU["7 transcript situations<br/>(lookup / synthesis / near-miss / far-miss)"]
        PROBE["Probe suite<br/>one-line guardrail checks"]
        SWEEP["Threshold sweep<br/>chosen threshold, and +/-0.1"]
    end
    REPLY -.-> EVAL

    MANIFEST --> STORE
```

## Reading the diagram

- **Route through the router is deterministic** (§9.2 of the record) — no model decides which of the five paths a question takes; a policy-pattern match, retrieval shape, or the top passage's own wording decides it.
- **The model is a narrator, never the source of a fact** — it only runs on the Compose path; Extract, Route, Cited-hand-off and Refuse never call it (or call it minimally, to paraphrase quoted text).
- **Guardrails run after generation, before printing** — a failed check on the Compose path routes into `REFUSE` in practice, shown here simplified as a straight line to keep the diagram readable; the full failure-mode table is in §10.5 of the record.
- **Two boxes are marked "yet to decide" on purpose**: embedding model and vector store (decision 2/3) and the local generation model (decision 1). These are the last technology decisions per §11.1 — the pipeline shape above does not depend on which one is picked.

## Source file

Raw Mermaid source: [`system-architecture.mmd`](system-architecture.mmd) — importable directly into mermaid.live or the Mermaid Chart editor.
