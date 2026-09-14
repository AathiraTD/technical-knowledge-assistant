# Architecture — C4-notation container view

Five containers of your own, two external systems, laid out as two flows that only meet at `Answer Engine`. This version is a hand-laid-out flowchart using C4 notation (`[Person]`, `[Container: ...]`, `[Software System]` labels, matching colors) rather than Mermaid's native `C4Container` diagram type — the native one kept fighting its own auto-layout once custom offsets were added, producing overlapping text and crossed lines. This version routes every line explicitly, so nothing overlaps regardless of renderer.

```mermaid
flowchart TD
    classDef person fill:#fff7ed,stroke:#fb923c,color:#9a3412,stroke-width:2px
    classDef website fill:#ecfeff,stroke:#22d3ee,color:#155e75,stroke-width:2px
    classDef llmClass fill:#f5f3ff,stroke:#a78bfa,color:#5b21b6,stroke-width:2px
    classDef receiver fill:#fefce8,stroke:#facc15,color:#854d0e,stroke-width:2px
    classDef cache fill:#f8fafc,stroke:#94a3b8,color:#334155,stroke-width:2px
    classDef worker fill:#f0fdfa,stroke:#2dd4bf,color:#115e59,stroke-width:2px
    classDef core fill:#eef2ff,stroke:#818cf8,color:#3730a3,stroke-width:2px
    classDef cli fill:#f0fdf4,stroke:#4ade80,color:#166534,stroke-width:2px

    subgraph QUERY["QUERY FLOW — every request, prototype and production"]
        direction TB
        USER(("Customer / trade / staff<br/>[Person]<br/>Asks product questions"))
        CLIN["CLI<br/>[Container: Python]<br/>Prints answer, sources, or a cited refusal"]
        USER --> CLIN
    end

    subgraph PROD["🔴 PRODUCTION ONLY — automatic delta pipeline (not built in the one-hour prototype)"]
        direction TB
        WEBN[["Lime Green website<br/>[Software System]<br/>Pages, FAQs, KB articles, datasheets"]]
        RECVN["Change Receiver<br/>[Container: webhook / poller]<br/>Detects new or changed content"]
        CACHEN["Content Cache<br/>[Container: disk store]<br/>Writes the delta immediately"]
        WORKN["Indexing Worker<br/>[Container: Python, async]<br/>Chunks + embeds only the delta<br/>atomic snapshot swap"]
        WEBN -->|"webhook on publish,<br/>or polled with conditional GET"| RECVN
        RECVN -->|"writes the delta"| CACHEN
        CACHEN -->|"queues the delta"| WORKN
    end

    COREN["Answer Engine<br/>[Container: Python]<br/>Retrieves, routes, verifies citations,<br/>answers or refuses"]
    LLMN[["Local LLM<br/>[Software System]<br/>Ollama — composes from retrieved passages only"]]

    CLIN -->|"passes question"| COREN
    WORKN -.->|"reads current<br/>index snapshot"| COREN
    COREN -->|"sends passages,<br/>gets composed text back"| LLMN

    class USER person
    class WEBN website
    class LLMN llmClass
    class RECVN receiver
    class CACHEN cache
    class WORKN worker
    class COREN core
    class CLIN cli

    style PROD fill:#fff1f2,stroke:#e11d48,stroke-width:3px,stroke-dasharray:6 4
    style QUERY fill:#f8fafc,stroke:#64748b,stroke-width:2px
```

## What's highlighted, and why

The whole delta pipeline (`Change Receiver → Content Cache → Indexing Worker`) sits inside a **red-bordered, red-titled box** labeled "🔴 PRODUCTION ONLY." That's deliberate, not decorative: everything in that box is roadmap material — none of it belongs in the one-hour prototype, which just does a single offline crawl-once script. The query flow (`user → CLI → Answer Engine → LLM`) is what both the prototype and production actually run; it sits in a plain gray box for contrast.

Say it this way on the slide: *"The core pipeline — ask, retrieve, answer — is what's built. The red box is what production adds: a hook that auto-triggers the delta flow, so only new or changed content is ever re-indexed."*

## Two flows, one meeting point

- **Background flow** (production only): `website → Change Receiver → Content Cache → Indexing Worker`. Runs on its own clock (webhook or short-interval poll), decoupled from query traffic entirely.
- **Query flow** (always): `user → CLI → Answer Engine → {index snapshot, Local LLM}`. Every query reads whichever snapshot is currently live; it never waits on indexing.

## Trigger: interval vs. event, and why

- **True webhook (push)** needs Lime Green's CMS to call an endpoint of ours on publish. Not buildable unilaterally against a site we don't own — it's a KTP-scope integration (the job description names "integration with the company's... website" directly), not a prototype feature.
- **Short-interval polling with conditional GET** (`If-None-Match` / `If-Modified-Since`) is the practical stand-in today: poll every few minutes, and an unchanged page costs a `304 Not Modified` — effectively free. It behaves like a hook without needing Lime Green's cooperation.
- **Write immediately, index asynchronously.** The `Content Cache` write happens the instant a change is detected; the `Indexing Worker` consumes a queue and may lag by seconds to minutes. This is why a query is never blocked on re-indexing, and why cost stays near zero — most polls and most indexing runs touch nothing.

## Three decisions resolved, not deferred

- **Retrieval index — in-memory numpy array, cosine similarity. No vector database.** At ~80 documents / a few hundred chunks, a dot product over the whole set is microseconds; Chroma or FAISS would add a dependency and nothing else at this scale.
- **Freshness — event-detected, asynchronously indexed. Not real-time-per-query, not blind daily batch.** A detected change is cached immediately; indexing catches up in the background; queries always read a consistent, if briefly stale, snapshot.
- **Embeddings — still needed, model still open.** Customers ask in their own words ("wall's gone damp and bubbly"); the corpus uses product vocabulary ("efflorescence"). Pure keyword search would miss too much of that gap, so *some* embedding model is worth having — which one (qwen3-embedding:0.6b vs nomic-embed-text) is a real decision 2, left open.
- **Local LLM — still open**, pending the latency test (decision 1 in the agent prompt). Shown here as an external system because, architecturally, it is one: a black box the Answer Engine calls over Ollama's API, swappable without touching the rest of the pipeline.

## Fallback if your Mermaid renderer doesn't support flowchart subgraphs with `style`

This should render anywhere modern (GitHub, mermaid.live, VS Code preview). If something strips `classDef`/`style` lines, [`system-architecture.mmd`](system-architecture.mmd) is a more detailed, older version of the same shape as a plain flowchart.
