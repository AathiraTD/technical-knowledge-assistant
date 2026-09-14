# Architecture — Technical Knowledge Assistant

Two diagrams, at two levels of detail, plus the live site inventory they're both grounded in. For the *why* behind each component (not just what connects to what), see [`architecture-rationale.md`](architecture-rationale.md).

## 1. Container view

The whole system inside one boundary, split into three groups: **Indexing path** (production only — the automatic re-index pipeline), **Retrieval data** (the shared store both paths touch), **Question-answering path** (MVP and production — what's actually built for the submission).

Source: [`diagrams/container-view.mmd`](diagrams/container-view.mmd) — paste directly into mermaid.live or Mermaid Chart.

```mermaid
flowchart TD
    classDef person fill:#fff7ed,stroke:#fb923c,color:#9a3412,stroke-width:2px
    classDef website fill:#ecfeff,stroke:#22d3ee,color:#155e75,stroke-width:2px
    classDef crm fill:#fdf4ff,stroke:#d946ef,color:#86198f,stroke-width:2px
    classDef llmClass fill:#f5f3ff,stroke:#a78bfa,color:#5b21b6,stroke-width:2px
    classDef receiver fill:#fefce8,stroke:#facc15,color:#854d0e,stroke-width:2px
    classDef cache fill:#f8fafc,stroke:#94a3b8,color:#334155,stroke-width:2px
    classDef queue fill:#fff1f2,stroke:#fb7185,color:#9f1239,stroke-width:2px
    classDef dlq fill:#fef2f2,stroke:#f87171,color:#991b1b,stroke-width:2px
    classDef worker fill:#f0fdfa,stroke:#2dd4bf,color:#115e59,stroke-width:2px
    classDef vectordb fill:#ecfeff,stroke:#22d3ee,color:#155e75,stroke-width:2px
    classDef metadb fill:#f0f9ff,stroke:#38bdf8,color:#0c4a6e,stroke-width:2px
    classDef core fill:#eef2ff,stroke:#818cf8,color:#3730a3,stroke-width:2px
    classDef cli fill:#f0fdf4,stroke:#4ade80,color:#166534,stroke-width:2px

    USER(("Customer, trade, or staff<br/>[Person]<br/>Asks product questions"))
    WEBN[["Lime Green website<br/>[Software System]<br/>Pages, FAQs, KB articles, datasheets"]]
    CRMN[["CRM<br/>[Software System]<br/>Future source — customer & enquiry records"]]
    LLMN[["Local LLM<br/>[Software System]<br/>Ollama — composes from retrieved passages only"]]

    subgraph APP["TECHNICAL KNOWLEDGE ASSISTANT"]
        direction TB

        subgraph INDEX["🔴 PRODUCTION — Indexing path<br/>(queue justified by multiple sources, not website size)"]
            direction TB
            RECVN["Change Receiver<br/>[Container: webhook / poller]<br/>Detects published or changed content"]
            CACHEN["Content Cache<br/>[Container: disk store]<br/>Stores raw changed content before indexing"]
            QUEUEN["Indexing Queue<br/>[Container: message queue]<br/>Buffers jobs from every source"]
            WORKN["Indexing Worker<br/>[Container: Python, async]<br/>Chunks + embeds changed docs<br/>retries with backoff in-process"]
            DLQN["Dead-Letter Queue<br/>[Container: message queue]<br/>Holds jobs that exceed the retry limit"]
            RECVN --> CACHEN --> QUEUEN --> WORKN
            WORKN -->|"after retry limit"| DLQN
        end

        subgraph DATA["Retrieval data"]
            direction TB
            VECN[("Vector Index<br/>[Container: vector database]<br/>Embeddings for semantic retrieval")]
            METAN[("Metadata Store<br/>[Container: document database]<br/>Source, chunk refs, indexing status")]
        end

        subgraph QA["Question-answering path — MVP and production"]
            direction TB
            CLIN["CLI<br/>[Container: Python]<br/>Displays answer, sources, or a cited refusal"]
            ENGN["Answer Engine<br/>[Container: Python]<br/>Retrieves passages, verifies citations,<br/>answers or refuses"]
            CLIN --> ENGN
        end
    end

    WEBN -->|"webhook on publish,<br/>or polled with conditional GET"| RECVN
    CRMN -.->|"webhook / export<br/>(future source)"| RECVN
    USER --> CLIN
    WORKN -->|"writes embeddings"| VECN
    WORKN -->|"writes metadata"| METAN
    ENGN -.->|"retrieves passages"| VECN
    ENGN -.->|"reads citations"| METAN
    ENGN -->|"composes from<br/>retrieved passages"| LLMN

    class USER person
    class WEBN website
    class CRMN crm
    class LLMN llmClass
    class RECVN receiver
    class CACHEN cache
    class QUEUEN queue
    class DLQN dlq
    class WORKN worker
    class VECN vectordb
    class METAN metadb
    class ENGN core
    class CLIN cli

    style APP fill:#fafafa,stroke:#334155,stroke-width:2px
    style INDEX fill:#fff1f2,stroke:#e11d48,stroke-width:3px,stroke-dasharray:6 4
    style DATA fill:#f0f9ff,stroke:#0ea5e9,stroke-width:2px
    style QA fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
```

## 2. Answer Engine detail

What "Answer Engine" in the diagram above actually does — the deterministic router and guardrail chain, derived from the mental-model record §9 (How), §10 (Bound), §11 (Build).

Source: [`diagrams/answer-engine-detail.mmd`](diagrams/answer-engine-detail.mmd)

```mermaid
flowchart TD
    subgraph QUERY["Query time"]
        Q["User question"]
        CLARIFY["Clarify gate<br/>underspecified? -> state assumptions"]
        POLICY["Policy gate<br/>price / stock / competitor / compliance / health?"]
        Q --> CLARIFY --> POLICY
    end

    subgraph RETRIEVE["Retrieval"]
        SIM["Similarity search<br/>per-document cap in top-k"]
        THRESH{"Best score<br/>vs threshold?"}
        POLICY -->|not routed| SIM --> THRESH
    end

    subgraph ROUTER["Deterministic router — 5 paths"]
        direction TB
        ROUTE["Route<br/>fixed referral text<br/>(no retrieval)"]
        EXTR["Extract<br/>one dominant doc<br/>quote verbatim + cite"]
        COMPOSE["Compose<br/>>=2 docs above threshold<br/>LLM synthesises with numbered markers"]
        DEFER["Cited hand-off<br/>top passage itself says 'contact us'"]
        REFUSE["Refuse<br/>nothing above threshold,<br/>or a guardrail check fails"]
    end

    POLICY -->|policy pattern matched| ROUTE
    THRESH -->|one doc dominates, factual| EXTR
    THRESH -->|multi-doc, synthesis-shaped| COMPOSE
    THRESH -->|top passage defers| DEFER
    THRESH -->|below threshold| REFUSE

    COMPOSE --> MODEL["Local LLM<br/>context-only prompt, temperature zero"]

    subgraph GUARD["Guardrails — post-generation checks"]
        CITE["Citation-marker check<br/>marker must point at a retrieved passage"]
        NUM["Numbers-verbatim check<br/>figure must appear word-for-word in cited passage"]
        QUAL["Qualifier/caveat travels with its number"]
        MODEL --> CITE --> NUM --> QUAL
    end

    subgraph MODE["Audience filter"]
        FLAG{"Staff or public mode?"}
        FLAG -->|staff, on refuse| CAND["Show nearest candidates + scores"]
        FLAG -->|public, on refuse| HANDOFF["Hand off to technical team"]
    end
    SIM -. audience-tagged filter .-> FLAG
    REFUSE --> FLAG

    subgraph OUT["Output — CLI"]
        REPLY["Answer / Sources / Refusal-or-Handoff<br/>(brief's exact format, doc name + URL)"]
    end
    ROUTE --> REPLY
    EXTR --> REPLY
    QUAL --> REPLY
    DEFER --> REPLY
    CAND --> REPLY
    HANDOFF --> REPLY
```

Three things worth saying out loud from this diagram:

- **Route through the router is deterministic** — no model decides which of the five paths a question takes; a policy-pattern match, retrieval shape, or the top passage's own wording decides it.
- **The model is a narrator, never the source of a fact** — it only runs on the Compose path.
- **Guardrails run after generation, before printing** — a failed check on the Compose path routes into `REFUSE`.

## Appendix: live site inventory (decision 0)

Pulled from `lime-green.co.uk`'s `sitemap.xml` and spot-checked pages. Corrects the mental-model record's hypothesis in §8.6 where noted.

- `robots.txt` allows all crawlers and points to a real, complete `sitemap.xml` — the record assumed this couldn't be confirmed; it can. Crawl by sitemap.
- **Product pages**: 6 categories (Lime Mortar, Lime Plaster, Lime Render, Insulation, Primers & Adhesives, Stone Repair), ~35 individual pages, plus `/products-by-colour` (24 colour pages).
- **Technical datasheets**: one PDF per product, plus up to 6 other PDF types per product (SDS, UK/EU DoP, Carbon Footprint, EPD, LRV) — confirms classifying by link text, not filename, and excluding everything except the TDS.
- **Knowledge base**: `/support/knowledgebase`, 15 articles (record assumed "about five" — correct this in §8.4/§8.7).
- **FAQ**: `/support/faq`, one page, **~41 questions across 6 sections** — record cites "17 items" throughout (§10.2, §8.7, §7.6); needs correcting wherever it appears.
- **Case studies**: 24, matching the named examples already in the record.

**Rough tier-one corpus size, corrected**: ~35 product pages + ~30 TDS PDFs + 1 FAQ page (41 Q&As) + 15 KB articles + contact/find-a-supplier ≈ **83 addressable units** — above the record's "about forty documents" cap. The spend-order in §8.7/§10.2 will need to actually cut, not just theoretically allow for it.

**Not yet checked**: Warmshell subsection pages (4); `find-a-supplier`/`contact` page structure (needed for hand-off text); whether datasheet PDFs keep qualifiers/caveats in the same section as their figures; actual PDF text-extraction quality.
