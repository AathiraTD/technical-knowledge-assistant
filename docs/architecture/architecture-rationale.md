# Architecture rationale — Technical Knowledge Assistant

Plain explanation of the design in [`architecture.md`](architecture.md): what each piece is, why it's there, what it buys, and where it's weak. One line per point, no padding.

## The shape, in one sentence

A question comes in, gets answered from retrieved passages only (never invented), with a citation for every fact — and separately, content changes get detected and re-indexed without touching the live query path. Two paths, one shared store between them.

## MVP vs production

Only the **Question-answering path** is built for the one-hour take-home. The **Indexing path** (queue, receiver, dead-letter handling) is roadmap — drawn to show the target architecture, not the submission.

---

## External systems

| Component | What it is | Why it's there |
|---|---|---|
| **Customer, trade, or staff** | The person asking a question | The whole system exists to serve this one interaction |
| **Lime Green website** | Product pages, FAQ, knowledge base, datasheets | The only content source in scope for the MVP |
| **CRM** | Customer and enquiry records | Future second content source — not built, shown to justify the queue (see below) |
| **Local LLM (Ollama)** | The generation model | Composes the answer text — never the source of a fact, only the narrator over retrieved passages |

---

## Indexing path (production only)

| Component | What it does | Why it exists | Shortcoming |
|---|---|---|---|
| **Change Receiver** | Detects new or changed content — webhook if the source can push it, polling with conditional GET if it can't | Nothing gets re-indexed until something changes; this is the trigger | Website has no webhook today — polling is a stand-in, not a true hook |
| **Content Cache** | Writes the raw changed content to disk immediately | Decouples "detect a change" from "process a change" — a slow or failing indexer never blocks detection | Adds a step; unnecessary at current scale (a direct write would work fine) |
| **Indexing Queue** | Buffers indexing jobs from every source | Only earns its place once more than one source (website + CRM + others) writes in independently — it's not for the website's size | At today's scale (one source, low change rate) this is pure overhead |
| **Indexing Worker** | Chunks and embeds changed documents; retries failures with backoff; swaps the index atomically | Keeps the live index always in a consistent, complete state — a query never sees a half-updated index | Async means brief staleness between a change landing and it being searchable |
| **Dead-Letter Queue** | Holds jobs that fail indexing repeatedly | A permanent failure is captured, not silently dropped | No operational story yet for who reviews it or how often |

## Retrieval data

| Component | What it does | Why it exists | Shortcoming |
|---|---|---|---|
| **Vector Index** | Stores embeddings for semantic (meaning-based) search | Customers ask in their own words ("wall's gone damp"); the corpus uses product vocabulary ("efflorescence") — keyword search alone would miss the match | Chosen as a real vector database ahead of need — at ~80 documents a flat in-memory array does the same job with one less dependency; kept anyway as the target architecture |
| **Metadata Store** | Holds source name, chunk reference, document date, indexing status | Every answer must cite its source by name — this is what makes that possible and lets "newest wins" work when two document versions disagree | A second store to keep in sync with the Vector Index; a bug here breaks citations even if retrieval is correct |

## Question-answering path (MVP and production)

| Component | What it does | Why it exists | Shortcoming |
|---|---|---|---|
| **CLI** | Takes the question, prints the answer, sources, or a refusal | The brief's whole required interface — nothing more | Single-user, single-turn only; no memory across questions |
| **Answer Engine** | Retrieves passages, decides how to answer (quote, compose, route, or refuse), checks every citation before printing | This is where hallucination is actually prevented — not by asking the model nicely, but by refusing to print anything that isn't traceable to a retrieved passage | Only checks that a citation points somewhere real — it doesn't fully guarantee the surrounding sentence is a faithful summary of that passage |

---

## Whole-system shortcomings, said plainly

- **The indexing path is over-built for what this project needs today.** It's a legitimate answer to "how would this scale," but none of it belongs in the actual one-hour submission.
- **The vector database is a deliberate choice, not a necessity.** At current scale a numpy array is equally correct and simpler. Kept because it's the shape the production system will actually need, at the cost of one dependency now.
- **No caching, no rate limiting, no concurrency handling.** Fine for one user at a time; the diagram doesn't yet show what changes under real traffic (see the "scale, scenario by scenario" table in the mental-model record for that).
- **Freshness has a lag, by design.** A change is cached instantly but may take seconds to minutes to become searchable — acceptable for a slow-moving corpus, worth stating out loud if asked.
- **The CRM box is speculative.** It's there to explain *why* a queue is justified at all, not because CRM integration is planned or scoped.

## Ready-made answers, if asked live

*"Why a vector database at only ~80 documents?"* — At today's scale a flat array would work just as well. This is designed as the architecture the KTP project actually needs, and a vector store gives a clean swap point without re-architecting the retrieval interface later, at the cost of one extra dependency now.

*"Why a queue at all?"* — Not because the website has a lot of pages — even at full scope this corpus stays in the low thousands. It's because production eventually has multiple independent sources writing into the same pipeline — the website, the CRM, eventually staff knowledge capture — and a queue is what decouples uncoordinated producers from one consumer.
