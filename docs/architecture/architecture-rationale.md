# Architecture rationale — Technical Knowledge Assistant

Plain explanation of the design in [`architecture.md`](architecture.md): what each piece is, why it's there, what it buys, where it's weak. One line per point. Row tags mark what is built for the submission (green on the diagram) and what is production roadmap (amber, rose and pink).

## The shape, in one sentence

A question is answered only from retrieved passages, every fact cited, with a refusal that still hands over whatever is published when the material runs out — and content is indexed once by hand for the submission, re-indexed only on change in production.

## Four decisions

| Decision | Call | Why | Cost accepted |
|---|---|---|---|
| **1 · Corpus boundary** | Everything technical the site publishes — 94 units, six to eight hundred chunks. Out: near-duplicate colour pages, boilerplate category pages, non-technical documents (the list is in the appendix of `architecture.md`). Three products have no datasheet — Solo Filler, Warmshell 660 Mesh, Silic8 AeroGel Adhesive — and a probe targets each | A rule the panel can check, not a curated list. Cutting datasheets to the products the test questions touch would be indexing what the evaluation needs. The real limits are retrieval pollution and evaluation honesty, not a round number | Embedding is minutes. The real cost is extraction QA on 31 unprobed datasheets and 3 unextracted guides, paid through the ingestion report and bounded by the spend order — anything that misses the box is dropped in that order and listed in the manifest as excluded, by name and link |
| **2 · Extract path** | Verbatim by code: the top passage printed whole with its citation (two passages on the calculation edge, coverage and pack size), by the same quote-and-cite code that prints the cited hand-off and the refusal. A relevance gate refuses with "not stated in the indexed material" when the asked-for property or substrate term, or a synonym, is absent from the passage — the near-miss case | The quote-and-cite printer exists anyway, so code on Extract adds no mechanism and removes a prompt variant. On a lookup the model can add nothing but paraphrase drift, which the checks would then refuse — a false refusal on the brief's first test type. The record's own estimate — prompt reading plus about 80 tokens at 15–25 tokens per second — puts a two-passage model call near the ten-second trade target; measured at build. Compliance: "use a local LLM to formulate an answer" is met where formulation happens, on Compose; the record already prints Route, Cited hand-off and Refuse without the model | The straightforward test question shows a quoted passage with a citation rather than model prose. Say so on the slide: the model formulates where there is something to formulate. A model quote-and-stop over two passages stays a roadmap option, gated on a measured warm call under the 10-second target; nothing in the diagrams assumes it |
| **3 · Chunk store** | One store: embeddings plus payload per chunk, and one manifest of documents, swapped together. Prototype: one array and one manifest file. Production: a vector database with payload, or a vector column in the database the company already runs | That is how vector stores work. A second store for metadata is a sync bug waiting to break citations | The prototype store is a file, not a service: one process, one user. Concurrency is a production change, not a code change |
| **4 · Second source for the queue** | Staff-knowledge capture: the agreed answer set, the failure library and the compatibility matrix as text. The policy list goes to the authored configuration. CRM becomes a channel adapter | The record classes the CRM as a system of record — "integration, not corpus", "route, never generate", personal data looked up exactly — so nothing in it is answer material. Staff capture is the KTP's core activity and produces retrievable text | The queue stays roadmap. If capture turns out to yield structured configuration only, there is no second chunk producer and the queue collapses to a scheduled re-index |

---

## External

| Component | What it is | Why it's there |
|---|---|---|
| **Lime Green website** | The source: 94 technical units, inventoried in the appendix of `architecture.md` | The only source in scope for the submission |
| **Embedding model** (Ollama) | Turns chunks and questions into vectors; qwen3-embedding:0.6b or nomic-embed-text, verified at build | Customers say "wall's gone damp"; the datasheet says "efflorescence" — keyword search misses that |
| **Generation model** (Ollama) | Composes over retrieved passages on the Compose path; qwen3:4b-instruct, granite4.2:3b only if it emits reasoning blocks | Never the source of a fact; the narrator over them |
| **Staff-knowledge capture** (production) | Agreed answer set, failure library, compatibility matrix as text | The second producer that justifies the queue |

## Indexing path

| Component | What it does | Why it exists | Shortcoming |
|---|---|---|---|
| **Indexer** (built; run once by hand) | Crawl by sitemap, cache, extract (PyMuPDF for PDFs), classify by link text, strip boilerplate and hazard blocks, chunk by heading with bullets and labelled sub-paragraphs kept whole, tag each document's caveat sentences (temperature limits, DIY suitability, incompatible boards — deferral sentences are not tagged, they have their own path), embed, write the index, the product, colour and merchant name lists, and the ingestion report (number–unit pairs, suspected errors flagged), swap atomically | Nothing answers without it; run once so the submission is offline and reproducible | Chunk quality depends on datasheet layout, and older sheets put temperature limits under Mixing rather than Application |
| **Content cache** (built; shipped) | Raw pages and PDFs on disk | The assessors run it without the network; the crawl is not repeated on their machine | Ships site content with the code — fine for a private submission, a licensing question for anything public |
| **Change receiver** (production) | Reads the sitemap's last-modified; falls back to conditional GET per page; enqueues changed URLs | Detects change in one request instead of one per document; the politest crawl possible | Not a true hook — the site cannot push; the polling interval is a choice |
| **Indexing queue** (production) | Buffers change jobs from both sources; retries with backoff; dead-letters after the limit | Decouples two producers from one consumer once staff capture joins the website | Overhead at one source; a change is searchable seconds to minutes after detection; nobody yet owns the dead-letter list |

## Retrieval data

| Component | What it does | Why it exists | Shortcoming |
|---|---|---|---|
| **Chunk store** (built) | Per chunk: embedding, product, document, section, date, authority rank, audience tags (public, trade, staff). Manifest: every document with type, authority rank, audience tag, date, link and its caveat sentences; excluded documents by name and link; contact text and the product, colour and merchant name lists from the crawl; embedding-model tag and chunking version — the engine refuses to run on a mismatch of either | Retrieval, citation by document name, newest-wins, the staff/public filter, document requests and the appended caveats all read from here | Prototype is an array in a file: one process, one user |
| **Authored configuration** (built) | Routing table; slot, calculation, symptom and property vocabularies with synonyms; deferral phrases; authority and audience rules per source; exclusion rules. The property vocabulary is seeded from the datasheets' own field names at ingestion (coverage, water, thickness, temperature, strength, permeability) plus hand-written synonyms | Half the guardrails read their data from here; the indexer applies its rules; hand-written, so the technical team can inspect it | Hand-authored lists miss paraphrases; an unknown property falls through to the threshold and citation checks |

## Question-answering path

| Component | What it does | Why it exists | Shortcoming |
|---|---|---|---|
| **Answer engine** (built) | Split by topic → policy gate per part → slot detection (values used when cued) → audience-filtered retrieval → deterministic router, evaluated threshold and deferral first, then symptoms, the relevance gate, uncued load-bearing slots (per option or ask back), calculation, extract, compose → model on Compose only → six checks → document caveats appended by code → hand-off with value | Hallucination is prevented by refusing to print what is not traceable, not by asking the model nicely | Qualitative synthesis — suitability, compatibility — is reduced, not eliminated: the stated weakest point |
| **CLI** (built) | Question and audience set in; answer, sources, refusal, diagnostics out | The brief's required interface, in the brief's format | Single user, single turn; real enquiries run six turns; the audience set is asserted, not authenticated |
| **Evaluation harness** (built) | Seven situations, probe suite, threshold sweep, pass/fail, self-describing header, one synthetic staff-tagged fixture that must be invisible in public mode | It is the artefact the panel reads; written before the code | Seven situations is evidence, not statistics |
| **Channel adapters** (production) | Website widget, CRM, training platform, calling the engine as a library | The job description's "shared data store used by multiple users and systems" | Not built; the engine is a library so that they can be thin |
| **Identity and audience** (production) | Resolves the caller to an audience set — public, trade or staff; anonymous gets public only | The audience filter is only as trustworthy as the claim behind it, and the three segments of §6.2 need three audiences, not the binary internal/external fork of §6.4: trade partners see lead times, stock and kit lists that the public does not | Not built. At the CLI today the audience set is asserted, so a caller could claim staff |
| **Serving layer** (production) | Generation queue with a visible wait, per-session rate limiting, extract-only degradation under load | Retrieval scales trivially — a snapshot read is lock-free; generation is the only bottleneck, one at a time per Ollama instance. Under load the system drops the expensive path, never a check | The compose answers that a busy hour needs most are the first to go; a queue with a visible wait is honest but still a wait |
| **Answer cache** (production) | Composite parts keyed on template, slots, audience set and index version | Roughly eighty per cent of demand is nineteen question templates (§7.5), and temperature-zero answers over a fixed snapshot are safe to reuse. The atomic swap is a free cache epoch — a re-index expires everything with no invalidation logic | The key must carry the audience set or a staff answer reaches a public caller; keyed too loosely it serves the right template for the wrong substrate |

---

## Whole-system shortcomings, said plainly

- **Qualitative synthesis is the weakest point.** The six checks bound numbers, names, attribution and the asked-for term; they reduce, not eliminate, an invented "this is fine on cob".
- **Single turn.** Real enquiries run six turns with photos; the prototype answers turn one and says it cannot see photographs. An uncued substrate becomes an ask-back, which the user answers by asking again.
- **The relevance gate is lexical, with synonyms.** A passage that names the property without answering it passes the gate; what then prints is what the sheet actually says ("high breathability", no figure), which is the honest result, not a hallucination.
- **The audience set is asserted, not authenticated.** The filter is real and enforced in code at retrieval; what is missing is the identity behind the claim. One user, one machine, no accounts — and the not-built list says so.
- **One user at a time.** No queue, no rate limit, no cache; the concurrency story is drawn as production boxes and argued, not built.
- **Staff mode always extracts; staff compose is roadmap.** The advisor verifies from the passage text, and time is the constraint. It means an uncued inside/outside on a staff selection question gets the passages, not a per-option answer.
- **Staff drafting mode is not built.** The record lists it as a requirement (10.3), traced to a specific staff question (10.6), and leaves its place in the first version to the author's call (10.8); it is generation by definition, needs pasted-email handling and a no-commitments rule, and does not fit the hour. Roadmap.
- **Datasheet quality is inherited.** Sheets date from 2015 to 2025; one prints "3m3" for a coverage figure and "8oC" for a temperature; a PDF title names a different product than its page. Quoted as printed, flagged in the ingestion report, never corrected.

## Ready-made answers, if asked live

- *Why no vector database in the prototype?* — See decision 3: at six to eight hundred chunks a dot product over one array is microseconds, behind the same interface a vector database sits behind in production.
- *Why a queue at all?* — See decision 4: for the moment a second producer writes into the pipeline, not for the website's size.
- *Why is the model only on Compose?* — See decision 2: every other path prints quoted passages by code; there is nothing to formulate, and a generated hand-off can invent a phone number.
- *Why 94 units, not forty?* — See decision 1: forty was a scope signal in a prompt; the boundary is a rule, and its cost is extraction QA, bounded by the spend order.
- *How does this handle hundreds of concurrent users?* — Retrieval doesn't care: a snapshot read is lock-free. Generation is the bottleneck, so production puts a queue, a rate limit and a cache in front of it, and under load serves only extract, route and refuse. Safety never degrades; coverage does.
- *How do you keep staff-only material out of public answers?* — Audience tags on every chunk, filtered at retrieval in code, never by prompt. In production the audience set comes from an authenticated session; in the prototype it is an asserted flag, and that is on the not-built list.
- *Where do the numbers come from?* — Only from a cited passage, word for word. Units are normalised for the comparison, never for the display. A figure that cannot be found in a retrieved passage does not print.
- *How do you catch the near-miss — right product, wrong property?* — The relevance gate: the property asked for, or a synonym, must appear in the passage, or the part refuses with "not stated in the indexed material". Retrieval being confident is not enough.
