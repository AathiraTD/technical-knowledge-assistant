# Decisions

Every decision behind the assistant: what was chosen, what else was possible, why, and where it breaks. The system it describes — diagrams, component reference and the site inventory — is in [`docs/architecture.md`](docs/architecture.md).

Each entry follows the same five questions, in the order a panel asks them: why does this decision exist, what else could have been done, why this one, prove it, where does it break.

## Index

| # | Decision | The call |
|---|---|---|
| 1 | Corpus boundary | Everything technical the site publishes — 94 units by rule, not a curated list |
| 2 | Extract path | Verbatim by code; the model runs on Compose only |
| 3 | Store | One store: a numpy array plus a manifest, swapped together. No vector database, no second metadata store |
| 4 | Framework | Hand-rolled retrieval, about 150 lines. No LangChain, no LlamaIndex, no agent loop |
| 5 | Chunking | Structure-aware: split at headings, keep bullets whole, tag caveats at document level |
| 6 | Embedding model | Open: decided by measurement before the build finishes |
| 7 | Generation model | qwen3:4b-instruct, granite4.2:3b if it emits reasoning blocks |
| 8 | Router order | Retrieval-shape steps first: threshold, then a published deferral, then the slot-driven branches |
| 9 | Relevance gate | The asked-for term, or a synonym, must appear in a cited passage — or refuse |
| 10 | Load-bearing slots | Substrate and inside/outside: cued means used, uncued means per-option or ask back |
| 11 | Caveats | Tagged per document at ingestion, appended by code — not a chunking problem |
| 12 | Audience and data model | Three audiences, eight classification axes, filtered at retrieval in code |
| 13 | Second source | Staff-knowledge capture justifies the queue; CRM is a channel, not a corpus |
| 14 | Caching | Production only, keyed on template, slots, audience and index version |
| 15 | Interface | CLI canonical; a thin Streamlit UI over the same library |

---

## 1. Corpus boundary: 94 units by rule

**Why it exists.** The agent prompt suggested "about forty documents" as a scope signal. The live inventory found 94 technical units, so the cap now cuts for real and has to be justified rather than inherited.

**Alternatives.** Keep forty and cut datasheets to the products the test questions touch. Index everything on the site including colour pages, category pages, case studies and news.

**Why this one.** A rule the panel can check beats a curated list, and cutting datasheets down to what the evaluation needs is indexing for the test. Forty was never a technical limit: at six to eight hundred chunks, embedding takes minutes and retrieval is instantaneous. The real reasons to exclude are retrieval pollution (24 near-identical colour pages, boilerplate category pages) and evaluation honesty (near-duplicates make the threshold sweep noisier). So the boundary is: everything technical the site publishes; out go near-duplicates, boilerplate and non-technical documents.

**Prove it.** The inventory in the architecture appendix, with counts and the excluded list by name and link.

**Where it breaks.** Embedding is minutes; the real cost is extraction QA across 31 unprobed datasheets and 3 unextracted guides. That is paid through the ingestion report and bounded by the spend order — anything that misses the time box is dropped in that order and listed in the manifest as excluded.

## 2. Extract path: verbatim by code

**Why it exists.** The brief's first recommended test is a straightforward product question. Something has to print "5 to 6 litres per 25 kg sack" without risking a paraphrase.

**Alternatives.** The model quotes and stops over two passages, keeping it on every path. The model composes on every path.

**Why this one.** The quote-and-cite printer is built regardless — the cited hand-off and the refusal template both print quoted passages by code — so putting the model on Extract adds a third prompt variant rather than removing one. On a lookup the model can contribute nothing but paraphrase drift, and the checks would then refuse it: "5–6 litres per 25 kg bag" rendered as "5 to 6 litres per bag" trips the qualifier check, which is a false refusal on the brief's first test type. Compliance with "use a local LLM to formulate an answer" is met where formulation actually happens, on Compose; the record already prints Route, Cited hand-off and Refuse with no model at all.

**Prove it.** Warm latency on a two-passage call, measured against the trade's ten-second target: the record's own estimate is prompt reading plus about 80 tokens at 15–25 tokens per second, which sits uncomfortably close to the limit.

**Where it breaks.** The straightforward test question shows a quoted passage with a citation rather than model prose, so say it on the slide: the model formulates where there is something to formulate. A model quote-and-stop stays a roadmap option, gated on a measured warm call with headroom; nothing in the design assumes it.

## 3. Store: one array and a manifest

**Why it exists.** Something has to hold six to eight hundred embeddings and the metadata that makes a citation possible.

**Alternatives.** Chroma, FAISS, pgvector or a hosted vector database. Two stores, vectors and metadata separately.

**Why this one.** Cosine over the whole array is a single matrix multiply — microseconds, nothing to tune. Approximate nearest-neighbour indexes earn their place around a hundred thousand to a million vectors, two to three orders of magnitude away. A separate metadata store is a synchronisation bug that breaks citations while retrieval still looks healthy, which is the worst failure shape available. One store, swapped atomically with its manifest.

**Prove it.** Index build time and mean query latency printed in the transcript header.

**Where it breaks.** One process, one user, no concurrent writers; the index is rebuilt rather than updated in place. Production swaps a vector database with payload behind the same interface — or a vector column in the database the company already runs, which is likelier once the CRM is in scope.

## 4. Framework: hand-rolled, not LangChain or LlamaIndex

**Why it exists.** The job description names both, so silence would read as not having considered them.

**Alternatives.** LangChain; LlamaIndex; a framework for ingestion only with hand-rolled retrieval.

**Why this one.** Every guardrail in this design sits exactly where a framework abstracts. The per-document cap lives inside the retriever. Caveat adjacency lives inside the splitter. The six post-generation checks live between generation and printing, which is not a seam most chains expose. A framework would have to be opened and explained anyway, so it buys convenience and costs the thing being assessed. The constraint from the problem statement is explicit: simple, cheap and explainable — nothing the technical team cannot inspect. At 94 documents, the loaders, splitters and store adapters are replaced by five functions.

**Prove it.** The dependency list is Ollama, an HTTP client, one HTML extractor, PyMuPDF and numpy. The pipeline is five stages, each one function, each independently runnable.

**Where it breaks.** No community components and nothing to inherit as the corpus grows to many sources and formats. This is the right call at this scale, not a general position: past a few thousand documents across heterogeneous sources, a framework's loader ecosystem starts paying for itself.

## 5. Chunking: structure-aware

**Why it exists.** A chunk is both the unit of retrieval and the unit of citation, and the corpus is datasheets where a number without its condition is dangerous.

**Alternatives.** Fixed-size with overlap (512 tokens / 50, the common default); recursive character splitting; semantic chunking on embedding boundaries; proposition extraction by an LLM; small-to-big, retrieving a small chunk and returning its parent.

**Why this one.** Decided on evidence from the datasheets rather than convention. All three probed sheets put headings on their own lines, so heading splitting is available. Solo's application section is one bullet per background at roughly 2,000 characters — a size-based splitter cuts a thickness away from the substrate it applies to. A section is also a citable unit: "the Mixing section of the Solo datasheet" is something a plasterer can check, which a 512-token window is not. Semantic chunking is non-deterministic, which costs reproducibility; proposition extraction puts an LLM in the ingestion path, rewriting a datasheet in a liability-sensitive corpus. Small-to-big is the closest good alternative, and retrieving whole sections approximates it without a second index.

**Prove it.** Three datasheets extracted and inspected: headings on their own lines, the 2,000-character bullet, and the caveat-adjacency finding that produced decision 11.

**Where it breaks.** Thirty-one datasheets and three system guides are unprobed. If any are multi-column or table-only, heading splitting degrades — the ingestion report will show it, and the fallback is per-page chunks for those documents, flagged in the manifest.

## 6. Embedding model: open, decided by measurement

**Why it exists.** The whole justification for semantic retrieval over keyword search is the vocabulary gap: "the wall's gone damp" has to reach a passage about efflorescence, and "bag" has to reach "sack".

**Alternatives.** qwen3-embedding:0.6b; nomic-embed-text.

**Why this one.** Not yet chosen, and it should not be chosen by argument. Criteria: retrieval quality on five questions with known answers; index build time over roughly 700 chunks; memory with the generation model also loaded; vector dimension, which sets the array size.

**Where it breaks.** Either way, the model tag goes into the index header and the engine refuses to run against a mismatch — an index built with one model and queried with another returns confident nonsense.

## 7. Generation model: qwen3:4b-instruct

**Why it exists.** The brief mandates a local model, and the hardware — a processor, 24 GB, no graphics card — bounds its size.

**Alternatives.** gemma3:4b; qwen3:8b or granite4.2:8b; llama3.2; the thinking variants of Qwen 3; granite4.2:3b.

**Why this one.** Apache 2.0, so commercial deployment is unencumbered — gemma3:4b is strong at this size but carries bespoke terms, and this partnership leads to deployment. Instruct-only, so there are no reasoning blocks to strip; the thinking variants double the output on a processor. 2.5 GB, the smallest download that follows quoting and refusal instructions reliably. The 8B models take thirty to sixty seconds per answer on a CPU — useful only as a transcript comparison to show what size buys. llama3.2 is older and weaker at this size.

**Prove it.** Warm latency on a five-passage compose, measured on the build machine; that number decides whether the demonstration is live or read from the transcript.

**Where it breaks.** A four-billion-parameter model is weakest exactly where this design leans on it — qualitative synthesis, suitability and compatibility. That is why the guardrails carry the risk rather than the model, and why Extract does not use it. Granite is a design-time fallback chosen by the author if the instruct build emits reasoning blocks, never a runtime switch: a missing model fails loudly by name.

## 8. Router order: retrieval shape before slots

**Why it exists.** Six conditions can hold at once — a question can have calculation words, a symptom, a deferring top passage and a weak score simultaneously — so precedence has to be stated or the behaviour is undefined.

**Alternatives.** Evaluate the slot-driven branches first, since slots are known before scores.

**Why this one.** Safety ordering beats information ordering. "How much Solo for an MgO board" retrieves the sheet's own "many MgO boards are not suitable for Solo — contact us", and that published deferral must beat a computed quantity; the register's rule is that the company's own instruction stands. Equally, nothing may generate below the threshold. Both of those are retrieval-shape facts, so they go first. The objection — that a symptom question below threshold would lose its "cannot see photographs" line — does not hold, because the hand-off renderer keys that line on the photograph slot, not on the path taken.

**Where it breaks.** A symptom question that retrieves nothing gets a refusal rather than a diagnosis hand-off. Since there are no published causes to quote in that case, the two replies say the same thing.

## 9. Relevance gate: the near-miss

**Why it exists.** The brief's third test type is a question the corpus cannot answer. Once Extract prints a retrieved passage verbatim, a confident retrieval on the right product but the wrong property — the vapour permeability of a sheet that never states it — would print a real, cited, irrelevant passage. Nothing else in the flow catches it: retrieval is above threshold, and the other checks run only on Compose.

**Alternatives.** Rely on the threshold alone. Rely on the citation and numbers checks. Use a second model call to judge relevance.

**Why this one.** A lexical gate is cheap, deterministic and explainable: the property or substrate asked for, or a synonym from the vocabulary, must appear in the cited passage, or the part refuses with "not stated in the indexed material". It runs as a router step on Extract and as check 6 on Compose, so both printing paths are covered. A second model call doubles latency on a CPU and is itself unverifiable.

**Where it breaks.** It is lexical. A passage that names the property without answering it passes the gate — and what then prints is what the sheet actually says ("high breathability", no figure), which is the honest result rather than a hallucination. The property vocabulary is seeded from the datasheets' own field names at ingestion plus hand-written synonyms; an unknown property falls through to the threshold and citation checks.

## 10. Load-bearing slots

**Why it exists.** The clarify gate was reduced to "answer with stated assumptions", which is fine for exposure or season and dangerous for substrate: a recommendation on an assumed wall is the costly error the whole design exists to avoid.

**Alternatives.** Every missing slot becomes a stated assumption. Every missing slot triggers an ask-back.

**Why this one.** Two slots are load-bearing — substrate, and inside versus outside. Cued in the question, the value is used and stated. Uncued: inside/outside is answered per option, because the datasheets split that way anyway and both answers fit in the same five passages; substrate triggers an ask-back through the hand-off renderer, carrying what is published about choosing by substrate. Everything else remains a stated assumption, so the terse trade question still gets numbers.

**Where it breaks.** An ask-back in a single-turn tool means the user asks again; that is a real cost of not carrying conversation state.

## 11. Caveats as document metadata

**Why it exists.** Fine Stuff puts its 8 °C limit under Mixing and again under Curing, and "not suitable for DIY plastering" under Application, while the steps a user asks about sit elsewhere. No chunking strategy keeps those together.

**Alternatives.** A check that looks across sections at answer time. Larger chunks. Accept the gap.

**Why this one.** It is an indexing problem, not a retrieval one. Caveat sentences are tagged per document at ingestion and appended by code — at most three, chosen by slot overlap — whenever any chunk of that document is printed or composed over. The qualifier check then verifies only within the printed passage, which is something it can actually do. Deferral sentences are excluded from the tags, because they have their own path and would otherwise print twice.

**Where it breaks.** Tagging is rule-based, so an unusually worded caveat is missed; and appending is capped at three, so a document with five relevant caveats shows the three that best match the question.

## 12. Audience and the data model

**Why it exists.** The job description asks for a shared data store used by multiple users and systems, and the segmentation found three groups with different rights to the same corpus.

**Alternatives.** One public corpus. A binary internal/external partition, which is what the record's constraint fork proposed.

**Why this one.** Three audiences — public, trade, staff. The binary fork loses the trade: stockists and contractors see lead times, stock and kit lists that the public does not, and they are the natural first user. Filtering happens at retrieval, in code, never by prompt, because a prompt instruction is not an access control. Eight classification axes are applied at ingestion:

| Axis | What it carries | What it enforces |
|---|---|---|
| Audience tags | public, trade, staff | The retrieval filter |
| Authority rank | datasheet > product page > knowledge-base article > FAQ | Conflict resolution; newest wins within a type |
| Document type | TDS, product page, FAQ entry, article, system guide, commercial page | Routing, and the name used in a citation |
| Dates | printed date, crawl date | Newest-wins, and the date shown beside an answer |
| Product | the product a chunk belongs to | The check that a number stays with its product |
| Section path | heading trail | Citation granularity |
| Caveat flags | temperature limits, DIY suitability, incompatible substrates | Appended by code whenever the document is used |
| Exclusion list | excluded documents by name and link | Answering "why isn't the safety sheet in here?" |

**Where it breaks.** Every document in the prototype corpus is public, so the filter is exercised only by a synthetic staff-tagged fixture in the evaluation set. The audience set is asserted at the command line, not authenticated — identity is the first thing production adds. Production also adds a sensitivity axis, because formulations and quality records are staff-only and belong in a separate store.

## 13. Second source: staff-knowledge capture, not CRM

**Why it exists.** An indexing queue is only justified by more than one producer writing into the same pipeline.

**Alternatives.** CRM records as the second source. No second source, in which case the queue collapses to a scheduled re-index.

**Why this one.** The record classes the CRM as a system of record — "integration, not corpus", "route, never generate" — and its contents are personal data looked up exactly, never searched by meaning. Nothing in it is answer material, so it belongs on the consuming side as a channel adapter. Staff-knowledge capture is the partnership's core activity and produces exactly what the store wants: the agreed answer set, the failure library and the compatibility matrix as text. The policy list goes to the authored configuration rather than the index.

**Where it breaks.** The queue stays roadmap. If capture turns out to yield structured configuration only, there is no second chunk producer and the queue should collapse to a scheduled re-index with one lock.

## 14. Caching: production only, template-keyed

**Why it exists.** A hundred concurrent public questions become a generation queue, and the same nineteen templates recur.

**Alternatives.** No cache. An exact-string cache. A semantic cache, embedding the query and serving a near neighbour's answer.

**Why this one.** The Ask stage found that roughly eighty per cent of demand is nineteen question templates, so a template-keyed cache hits often where an exact-string cache would barely hit — two people never phrase it identically. The key is the template identifier, the detected slot values, the caller's audience set and the index version. A semantic cache is rejected on principle rather than cost: serving a "close enough" stored answer is exactly what a system built on refusing rather than approximating must not do. Reuse is safe because answers are deterministic — temperature zero, fixed seed, fixed snapshot — and invalidation is free, because the index version is in the key and the atomic swap acts as a cache epoch. Refusals are cached too: a refusal costs a full retrieval to produce.

**Where it breaks.** The audience set must be in the key or a staff answer reaches a public caller — a leak, not a performance bug. Keyed too loosely, it serves the right template for the wrong substrate.

## 15. Interface: CLI canonical, a thin UI over it

**Why it exists.** The brief asks for a demonstrable assistant and the presentation is seven minutes.

**Alternatives.** Command line only. FastAPI with a single HTML page. A chat widget.

**Why this one.** The answer engine is a library by design — question and audience set in; answer, sources, status and diagnostics out — so a second surface is a wrapper, not a rewrite, and building one demonstrates the property the production channel adapters depend on rather than asserting it. Streamlit is one dependency and one command on the assessors' machine. The command line stays canonical because the evidence lives there: the transcript and the evaluation harness run through it, so a UI failure must not cost the evidence.

**Where it breaks.** One more dependency that has to install cleanly from a clean clone. The UI is single-user and single-turn like everything else. If it misbehaves on the assessors' machine, the command line and the transcript still stand.

---

## Known weaknesses

- **Qualitative synthesis is the weakest point.** The six checks bound numbers, names, attribution and the asked-for term; they reduce, not eliminate, an invented "this is fine on cob".
- **Single turn.** Real enquiries run six turns with photographs; the prototype answers turn one, says it cannot see photographs, and an uncued substrate becomes an ask-back the user answers by asking again.
- **The audience set is asserted, not authenticated.** The filter is real and enforced in code at retrieval; the identity behind the claim is missing.
- **One user at a time.** No queue, no rate limit, no cache: the concurrency story is drawn and argued, not built.
- **Staff mode always extracts.** Staff compose on request, and the staff drafting mode, are roadmap — the advisor verifies from the passage text, and time is the constraint.
- **Datasheet quality is inherited.** Sheets date from 2015 to 2025; one prints "3m3" for a coverage figure and "8oC" for a temperature; a PDF title names a different product than its page. Quoted as printed, flagged in the ingestion report, never corrected.

## Still open

| Open | Closed by | When |
|---|---|---|
| Embedding model | Building the index with each and comparing five known-answer questions | Slice 1 |
| Generation latency, and so live demo versus transcript | Warm timing of a five-passage compose | Slice 1 |
| Abstention threshold value | The sweep, printed at the chosen value and at plus and minus 0.1 | Evaluation |
| HTML extraction library | One test across a product page, the FAQ and an article, judged on boilerplate removal | Slice 1 |

Four decisions of no consequence — the Ollama client (raw HTTP, to avoid a dependency wrapping two endpoints), argument parsing (`argparse`), the configuration format (JSON) and dependency pinning (a pinned `requirements.txt`) — are recorded only so nobody assumes they were overlooked.

## Quick answers

- *Why no vector database?* — Decision 3: at six to eight hundred chunks a dot product over one array is microseconds, behind the same interface a vector database sits behind in production.
- *Why not LangChain?* — Decision 4: every guardrail sits where a framework abstracts, and the constraint is that the technical team can inspect it.
- *Why is the model only on Compose?* — Decision 2: every other path prints quoted passages by code; there is nothing to formulate, and a generated hand-off can invent a phone number.
- *Why 94 units, not forty?* — Decision 1: forty was a scope signal, not a limit; the boundary is a rule and its cost is extraction QA.
- *How do you catch the near-miss?* — Decision 9: the asked-for property, or a synonym, must appear in the cited passage, or it refuses. A confident retrieval is not enough.
- *How would this handle hundreds of concurrent users?* — Decision 14 and the serving layer: retrieval is lock-free, generation is the bottleneck, so a queue, a rate limit and a cache go in front of it, and under load only extract, route and refuse are served. Safety never degrades; coverage does.
- *How do you keep staff-only material out of public answers?* — Decision 12: audience tags filtered at retrieval in code, never by prompt; in production the audience set comes from an authenticated session.
- *Where do the numbers come from?* — Only from a cited passage, word for word. Units are normalised for the comparison, never for the display. A figure not found in a retrieved passage does not print.
