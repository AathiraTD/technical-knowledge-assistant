# Decisions

Every decision behind the assistant: what was chosen, what else was possible, why, and where it breaks. The system it describes — diagrams, component reference and the site inventory — is in [`docs/architecture.md`](docs/architecture.md).

Each entry follows the same five questions, in the order a panel asks them: why does this decision exist, what else could have been done, why this one, prove it, where does it break.

## Index

| # | Decision | The call |
|---|---|---|
| 1 | Corpus boundary | Everything technical the site publishes — 94 units by rule, not a curated list |
| 2 | Extract path | Verbatim by code; the model runs on Compose only |
| 3 | Store | One repository boundary, two adapters: SQLite ships for assessment, PostgreSQL + pgvector is the deployment target |
| 4 | Framework | Hand-rolled retrieval, about 150 lines. No LangChain, no LlamaIndex, no agent loop |
| 5 | Chunking | Structure-aware: split at headings, keep bullets whole, tag caveats at document level |
| 6 | Embedding model | Open: decided by measurement before the build finishes |
| 7 | Generation model | qwen3.5:4b — one family for text now and vision later; qwen3:4b-instruct is the fallback |
| 8 | Router order | Retrieval-shape steps first: threshold, then a published deferral, then the slot-driven branches |
| 9 | Relevance gate | The asked-for term, or a synonym, must appear in a cited passage — or refuse |
| 10 | Load-bearing slots | Substrate and inside/outside: cued means used, uncued means per-option or ask back |
| 11 | Caveats | Tagged per document at ingestion, appended by code — not a chunking problem |
| 12 | Audience and data model | Three audiences, eight classification axes, filtered at retrieval in code |
| 13 | Second source | Staff-knowledge capture justifies the queue; CRM is a channel, not a corpus |
| 14 | Caching | Production only, keyed on template, slots, audience and index version |
| 15 | Interface | CLI canonical; a thin Streamlit UI over the same library |
| 16 | Images | Handled by policy, not by capability — detected, declared, handed off |
| 17 | Embedding cache | Content-addressed and shipped: a clean clone indexes in seconds, not forty minutes |

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

## 3. Store: one boundary, two adapters

**Why it exists.** Something has to hold the embeddings, the metadata that makes a citation possible, and the version history that stops a changed datasheet putting two coverage figures into the same answer. And it has to do that under two conditions that pull apart: the assessors run it offline from a clean clone with one command, while Lime Green would run it as a deployed service.

**Alternatives.** A numpy array plus a JSON manifest, which is the smallest thing that works at this scale. A dedicated vector database — Chroma, FAISS, Qdrant. PostgreSQL with pgvector everywhere, including the assessment path. Two stores, vectors and metadata separately.

**Why this one.** Serving both conditions with one codebase means the storage has to sit behind an interface, so the answer engine depends on `KnowledgeRepository` and never on a database driver. Then:

| | Assessment path | Deployment path |
|---|---|---|
| Store | SQLite — standard library, one file, ships in the repository | PostgreSQL + pgvector |
| Similarity | Cosine in numpy after loading; a single matrix multiply over six to eight hundred chunks | `<=>` inside the query, with an HNSW index |
| Service required | None | Postgres |
| Schema | `db/schema.sqlite.sql` | `db/schema.postgres.sql` |

The two schemas carry the same eight tables, the same column names and the same version semantics. That is what makes the claim "the assessment adapter and the production adapter are the same system" literally true rather than approximately true, and it makes the port mechanical rather than a rewrite.

SQLite rather than numpy-plus-JSON for the embedded adapter is the choice that buys this. A flat array would have worked for retrieval, but it would not have enforced one active version per document, and the two adapters would then have resembled each other rather than matched.

pgvector rather than a dedicated vector database, because the metadata filtering and the similarity search then happen in **one query** — active version, audience set, authority order and cosine distance together — instead of a numpy search followed by Python filtering followed by a separate metadata lookup. It is also one service for Lime Green to run rather than two, and free.

**Prove it.** Both schemas execute. The partial unique index refuses a second active version of the same document — an attempt to activate version 2 while version 1 is live raises an integrity error from the database, not from application code. Index build time and mean query latency are printed in the transcript header.

**Where it breaks.** The assessment adapter is one file and one process: no concurrent writers, and the index is rebuilt rather than updated in place. The Postgres adapter is written against the same interface but is exercised far less than the SQLite one, because the submission runs on SQLite — so treat its test coverage as the weaker of the two and say so rather than implying parity. Raw HTML and PDFs stay on a versioned filesystem in both paths: Postgres holds identity, history and chunks, never the bytes, because a 29 MB blob column buys nothing that a path and a content hash do not.

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

**Prove it.** All 37 PDFs were then probed, not three. The result changed the implementation and is worth stating plainly, because a font-only heading detector would have shipped broken:

| Detector | Documents where it finds headings |
|---|---|
| Font signal alone — heading is bolder or larger than body | 17 of 37 |
| Layout signal alone — heading is a short line above a paragraph | 20 of 37 |
| Both, union | 36 of 37, the last being a 34-page guide that splits anyway |

The corpus is two families of datasheet. The older sheets mark a heading with a heavier font; the newer ones use one font throughout and mark a heading by putting it alone on a short line. A font-only detector reports the second family as flat and falls back to whole-page chunks — which is how `medium-mortar-tds.pdf` first came out as two 3,000-character blobs despite having thirteen clean sections. Running both detectors and taking the union recovers Description / Mixing / Application / Aftercare across the corpus.

Final counts: 94 published documents plus one staff-tagged evaluation fixture, 579 passages, none over the 4,000-character ceiling, 52 documents classed clean and 43 partial, none failed. The passage count fell from 644 once the download furniture on product pages was removed — see the note under decision 8.

**Where it breaks.** "Partial" mostly means a short page rather than a bad extraction — a product page with two sections is a two-section page. The genuine weakness is the 34-page roof design guide, which yields 74 passages whose headings are drawing references rather than section names; it is retrievable but its citations read less well than a datasheet's. The ingestion report names every document and its quality, so this is inspectable rather than asserted.

## 6. Embedding model: open, decided by measurement

**Why it exists.** The whole justification for semantic retrieval over keyword search is the vocabulary gap: "the wall's gone damp" has to reach a passage about efflorescence, and "bag" has to reach "sack".

**Alternatives.** qwen3-embedding:0.6b; nomic-embed-text.

**Why this one.** Not yet chosen, and it should not be chosen by argument. Criteria: retrieval quality on five questions with known answers; index build time over roughly 700 chunks; memory with the generation model also loaded; vector dimension, which sets the array size; and input window against chunk size — `nomic-embed-text` has a 2K-token window, and while the largest observed section (Solo's application bullets, about 2,000 characters) fits comfortably, heading-based chunks are variable by nature and a silent truncation would degrade retrieval invisibly.

**Tags verified against the Ollama library, 15 September 2026.** `qwen3-embedding:0.6b` (639 MB) and `nomic-embed-text` (274 MB, default is v1.5) both exist.

**Where it breaks.** Either way, the model tag goes into the index header and the engine refuses to run against a mismatch — an index built with one model and queried with another returns confident nonsense.

## 7. Generation model: qwen3.5:4b

**Why it exists.** The brief mandates a local model. The hardware — a processor, 24 GB, no graphics card — bounds its size. And the model does exactly one job in this design: compose an answer over five retrieved passages, on the Compose path only.

### 7.1 The rubric

Two parts. Gates are pass or fail, assessed from published facts; a candidate failing any gate is not assessed further. Capability criteria are scored, and every one is measured on our own evaluation harness rather than taken from a leaderboard — general benchmarks (reasoning, maths, coding, multilingual, agentic tool use) measure almost nothing this design consumes.

**Gates**

| # | Gate | Why it is a gate, not a score |
|---|---|---|
| G1 | Licence permits commercial deployment | The partnership leads to deployment; bespoke terms are a legal question, not a trade-off |
| G2 | Runs usefully on a processor with no graphics card | A model taking a minute per answer cannot be demonstrated in seven |
| G3 | Published under a stable tag | An assessor pulls it from a clean clone; a preview cannot be a dependency |
| G4 | Can be made deterministic — fixed tag, temperature zero, fixed seed | The reproducibility target is an identical transcript on a second run |
| G5 | No mandatory reasoning tokens | Output tokens are the whole latency budget on a processor; thinking must be absent or switchable off |

**Two kinds of reasoning, and only one of them is wanted.** This design needs the model to reason *over supplied passages* — recognise that a passage about solid masonry covers a 1930s solid brick wall, join a base coat to a compatible finish coat because a passage links them, pick the right curing figure when a section lists two finish coats with different waits. That is bounded, grounded reasoning and it is the entire Compose job.

What it must not do is reason *from its own knowledge*. An inference like "Ultra is probably fine on cob" is the design's named weakest point, and check 1 exists to kill it: every sentence must carry word overlap with a cited passage, so a conclusion the passages do not support cannot print. Most of the reasoning a question needs has already been moved out of the model and into code — which path to take, which passages are relevant, whether to refuse, whether to answer per option, and arithmetic, which the model never performs.

This is the real reason thinking variants are gated out at G5, stronger than the latency argument: a chain of thought is an invitation to reason past the evidence, and here that is a defect rather than a feature.

**Capability criteria**, weighted by what each failure costs. Note the consequence peculiar to this architecture: the six checks are deterministic and run before printing, so a weaker model does not produce wrong answers here — it produces refusals. Capability shows up as **coverage**, and the weights follow the checks.

| # | Criterion | Weight | How it is measured |
|---|---|---|---|
| C1 | Citation-marker discipline — one marker per sentence, only for supplied passages | 20 | Check 1 pass rate across the seven situations |
| C2 | Verbatim fidelity — figures and qualifiers copied, not paraphrased | 20 | Checks 2 and 4 pass rate; the qualifier probes |
| C3 | **Grounded reasoning** — joins two or more passages into a correct combination, without inventing the link and without reaching past the text | 20 | The two multi-source situations: does the answer actually combine the documents, or quote them side by side? Does the joining sentence itself carry a citation? Read by the technical team |
| C4 | Refusal compliance — refuses on instruction instead of helping | 15 | Refusal state on the near-miss and far-miss |
| C5 | Latency and length — warm five-passage compose, about 200 tokens | 15 | Timed on the build machine; word counts in the probes. Decides live demonstration versus transcript |
| C6 | Attribution discipline — no blending across products or versions | 10 | Checks 3 and 5 pass rate; the two-product probe |
| — | Over-refusal rate on answerable questions | tie-break | A model that trips checks constantly yields a system that refuses everything |

### 7.2 Gate assessment

| Candidate | G1 Licence | G2 CPU | G3 Stable | G4 Deterministic | G5 No forced thinking | Result |
|---|---|---|---|---|---|---|
| `qwen3.5:4b` (3.4 GB) | **Not stated — check at pull** | 4B | yes | yes | **No instruct-only variant published; default not stated** | **Chosen, with two rows to verify** |
| `qwen3:4b-instruct` (2.5 GB) | Apache 2.0 | 4B | yes | yes | Instruct-only build | **Fallback — passes all five cleanly** |
| `granite4.2:3b` (2.2 GB) | Apache 2.0 | 3B | yes | yes | On by default, `enable_thinking=false` | **Assess** |
| `qwen3.8:27b` (18 GB) | Not stated | **27B — two to four minutes per answer on this hardware; 18 GB against 24 GB RAM** | yes | yes | On by default, disableable | **Out at G2** |
| `qwen3.8-flash-next` (105 GB+) | Not stated | **Impossible** | **Experimental preview of the Qwen4 architecture** | yes | — | **Out at G2 and G3** |
| `gemma4:e2b` / `e4b` (7.2 / 9.6 GB) | **Gemma terms** | Borderline | yes | yes | Configurable | **Out at G1** |
| `gemma3:4b` (3.3 GB) | **Gemma terms** | yes | yes | yes | yes | **Out at G1** |
| `llama3.2` | **Llama community terms** | yes | yes | yes | yes | **Out at G1** |
| `qwen3:8b`, `granite4.2:8b` | Apache 2.0 | **30–60 s per answer** | yes | yes | yes | **Out at G2** |
| `glm-5.3-flash` (18B active), `lfm2.5` (8B), `minicpm-v4.5` (8B) | Varies | **Too large** | yes | yes | Varies | **Out at G2** |
| `qwen3.8-flash-next` | Varies | yes | **Experimental preview** | yes | Varies | **Out at G3** |
| Qwen 3 thinking variants | Apache 2.0 | yes | yes | yes | **Thinking is the build** | **Out at G5** |

Two survive. **Gemma 4 does not lose on capability — it never reaches the capability assessment**, because licence is a gate and this work leads to deployment. That is the honest answer to "why not Gemma 4": had it been Apache-licensed it would have been assessed, and its multimodality and reasoning-first design would then have counted against it on cost-per-capability-consumed rather than on quality.

### 7.3 Capability assessment of the survivors

Evidence status is marked, because most of this is not yet measured and pretending otherwise would repeat the error of quoting a leaderboard.

| # | Criterion | Wt | `qwen3.5:4b` (chosen) | `qwen3:4b-instruct` (fallback) | `granite4.2:3b` | Evidence |
|---|---|---|---|---|---|---|
| C1 | Citation markers | 20 | Expected strong; newest of the three | Expected strong | Expected strong — IBM names structured JSON output | **To measure** |
| C2 | Verbatim fidelity | 20 | Unknown | Unknown | Unknown | **To measure** — the likely discriminator; no card claims it |
| C3 | Grounded reasoning | 20 | 4B, and the newest training run of the three | 4B | 3B, but IBM names retrieval-augmented generation explicitly | Architectural prior; **to measure** |
| C4 | Refusal compliance | 15 | Unknown | Unknown | Unknown | **To measure** |
| C5 | Latency and length | 15 | 3.4 GB — carries a vision encoder it will not use on the text path | 2.5 GB — lightest | 2.2 GB — lightest of all | **To measure** |
| C6 | Attribution | 10 | Unknown | Unknown | Unknown | **To measure** |
| — | Operational simplicity | — | No instruct-only build; thinking default unknown | No thinking path exists at all | One documented flag to set | Published fact |
| — | Roadmap fit | — | **Same family serves the multimodal path (16.1)** | Text only | Text only | Published fact |

**Reading it honestly**: C2, C4 and C6 — 45 of the 100 points — are unknown for all three, and no model card claims them, because verbatim copying and refusal-on-command are not what vendors benchmark. C5 favours the two smaller models. The rubric does not resolve on paper.

### 7.4 Decision

**`qwen3.5:4b` (3.4 GB), with `qwen3:4b-instruct` as the fallback and `granite4.2:3b` measured alongside.**

The deciding argument is not a capability row — those are unmeasured for all three — but roadmap fit: `qwen3.5` is multimodal, so **one model family serves the text assistant now and the vision perception step in 16.1 later**. That is a real architectural economy and a coherent thing to say in the room, rather than swapping families at phase three.

Two gate rows are unverified and must be checked when the model is pulled, not assumed:

| To verify at pull time | If it fails |
|---|---|
| **Licence** — not stated on the library page. Qwen 3 is Apache 2.0, so 3.5 probably is, but "probably" is not a licence | Fall back to `qwen3:4b-instruct`, which is confirmed Apache 2.0 |
| **Thinking default** — no instruct-only variant is published and the default is undocumented. `qwen3.8` has it on by default and disableable per request, so 3.5 likely does too | Disable it per request; if it cannot be disabled, fall back |

`qwen3:4b-instruct` is the fallback precisely because it passes all five gates cleanly and has no reasoning path to switch off — the safe option if either check goes the wrong way. All three are under 3.5 GB, so pull them together and let the harness print the rubric as a table. The model carrying more answerable questions through the six checks wins, and swapping is one configuration value recorded in the index header.

**Where it breaks.** A four-billion-parameter model is weakest exactly where this design leans on it — suitability and compatibility claims in prose — which is why the guardrails carry that risk rather than the model, and why Extract does not use it at all. `qwen3.5:4b` also carries a vision encoder the text path never touches, which is roughly a gigabyte of download and memory bought for the roadmap rather than for the submission: a deliberate trade, not an oversight. If both candidates score poorly on C2 or C3, the design absorbs it by refusing more often, and the honest thing on slide 3 is the over-refusal number, not a claim about the model.

## 8. Router order: retrieval shape before slots

**Why it exists.** Six conditions can hold at once — a question can have calculation words, a symptom, a deferring top passage and a weak score simultaneously — so precedence has to be stated or the behaviour is undefined.

**Alternatives.** Evaluate the slot-driven branches first, since slots are known before scores.

**Why this one.** Safety ordering beats information ordering. "How much Solo for an MgO board" retrieves the sheet's own "many MgO boards are not suitable for Solo — contact us", and that published deferral must beat a computed quantity; the register's rule is that the company's own instruction stands. Equally, nothing may generate below the threshold. Both of those are retrieval-shape facts, so they go first. The objection — that a symptom question below threshold would lose its "cannot see photographs" line — does not hold, because the hand-off renderer keys that line on the photograph slot, not on the path taken.

**Where it breaks.** A symptom question that retrieves nothing gets a refusal rather than a diagnosis hand-off. Since there are no published causes to quote in that case, the two replies say the same thing.

## 9. Relevance gate: the near-miss

**Why it exists.** The brief's third test type is a question the corpus cannot answer. Once Extract prints a retrieved passage verbatim, a confident retrieval on the right product but the wrong property — the vapour permeability of a sheet that never states it — would print a real, cited, irrelevant passage. Nothing else in the flow catches it: retrieval is above threshold, and the other checks run only on Compose.

**Alternatives.** Rely on the threshold alone. Rely on the citation and numbers checks. Use a second model call to judge relevance.

**Why this one.** A lexical gate is cheap, deterministic and explainable: the property or substrate asked for, or a synonym from the vocabulary, must appear in the cited passage, or the part refuses with "not stated in the indexed material". It runs as a router step on Extract and as check 6 on Compose, so both printing paths are covered. A second model call doubles latency on a CPU and is itself unverifiable.

**Depends on.** The unanswerable evaluation questions must be verified absent before they are fixed: grep the built corpus for the property, its synonyms and its units (vapour permeability, µ value, thermal conductivity, lambda, W/mK). If the property turns out to be published under another name, the assistant answers it correctly and the evaluation scores that as a failure, in front of the panel.

**Where it breaks.** It is lexical. A passage that names the property without answering it passes the gate — and what then prints is what the sheet actually says ("high breathability", no figure), which is the honest result rather than a hallucination. The property vocabulary is seeded from the datasheets' own field names at ingestion plus hand-written synonyms; an unknown property falls through to the threshold and citation checks.

## 10. Load-bearing slots

**Why it exists.** The clarify gate was reduced to "answer with stated assumptions", which is fine for exposure or season and dangerous for substrate: a recommendation on an assumed wall is the costly error the whole design exists to avoid.

**Alternatives.** Every missing slot becomes a stated assumption. Every missing slot triggers an ask-back.

**Why this one.** Two slots are load-bearing — substrate, and inside versus outside. Cued in the question, the value is used and stated. Uncued: inside/outside is answered per option, because the datasheets split that way anyway and both answers fit in the same five passages; substrate triggers an ask-back through the hand-off renderer, carrying what is published about choosing by substrate. Everything else remains a stated assumption, so the terse trade question still gets numbers.

**Where it breaks.** An ask-back in a single-turn tool means the user asks again; that is a real cost of not carrying conversation state. Two globally privileged slots is also a simplification: the facts a recommendation actually requires vary by intent, and the general form is required, conditional and blocking facts per intent. That generalisation is deferred, not rejected — the two chosen slots are the ones that matter for the selection questions the Ask stage found most common.

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

**Derived at ingestion, not authored.** Four artefacts are harvested from the crawl and written into the manifest, because typing them by hand would be inventing them:

| Artefact | Source | What depends on it |
|---|---|---|
| Product name list | Crawled product pages | Check 5, real names only |
| Colour name list | The repeated colour block on product pages — **harvested before boilerplate stripping removes it** | Check 5, and the invented-colour probe |
| Merchant list | The find-a-supplier page — 39 named stockists, read from the `alt` text of the map pins, because the page publishes them as an image overlay rather than as prose | Check 5, and never inventing a merchant |
| Contact line and hours | The contact page, verbatim | Every refusal and hand-off |

The tagging rule in the prototype is trivial but still explicit: everything crawled from the public website is tagged `public`; nothing is tagged `trade` or `staff`, because no such material is published. The one staff-tagged document is the synthetic fixture created for the evaluation set.

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

## 16. Images: handled by policy, not by capability

**Why it exists.** Photographs arrive in eight of the fifteen external situation archetypes — the damp bedroom wall, the crazed hallway, the patchy gable. It is the most common thing a customer attaches, and the job description names multimodal guardrails for text and images as a partnership activity. The absence of vision therefore has to be a decision, not an oversight.

**Alternatives.** A multimodal generation model reading the user's photograph (`gemma4:e2b`/`e4b`, `minicpm-v4.5`). A separate vision model behind the diagnosis path. Text only, with images detected and handed off.

**Why this one.** Four reasons, heaviest first.

1. *Vision would not change the answer.* Rung 4 of the ask ladder — diagnosis from the user's own evidence — is refuse-and-hand-off **by policy**, because judging a wall is a liability decision the technical team makes and stands behind. The register forbids diagnosing from symptoms or photographs, judging workmanship, and promising an outcome. A model that could see the photograph would still be forbidden from acting on what it saw.
2. *There is nothing to ground it against.* The corpus is published text. The failure library — labelled photographs of crazing, bloom and debonding with causes and remedies — is recorded in the data inventory as not existing. Vision would be reading the question, not the corpus, so a visual judgement could not be traced to a cited passage. Every other answer in this system can be.
3. *It could not be evaluated.* With no labelled failure photographs there is no way to measure whether a reading was right. Shipping an unevaluated capability contradicts the rest of the design, where the threshold, the checks and the refusals are all measured.
4. *A confident wrong visual diagnosis is the worst failure available here.* "That is efflorescence, it will wash off" over a photograph of sulphate attack ends in a damaged building. The thesis is that a refusal is cheaper than a wrong answer, and nowhere is that truer than on a photograph.

**What is built instead.** Images are handled as a detected condition with a designed response: the photograph slot fires on any mention of one, the router takes the diagnosis path, and the hand-off renderer states plainly that the assistant cannot see photographs, quotes what the published material does say about that symptom, and asks for the two details that matter. Detected, declared, handed off.

**Consequence for decision 7.** Because vision is out of scope by policy rather than by inability, a multimodal model is not a capability advantage here — it is weight, download and load time for something the design refuses to use. That is precisely what makes Gemma 4's multimodality a cost rather than a benefit in the gate assessment, and the reasoning only holds because this decision is made explicitly.

### 16.1 The roadmap architecture, if and when images are read

Three stages, strictly separated, with one rule that makes the whole thing safe: **the vision model never names a product.**

```
perception            interpretation          recommendation
what can I see?  →    what does that     →    what should we advise,
                      imply about             from the published
                      the building?           corpus?
```

Collapsing those stages is the failure mode. "I see rising damp, therefore use Product X" fuses an uncertain visual inference with a commercial recommendation and produces confident nonsense with liability attached.

**The observation contract.** The vision model returns structured observations, never prose and never an answer. Every attribute carries a value, a confidence, the image it came from and the region within it:

```json
{
  "observation": "white crystalline deposits near wall base",
  "confidence": 0.91,
  "image": "IMG_002",
  "region": [0.12, 0.67, 0.81, 0.94],
  "possible_interpretations": [
    { "cause": "salt deposition", "confidence": 0.63 }
  ]
}
```

together with a required `cannot_determine_from_image` list — existing plaster composition, moisture source, wall construction depth, substrate suction, structural movement. Forcing the model to enumerate what it *cannot* tell is the visual equivalent of refusing: it is the same discipline as declaring that something is not stated in the indexed material.

**Why this fits what is already built.** The region is to a visual claim what a cited passage is to a textual one — it makes the claim auditable, so a technical advisor can see exactly which pixels produced the observation. That is the same property the six checks enforce on text: nothing prints that cannot be traced. And a confidence below the floor simply leaves the slot uncued, which the load-bearing-slot rule already handles by asking back. The vision path therefore adds **no new answer route**; it fills slots on the router that exists.

**The guided visual survey.** One photograph rarely carries the recommendation-critical facts — a rendered wall hides its own substrate — so the assistant asks for more: a wider shot, an exposed section where the render has fallen away, the ground line and drainage. That turns image upload into a remote visual survey rather than a chatbot with an attachment, and it is the honest interaction, because it is what an advisor does on the phone today. It depends on carrying the situation across turns, which is already on the roadmap for the same reason.

**Retrieval from a building profile, not from the question.** Retrieval would run on the structured profile — external replastering; old brick masonry; traditional solid construction; failing render; possible salts; breathability required; moisture source unknown — rather than on the embedded sentence "what plaster should I use?". Slot-driven retrieval already works this way for text; images simply fill more slots, and fill them better than a typed description does.

**Candidates, verified on the Ollama library, with the hardware caveat.**

| Model | Size | Note |
|---|---|---|
| `qwen3-vl:2b` | 1.9 GB | Smallest viable; the realistic processor-only option |
| `qwen3-vl:4b` | 3.3 GB | Likely the balance point |
| `qwen3-vl:8b` | 6.1 GB | Strongest of the family; needs a graphics card in practice |
| `qwen3.5:2b` / `:4b` | 2.7 / 3.4 GB | Multimodal, 256K context, same lineage |
| `minicpm-v4.6` | 1B | Built for on-device inference |
| `gemma4:e4b`, `medgemma` | 9.6 GB / 4B | Gemma licence applies; `medgemma` is the precedent — a general family specialised to one visual domain |

The caveat the prototype hardware imposes: vision encoders are compute-heavy, and an 8B vision-language model on a processor with no graphics card is not a demonstration, it is a wait. This architecture assumes the graphics card or hosted inference that production adds, which is consistent with it being roadmap rather than build.

**Refinements to fold in when this is built.** Raised in review, correct, and deliberately not built now because none of it ships on Thursday:

| Refinement | Why it matters |
|---|---|
| **A profile resolver between observations and the profile** | Three photographs may disagree — brick 0.82, stone 0.61, brick 0.91. The later image must not simply overwrite the earlier. Aggregate with provenance and mark attributes `CONFIRMED`, `UNCERTAIN`, `CONFLICTING` or `NOT_DETERMINABLE` |
| **Treat model confidence as a routing band, not a probability** | A vision model reporting 0.91 is not correct 91 per cent of the time. Use `HIGH` / `UNCERTAIN` / `NOT_DETERMINABLE` at first, then set real thresholds from labelled examples with the threshold sweep the harness already does. That is what makes a claim like "at threshold X, substrate identification reached Y precision, so only observations above it may fill a recommendation-critical slot" defensible |
| **Required facts per intent, not two privileged slots** | "Substrate and interior or exterior" is right for replastering and wrong in general. Model it as required, conditional and blocking facts per intent, using the slot vocabulary that already exists |
| **Failure library schema** | Store image, model prediction, **expert-corrected label**, error type and visibility — not the photograph paired with the advisor's prose answer. Otherwise the model is trained on its own mistakes. Use it for **evaluation and calibration first**; fine-tune only if the measurements show it is needed |
| **Structured outputs for the observation contract** | Ollama supports constraining a response to a JSON schema. Use it rather than parsing prose from the vision model |

**The prerequisite has not changed.** None of these models diagnoses lime render off the shelf — a general vision-language model has no idea what crazing looks like against sulphate attack on a lime surface. The accuracy comes from fine-tuning on the labelled failure library, and that library is built by logging the photographs the hand-off already collects, paired with what the advisor answered. The model is the easy part; the data is the partnership.

**Where it breaks.** The customer still has a photograph and still wants an answer, so the hand-off has to be good — which is why the refusal carries the published causes and the contact line rather than a bare referral. This is also the decision most likely to be revisited first: it is the strongest demand-side case for the partnership's multimodal guardrails strand, and 16.1 sets out the architecture. The prerequisite is not a model, though — it is a labelled failure library, built with the technical team as annotators, which is exactly the knowledge capture the partnership exists to do.

---

## 17. Embedding cache: content-addressed, and shipped

**Why it exists.** Embedding 579 passages on this laptop takes about fifteen minutes on CPU. That is paid on a first build, which is tolerable, and then paid again on every rebuild, which is not — and a rebuild happens whenever chunking changes, which during development is constantly. It was paid twice before this existed, once when a transient HTTP 400 ended a run at 97 per cent.

**Alternatives.** Accept the rebuild cost. Ship the built index instead of the cache. Keep no cache and tell the assessor to wait.

**Why this one.** The key is the SHA-256 of the passage text together with the model tag and the dimension count, which is the whole correctness argument: a vector is returned only for the exact text it was computed from, by the exact model that computed it. Point the indexer at a different embedding model and every lookup misses and the run recomputes — which is the required behaviour, because a cached vector from another model is precisely the confident nonsense the index header exists to prevent.

Shipping the cache rather than the index is the part worth defending. The index is a build artefact tied to one machine's model; the cache is a lookup table that is either valid or misses. An assessor with the same model tag gets a build in seconds and can still verify it by deleting the cache; an assessor with a different model gets a correct, slower build rather than a wrong fast one.

**Prove it.** The same build, twice: 892 seconds computing 644 vectors, then 11 seconds with 644 cache hits and none computed. The counts appear in the ingestion report, so a run that quietly recomputed everything cannot be mistaken for one that did not.

**Where it breaks.** It is a cache of a pure function, so the failure modes are small, but it is derived data in version control — 5.6 MB as committed, which is larger than the live index needs. The cache is append-only and keyed by text, so every superseded chunking run leaves its vectors behind; the current 579 passages account for under half of it. That is honest but untidy, and the fix is a prune step that drops keys no live passage hashes to. Above ten megabytes the right move is to build it in CI and attach it to a release rather than commit it.

## Known weaknesses

- **The Postgres adapter is the less-exercised of the two.** The submission runs on SQLite, so the deployment adapter is written against the same interface and the same schema but sees far less use. Parity of design is not parity of testing, and the transcript only evidences one of them.
- **Compatibility is enforced by citation, not by a rules gate.** Nothing deterministic decides which products are eligible for a substrate before retrieval runs; the assistant can only say what a cited passage says, which prevents invention but does not actively exclude an incompatible product. The proper mechanism is an eligibility stage — substrate and exposure in, candidate products out, retrieval restricted to those — and it needs the product-to-substrate compatibility matrix, which the data inventory records as existing nowhere on the site, scattered across datasheets and advisors' heads. Building that matrix is partnership work; the eligibility gate follows it.
- **Qualitative synthesis is the weakest point.** The six checks bound numbers, names, attribution and the asked-for term; they reduce, not eliminate, an invented "this is fine on cob".
- **Single turn, and blind to photographs.** Real enquiries run six turns and eight of fifteen external situations attach a photograph; the prototype answers turn one, declares it cannot see images and hands them to a person (decision 16), and an uncued substrate becomes an ask-back the user answers by asking again.
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
| ~~HTML extraction library~~ | **Closed: BeautifulSoup + lxml.** The pipeline needs the DOM regardless — link-text classification and name-list harvesting both require it, and trafilatura's automatic main-content extraction would discard the colour block that must be harvested before stripping | Decided |

Five decisions of no consequence — the Ollama client (raw HTTP, to avoid a dependency wrapping two endpoints), argument parsing (`argparse`), the configuration format (JSON), dependency pinning (a pinned `requirements.txt`) and **no headless browser** (the site is server-rendered: plain fetches return the sitemap, full FAQ text, document links and the stockist list, so Playwright would add a browser download and an install step for nothing) — are recorded only so nobody assumes they were overlooked.

## Quick answers

- *Why SQLite in the demo and Postgres in production?* — Decision 3: the assessors need an offline clean-clone run, so the domain layer is not coupled to a database. `KnowledgeRepository` is the boundary; the assessment adapter is SQLite, the deployment adapter is PostgreSQL with pgvector, and the schema and version semantics are identical.
- *Why not LangChain?* — Decision 4: every guardrail sits where a framework abstracts, and the constraint is that the technical team can inspect it.
- *Why is the model only on Compose?* — Decision 2: every other path prints quoted passages by code; there is nothing to formulate, and a generated hand-off can invent a phone number.
- *Why 94 units, not forty?* — Decision 1: forty was a scope signal, not a limit; the boundary is a rule and its cost is extraction QA.
- *How do you catch the near-miss?* — Decision 9: the asked-for property, or a synonym, must appear in the cited passage, or it refuses. A confident retrieval is not enough.
- *How would this handle hundreds of concurrent users?* — Decision 14 and the serving layer: retrieval is lock-free, generation is the bottleneck, so a queue, a rate limit and a cache go in front of it, and under load only extract, route and refuse are served. Safety never degrades; coverage does.
- *How do you keep staff-only material out of public answers?* — Decision 12: audience tags filtered at retrieval in code, never by prompt; in production the audience set comes from an authenticated session.
- *Why can't it read the photograph the customer attached?* — Decision 16: it could be made to see one, but it would still be forbidden to act on it. Diagnosing a wall is the technical team's call, there is no labelled failure library to ground or evaluate a reading against, and a confident wrong visual diagnosis is the worst failure this system could produce. It says it cannot see the image, quotes what the sheets do say about the symptom, and hands over.
- *Where do the numbers come from?* — Only from a cited passage, word for word. Units are normalised for the comparison, never for the display. A figure not found in a retrieved passage does not print.
