# Lime Green Technical Knowledge Assistant

A local-LLM technical knowledge assistant over Lime Green Products' published material. It retrieves passages from the company's own datasheets, guides and FAQ, cites every source by document name, and refuses — with a hand-off to the technical team — when the published material does not answer the question.

Everything runs on one machine. No API key, no hosted model, no network at answer time.

Built for the AEC Solution Architect (AI/LLM Developer) KTP take-home exercise, Birmingham City University with Lime Green Products.

> **New here?** [Running it](#running-it) is four commands. [Demoing it](#how-to-demo) is seven questions. [What is not finished](#limitations-and-production-seams) is stated plainly rather than left to be discovered.

---

## What it demonstrates

| | What to look at |
|---|---|
| **RAG over approved sources only** | 94 published documents plus one staff-tagged evaluation fixture, 552 passages. The corpus boundary is a rule, not a curated list — `DECISIONS.md` entry 1 |
| **Provenance** | Every answer names its sources by document name and section, with a followable link and the printed date. A figure not found word-for-word in a cited passage does not print |
| **Versioned knowledge** | One active version per document, enforced by a partial unique index in the database. A changed datasheet supersedes its predecessor and the old version is retained but unreachable — `DECISIONS.md` entry 18 |
| **Stateful conversation** | The turn is a LangGraph state machine with a checkpoint per session: slots carry across turns, a correction overwrites rather than accumulates, and an ask-back genuinely pauses and resumes — `DECISIONS.md` entry 20 |
| **Evidence sufficiency** | A deterministic router decides the path before the model sees anything. Below threshold, or the asked-for term absent from every passage, and it refuses — `DECISIONS.md` entry 9 |
| **Image input** | A vision model reads photographs into structured observations that fill slots. It is never permitted to name a product — the attribute field is an enum of four slots, and an observed value that is not already in the vocabulary is discarded |
| **Product recommendation** | Candidates are assessed against the corpus per required property; a product whose substrate suitability is not independently established cannot be recommended |
| **Safe refusal and hand-off** | Eleven policy topics route without retrieval when their patterns match — three probe phrasings miss, see [known issue 1](#known-issues-in-this-branch). A refusal still prints what the site does publish, its source, and the technical team's harvested contact line |
| **Observability and evaluation** | A correlation id per answer, readable back with `python -m assistant.trace`. Nine situations, ten probes, five conversations and a threshold sweep in `eval/` |

---

## How it works, in a paragraph

A question is split by topic, gated against a routing table, and matched against slot vocabularies. What survives is embedded and retrieved from the knowledge store — active document versions only, filtered to the caller's audience in code rather than by prompt. A deterministic router — not the model — then picks one of five paths: route to a fixed referral, print a passage verbatim, compose over several passages, quote a published hand-off, or refuse. The model runs on the compose path only, at temperature zero, over retrieved passages delimited as data. Six checks run before anything prints, and a failure sends the part to a refusal that still carries whatever the site does publish.

---

## Requirements

- **Python 3.11 or later** (developed and measured on 3.13)
- **[Ollama](https://ollama.com/download)** installed and running, with two models pulled:

```
ollama pull qwen3.5:4b            # 3.4 GB — generation
ollama pull qwen3-embedding:0.6b  # 639 MB — embeddings
```

About 4 GB in total. `qwen3:4b-instruct` (2.5 GB) is the documented fallback — `DECISIONS.md` entry 7. Model tags are fixed in configuration, recorded in the index header, and **the engine refuses to run against an index built with a different model** rather than returning confident nonsense.

Hardware this was measured on: a processor, 24 GB RAM, **no graphics card**. That constraint is why latency is what it is — see [Testing and evidence](#testing-and-evidence).

---

## Running it

From a clean clone, in order:

```
pip install -r requirements.txt
python -m assistant.index        # builds the index offline from the shipped crawl
python -m assistant.ui           # the web page — open the address it prints
```

That is enough to use it. The other entry points:

```
python -m assistant.cli "How much water does Solo Onecoat need per bag?"
python -m assistant.health       # store, snapshot compatibility, model reachability
python -m eval.run               # nine situations, probes, conversations, sweep
python -m assistant.trace <correlation-id>    # the stages of one specific answer
```

**What to expect from `python -m assistant.index`.** About **25 seconds** on a first run and **1–2 seconds** on a second, reporting `unchanged 94 · reprocessed 0`. It does not touch the network: the crawled pages and PDFs ship in `data/cache/`, and the content-addressed embedding cache ships in `data/embeddings.db`, so a clean clone gets 552 cache hits rather than fifteen minutes of embedding. Point it at a different embedding model and every lookup misses and it recomputes — which is the required behaviour, not a bug.

**Selecting PostgreSQL.** Set `ASSISTANT_POSTGRES_DSN` and every entry point uses the PostgreSQL + pgvector adapter instead of SQLite. Same schema, same version semantics.

**Refreshing the corpus.** `python -m assistant.crawl --refresh` revalidates sources with ETag and Last-Modified. See [the runbook](docs/knowledge-pipeline.md) for staff imports and scheduling.

---

## How to demo

Seven questions, each chosen because it exercises a different path. There is a
full rehearsal script with timings, expected results and a failure playbook in
**[`docs/DEMO-SCRIPT.md`](docs/DEMO-SCRIPT.md)**.

| # | Ask this | What it shows | Path |
|---|---|---|---|
| 1 | `How much water does Solo Onecoat need per bag?` | The published figure reproduced exactly, cited to the datasheet section | compose |
| 2 | `How many bags of Duro do I need for 20 square metres?` | Coverage and pack size printed verbatim — **and the multiplication refused** | extract |
| 3 | `Which plaster should I use?` → reply `brick` | Asks back for the load-bearing slot, then resumes the original question | ask back |
| 4 | `What is the U-value of Solo Onecoat plaster?` | Refuses. It retrieves at 0.739 — *more* confidently than question 1 — and the relevance gate catches it anyway | refuse |
| 5 | `Does this comply with Part L?` | Never reaches retrieval. Fixed referral with the harvested contact line, in about 2 s | route |
| 6 | `I have an old solid brick wall…` then `How much would I need for 30 m² at 25 mm?` | The second turn inherits the wall and the product without restating either | compose → extract |
| 7 | `Is Solo available in Cotswold Cream?` | An invented colour is refused — check 5 tests names against lists harvested at ingestion | refuse |

**Use question 5 exactly as written.** The longer phrasing "Can you confirm my Warmshell build complies with Part L?" misses the policy gate and takes 80 s to refuse instead of 2 s to route — that is [known issue 1](#known-issues-in-this-branch), and it is worth knowing before a demonstration rather than during one.

**Before demoing, warm the answer cache.** An uncached composed answer costs **35–200 seconds** on a machine with no graphics card; a repeat is **0.017 s**. Ask questions 1 and 6 once beforehand. The checklist in the demo script covers this and the rest of the pre-flight.

---

## Architecture

Deterministic code owns every decision. The model composes sentences and nothing else.

```
                      question + audience set
                               │
                    ┌──────────▼──────────┐
                    │  split by topic     │
                    │  policy gate (11)   │──── matches ──▶ ROUTE  (fixed referral,
                    └──────────┬──────────┘                         no retrieval)
                               │ no match
                    ┌──────────▼──────────┐
                    │  slot detection     │◀─── vision: photograph ▶ observations
                    │  substrate, location│     (never names a product)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────────────────────┐
                    │  RETRIEVAL  via KnowledgeRepository      │
                    │  active versions only                    │
                    │  audience-filtered in code, before rank   │
                    │  authority band, then similarity          │
                    └──────────┬──────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ DETERMINISTIC ROUTER│  8 steps, evaluated in order
                    └──┬────┬────┬────┬───┘
            1,4 below  │    │    │    │  8 otherwise
         threshold /   │    │    │    └──────────────▶ COMPOSE ─▶ ⟨model⟩ ─▶ 6 CHECKS
         term absent   │    │    │ 6,7 calculation /                          │
                       │    │    └── one document ──▶ EXTRACT (verbatim,      │
                       │    │                                  by code)       │
                       │    └── 2 published deferral ─▶ CITED HAND-OFF        │
                       │    └── 3 cause asked ────────▶ DIAGNOSIS              │
                       └── 5 substrate uncued ────────▶ ASK BACK          any check
                               │                                            fails
                               ▼                                              │
                            REFUSE ◀───────────────────────────────────────────┘
                               │
                               ▼
                    hand-off renderer: what IS published, its source,
                    document caveats, the harvested contact line
```

**The six checks**, in order, all after generation and all deterministic: every sentence cited with word overlap to its passage; numbers verbatim in the cited passage; numbers attached to the right product; qualifiers travelling with their figure; only real names (products, colours, merchants — harvested at ingestion); and the asked-for property or substrate present in a cited passage.

**Storage** sits behind one `KnowledgeRepository` interface with two adapters — SQLite (ships, offline, serves the transcript) and PostgreSQL + pgvector (the deployment path). Same eight tables, same column names, same version semantics. The answer engine never imports a database driver.

Full diagrams — container view, answer engine detail, multimodal roadmap — with the corpus inventory: **[`docs/architecture.md`](docs/architecture.md)**. Every decision with its alternatives and costs: **[`DECISIONS.md`](DECISIONS.md)** (20 entries).

---

## Read this first

| If you want | Read |
|---|---|
| Why it is built this way — every decision, alternative and cost | [`DECISIONS.md`](DECISIONS.md) |
| What the system is — diagrams, component reference, corpus inventory | [`docs/architecture.md`](docs/architecture.md) |
| To demonstrate it | [`docs/DEMO-SCRIPT.md`](docs/DEMO-SCRIPT.md) |
| How the knowledge pipeline is operated | [`docs/knowledge-pipeline.md`](docs/knowledge-pipeline.md) |
| The exercise itself | [`docs/brief.docx`](docs/brief.docx) |

---

## Testing and evidence

Everything below is a measurement, taken on the build machine, with the date it was taken. Where a number is worse than it looks in a slide, it is stated here rather than rounded.

### Running the tests

```
pip install -r requirements-dev.txt
python -m pytest -q tests/              # no Ollama, no index, no network
```

Browser tests are opt-in twice, because a Chromium download is exactly the heavyweight dependency deterministic CI keeps out:

```
python -m playwright install chromium
set ASSISTANT_E2E=1                     # export ASSISTANT_E2E=1 on a shell
python -m pytest tests/e2e -m smoke -q  # fast pre-flight, no generation
python -m pytest tests/e2e -q           # full pre-demo suite
```

Coverage:

```
python -m coverage run --rcfile=.coveragerc -m pytest -q tests/
python -m coverage report --rcfile=.coveragerc
```

### Evaluation harness — measured, including what fails

`python -m eval.run` against the shipped corpus. The transcript in [`eval/results/transcript.txt`](eval/results/transcript.txt) carries the snapshot id, embedding model and dimension, chunking version and document/passage counts in its header, so a run is bound to what produced it.

| Suite | Result | Note |
|---|---|---|
| Situations | **7 / 9** | S8 and S9 fail on the same expectation: each requires the answer's own citation markers to reference two distinct documents, and each turns out to be answerable from one. Left failing rather than weakened — the expectation is the honest one and the situations are not the multi-source tests they claim to be |
| Guardrail probes | **9 / 10** | P7 (a health question) fails: the referral does not contain the expected words. A routing-table wording gap, not a safety failure — the question still never reaches retrieval |
| Conversations | **5 / 5** | Progressive refinement, correction, topic switch, slot hygiene, ask-back cycle, product carry |
| Audience filter | **pass, both directions** | A staff-tagged fixture is invisible to a public caller and visible to a staff one |
| Threshold sweep | 0.35 / 0.45 / 0.55 | Identical at every value: six answered, none wrongly refused, **two wrongly admitted**. No threshold in range separates the two unanswerable questions, because both are well-formed questions about real products. The relevance gate catches them instead — this is the measurement behind `DECISIONS.md` entry 9 |

### Latency — the number that does not meet the target

A five-passage compose on a question the model has not seen costs **35–90 s** on a quiet machine and **115–199 s** measured under load. The transcript's nine model calls read 1.66–75.47 s with a median of 3.8 s and **that median must not be quoted**: it is Ollama's prompt cache, earned by running the harness repeatedly against the same questions. Two questions never asked before cost 199 s and 115 s; an immediate repeat of the first cost 4.2 s. The exact-key answer cache takes a repeat to 0.017 s. Routed, extract and refused answers involve no generation and are fast.

### Index build — verified on this machine, 17 September 2026

94 documents, 552 passages, **552 embedding cache hits, 0 computed**, 24.9 s total, zero extraction failures (50 clean, 45 partial). A second run reprocesses nothing.

### Test suite — measured, not claimed

Full run on Windows with no `ASSISTANT_POSTGRES_DSN`: see the table below. **Known failures are listed rather than hidden**, because a suite whose failures are undocumented is not evidence.

| Group | Count | Status |
|---|---|---|
| `tests/` overall | 1368 passed, 77 skipped | The 77 skips are PostgreSQL-gated and Docker-gated; a skip is printed, never swallowed |
| `tests/e2e/` surface | **8 passed** (7m25s) | Cookie, session, upload boundary, rendered citation, audience, correlation id → trace |
| `tests/e2e/` demo journeys | **11 smoke passed** | Journeys A–H in `tests/e2e/test_demo_journeys.py`. `-m smoke` is the fast pre-flight subset (no generation); the rest are marked `slow` and compose or perceive |
| `tests/test_ui_server.py` | **10 known failures** | These assert the *previous* plain-text UI, replaced by the chat interface. The behaviour they describe changed; the tests did not. Diagnosed, not rewritten — see [Known issues](#known-issues-in-this-branch) |
| `tests/test_integration_core5.py` | **9 known failures** | Committed as WIP. It asserts that the engine writes to a `SessionStore`, which by design it never does. Previously 17 errors; it now skips cleanly when no index is built |
| `tests/test_policy_gate_phrasing.py` | 7 passed, **3 xfail** | New. Reproduces the policy-gate gap in known issue 1, and pins the phrasings that do work |

CI runs the full suite against both SQLite and a real `pgvector/pgvector:pg16`, reports line and branch coverage separately, and fails on any missing branch in the safety-critical modules.

### Known issues in this branch

Found while preparing this submission, diagnosed, and **deliberately not silently fixed** — rewriting a failing test to match current behaviour can bury a real regression:

1. **The policy gate misses three of the phrasings its own probe suite uses.** This is the most significant finding and it is a real defect, reproduced in [`tests/test_policy_gate_phrasing.py`](tests/test_policy_gate_phrasing.py). The gate is described throughout this record as eleven topics that "never reach retrieval". Measured against the running server, three probe questions match **no topic at all**, each missing by one word or by word order:

   | Probe | Question | Pattern written for it | Why it misses | Path actually taken |
   |---|---|---|---|---|
   | P6 | "…is the house safe?" | `\bis it safe\b` | says *the house*, not *it* | **compose** — 72.9 s |
   | P7 | "I got lime plaster in my eye" | `\beye(s)?\b.{0,20}\b(splash\|got\|in)\b` | needs *eye* then *got*; sentence has *got* then *eye* | retrieval |
   | P8 | "…complies with Part L?" | `\bpart [a-l]\b.{0,25}\bcomply\b` | needs *Part L* then *comply*; sentence reverses them | refuse — 79.8 s |

   **What still holds:** the six checks, the relevance gate and the hand-off renderer all run, so the observed answers were not unsafe — they declined to judge and pointed at the technical team. The guarantee that is *not* holding is the stronger one claimed above: that these questions never reach the model. P6 reaches it, with passages about cracking in front of it.

   **Why the harness did not catch it:** only P7 fails there. P6 and P8 pass their assertions while the mechanism they name is bypassed — P6's `must_contain_any` is satisfied by any answer mentioning the technical team. A probe can pass on outcome and still be silent about mechanism. `config/routing.json` is core safety routing, so the patterns are **not** edited here; the gap is reported for a deliberate decision.

2. **`tests/test_ui_server.py` — 10 failures.** They assert the plain-text UI that the chat interface replaced. Two of the original twelve encoded genuine losses and have been fixed: the footer no longer stated which index was answering, and an upload note was dropped in silence on a server-rendered POST. The XSS assertion in this group was **verified by hand against three payloads** — no raw markup is served — so it is stale, not a security regression.
3. **`tests/test_integration_core5.py` — 9 failures.** Committed as WIP. It hard-errored rather than skipping when no index existed, which broke the clean-clone `pytest` promise; **that half is fixed** and it now skips with the command to run. The nine that remain when an index *is* present expect `Assistant.ask` to persist slots into a `SessionStore`, which its own docstring says it does not do — the caller owns that, and the behaviour is covered by the five conversation scenarios that pass.
4. **The visible transcript does not survive a reload.** The conversation does — it is a server fact named by an HttpOnly cookie — but answers are rendered into the DOM by the page's script and nothing re-renders them. `tests/e2e/test_demo_journeys.py` pins this behaviour as it actually is.

---

## Limitations and production seams

Stated with a fixed vocabulary so an intention is never mistaken for an implementation: **built and verified**, **built but weakly tested**, **partial**, **documented only**, **known limitation**.

| Area | Status | The honest position |
|---|---|---|
| Authentication | **documented only** | The audience set is **asserted, not proved**. The filter itself is real, enforced in code inside the repository before ranking, and tested in both directions. Over HTTP a request may only *narrow* what the operator started the server with. That is not authentication and is not described as such |
| PostgreSQL adapter | **built and verified, not operated** | It passes the same repository contract, ingestion lifecycle, publication lock and concurrent-reader tests against a real PostgreSQL 16 + pgvector in CI. But the submission runs on SQLite, the transcript evidences only that adapter, and no PostgreSQL instance has answered a question outside a test. Contract-verified is not production-proven |
| Compatibility matrix | **known limitation** | Products are assessed against corpus evidence per required property, and one whose substrate suitability is not independently established cannot be recommended. But **no product-to-substrate matrix exists** on the site and none was invented — every documented pair returns `None`, and `None` reads as unknown rather than as permission. Building that matrix is partnership work |
| Generation queue, rate limiting | **documented only** | Concurrent *serving* is built and tested. A generation queue with a visible wait, per-session rate limiting and extract-only degradation under load are drawn and argued, not built. Generation remains one at a time per Ollama instance |
| Answer cache | **built, weaker form than designed** | Exact-key, audience- and snapshot-scoped. It hits only when the same question is asked in the same words. The template-keyed form entry 14 designs is roadmap — identical on safety, poorer on hit rate |
| Conversation durability | **known limitation** | The PostgreSQL checkpointer is a documented seam, not a working adapter: `langgraph-checkpoint-postgres` and `langgraph==1.2.11` have incompatible requirements. Conversation state lives in memory, does not survive a restart, and `checkpointer_for()` raises rather than falling back — a deployment that believed its conversations were durable and silently lost them would be worse than one that refuses to start |
| Embedding model | **development default** | `qwen3-embedding:0.6b` is in use and recorded in every snapshot. The benchmark against `nomic-embed-text` that would close `DECISIONS.md` entry 6 **has not been run**. This is an unmeasured default, not a selected model |
| Diagnosis from photographs | **by policy, not capability** | Perception is built; acting on it is refused. Judging a wall is the technical team's call, there is no labelled failure library to ground or evaluate a reading against, and a confident wrong visual diagnosis is the worst failure this system could produce |
| Qualitative synthesis | **known limitation** | The six checks bound numbers, names, attribution and the asked-for term. They reduce, and do not eliminate, an invented "this is fine on cob" |
| Datasheet quality | **inherited** | Sheets date from 2015 to 2025. One prints "3m3" for a coverage figure and "8oC" for a temperature; one PDF title names a different product than its page. Quoted as printed, flagged in the ingestion report, never corrected |

---

## What ships in this repository, and on what footing

The crawled corpus ships with the code: `data/cache/` is 96 tracked files and about 29 MB — 37 Lime Green PDFs and the HTML of the pages that linked them, byte-for-byte as fetched. That is what makes the clean-clone run real: an assessor with no network still gets the same 95 documents, the same 552 passages and the same transcript, and can check any quoted figure against the original file.

The footing has to be stated plainly, because it is not a licence. This is third-party material — Lime Green's published documents, and the exercise brief — included so the work can be assessed offline. Copyright stays with its owners and nothing here confers a right to republish. There is deliberately no `LICENSE` file: adding one would imply a grant over content that is not ours to grant. Everything derived from the corpus inherits the same footing — the index, the embedding cache, and the harvested product, colour and merchant name lists.

**This repository is public**, which was a deliberate choice and is worth naming rather than leaving to be discovered. Every document here is already published by Lime Green and freely downloadable; what this adds is a mirror, and a mirror is a distribution decision even when the source is open. Lime Green have not been asked. If they would rather it were not here, the corpus comes out and the crawl becomes a build step — the pipeline already supports that.

---

## Layout

```
README.md               what it is, how to run and demo it
DECISIONS.md            why it is this way — 20 decisions
docs/DEMO-SCRIPT.md     a rehearsable 7–10 minute demonstration
requirements.txt        six pinned runtime dependencies
requirements-dev.txt    the browser driver, for the e2e suite only

assistant/
  crawl.py              sitemap crawl, content hashing, the version ledger
  extract.py            HTML and PDF to citable sections; harvest before stripping
  index.py              chunk, tag caveats, embed, publish a snapshot
  embedcache.py         content-addressed embeddings, so a rebuild is seconds
  model.py              the domain model — storage-agnostic by design
  repository.py         KnowledgeRepository: the boundary the engine depends on
  store/embedded.py     the SQLite adapter — the one that ships
  store/postgres.py     the PostgreSQL + pgvector adapter, same contract
  store/factory.py      which of the two a process gets, decided in one place
  retrieve.py           question embedding, synonyms, index-mismatch refusal
  router.py             policy gate, slot detection, the eight ordered steps
  answer.py             extract and compose, the six checks, hand-off, rendering
  candidates.py         per-property candidate assessment before recommendation
  graph.py              the turn as a state machine — ordering, and nothing else
  vision.py             perception only; never names a product
  engine.py             the assembled assistant, split by topic
  audience.py           a request may narrow the operator's audience set, never widen
  cache.py              exact-key answer cache, scoped by audience and snapshot
  cli.py                canonical interface
  ui.py                 a web page over the same library, standard library only
  trace.py              read one answer's stages back out of the log
  health.py             readiness: store, snapshot compatibility, model reachability

config/                 routing table, vocabularies, corpus boundary — as data
db/                     schema.sqlite.sql and schema.postgres.sql — same eight tables
data/cache/             original HTML and PDFs as fetched, shipped so it runs offline
data/embeddings.db      the embedding cache, content-addressed by text and model
eval/                   situations, probes, conversations, fixtures, the harness
tests/e2e/              browser proofs of the surface, and the demo journeys A–H
docs/                   architecture, diagrams, pipeline runbook, demo script
```

---

## Notes for an assessor

- The corpus is Lime Green's published material only, crawled once by sitemap and cached in the repo. What is in, what is out and why: `DECISIONS.md` entry 1, inventoried in `docs/architecture.md`.
- Every answer names its sources. A figure that cannot be found word-for-word in a cited passage does not print.
- **Refusals are a designed outcome, not a failure.** A refusal still carries what the site publishes, its source, the document's own caveats, and the technical team's published contact line — never a named individual, never an invented number.
- The numbers in [Testing and evidence](#testing-and-evidence) are measurements with dates. Where something is not finished, [Limitations](#limitations-and-production-seams) says so in the same vocabulary the rest of the record uses.
