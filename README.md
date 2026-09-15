# Lime Green Technical Assistant

A local-LLM technical knowledge assistant over Lime Green's published material. It retrieves passages from the company's own datasheets, guides and FAQ, cites every source by document name, and refuses — with a hand-off to the technical team — when the published material does not answer the question.

Built for the AEC Solution Architect (AI/LLM Developer) KTP take-home exercise, Birmingham City University with Lime Green Products.

## How it works, in a paragraph

A question is split by topic, gated against a routing table, and matched against slot vocabularies. What survives is embedded and retrieved from the knowledge store — active document versions only, filtered to the caller's audience in the query rather than by prompt. A deterministic router — not the model — then picks one of five paths: route to a fixed referral, print a passage verbatim, compose over several passages, quote a published hand-off, or refuse. The model runs on the compose path only, at temperature zero, over retrieved passages delimited as data. Six checks run before anything prints, and a failure sends the part to a refusal that still carries whatever the site does publish.

## Status

Design complete and evidenced against the live site. The build is in progress.

| Area | State |
|---|---|
| Decisions | 16 recorded, with alternatives and costs — see `DECISIONS.md` |
| Site inventory | Confirmed against the live site, then crawled: **94 documents, 0 errors** |
| Crawler | **Built.** Sitemap-driven, rate limited, robots-aware, cache-backed, with SHA-256 content hashing and a version ledger — a re-run reports 94 unchanged |
| Storage boundary | **Built.** `KnowledgeRepository` with two schemas: SQLite (ships) and PostgreSQL + pgvector (deployment) |
| Indexer | In progress — extract, chunk, tag caveats, embed, publish |
| Answer engine | Not started |
| Evaluation harness | Not started |
| Web UI | Not started |
| Environment | Ollama 0.34.0 with `qwen3.5:4b` and `qwen3-embedding:0.6b` pulled |

## Read this first

| If you want | Read |
|---|---|
| Why it is built this way — every decision, alternative and cost | [`DECISIONS.md`](DECISIONS.md) |
| What the system is — diagrams, component reference, corpus inventory | [`docs/architecture.md`](docs/architecture.md) |
| The analysis it came from — segments, jobs, question types, data, guardrails | [`docs/working-record.docx`](docs/working-record.docx) |
| The exercise itself | [`docs/brief.docx`](docs/brief.docx) |

## Requirements

- Python 3.11 or later
- [Ollama](https://ollama.com/download) installed and running locally, with two models pulled:
  ```
  ollama pull qwen3.5:4b            # 3.4 GB — generation, 256K context
  ollama pull qwen3-embedding:0.6b  # 639 MB — embeddings
  ```
  About 4 GB in total; allow time on a slow connection. `qwen3:4b-instruct` (2.5 GB) is the documented fallback if `qwen3.5` turns out to emit reasoning blocks — see `DECISIONS.md` entry 7. Tags were verified against the Ollama library on 15 September 2026. They are fixed in `config/`, recorded in the index header, and the engine refuses to run against a mismatch rather than returning confident nonsense.

## Running it

Once the build lands, from a clean clone:

```
pip install -r requirements.txt
python -m assistant.index      # crawl is cached in the repo; builds the index offline
python -m assistant.cli        # ask a question
python -m assistant.ui         # the same library behind a web page
python -m eval.run             # seven situations, probe suite, threshold sweep
```

The crawled pages and PDFs ship in `data/cache/`, so the indexer runs without network access. Only the index is rebuilt locally, because it is tied to the embedding model on your machine.

`python -m assistant.crawl` re-fetches from the site, and you should not need it: it reads the cache when present, and only documents whose content hash has changed are fetched again.

## Layout

```
README.md               what it is, how to run it
DECISIONS.md            why it is this way — 15 decisions
requirements.txt        pinned dependencies

assistant/              the package (five to eight modules)
  crawl.py              sitemap crawl, content hashing, the version ledger
  model.py              the domain model — storage-agnostic by design
  repository.py         KnowledgeRepository: the boundary the engine depends on
  store/                the two adapters behind it
  index.py              extract, chunk, tag caveats, embed, publish a snapshot
  retrieve.py           embed the question, cosine, top-k, per-document cap, audience filter
  router.py             policy gate, slot detection, the ordered router
  answer.py             extract and compose, the six checks, hand-off, rendering
  cli.py                canonical interface
  ui.py                 a web page over the same library, served from the standard library

config/                 hand-written: routing table, vocabularies, authority and audience rules
db/                     one data model, two dialects
  schema.sqlite.sql     the assessment adapter — stdlib, ships, offline
  schema.postgres.sql   the deployment adapter — PostgreSQL + pgvector
data/
  cache/                original HTML and PDFs as fetched, versioned by content hash,
                        shipped so it runs offline
  index/                generated — the SQLite knowledge store and the ingestion
                        report (not committed; rebuilt from the cache)
eval/
  situations.*          seven transcript situations with expected sources and refusal states
  probes.*              one-line guardrail probes
  fixtures/             a synthetic staff-tagged document that must stay invisible in public mode
  results/              transcripts and the threshold sweep
docs/
  architecture.md       diagrams, component reference, corpus inventory
  diagrams/             Mermaid sources for the three diagrams
  brief.docx            the exercise
  working-record.docx   the analysis behind the design
  DECISIONS.docx        generated Word export of DECISIONS.md
```

Directories not yet present appear as the build reaches them.

## Notes for an assessor

- The corpus is Lime Green's published material only, crawled once by sitemap and cached in the repo. The boundary — what is in, what is out, and why — is in `DECISIONS.md` entry 1 and inventoried in `docs/architecture.md`.
- Every answer names its sources by document name. A figure that cannot be found word-for-word in a cited passage does not print.
- The assistant refuses rather than guesses, and a refusal still carries what the site publishes plus the technical team's published contact line. Refusals are a designed outcome, not a failure.
