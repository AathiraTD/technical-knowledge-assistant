# Lime Green Technical Assistant

A local-LLM technical knowledge assistant over Lime Green's published material. It retrieves passages from the company's own datasheets, guides and FAQ, cites every source by document name, and refuses — with a hand-off to the technical team — when the published material does not answer the question.

Built for the AEC Solution Architect (AI/LLM Developer) KTP take-home exercise, Birmingham City University with Lime Green Products.

## How it works, in a paragraph

A question is split by topic, gated against a routing table, and matched against slot vocabularies. What survives is embedded and retrieved from the knowledge store — active document versions only, filtered to the caller's audience in the query rather than by prompt. A deterministic router — not the model — then picks one of five paths: route to a fixed referral, print a passage verbatim, compose over several passages, quote a published hand-off, or refuse. The model runs on the compose path only, at temperature zero, over retrieved passages delimited as data. Six checks run before anything prints, and a failure sends the part to a refusal that still carries whatever the site does publish.

## Status

Design complete and evidenced against the live site. The build is in progress.

Status is reported with a fixed vocabulary, so an intention is never mistaken for an implementation: **built and verified**, **built but weakly tested**, **partial**, **documented only**, **known limitation**.

| Area | Status | Evidence |
|---|---|---|
| Decisions | built and verified | 18 recorded with alternatives and costs — `DECISIONS.md` |
| Crawler | built and verified | 94 documents, 0 errors; a re-run reports 94 unchanged |
| Extraction | built and verified | 37 PDFs probed; two heading detectors; per-document quality in the ingestion report |
| Indexer | built and verified | 94 documents + 1 staff fixture → 579 passages, 102 caveats, 39 stockists, 24 colours, 0 failures |
| Delta ingestion | built and verified | A second crawl of an unchanged site reprocesses **nothing**: 1.4s against 23.8s. Changed documents supersede and keep their history; withdrawn ones are deactivated, not deleted. Four-run proof in `DECISIONS.md` entry 18 |
| SQLite adapter | built and verified | The index builds, publishes atomically, and serves every answer in the transcript |
| Retrieval | built and verified | Audience filtering in the query; authority banding; index-mismatch refusal |
| Router | built and verified | 11-topic policy gate, 8 slots, 8 ordered steps |
| Six checks | built and verified | 16 unit tests, one per failure they exist to catch — `tests/test_checks.py` |
| Answer engine | built and verified | Seven paths, 100% branch coverage, and a full evaluation transcript in `eval/results/` |
| CLI and web page | built and verified | Both over one library; the harness drives the library. `python -m assistant.health` reports ready |
| Evaluation harness | built and verified | **7/7 situations, 10/10 probes**, audience filter passing in both directions, threshold sweep. Transcript in `eval/results/transcript.txt` |
| Embedding model | **development default** | `qwen3-embedding:0.6b`. `eval/embedding_choice.py` is the benchmark that closes decision 6 |
| PostgreSQL adapter | written, never run | Full adapter against the same Protocol and the same 8 tables, with filtering and similarity in one query. `psycopg[binary]` has no Windows ARM64 wheel and Docker's daemon is not running here, so the contract suite **skips** it rather than passing it |
| Docker Compose | written, never booted | App, PostgreSQL + pgvector and Ollama, with pinned images, named volumes, a non-root app container and readiness checks. Not yet started once |
| Authentication | documented only | The audience set is asserted, not proved — the filter itself is real and tested |
| Vision | documented only | Refused by policy, not by capability — decision 16 |
| Re-crawl hook | documented only | Change detection is built; the trigger is manual |
| Answer cache, queueing | documented only | Designed for production; nothing to cache at one user |
| Test coverage | built and verified | **336 tests, 99% overall, 100% line and branch on every safety-critical module.** The residual is four defensive branches in extraction, named in the testing section |

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
                               # a second run reprocesses only what changed
python -m assistant.cli        # ask a question
python -m assistant.ui         # the same library behind a web page
python -m eval.run             # seven situations, probe suite, threshold sweep
```

The crawled pages and PDFs ship in `data/cache/`, so the indexer runs without network access. Only the index is rebuilt locally, because it is tied to the embedding model on your machine.

`python -m assistant.crawl` re-fetches from the site, and you should not need it: it reads the cache when present, and only documents whose content hash has changed are fetched again.

## Layout

```
README.md               what it is, how to run it
DECISIONS.md            why it is this way — 18 decisions
requirements.txt        five pinned dependencies

assistant/
  crawl.py              sitemap crawl, content hashing, the version ledger
  model.py              the domain model — storage-agnostic by design
  repository.py         KnowledgeRepository: the boundary the engine depends on
  store/embedded.py     the SQLite adapter behind it
  extract.py            HTML and PDF to citable sections; harvest before stripping
  index.py              chunk, tag caveats, embed, publish a snapshot
  embedcache.py         content-addressed embeddings, so a rebuild is seconds
  ollama.py             two HTTP endpoints, retried; no client library
  retrieve.py           question embedding, synonyms, index-mismatch refusal
  router.py             policy gate, slot detection, the eight ordered steps
  answer.py             extract and compose, the six checks, hand-off, rendering
  engine.py             the assembled assistant, split by topic
  cli.py                canonical interface
  ui.py                 a web page over the same library, standard library only

config/
  sources.json          the corpus boundary as executable configuration
  routing.json          the policy gate: eleven topics that never reach retrieval
  vocabularies.json     slot vocabularies, synonyms, deferral markers
db/
  schema.sqlite.sql     the assessment adapter — stdlib, ships, offline
  schema.postgres.sql   the deployment adapter — PostgreSQL + pgvector
data/
  cache/                original HTML and PDFs as fetched, shipped so it runs offline
  embeddings.db         the embedding cache, content-addressed by text and model
  index/                generated — the knowledge store and the ingestion report
eval/
  situations.json       seven situations with mechanical expectations
  probes.json           ten guardrail probes
  fixtures/             a synthetic staff-tagged document that must stay invisible
  run.py                the harness
  embedding_choice.py   the measurement behind decision 6
tests/
  test_checks.py        the six checks, against the failures they exist to catch
docs/
  architecture.md       diagrams, component reference, corpus inventory
  diagrams/             Mermaid sources for the three diagrams
```

## Tests

```
pip install pytest coverage
python -m pytest -q tests/                 # no Ollama, no index, no network
python -m coverage run --rcfile=.coveragerc -m pytest -q tests/
python -m coverage report --rcfile=.coveragerc
```

Nothing in `tests/` contacts Ollama, reads the network or needs a built index.
That is deliberate: the part of the system that decides whether an answer prints
is a set of pure functions over text and passages, and it should be testable
without a model running.

Two targets, and they are different on purpose.

**Safety-critical logic carries a 100% line and branch target** — the router and
its precedence, the six checks, the relevance gate, audience filtering,
active-version filtering, snapshot publication, the index-compatibility refusal,
and the load-bearing-slot rules. These are the branches where a miss is not a
missing feature but a wrong answer printed with a citation. CI fails the build
if any of them drops below 100%.

**Everything else targets 90%.** `crawl.py` talks to a live website, the CLI and
the web page are argument parsing and HTML, and the PostgreSQL adapter cannot
run on the build machine, so all four are excluded from measurement rather than
counted at a number that would describe the environment instead of the tests.

Measured, with 336 tests: **99% overall, one missed statement and four partial
branches, all in `extract.py` and `index.py` at 99% each.** Naming them rather
than hiding them, because they are the only gap:

| Location | Branch | Why it is not covered |
|---|---|---|
| `extract.py:200` | text empty after cleaning a PDF span | The span list is already filtered on `sp["text"].strip()`, so cleaning cannot empty it |
| `extract.py:282, 290` | section body empty after cleaning | Same guard, on a buffer that only receives non-empty lines |
| `index.py:129` | a chunk buffer that is only whitespace | `_split_on_bullets` discards whitespace-only parts before this runs |

Each is a defensive guard against a condition an earlier filter already
prevents. They are kept because extraction is the part most likely to change
when the site does, and cheap insurance there is worth one uncovered branch.
None of them is in the safety-critical set.

The repository contract is one behavioural suite run against every adapter. It
passes against SQLite here and **skips** against PostgreSQL, visibly, because
`psycopg` publishes no Windows ARM64 wheel and Docker's daemon is not running on
this machine. CI runs the same suite against a real pgvector service, which is
the only thing that turns "the two adapters are one system" into a check rather
than a claim.

## Notes for an assessor

- The corpus is Lime Green's published material only, crawled once by sitemap and cached in the repo. The boundary — what is in, what is out, and why — is in `DECISIONS.md` entry 1 and inventoried in `docs/architecture.md`.
- Every answer names its sources by document name. A figure that cannot be found word-for-word in a cited passage does not print.
- The assistant refuses rather than guesses, and a refusal still carries what the site publishes plus the technical team's published contact line. Refusals are a designed outcome, not a failure.
