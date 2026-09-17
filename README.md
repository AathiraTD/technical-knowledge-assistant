# Lime Green Technical Knowledge Assistant

A local-LLM technical knowledge assistant over Lime Green Products' published material. It retrieves passages from approved Lime Green datasheets, guides and FAQ content, preserves provenance and version context, cites its evidence, and refuses or hands off when the published material is insufficient to support an answer.

The submitted demo runs locally with SQLite, LangGraph in-memory conversational state, and Ollama-served local models. No hosted model API key is required for answering.

Built for the AEC Solution Architect (AI/LLM Developer) KTP take-home exercise, Birmingham City University with Lime Green Products.

> \*\*New here?\*\* Start with \[Running it](#running-it), then \[How to demo](#how-to-demo). \[Limitations and production seams](#limitations-and-production-seams) states what is not production-complete.

\---

## What it demonstrates

|Capability|What is implemented|
|-|-|
|**RAG over approved sources**|The assistant indexes the approved Lime Green corpus rather than indiscriminately crawling arbitrary web content.|
|**Semantic retrieval**|Queries are embedded with `qwen3-embedding:0.6b`; semantic similarity search is combined with product, audience, active-version, authority and de-duplication constraints.|
|**Versioned knowledge**|Source identity, active document version, snapshot and chunking metadata are retained. Older versions can remain for provenance while only the active version participates in retrieval.|
|**Stateful conversation**|LangGraph is the orchestration/state layer. Facts carry across turns, corrections supersede earlier values, ask-back can interrupt and resume, and case/topic boundaries prevent stale context leaking into a new wall/case.|
|**Evidence sufficiency**|Retrieved text is not automatically treated as sufficient. Product/property evidence is checked before recommendation or composition.|
|**Image input**|Optional multimodal perception converts wall photographs into typed observations with explicit uncertainty. Visual observations do not directly authorize a product recommendation.|
|**Product recommendation containment**|A product may be recommended only if it survives candidate assessment against approved evidence; final recommendation containment is independent of intent classification.|
|**Deterministic safety routing**|Safety, health, compliance, price and other governed topics can terminate before retrieval/generation. Paraphrase and word-order regressions are covered by tests.|
|**Deterministic calculations and checks**|Published numeric values are bound to the correct product/evidence and calculations are performed outside the LLM where appropriate.|
|**Observability**|Each answer receives a correlation id and stage-level trace information that can be inspected with `python -m assistant.trace`.|
|**Evaluation**|Unit/integration tests, structured conversation evaluation, adversarial guardrail tests, browser smoke/E2E journeys and multimodal false-positive evaluation are included.|
|**Reproducible startup**|Preflight/readiness checks and a production Docker image are included. The live demo remains intentionally lightweight.|

\---

## How it works

The assistant is a **governed agentic workflow**, not an unconstrained autonomous agent.

```text
user question / image
        |
        v
deterministic policy gate
        |
        v
structured understanding
        |
        v
conversation + case state
        |
        +---- missing critical fact ----> ask back / resume
        |
        v
semantic retrieval over approved evidence
        |
        v
evidence-sufficiency / candidate assessment
        |
        +---- insufficient -------------> refuse / conditional answer / hand-off
        |
        v
extract / deterministic calculation / grounded composition
        |
        v
final checks + recommendation containment
        |
        v
cited answer + trace
```

The LLM is used where model intelligence is useful — structured language understanding and grounded natural-language composition. Deterministic/domain-controlled components retain authority over policy routing, product identity, source/version constraints, calculations, evidence sufficiency, recommendation eligibility, citations and refusal behaviour.

The image path follows the same principle:

```text
photograph
   |
   v
typed observations + uncertainty
   |
   v
trusted conversation state
   |
   v
missing-information gate
   |
   v
official evidence retrieval
   |
   v
candidate/evidence assessment
   |
   v
supported recommendation or hand-off
```

A VLM observation is not automatically a fact, and a model-proposed product is not automatically an eligible product.

\---

## Requirements

* Python 3.11 or later
* [Ollama](https://ollama.com/download) installed and running locally
* Models:

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

`qwen3.5:4b` is used for local generation/structured model tasks and, when the vision demo is enabled, for multimodal perception on the current demo machine.

`qwen3-embedding:0.6b` is used for query/passage embeddings. The embedding model identity and dimensionality are tied to the knowledge snapshot so an incompatible index is rejected rather than silently queried.

\---

## Running it

### Recommended Windows demo startup

```powershell
.\\scripts\\start-demo.ps1 -CheckOnly
.\\scripts\\start-demo.ps1
```

`-CheckOnly` verifies the environment and starts nothing. The normal command performs the same checks and then starts the web page.

The preflight/readiness path checks, in dependency order:

* Python/runtime dependencies
* Ollama reachability
* required model tags
* knowledge index availability
* snapshot/model compatibility

### Manual startup

```bash
pip install -r requirements.txt
python -m assistant.index
python -m assistant.ui
```

Other useful entry points:

```bash
python -m assistant.health
python -m assistant.cli -q "How much water does Solo Onecoat need per bag?"
python -m assistant.trace
python -m assistant.trace <correlation-id>
python -m eval.run
```

### Readiness

`/health` is liveness-only.

`/ready` reports whether the running process can actually serve answers and returns `200` when ready or `503` when a required dependency has become unavailable after startup.

The CLI readiness report includes the knowledge store, active snapshot, embedding model, generation model and Ollama state. Sensitive connection details are not exposed by the unauthenticated HTTP readiness endpoint.

\---

## Vision demo

Image support is implemented but **disabled by default** in the live demo because local CPU-only perception is slow.

Enable it with:

```powershell
$env:ASSISTANT\_VISION\_DEMO="1"
```

or:

```bash
export ASSISTANT\_VISION\_DEMO=1
```

Accepted truthy values are `1`, `true`, `yes`, `on`, and `enabled` (case-insensitive).

When disabled, image upload remains a supported state: the system returns a truthful hand-off rather than pretending perception occurred.

When enabled:

* the image is bounded/sniffed before inference,
* the VLM returns typed observations,
* `location` and `exposure` are not inferred from a wall photograph,
* a substrate claim is gated when a covering finish is visible,
* user-stated facts outrank conflicting visual observations,
* image observations still have to pass the same evidence-sufficiency and recommendation rules as text.

### Important demo limitation

Real local VLM inference on the current machine has been measured in the roughly **156–198 second** range for successful examples, with worse degradation under memory pressure. One evaluation image exceeded a 600-second timeout under load.

For the interview/demo, pre-run the representative image scenario or use `tools/vision\_demo.py` / the recorded trace rather than depending on live perception in the room.

The multimodal evaluation is designed around **dangerous false positives**, not just generic visual accuracy: a photograph may support an observation such as visible staining, but it must not turn that into an unsupported diagnosis such as rising damp, structural safety, regulatory compliance or hidden-substrate certainty.

\---

## How to demo

A compact sequence:

1. **Published technical fact**

   * `How much water does Solo Onecoat need per bag?`
   * Shows grounded evidence and citation.
2. **Unsupported property**

   * `What is the U-value of Solo Onecoat plaster?`
   * Shows evidence-sufficiency refusal rather than an invented figure.
3. **Ask-back / resume**

   * `Which plaster should I use?`
   * Reply with the missing substrate when asked.
   * Shows real interrupt/resume and trusted state.
4. **Multi-turn product continuity**

   * `I have an old solid brick wall and want to improve its insulation. Would Lime Green Ultra be suitable internally, and what thickness can it be applied at?`
   * Then: `How much would I need for 30 m² at 25 mm?`
   * Shows product/context carry and product-bound evidence.
5. **False premise correction**

   * `Ultra covers 0.8 m² per bag at 25 mm, right?`
   * The system should correct against the published 0.6 m² figure rather than agree.
6. **Diagnosis-style question**

   * `My external lime render is showing patchy colour after drying. What could be causing it?`
   * Shows evidence-backed possibilities without claiming a definitive diagnosis.
7. **Deterministic safety routing**

   * `I got lime plaster in my eye, what should I do?`
   * The deterministic policy route should complete without retrieval/generation.
8. **Image-assisted case**

   * Enable `ASSISTANT\_VISION\_DEMO=1` only when you intentionally want live perception, or use the prepared image demo/trace.
   * Ask what can be reliably observed, what remains uncertain, and what published guidance is relevant.

The full rehearsal/failure playbook is in `docs/DEMO-SCRIPT.md`.

### Warm the server process

The answer cache is in-process. Warming via the CLI does **not** warm the web server.

Before the interview, start the web server and ask the questions you plan to demonstrate through that running server. For cache-sensitive examples, use a fresh chat where appropriate because conversation state is part of the cache key.

\---

## Architecture

### Retrieval

Vector/semantic search does **not** require a dedicated vector database in the submitted demo.

The demo path uses the local repository/storage adapter and local embedding similarity. Retrieval then applies domain constraints such as:

* active document version
* audience
* product identity
* requested property
* source authority
* per-document diversity/de-duplication

The repository boundary allows a production deployment to use PostgreSQL + pgvector without changing the answer/domain logic.

### Storage

`KnowledgeRepository` is the storage/retrieval boundary.

Current demo:

```text
SQLite
+ local semantic retrieval
+ LangGraph InMemorySaver
+ Ollama
```

Production-shaped seam:

```text
PostgreSQL
+ optional pgvector
+ durable shared conversation checkpointing
+ replicated API/model-serving tiers
```

PostgreSQL is useful even apart from vectors for transactional document/version metadata, audit/provenance, shared state and production concurrency.

The PostgreSQL repository adapter is contract-tested, but the submitted demo does **not** rely on PostgreSQL.

### Ollama

Ollama is the local model server.

Conceptually:

```text
assistant
   |
   | HTTP
   v
Ollama
   +-- qwen3.5:4b
   +-- qwen3-embedding:0.6b
```

This keeps model lifecycle/serving outside the Python application and provides one local interface for generation, structured model output, embeddings and optional vision.

\---

## Safety and hallucination controls

The design does not assume that prompt engineering can eliminate hallucination.

Controls include:

* approved-source corpus boundary
* versioned evidence
* product-aware retrieval
* audience filtering before ranking
* evidence-sufficiency gate
* candidate assessment before recommendation
* recommendation containment independent of intent classification
* deterministic calculations
* numeric/product binding checks
* caveat/qualifier checks
* citation/evidence checks
* deterministic policy routes
* explicit ask-back for missing decision-critical facts
* refusal/handoff on insufficient evidence
* user-stated facts outranking VLM/model guesses
* adversarial and regression tests
* per-turn observability

A generated answer is not treated as trusted merely because the model produced it.

\---

## Privacy and responsible use

The demo is designed to operate locally:

* local knowledge store
* local embeddings
* local Ollama models
* no hosted model API required for answering
* hosted LangSmith tracing is explicitly disabled, including hostile environment-variable cases

Operational tracing intentionally avoids recording full question/answer/passage text in the stage spans. Correlation ids, routes, timings and model identifiers provide observability without exposing chain-of-thought.

Image uncertainty is represented rather than hidden. The system does not claim that a wall photograph alone can establish:

* exact cause of damp
* structural safety
* Building Regulations compliance
* exact mortar chemistry
* hidden substrate
* final compatibility

Those are ask-back / qualification / technical hand-off cases.

\---

## Testing and evidence

### Focused regression evidence

Before integration:

* Core focused regression: **346 passed**
* Vision focused regression: **562 passed, 7 skipped, 0 failed**
* Browser smoke subset: **11 passed** and was stable across repeated runs

### Browser journeys

The E2E suite covers:

* basic RAG and citations
* multi-turn context
* image upload
* ask-back/resume
* deterministic safety hand-off
* New Chat reset
* reload behaviour
* trace/correlation-id flow

The full browser suite is intentionally much slower because local Ollama inference is serialised and vision can take minutes on the current machine.

### Policy-gate regression

Core specifically fixed the cross-branch policy phrasing failures for:

* structural-safety paraphrases
* eye/health word-order variation
* Part L/compliance word-order variation

Measured fixed routes completed in **22–48 ms**, with zero retrieved sources and no model invocation.

### Final integrated numbers

Run and record the final integrated results after this merge:

```bash
python -m pytest -q tests/
python -m pytest tests/e2e -m smoke -q
python -m eval.run --gold-only
```

Branch-level counts above are pre-integration evidence and should not be presented as the final post-merge count.

\---

## Performance findings

A small concurrency benchmark was used to identify the bottleneck rather than claim production capacity.

Representative findings on the local machine:

* deterministic policy fast path: \~0.31 s class
* cached repeat: \~0.51 s class
* single uncached generated answer: \~28.5 s in the measured benchmark, with substantially higher latency under load
* semantic/vector store search itself: \~36 ms in the measured trace
* under concurrent requests, query embedding also slows because it queues behind generation on the same Ollama instance
* zero observed request drops in the small 1/3/5-request benchmark; requests queue

The primary bottleneck is **local model serving**, not SQLite similarity search.

A production scale-out would separate/replicate model-serving concerns, add queueing/rate limiting, externalise shared state, and load-test against explicit SLOs before making concurrency claims.

\---

## Docker / deployment

Docker is used for reproducibility and runtime isolation, not because the system needs a vector database.

The production image was verified to:

* build successfully after fixing `.dockerignore`/`COPY` incompatibilities,
* run as non-root uid `10001`,
* contain the expected application files,
* import the application inside the image,
* keep hosted tracing disabled even when hostile tracing environment variables are present.

A full live-model `docker compose up` was **not** completed in the platform verification, so this README does not claim that it was.

Useful verification command:

```powershell
.\\scripts\\verify-docker-deployment.ps1 -PipIndexUrl "https://packagefeedproxy.microsoft.io/pypi/simple/"
```

The `deploy/` stack is the production-oriented container path.

\---

## Limitations and production seams

|Area|Status|Position|
|-|-|-|
|Authentication|**documented only**|Audience filtering is real, but the HTTP audience is asserted/narrowed, not authenticated. A production deployment needs SSO/account-backed claims.|
|PostgreSQL|**contract-verified, not the demo runtime**|The adapter exists and is tested; the submitted demo uses SQLite.|
|LangGraph durability|**known limitation**|Demo conversation state uses `InMemorySaver`; it does not survive restart or share state across app replicas. The current LangGraph/Postgres-checkpointer dependency versions are incompatible, and the code fails explicitly rather than pretending durability exists.|
|Vision latency|**known limitation**|Image inference works but is too slow/unpredictable on this CPU-only demo machine for a time-critical live demonstration.|
|Generation scalability|**documented seam**|One Ollama instance serialises expensive work. No production queue, rate limiting or load shedding is claimed.|
|Compatibility knowledge|**known limitation**|No product/substrate compatibility matrix is invented where Lime Green has not published one. Unknown stays unknown.|
|Embedding choice|**development default**|`qwen3-embedding:0.6b` is in use and snapshot-bound, but a formal embedding-model benchmark has not selected it as globally optimal.|
|Qualitative synthesis|**bounded, not infallible**|Deterministic checks reduce unsupported output; they do not prove that every qualitative sentence is universally correct.|
|Visible transcript reload|**known UI limitation**|Server-side conversation state can remain while browser-rendered transcript history is not reconstructed after reload.|

\---

## Repository hygiene before making the repository public

Before final public submission:

1. **Remove or redact `docs/brief.docx`** if it still contains BCU staff email addresses / author metadata.
2. Remove tracked synthetic `data/failure\_library/...` artifacts if they are not intended to ship.
3. Remove or sanitise generated presentation metadata that identifies `OpenAI` as `dc:creator` if those files are retained.
4. Remove unreferenced walkthrough PNGs if they are not needed.
5. Treat `deploy/` as the authoritative Docker path and avoid pointing assessors to the stale root stack.
6. Do not publish secrets, machine-specific paths, generated test logs or browser reports.

\---

## Read this first

|If you want|Read|
|-|-|
|Why it is built this way|`DECISIONS.md`|
|Architecture/component reference|`docs/architecture.md`|
|How to demonstrate it|`docs/DEMO-SCRIPT.md`|
|Demo startup/recovery|`docs/demo-runbook.md`|
|Knowledge-pipeline operation|`docs/knowledge-pipeline.md`|
|Container deployment|`docs/deployment.md`|

\---

## Layout

```text
README.md
DECISIONS.md
CLAUDE.md

assistant/
  audience.py
  answer.py
  cache.py
  candidates.py
  cli.py
  engine.py
  graph.py
  health.py
  model.py
  repository.py
  retrieve.py
  router.py
  trace.py
  ui.py
  understanding.py
  vision.py
  store/
    embedded.py
    postgres.py

config/
  routing.json
  vocabularies.json
  sources.json

db/
  schema.sqlite.sql
  schema.postgres.sql

data/
  cache/
  embeddings.db
  index/

eval/
  gold.json
  situations.json
  probes.json
  conversations.json
  vision\_eval.py
  fixtures/

tests/
  e2e/

tools/
  make\_image\_fixtures.py
  vision\_demo.py

scripts/
  preflight.py
  start-demo.ps1
  benchmark.py
  verify-docker-deployment.ps1

deploy/
  Dockerfile
  docker-compose.yml

docs/
  architecture.md
  DEMO-SCRIPT.md
  demo-runbook.md
  deployment.md
  knowledge-pipeline.md
```

\---

## Notes for an assessor

* The assistant is intentionally **not** a generic chatbot. It is a governed evidence-backed technical assistant.
* Semantic retrieval helps locate evidence; retrieval confidence alone does not authorise an answer.
* Refusal and hand-off are designed outcomes, not failures.
* The submitted demo is intentionally lightweight and local; production seams are stated explicitly rather than presented as completed deployment work.
* The important reliability claim is not that the system can never be wrong. It is that important failure modes are represented, tested, observable, and designed to fail closed where evidence is insufficient.

