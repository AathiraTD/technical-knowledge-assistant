# Lime Green Technical Knowledge Assistant

A technical knowledge assistant that answers product questions using only approved sources. It retrieves passages from Lime Green datasheets and guides, verifies citation accuracy and product scope, checks evidence sufficiency, and refuses or hands off when the published material is insufficient.

The system separates concerns: LLMs provide language understanding, retrieval provides evidence, deterministic code provides control. Local mode changes infrastructure, not safety semantics.

---

## Why this exists

Technical product recommendations require more than semantic search. A useful answer must be:

- **Grounded in approved sources** — retrieved from indexed, versioned corpus, not from model knowledge
- **Product-scoped** — not just cited correctly, but about the product the question asked
- **Evidence-sufficient** — candidate assessment and numeric binding verified before printing
- **Numerically consistent** — figures are verbatim, not paraphrased; units normalised for comparison only
- **Properly cited** — every factual sentence traceable to a passage, with word overlap verified
- **Safe to refuse** — fail-closed when evidence is insufficient rather than inventing

---

## Architecture

```
user question + conversation state
        ↓
deterministic policy gate (price, safety, health, compliance)
        ↓
structured understanding (substrate, location, exposure, symptom, property)
        ↓
approved-source retrieval (active versions, audience-filtered, authority-ranked)
        ↓
evidence-sufficiency gate + candidate assessment
        ↓
extract (verbatim) | compose (LLM over passages) | refuse | hand-off
        ↓
seven verification checks (citation, numbers, product, qualifiers, names, property, scope)
        ↓
cited answer + trace
```

**Core principle**: Evidence and reasoning are decoupled. Retrieved evidence does not authorize recommendation by itself; retrieved product does not authorize that product's recommendations; cited passage does not authorize all products mentioned in it.

---

## Key design decisions

- **Approved-source boundary** — no web crawl, no speculative knowledge
- **Conversation state separated from transcript** — trusted slots (substrate, location, exposure) are tracked; prior assistant answers do not re-enter as evidence
- **Product-aware retrieval** — when product is known, generic (FAQ, article, system guide) documents are deprioritised if product-specific evidence covers the property
- **Deterministic routing** — policy rules, calculation rules, and refusal rules are code, not prompts
- **Seven post-generation checks** — citation presence, numeric verbatim match, product attribution, qualifier/caveat adjacency, real names only, property presence in evidence, product-scope correctness
- **Vision observations typed and vocabulary-gated** — substrate and symptom may be filled; location/exposure are not inferred from photos; observations carry confidence and are user-correctable
- **Fail-closed architecture** — weak evidence triggers refusal (with published material printed) rather than weaker verification

See [Reliability and design evolution](docs/RELIABILITY_AND_DESIGN_EVOLUTION.md) for how these were validated through measured failures.

---

## Local runtime

The interview build intentionally uses SQLite, local embeddings, Ollama and in-memory conversation state. The corpus is small and the exercise prioritises a self-contained local system. This keeps the system runable from a clean clone without external dependencies.

**What is local:**
- Knowledge store: SQLite
- Vector similarity: local numpy-based retrieval
- Embedding model: `qwen3-embedding:0.6b` via Ollama (local CPU)
- Generation model: `qwen3.5:4b` via Ollama (local CPU)
- Conversation state: LangGraph in-memory (lost on restart)
- Vision: optional, runs locally if enabled (CPU-only, slow)
- Authentication: none (audience asserted, not verified)
- Tracing: hosted LangSmith disabled in code

**What is NOT local:**
- No PostgreSQL requirement
- No pgvector requirement
- No rate limiting or request queue
- No durable checkpointing
- No message broker

Selecting a different backend does not change the answer engine's logic, citations, checks, or fail-closed behaviour — only where bytes are stored.

---

## Production direction

These are explicitly seams, not part of the submitted demo:

| Capability | Status |
|---|---|
| PostgreSQL + pgvector storage | adapter/contract-tested; not operated locally |
| Durable conversation checkpointing | designed seam; dependency-blocked (`langgraph-checkpoint-postgres` version conflict) |
| Authenticated identity | designed seam; currently asserted over HTTP, not verified |
| Scalable inference | designed seam; one Ollama instance serialises generation |
| Rate limiting / queueing | not implemented |

---

## Quick start

### Requirements

- Python 3.11+
- [Ollama](https://ollama.com/download) running locally

### Pull models

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

### Check readiness

```powershell
.\scripts\start-demo.ps1 -CheckOnly
```

### Start demo (Windows)

```powershell
.\scripts\start-demo.ps1
```

Opens http://localhost:5000 in your browser.

### Start demo (manual)

```bash
pip install -r requirements.txt
python -m assistant.ui
```

### Enable vision (optional, slow on CPU)

```powershell
$env:ASSISTANT_VISION_DEMO="1"
.\scripts\start-demo.ps1
```

### Evaluate

```bash
python -m eval.run              # Structured conversation evaluation
python -m assistant.cli -q "..."  # Single-turn CLI
python -m assistant.health      # Readiness check
python -m pytest tests/          # Unit/integration tests
```

---

## Repository structure

```
assistant/        answer engine, LLM orchestration, retrieval, verification
data/            indexed knowledge, embeddings, local cache
eval/            evaluation scenarios, gold fixtures, test harness
tests/           automated regression, contract tests for storage/repository
scripts/         operational commands (startup, verification)
docs/            architecture, decisions, reliability history
deploy/          production container / Docker Compose
config/          routing tables, vocabularies, source metadata
db/              schema (SQLite + PostgreSQL)
```

---

## Documentation

- **[Architecture](docs/architecture.md)** — System design, component reference, decision rationale
- **[Reliability and design evolution](docs/RELIABILITY_AND_DESIGN_EVOLUTION.md)** — How measured failures drove design fixes and validation
- **[Architecture decisions](DECISIONS.md)** — Why each choice was made, alternatives considered, where it breaks
- **[Runtime profiles](docs/architecture.md#runtime-profiles-local-demo-vs-production)** — Local vs production infrastructure, honest status of each capability
- **[Demo script](docs/DEMO-SCRIPT.md)** — Interview demo sequence with test questions and expected behaviour
- **[Knowledge pipeline](docs/knowledge-pipeline.md)** — Indexing, versioning, publication, delta updates

---

## Known limitations

- **Local CPU inference latency** — 20–40+ seconds per uncached question (Ollama prompt cache significantly speeds repeats)
- **Conversation state not durable** — in-memory only; lost on process restart
- **Vision perception latency** — 150–200+ seconds per image on local CPU; timeouts possible under load
- **Checkpointing dependency blocked** — PostgreSQL checkpoint backend cannot install due to version conflict
- **Single Ollama instance** — generation serialised; no parallelism without additional infrastructure
- **Embedding model not formally selected** — `qwen3-embedding:0.6b` is development default; formal benchmark deferred
- **Limited property retrieval hardening** — some targeted property queries may still need stronger evidence scoping
- **No production authentication** — audience is asserted CLI/HTTP parameter, not issued by identity system

See [Limitations](docs/architecture.md#remaining-known-limitations) in architecture docs for non-infrastructure constraints.

---

## Design philosophy

The system does not treat prompt engineering as the primary safety mechanism. Instead:

- **Evidence is retrieved**, not generated
- **Policy is deterministic code**, not prompt instructions
- **Routing is explicit and ordered**, with clear precedence
- **Verification happens after generation**, not inside the prompt
- **Fail-closed** means refusal with published material, not weaker checks

A generated answer is not treated as trusted merely because the LLM produced it. Seven distinct checks verify citation, numeric accuracy, product scope, qualifier adjacency, real names, property presence, and product consistency before anything is printed.

---

Built for the AEC Solution Architect KTP exercise, Birmingham City University with Lime Green Products.
