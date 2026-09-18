# Lime Green Technical Knowledge Assistant

A technical knowledge assistant that answers product questions using only approved sources. It retrieves passages from Lime Green datasheets and guides, verifies citation accuracy and product scope, checks evidence sufficiency, and refuses or hands off when the published material is insufficient.

The system separates responsibilities: LLMs provide language intelligence, retrieval provides evidence, and deterministic software provides control. Local mode changes infrastructure, not safety semantics.

---

## Why this exists

Technical product recommendations require more than semantic search. A useful answer must be:

- **Grounded in approved sources** — retrieved from indexed, versioned corpus, not from model knowledge
- **Product-scoped** — not just cited correctly, but about the product the question asked
- **Evidence-sufficient** — evidence support and numeric claims verified before printing
- **Numerically consistent** — numeric claims are checked against published evidence; units may be normalised for comparison
- **Properly cited** — factual product claims must be grounded in cited passages and pass post-generation support checks
- **Safe to refuse** — fail-closed when evidence is insufficient rather than inventing

---

## Architecture

```
user question + trusted conversation state
        ↓
deterministic policy + request resolution
        ↓
structured understanding
        ↓
approved-source retrieval
        ↓
evidence binding + sufficiency assessment
        ↓
extract | compose | refuse | hand-off
        ↓
post-generation verification
        ↓
cited answer + trace
```

**Core principle**: Evidence and reasoning are decoupled. Retrieved evidence does not authorize recommendation by itself; retrieved product does not authorize that product's recommendations; cited passage does not authorize all products mentioned in it.

---

## Key design decisions

- **Approved-source boundary** — no web crawl, no speculative knowledge
- **Conversation state separated from transcript** — trusted slots (substrate, location, exposure) are tracked; prior assistant answers do not re-enter as evidence
- **Product-aware evidence scoping** — when product-specific evidence already covers the requested property, generic non-product passages can be excluded from the composition evidence set without changing retrieval itself
- **Deterministic routing** — policy rules, calculation rules, and refusal rules are code, not prompts
- **Post-generation verification** — citation support, numeric grounding, product attribution, qualifier/caveat adjacency, real names only, property presence in evidence, and product-scope correctness are checked after generation and before printing
- **Vision observations typed and vocabulary-gated** — substrate and symptom may be filled; location/exposure are not inferred from photos; observations carry confidence and are user-correctable
- **Fail-closed architecture** — weak or insufficient evidence triggers a fail-closed response rather than weakening verification

See [Reliability and design evolution](docs/RELIABILITY_AND_DESIGN_EVOLUTION.md) for how these were validated through measured failures.

---

## Local runtime

The local build intentionally uses SQLite, local embeddings, Ollama and in-memory conversation state. The corpus is small, so the local runtime prioritises a self-contained system without requiring external databases or hosted AI services.

**What is local:**
- Knowledge store: SQLite
- Vector similarity: local numpy-based retrieval
- Embedding model: `qwen3-embedding:0.6b` via Ollama (local CPU)
- Generation model: `qwen3.5:4b` via Ollama (local CPU)
- Conversation state: LangGraph in-memory (lost on restart)
- Vision: optional, runs locally if enabled (CPU-only, slow)
- Authentication: none (audience asserted, not verified)
- Tracing: hosted LangSmith disabled in code

**Not required in local mode:**
- No PostgreSQL requirement
- No pgvector requirement
- No rate limiting or request queue
- No durable checkpointing
- No message broker

Infrastructure backends are isolated from the answer-safety contract: changing persistence or serving infrastructure should not change evidence-sufficiency, citation, product-scope or fail-closed semantics.

---

## Production direction

These capabilities represent deployment-oriented seams rather than requirements of the local runtime:

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

Opens http://127.0.0.1:8765 in your browser.

### Start demo (manual)

```bash
pip install -r requirements.txt
python -m assistant.interfaces.ui
```

### Enable vision (optional, slow on CPU)

```powershell
.\scripts\start-demo.ps1 -VisionDemo
```

### Evaluate

```bash
python -m eval.run              # Structured conversation evaluation
python -m assistant.interfaces.cli -q "..."  # Single-turn CLI
python -m assistant.infrastructure.health      # Readiness check
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
- **[Demo script](docs/DEMO-SCRIPT.md)** — Walkthrough sequence with test questions and expected behaviour
- **[Knowledge pipeline](docs/knowledge-pipeline.md)** — Indexing, versioning, publication, delta updates

---

## Known limitations

- **Local CPU inference latency** — 20–40+ seconds per uncached question; Ollama prompt cache significantly speeds repeats
- **Conversation state not durable** — in-memory only; lost on process restart
- **Optional vision is slow** — local CPU inference can take 150+ seconds per image
- **Single Ollama instance** — generation serialised; no parallelism on local machine
- **Production capabilities not implemented** — no authentication, rate limiting, durable checkpointing or scalable inference

See [Reliability and design evolution](docs/RELIABILITY_AND_DESIGN_EVOLUTION.md) and [Architecture decisions](DECISIONS.md) for detailed trade-offs and design rationale.

---

## Design philosophy

The system does not treat prompt engineering as the primary safety mechanism. Instead:

- **Evidence is retrieved**, not generated
- **Policy is deterministic code**, not prompt instructions
- **Routing is explicit and ordered**, with clear precedence
- **Verification happens after generation**, not inside the prompt
- **Fail-closed** means returning a safe limitation or refusal rather than weakening verification

The model is not treated as the source of truth. Language generation is allowed flexibility; product identity, evidence sufficiency, numeric support, citations and refusal boundaries remain governed by structured state and deterministic verification.
