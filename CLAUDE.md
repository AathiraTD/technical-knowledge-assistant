# CLAUDE.md — Technical Knowledge Assistant

@docs/architecture.md
@DECISIONS.md

## Purpose

This repository implements the Lime Green Technical Knowledge Assistant. Treat it as a production-credible system, not a throwaway RAG demo.

The architecture and decision record are authoritative for intended behaviour. The repository is authoritative for current implementation. If they disagree, identify the discrepancy and determine whether code is incomplete or documentation is stale; do not silently change either side to make them appear consistent.

Preserve working behaviour and evolve incrementally. Do not restart the project from scratch.

## Core product principles

1. Answer only from approved retrieved evidence.
2. Unsupported information must not print.
3. Deterministic code owns routing, permissions, version filtering, policy gates, calculations, citation validation and safety checks.
4. The LLM is not the controller.
5. Failure should reduce coverage, not safety.
6. Refusal or hand-off is preferable to unsupported inference.
7. Every factual answer must remain traceable to source evidence.
8. Historical document versions must remain auditable.
9. Audience isolation happens before generation.
10. Production dependencies must be free, self-hostable and commercially deployable unless explicitly approved otherwise.
11. Production claims require executable evidence.
12. Documentation must describe reality, not aspiration.

## Architecture boundaries

### Knowledge repository

The application core depends on a database-agnostic `KnowledgeRepository` boundary.

Required adapters:

- `SQLiteKnowledgeRepository` for the embedded/offline assessment path.
- `PostgresKnowledgeRepository` for production/deployment.

The answer engine must not directly depend on `sqlite3`, `psycopg`, pgvector SQL, or other database-specific infrastructure. Infrastructure implements the repository boundary; domain/application code does not import concrete storage adapters except at composition/bootstrap.

The two adapters represent one application and must share equivalent domain semantics.

### Storage

Assessment path:

- SQLite
- versioned local source filesystem
- local vector representation where appropriate
- local Ollama models

Production path:

- PostgreSQL
- pgvector
- versioned source filesystem
- Ollama

Raw HTML, PDFs and original source files remain on a versioned filesystem. PostgreSQL stores identity, source paths, hashes, metadata, versions, chunks, embeddings and provenance. Do not store whole PDFs/HTML as database BLOBs without a demonstrated need.

The logical schema includes, at minimum:

- `documents`
- `document_versions`
- `chunks`
- `document_caveats`
- `excluded_documents`
- `crawl_runs`
- `index_snapshots`

Use primary keys, foreign keys, uniqueness constraints, `NOT NULL`, check constraints and indexes where they protect real invariants or access paths.

Critical invariant: a document may have exactly one active version. Enforce this in the database. In PostgreSQL, use an appropriate partial unique index. New-version activation and old-version deactivation must be transactional.

### Delta ingestion and recency

Preserve the existing delta/version model and semantics, including concepts equivalent to:

- `version`
- `content_hash`
- `etag`
- `last_modified`
- `first_seen_at`
- `fetched_at`
- `checked_at`
- `is_active`
- `supersedes`

Understand what bytes/content are hashed before changing hash behaviour.

A crawl must distinguish new, changed, unchanged, removed and failed documents.

Unchanged documents must not be needlessly re-extracted, re-chunked or re-embedded.

Changed documents preserve the previous version, create a new version, reprocess only what is necessary, and publish atomically. A failed crawl/indexing attempt must never corrupt the currently serving snapshot.

Retrieval must never blend active and superseded versions.

### Snapshots and reproducibility

An index snapshot must capture enough state to explain what the system knew at the time of an answer, including the embedding model/configuration, chunking version and active document versions.

Internal answer diagnostics should make it possible to identify the snapshot, retrieved chunk IDs, document versions, route taken and model/configuration used.

The engine must refuse to query an index built with an incompatible embedding model/configuration.

### Retrieval

Production retrieval uses PostgreSQL + pgvector as real executable functionality, not as a documentation-only target.

Where practical, filtering and similarity search should be combined in the database query. Retrieval must respect:

- active version only
- audience
- authority
- semantic similarity
- per-document result caps
- any approved product/category constraints

Never retrieve restricted staff evidence and then rely on the LLM to hide it.

Preserve the documented authority policy. Newer content does not automatically outrank a more authoritative source; recency is a tiebreaker within equivalent authority/source classes.

Do not cargo-cult ANN/HNSW at this corpus size. Benchmark exact pgvector search first. Add an ANN index only when measurement justifies it.

### Product eligibility

Do not invent a product/substrate compatibility matrix from model knowledge.

If approved structured compatibility data exists, a deterministic eligibility stage may restrict candidate products before retrieval. If the required matrix does not exist, preserve citation-based safety, keep a clean extension boundary and document the dependency on technical-team knowledge capture.

### Answer engine

Routing is deterministic code. Preserve documented router precedence.

Conceptual output paths remain:

- route
- extract
- compose
- cited hand-off
- refusal

The model runs only where composition is required. Do not move permissions, policy routing, arithmetic, version logic or validation into prompts.

Prompts are not security boundaries.

Preserve and test the documented post-generation checks, including:

1. generated factual sentences require citations;
2. numeric values must exist in cited evidence;
3. numbers stay attached to the correct product;
4. qualifiers/caveats remain attached;
5. named products/documents/merchants must be approved real names;
6. the requested property/substrate or synonym must appear in supporting evidence.

Any failed safety check fails closed into the documented refusal/hand-off behaviour.

### Audience security

`public`, `trade` and `staff` access must be enforced in code/database retrieval, not prompts.

Production identity can determine the audience set later. The assessment path may use an asserted audience flag.

Maintain a test proving public retrieval cannot see staff-tagged evidence.

### Multimodal

Do not destabilise the text/retrieval foundation to rush image support.

Intended architecture:

`image -> vision perception -> structured observations -> profile resolver -> evidence/confidence gate -> existing answer pipeline`

The VLM must never directly choose a product.

Structured observations should use explicit contracts containing concepts such as attribute, value/status, confidence, source image, region and `cannot_determine`.

Treat VLM confidence as a routing signal unless empirically calibrated.

Diagnosis remains subject to the documented human hand-off policy.

A future failure library should store the image, model prediction, expert-corrected label, error type and visibility/quality metadata. Evaluation/calibration comes before fine-tuning. Never use the model's own answer as ground truth.

## Embedding/model decisions

Do not prematurely hard-code a final embedding model because it is currently favoured.

Select the embedding model by the documented benchmark using the same corpus/questions and compare:

- known-answer retrieval quality
- near-miss/far-miss behaviour
- index build time
- query latency
- memory/runtime constraints
- embedding dimension
- actual chunk-size/input-window compatibility

A temporary candidate must be labelled `development default`, not `final selected model`.

Record exact model tags/configuration in snapshots.

## Docker and deployment

Maintain a simple production-like Docker Compose stack using free/self-hostable components. Expected services are the application, PostgreSQL + pgvector and Ollama; add other services only when justified.

Requirements:

- persistent volumes
- externalised configuration
- `.env.example`
- no committed secrets
- meaningful health/readiness checks
- deterministic startup
- clean shutdown
- explicit network exposure
- controlled/pinned images where practical
- non-root application container where practical

Inside Compose, use service DNS names such as `db:5432` and `ollama:11434`, not `localhost`.

Do not add Kubernetes or unnecessary microservices.

Liveness and readiness are different. Readiness should consider database connectivity, schema state, repository usability, model availability and snapshot compatibility where applicable.

## Git and data safety

Before substantial work inspect:

- `git status`
- current branch
- recent commits

Never overwrite unrelated user changes.

Do not push to remote Git, force-push, rewrite remote history, change repository visibility, delete remote branches or publish artifacts unless the user explicitly authorises that action in the current session.

A request to "continue development" is not permission to push.

Do not assume publicly downloadable Lime Green documents may be redistributed through a public repository.

The source-code repository should normally contain code, schemas, tests, synthetic fixtures, configuration, diagrams and documentation. Crawled production/assessment corpus files require an explicit distribution decision.

If crawled PDFs/HTML/cache/embeddings/databases are tracked publicly, report the exposure. Do not perform destructive history rewriting without explicit approval.

## Dependency policy

Prefer existing dependencies and standard-library functionality when they are adequate.

Significant new dependencies must be free/self-hostable, commercially deployable and justified by a concrete need.

Do not introduce LangChain, LlamaIndex or an agent framework into the product runtime merely because development is being done by agents. The runtime is intentionally inspectable and deterministic.

## Security

At external boundaries, deliberately review and test plausible risks including:

- SQL injection
- SSRF/arbitrary URL fetching
- path traversal
- unsafe uploaded filenames
- malformed PDFs/HTML
- prompt injection from indexed content
- oversized input / denial-of-service paths
- audience escalation/leakage
- secret exposure
- unsafe deserialisation

Use parameterised SQL. Validate strongly at system boundaries. Do not silently swallow failures.

Avoid logging secrets, credentials, unnecessary personal data, complete customer conversations, or sensitive internal material.

## Observability and operations

Use useful structured logs for significant events such as crawl start/complete, document new/changed/unchanged, extraction failures, snapshot publication, retrieval, route selection, generation timing, safety-check failure, refusal, DB errors and Ollama errors.

Use request/correlation IDs where they help trace a single answer.

Measure before optimising. Track relevant timings such as crawl/index duration, retrieval latency and generation latency.

Do not claim serving features such as queueing, rate limiting, caching or graceful degradation are built unless they are actually implemented and tested.

## Testing and quality gates

Testing is a production requirement, not a final polish step.

A feature is not complete because it worked once. Its intended behaviour, invariants and failure paths must be demonstrated by automated tests.

### Safety-critical coverage

Target 100% branch coverage for safety-critical deterministic logic, especially:

- router and precedence
- policy/refusal/handoff paths
- relevance gates
- audience filtering
- active-version filtering
- version supersession/activation
- snapshot publication
- embedding/index compatibility
- all post-generation checks
- citation/numeric/qualifier/attribution checks
- load-bearing-slot handling
- calculation restrictions

Do not create meaningless tests solely to inflate coverage.

### Core coverage

For core application/domain/retrieval/indexing modules, target:

- >= 90% line coverage
- >= 90% branch coverage

If below target, report exactly which paths remain uncovered and why.

### Repository contract

Maintain one behavioural `KnowledgeRepository` contract suite and run it against both SQLite and a real PostgreSQL + pgvector instance where environment support exists.

The contract should cover at least:

- create/read document
- version creation
- one-active-version invariant
- supersession
- transactional activation
- active-version-only retrieval
- audience filtering
- authority/similarity ordering
- per-document cap
- provenance
- crawl-run persistence
- snapshot publication
- model/configuration compatibility
- rollback/error behaviour

Postgres is not production-ready merely because its schema or class exists.

### Integration / E2E

Prefer real boundary tests over mocks when practical. Important integration scenarios include SQLite, PostgreSQL/pgvector, indexer/repository, retrieval/embeddings, engine/repository and Docker readiness/persistence.

Keep deterministic CI independent of live websites, paid services and multi-GB model downloads. Real Ollama/model smoke tests may be separate opt-in/local checks.

Maintain end-to-end scenarios for exact lookup, synonym gaps, multi-document synthesis, near/far misses, missing load-bearing slots, calculations, published deferrals, superseded versions, similarly named products, public-vs-staff isolation, malformed source, unavailable model/database, failed safety check, changed crawl, unchanged crawl and rollback after indexing failure.

Use property-based or mutation testing where it adds real value, especially around router/safety/version/audience invariants.

Every discovered defect should receive a regression test where practical before the fix.

## CI

CI should fail on applicable:

- unit/integration test failure
- repository-contract failure
- safety regression
- lint/format/type-check failure
- schema/migration failure
- production container build failure
- agreed core coverage threshold failure

Report line and branch coverage separately. Do not exclude difficult modules merely to improve the percentage.

## Code quality

Prefer small cohesive functions, explicit domain types, narrow interfaces, meaningful names, deterministic logic and dependency injection at infrastructure boundaries.

Avoid god objects, speculative abstraction, deep wrapper layers, hidden global state and "enterprise architecture theatre".

Production quality means understandable, testable and operationally credible, not maximum abstraction.

## Multi-agent development

For substantial cross-cutting work, use multiple specialised subagents rather than one monolithic context.

A strong default is parallel read-only investigation for independent concerns such as:

- repository/current-state audit
- data/database architecture
- retrieval/RAG
- QA/testing
- DevOps/security
- adversarial review

The primary agent is always integration owner.

Before parallel implementation, assign explicit file/module ownership. Do not allow two agents to edit overlapping files concurrently.

After each substantial vertical slice, use independent read-only reviewers for:

1. architecture drift;
2. adversarial/security failure modes;
3. tests/operations.

Evaluate reviewer findings rather than blindly applying them.

Do not use subagents for trivial reads/grep operations where delegation costs more than it saves.

## Engineering workflow

For substantial tasks:

1. establish baseline behaviour and tests;
2. inspect relevant code before making claims;
3. identify architecture/code discrepancies;
4. plan in dependency order;
5. implement vertical slices;
6. add/update tests;
7. run narrow tests;
8. run subsystem/integration tests;
9. run full relevant tests;
10. inspect the diff;
11. run architecture/adversarial/operations review;
12. update documentation only after implementation evidence exists.

Do not stop after producing a plan when the user asked for implementation.

Only ask for user input when blocked by destructive remote Git actions, deletion of real data, repository visibility changes, credentials/secrets, unresolved materially different business rules, or licensing/ownership decisions requiring approval.

## Status honesty

Use precise implementation status:

- `built and verified`
- `built but weakly tested`
- `partial`
- `documented/roadmap only`
- `known limitation`

Never turn an architectural intention into an implementation claim.

Production PostgreSQL + pgvector is not "done" until it boots, schema setup works, repository contract tests pass and real vector retrieval succeeds.

## Production-foundation definition of done

Do not call the production foundation complete until applicable evidence shows:

- application core is storage-agnostic;
- SQLite adapter passes the repository contract;
- PostgreSQL adapter passes the same contract;
- pgvector retrieval works;
- audience filtering occurs inside retrieval;
- inactive versions cannot enter retrieval;
- the DB enforces one active version;
- version activation is transactional;
- unchanged documents avoid redundant work;
- changed documents create auditable new versions;
- source provenance is preserved;
- snapshots bind model/configuration;
- incompatible indexes are rejected;
- Docker Compose boots the production-like stack;
- DB state survives restart;
- health/readiness behave meaningfully;
- configuration/secrets are externalised;
- safety-critical coverage meets its target;
- core coverage meets target or documented exceptions exist;
- integration/evaluation tests show no safety regression;
- verified setup/test/run commands are documented.

Evidence over confidence.
