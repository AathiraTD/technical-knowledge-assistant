# Knowledge pipeline completion

## Contract

The supplied decks define a controlled knowledge release: discover changes,
preserve source evidence, extract and chunk only what needs processing, embed
locally, validate, publish atomically, and serve one consistent snapshot.

Approved scope: complete that lifecycle and its PostgreSQL integration. The
existing extraction edits are preserved. Answer-generation policy and vision
remain separate work. Approved staff documents use an explicit local import;
this does not invent or approve technical content.

## Acceptance criteria and order

1. Sources: explicit offline/revalidation modes; conditional HTTP requests;
   content-addressed original files; crawl history; failed discovery cannot
   withdraw live evidence. Retry transient failures with bounded backoff.
2. Indexing: validate source hashes, extraction and vectors before publication;
   distinguish failed from removed; preserve history even on rebuild; rebuild
   automatically when model/chunking configuration changes.
3. Storage and readers: transactional delta activation; auditable snapshot
   membership; unique release IDs; persist crawl outcomes; consistent reads
   across publication and refresh between answers.
4. Integration: one repository factory for indexer, CLI, UI and readiness;
   actual PostgreSQL/pgvector contract and lifecycle tests; approved staff
   import and a repeatable scheduled pipeline command.
5. Verification: 100% line and branch coverage for knowledge-pipeline modules
   (crawl, extract, index, embedding client/cache, model, repository, storage,
   and new lifecycle/bootstrap modules). Do not omit crawler/PostgreSQL or add
   coverage exclusions to meet the target. Keep full existing suite green.
   Note that the enforced configuration is wider than this criterion: `.coveragerc`
   sets `source = assistant` and omits only `assistant/__init__.py`, so the CLI
   and the web page are measured to the same target rather than excused as
   presentation wrappers.
6. Document final architecture, decisions, commands, evidence and limitations.

## Completion

Implemented and verified on 16 September 2026. See
[the requirements mapping and runbook](knowledge-pipeline.md) and
[the measured verification evidence](knowledge-pipeline-verification.md) — which
carries a dating caveat worth reading first: the package was restructured on
18 September 2026 and those figures have not been re-measured against the
current tree.

## Verification commands

`python -m pytest -q tests/`

`python -m coverage run --rcfile=.coveragerc -m pytest -q tests/`

`python -m coverage report --rcfile=.coveragerc`

PostgreSQL tests require an isolated local test database via
`ASSISTANT_POSTGRES_DSN`; configured database failures must fail tests, not skip.
No tests fetch the live partner website or mutate the shipped corpus.
