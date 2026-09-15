"""Readiness check, for a container orchestrator and for a person.

`python -m assistant.health` exits 0 when the system could actually answer a
question, and non-zero with a reason when it could not.

Liveness and readiness are different things and conflating them is how a
service stays in rotation while returning nothing useful. A process that starts
is alive. This system is *ready* only when the store has an active snapshot,
that snapshot's embedding model matches the one configured, the model is pulled,
and there are embedded chunks to search. Each of those can be false while the
process runs perfectly well.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import ollama, use_utf8


def check(db: str = "", dsn: str = "") -> dict:
    report: dict = {"ready": False, "checks": {}}

    dsn = dsn or os.environ.get("ASSISTANT_POSTGRES_DSN", "")
    try:
        if dsn:
            from .store.postgres import PostgresKnowledgeRepository
            repo = PostgresKnowledgeRepository(dsn, apply_schema=False)
            report["store"] = "postgresql+pgvector"
        else:
            from .store import SQLiteKnowledgeRepository
            repo = SQLiteKnowledgeRepository(db or "data/index/knowledge.db")
            report["store"] = "sqlite"
        report["checks"]["store_reachable"] = True
    except Exception as exc:
        report["checks"]["store_reachable"] = False
        report["error"] = f"store unreachable: {exc}"
        return report

    try:
        snapshot = repo.snapshot()
        report["checks"]["active_snapshot"] = snapshot is not None
        if snapshot is None:
            report["error"] = ("no active index snapshot — run "
                               "python -m assistant.index")
            return report
        report["snapshot"] = snapshot.snapshot_id
        report["documents"] = snapshot.document_count
        report["chunks"] = snapshot.chunk_count
        report["embedding_model"] = snapshot.embedding_model

        report["checks"]["chunks_present"] = snapshot.chunk_count > 0

        try:
            have = ollama.available()
            report["checks"]["ollama_reachable"] = True
            bare = {m.split(":")[0] for m in have}
            for label, tag in (("embedding_model_pulled", snapshot.embedding_model),
                               ("generation_model_pulled", ollama.GENERATION_MODEL)):
                report["checks"][label] = tag in have or tag.split(":")[0] in bare
        except ollama.OllamaUnavailable as exc:
            report["checks"]["ollama_reachable"] = False
            report["error"] = str(exc)
            return report

        report["checks"]["embedding_model_matches_index"] = (
            snapshot.embedding_model == ollama.EMBED_MODEL)
        if not report["checks"]["embedding_model_matches_index"]:
            report["error"] = (
                f"index built with {snapshot.embedding_model!r} but the engine is "
                f"configured for {ollama.EMBED_MODEL!r}; rebuild the index")
    finally:
        if hasattr(repo, "close"):
            repo.close()

    report["ready"] = all(report["checks"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(prog="assistant.health")
    parser.add_argument("--db", default="data/index/knowledge.db")
    parser.add_argument("--dsn", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = check(args.db, args.dsn)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print("ready" if report["ready"] else "NOT READY")
        for name, ok in report["checks"].items():
            print(f"  {'ok ' if ok else 'no '} {name.replace('_', ' ')}")
        if report.get("error"):
            print(f"\n  {report['error']}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
