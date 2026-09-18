"""Readiness check, for a container orchestrator and for a person.

`python -m assistant.infrastructure.health` exits 0 when the system could actually answer a
question, and non-zero with a reason when it could not.

Liveness and readiness are different things and conflating them is how a
service stays in rotation while returning nothing useful. A process that starts
is alive. This system is *ready* only when the store has an active snapshot,
that snapshot's embedding model matches the one configured, the model is pulled,
and there are embedded chunks to search. Each of those can be false while the
process runs perfectly well.

Three things this module deliberately will not do.

**It does not print a connection string.** A PostgreSQL DSN carries a password,
and a readiness report is the most-copied output in an incident -- into a
ticket, a chat window, a screenshot. `_redact()` removes the credentials and
keeps the host and database, which is the part an operator actually needs.

**Vision is reported but is not part of readiness by default.** The image path
is roadmap (decision 16), so a machine with no vision model is not a machine
that cannot answer questions. That check becomes load-bearing only when
`ASSISTANT_VISION_DEMO` is set, which is the one case where a missing model
would break something the operator is about to demonstrate.

**It does not retry on its own.** `--wait` polls, because a startup script has
to wait for Ollama to finish loading a model and a readiness probe must not:
a probe that waits reports healthy slowly instead of reporting unhealthy. The
two callers want opposite things, so waiting is a flag rather than a default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from . import ollama
from .. import use_utf8
from ..knowledge.store.factory import open_repository
from ..indexing.index import CHUNKING_VERSION

# Credentials in a DSN, and nothing else. Deliberately narrow: the host, port
# and database name are what make the message useful, and reducing the whole
# string to "postgresql" would trade a leak for an unusable error.
_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+:[^/@\s]+@")

# Set this when the demonstration includes an image upload. Off by default
# because vision is roadmap: see the module docstring.
VISION_DEMO_VAR = "ASSISTANT_VISION_DEMO"


def _redact(text: str) -> str:
    """A DSN with its password removed, safe to print and to paste."""
    return _CREDENTIALS.sub("***@", text)


def vision_demo_enabled() -> bool:
    """Whether a missing vision model should count against readiness.

    Off unless the operator says otherwise, and the several spellings of yes
    are accepted because the alternative — a capability flag that silently
    means "no" when someone writes `true` instead of `1` — fails in the
    direction of a demonstration discovering at the wrong moment that its
    model was never checked.
    """
    return os.environ.get(VISION_DEMO_VAR, "").strip().lower() in (
        "1", "true", "yes", "on")


def _pulled(tag: str, have: list[str]) -> bool:
    """Whether Ollama holds this model, tolerating an omitted `:latest`."""
    return tag in have or tag.split(":")[0] in {m.split(":")[0] for m in have}


def check(db: str = "", dsn: str = "") -> dict:
    """Everything that has to be true before this process can answer anything.

    Readiness is computed, never assumed: `report["ready"]` is the conjunction
    of `report["checks"]`, so a check added to that mapping is load-bearing the
    moment it is written and there is no separate verdict to forget to update.

    The order is a dependency chain, and each stage returns early rather than
    reporting a cascade of consequences. A store that cannot be opened makes
    every later question meaningless, so the report stops there with one fault
    named instead of six. The same applies to a missing snapshot and to an
    unreachable Ollama: an operator reading a probe at three in the morning
    needs the first true statement, not the transitive closure of it.

    What the chain actually asserts, in the order the code runs it:

    * the store opens — PostgreSQL when a DSN is configured, SQLite otherwise,
      selected by the same factory every entry point uses so the probe cannot
      be inspecting a different backend from the one serving;
    * an index snapshot is published and active;
    * that snapshot holds passages at all, and was chunked by the version this
      build of `assistant/indexing/index.py` implements;
    * the vector width matches what the engine is configured to produce;
    * Ollama answers, and holds both the embedding and the generation model;
    * **the snapshot's embedding model is the one the engine is configured to
      use** — the check this whole module exists for, because an index built
      by one model and queried by another returns confident nonsense rather
      than an error, and nothing downstream would notice.

    `apply_schema=False` on the PostgreSQL branch is deliberate: a readiness
    probe runs constantly and must observe the system, never migrate it. The
    store is closed in a `finally`, because a probe called every fifteen
    seconds that leaked a connection would exhaust the server it is meant to be
    watching over.

    Vision is reported and, unless `ASSISTANT_VISION_DEMO` is set, excluded
    from the verdict — see the module docstring for why a roadmap stage must
    not be able to declare a working text system unready.
    """
    report: dict = {"ready": False, "checks": {}}

    dsn = dsn or os.environ.get("ASSISTANT_POSTGRES_DSN", "")
    try:
        if dsn:
            repo = open_repository(dsn=dsn, apply_schema=False)
            report["store"] = "postgresql+pgvector"
            report["store_target"] = _redact(dsn)
        else:
            path = db or "data/index/knowledge.db"
            repo = open_repository(path, dsn="")
            report["store"] = "sqlite"
            report["store_target"] = path
        report["checks"]["store_reachable"] = True
    except Exception as exc:
        report["checks"]["store_reachable"] = False
        report["error"] = f"store unreachable: {_redact(str(exc))}"
        return report

    try:
        snapshot = repo.snapshot()
        report["checks"]["active_snapshot"] = snapshot is not None
        if snapshot is None:
            report["error"] = ("no active index snapshot — run "
                               "python -m assistant.indexing.index")
            return report
        report["snapshot"] = snapshot.snapshot_id
        report["documents"] = snapshot.document_count
        report["chunks"] = snapshot.chunk_count
        report["embedding_model"] = snapshot.embedding_model
        report["embedding_dimensions"] = snapshot.embedding_dimensions
        report["chunking_version"] = snapshot.chunking_version
        report["generation_model"] = ollama.GENERATION_MODEL
        report["ollama_host"] = ollama.HOST

        report["checks"]["chunks_present"] = snapshot.chunk_count > 0
        report["checks"]["chunking_matches_index"] = snapshot.chunking_version == CHUNKING_VERSION
        report["checks"]["dimensions_match_index"] = snapshot.embedding_dimensions == ollama.EMBED_DIMENSIONS

        try:
            have = ollama.available()
            report["checks"]["ollama_reachable"] = True
            for label, tag in (("embedding_model_pulled", snapshot.embedding_model),
                               ("generation_model_pulled", ollama.GENERATION_MODEL)):
                report["checks"][label] = _pulled(tag, have)
        except ollama.OllamaUnavailable as exc:
            report["checks"]["ollama_reachable"] = False
            report["error"] = str(exc)
            return report

        # Reported either way; load-bearing only for an image demonstration.
        #
        # Imported here rather than at module scope because `assistant.answering.vision`
        # pulls in the router and its vocabularies, and a readiness probe that
        # runs every fifteen seconds should not pay for them.
        #
        # Guarded because the text path does not need this module at all. A
        # readiness check that raised on a broken import would report the whole
        # system unable to answer questions on account of a roadmap feature it
        # does not use -- turning an unused stage into an outage, which is the
        # opposite of what a probe is for.
        demo = vision_demo_enabled()
        try:
            from ..answering.vision import VISION_MODEL
        except Exception as exc:                        # pragma: no cover
            report["vision"] = {"model": "", "pulled": False, "required": demo,
                                "error": str(exc)}
        else:
            report["vision"] = {"model": VISION_MODEL,
                                "pulled": _pulled(VISION_MODEL, have),
                                "required": demo}
        if demo:
            report["checks"]["vision_model_pulled"] = report["vision"]["pulled"]

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


# --------------------------------------------------------------- the report

# What to do about each check that can be false. A readiness report saying
# "no generation_model_pulled" has named the fault and left the operator to
# work out the command; five minutes before a demonstration, that is the wrong
# half of the job to leave undone.
REMEDIES = {
    "store_reachable":
        "check the --db path exists, or that ASSISTANT_POSTGRES_DSN points at a "
        "running database",
    "active_snapshot":
        "build the index: python -m assistant.indexing.index",
    "chunks_present":
        "the snapshot published no passages; rebuild: python -m assistant.indexing.index --rebuild",
    "chunking_matches_index":
        "the chunker changed since this index was built; rebuild: python -m assistant.indexing.index",
    "dimensions_match_index":
        "EMBED_DIMENSIONS does not match the index; unset it, or rebuild the index",
    "ollama_reachable":
        "start the model server: ollama serve",
    "embedding_model_pulled":
        f"ollama pull {ollama.EMBED_MODEL}",
    "generation_model_pulled":
        f"ollama pull {ollama.GENERATION_MODEL}",
    "vision_model_pulled":
        "pull the vision model, or unset ASSISTANT_VISION_DEMO to demonstrate "
        "the text path only",
    "embedding_model_matches_index":
        "rebuild the index with the configured model: python -m assistant.indexing.index --rebuild",
}


def _rows(report: dict) -> list[tuple[str, str, str]]:
    """Label, state and detail for each line of the readiness table."""
    checks = report["checks"]
    rows: list[tuple[str, str, str]] = [("application", "ready", "process is serving")]

    rows.append(("database",
                 "ready" if checks.get("store_reachable") else "UNREACHABLE",
                 f"{report.get('store', 'unknown')} · "
                 f"{report.get('store_target', '')}".strip(" ·")))

    if "active_snapshot" not in checks:
        return rows
    if not checks["active_snapshot"]:
        rows.append(("knowledge index", "NOT BUILT", REMEDIES["active_snapshot"]))
        rows.append(("active snapshot", "none", "nothing published"))
        return rows

    rows.append(("knowledge index",
                 "ready" if checks.get("chunks_present") else "EMPTY",
                 f"{report.get('documents', 0)} documents · "
                 f"{report.get('chunks', 0)} passages"))
    rows.append(("active snapshot", report.get("snapshot", "none"),
                 f"chunking {report.get('chunking_version', '?')}"
                 + ("" if checks.get("chunking_matches_index") else
                    f" (engine expects {CHUNKING_VERSION})")))
    rows.append(("embedding model", report.get("embedding_model", "?"),
                 ("matches index" if checks.get("embedding_model_matches_index")
                  else f"MISMATCH — engine configured for {ollama.EMBED_MODEL}")
                 + f" · {report.get('embedding_dimensions', '?')} dimensions"))
    rows.append(("generation model", report.get("generation_model", "?"),
                 "pulled" if checks.get("generation_model_pulled") else "NOT PULLED"))

    vision = report.get("vision")
    if vision:
        rows.append(("vision model", vision["model"],
                     ("pulled" if vision["pulled"] else "NOT PULLED")
                     + ("" if vision["required"] else " · image demo off")))

    rows.append(("Ollama",
                 "reachable" if checks.get("ollama_reachable") else "UNREACHABLE",
                 report.get("ollama_host", ollama.HOST)))
    return rows


def summary(report: dict) -> str:
    """The readiness table, and what to do about anything that is not ready."""
    rows = _rows(report)
    width = max(len(label) for label, _, _ in rows)
    state = max(len(value) for _, value, _ in rows)
    lines = ["READY" if report["ready"] else "NOT READY", ""]
    lines += [f"{label:<{width}}  {value:<{state}}  {detail}".rstrip()
              for label, value, detail in rows]

    broken = [name for name, good in report["checks"].items() if not good]
    if broken:
        lines += ["", "To fix:"]
        for name in broken:
            lines.append(f"  {name.replace('_', ' ')}")
            remedy = REMEDIES.get(name)
            if remedy:
                lines.append(f"      {remedy}")
    if report.get("error"):
        lines += ["", f"  {report['error']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """The same verdict for an orchestrator and for a person, one exit code.

    The exit status is the machine-readable part: 0 ready, 1 not. A container
    healthcheck and a startup script both act on that alone, so the output
    format is free to serve whoever is reading — `--json` for something
    parsing it, the table from `summary()` for someone standing in front of a
    demonstration five minutes before it starts, with the remedy printed under
    each failing line.

    **Liveness is not in this module at all.** It is one static line in
    `assistant/interfaces/ui.py` at `/health`, answering "is this process
    running" without touching the store. Everything here is readiness, and the
    distinction is the reason both exist: a process can start perfectly and be
    unable to answer a single question, and a probe that conflates the two
    keeps such a container in rotation.

    `--wait` is the one place the two callers genuinely want opposite
    behaviour. A startup script has to give Ollama time to finish loading a
    model; a readiness probe must never wait, because a probe that waits
    reports healthy slowly instead of reporting unhealthy. The loop runs at
    least once whatever the deadline, so `--wait 0` is a single check rather
    than no check.
    """
    use_utf8()
    parser = argparse.ArgumentParser(
        prog="assistant.infrastructure.health",
        description="Whether this process could actually answer a question.")
    parser.add_argument("--db", default="data/index/knowledge.db")
    parser.add_argument("--dsn", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--wait", type=float, default=0.0, metavar="SECONDS",
        help="poll until ready, or until this many seconds have passed. For a "
             "startup script waiting on Ollama; a readiness probe must not use it.")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="seconds between polls while --wait counts down")
    args = parser.parse_args(argv)

    deadline = time.monotonic() + args.wait
    while True:
        report = check(args.db, args.dsn)
        if report["ready"] or time.monotonic() >= deadline:
            break
        time.sleep(args.interval)

    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(summary(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
