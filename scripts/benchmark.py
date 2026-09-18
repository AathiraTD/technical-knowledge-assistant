"""A modest concurrency measurement against a running page, and where the time goes.

The purpose is to find the bottleneck and to be able to say what it is, not to
claim a throughput number. Five concurrent questions on one laptop with one
Ollama is not a load test and nothing here should be quoted as one -- the
honest output is a breakdown, and the breakdown is the point: generation
dominates by so much that everything else is rounding error, and generation is
serialised per Ollama instance no matter how many threads ask.

    python scripts/benchmark.py --base http://127.0.0.1:8765
    python scripts/benchmark.py --levels 1,3,5 --json results.json

**Timings come from the spans, not from a stopwatch around the request.**
`assistant/infrastructure/observability.py` already records a span per stage and the store
already persists them, so retrieval and generation are read back through
`KnowledgeRepository.traces()` keyed on the correlation id the page returns in
`X-Correlation-Id`. Wrapping a timer around the HTTP call would measure the
same total and would be unable to split it, and a second timing model that
disagreed with the first is worse than no second timing model.

**Questions are distinct and asked once.** The answer cache is exact-key
(decision 14), so the same question twice measures the cache -- 0.017s against
40s -- and a benchmark that quietly did that would report a system forty times
faster than the one being demonstrated. `--warm` measures the cache
deliberately, as its own row, which is the useful version of that number.

**The policy fast path is measured separately.** A question matching the
routing table never reaches retrieval or a model, so it answers in
milliseconds. Averaging it together with a compose would produce a mean that
describes no question anyone asks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.knowledge.store.factory import open_repository       # noqa: E402

# Distinct on purpose: each is asked once per run, so none of them is served
# from the answer cache. They are also deliberately ordinary -- the questions a
# demonstration actually gets -- rather than chosen to be fast.
RETRIEVAL_QUESTIONS = [
    "How much water does Solo Onecoat need per bag?",
    "What is the coverage of Forte?",
    "Can I use Ultra on a solid brick wall?",
    "What temperature can Fine Stuff be applied at?",
    "How long should lime render cure before the next coat?",
    "What is the pot life of Duro?",
    "Which plaster suits a lath and plaster ceiling?",
    "What background preparation does lime rendering need?",
    "Is hydraulic lime different from hydrated lime?",
    "What thickness should a base coat be applied at?",
    "Does Warmshell need a mesh layer?",
]

# Every one of these matches the policy gate, so none reaches retrieval or a
# model: price, stock, delivery and where-to-buy route to a referral by code.
POLICY_QUESTIONS = [
    "How much does Solo cost?",
    "Where can I buy Lime Green products?",
    "Do you have Forte in stock?",
    "How long does delivery take?",
    "What is your warranty?",
]


def ask(base: str, question: str, timeout: float) -> dict:
    """One question over HTTP, returning its wall time and correlation id."""
    url = f"{base.rstrip('/')}/ask?" + urllib.parse.urlencode({"q": question})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            correlation = response.headers.get("X-Correlation-Id", "")
            status = response.status
    except urllib.error.HTTPError as exc:
        body, correlation, status = "", exc.headers.get("X-Correlation-Id", ""), exc.code
    except (urllib.error.URLError, OSError) as exc:
        return {"question": question, "ok": False, "error": str(exc),
                "seconds": time.perf_counter() - started, "trace_id": ""}

    elapsed = time.perf_counter() - started
    result = {"question": question, "ok": 200 <= status < 300, "status": status,
              "seconds": elapsed, "trace_id": correlation}
    if not result["ok"]:
        result["error"] = f"HTTP {status}"
    # A question splits by topic, so a reply carries a path per part. Joined
    # rather than reduced to the first, because "route+compose" is a real and
    # informative shape and reporting it as "route" would hide the model call
    # that dominated the measurement.
    try:
        paths = [p.get("path", "") for p in json.loads(body).get("parts", [])]
        result["path"] = "+".join(sorted({p for p in paths if p}))
    except (ValueError, AttributeError, TypeError):
        result["path"] = ""
    return result


def stage_times(repo, trace_id: str) -> dict:
    """Retrieval and generation, in milliseconds, read back from the spans.

    Returns what it found rather than zeroes for what it did not: a refusal
    never generates, and recording 0 ms of generation for it would drag a mean
    towards a number no answer took.
    """
    if not trace_id:
        return {}
    found: dict = {}
    for span in repo.traces(trace_id=trace_id, limit=200):
        if span.name in ("retrieval", "generation", "answer", "checks"):
            # A turn can retrieve more than once when a question splits into
            # parts, so accumulate rather than overwrite.
            found[span.name] = found.get(span.name, 0.0) + float(span.duration_ms)
    return found


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run_level(base: str, questions: list[str], concurrency: int,
              timeout: float) -> list[dict]:
    """`concurrency` questions in flight at once, one batch."""
    batch = questions[:concurrency]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda q: ask(base, q, timeout), batch))


def summarise(results: list[dict], repo) -> dict:
    """Wall times, stage breakdown and error rate for one level."""
    good = [r for r in results if r["ok"]]
    times = [r["seconds"] for r in good]

    stages: dict[str, list[float]] = {}
    for result in good:
        for name, ms in stage_times(repo, result["trace_id"]).items():
            stages.setdefault(name, []).append(ms / 1000.0)

    summary = {
        "requests": len(results),
        "errors": len(results) - len(good),
        "error_rate": round((len(results) - len(good)) / max(len(results), 1), 3),
        "seconds": {
            "min": round(min(times), 2) if times else None,
            "median": round(statistics.median(times), 2) if times else None,
            "p95": round(percentile(times, 0.95), 2) if times else None,
            "max": round(max(times), 2) if times else None,
        },
        "stages_seconds": {
            name: {"median": round(statistics.median(values), 2),
                   "max": round(max(values), 2), "samples": len(values)}
            for name, values in sorted(stages.items())
        },
        "paths": sorted({r.get("path", "") for r in good if r.get("path")}),
    }
    if any(not r["ok"] for r in results):
        summary["first_error"] = next(r.get("error", "") for r in results if not r["ok"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/benchmark.py",
        description="A modest concurrency measurement against a running page.")
    parser.add_argument("--base", default="http://127.0.0.1:8765",
                        help="the running server")
    parser.add_argument("--levels", default="1,3,5",
                        help="concurrency levels, comma separated")
    parser.add_argument("--db", default="data/index/knowledge.db",
                        help="the store to read spans back from")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-request ceiling; generation on a CPU is minutes")
    parser.add_argument("--policy-only", action="store_true",
                        help="only the fast path, for a quick check that the "
                             "server is answering")
    parser.add_argument("--warm", action="store_true",
                        help="repeat the level-1 question to measure the answer "
                             "cache, reported as its own row")
    parser.add_argument("--json", default="", metavar="PATH",
                        help="write the whole measurement to a file")
    args = parser.parse_args(argv)

    levels = [int(n) for n in args.levels.split(",") if n.strip()]
    # The sum, not the maximum: every level consumes its own unasked questions
    # so that no level is served one an earlier level already cached, which
    # means 1,3,5 needs nine of them rather than five.
    if sum(levels) > len(RETRIEVAL_QUESTIONS):
        print(f"Levels {args.levels} need {sum(levels)} distinct questions and "
              f"only {len(RETRIEVAL_QUESTIONS)} are defined; asking a repeat "
              f"would measure the answer cache instead of the system.",
              file=sys.stderr)
        return 2

    repo = open_repository(args.db, args.dsn, apply_schema=False)
    measurement: dict = {"base": args.base, "levels": {}, "policy": {}}

    try:
        print(f"Policy fast path — {len(POLICY_QUESTIONS)} questions, "
              f"no retrieval, no model")
        policy = run_level(args.base, POLICY_QUESTIONS, len(POLICY_QUESTIONS),
                           args.timeout)
        measurement["policy"] = summarise(policy, repo)
        seconds = measurement["policy"]["seconds"]
        print(f"  median {seconds['median']}s   max {seconds['max']}s   "
              f"errors {measurement['policy']['errors']}")

        if not args.policy_only:
            asked = 0
            for level in levels:
                # A fresh slice each level, so no level is served a question an
                # earlier level already cached.
                questions = RETRIEVAL_QUESTIONS[asked:asked + level]
                if len(questions) < level:
                    print(f"  (only {len(questions)} unasked questions left; "
                          f"skipping level {level})")
                    break
                asked += level
                print(f"\nConcurrency {level} — {level} distinct question(s), "
                      f"none cached")
                started = time.perf_counter()
                results = run_level(args.base, questions, level, args.timeout)
                wall = time.perf_counter() - started
                summary = summarise(results, repo)
                summary["wall_seconds"] = round(wall, 2)
                measurement["levels"][str(level)] = summary

                seconds = summary["seconds"]
                print(f"  wall {summary['wall_seconds']}s   "
                      f"median {seconds['median']}s   max {seconds['max']}s   "
                      f"errors {summary['errors']}")
                for name, stage in summary["stages_seconds"].items():
                    print(f"    {name:<12} median {stage['median']}s   "
                          f"max {stage['max']}s")

            if args.warm:
                print("\nAnswer cache — the level-1 question asked again")
                repeat = run_level(args.base, RETRIEVAL_QUESTIONS[:1], 1,
                                   args.timeout)
                measurement["cached"] = summarise(repeat, repo)
                print(f"  {measurement['cached']['seconds']['median']}s")
    finally:
        if hasattr(repo, "close"):
            repo.close()

    if args.json:
        Path(args.json).write_text(json.dumps(measurement, indent=1),
                                   encoding="utf-8")
        print(f"\nwritten to {args.json}")

    errors = (measurement["policy"].get("errors", 0)
              + sum(l.get("errors", 0) for l in measurement["levels"].values()))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
