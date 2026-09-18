"""Prometheus text, derived from the trace table. No new instrumentation.

`docs/conversation-observability-review.md` §2.7 asks for six things — answer,
refusal and hand-off rate; route distribution; the retrieval top-score
distribution; check failures by check number; cache hit rate; and latency
percentiles per span name — and says of all six that they are *derived*: "every
metric in the operational list is a query over `turn_traces` and `answer_log`.
Nothing that requires a counter the code does not already produce."

This module holds to that literally. It adds no counter, no call site and no
field. It reads spans back through `KnowledgeRepository.traces()` and counts
them. If a metric cannot be computed from what the answering path already
records, it is not exposed — one such omission is named under *What is not
here*, below, rather than fixed by adding a counter, because a counter added to
satisfy a dashboard is instrumentation that exists for the metric instead of a
metric that exists for the instrumentation.

---

## Why this is its own module and not a method on the page

`assistant/interfaces/ui.py` is scheduled to be rewritten wholesale by a later slice. A
`/metrics` renderer living inside it would be rewritten with it, or — worse —
survive as the one part of the old page nobody dared touch. The page's whole
involvement is therefore two lines in `do_GET` that call `render()` and hand
back its bytes, which the next author can carry forward without reading a word
of this file.

## Why the format is hand-written

Decision 4 rests on five inspectable dependencies, and `prometheus_client` would
be a sixth for string formatting. The exposition format is `# HELP`, `# TYPE`,
and `name{label="value"} number` — the whole of what is needed is below, in
about forty lines including the escaping rules. What a client library would
actually buy is process-wide counter *registries*, which is precisely what this
design does not want: the numbers here come from the database, so they survive a
restart, they are the same on every worker, and they cannot drift from the
trace an operator would open to check one.

## Why everything is a gauge, including the counts

The honest shape, and the one most likely to be questioned. Prometheus counters
must be monotonic — `rate()` and `increase()` are defined in terms of resets —
and these numbers are not: they are counted over **the most recent N spans**,
so they rise, fall, and jump when retention prunes. Declaring them counters
would make `rate(assistant_answers_total[5m])` return a plausible number that
means nothing. Declared gauges, the same expression is obviously wrong and the
right one — comparing the window against itself — is available.

The cost is that this endpoint reports a *window*, not a lifetime. It is stated
in every `# HELP` line and in `assistant_window_spans`, which says how many rows
the window actually held, and `assistant_window_truncated`, which says whether
the cap bound. An operator reading 40 refusals with `truncated 1` knows to look
at the trace table rather than at this page.

## Why `source` is a label and not a filter

This is the point of the module, and it has a history. Commit e5bd236 added
`source` to `answer_log` and to `turn_traces` because the evaluation harness
answers through the same `Assistant` a person does, and its question set is
deliberately loaded with the near-miss and far-miss probes that are *supposed*
to refuse. Counted undifferentiated, the better the guardrail worked the worse
the refusal rate looked. Review §8.2 puts the consequence bluntly: "a `source`
column nobody filters on is worse than no column, because it looks like the
problem was solved."

Two ways to honour that. **Exclude** `evaluation` rows outright, which is safe
and throws away something §8.2 says is wanted — "a change in the harness's route
distribution between two builds is a regression signal". **Label** every series
with its source, which keeps both and risks somebody summing across the label
and recreating the defect exactly.

Labelled, with one rule that makes the risk explicit rather than latent:

> **No series is emitted without a `source` label, and no unlabelled total is
> emitted anywhere.**

There is no `assistant_answers_total` to sum by accident; there is only
`assistant_answers_total{source="web",path="refuse"}`. Re-mixing evaluation with
real traffic is then something a person types — `sum(...) without (source)` — in
a query they can be asked to justify, rather than something the endpoint did to
them silently. That is the whole difference between this and the state
e5bd236 fixed.

The label values are closed: `cli`, `web`, `evaluation`, `test`, `unknown`, and
anything else folded into `other`. `source` is caller-claimed and unvalidated by
design (§8.2 declines a CHECK constraint, because a rejected insert would be an
observability write failing an answer) — so a typo in a harness flag must not be
able to mint an unbounded number of Prometheus series through a table that has
no constraint to stop it. It is counted under `other`, visibly.

## What this endpoint is allowed to say

`/metrics` is unauthenticated and may be scraped by anything that can reach the
port. It is treated as public, and the rule is aggregate-only:

* No question text, no answer text, no passage text. `turn_traces` holds none
  of those by construction — that is enforced in `tests/test_spans.py` — but
  this module does not rely on that: it reads six named attributes, all of them
  numbers or booleans, and never renders `attributes` wholesale.
* No fingerprints either, although the table has them. A question fingerprint
  is stable across askers, so a scraper could confirm a guess ("was anyone
  asked *this*?") by hashing a candidate and looking for it. Counting is
  enough for every metric here, so the fingerprint is simply not read.
* No ids. No `trace_id`, `session_id` or `turn_id` reaches the output, so the
  endpoint cannot be used to enumerate conversations and then fetch them.
* Every label value is drawn from a closed set decided in this file — sources,
  outcomes, span names, check numbers, histogram bounds — never passed through
  from a row. A row can influence a *count*; it cannot influence a *name*.

## What bounds the work

A metrics scrape must not table-scan a fortnight of history, and this one
cannot: it makes exactly one repository call, `traces(limit=WINDOW)`, which both
adapters answer with `ORDER BY id DESC LIMIT ?` over the primary key. `WINDOW`
is a module constant and `render()`'s `window` argument is clamped to it, so no
caller — and no query string a later route might pass through — can widen it.
Everything after that is a single pass over at most `WINDOW` rows in memory.

There is no pagination, no `?since=`, no filter parameter. Every one of those
would be a knob on an unauthenticated endpoint whose whole defence is that it
takes no input.

## What is not here, and why

**Hand-off rate needs a definition, and this file makes it.** The router has
seven outcomes and the review names three. `route`, `extract` and `compose`
count as answered; `refuse` counts as refused; `cited hand-off`, `diagnosis` and
`ask back` count as hand-offs, because all three end with the hand-off renderer
and a person. The path distribution is exposed separately and unrolled, so
anyone who disagrees with that grouping can recompute it from
`assistant_answers_total` without arguing with this module.

**No per-check latency, and no per-route latency.** Span durations are keyed by
span name only. The join exists in the data — `checks` and `route` share a
`trace_id` — but reconstructing trees here would turn a single pass into a
grouping, and nothing in §2.7 asks for it.

**`answer_log` is deliberately not read**, which is a departure from §2.7's
wording and worth defending rather than hiding. Everything §2.7 lists is
present in `turn_traces`: the path is on the `part` span, the failed check
numbers are on the `checks` span, the cache hit is on `cache_lookup`, the top
score is on `retrieval`. `answer_log` would add nothing except two problems —
it is the one table in this system that retains question text, so reading it
from a public endpoint puts that text one careless line away from the response
body; and it is unbounded while `turn_traces` is pruned at fourteen days, so a
rate mixing the two would divide a fortnight by a lifetime. One table, one
window, one meaning.
"""

from __future__ import annotations

from typing import Any, Iterable

try:
    from . import otel_export
except ImportError:
    otel_export = None  # type: ignore

# How many spans one scrape may read. Roughly a day of heavy development
# traffic, by the same measurement that sized TRACE_ROW_CAP — about two thousand
# span rows on a busy day — so a normal window covers everything and the clamp
# binds only when something is generating traffic worth noticing separately.
#
# It is the hard ceiling as well as the default: `render(window=...)` clamps to
# it. An unauthenticated endpoint does not take a size from its caller.
WINDOW = 5_000

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# The surfaces that may appear as a label value. Anything else is `other`; see
# the module docstring on why a caller-claimed, constraint-free column is not
# allowed to mint series.
SOURCES = ("cli", "web", "evaluation", "test", "unknown")
OTHER_SOURCE = "other"

# Router outcomes, as Path_ spells them, mapped to the three rates §2.7 names.
# Spaces are not legal-looking in a label value even though the format permits
# them, so the path label is the underscored form and the mapping is here rather
# than inferred anywhere else.
PATHS = {
    "route": ("route", "answered"),
    "extract": ("extract", "answered"),
    "compose": ("compose", "answered"),
    "cited hand-off": ("cited_handoff", "handoff"),
    "diagnosis": ("diagnosis", "handoff"),
    "ask back": ("ask_back", "handoff"),
    "refuse": ("refuse", "refused"),
}
OUTCOMES = ("answered", "refused", "handoff")

# Cumulative upper bounds for the retrieval top score. Cosine similarity, so
# [0, 1]; the bounds cluster around the abstention threshold because that is the
# only region where the shape of the distribution changes an operational
# decision. Decision 9's sweep is the reason the resolution is there and not
# spread evenly: the two unanswerable situations scored 0.717 and 0.739 against
# answerable ones at 0.595 and 0.615, so the interesting question is how much of
# the traffic sits in the band no threshold separates.
SCORE_BUCKETS = (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)

QUANTILES = (0.5, 0.9, 0.99)

# Span names that may appear as a label value. Closed for the same reason the
# source list is, although the risk is smaller: span names come from call sites
# in this repository, not from callers. Listing them means a span renamed or
# added upstream shows up as `other` in the latency series instead of silently
# adding a series nobody expected — which is a visible prompt to update this
# list, and is what the `metrics` test asserting the list matches the call sites
# would otherwise have to be.
SPAN_NAMES = (
    "answer", "part", "split_by_topic", "cache_lookup", "perception",
    "slot_detection", "retrieval", "embed_question", "search",
    "targeted_retrieval", "route", "render", "generation", "checks",
)
OTHER_SPAN = "other"

# The six checks, so a check that never fails still reports a zero rather than
# vanishing from the output. A missing series and a zero series read very
# differently on a dashboard, and only one of them is true.
CHECKS = ("check 1", "check 2", "check 3", "check 4", "check 5", "check 6")


# ------------------------------------------------------------------ format
#
# The whole of the Prometheus text exposition format that this module needs.
# Kept together and kept short, because the argument for hand-writing it is that
# it is short.


def _escape_label(value: str) -> str:
    """Backslash, double quote and newline, in that order.

    Order matters: escaping the quote first and the backslash second would
    escape the backslash this function just inserted.
    """
    return (value.replace("\\", "\\\\")
                 .replace('"', '\\"')
                 .replace("\n", "\\n"))


def _escape_help(text: str) -> str:
    """HELP escapes backslash and newline only. A quote is literal there."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _number(value: float) -> str:
    """Render a sample value.

    Integers print without a decimal point because most of these are counts and
    `13` reads better than `13.0`; floats are rounded to six places, which is
    well inside the resolution of a cosine score and stops a repeating binary
    fraction filling a line. Prometheus accepts both.
    """
    if isinstance(value, bool):                 # bool is an int; be explicit
        return "1" if value else "0"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _sample(name: str, labels: dict[str, str], value: float) -> str:
    if not labels:
        return f"{name} {_number(value)}"
    rendered = ",".join(f'{k}="{_escape_label(str(v))}"'
                        for k, v in labels.items())
    return f"{name}{{{rendered}}} {_number(value)}"


def _family(name: str, kind: str, help_text: str,
            samples: Iterable[tuple[dict[str, str], float]]) -> list[str]:
    """One metric family: HELP, TYPE, then its samples.

    Emitted even when `samples` is empty, so a freshly built index serves a
    complete, parseable document that says zero everywhere rather than an empty
    body a scraper cannot distinguish from a broken endpoint.
    """
    lines = [f"# HELP {name} {_escape_help(help_text)}", f"# TYPE {name} {kind}"]
    lines.extend(_sample(name, labels, value) for labels, value in samples)
    return lines


# ------------------------------------------------------------------ reading
#
# Every reader below is defensive in the same way and for the same reason: these
# rows come out of a database that has no CHECK constraints on them, may have
# been written by an older build, and — in the case of `attributes` — round-trip
# through JSON. A metrics endpoint that raised on one malformed row would take
# out the observability of the whole system to report a number, which inverts
# the priority `record_spans` already establishes by refusing to fail an answer.


def _source_of(span: Any) -> str:
    value = str(getattr(span, "source", "") or "unknown").strip().lower()
    return value if value in SOURCES else OTHER_SOURCE


def _span_name(span: Any) -> str:
    name = str(getattr(span, "name", "") or "")
    return name if name in SPAN_NAMES else OTHER_SPAN


def _attributes(span: Any) -> dict:
    attributes = getattr(span, "attributes", None)
    return attributes if isinstance(attributes, dict) else {}


def _duration(span: Any) -> float | None:
    """Milliseconds, or None if the row does not carry a usable number.

    Negative durations are dropped rather than clamped to zero. A negative
    duration means something is wrong with a clock or a writer, and a zero would
    quietly pull a percentile down while looking like a fast answer.
    """
    raw = getattr(span, "duration_ms", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) if raw >= 0 else None


def _quantile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank, on values already sorted.

    Nearest-rank rather than interpolation because every value here is an
    observed duration and an interpolated one is a duration nothing took. On a
    handful of samples that distinction is the difference between "the slowest
    answer took 94 seconds" and a number between two real answers.
    """
    if not sorted_values:
        return 0.0
    rank = max(1, min(len(sorted_values),
                      int(-(-q * len(sorted_values) // 1))))  # ceil
    return sorted_values[rank - 1]


# ------------------------------------------------------------------ counting


class _Tally:
    """Everything one pass over the window learns, keyed by source.

    A class rather than eight dictionaries threaded through eight functions,
    and deliberately not a general aggregation framework: it holds exactly the
    six families §2.7 names, and a seventh family would be a decision to take
    rather than a key to add.
    """

    def __init__(self) -> None:
        self.spans: dict[str, int] = {}
        self.paths: dict[tuple[str, str], int] = {}
        self.outcomes: dict[tuple[str, str], int] = {}
        self.cache_lookups: dict[str, int] = {}
        self.cache_hits: dict[str, int] = {}
        self.check_runs: dict[str, int] = {}
        self.check_failures: dict[tuple[str, str], int] = {}
        self.scores: dict[str, list[float]] = {}
        self.durations: dict[tuple[str, str], list[float]] = {}
        self.errors: dict[tuple[str, str], int] = {}

    @staticmethod
    def _bump(store: dict, key, amount: int = 1) -> None:
        store[key] = store.get(key, 0) + amount

    def add(self, span: Any) -> None:
        source = _source_of(span)
        name = _span_name(span)
        attributes = _attributes(span)
        self._bump(self.spans, source)

        if str(getattr(span, "status", "ok") or "ok") != "ok":
            self._bump(self.errors, (source, name))

        duration = _duration(span)
        if duration is not None:
            self.durations.setdefault((source, name), []).append(duration)

        # The path a part took. Read off `part` rather than off `route`,
        # because a cached answer never reaches the router and would be missing
        # from a route-derived distribution — which is the one population a
        # cache-hit-rate metric on the same page would tell you had been
        # growing.
        if name == "part":
            path = str(attributes.get("path", "") or "")
            label, outcome = PATHS.get(path, ("", ""))
            if label:
                self._bump(self.paths, (source, label))
                self._bump(self.outcomes, (source, outcome))

        elif name == "cache_lookup":
            self._bump(self.cache_lookups, source)
            if attributes.get("hit") is True:
                self._bump(self.cache_hits, source)

        elif name == "checks":
            self._bump(self.check_runs, source)
            failed = attributes.get("failed")
            # A list of check labels, as `answer.py` writes it. Anything else
            # is an older or corrupted row and is counted as a run with no
            # failures rather than dropped: the denominator stays honest.
            if isinstance(failed, list):
                for entry in failed:
                    label = str(entry).strip().lower()
                    if label in CHECKS:
                        self._bump(self.check_failures, (source, label))

        elif name == "retrieval":
            score = attributes.get("top_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                self.scores.setdefault(source, []).append(float(score))


# ------------------------------------------------------------------ render


def render(repo: Any, window: int = WINDOW) -> str:
    """The whole exposition document, from one bounded read of `turn_traces`.

    Returns text, not bytes, so the caller owns the encoding and the header —
    `CONTENT_TYPE` is here for it to use.

    A repository that cannot answer `traces()` — an adapter predating the
    table, a store that has gone away — produces a valid document reporting an
    empty window rather than a 500. The endpoint's job is to be scrapeable; a
    scraper reading `assistant_window_spans 0` learns something, and a scraper
    reading a stack trace learns that monitoring is down when it is the store
    that is.
    """
    window = max(1, min(int(window or WINDOW), WINDOW))
    try:
        spans = repo.traces(limit=window) or []
    except Exception:                                  # noqa: BLE001
        # No event emitted. This is called from an HTTP handler that may be
        # scraped every fifteen seconds, and an observability failure that
        # writes a log line per scrape is a log flood dressed as diagnostics.
        # The empty window is itself the signal.
        spans = []

    tally = _Tally()
    for span in spans:
        try:
            tally.add(span)
        except Exception:                              # noqa: BLE001
            # One unreadable row must not cost the other four thousand nine
            # hundred and ninety-nine.
            continue

    # Only sources actually present, in a stable order, so the output diffs
    # cleanly between scrapes and a source that has sent nothing does not
    # advertise itself. `unknown` appears exactly when rows say `unknown`,
    # which is the visibility e5bd236's default was chosen for.
    present = [s for s in (*SOURCES, OTHER_SOURCE) if s in tally.spans]

    lines: list[str] = []

    lines += _family(
        "assistant_window_spans", "gauge",
        "Trace rows read by this scrape, per source. Every other metric here is "
        "counted over this window and not over all history.",
        [({"source": s}, tally.spans[s]) for s in present])

    lines += _family(
        "assistant_window_truncated", "gauge",
        "1 when the scrape hit its row cap, so the window is a slice of recent "
        "traffic rather than everything retained.",
        [({}, 1 if len(spans) >= window else 0)])

    lines += _family(
        "assistant_answers_total", "gauge",
        "Answered parts by router path, over the window. A question splits by "
        "topic, so parts exceed questions.",
        [({"source": s, "path": p}, n)
         for (s, p), n in sorted(tally.paths.items())])

    lines += _family(
        "assistant_outcomes_total", "gauge",
        "The same parts grouped as answered, refused or handed off. route, "
        "extract and compose answer; cited hand-off, diagnosis and ask back "
        "hand off; refuse refuses.",
        [({"source": s, "outcome": o}, tally.outcomes.get((s, o), 0))
         for s in present for o in OUTCOMES])

    lines += _family(
        "assistant_cache_lookups_total", "gauge",
        "Answer-cache lookups over the window.",
        [({"source": s}, tally.cache_lookups[s])
         for s in present if s in tally.cache_lookups])

    lines += _family(
        "assistant_cache_hits_total", "gauge",
        "Answer-cache hits over the window. Hit rate is hits over lookups; the "
        "cache is exact-key, so a low rate is decision 14's known limitation "
        "rather than a fault.",
        [({"source": s}, tally.cache_hits.get(s, 0))
         for s in present if s in tally.cache_lookups])

    lines += _family(
        "assistant_checks_runs_total", "gauge",
        "Runs of the six post-generation checks, which is the denominator for "
        "the failure counts below.",
        [({"source": s}, tally.check_runs[s])
         for s in present if s in tally.check_runs])

    lines += _family(
        "assistant_check_failures_total", "gauge",
        "Failures by check number. A check that never fails reports zero "
        "rather than disappearing.",
        [({"source": s, "check": c}, tally.check_failures.get((s, c), 0))
         for s in present if s in tally.check_runs for c in CHECKS])

    lines += _family(
        "assistant_span_errors_total", "gauge",
        "Spans that completed with status=error, by stage.",
        [({"source": s, "span": n}, v)
         for (s, n), v in sorted(tally.errors.items())])

    lines += _histogram(tally, present)
    lines += _summary(tally)

    # A trailing newline: the format is line-oriented and a parser reading the
    # last sample without one is entitled to treat it as truncated.
    return "\n".join(lines) + "\n"


def _histogram(tally: _Tally, present: list[str]) -> list[str]:
    """Retrieval top score as a cumulative histogram.

    A histogram rather than a summary because the operational question is not
    "what is the median score" — decision 9 already established that the median
    says nothing useful — but "how much traffic sits in the band around the
    threshold where no cut-off separates an answerable question from a
    near-miss". That is a bucket question.
    """
    lines = ["# HELP assistant_retrieval_top_score Top retrieval score per "
             "question part, bucketed. Buckets cluster around the abstention "
             "threshold, where decision 9 showed unanswerable questions "
             "outscoring answerable ones.",
             "# TYPE assistant_retrieval_top_score histogram"]
    for source in present:
        scores = tally.scores.get(source)
        if not scores:
            continue
        running = 0
        ordered = sorted(scores)
        for bound in SCORE_BUCKETS:
            running = sum(1 for s in ordered if s <= bound)
            lines.append(_sample("assistant_retrieval_top_score_bucket",
                                 {"source": source, "le": _number(bound)},
                                 running))
        lines.append(_sample("assistant_retrieval_top_score_bucket",
                             {"source": source, "le": "+Inf"}, len(ordered)))
        lines.append(_sample("assistant_retrieval_top_score_sum",
                             {"source": source}, sum(ordered)))
        lines.append(_sample("assistant_retrieval_top_score_count",
                             {"source": source}, len(ordered)))
    return lines


def _summary(tally: _Tally) -> list[str]:
    """Per-stage latency, as quantiles over the window.

    A summary and not a histogram, because the useful bounds differ by three
    orders of magnitude between stages: `slot_detection` is sub-millisecond and
    `generation` is measured in tens of seconds — decision 7's own closed
    latency item puts a cold compose at 115 to 199 seconds. One set of histogram
    buckets covering both would waste every bucket on one of them. The cost is
    that these quantiles cannot be aggregated across scrapes, which is the
    standard objection to summaries and is already true of everything else here:
    this endpoint reports a window.
    """
    lines = ["# HELP assistant_span_duration_ms Stage latency over the window, "
             "by span name. Nearest-rank quantiles over observed durations, not "
             "interpolated.",
             "# TYPE assistant_span_duration_ms summary"]
    for (source, name), values in sorted(tally.durations.items()):
        ordered = sorted(values)
        for q in QUANTILES:
            lines.append(_sample(
                "assistant_span_duration_ms",
                {"source": source, "span": name, "quantile": _number(q)},
                _quantile(ordered, q)))
        lines.append(_sample("assistant_span_duration_ms_sum",
                             {"source": source, "span": name}, sum(ordered)))
        lines.append(_sample("assistant_span_duration_ms_count",
                             {"source": source, "span": name}, len(ordered)))
    return lines


def export_to_otel(repo: Any, window: int = WINDOW) -> None:
    """Export metrics to OTLP if configured.

    Reads the same trace data as render() and sends key metrics to the OTLP endpoint.
    If endpoint is not configured or unreachable, silently continues.
    """
    if otel_export is None:
        return

    window = max(1, min(int(window or WINDOW), WINDOW))
    try:
        spans = repo.traces(limit=window) or []
    except Exception:
        return

    tally = _Tally()
    for span in spans:
        try:
            tally.add(span)
        except Exception:
            continue

    # Build OTel metrics list
    metrics = []
    present = [s for s in (*SOURCES, OTHER_SOURCE) if s in tally.spans]

    # Window info
    for source in present:
        metrics.append({
            "name": "assistant_window_spans",
            "gauge": {"dataPoints": [
                {"attributes": [{"key": "source", "value": {"stringValue": source}}],
                 "asInt": tally.spans.get(source, 0)}
            ]}
        })

    # Answer outcomes
    for (source, outcome), count in tally.outcomes.items():
        metrics.append({
            "name": "assistant_outcomes_total",
            "gauge": {"dataPoints": [
                {"attributes": [
                    {"key": "source", "value": {"stringValue": source}},
                    {"key": "outcome", "value": {"stringValue": outcome}}
                ], "asInt": count}
            ]}
        })

    # Cache hits
    for source in present:
        if source in tally.cache_lookups:
            metrics.append({
                "name": "assistant_cache_hits_total",
                "gauge": {"dataPoints": [
                    {"attributes": [{"key": "source", "value": {"stringValue": source}}],
                     "asInt": tally.cache_hits.get(source, 0)}
                ]}
            })

    # Check failures
    for (source, check), count in tally.check_failures.items():
        metrics.append({
            "name": "assistant_check_failures_total",
            "gauge": {"dataPoints": [
                {"attributes": [
                    {"key": "source", "value": {"stringValue": source}},
                    {"key": "check", "value": {"stringValue": check}}
                ], "asInt": count}
            ]}
        })

    # Span errors
    for (source, name), count in tally.errors.items():
        metrics.append({
            "name": "assistant_span_errors_total",
            "gauge": {"dataPoints": [
                {"attributes": [
                    {"key": "source", "value": {"stringValue": source}},
                    {"key": "span", "value": {"stringValue": name}}
                ], "asInt": count}
            ]}
        })

    # Latency summaries (simplified: just send one quantile per span)
    for (source, name), values in tally.durations.items():
        if values:
            ordered = sorted(values)
            p95 = _quantile(ordered, 0.95)
            metrics.append({
                "name": "assistant_span_duration_ms_p95",
                "gauge": {"dataPoints": [
                    {"attributes": [
                        {"key": "source", "value": {"stringValue": source}},
                        {"key": "span", "value": {"stringValue": name}}
                    ], "asDouble": p95}
                ]}
            })

    otel_export.export_metrics({"metrics": metrics})
