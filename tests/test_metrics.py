"""What `/metrics` may say, and what it may not.

Four groups, in the order they matter.

**The rates must not re-mix evaluation with real traffic.** This is the whole
reason the module exists rather than a tidy-up after it: commit e5bd236 added
`source` because the harness's near-miss and far-miss probes — which are
*supposed* to refuse — were being counted as the system's refusals, so the
better the guardrail worked the worse the number looked. Review §8.2 says a
`source` column nobody filters on is worse than no column. The tests below
therefore assert the invariant that makes that impossible rather than the output
of one query: **no series is emitted without a `source` label**, so re-mixing is
something a person types, not something the endpoint does to them.

**The endpoint is unauthenticated, so its inputs and its outputs are bounded.**
One repository call, clamped to a constant no caller can widen; label values
drawn from closed sets so a caller-claimed `source` cannot mint series; and
nothing in the body that could enumerate a conversation — no ids, no
fingerprints, no text. The privacy assertions are written against the rendered
document rather than against the reader, because the document is what leaves
the process.

**It must not be the thing that breaks.** An empty store, an adapter with no
`traces`, a store that raises, a row with a missing attribute, a negative
duration, a `failed` list that is not a list — each produces a valid document.
A metrics endpoint that 500s on a malformed row has taken out the observability
of the whole system in order to report a number.

**It must parse.** A hand-written exposition format is only defensible if
something checks it, and no parser is available without the dependency the
module exists to avoid — so there is a parser here, written against the format's
own rules (HELP before TYPE before samples, one TYPE per family, `le` on every
histogram bucket, a `+Inf` bucket, `_sum`/`_count` siblings, quantiles between
zero and one). It is deliberately strict: it rejects documents Prometheus would
also reject, and the point is to fail here rather than in a scrape.

Spans are constructed directly rather than answered into existence. A metric is
a function of rows, and building the rows states exactly which ones produce
which number — an end-to-end run would prove the same thing while leaving "why
is this number 3?" to be reconstructed from the engine.
"""

from __future__ import annotations

import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant import metrics, ollama, ui                           # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.model import TraceSpan                               # noqa: E402
from assistant.store.factory import open_repository                 # noqa: E402
from assistant.ui import Handler                                    # noqa: E402

from test_engine import build_repo, quoting, unit                   # noqa: E402


# --------------------------------------------------------------- fixtures


def span(name: str, *, source: str = "web", duration_ms: int = 5,
         status: str = "ok", **attributes) -> TraceSpan:
    """One row as `record_spans` would have written it."""
    return TraceSpan(trace_id="t1", span_id="s1", parent_span_id="",
                     turn_id="u1", session_id="", name=name,
                     started_at="2026-09-16T10:00:00", duration_ms=duration_ms,
                     status=status, source=source, attributes=dict(attributes))


class FakeRepo:
    """A store that is only its trace table, and remembers how it was asked.

    `calls` is the point of it: the boundedness test asserts on the `limit`
    that actually reached the repository, which is the only place the clamp can
    be observed from outside the module.
    """

    def __init__(self, spans=(), error: Exception | None = None) -> None:
        self.spans = list(spans)
        self.error = error
        self.calls: list[int] = []

    def traces(self, trace_id: str = "", session_id: str = "",
               limit: int = 1000):
        self.calls.append(limit)
        if self.error is not None:
            raise self.error
        return self.spans[-limit:]


# ------------------------------------------------------------ the parser


def parse(text: str) -> dict:
    """Parse the exposition format, strictly, and return name -> samples.

    Strict on purpose: every rule it enforces is one Prometheus enforces, so a
    document that passes here is one a scrape will accept. It raises rather
    than returning errors, because a malformed document has no partial reading
    worth having.
    """
    assert text.endswith("\n"), "the document must end with a newline"
    families: dict[str, dict] = {}
    seen_help: set[str] = set()
    for line in text.split("\n"):
        if not line:
            continue
        if line.startswith("# HELP "):
            name, _, help_text = line[len("# HELP "):].partition(" ")
            assert name not in seen_help, f"HELP repeated for {name}"
            assert help_text, f"HELP for {name} is empty"
            assert "\n" not in help_text
            seen_help.add(name)
            families.setdefault(name, {"type": "", "samples": []})
            continue
        if line.startswith("# TYPE "):
            name, _, kind = line[len("# TYPE "):].partition(" ")
            assert name in seen_help, f"TYPE before HELP for {name}"
            assert kind in ("counter", "gauge", "histogram", "summary",
                            "untyped"), kind
            assert not families[name]["type"], f"TYPE repeated for {name}"
            families[name]["type"] = kind
            continue
        assert not line.startswith("#"), f"unknown comment line: {line!r}"

        metric, _, value = line.rpartition(" ")
        float(value)                       # a sample value must be a number
        labels: dict[str, str] = {}
        if "{" in metric:
            metric, _, rest = metric.partition("{")
            assert rest.endswith("}"), line
            for pair in _split_labels(rest[:-1]):
                key, _, raw = pair.partition("=")
                assert raw.startswith('"') and raw.endswith('"'), line
                labels[key] = raw[1:-1]
        # A histogram's samples are named `_bucket`, `_sum` and `_count`; a
        # summary's are the bare name plus `_sum` and `_count`. Resolve each
        # back to the family that declared it.
        family = metric
        for suffix in ("_bucket", "_sum", "_count"):
            if family.endswith(suffix) and family[:-len(suffix)] in families:
                family = family[:-len(suffix)]
                break
        assert family in families, f"sample {metric} has no HELP/TYPE: {line!r}"
        families[family]["samples"].append((metric, labels, float(value)))
    for name, body in families.items():
        assert body["type"], f"{name} declared no TYPE"
    return families


def _split_labels(text: str) -> list[str]:
    """Split on commas that are not inside a quoted value."""
    out, current, quoted, escaped = [], "", False, False
    for char in text:
        if escaped:
            current, escaped = current + char, False
        elif char == "\\":
            current, escaped = current + char, True
        elif char == '"':
            current, quoted = current + char, not quoted
        elif char == "," and not quoted:
            out.append(current)
            current = ""
        else:
            current += char
    if current:
        out.append(current)
    return out


def samples(text: str, name: str) -> dict:
    """Samples of one family, keyed by their label set, for readable asserts."""
    body = parse(text)[name]
    return {tuple(sorted(labels.items())): value
            for metric, labels, value in body["samples"] if metric == name}


def check_histogram(text: str, name: str) -> None:
    """A histogram must be cumulative, carry `le` everywhere, and end at +Inf."""
    body = parse(text)[name]
    assert body["type"] == "histogram"
    per_source: dict[str, list] = {}
    for metric, labels, value in body["samples"]:
        if metric.endswith("_bucket"):
            assert "le" in labels, labels
            per_source.setdefault(labels["source"], []).append((labels["le"], value))
    for source, buckets in per_source.items():
        assert buckets[-1][0] == "+Inf", f"{source} has no +Inf bucket"
        values = [v for _, v in buckets]
        assert values == sorted(values), f"{source} buckets are not cumulative"


# ------------------------------------------------- the source invariant


def test_no_series_is_emitted_without_a_source_label():
    """The invariant that makes the e5bd236 defect unreachable.

    Not "the evaluation rows are excluded" and not "the labels are correct on
    the families I remembered to check": *every* sample carries `source`, so
    there is no total anywhere in the document that silently contains the probe
    set. A family added later without the label fails here.
    """
    text = metrics.render(FakeRepo([
        span("part", path="refuse", source="evaluation"),
        span("part", path="compose", source="web"),
        span("retrieval", top_score=0.7, source="evaluation"),
        span("checks", failed=["check 6"], source="evaluation"),
        span("cache_lookup", hit=True, source="web"),
    ]))
    for name, body in parse(text).items():
        if name == "assistant_window_truncated":
            continue            # a property of the scrape, not of any traffic
        for metric, labels, _ in body["samples"]:
            assert "source" in labels, f"{metric} has no source label"


def test_evaluation_refusals_do_not_appear_in_the_real_refusal_rate():
    """The defect itself, as an assertion.

    Three refusals from the harness's probes and one answered question from the
    web page. Read by source, the page refused nothing; summed, it would look
    like a system refusing three questions in four.
    """
    text = metrics.render(FakeRepo([
        span("part", path="refuse", source="evaluation"),
        span("part", path="refuse", source="evaluation"),
        span("part", path="refuse", source="evaluation"),
        span("part", path="compose", source="web"),
    ]))
    outcomes = samples(text, "assistant_outcomes_total")
    assert outcomes[(("outcome", "refused"), ("source", "web"))] == 0
    assert outcomes[(("outcome", "answered"), ("source", "web"))] == 1
    assert outcomes[(("outcome", "refused"), ("source", "evaluation"))] == 3
    assert outcomes[(("outcome", "answered"), ("source", "evaluation"))] == 0


def test_an_unrecognised_source_cannot_mint_a_series():
    """`source` has no CHECK constraint, so it must not reach a label value.

    §8.2 declines the constraint deliberately — a rejected insert would be an
    observability write failing an answer — which leaves this the only place the
    value is bounded. A mistyped harness flag is counted under `other` and is
    visible there; it does not add a Prometheus series per typo.
    """
    text = metrics.render(FakeRepo([
        span("part", path="compose", source="evaluatoin"),
        span("part", path="compose", source="Ｗ" * 300),
    ]))
    sources = {labels["source"]
               for body in parse(text).values()
               for _, labels, _ in body["samples"] if "source" in labels}
    assert sources == {"other"}
    assert samples(text, "assistant_window_spans")[(("source", "other"),)] == 2


# ------------------------------------------------------ the six metrics


def test_route_distribution_and_the_three_rates():
    """Paths unrolled, and grouped, from the same rows.

    Grouped and unrolled both, because the grouping is this module's judgement
    — diagnosis and ask back are hand-offs because both end at the hand-off
    renderer — and anyone who disagrees has to be able to recompute it.
    """
    text = metrics.render(FakeRepo([
        span("part", path="compose"), span("part", path="extract"),
        span("part", path="route"), span("part", path="refuse"),
        span("part", path="cited hand-off"), span("part", path="diagnosis"),
        span("part", path="ask back"),
    ]))
    paths = samples(text, "assistant_answers_total")
    assert paths[(("path", "cited_handoff"), ("source", "web"))] == 1
    assert paths[(("path", "ask_back"), ("source", "web"))] == 1
    outcomes = samples(text, "assistant_outcomes_total")
    assert outcomes[(("outcome", "answered"), ("source", "web"))] == 3
    assert outcomes[(("outcome", "refused"), ("source", "web"))] == 1
    assert outcomes[(("outcome", "handoff"), ("source", "web"))] == 3


def test_a_path_the_router_cannot_produce_is_counted_nowhere():
    """An unknown path is dropped rather than bucketed into a real outcome.

    Filing it under `answered` would overstate coverage and filing it under
    `refused` would understate it; the window count still shows the row, so it
    is visible as a discrepancy rather than absorbed into a rate.
    """
    text = metrics.render(FakeRepo([span("part", path="teleport"),
                                    span("part", path="compose")]))
    outcomes = samples(text, "assistant_outcomes_total")
    assert sum(outcomes.values()) == 1
    assert samples(text, "assistant_window_spans")[(("source", "web"),)] == 2


def test_cache_hit_rate_counts_lookups_and_hits():
    text = metrics.render(FakeRepo([
        span("cache_lookup", hit=True), span("cache_lookup", hit=False),
        span("cache_lookup", hit=False), span("cache_lookup", hit=False),
    ]))
    assert samples(text, "assistant_cache_lookups_total")[(("source", "web"),)] == 4
    assert samples(text, "assistant_cache_hits_total")[(("source", "web"),)] == 1


def test_check_failures_are_counted_by_number_and_zeroes_are_printed():
    """A check that never fires reports zero rather than vanishing.

    The distinction is the whole value of the metric: an absent series reads as
    "not measured" and a zero reads as "measured, never fired", and only one of
    those supports the over-refusal argument decision 7 rests on.
    """
    text = metrics.render(FakeRepo([
        span("checks", failed=["check 2", "check 6"]),
        span("checks", failed=["check 6"]),
        span("checks", failed=[]),
    ]))
    failures = samples(text, "assistant_check_failures_total")
    assert failures[(("check", "check 6"), ("source", "web"))] == 2
    assert failures[(("check", "check 2"), ("source", "web"))] == 1
    assert failures[(("check", "check 3"), ("source", "web"))] == 0
    assert samples(text, "assistant_checks_runs_total")[(("source", "web"),)] == 3


def test_top_scores_bucket_around_the_threshold():
    """Cumulative, with a +Inf bucket, and the sum and count that go with it."""
    text = metrics.render(FakeRepo([
        span("retrieval", top_score=0.2), span("retrieval", top_score=0.5),
        span("retrieval", top_score=0.72), span("retrieval", top_score=0.95),
    ]))
    check_histogram(text, "assistant_retrieval_top_score")
    buckets = {(labels["le"]): value
               for metric, labels, value in
               parse(text)["assistant_retrieval_top_score"]["samples"]
               if metric.endswith("_bucket")}
    assert buckets["0.45"] == 1          # only 0.2
    assert buckets["0.5"] == 2           # the boundary is inclusive
    assert buckets["0.75"] == 3
    assert buckets["+Inf"] == 4


def test_latency_quantiles_are_observed_durations_per_span_name():
    """Nearest-rank, so every quantile printed is a duration something took."""
    text = metrics.render(FakeRepo(
        [span("generation", duration_ms=d) for d in (10, 20, 30, 40, 90000)]
        + [span("search", duration_ms=3)]))
    durations = samples(text, "assistant_span_duration_ms")
    assert durations[(("quantile", "0.5"), ("source", "web"),
                      ("span", "generation"))] == 30
    assert durations[(("quantile", "0.99"), ("source", "web"),
                      ("span", "generation"))] == 90000
    assert durations[(("quantile", "0.5"), ("source", "web"),
                      ("span", "search"))] == 3
    assert parse(text)["assistant_span_duration_ms"]["type"] == "summary"


def test_a_span_name_the_code_does_not_emit_is_folded_into_other():
    text = metrics.render(FakeRepo([span("wat", duration_ms=1)]))
    assert (("source", "web"), ("span", "other")) in {
        tuple(sorted((k for k in labels.items() if k[0] != "quantile")))
        for _, labels, _ in
        parse(text)["assistant_span_duration_ms"]["samples"]}


def test_an_error_status_is_counted_by_stage():
    text = metrics.render(FakeRepo([
        span("generation", status="error"), span("generation", status="ok")]))
    errors = samples(text, "assistant_span_errors_total")
    assert errors[(("source", "web"), ("span", "generation"))] == 1


def test_every_family_is_a_gauge_or_a_distribution_never_a_counter():
    """The window is not monotonic, so nothing here may claim to be.

    Declared a counter, `rate(...)` over these numbers would return a plausible
    figure that means nothing, because the window shrinks and jumps when
    retention prunes. Declared a gauge, the same expression is obviously wrong.
    """
    text = metrics.render(FakeRepo([span("part", path="compose")]))
    kinds = {body["type"] for body in parse(text).values()}
    assert kinds <= {"gauge", "histogram", "summary"}
    assert "counter" not in kinds


# ------------------------------------------------------------- boundedness


def test_the_scrape_reads_one_bounded_page_however_it_is_asked():
    """The clamp is the module's, not the caller's.

    One repository call, and a `limit` that no argument can widen past the
    constant. An unauthenticated endpoint takes no size from its caller, and
    this is where that is true or not.
    """
    repo = FakeRepo([span("part", path="compose") for _ in range(10)])
    metrics.render(repo)
    metrics.render(repo, window=10 ** 9)
    metrics.render(repo, window=-1)
    metrics.render(repo, window=0)
    assert repo.calls == [metrics.WINDOW, metrics.WINDOW, 1, metrics.WINDOW]


def test_a_full_window_is_declared_truncated():
    """So nobody reads a windowed count as a lifetime total."""
    rows = [span("part", path="compose") for _ in range(4)]
    assert samples(metrics.render(FakeRepo(rows), window=4),
                   "assistant_window_truncated") == {(): 1}
    assert samples(metrics.render(FakeRepo(rows), window=5),
                   "assistant_window_truncated") == {(): 0}


def test_a_large_window_renders_in_one_pass():
    """Five thousand rows is the cap, and it has to be cheap enough to scrape."""
    rows = [span("part", path="compose"), span("retrieval", top_score=0.5),
            span("checks", failed=["check 1"]), span("cache_lookup", hit=False)]
    text = metrics.render(FakeRepo(rows * (metrics.WINDOW // 4)))
    assert samples(text, "assistant_window_spans")[(("source", "web"),)] == \
        metrics.WINDOW
    parse(text)


# ------------------------------------------------------------- degradation


def test_an_empty_store_renders_a_complete_document():
    """No traces yet is a state, not an error.

    A fresh index is the first thing anyone points a scraper at, and a 500 there
    reads as "monitoring is broken" rather than "nothing has been asked".
    """
    text = metrics.render(FakeRepo([]))
    families = parse(text)
    assert "assistant_outcomes_total" in families
    assert "assistant_retrieval_top_score" in families
    assert samples(text, "assistant_window_truncated") == {(): 0}


def test_a_store_that_raises_renders_an_empty_window():
    """Including an adapter that has no trace table at all."""
    text = metrics.render(FakeRepo(error=RuntimeError("no such table")))
    parse(text)
    assert samples(text, "assistant_window_spans") == {}

    class NoTraces:
        pass

    parse(metrics.render(NoTraces()))


@pytest.mark.parametrize("bad", [
    span("part"),                                   # no path attribute at all
    span("cache_lookup"),                           # no hit
    span("checks", failed="check 1"),               # a string, not a list
    span("checks", failed=[None, 7, "check 9"]),    # nothing that is a check
    span("retrieval", top_score="high"),            # not a number
    span("retrieval", top_score=True),              # a bool is an int; not a score
    span("part", duration_ms=-5, path="compose"),   # a clock went backwards
    span("part", duration_ms=None, path="compose"),
    TraceSpan(trace_id="", span_id="", parent_span_id="", turn_id="",
              session_id="", name="", started_at="", duration_ms=0,
              status="", source="", attributes=None),   # type: ignore[arg-type]
])
def test_a_malformed_row_does_not_break_the_document(bad):
    """Every one of these is a row a database with no CHECK constraints allows.

    The endpoint counts what it can read and renders the rest as absence. It
    does not raise, and it does not guess — a non-numeric score is not counted
    as zero, which would drag the histogram's first bucket up with traffic that
    never happened.
    """
    parse(metrics.render(FakeRepo([bad, span("part", path="compose")])))


def test_a_negative_duration_is_dropped_rather_than_clamped():
    """A clock that went backwards is not a fast answer."""
    text = metrics.render(FakeRepo([span("generation", duration_ms=-5),
                                    span("generation", duration_ms=40)]))
    counts = {tuple(sorted(labels.items())): value
              for metric, labels, value in
              parse(text)["assistant_span_duration_ms"]["samples"]
              if metric.endswith("_count")}
    assert counts[(("source", "web"), ("span", "generation"))] == 1


# ------------------------------------------------------------------ privacy


def test_nothing_identifying_reaches_the_document():
    """Aggregate only, asserted against what leaves the process.

    The table holds ids and question fingerprints and this endpoint reads
    neither — not because `turn_traces` is trusted to be clean (it is, and
    `tests/test_spans.py` enforces that) but because a scraper able to confirm
    "was anyone asked *this*?" by hashing a candidate would be an enumeration
    the aggregate contract does not permit.
    """
    secret = "9f2b17c4de01"
    text = metrics.render(FakeRepo([
        TraceSpan(trace_id=secret, span_id=secret, parent_span_id=secret,
                  turn_id=secret, session_id=secret, name="part",
                  started_at="2026-09-16T10:00:00", duration_ms=5, status="ok",
                  source="web",
                  attributes={"path": "compose", "question": secret,
                              "product": "Lime Green Solo",
                              "gen_ai.request.model": "qwen3.5:4b"}),
    ]))
    for forbidden in (secret, "Lime Green Solo", "qwen3.5:4b",
                      "trace_id", "session_id", "fingerprint"):
        assert forbidden not in text, forbidden


def test_label_values_are_escaped_not_trusted():
    """The escaping rules, exercised on a value that would otherwise break out.

    Nothing in the current output can reach here — every label value comes from
    a closed set — which is the reason to test the helper directly: the day a
    label is added that does pass a value through, this is the assertion that
    was already in place.
    """
    rendered = metrics._sample("m", {"l": 'a"b\\c\nd'}, 1)
    assert rendered == 'm{l="a\\"b\\\\c\\nd"} 1'
    # And it survives the round trip, which is the property that matters: an
    # escape that renders but does not parse is not an escape.
    document = "# HELP m h\n# TYPE m gauge\n" + rendered + "\n"
    assert parse(document)["m"]["samples"][0][1]["l"] == 'a\\"b\\\\c\\nd'


# ---------------------------------------------------------------- the route


@pytest.fixture
def server(tmp_path, monkeypatch):
    """The real page, serving the real route, over a real store.

    A live server rather than a hand-built handler because the thing under test
    is the wiring: that `/metrics` exists, answers before the 404, carries the
    right content type, and — the one that would otherwise be found in
    production — does not mint a session per scrape.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)
    build_repo(tmp_path, two_documents=True).close()
    repo = open_repository(str(tmp_path / "index" / "knowledge.db"), dsn="",
                           thread_safe=True)
    saved = {name: Handler.__dict__.get(name) for name in
             ("assistant", "meta", "audiences", "sessions", "uploads")}
    Handler.assistant = Assistant(repo, source="web")
    Handler.meta = "test"
    Handler.audiences = ("public",)
    Handler.sessions = ui.SessionStore()
    Handler.uploads = ui.UploadBudget()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{httpd.server_address[1]}", repo
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        repo.close()
        for name, value in saved.items():
            if value is None:
                if name in Handler.__dict__:
                    delattr(Handler, name)
            else:
                setattr(Handler, name, value)


def fetch(address: str, path: str) -> tuple[str, dict, str]:
    """GET, spoken to the socket.

    Through the socket rather than `urllib` for the reason the upload-guard
    tests give: the handler closes the connection in some paths, and a client
    still in a library's retry logic reports the close instead of the status.
    Here the body length is declared, so this is simply the most direct reading
    of exactly what the server wrote.
    """
    host, port = address.split(":")
    request = (f"GET {path} HTTP/1.1\r\nHost: {address}\r\n"
               "Connection: close\r\n\r\n")
    with socket.create_connection((host, int(port)), timeout=30) as client:
        client.sendall(request.encode("ascii"))
        raw = b""
        while True:
            block = client.recv(65536)
            if not block:
                break
            raw += block
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    headers = {}
    for line in lines[1:]:
        key, _, value = line.partition(":")
        headers.setdefault(key.strip().lower(), value.strip())
    return lines[0], headers, body.decode("utf-8")


def test_the_endpoint_serves_a_parseable_document(server):
    address, repo = server
    repo.record_spans([span("part", path="compose", source="web"),
                       span("retrieval", top_score=0.61, source="web")])
    status, headers, body = fetch(address, "/metrics")
    assert "200" in status, status
    assert headers["content-type"] == metrics.CONTENT_TYPE
    parse(body)
    assert samples(body, "assistant_outcomes_total")[
        (("outcome", "answered"), ("source", "web"))] == 1


def test_a_scrape_does_not_mint_a_session(server):
    """Otherwise an unauthenticated endpoint evicts real conversations.

    A scrape every fifteen seconds would open a `SessionStore` entry each time,
    and the store is bounded and least-recently-used — so the people actually
    using the page would lose their carried slots to the monitoring.
    """
    address, _ = server
    before = dict(Handler.sessions.__dict__)
    _, headers, _ = fetch(address, "/metrics")
    assert "set-cookie" not in headers
    # And nothing was opened behind the scenes either: the store is unchanged,
    # not merely uncommunicated.
    assert Handler.sessions.__dict__ == before


def test_the_endpoint_takes_no_parameters(server):
    """No window, no filter, no page. A query string changes nothing."""
    address, repo = server
    repo.record_spans([span("part", path="compose", source="web")])
    _, _, plain = fetch(address, "/metrics")
    _, _, fiddled = fetch(address, "/metrics?window=999999&source=staff&limit=1")
    assert plain == fiddled


def test_a_real_answer_is_countable_without_being_readable(server):
    """The one test that reads spans the engine actually wrote.

    Every other test builds its rows, which states precisely which row produces
    which number and says nothing about whether the engine still writes that
    row. This asks a real question through the real page and then scrapes, so a
    renamed span or a renamed attribute — `part`'s `path`, `retrieval`'s
    `top_score`, `cache_lookup`'s `hit` — shows up here as a metric that went to
    zero rather than as a dashboard nobody noticed had gone flat.
    """
    address, _ = server
    status, _, _ = fetch(address, "/ask?q=how%20much%20water%20per%20bag")
    assert "200" in status, status

    _, _, body = fetch(address, "/metrics")
    parse(body)
    outcomes = samples(body, "assistant_outcomes_total")
    assert sum(outcomes.values()) >= 1
    assert samples(body, "assistant_cache_lookups_total")[(("source", "web"),)] >= 1
    assert samples(body, "assistant_window_spans")[(("source", "web"),)] > 1
    # A real question, and still nothing of it in the document.
    for word in ("water", "bag", "how much"):
        assert word not in body


def test_an_unknown_path_is_still_a_404(server):
    """The route was added ahead of the 404 and must not have swallowed it."""
    address, _ = server
    status, _, _ = fetch(address, "/metrics/all")
    assert "404" in status, status
