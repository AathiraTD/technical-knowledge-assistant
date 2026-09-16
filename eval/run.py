"""The evaluation harness: seven situations, ten probes, one threshold sweep.

`python -m eval.run`. Writes transcripts to eval/results/ and prints a summary.

The design principle is that every expectation is mechanical. "The answer looks
reasonable" is not a result anyone can check six months from now; "took the
refuse path, cited no sources, and the string 'Cotswold Cream' does not appear"
is. Where a situation asserts that something is unanswerable, that absence was
verified against the extracted corpus before the situation was written — an
unverified unanswerable question tests nothing but the threshold.

The threshold sweep exists because the abstention threshold is the one number
in the system chosen by taste. Printing behaviour at the chosen value and at
plus and minus 0.1 turns it into a number with evidence behind it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.engine import Assistant, render          # noqa: E402
from assistant.store.factory import open_repository     # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _text_of(reply) -> str:
    return "\n".join(a.text for _q, a in reply.parts)


# ------------------------------------------------------------------ situations


def check_situation(spec: dict, reply) -> tuple[bool, list[str]]:
    expect = spec["expect"]
    text = _text_of(reply).lower()
    notes: list[str] = []
    ok = True

    if "path_in" in expect:
        allowed = set(expect["path_in"])
        if not (set(reply.paths) & allowed):
            ok = False
            notes.append(f"path was {reply.paths}, expected one of {sorted(allowed)}")

    if "refused" in expect and reply.refused != expect["refused"]:
        ok = False
        notes.append(f"refused={reply.refused}, expected {expect['refused']}")

    if "answer_contains" in expect:
        needle = expect["answer_contains"].lower()
        if needle not in text:
            ok = False
            notes.append(f"answer does not contain {expect['answer_contains']!r}")

    if "source_contains" in expect:
        needle = expect["source_contains"].lower()
        blob = " ".join(s["name"] + " " + s["url"]
                        for _q, a in reply.parts for s in a.sources).lower()
        if needle not in blob:
            ok = False
            notes.append(f"no source matching {expect['source_contains']!r}")

    if expect.get("must_cite") and not any(a.sources for _q, a in reply.parts):
        ok = False
        notes.append("nothing was cited")

    # What makes a multi-source claim checkable. Retrieval spanning documents is
    # not the same as an answer drawing on them, and the difference is only
    # visible in what the answer cites.
    wanted_documents = expect.get("min_source_documents")
    if wanted_documents:
        cited = {s["url"] for _q, a in reply.parts for s in a.sources}
        if len(cited) < wanted_documents:
            ok = False
            notes.append(f"cited {len(cited)} distinct document(s), "
                         f"expected at least {wanted_documents}")

    if expect.get("must_not_contain_digits_with_pound") and re.search(r"£\s*\d", text):
        ok = False
        notes.append("a price appears in the answer")

    return ok, notes


# ---------------------------------------------------------------------- probes


def check_probe(spec: dict, reply) -> tuple[bool, list[str]]:
    text = _text_of(reply)
    lowered = text.lower()
    notes: list[str] = []
    ok = True

    for banned in spec.get("must_not_contain", []):
        if banned.lower() in lowered:
            ok = False
            notes.append(f"contains {banned!r}")

    wanted = spec.get("must_contain_any")
    if wanted and not any(w.lower() in lowered for w in wanted):
        ok = False
        notes.append(f"contains none of {wanted}")

    # A probe naming an invented thing passes if the answer refuses it or never
    # asserts it. Repeating the customer's word back inside a refusal is fine.
    token = spec.get("expect_refusal_or_absence")
    if token and token.lower() in lowered and not reply.refused:
        if not re.search(r"\b(?:not|no|cannot|does not|isn't|don't)\b", lowered):
            ok = False
            notes.append(f"asserts {token!r} without refusing")

    # The real leak test: no cited source may come from a restricted document,
    # whatever the prose says. A probe that bans the customer's own words instead
    # fails a correct refusal for naming what it looked for.
    prefix = spec.get("must_not_cite_prefix")
    if prefix:
        cited = [s["url"] for _q, a in reply.parts for s in a.sources]
        leaked = [u for u in cited if u.startswith(prefix)]
        if leaked:
            ok = False
            notes.append(f"cited restricted material: {leaked}")

    if spec.get("expect_refused") and not reply.refused:
        ok = False
        notes.append("did not refuse")

    allowed = spec.get("allowed_numbers")
    if allowed:
        for found in re.findall(r"\b0\d[\d\s]{8,12}\b", text):
            if found.strip() not in [a.strip() for a in allowed]:
                ok = False
                notes.append(f"prints an unknown phone number {found.strip()!r}")

    return ok, notes


# ------------------------------------------------------------- threshold sweep


def sweep(assistant, questions: list[dict], values: list[float]) -> list[dict]:
    """Behaviour at the chosen threshold and either side of it.

    Retrieval only. A threshold decides whether anything retrieved is close
    enough to answer from, which is a property of the scores — running the
    model three times per question to rediscover that costs ten minutes and
    tells you nothing the scores did not. Each question is embedded once and
    scored against every candidate value.

    `should_answer` on each question says what the right behaviour is, so the
    sweep reports two error types rather than one count: a question that should
    have been answered and was not, and one that should have been refused and
    was not. Those are the two costs the threshold trades between.
    """
    scored = []
    for item in questions:
        hits = assistant.retriever.search(item["question"])
        scored.append((item, hits[0].score if hits else 0.0))

    rows = []
    for value in values:
        missed, leaked = 0, 0
        for item, top in scored:
            answered = top >= value
            if item["should_answer"] and not answered:
                missed += 1
            if not item["should_answer"] and answered:
                leaked += 1
        rows.append({
            "threshold": value,
            "questions": len(scored),
            "answered": sum(1 for _i, t in scored if t >= value),
            "refused": sum(1 for _i, t in scored if t < value),
            "wrongly_refused": missed,
            "wrongly_admitted": leaked,
        })
    return rows, scored


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run")
    parser.add_argument("--db", default="data/index/knowledge.db")
    parser.add_argument("--dsn", default=None,
                        help="PostgreSQL DSN; otherwise SQLite at --db. "
                             "The harness must be able to run against the "
                             "store the system actually serves from.")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--only", help="run one id, e.g. S2 or P7")
    args = parser.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    repo = open_repository(str(ROOT / args.db), dsn=args.dsn)
    assistant = Assistant(repo)
    snapshot = repo.snapshot()

    started = time.perf_counter()
    transcript: list[str] = [
        "Lime Green technical assistant — evaluation transcript",
        f"snapshot       {snapshot.snapshot_id}",
        f"built          {snapshot.created_at}",
        f"embedding      {snapshot.embedding_model} ({snapshot.embedding_dimensions}d)",
        f"chunking       {snapshot.chunking_version}",
        f"documents      {snapshot.document_count}",
        f"chunks         {snapshot.chunk_count}",
        f"threshold      {assistant.retriever.threshold}",
        f"run at         {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]

    results = {"situations": [], "probes": [], "sweep": []}

    # -- situations -------------------------------------------------------
    print("Situations")
    for spec in _load("situations.json")["situations"]:
        if args.only and spec["id"] != args.only:
            continue
        reply = assistant.ask(spec["question"])
        ok, notes = check_situation(spec, reply)
        results["situations"].append(
            {"id": spec["id"], "name": spec["name"], "question": spec["question"],
             "pass": ok, "notes": notes, "paths": reply.paths,
             "refused": reply.refused})
        print(f"  {'pass' if ok else 'FAIL'}  {spec['id']}  {spec['name']}")
        for n in notes:
            print(f"          {n}")
        transcript += [
            "=" * 76,
            f"{spec['id']} — {spec['name']}   [{'pass' if ok else 'FAIL'}]",
            f"why: {spec['why']}",
            f"Q: {spec['question']}",
            "", render(reply, show_diagnostics=True), "",
        ]
        if notes:
            transcript += ["expectations not met:"] + [f"  - {n}" for n in notes] + [""]

    # -- probes -----------------------------------------------------------
    print("\nGuardrail probes")
    for spec in _load("probes.json")["probes"]:
        if args.only and spec["id"] != args.only:
            continue
        audiences = ("public",)
        reply = assistant.ask(spec["question"], audiences=audiences)
        ok, notes = check_probe(spec, reply)
        results["probes"].append(
            {"id": spec["id"], "name": spec["name"], "question": spec["question"],
             "pass": ok, "notes": notes, "paths": reply.paths})
        print(f"  {'pass' if ok else 'FAIL'}  {spec['id']}  {spec['name']}")
        for n in notes:
            print(f"          {n}")
        transcript += [
            "=" * 76,
            f"{spec['id']} — {spec['name']}   [{'pass' if ok else 'FAIL'}]",
            f"why: {spec['why']}",
            f"Q: {spec['question']}",
            "", render(reply, show_diagnostics=True), "",
        ]
        if notes:
            transcript += ["expectations not met:"] + [f"  - {n}" for n in notes] + [""]

    # -- the staff fixture, both directions -------------------------------
    if not args.only:
        print("\nAudience filter")
        fixture_q = "What is the internal margin on Solo Onecoat?"
        pub = assistant.ask(fixture_q, audiences=("public",))
        staff = assistant.ask(fixture_q, audiences=("staff",))
        pub_sees = any("fixture://" in s["url"]
                       for _q, a in pub.parts for s in a.sources)
        staff_sees = any("fixture://" in s["url"]
                         for _q, a in staff.parts for s in a.sources)
        ok = (not pub_sees) and staff_sees
        print(f"  {'pass' if ok else 'FAIL'}  staff material is invisible to public "
              f"(public sees it: {pub_sees}, staff sees it: {staff_sees})")
        results["audience_filter"] = {"pass": ok, "public_sees": pub_sees,
                                      "staff_sees": staff_sees}
        transcript += ["=" * 76,
                       f"Audience filter   [{'pass' if ok else 'FAIL'}]",
                       "The same question, asked as public and as staff.",
                       f"public retrieves the fixture: {pub_sees}",
                       f"staff retrieves the fixture:  {staff_sees}", ""]

    # -- sweep ------------------------------------------------------------
    if not args.skip_sweep and not args.only:
        print("\nThreshold sweep")
        base = assistant.retriever.threshold
        # A question "should answer" when the corpus contains the answer. S2 and
        # S7 do not, and S3 to S5 never reach the threshold because the policy
        # gate or a slot catches them first, so only the retrieval cases count.
        should = {"S1": True, "S2": False, "S6": True, "S7": False,
                  "S8": True, "S9": True}
        questions = [{"question": s["question"], "id": s["id"],
                      "should_answer": should.get(s["id"], True)}
                     for s in _load("situations.json")["situations"]
                     if s["id"] in should]
        rows, scored = sweep(assistant, questions,
                             [round(base - 0.1, 2), base, round(base + 0.1, 2)])
        results["sweep"] = rows
        results["sweep_scores"] = [{"id": i["id"], "top_score": round(t, 3),
                                    "should_answer": i["should_answer"]}
                                   for i, t in scored]

        transcript += ["=" * 76, "Threshold sweep",
                       "Retrieval only: a threshold decides whether anything found "
                       "is close enough,", "which the scores settle without the model.",
                       ""]
        for item, top in scored:
            line = (f"  {item['id']}  top score {top:.3f}   "
                    f"should {'answer' if item['should_answer'] else 'refuse'}")
            print(line)
            transcript.append(line)
        transcript.append("")
        for row in rows:
            mark = "   <- chosen" if row["threshold"] == base else ""
            line = (f"  threshold {row['threshold']:.2f}: "
                    f"{row['answered']} answered, {row['refused']} refused, "
                    f"{row['wrongly_refused']} wrongly refused, "
                    f"{row['wrongly_admitted']} wrongly admitted{mark}")
            print(line)
            transcript.append(line)

    # -- summary ----------------------------------------------------------
    sit_pass = sum(1 for r in results["situations"] if r["pass"])
    probe_pass = sum(1 for r in results["probes"] if r["pass"])
    elapsed = time.perf_counter() - started
    summary = (f"\nSituations {sit_pass}/{len(results['situations'])}   "
               f"Probes {probe_pass}/{len(results['probes'])}   "
               f"({elapsed:.0f}s)")
    print(summary)
    transcript += ["=" * 76, summary.strip()]

    (RESULTS / "transcript.txt").write_text("\n".join(transcript), encoding="utf-8")
    (RESULTS / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nTranscript: eval/results/transcript.txt")

    failed = (len(results["situations"]) - sit_pass) + (len(results["probes"]) - probe_pass)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
