"""Everything that must be true before `assistant` can even be imported.

`python -m assistant.health` is the readiness check and it is the right one --
but it imports the application, so on a machine where `pip install` has not been
run it does not report a missing dependency, it raises `ModuleNotFoundError`
from inside a package the reader has never opened. That is the first thing an
assessor sees on a clean clone, and it reads like the project is broken rather
than like a step was skipped.

So this runs first and uses the standard library only. It checks the things
whose failure would turn the readiness check itself into a traceback -- the
interpreter version, the five pinned dependencies, the orchestration package --
and then hands over to `assistant.health` for everything that needs the
application loaded.

    python scripts/preflight.py            the checks, and what to do about them
    python scripts/preflight.py --json     the same, for a script to read

Exit code 0 when the application can be imported, 1 when it cannot. That is
deliberately a lower bar than readiness: a machine with no index passes here and
fails `assistant.health`, and those are different problems with different fixes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MINIMUM_PYTHON = (3, 11)

# Import name on the left, because that is what fails; the distribution name on
# the right, because that is what `pip install` wants. They differ for three of
# the six, which is exactly the confusion worth spending a dictionary on.
DEPENDENCIES = {
    "httpx": "httpx",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "pymupdf": "pymupdf",
    "numpy": "numpy",
    "langgraph": "langgraph",
}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _missing() -> list[str]:
    """Distribution names for the dependencies that cannot be imported.

    `find_spec` rather than `import`: importing pymupdf costs a shared library
    load, and this runs on every start of the demonstration.
    """
    absent = []
    for module, distribution in DEPENDENCIES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):       # a broken or partial install
            found = False
        if not found:
            absent.append(distribution)
    return absent


def ollama_tags(timeout: float = 5.0) -> list[str] | None:
    """The model tags Ollama holds, or None when it cannot be reached.

    Raw urllib rather than the application's own client, because the point of
    this file is to run before the application is importable.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST.rstrip('/')}/api/tags",
                                    timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return [m.get("name", "") for m in payload.get("models", [])]


def check() -> dict:
    """What is true, what is not, and the command that fixes each."""
    report: dict = {"ok": False, "checks": {}, "fix": []}

    version = tuple(sys.version_info[:2])
    report["python"] = ".".join(str(n) for n in sys.version_info[:3])
    report["checks"]["python_version"] = version >= MINIMUM_PYTHON
    if not report["checks"]["python_version"]:
        report["fix"].append(
            f"Python {'.'.join(str(n) for n in MINIMUM_PYTHON)} or later is "
            f"required; this is {report['python']}")

    absent = _missing()
    report["checks"]["dependencies_installed"] = not absent
    report["missing_dependencies"] = absent
    if absent:
        report["fix"].append(
            "pip install -r requirements.txt"
            f"   ({', '.join(absent)} not importable)")

    tags = ollama_tags()
    report["checks"]["ollama_reachable"] = tags is not None
    report["ollama_host"] = OLLAMA_HOST
    if tags is None:
        report["fix"].append(
            f"start the model server: ollama serve   (nothing answering at "
            f"{OLLAMA_HOST})")
    else:
        report["models"] = tags

    # The index is not required for `assistant` to import, so it is reported
    # rather than failed here -- `assistant.health` is what decides readiness.
    index = ROOT / "data" / "index" / "knowledge.db"
    report["checks"]["index_present"] = index.exists()
    if not index.exists():
        report["fix"].append(
            "build the index: python -m assistant.index   (about two minutes; "
            "the embedding cache ships, so nothing is re-embedded)")

    # Importable is the bar this file sets. The index and the models are
    # reported so one run explains the whole machine, and are left to the
    # readiness check to fail on.
    report["ok"] = (report["checks"]["python_version"]
                    and report["checks"]["dependencies_installed"])
    return report


def summary(report: dict) -> str:
    rows = [
        ("python", report["python"],
         "" if report["checks"]["python_version"]
         else f"needs {'.'.join(str(n) for n in MINIMUM_PYTHON)} or later"),
        ("dependencies",
         "installed" if report["checks"]["dependencies_installed"] else "MISSING",
         ", ".join(report["missing_dependencies"]) or "six packages, pinned"),
        ("ollama",
         "reachable" if report["checks"]["ollama_reachable"] else "UNREACHABLE",
         report["ollama_host"]),
        ("knowledge index",
         "present" if report["checks"]["index_present"] else "NOT BUILT",
         "data/index/knowledge.db"),
    ]
    width = max(len(label) for label, _, _ in rows)
    state = max(len(value) for _, value, _ in rows)
    lines = [f"{label:<{width}}  {value:<{state}}  {detail}".rstrip()
             for label, value, detail in rows]
    if report["fix"]:
        lines += ["", "To fix:"]
        lines += [f"  {step}" for step in report["fix"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/preflight.py",
        description="What must be true before the application can be imported.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = check()
    print(json.dumps(report, indent=1) if args.json else summary(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
