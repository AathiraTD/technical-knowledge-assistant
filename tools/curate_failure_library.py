"""CLI tool: curate the failure library from captured diagnosis hand-offs.

Process all captured diagnosis cases (from DiagnosisCapture), organize by
substrate/symptom/root-cause, and export as a curated dataset ready for
fine-tuning analysis and evaluation.

This tool reads complete cases (those with expert_diagnosis filled in) and
produces a summary report with statistics for machine learning workflows.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from assistant import observability as obs


def load_cases(input_dir: Path) -> list[dict]:
    """Load all diagnosis cases from date-partitioned directories.

    Args:
        input_dir: Path to data/failure_library/

    Returns:
        List of case dicts (complete cases only, with expert_diagnosis)
    """
    cases = []

    if not input_dir.exists():
        obs.event("curate_input_dir_not_found", path=str(input_dir))
        return cases

    # Recursively find all .json files (excluding .tmp files)
    for case_file in input_dir.glob("[0-9]*-[0-9]*-[0-9]*/*.json"):
        if case_file.name.endswith(".tmp"):
            continue

        try:
            data = json.loads(case_file.read_text())

            # Only include complete cases (with expert_diagnosis)
            if data.get("expert_diagnosis"):
                cases.append(data)

        except (json.JSONDecodeError, IOError) as e:
            obs.event("curate_case_read_error", path=str(case_file), error=str(e))
            continue

    obs.event("curate_cases_loaded", count=len(cases))
    return cases


def extract_tags(case: dict) -> dict[str, Optional[str]]:
    """Extract structured tags from a case.

    Args:
        case: Diagnosis case dict

    Returns:
        Dict with substrate, symptom, root_cause (or None if not in tags)
    """
    tags = case.get("tags", [])

    substrate = None
    symptom = None
    root_cause = None

    for tag in tags:
        if tag.startswith("substrate_"):
            substrate = tag.replace("substrate_", "")
        elif tag.startswith("symptom_"):
            symptom = tag.replace("symptom_", "")
        elif tag.startswith("cause_"):
            root_cause = tag.replace("cause_", "")

    return {
        "substrate": substrate,
        "symptom": symptom,
        "root_cause": root_cause,
    }


def compute_statistics(cases: list[dict]) -> dict:
    """Compute statistics over the curated dataset.

    Args:
        cases: List of complete cases

    Returns:
        Dict with statistics
    """
    stats = {
        "total_cases": len(cases),
        "by_substrate": defaultdict(int),
        "by_symptom": defaultdict(int),
        "by_root_cause": defaultdict(int),
    }

    for case in cases:
        tags = extract_tags(case)

        if tags["substrate"]:
            stats["by_substrate"][tags["substrate"]] += 1
        if tags["symptom"]:
            stats["by_symptom"][tags["symptom"]] += 1
        if tags["root_cause"]:
            stats["by_root_cause"][tags["root_cause"]] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    stats["by_substrate"] = dict(stats["by_substrate"])
    stats["by_symptom"] = dict(stats["by_symptom"])
    stats["by_root_cause"] = dict(stats["by_root_cause"])

    return stats


def curate_cases(
    input_dir: Path,
    output_path: Path,
    stats_only: bool = False,
) -> dict:
    """Curate failure library and write output.

    Args:
        input_dir: Path to data/failure_library/
        output_path: Path where curated.json will be written
        stats_only: If True, only compute and return statistics

    Returns:
        Dict with result summary
    """
    cases = load_cases(input_dir)

    if not cases:
        obs.event("curate_no_cases", input_dir=str(input_dir))
        return {"status": "no_cases", "message": "No complete cases found"}

    statistics = compute_statistics(cases)

    if stats_only:
        return {
            "status": "stats_only",
            "statistics": statistics,
        }

    # Build curated dataset
    curated = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "cases": [
            {
                "id": case["case_id"],
                "question": case["question"],
                "images": case.get("image_hashes", []),
                "substrate": extract_tags(case)["substrate"],
                "symptom": extract_tags(case)["symptom"],
                "expert_diagnosis": case["expert_diagnosis"],
                "confidence": case.get("confidence_level", "medium"),
                "advisor": case.get("advisor_id", "unknown"),
                "captured_at": case.get("timestamp", ""),
            }
            for case in cases
        ],
        "statistics": statistics,
    }

    # Write atomically
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tmp_path.write_text(json.dumps(curated, indent=2))
        tmp_path.replace(output_path)
        obs.event(
            "curate_output_written",
            path=str(output_path),
            num_cases=len(curated["cases"]),
        )
        return {
            "status": "success",
            "cases_curated": len(curated["cases"]),
            "output_path": str(output_path),
            "statistics": statistics,
        }

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        obs.event("curate_write_error", path=str(output_path), error=str(e))
        return {
            "status": "error",
            "message": f"Failed to write output: {str(e)}",
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Curate failure library from captured diagnosis cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/failure_library"),
        help="Input directory (default: data/failure_library/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/failure_library/curated.json"),
        help="Output file (default: data/failure_library/curated.json)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics only, do not write output file",
    )

    args = parser.parse_args()

    result = curate_cases(args.input, args.output, stats_only=args.stats)

    if result["status"] == "success":
        print(f"✓ Curated {result['cases_curated']} cases")
        print(f"  Output: {result['output_path']}")
        print(f"\n  Statistics:")
        stats = result["statistics"]
        print(f"    Total cases: {stats['total_cases']}")
        if stats["by_substrate"]:
            print(f"    By substrate: {stats['by_substrate']}")
        if stats["by_symptom"]:
            print(f"    By symptom: {stats['by_symptom']}")
        if stats["by_root_cause"]:
            print(f"    By root cause: {stats['by_root_cause']}")

    elif result["status"] == "stats_only":
        print("Statistics:")
        stats = result["statistics"]
        print(f"  Total complete cases: {stats['total_cases']}")
        if stats["by_substrate"]:
            print(f"  By substrate: {stats['by_substrate']}")
        if stats["by_symptom"]:
            print(f"  By symptom: {stats['by_symptom']}")
        if stats["by_root_cause"]:
            print(f"  By root cause: {stats['by_root_cause']}")

    else:
        print(f"✗ {result['status']}: {result.get('message', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
