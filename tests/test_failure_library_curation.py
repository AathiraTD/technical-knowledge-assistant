"""Tests for failure library curation.

Integration with diagnosis capture: reads complete cases and outputs
curated dataset ready for fine-tuning analysis.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tools.curate_failure_library import (
    compute_statistics,
    curate_cases,
    extract_tags,
    load_cases,
)


@pytest.fixture
def sample_cases():
    """Sample diagnosis cases (complete, with expert_diagnosis)."""
    return [
        {
            "case_id": "case-001",
            "question": "How to diagnose rising damp",
            "image_hashes": ["hash1"],
            "retrieved_chunk_ids": ["chunk1"],
            "refusal_reason": "diagnosis_handed_off",
            "timestamp": "2026-09-16T10:00:00Z",
            "tags": ["substrate_lime_render", "symptom_efflorescence", "cause_rising_damp"],
            "expert_diagnosis": "Salt deposition from rising damp",
            "advisor_id": "staff_001",
            "confidence_level": "high",
        },
        {
            "case_id": "case-002",
            "question": "What's wrong with my wall",
            "image_hashes": [],
            "retrieved_chunk_ids": ["chunk2"],
            "refusal_reason": "diagnosis_handed_off",
            "timestamp": "2026-09-16T10:30:00Z",
            "tags": ["substrate_lime_mortar", "symptom_crazing"],
            "expert_diagnosis": "Drying shrinkage cracks",
            "advisor_id": "staff_002",
            "confidence_level": "medium",
        },
        {
            "case_id": "case-003",
            "question": "Incomplete case",
            "image_hashes": [],
            "retrieved_chunk_ids": [],
            "refusal_reason": "diagnosis_handed_off",
            "timestamp": "2026-09-16T11:00:00Z",
            "tags": [],
            "expert_diagnosis": None,  # Incomplete
            "advisor_id": None,
            "confidence_level": None,
        },
    ]


def test_extract_tags_with_structured_tags(sample_cases):
    """Extract substrate, symptom, root_cause from tags."""
    case = sample_cases[0]
    tags = extract_tags(case)

    assert tags["substrate"] == "lime_render"
    assert tags["symptom"] == "efflorescence"
    assert tags["root_cause"] == "rising_damp"


def test_extract_tags_partial(sample_cases):
    """Extract tags when not all categories present."""
    case = sample_cases[1]
    tags = extract_tags(case)

    assert tags["substrate"] == "lime_mortar"
    assert tags["symptom"] == "crazing"
    assert tags["root_cause"] is None


def test_extract_tags_empty(sample_cases):
    """Extract tags from case with no tags."""
    case = sample_cases[2]
    tags = extract_tags(case)

    assert tags["substrate"] is None
    assert tags["symptom"] is None
    assert tags["root_cause"] is None


def test_load_cases_from_directory(sample_cases):
    """Load cases from date-partitioned directory structure."""
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        date_dir = base / "2026-09-16"
        date_dir.mkdir()

        # Write complete cases
        (date_dir / "case-001.json").write_text(json.dumps(sample_cases[0]))
        (date_dir / "case-002.json").write_text(json.dumps(sample_cases[1]))
        (date_dir / "case-003.json").write_text(json.dumps(sample_cases[2]))

        cases = load_cases(base)

        # Only complete cases (with expert_diagnosis)
        assert len(cases) == 2
        assert cases[0]["case_id"] == "case-001"
        assert cases[1]["case_id"] == "case-002"


def test_load_cases_ignores_incomplete():
    """Load cases ignores cases without expert_diagnosis."""
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        date_dir = base / "2026-09-16"
        date_dir.mkdir()

        incomplete = {
            "case_id": "incomplete",
            "question": "test",
            "expert_diagnosis": None,  # Incomplete
        }
        (date_dir / "incomplete.json").write_text(json.dumps(incomplete))

        cases = load_cases(base)
        assert len(cases) == 0


def test_load_cases_empty_directory():
    """Load cases from empty directory returns empty list."""
    with TemporaryDirectory() as tmpdir:
        cases = load_cases(Path(tmpdir))
        assert cases == []


def test_load_cases_nonexistent_directory():
    """Load cases from non-existent directory returns empty list."""
    cases = load_cases(Path("/nonexistent/path"))
    assert cases == []


def test_compute_statistics(sample_cases):
    """Compute statistics from complete cases."""
    complete_cases = sample_cases[:2]  # Exclude incomplete case
    stats = compute_statistics(complete_cases)

    assert stats["total_cases"] == 2
    assert stats["by_substrate"]["lime_render"] == 1
    assert stats["by_substrate"]["lime_mortar"] == 1
    assert stats["by_symptom"]["efflorescence"] == 1
    assert stats["by_symptom"]["crazing"] == 1
    assert stats["by_root_cause"]["rising_damp"] == 1


def test_compute_statistics_empty_list():
    """Compute statistics from empty list."""
    stats = compute_statistics([])
    assert stats["total_cases"] == 0
    assert stats["by_substrate"] == {}
    assert stats["by_symptom"] == {}
    assert stats["by_root_cause"] == {}


def test_curate_cases_success(sample_cases):
    """Curate cases writes output file."""
    with TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        date_dir = input_dir / "2026-09-16"
        date_dir.mkdir(parents=True)

        # Write complete cases only
        for i, case in enumerate(sample_cases[:2]):
            (date_dir / f"case-{i:03d}.json").write_text(json.dumps(case))

        output_path = Path(tmpdir) / "curated.json"

        result = curate_cases(input_dir, output_path)

        assert result["status"] == "success"
        assert result["cases_curated"] == 2
        assert output_path.exists()

        # Verify output format
        curated = json.loads(output_path.read_text())
        assert curated["version"] == "1.0"
        assert "generated_at" in curated
        assert len(curated["cases"]) == 2
        assert "statistics" in curated


def test_curate_cases_no_cases(sample_cases):
    """Curate cases with no complete cases returns no_cases status."""
    with TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "empty"
        output_path = Path(tmpdir) / "curated.json"

        result = curate_cases(input_dir, output_path)

        assert result["status"] == "no_cases"
        assert not output_path.exists()


def test_curate_cases_stats_only(sample_cases):
    """Curate with stats_only returns statistics without writing file."""
    with TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        date_dir = input_dir / "2026-09-16"
        date_dir.mkdir(parents=True)

        for i, case in enumerate(sample_cases[:2]):
            (date_dir / f"case-{i:03d}.json").write_text(json.dumps(case))

        output_path = Path(tmpdir) / "curated.json"

        result = curate_cases(input_dir, output_path, stats_only=True)

        assert result["status"] == "stats_only"
        assert "statistics" in result
        assert not output_path.exists()


def test_curate_output_format(sample_cases):
    """Curate output has correct structure and fields."""
    with TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        date_dir = input_dir / "2026-09-16"
        date_dir.mkdir(parents=True)

        (date_dir / "case.json").write_text(json.dumps(sample_cases[0]))

        output_path = Path(tmpdir) / "curated.json"
        curate_cases(input_dir, output_path)

        curated = json.loads(output_path.read_text())

        # Verify top-level structure
        assert isinstance(curated, dict)
        assert "version" in curated
        assert "generated_at" in curated
        assert "cases" in curated
        assert "statistics" in curated

        # Verify case structure
        assert len(curated["cases"]) == 1
        case = curated["cases"][0]
        assert "id" in case
        assert "question" in case
        assert "expert_diagnosis" in case
        assert "images" in case
        assert "substrate" in case
        assert "symptom" in case
        assert "confidence" in case
        assert "advisor" in case
        assert "captured_at" in case

        # Verify statistics structure
        stats = curated["statistics"]
        assert "total_cases" in stats
        assert "by_substrate" in stats
        assert "by_symptom" in stats
        assert "by_root_cause" in stats
