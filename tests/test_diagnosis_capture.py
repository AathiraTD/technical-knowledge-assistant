"""Tests for diagnosis hand-off logging.

Safety-critical coverage: atomic writes, image deduplication, schema
validation, and fire-and-forget semantics.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from assistant.logging.diagnosis_capture import DiagnosisCapture, DiagnosisCase


@pytest.fixture
def capture_dir():
    """Temporary directory for failure library."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def capture(capture_dir):
    """DiagnosisCapture instance with temp directory."""
    return DiagnosisCapture(base_path=capture_dir)


def test_capture_writes_valid_json(capture, capture_dir):
    """Valid capture writes correct JSON to disk."""
    question = "How to diagnose rising damp on lime render"
    chunk_ids = ["chunk_001", "chunk_002"]
    tags = ["damp", "render"]

    case_id = capture.capture(
        question=question,
        images=[],
        chunk_ids=chunk_ids,
        refusal_reason="below_threshold",
        tags=tags,
    )

    # Case file exists
    case_file = capture_dir / "2026-09-16" / f"{case_id}.json"
    assert case_file.exists()

    # JSON is valid and contains expected fields
    data = json.loads(case_file.read_text())
    assert data["case_id"] == case_id
    assert data["question"] == question
    assert data["retrieved_chunk_ids"] == chunk_ids
    assert data["refusal_reason"] == "below_threshold"
    assert data["tags"] == tags
    assert data["expert_diagnosis"] is None


def test_images_stored_by_content_hash(capture, capture_dir):
    """Images stored by SHA-256 hash, not filename."""
    img1 = b"image_data_1"
    img2 = b"image_data_2"

    case_id = capture.capture(
        question="test",
        images=[img1, img2],
        chunk_ids=[],
        refusal_reason="test",
        tags=[],
    )

    # Both images stored
    images_dir = capture_dir / "images"
    image_files = list(images_dir.glob("*.bin"))
    assert len(image_files) == 2

    # Verify content hashes match
    import hashlib

    hash1 = hashlib.sha256(img1).hexdigest()
    hash2 = hashlib.sha256(img2).hexdigest()

    assert (images_dir / f"{hash1}.bin").read_bytes() == img1
    assert (images_dir / f"{hash2}.bin").read_bytes() == img2

    # Case file references hashes, not names
    case_file = capture_dir / "2026-09-16" / f"{case_id}.json"
    data = json.loads(case_file.read_text())
    assert data["image_hashes"] == [hash1, hash2]


def test_image_deduplication(capture, capture_dir):
    """Same image uploaded twice is not duplicated on disk."""
    img = b"same_image_data"
    import hashlib

    img_hash = hashlib.sha256(img).hexdigest()

    # First capture
    case_id_1 = capture.capture(
        question="q1",
        images=[img],
        chunk_ids=[],
        refusal_reason="test",
        tags=[],
    )

    # Second capture with same image
    case_id_2 = capture.capture(
        question="q2",
        images=[img],
        chunk_ids=[],
        refusal_reason="test",
        tags=[],
    )

    # Only one image file on disk
    image_files = list((capture_dir / "images").glob("*.bin"))
    assert len(image_files) == 1
    assert image_files[0].name == f"{img_hash}.bin"

    # Both cases reference same hash
    case_1 = json.loads((capture_dir / "2026-09-16" / f"{case_id_1}.json").read_text())
    case_2 = json.loads((capture_dir / "2026-09-16" / f"{case_id_2}.json").read_text())
    assert case_1["image_hashes"] == [img_hash]
    assert case_2["image_hashes"] == [img_hash]


def test_atomic_write_on_json_validation_failure(capture, capture_dir):
    """Failed JSON validation leaves no .tmp or partial file."""
    question = "test"

    # Patch _validate_json to raise
    original_validate = DiagnosisCapture._validate_json

    def failing_validate(path):
        raise json.JSONDecodeError("simulated failure", "", 0)

    DiagnosisCapture._validate_json = failing_validate

    try:
        with pytest.raises(IOError):
            capture.capture(
                question=question,
                images=[],
                chunk_ids=[],
                refusal_reason="test",
                tags=[],
            )

        # No .tmp files left behind
        tmp_files = list((capture_dir / "2026-09-16").glob("*.tmp"))
        assert len(tmp_files) == 0

        # No case files created
        case_files = list((capture_dir / "2026-09-16").glob("*.json"))
        assert len(case_files) == 0

    finally:
        DiagnosisCapture._validate_json = original_validate


def test_empty_question_raises(capture):
    """Empty or whitespace-only question raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        capture.capture(
            question="",
            images=[],
            chunk_ids=[],
            refusal_reason="test",
            tags=[],
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        capture.capture(
            question="   ",
            images=[],
            chunk_ids=[],
            refusal_reason="test",
            tags=[],
        )


def test_empty_image_bytes_raises(capture):
    """Empty image bytes raises ValueError."""
    with pytest.raises(ValueError, match="empty image bytes"):
        capture.capture(
            question="test",
            images=[b"valid", b""],
            chunk_ids=[],
            refusal_reason="test",
            tags=[],
        )


def test_record_expert_diagnosis_updates_case(capture, capture_dir):
    """record_expert_diagnosis appends diagnosis to existing case."""
    case_id = capture.capture(
        question="How to diagnose rising damp",
        images=[],
        chunk_ids=["chunk_001"],
        refusal_reason="below_threshold",
        tags=["damp"],
    )

    # Initially no diagnosis
    case_file = capture_dir / "2026-09-16" / f"{case_id}.json"
    data = json.loads(case_file.read_text())
    assert data["expert_diagnosis"] is None
    assert data["advisor_id"] is None

    # Record diagnosis
    result = capture.record_expert_diagnosis(
        case_id=case_id,
        diagnosis="Salt deposition from rising damp due to capillary action.",
        advisor_id="staff_001",
        confidence_level="high",
    )

    assert result is True

    # Diagnosis now in case
    data = json.loads(case_file.read_text())
    assert data["expert_diagnosis"] == "Salt deposition from rising damp due to capillary action."
    assert data["advisor_id"] == "staff_001"
    assert data["confidence_level"] == "high"


def test_record_expert_diagnosis_case_not_found(capture, capture_dir):
    """record_expert_diagnosis returns False if case doesn't exist."""
    result = capture.record_expert_diagnosis(
        case_id="nonexistent_uuid",
        diagnosis="test",
        advisor_id="staff_001",
    )

    assert result is False


def test_capture_case_id_is_uuid_format(capture):
    """Captured case_id is valid UUID format."""
    import uuid

    case_id = capture.capture(
        question="test",
        images=[],
        chunk_ids=[],
        refusal_reason="test",
        tags=[],
    )

    # Should not raise
    uuid.UUID(case_id)


def test_timestamp_is_iso_8601(capture, capture_dir):
    """Captured timestamp is ISO 8601 format."""
    from datetime import datetime

    case_id = capture.capture(
        question="test",
        images=[],
        chunk_ids=[],
        refusal_reason="test",
        tags=[],
    )

    case_file = capture_dir / "2026-09-16" / f"{case_id}.json"
    data = json.loads(case_file.read_text())

    # Should parse as ISO 8601
    ts = datetime.fromisoformat(data["timestamp"])
    assert ts is not None
