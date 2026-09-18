"""Tests for staff knowledge ingestion.

Safety-critical: approval workflow validation, audience filtering,
schema compliance, and versioning.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from assistant.indexing.staff_knowledge import StaffKnowledgeIngestor, StaffSubmission


@pytest.fixture
def ingestor():
    """StaffKnowledgeIngestor instance."""
    return StaffKnowledgeIngestor()


@pytest.fixture
def valid_submission():
    """Valid staff submission dict."""
    return {
        "question": "How to diagnose rising damp on lime render",
        "answer": "Rising damp appears as white crystalline deposits (salts) at the base of internal walls, typically rising 600-900mm up the wall.\n\nThe salts are carried by water from the ground through the masonry by capillary action.",
        "substrate": "lime_render",
        "submitted_by": "staff_001",
        "submitted_at": "2026-09-16T10:30:00Z",
        "approved_by": "supervisor_001",
        "approved_at": "2026-09-16T11:00:00Z",
        "tags": ["diagnosis", "damp", "render"],
        "confidence": "high",
    }


def test_valid_submission_passes_validation(ingestor, valid_submission):
    """Valid submission passes schema validation."""
    valid, error = ingestor.validate_submission(valid_submission)
    assert valid is True
    assert error is None


def test_missing_required_field_fails_validation(ingestor, valid_submission):
    """Missing required field fails validation."""
    del valid_submission["approved_by"]
    valid, error = ingestor.validate_submission(valid_submission)
    assert valid is False
    assert "approved_by" in error or "must be approved" in error


def test_missing_approval_timestamp_fails_validation(ingestor, valid_submission):
    """Missing approval timestamp fails validation."""
    del valid_submission["approved_at"]
    valid, error = ingestor.validate_submission(valid_submission)
    assert valid is False
    assert "approved_at" in error or "approval timestamp" in error


def test_wrong_field_type_fails_validation(ingestor, valid_submission):
    """Wrong field type fails validation."""
    valid_submission["question"] = 123  # Should be string
    valid, error = ingestor.validate_submission(valid_submission)
    assert valid is False
    assert "str" in error or "type" in error.lower()


def test_malformed_timestamp_fails_validation(ingestor, valid_submission):
    """Malformed ISO 8601 timestamp fails validation."""
    valid_submission["submitted_at"] = "2026-09-16 10:30:00"  # Wrong format
    valid, error = ingestor.validate_submission(valid_submission)
    assert valid is False
    assert "ISO 8601" in error or "submitted_at" in error


def test_invalid_confidence_level_fails_validation(ingestor, valid_submission):
    """Invalid confidence level fails validation."""
    valid_submission["confidence"] = "uncertain"
    valid, error = ingestor.validate_submission(valid_submission)
    assert valid is False
    assert "confidence" in error


def test_tags_must_be_list(ingestor, valid_submission):
    """Tags field must be a list."""
    valid_submission["tags"] = "diagnosis, damp"  # Should be list
    valid, error = ingestor.validate_submission(valid_submission)
    assert valid is False
    assert "tags" in error and "list" in error


def test_successful_ingestion_returns_doc_id(ingestor, valid_submission):
    """Successful ingestion returns doc_id."""
    success, result = ingestor.ingest(valid_submission)
    assert success is True
    assert isinstance(result, str)
    assert "staff_knowledge" in result


def test_failed_validation_on_ingest_returns_error(ingestor, valid_submission):
    """Invalid submission on ingest returns error."""
    del valid_submission["approved_by"]
    success, error = ingestor.ingest(valid_submission)
    assert success is False
    assert isinstance(error, str)
    assert len(error) > 0


def test_answer_is_chunked_by_paragraphs(ingestor):
    """Answer text is split into paragraph chunks."""
    answer = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = StaffKnowledgeIngestor._chunk_answer(answer, "lime_render")

    assert len(chunks) == 3
    assert chunks[0]["content"] == "First paragraph."
    assert chunks[1]["content"] == "Second paragraph."
    assert chunks[2]["content"] == "Third paragraph."


def test_empty_paragraphs_are_skipped(ingestor):
    """Empty paragraphs are skipped in chunking."""
    answer = "First.\n\n\n\nSecond."
    chunks = StaffKnowledgeIngestor._chunk_answer(answer, "lime_render")

    assert len(chunks) == 2
    assert chunks[0]["content"] == "First."
    assert chunks[1]["content"] == "Second."


def test_empty_answer_produces_no_chunks(ingestor):
    """Empty answer produces no chunks."""
    chunks = StaffKnowledgeIngestor._chunk_answer("", "lime_render")
    assert len(chunks) == 0

    chunks = StaffKnowledgeIngestor._chunk_answer("   ", "lime_render")
    assert len(chunks) == 0


def test_chunk_carries_metadata(ingestor):
    """Each chunk carries section, substrate, and index metadata."""
    answer = "Para 1.\n\nPara 2."
    chunks = StaffKnowledgeIngestor._chunk_answer(answer, "solid_brick")

    for i, chunk in enumerate(chunks):
        assert "content" in chunk
        assert "section" in chunk
        assert chunk["substrate"] == "solid_brick"
        assert chunk["chunk_index"] == i
        assert "Paragraph" in chunk["section"]


def test_load_submissions_from_file(ingestor):
    """Load staff submissions from JSON file."""
    with TemporaryDirectory() as tmpdir:
        submissions = [
            {
                "question": "Q1",
                "answer": "A1",
                "substrate": "lime",
                "submitted_by": "s1",
                "submitted_at": "2026-09-16T10:00:00Z",
                "approved_by": "a1",
                "approved_at": "2026-09-16T11:00:00Z",
            },
            {
                "question": "Q2",
                "answer": "A2",
                "substrate": "lime",
                "submitted_by": "s2",
                "submitted_at": "2026-09-16T10:30:00Z",
                "approved_by": "a2",
                "approved_at": "2026-09-16T11:30:00Z",
            },
        ]

        path = Path(tmpdir) / "staff_submissions.json"
        path.write_text(json.dumps({"submissions": submissions}))

        loaded = StaffKnowledgeIngestor.load_submissions_file(path)

        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1"
        assert loaded[1]["question"] == "Q2"


def test_load_submissions_missing_file_raises(ingestor):
    """Missing submissions file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        StaffKnowledgeIngestor.load_submissions_file(Path("/nonexistent/path.json"))


def test_load_submissions_invalid_json_raises(ingestor):
    """Invalid JSON raises JSONDecodeError."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.json"
        path.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            StaffKnowledgeIngestor.load_submissions_file(path)


def test_load_submissions_missing_submissions_key_raises(ingestor):
    """Missing 'submissions' key in JSON raises ValueError."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "no_submissions.json"
        path.write_text(json.dumps({"data": []}))

        with pytest.raises(ValueError, match="submissions"):
            StaffKnowledgeIngestor.load_submissions_file(path)


def test_staff_submission_dataclass_to_dict(ingestor, valid_submission):
    """StaffSubmission dataclass converts to dict."""
    sub = StaffSubmission(**valid_submission)
    data = sub.to_dict()

    assert isinstance(data, dict)
    assert data["question"] == valid_submission["question"]
    assert data["approved_by"] == valid_submission["approved_by"]
    assert data["tags"] == valid_submission["tags"]


def test_ingestion_empty_answer_fails(ingestor, valid_submission):
    """Empty answer during ingestion fails."""
    valid_submission["answer"] = ""
    success, error = ingestor.ingest(valid_submission)
    assert success is False
    assert "chunks" in error.lower() or "answer" in error.lower()


def test_approval_is_required_field(ingestor, valid_submission):
    """Approval fields are strictly required."""
    # No approved_by
    sub = valid_submission.copy()
    sub["approved_by"] = None
    valid, error = ingestor.validate_submission(sub)
    assert valid is False

    # No approved_at
    sub = valid_submission.copy()
    sub["approved_at"] = None
    valid, error = ingestor.validate_submission(sub)
    assert valid is False
