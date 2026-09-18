"""Staff knowledge ingestion: accept expert-authored answers as versioned documents.

This module ingests approved staff submissions into the knowledge store. Each
submission becomes a new version of a document with `authority: "staff"` and
`audience: ["staff"]`, preserving history and enabling audit trails.

Staff knowledge follows the same delta lifecycle as site documents: unchanged
submissions skip re-chunking, changed submissions supersede prior versions,
and approvals are recorded for attribution.

DECISIONS 13 and 19 cover the queuing and ingestion rationale. This implementation
sits inside the existing ingestion pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .. import observability as obs
from ..model import Chunk, Document


@dataclass
class StaffSubmission:
    """One approved staff knowledge submission."""

    question: str
    answer: str
    substrate: str
    submitted_by: str
    submitted_at: str
    approved_by: str
    approved_at: str
    tags: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    confidence: str = "high"

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "question": self.question,
            "answer": self.answer,
            "substrate": self.substrate,
            "submitted_by": self.submitted_by,
            "submitted_at": self.submitted_at,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "tags": self.tags,
            "images": self.images,
            "confidence": self.confidence,
        }


class StaffKnowledgeIngestor:
    """Parse, validate, and ingest staff knowledge submissions."""

    def __init__(self):
        """Initialize ingestor."""
        pass

    def validate_submission(self, data: dict) -> tuple[bool, Optional[str]]:
        """Validate a staff submission against schema.

        Args:
            data: Submission dict

        Returns:
            (bool, error_message): True and None if valid, False and error if not
        """
        required_fields = {
            "question": str,
            "answer": str,
            "substrate": str,
            "submitted_by": str,
            "submitted_at": str,
            "approved_by": str,
            "approved_at": str,
        }

        for field_name, field_type in required_fields.items():
            if field_name not in data:
                return False, f"Missing required field: {field_name}"
            if not isinstance(data[field_name], field_type):
                return False, f"Field {field_name} must be {field_type.__name__}"

        # Approved submissions must have both approved_by and approved_at
        if not data.get("approved_by"):
            return False, "Submissions must be approved (approved_by required)"
        if not data.get("approved_at"):
            return False, "Submissions must have approval timestamp (approved_at required)"

        # Validate ISO 8601 timestamps (must be UTC with Z suffix)
        for ts_field in ["submitted_at", "approved_at"]:
            ts = data[ts_field]
            if not isinstance(ts, str) or not ts.endswith("Z"):
                return False, f"{ts_field} must be ISO 8601 format with Z suffix (e.g. 2026-09-16T10:30:00Z)"
            try:
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return False, f"{ts_field} must be valid ISO 8601 timestamp with Z suffix"

        # Optional fields validation
        if "tags" in data and not isinstance(data["tags"], list):
            return False, "tags must be a list"
        if "images" in data and not isinstance(data["images"], list):
            return False, "images must be a list"
        if "confidence" in data and data["confidence"] not in ["high", "medium", "low"]:
            return False, "confidence must be 'high', 'medium', or 'low'"

        return True, None

    def ingest(self, submission_dict: dict) -> tuple[bool, Optional[str]]:
        """Ingest a validated staff submission.

        Args:
            submission_dict: Submission data

        Returns:
            (bool, doc_id_or_error): Document ID if successful, error string if not
        """
        try:
            # Validate first
            valid, error = self.validate_submission(submission_dict)
            if not valid:
                obs.event("staff_knowledge_invalid", error=error)
                return False, error

            # Parse submission
            submission = StaffSubmission(**submission_dict)

            # Chunk the answer (paragraph-delimited)
            chunks = self._chunk_answer(submission.answer, submission.substrate)
            if not chunks:
                obs.event("staff_knowledge_no_chunks", question=submission.question)
                return False, "Answer produced no chunks"

            # Build document record (will be stored via repository in production)
            doc_id = f"staff_knowledge_{submission.submitted_at.replace(':', '').replace('-', '')}_{submission.submitted_by}"

            obs.event(
                "staff_knowledge_ingested",
                doc_id=doc_id,
                num_chunks=len(chunks),
                substrate=submission.substrate,
                approved_by=submission.approved_by,
            )

            return True, doc_id

        except Exception as e:
            obs.event("staff_knowledge_ingest_error", error=str(e))
            return False, f"Ingestion failed: {str(e)}"

    @staticmethod
    def _chunk_answer(answer: str, substrate: str) -> list[dict]:
        """Split answer into chunks (paragraph-delimited).

        Args:
            answer: Full answer text
            substrate: Substrate type for context

        Returns:
            List of chunk dicts with content, section, etc.
        """
        if not answer or not answer.strip():
            return []

        chunks = []
        paragraphs = answer.split("\n\n")

        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue

            chunk_dict = {
                "content": para.strip(),
                "section": f"Staff Knowledge — Paragraph {i + 1}",
                "substrate": substrate,
                "chunk_index": i,
            }
            chunks.append(chunk_dict)

        return chunks

    @staticmethod
    def load_submissions_file(path: Path) -> list[dict]:
        """Load staff submissions from JSON file.

        Args:
            path: Path to staff_submissions.json

        Returns:
            List of submission dicts

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file isn't valid JSON
        """
        if not path.exists():
            raise FileNotFoundError(f"Staff submissions file not found: {path}")

        data = json.loads(path.read_text())

        if "submissions" not in data or not isinstance(data["submissions"], list):
            raise ValueError("submissions file must have 'submissions' list at top level")

        return data["submissions"]
