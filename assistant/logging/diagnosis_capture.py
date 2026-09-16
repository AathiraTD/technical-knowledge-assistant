"""Diagnosis hand-off logging: capture evidence for the failure library.

Every time the system hands off a diagnosis question to the technical team,
this module captures:
- The question and any uploaded images
- What the system retrieved and why it refused to diagnose
- The expert's corrected answer (added later by an advisor)

Captures are atomic: write to .tmp, validate, then rename. Failed captures
are logged as observability events but do not crash the answer engine.

Storage is content-addressed: images stored by SHA-256 hash, captures by case
UUID. This prevents duplicates and keeps the evidence auditable.

DECISIONS 16.1 names this as the mechanism for building the labelled failure
library: "The hand-off is the data-collection mechanism — logging these
photographs against what the advisor answered is what builds the labelled
failure library."
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .. import observability as obs


@dataclass
class DiagnosisCase:
    """One captured hand-off: the raw facts only, not the model's reasoning."""

    case_id: str
    question: str
    image_hashes: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    refusal_reason: str = ""
    timestamp: str = ""
    tags: list[str] = field(default_factory=list)
    expert_diagnosis: Optional[str] = None
    advisor_id: Optional[str] = None
    confidence_level: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return asdict(self)


class DiagnosisCapture:
    """Fire-and-forget capture of diagnosis hand-offs."""

    def __init__(self, base_path: Path = Path("data/failure_library")):
        """Initialize capture paths.

        Args:
            base_path: Root directory for failure library. Creates
                data/failure_library/{date}/ and
                data/failure_library/images/ subdirectories.
        """
        self.base_path = Path(base_path)
        self.cases_dir = self.base_path / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.images_dir = self.base_path / "images"

        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        question: str,
        images: list[bytes],
        chunk_ids: list[str],
        refusal_reason: str,
        tags: list[str],
    ) -> str:
        """Capture a diagnosis hand-off atomically.

        Args:
            question: The user's question
            images: Raw image bytes (empty list if no images)
            chunk_ids: IDs of retrieved chunks that influenced the refusal
            refusal_reason: Why the system refused (e.g., "photograph_only", "no_results")
            tags: Structured tags for grouping later (e.g., ["damp", "render"])

        Returns:
            case_id (UUID string) on success

        Raises:
            ValueError: If question is empty or images are corrupt
            IOError: If filesystem operation fails (atomic rollback on write)
        """
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        try:
            case_id = str(uuid.uuid4())
            image_hashes = []

            # Store images, content-addressed by SHA-256
            for img_bytes in images:
                if not img_bytes:
                    raise ValueError("empty image bytes")
                img_hash = hashlib.sha256(img_bytes).hexdigest()
                image_hashes.append(img_hash)

                # Only write if not already present (deduplication)
                img_path = self.images_dir / f"{img_hash}.bin"
                if not img_path.exists():
                    tmp_path = img_path.with_suffix(img_path.suffix + ".tmp")
                    try:
                        tmp_path.write_bytes(img_bytes)
                        tmp_path.replace(img_path)
                    except Exception as e:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        raise IOError(f"Failed to write image {img_hash}: {e}") from e

            # Build case record
            case = DiagnosisCase(
                case_id=case_id,
                question=question.strip(),
                image_hashes=image_hashes,
                retrieved_chunk_ids=chunk_ids,
                refusal_reason=refusal_reason,
                timestamp=datetime.now(timezone.utc).isoformat(),
                tags=tags,
            )

            # Atomic write: write to .tmp, validate, rename
            case_path = self.cases_dir / f"{case_id}.json"
            tmp_path = case_path.with_suffix(case_path.suffix + ".tmp")

            try:
                tmp_path.write_text(json.dumps(case.to_dict(), indent=2))
                DiagnosisCapture._validate_json(tmp_path)
                tmp_path.replace(case_path)
            except Exception as e:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise IOError(f"Failed to write case {case_id}: {e}") from e

            obs.event("diagnosis_captured", case_id=case_id, num_images=len(images), tags=tags)
            return case_id

        except Exception as e:
            obs.event("diagnosis_capture_failed", error=str(e), question_len=len(question))
            raise

    def record_expert_diagnosis(
        self,
        case_id: str,
        diagnosis: str,
        advisor_id: str,
        confidence_level: str = "high",
    ) -> bool:
        """Record the expert's corrected diagnosis (called later, asynchronously).

        Args:
            case_id: UUID returned from capture()
            diagnosis: The advisor's diagnosis (the corrected answer)
            advisor_id: Staff member ID for attribution
            confidence_level: "high", "medium", or "low"

        Returns:
            True if successful, False if case not found
        """
        try:
            # Search for case in date-partitioned directories
            case_path = None
            for date_dir in self.base_path.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"):
                candidate = date_dir / f"{case_id}.json"
                if candidate.exists():
                    case_path = candidate
                    break

            if case_path is None:
                obs.event("diagnosis_update_not_found", case_id=case_id)
                return False

            case_data = json.loads(case_path.read_text())
            case_data["expert_diagnosis"] = diagnosis
            case_data["advisor_id"] = advisor_id
            case_data["confidence_level"] = confidence_level

            tmp_path = case_path.with_suffix(case_path.suffix + ".tmp")
            try:
                tmp_path.write_text(json.dumps(case_data, indent=2))
                DiagnosisCapture._validate_json(tmp_path)
                tmp_path.replace(case_path)
            except Exception as e:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise IOError(f"Failed to update case {case_id}: {e}") from e

            obs.event("diagnosis_updated", case_id=case_id, advisor_id=advisor_id)
            return True

        except Exception as e:
            obs.event("diagnosis_update_failed", case_id=case_id, error=str(e))
            return False

    @staticmethod
    def _validate_json(path: Path) -> None:
        """Validate JSON is valid before rename.

        Args:
            path: Path to JSON file

        Raises:
            json.JSONDecodeError: If JSON is invalid
        """
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise json.JSONDecodeError("Expected dict at top level", "", 0)
        if "case_id" not in data:
            raise json.JSONDecodeError("Missing case_id", "", 0)
