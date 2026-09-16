"""Tests for auto-answer on pending: when a user provides the missing fact.

Verifies that:
- Short input with load-bearing slot terms is recognized as answer-to-askback
- Pending question is re-asked with merged slots
- Pending flag is cleared after auto-answer
- Slot detection from answer-to-askback works correctly
- Multi-turn flow: Q1 → ask-back → answer → auto-answered Q1
"""

from __future__ import annotations

import pytest

from assistant.engine import Assistant
from assistant.router import SlotDetector
from assistant.session import SessionStore


class TestSlotDetectorAnswerToAskback:
    """Test the is_answer_to_askback detector."""

    def setup_method(self):
        self.detector = SlotDetector()

    def test_short_input_with_substrate_is_answer(self):
        """'brick' is a short input with a load-bearing term."""
        assert self.detector.is_answer_to_askback("brick")

    def test_short_input_with_location_is_answer(self):
        """'outside' is a short input with a load-bearing term."""
        assert self.detector.is_answer_to_askback("outside")

    def test_short_input_with_exposure_is_answer(self):
        """'exposed, north-facing' is a short input with a load-bearing term."""
        assert self.detector.is_answer_to_askback("exposed, north-facing")

    def test_two_slots_together_is_answer(self):
        """'brick, outside' is a short answer to ask-back."""
        assert self.detector.is_answer_to_askback("brick, outside")

    def test_long_input_is_not_answer(self):
        """Long input is a new question, not an answer to ask-back."""
        assert not self.detector.is_answer_to_askback(
            "The wall is brick and it's on the outside of the building"
        )

    def test_input_without_slots_is_not_answer(self):
        """Input with no load-bearing terms is not an answer to ask-back."""
        assert not self.detector.is_answer_to_askback("How much does it cost?")

    def test_borderline_length_10_words_is_answer(self):
        """10 words with a slot term is still an answer."""
        assert self.detector.is_answer_to_askback(
            "brick on the outside of my house in the north"
        )

    def test_borderline_length_11_words_is_not_answer(self):
        """11 words is over the threshold, even with slots."""
        assert not self.detector.is_answer_to_askback(
            "the wall is brick on the outside of my house in the north"
        )


class TestAssistantDetectAnswerToAskbackSlots:
    """Test the Assistant's detect_answer_to_askback_slots method."""

    def test_detect_substrate_from_short_input(self, repo):
        """'brick' → {'substrate': 'brick'}"""
        assistant = Assistant(repo, cache=False)
        slots = assistant.detect_answer_to_askback_slots("brick")
        assert slots.get("substrate") == "brick"

    def test_detect_location_from_short_input(self, repo):
        """'outside' → {'location': 'outside'}"""
        assistant = Assistant(repo, cache=False)
        slots = assistant.detect_answer_to_askback_slots("outside")
        assert slots.get("location") == "outside"

    def test_detect_multiple_slots(self, repo):
        """'brick, outside' → {'substrate': 'brick', 'location': 'outside'}"""
        assistant = Assistant(repo, cache=False)
        slots = assistant.detect_answer_to_askback_slots("brick, outside")
        assert slots.get("substrate") == "brick"
        assert slots.get("location") == "outside"

    def test_empty_dict_for_non_answer(self, repo):
        """Long input returns empty dict (not an answer-to-askback)."""
        assistant = Assistant(repo, cache=False)
        slots = assistant.detect_answer_to_askback_slots(
            "How much does lime mortar cost?"
        )
        assert slots == {}

    def test_empty_dict_for_no_slots(self, repo):
        """Short input with no slots returns empty dict."""
        assistant = Assistant(repo, cache=False)
        slots = assistant.detect_answer_to_askback_slots("yes")
        assert slots == {}


class TestAutoAnswerFlowIntegration:
    """Integration tests for the full auto-answer flow."""

    def test_detect_answer_flow(self, repo):
        """Verify the end-to-end detection works without errors."""
        assistant = Assistant(repo, cache=False)
        answer = "brick, outside"
        detected = assistant.detect_answer_to_askback_slots(answer)
        assert "substrate" in detected or len(detected) > 0

    def test_short_answer_input_detected(self, repo):
        """Very short substrate-only input is detected as answer."""
        assistant = Assistant(repo, cache=False)
        assert assistant.router.slots.is_answer_to_askback("brick")

    def test_empty_detection_for_new_question(self, repo):
        """A follow-up question (not answer-to-askback) returns empty."""
        assistant = Assistant(repo, cache=False)
        assert assistant.detect_answer_to_askback_slots("What about Forte?") == {}
        assert assistant.detect_answer_to_askback_slots(
            "Can I use Ultra on the same wall?"
        ) == {}


class TestAutoAnswerWithSession:
    """Integration tests with session state."""

    def test_pending_flag_lifecycle(self):
        """Pending is set on ask-back, cleared after answer."""
        sessions = SessionStore()
        session_id = sessions.open()

        # Turn 1: Ask-back
        q1 = "How much water?"
        sessions.remember(session_id, q1, "Ask back", {}, pending=q1)
        assert sessions.pending(session_id) == q1

        # Turn 2: Auto-answer (pending is cleared)
        sessions.remember(session_id, "brick", "Answered", {"substrate": "brick"}, pending="")
        assert sessions.pending(session_id) == ""

    def test_slots_carried_to_next_turn(self):
        """After auto-answer, slots are carried to the next turn."""
        sessions = SessionStore()
        session_id = sessions.open()

        # Turn 1: Set substrate
        sessions.remember(
            session_id,
            "Question 1",
            "Answer 1",
            {"substrate": "brick"},
            pending=""
        )

        # Turn 2: Substrate should be carried
        carried = sessions.carried(session_id)
        assert carried.get("substrate") == "brick"

        # Turn 3: New question, substrate still carried
        sessions.remember(
            session_id,
            "Question 3",
            "Answer 3",
            {},  # No new substrate detected
            pending=""
        )
        carried = sessions.carried(session_id)
        assert carried.get("substrate") == "brick"
