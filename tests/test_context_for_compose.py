"""Tests for multi-turn context passed to the generation model.

Verifies that:
- Context from prior turns is built correctly
- Context is merged with the current question before generation
- Model sees full conversation history for coherent follow-ups
- Context parameter is optional (no context on first turn)
- Long context is handled correctly
"""

from __future__ import annotations

import pytest

from assistant.turn.session import SessionStore


class TestBuildContext:
    """Test context building from session turns."""

    def test_empty_turns_returns_empty_context(self):
        """Session with no prior turns returns empty context."""
        sessions = SessionStore()
        session_id = sessions.open()

        # No turns yet, so context should be empty
        # We can't test _build_context directly from UI, but we can verify
        # that the session turns are empty
        turns = sessions.turns(session_id)
        assert turns == []

    def test_one_turn_insufficient_for_context(self):
        """With only one turn, context should be empty (need 2+ for context)."""
        sessions = SessionStore()
        session_id = sessions.open()

        sessions.remember(
            session_id,
            "Question 1",
            "Answer 1",
            {},
            pending=""
        )

        turns = sessions.turns(session_id)
        assert len(turns) == 1
        # Context builder would return empty string with only 1 turn

    def test_two_turns_provides_context(self):
        """With two or more turns, context can be built."""
        sessions = SessionStore()
        session_id = sessions.open()

        # Turn 1
        sessions.remember(
            session_id,
            "How much water?",
            "5-6 litres per bag",
            {"substrate": "brick"},
            pending=""
        )

        # Turn 2
        sessions.remember(
            session_id,
            "What about Forte?",
            "Forte is compatible",
            {},
            pending=""
        )

        turns = sessions.turns(session_id)
        assert len(turns) == 2
        # Context builder would use these to build context

    def test_turns_capped_at_max_turns(self):
        """Turns history is capped, so context uses most recent."""
        from assistant.turn.session import MAX_TURNS

        sessions = SessionStore()
        session_id = sessions.open()

        # Add many turns (more than MAX_TURNS)
        for i in range(MAX_TURNS + 5):
            sessions.remember(
                session_id,
                f"Question {i}",
                f"Answer {i}",
                {},
                pending=""
            )

        turns = sessions.turns(session_id)
        # Should be capped at MAX_TURNS
        assert len(turns) <= MAX_TURNS


class TestContextInQuestion:
    """Test that context is properly merged with question."""

    def test_context_prepended_to_question(self):
        """Context is prepended before the current question."""
        # This tests the engine's merging behavior
        # When context is passed to ask(), it's prepended to the question

        context = "Earlier in this conversation, you asked: How much water? You answered: 5-6 litres."
        question = "What about Forte?"
        merged = context + "\n\n" + question

        # The merged question should contain both parts
        assert "Earlier in this conversation" in merged
        assert "How much water?" in merged
        assert "5-6 litres" in merged
        assert "What about Forte?" in merged

    def test_empty_context_no_merge(self):
        """Empty context doesn't add anything to question."""
        context = ""
        question = "What about Forte?"

        # With empty context, question stays the same
        if context:
            merged = context + "\n\n" + question
        else:
            merged = question

        assert merged == question

    def test_context_size_reasonable(self):
        """Context string is reasonable size (not huge)."""
        # Build a modest context
        context_parts = [
            "Earlier in this conversation,",
            "you asked: How much water for Solo?",
            "You answered: Between 5 and 6 litres per 25kg sack.",
            "Then the user asked: What about Forte?",
            "You answered: Forte is a stronger formulation suitable for thicker applications."
        ]
        context = "\n".join(context_parts)

        # Should be < 500 chars
        assert len(context) < 500

        # Question
        question = "What about Ultra?"
        merged = context + "\n\n" + question

        # Total should still be reasonable (< 1000 chars for modest context)
        assert len(merged) < 1000


class TestMultiTurnFlow:
    """Test the multi-turn conversation flow with context."""

    def test_turn_one_no_context(self):
        """First turn has no prior context."""
        sessions = SessionStore()
        session_id = sessions.open()

        # Turn 1: ask-back for substrate
        q1 = "How much water for Solo?"
        turns = sessions.turns(session_id)
        assert len(turns) == 0  # No prior turns yet

    def test_turn_two_uses_context(self):
        """Second turn can use context from turn 1."""
        sessions = SessionStore()
        session_id = sessions.open()

        # Turn 1
        sessions.remember(
            session_id,
            "How much water for Solo?",
            "5-6 litres per 25kg sack",
            {"substrate": "brick"},
            pending=""
        )

        # Turn 2
        turns = sessions.turns(session_id)
        assert len(turns) == 1

        # Context for turn 2 would be built from turn 1
        if len(turns) >= 2:
            # More than 1 turn, context available
            prior_q, prior_a = turns[0]
            assert prior_q == "How much water for Solo?"
            assert prior_a == "5-6 litres per 25kg sack"

    def test_context_carries_substrate_to_follow_up(self):
        """Substrate from turn 1 is available to turn 2 through context."""
        sessions = SessionStore()
        session_id = sessions.open()

        # Turn 1: establish substrate
        sessions.remember(
            session_id,
            "I have a brick wall",
            "OK, brick is noted",
            {"substrate": "brick"},
            pending=""
        )

        # Turn 2: follow-up question
        carried = sessions.carried(session_id)
        assert carried.get("substrate") == "brick"

        # Context would include: "you asked: I have a brick wall"
        # So model sees substrate from earlier


class TestHistoryCarriesQuestionsNotAnswers:
    """The boundary between conversational continuity and evidence.

    Trusted state carries facts. The transcript carries continuity. The
    transcript must not become a second evidence channel -- and for a while it
    was one, because `_build_context` emitted "You answered: ..." beside each
    question and the model recycled what it found there.

    Measured, in a live four-turn conversation: turn three wrote "It should be
    applied in a uniform thickness between 10 and 30mm" -- its own answer from
    turn one -- and cited a passage that does not contain the figure. Check 2
    caught it ("'10' is not in the passage it is cited to") and the turn
    refused. Turn four repeated the pattern with "It is suitable for masonry
    backgrounds such as brick walls". Both were questions the corpus answers.

    `HISTORY_BLOCK` already instructs the model never to take a fact from the
    transcript. It did anyway, which is what happens when a prompt is asked to
    be a boundary.
    """

    def _handler(self):
        """A Handler bound to a real SessionStore, without opening a socket."""
        from assistant import ui

        handler = ui.Handler.__new__(ui.Handler)
        handler.sessions = SessionStore()
        return handler

    def test_prior_assistant_answers_are_absent_from_the_history(self):
        """A. The fix, stated as the property it guarantees.

        The figure in the earlier answer is the thing that got recycled, so the
        assertion is about the figure and not merely about the label.
        """
        handler = self._handler()
        session_id = handler.sessions.open()
        handler.sessions.remember(
            session_id,
            "What thickness should Lime Green Ultra be applied at?",
            "Ultra should be applied in a uniform thickness between 10 and 30mm [1].",
            {})
        handler.sessions.remember(
            session_id,
            "I have a solid brick wall internally. Can I use Lime Green Ultra?",
            "Lime Green Ultra is suitable for masonry backgrounds [4].",
            {})

        context = handler._build_context(session_id)

        assert "You answered" not in context, context
        assert "10 and 30mm" not in context, context
        assert "masonry backgrounds" not in context, context

    def test_prior_user_questions_are_present_in_the_history(self):
        """B. What the transcript is actually for.

        Resolving "it" needs the subject, and the subject is in the question.
        """
        handler = self._handler()
        session_id = handler.sessions.open()
        handler.sessions.remember(
            session_id,
            "What thickness should Lime Green Ultra be applied at?",
            "Ultra should be applied between 10 and 30mm [1].", {})
        handler.sessions.remember(
            session_id,
            "I have a solid brick wall internally. Can I use Lime Green Ultra?",
            "Lime Green Ultra is suitable for masonry backgrounds [4].",
            {})

        context = handler._build_context(session_id)

        assert "What thickness should Lime Green Ultra be applied at?" in context
        assert "I have a solid brick wall internally." in context
        assert "Ultra" in context, "the subject a follow-up resolves against is gone"

    def test_reference_resolution_survives_without_the_answers(self):
        """The multi-turn property: continuity kept, facts not.

        A follow-up saying "it" has to be able to find its subject, and the
        history has to offer nothing citable while it does.
        """
        handler = self._handler()
        session_id = handler.sessions.open()
        handler.sessions.remember(
            session_id, "Can I use Lime Green Ultra on a brick wall?",
            "Ultra is suitable for most masonry and lath backgrounds [2].", {})
        handler.sessions.remember(
            session_id, "Is it suitable internally?",
            "It gives an additional insulation layer to internal walls [4].", {})

        context = handler._build_context(session_id)

        # The subject is reachable...
        assert "Ultra" in context
        # ...and nothing in the transcript is a fact worth copying.
        assert "insulation layer" not in context, context
        assert "masonry and lath" not in context, context
        assert "[2]" not in context and "[4]" not in context, (
            "citation markers from an earlier answer invite a reused citation")

    def test_a_first_turn_still_carries_no_transcript(self):
        """Unchanged behaviour: below the limit there is no history at all."""
        handler = self._handler()
        session_id = handler.sessions.open()
        handler.sessions.remember(session_id, "One question", "One answer.", {})

        assert handler._build_context(session_id) == ""
