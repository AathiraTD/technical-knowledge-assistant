"""The policy gate must survive paraphrase and word order.

Reported from `feat/e2e-submission`, which found three semantically equivalent
phrasings the gate let through. All three were real and all three are fixed
here, at the routing layer, in `config/routing.json`:

  * **structural** — "is it safe" matched, "is the house safe" did not. Only the
    literal pronoun was covered, so every phrasing naming the thing being asked
    about fell through.
  * **health** — the pattern wanted the verb *after* the word "eye", so "I got
    lime plaster in my eye" missed. Fixed earlier on this branch, for the same
    reason and with the same consequence: a 142-second generation where an
    instant referral was written and waiting.
  * **compliance** — "does *it* comply" matched; "does my Warmshell build
    comply" did not, and neither did the reversed order, "complies with Part L".

Why this is a routing problem and not a language problem. The gate runs before
slot detection, before query understanding and before anything is embedded, and
that ordering is the whole reason a price question costs milliseconds. Handing
these phrasings to a model to classify would move a deterministic control into
the one component this system does not let decide anything, and it would buy
paraphrase coverage with a network round trip on the fastest path in the
system. A regular expression that reads the way people write is cheaper and can
be reviewed by the technical team, which is the standard the rest of the
routing table is held to.

The second half of this file is the part that stops the first half being a
blunt instrument. A gate that fires on everything has moved the failure rather
than fixed it: it turns answerable technical questions into referrals, which is
the over-refusal the evaluation measures separately and deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answering.router import PolicyGate                        # noqa: E402


# ------------------------------------------------- the three reported misses

STRUCTURAL = [
    # The exact phrasing reported.
    "My gable wall has a 10 mm crack, is the house safe?",
    "Is the house safe?",
    # The same question with the subject varied, because the defect was that
    # only one subject was covered and a second literal would repeat it.
    "Is this wall safe?",
    "Is my chimney safe?",
    "Is the building safe to live in?",
    "Is the masonry safe?",
    # Asked as a consequence rather than as a state.
    "Will the wall collapse?",
    "Will it fall down?",
    # Already covered, kept so a rewrite cannot quietly drop them.
    "Is it safe?",
    "Is this a load bearing wall?",
]

HEALTH = [
    "I got lime plaster in my eye, what should I do?",
    "Lime splashed into my eyes",
    "Some render went in my eye",
    "My eyes got some lime in them",
    "I swallowed some of the powder",
]

COMPLIANCE = [
    # The reported miss: a subject that is not "it" or "this".
    "Does my Warmshell build comply with Part L?",
    # The reported reversed order.
    "Can you confirm my Warmshell build complies with Part L?",
    "Is my build compliant with Part L?",
    "Will this comply with the building regulations?",
    # Already covered.
    "Does this comply with Part L?",
    "Can you sign off my specification?",
]


@pytest.mark.parametrize("question", STRUCTURAL)
def test_a_structural_question_routes_whatever_it_calls_the_building(question):
    matched = PolicyGate().match(question)
    assert matched and matched[0] == "structural_judgement", question


@pytest.mark.parametrize("question", HEALTH)
def test_a_health_question_routes_whatever_order_it_is_written_in(question):
    matched = PolicyGate().match(question)
    assert matched and matched[0] == "health", question


@pytest.mark.parametrize("question", COMPLIANCE)
def test_a_compliance_question_routes_whatever_its_subject(question):
    matched = PolicyGate().match(question)
    assert matched and matched[0] == "compliance_signoff", question


def test_a_question_naming_both_takes_one_of_the_two_and_neither_is_answered():
    """Two policy topics in one sentence is still a referral, not an answer.

    Which of the two wins is dictionary order and does not matter: both refuse
    to judge and both name a person. What would matter is the question reaching
    retrieval, and this is the assertion that says it cannot.
    """
    matched = PolicyGate().match(
        "Can you certify this wall is structurally safe and "
        "Building Regulations compliant?")
    assert matched and matched[0] in ("structural_judgement",
                                      "compliance_signoff")


# ----------------------------------------- and not at the cost of everything

# Ordinary technical questions, several of them using the very words the
# widened patterns look for -- safe, comply, compatible. A gate that catches
# these has not become safer; it has stopped answering the questions the corpus
# exists to answer.
TECHNICAL = [
    "What is the coverage of Solo Onecoat?",
    "How much water does Solo Onecoat need per bag?",
    "What thickness should Lime Green Ultra be applied at?",
    "What temperature range can Duro be applied in?",
    "Which finish coats are compatible with Forte render, "
    "including the hardening time required first?",
    "What preparation does a dense smooth masonry background need "
    "before applying Lime Green Ultra?",
    "My external lime render is showing patchy colour after drying. "
    "What could be causing it?",
    "Why is my lime render cracking?",
    "What plaster should I use?",
    "I'm rendering an old masonry wall in a very exposed location. How thick "
    "should the lime render be, and what preparation does the background need?",
    "Is Natural Finish compatible with Duro, including the temperature range "
    "Duro needs?",
    "Can I use Lime Green Solo directly over old gypsum plaster, or do I need "
    "Solo Primer first?",
    "Can Solo Onecoat be used on an external wall?",
    # Uses "safe" about a product rather than about a building, which is the
    # distinction the structural pattern turns on: it requires a building noun.
    "Is this product safe to use indoors?",
]


@pytest.mark.parametrize("question", TECHNICAL)
def test_an_ordinary_technical_question_is_left_alone(question):
    assert PolicyGate().match(question) is None, question


def test_the_gate_is_pure_pattern_matching_and_reaches_nothing(monkeypatch):
    """No retrieval, no embedding, no generation — by construction.

    `PolicyGate` holds compiled patterns and a configuration dictionary and has
    no repository, no retriever and no model client to call. That is what makes
    the referral instant, and it is asserted here rather than assumed because
    the whole value of the gate is what it does *not* do.
    """
    import assistant.ollama as ollama

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the policy gate reached the model")

    monkeypatch.setattr(ollama, "generate", forbidden)
    monkeypatch.setattr(ollama, "embed_one", forbidden)
    monkeypatch.setattr(ollama, "embed", forbidden)

    gate = PolicyGate()
    for question in STRUCTURAL + HEALTH + COMPLIANCE:
        matched = gate.match(question)
        assert matched, question
        topic, spec = matched
        # The referral is authored text, read from configuration. Nothing here
        # is generated, so there is nothing to check afterwards.
        assert spec["referral"] and spec["next_step"]
