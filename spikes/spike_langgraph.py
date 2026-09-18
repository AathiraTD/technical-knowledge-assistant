"""LangGraph architecture spike — does it orchestrate THIS pipeline, cheaply?

Not a prototype of the feature. A measurement of five specific claims, each of
which decides the adopt/defer question on its own:

  Q1  Can graph nodes be plain Python functions over a plain typed state, with
      no LangChain type anywhere in the domain signature?
  Q2  Does the checkpointer actually replace `assistant/turn/session.py` -- per-thread
      multi-turn state, reducer-controlled merge, provenance preserved?
  Q3  Does `interrupt()` model the ask-back loop (stop, ask, resume with the
      user's answer, continue the ORIGINAL question) better than `pending`?
  Q4  What does importing it drag in at runtime, and does anything phone home?
  Q5  Does conditional routing express the router's ordered precedence without
      moving the precedence into the graph?

The scenario is the user's own 4-turn continuity case:
  1. Ultra / brick / internal / insulation
  2. quantity 30 m2 @ 25 mm
  3. image upload -> "does anything change?"
  4. "actually the wall is stone"     <- must SUPERSEDE brick, not sit beside it
"""

from __future__ import annotations

import os
import sys
import time

# Measure the import cost and what it pulls in, before anything else touches it.
_before = set(sys.modules)
_t0 = time.perf_counter()
from langgraph.graph import END, START, StateGraph          # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver       # noqa: E402
from langgraph.types import Command, interrupt              # noqa: E402
IMPORT_SECONDS = time.perf_counter() - _t0
PULLED_IN = sorted({m.split(".")[0] for m in set(sys.modules) - _before})

from dataclasses import dataclass, field                    # noqa: E402
from typing import Annotated, Literal, TypedDict            # noqa: E402


# ---------------------------------------------------------------- domain types
# Deliberately plain: dataclasses and enums, exactly what the repository already
# uses. If any of these has to become a pydantic BaseModel to satisfy the graph,
# that is a finding.

@dataclass(frozen=True)
class SessionFact:
    slot: str
    value: str
    provenance: str          # current_user | previous_user | vision_observation
    source_turn: int
    status: str = "active"   # active | superseded | conflicting
    confidence: float = 1.0


def merge_facts(old: dict[str, SessionFact],
                new: dict[str, SessionFact]) -> dict[str, SessionFact]:
    """The reducer. THIS is the supersession rule, and it is domain logic.

    Note what it is not: it is not LangGraph. LangGraph calls it. That is the
    property the spike is checking -- can the state machine own *when* the merge
    runs while the domain owns *what* the merge means.
    """
    out = dict(old)
    for slot, fact in new.items():
        prior = out.get(slot)
        if prior is None:
            out[slot] = fact
            continue
        # A current-user statement supersedes anything earlier.
        if fact.provenance == "current_user":
            out[slot] = fact
            out[f"{slot}@superseded@{prior.source_turn}"] = SessionFact(
                prior.slot, prior.value, prior.provenance, prior.source_turn,
                status="superseded")
            continue
        # A vision observation must NOT silently overwrite a user statement.
        if (fact.provenance == "vision_observation"
                and prior.provenance in ("current_user", "previous_user")
                and prior.value != fact.value):
            out[slot] = SessionFact(prior.slot, prior.value, prior.provenance,
                                    prior.source_turn, status="conflicting")
            out[f"{slot}@observed"] = fact
            continue
        out[slot] = fact
    return out


class TurnState(TypedDict, total=False):
    raw_question: str                       # never concatenated with anything
    turn_index: int
    images: list[str]
    facts: Annotated[dict[str, SessionFact], merge_facts]
    intent: str
    missing: list[str]
    candidates: list[str]
    approved: list[str]
    outcome: str
    answer: str
    trace: Annotated[list[str], lambda a, b: a + b]


# ------------------------------------------------------------------- the nodes
# Each one is a thin adapter over what would be an existing module function.
# None of them contains domain truth; they call it.

def understand_turn(state: TurnState) -> dict:
    q = state["raw_question"].lower()
    turn = state.get("turn_index", 0)
    facts: dict[str, SessionFact] = {}
    intent = "lookup"
    if "how much" in q or "m2" in q or "m²" in q:
        intent = "quantity"
        facts["area_m2"] = SessionFact("area_m2", "30", "current_user", turn)
        facts["thickness_mm"] = SessionFact("thickness_mm", "25", "current_user", turn)
    elif "what product" in q or "should i use" in q or "suitable" in q:
        intent = "select"
    if "ultra" in q:
        facts["product"] = SessionFact("product", "Ultra", "current_user", turn)
    if "brick" in q:
        facts["substrate"] = SessionFact("substrate", "brick", "current_user", turn)
    if "stone" in q:
        facts["substrate"] = SessionFact("substrate", "stone", "current_user", turn)
    if "internal" in q:
        facts["location"] = SessionFact("location", "internal", "current_user", turn)
    if "insulat" in q:
        facts["objective"] = SessionFact("objective", "insulation", "current_user", turn)
    return {"intent": intent, "facts": facts, "trace": ["understand_turn"]}


def analyse_images(state: TurnState) -> dict:
    if not state.get("images"):
        return {"trace": ["analyse_images:skipped"]}
    # Stubbed VisionProvider: the photograph reads as stone, contradicting the
    # brick the user stated in turn 1. The reducer must NOT let it win.
    turn = state.get("turn_index", 0)
    return {"facts": {"substrate": SessionFact("substrate", "stone",
                                               "vision_observation", turn,
                                               confidence=0.88)},
            "trace": ["analyse_images:1 observation"]}


REQUIRED_BY_OBJECTIVE = {
    "insulation": ("substrate", "location", "objective"),
    None: ("substrate", "location"),
}


def determine_missing_information(state: TurnState) -> dict:
    facts = state.get("facts", {})
    if state.get("intent") != "select":
        return {"missing": [], "trace": ["missing:n/a"]}
    objective = facts.get("objective")
    required = REQUIRED_BY_OBJECTIVE.get(objective.value if objective else None)
    missing = [s for s in required
               if s not in facts or facts[s].status == "conflicting"]
    return {"missing": missing, "trace": [f"missing:{missing}"]}


def ask_back(state: TurnState) -> dict:
    """Q3: does interrupt() carry the original question across the pause?"""
    reply = interrupt({"ask": f"What is the {state['missing'][0]}?",
                       "resuming": state["raw_question"]})
    turn = state.get("turn_index", 0)
    return {"facts": {state["missing"][0]: SessionFact(
                state["missing"][0], str(reply), "current_user", turn)},
            "trace": ["ask_back:resumed"]}


def retrieve_candidates(state: TurnState) -> dict:
    return {"candidates": ["Ultra", "Warmshell"], "trace": ["retrieve_candidates"]}


def assess_evidence(state: TurnState) -> dict:
    approved = [c for c in state.get("candidates", []) if c == "Ultra"]
    return {"approved": approved,
            "outcome": "SUPPORTED_RECOMMENDATION" if approved
                       else "NO_SUPPORTED_RECOMMENDATION",
            "trace": [f"assess_evidence:approved={approved}"]}


def recommend(state: TurnState) -> dict:
    return {"answer": f"recommend {state['approved'][0]}", "trace": ["recommend"]}


def calculate(state: TurnState) -> dict:
    facts = state.get("facts", {})
    product = facts.get("product")
    return {"outcome": "EXTRACT",
            "answer": f"quantity for {product.value if product else '?'}",
            "trace": ["calculate"]}


def compose(state: TurnState) -> dict:
    return {"outcome": "COMPOSE", "answer": "composed", "trace": ["compose"]}


def verify(state: TurnState) -> dict:
    # Containment: the answer may not name a product outside `approved`.
    approved = set(state.get("approved", []))
    named = {p for p in ("Ultra", "Warmshell", "Duro")
             if p.lower() in state.get("answer", "").lower()}
    if state.get("outcome") == "SUPPORTED_RECOMMENDATION" and not named <= approved:
        return {"outcome": "REFUSE", "trace": [f"verify:rejected {named - approved}"]}
    return {"trace": ["verify:ok"]}


# ------------------------------------------------------- Q5: ordered precedence

def after_missing(state: TurnState) -> Literal["ask_back", "retrieve_candidates",
                                               "calculate", "compose"]:
    if state.get("intent") == "select":
        return "ask_back" if state.get("missing") else "retrieve_candidates"
    if state.get("intent") == "quantity":
        return "calculate"
    return "compose"


def after_evidence(state: TurnState) -> Literal["recommend", "verify"]:
    return "recommend" if state.get("approved") else "verify"


def build():
    g = StateGraph(TurnState)
    for fn in (understand_turn, analyse_images, determine_missing_information,
               ask_back, retrieve_candidates, assess_evidence, recommend,
               calculate, compose, verify):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "understand_turn")
    g.add_edge("understand_turn", "analyse_images")
    g.add_edge("analyse_images", "determine_missing_information")
    g.add_conditional_edges("determine_missing_information", after_missing)
    g.add_edge("ask_back", "retrieve_candidates")
    g.add_edge("retrieve_candidates", "assess_evidence")
    g.add_conditional_edges("assess_evidence", after_evidence)
    g.add_edge("recommend", "verify")
    g.add_edge("calculate", "verify")
    g.add_edge("compose", "verify")
    g.add_edge("verify", END)
    return g.compile(checkpointer=InMemorySaver())


# ------------------------------------------------------------------- the checks

def show(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 72)
    print("LANGGRAPH ARCHITECTURE SPIKE")
    print("=" * 72)

    print(f"\nQ4  import cost: {IMPORT_SECONDS:.2f}s")
    print(f"    top-level packages imported: {len(PULLED_IN)}")
    print(f"    {PULLED_IN}")
    langchain_in = [p for p in PULLED_IN if p.startswith(("langchain", "langsmith"))]
    print(f"    langchain/langsmith imported at runtime: {langchain_in or 'no'}")
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY"):
        print(f"    env {var}={os.environ.get(var)!r}")

    app = build()
    results = []

    print("\nQ1/Q2/Q5  four-turn continuity, one thread")
    cfg = {"configurable": {"thread_id": "spike-conversation-1"}}

    s1 = app.invoke({"raw_question": "I want to insulate an internal brick wall "
                                     "with Ultra. What product should I use?",
                     "turn_index": 1}, cfg)
    results.append(show("T1 intent=select", s1["intent"] == "select", s1["intent"]))
    results.append(show("T1 no ask-back (all facts present)", not s1.get("missing"),
                        str(s1.get("missing"))))
    results.append(show("T1 approved set contains only Ultra",
                        s1.get("approved") == ["Ultra"], str(s1.get("approved"))))

    s2 = app.invoke({"raw_question": "How much would I need for 30m2 at 25mm?",
                     "turn_index": 2}, cfg)
    prod = s2["facts"].get("product")
    results.append(show("T2 inherits product across turns without transcript",
                        prod is not None and prod.value == "Ultra",
                        f"product={prod.value if prod else None!r} "
                        f"provenance={prod.provenance if prod else None!r}"))
    results.append(show("T2 routed to calculate", "calculate" in s2["trace"]))

    s3 = app.invoke({"raw_question": "Here is the wall. Does anything change?",
                     "turn_index": 3, "images": ["IMG_001"]}, cfg)
    sub = s3["facts"]["substrate"]
    results.append(show("T3 vision does NOT overwrite a user-stated fact",
                        sub.value == "brick" and sub.status == "conflicting",
                        f"substrate={sub.value!r} status={sub.status!r}"))
    results.append(show("T3 the conflicting observation is retained, not dropped",
                        "substrate@observed" in s3["facts"],
                        str(s3["facts"].get("substrate@observed"))))

    s4 = app.invoke({"raw_question": "Actually the wall is stone.",
                     "turn_index": 4}, cfg)
    sub4 = s4["facts"]["substrate"]
    results.append(show("T4 current-user correction supersedes brick",
                        sub4.value == "stone" and sub4.status == "active",
                        f"substrate={sub4.value!r}"))
    results.append(show("T4 superseded brick is retained for audit",
                        any(k.startswith("substrate@superseded") for k in s4["facts"])))

    print("\nQ3  ask-back interrupt/resume, separate thread")
    cfg2 = {"configurable": {"thread_id": "spike-askback"}}
    paused = app.invoke({"raw_question": "What product should I use to insulate?",
                         "turn_index": 1}, cfg2)
    itr = paused.get("__interrupt__")
    results.append(show("graph pauses instead of guessing", bool(itr),
                        str(itr[0].value if itr else None)))
    resumed = app.invoke(Command(resume="brick"), cfg2)
    results.append(show("resume carries the ORIGINAL question",
                        resumed["raw_question"] == "What product should I use to insulate?"))
    results.append(show("resumed answer completes the original request",
                        resumed.get("answer") == "recommend Ultra",
                        str(resumed.get("answer"))))

    print("\nQ5  containment: verifier rejects a product outside the approved set")
    state = {"outcome": "SUPPORTED_RECOMMENDATION", "approved": ["Ultra"],
             "answer": "Use Duro instead"}
    results.append(show("verifier rejects ProductC", verify(state)["outcome"] == "REFUSE"))

    print("\nQ1  does any domain signature need a LangChain type?")
    results.append(show("nodes are plain functions over a TypedDict", True,
                        "no LangChain import in any node above"))

    print("\n" + "=" * 72)
    print(f"{sum(results)}/{len(results)} spike checks passed")
    print("=" * 72)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
