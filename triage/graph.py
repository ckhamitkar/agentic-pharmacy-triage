"""The agentic triage pipeline as a LangGraph StateGraph.

    intake -> classify -> risk -> route --(needs human?)--> HITL --> critic -> finalize
                                              |                         ^
                                              +----------(auto)---------+

Design rationale (the "why"):
  * Why a graph of agents, not one call? It lets us SCALE triage beyond what
    humans alone can handle — each step is independently testable/evaluable,
    we can tier models per step, and (critically) we can pause for a human
    only where it matters.
  * Why human-in-the-loop? First, do no harm. A licensed human decides anything
    life-impacting (clinical questions, adverse events, controlled substances,
    high urgency, low confidence); the agent handles the routine rest.
  * Why a separate risk node? It screens the FULL message for red-flag symptoms
    independently of intent — so "refill my metformin, also thirsty & blurry"
    is caught as possible hyperglycemia even though the intent is just a refill.

The LLM is MedGemma (self-hosted), reached through triage.llm.MedGemmaClient.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .llm import MedGemmaClient
from .schemas import (
    Classification,
    CriticResult,
    Intake,
    RiskAssessment,
    RoutingDecision,
    TriageRecord,
    decide_human_review,
)


class TriageState(TypedDict, total=False):
    raw_message: str
    intake: Intake
    classification: Classification
    risk: RiskAssessment
    routing: RoutingDecision
    requires_human: bool
    human_decision: dict
    critic: CriticResult
    final: dict
    trace: Annotated[List[str], operator.add]


def build_graph(client: Optional[MedGemmaClient] = None):
    """Compile the triage graph. Pass a client (or one is built from env)."""
    llm = client or MedGemmaClient()

    def intake_node(state: TriageState) -> dict:
        out = llm.complete_json(
            f"Extract structured fields from this inbound pharmacy message:\n\n{state['raw_message']}",
            Intake,
        )
        return {"intake": out, "trace": [f"intake: {out.requested_action} (meds: {', '.join(out.medications) or 'none'})"]}

    def classify_node(state: TriageState) -> dict:
        out = llm.complete_json(
            "Classify the patient's intent. Categories: refill, new_rx_transfer, "
            "clinical_question, interaction_check, adverse_event, admin_billing, "
            "controlled_substance. Give a calibrated 0-1 confidence.\n\n"
            f"Message: {state['raw_message']}",
            Classification,
        )
        return {"classification": out, "trace": [f"classify: {out.category} (conf {out.confidence:.2f})"]}

    def risk_node(state: TriageState) -> dict:
        out = llm.complete_json(
            "You are a pharmacist doing a SAFETY SCREEN. Read the FULL message and look "
            "for clinical red flags — dangerous symptoms, adverse drug reactions, risky "
            "interactions, contraindications (e.g. medications unsafe in pregnancy), and "
            "any sign of a MENTAL-HEALTH CRISIS or SELF-HARM / SUICIDAL IDEATION — even if "
            "the patient is only asking for something routine like a refill. Any indication "
            "of suicidal thoughts, hopelessness, self-harm, or intent to harm is EMERGENT: "
            "set severity P0 and require escalation. Assign severity P0 (emergent) to P3 "
            "(routine) and decide if a prescriber must be looped in.\n\n"
            f"Message: {state['raw_message']}",
            RiskAssessment,
        )
        flags = ", ".join(out.red_flags) or "none"
        return {"risk": out, "trace": [f"risk: {out.severity}, red_flags: {flags}"]}

    def route_node(state: TriageState) -> dict:
        cls, risk = state["classification"], state["risk"]
        out = llm.complete_json(
            "Decide who should handle this and draft a first reply for THEM to review "
            "(never final clinical advice to the patient). Assignees: pharmacy_technician "
            "(admin/simple refills), pharmacist (clinical, interactions, counseling, "
            "controlled substances), prescriber (adverse events, red flags, dose changes).\n\n"
            f"Message: {state['raw_message']}\n"
            f"Category: {cls.category} (conf {cls.confidence:.2f})\n"
            f"Risk: {risk.severity}; red flags: {risk.red_flags}; escalate: {risk.requires_escalation}",
            RoutingDecision,
        )
        needs_human = decide_human_review(cls, risk)
        reason = "first-do-no-harm: life-impacting / uncertain → pharmacist sign-off" if needs_human else "routine → auto-routed"
        return {
            "routing": out,
            "requires_human": needs_human,
            "trace": [f"route: -> {out.assignee} ({reason})"],
        }

    def hitl_node(state: TriageState) -> dict:
        """Pause for a licensed pharmacist. The graph cannot finalize a clinical
        decision without passing through here — the hard stop is structural."""
        decision = interrupt(
            {
                "message": state["raw_message"],
                "proposed_assignee": state["routing"].assignee,
                "category": state["classification"].category,
                "severity": state["risk"].severity,
                "red_flags": state["risk"].red_flags,
                "draft_response": state["routing"].draft_response,
                "reason": "Pharmacist review required before this triage is finalized.",
            }
        )
        return {"human_decision": decision, "trace": [f"HITL: pharmacist {decision.get('action', 'reviewed')}"]}

    def critic_node(state: TriageState) -> dict:
        out = llm.complete_json(
            "Adversarially review this triage decision. Did it miss any clinical danger "
            "in the original message? Be skeptical.\n\n"
            f"Message: {state['raw_message']}\n"
            f"Severity: {state['risk'].severity}; red flags: {state['risk'].red_flags}\n"
            f"Routed to: {state['routing'].assignee}",
            CriticResult,
        )
        return {"critic": out, "trace": [f"critic: missed_red_flag={out.missed_red_flag}"]}

    def finalize_node(state: TriageState) -> dict:
        routing, cls, risk = state["routing"], state["classification"], state["risk"]
        decision = state.get("human_decision") or {}
        final_assignee = decision.get("final_assignee", routing.assignee)
        record = TriageRecord(
            summary=state["intake"].summary,
            category=cls.category,
            confidence=cls.confidence,
            severity=risk.severity,
            red_flags=risk.red_flags,
            final_assignee=final_assignee,
            human_reviewed=bool(state.get("human_decision")),
            human_note=decision.get("note", ""),
            draft_response=routing.draft_response,
            audit_trail=state.get("trace", []) + ["finalize: triage record sealed"],
        )
        return {"final": record.model_dump()}

    def needs_human(state: TriageState) -> str:
        return "hitl" if state.get("requires_human") else "critic"

    g = StateGraph(TriageState)
    g.add_node("intake", intake_node)
    g.add_node("classify", classify_node)
    g.add_node("risk", risk_node)
    g.add_node("route", route_node)
    g.add_node("hitl", hitl_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "classify")
    g.add_edge("classify", "risk")
    g.add_edge("risk", "route")
    g.add_conditional_edges("route", needs_human, {"hitl": "hitl", "critic": "critic"})
    g.add_edge("hitl", "critic")
    g.add_edge("critic", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())
