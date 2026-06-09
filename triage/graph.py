"""The agentic triage pipeline as a LangGraph StateGraph.

    intake -> classify -> risk -> route --(needs human?)--> HITL --> critic -> finalize
                                              |                ^        |
                                              +----(auto)------|--------+
                                                               |        |
                                              (critic caught a miss?)---+

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
    detect_crisis,
    find_controlled_substances,
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
        meds = ", ".join(out.medications) or "none"
        return {"intake": out, "trace": [f"Read the message — request: {out.requested_action}; medications: {meds}"]}

    def classify_node(state: TriageState) -> dict:
        out = llm.complete_json(
            "Classify the patient's intent. Categories: refill, new_rx_transfer, "
            "clinical_question, interaction_check, adverse_event, admin_billing, "
            "controlled_substance. Give a calibrated 0-1 confidence.\n\n"
            f"Message: {state['raw_message']}",
            Classification,
        )
        return {"classification": out, "trace": [f"Identified intent: {out.category.replace('_', ' ')}"]}

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
        # Deterministic crisis floor: self-harm / suicidal ideation is ALWAYS emergent.
        # The model caught the red flag here but mis-rated severity (P2); we never
        # leave the most dangerous classification to a stochastic call.
        if detect_crisis(state["raw_message"]):
            out.severity = "P0"
            out.requires_escalation = True
            if not any(("suicid" in f.lower() or "self-harm" in f.lower() or "self harm" in f.lower()) for f in out.red_flags):
                out.red_flags.append("Self-harm / suicidal ideation")
        flags = ", ".join(out.red_flags) or "none"
        sev_label = {"P0": "emergent", "P1": "urgent", "P2": "elevated", "P3": "routine"}.get(out.severity, out.severity)
        return {"risk": out, "trace": [f"Safety screen: {out.severity} · {sev_label}; red flags: {flags}"]}

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
        # Deterministic guard: a named controlled substance always sees a human,
        # even if the classifier labeled it a routine refill.
        controlled = find_controlled_substances(state["raw_message"])
        if controlled:
            note = f"Controlled substance ({', '.join(controlled)}) — pharmacist dispensing/early-refill review"
            if note not in risk.red_flags:
                risk.red_flags.append(note)
            # A controlled substance must never default to a technician.
            if out.assignee == "pharmacy_technician":
                out.assignee = "pharmacist"
        needs_human = decide_human_review(cls, risk) or bool(controlled)
        reason = "flagged as life-impacting, so it needs a pharmacist's sign-off" if needs_human else "routine, so it was handled automatically"
        return {
            "routing": out,
            "risk": risk,
            "requires_human": needs_human,
            "trace": [f"Routed to {out.assignee.replace('_', ' ')} — {reason}"],
        }

    def hitl_node(state: TriageState) -> dict:
        """Pause for a licensed pharmacist. The graph cannot finalize a clinical
        decision without passing through here — the hard stop is structural.

        Two ways in: the router sends life-impacting/uncertain cases here, OR the
        downstream critic catches a danger the risk screen missed and escalates."""
        crit = state.get("critic")
        if crit and crit.missed_red_flag:
            reason = f"Critic caught a possible missed danger ({crit.concern}) — pharmacist review required."
        else:
            reason = "Pharmacist review required before this triage is finalized."
        decision = interrupt(
            {
                "message": state["raw_message"],
                "proposed_assignee": state["routing"].assignee,
                "category": state["classification"].category,
                "severity": state["risk"].severity,
                "red_flags": state["risk"].red_flags,
                "draft_response": state["routing"].draft_response,
                "reason": reason,
            }
        )
        return {"human_decision": decision, "trace": [f"Pharmacist {decision.get('action', 'reviewed')} the case"]}

    def critic_node(state: TriageState) -> dict:
        out = llm.complete_json(
            "You are the LAST safety check before this triage is finalized. The risk "
            "screen already caught these red flags: "
            f"{state['risk'].red_flags or 'none'}.\n"
            "Your ONLY job: is there a CONCRETE clinical danger present in the message "
            "that is NOT already on that list — a specific symptom, a dangerous drug "
            "interaction, an adverse reaction, or a self-harm / crisis signal?\n"
            "Set missed_red_flag=True ONLY if you can name such a specific, concrete "
            "danger. A routine refill, an admin / billing / insurance question, or any "
            "message with no clinical symptoms is NOT a missed red flag — set "
            "missed_red_flag=False and pass it. Do not invent risks; when the message "
            "is clearly routine, do not flag it.\n\n"
            f"Message: {state['raw_message']}\n"
            f"Routed to: {state['routing'].assignee}",
            CriticResult,
        )
        # If the critic names a CONCRETE missed danger AND no human has reviewed yet,
        # it escalates back to the HITL gate (defense in depth). Once a human has
        # signed off, the critic can only annotate — it cannot loop the case back.
        escalating = out.missed_red_flag and not state.get("human_decision")
        if out.missed_red_flag:
            line = "Final safety check: caught a possible missed danger" + (" → escalating to pharmacist" if escalating else "")
        else:
            line = "Final safety check: nothing missed — cleared"
        return {"critic": out, "trace": [line]}

    def critic_route(state: TriageState) -> str:
        """After the critic: a flagged miss with no prior human review goes to the
        HITL gate; anything else finalizes. The `human_decision` guard makes the
        critic->hitl->critic path terminate (a human signs off at most once)."""
        crit = state["critic"]
        if crit.missed_red_flag and not state.get("human_decision"):
            return "hitl"
        return "finalize"

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
            audit_trail=state.get("trace", []) + ["Triage finalized and recorded"],
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
    g.add_conditional_edges("critic", critic_route, {"hitl": "hitl", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())
