"""Wiring + policy tests — no network, no API key (a stub stands in for MedGemma)."""

import uuid

from triage.graph import build_graph
from triage.schemas import (
    Classification,
    CriticResult,
    Intake,
    RiskAssessment,
    RoutingDecision,
    decide_human_review,
)


class StubClient:
    """Returns canned, schema-valid objects so we can test graph flow offline."""

    def __init__(self, classification: Classification, risk: RiskAssessment, critic: CriticResult = None):
        self._cls = classification
        self._risk = risk
        self._critic = critic or CriticResult(missed_red_flag=False, concern="none", recommend_escalate=False)

    def complete_json(self, prompt, schema, system=None, retries=1):
        name = schema.__name__
        if name == "Intake":
            return Intake(patient_ref="x", medications=["metformin"], requested_action="refill", summary="refill request")
        if name == "Classification":
            return self._cls
        if name == "RiskAssessment":
            return self._risk
        if name == "RoutingDecision":
            return RoutingDecision(assignee="pharmacy_technician", rationale="r", draft_response="d")
        if name == "CriticResult":
            return self._critic
        raise AssertionError(f"unexpected schema {name}")


def _cfg():
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def test_decide_human_review_policy():
    safe_cls = Classification(category="refill", confidence=0.95, rationale="")
    safe_risk = RiskAssessment(severity="P3", red_flags=[], clinical_reasoning="", requires_escalation=False)
    assert decide_human_review(safe_cls, safe_risk) is False

    # clinical category -> human
    clinical = Classification(category="adverse_event", confidence=0.95, rationale="")
    assert decide_human_review(clinical, safe_risk) is True

    # low confidence -> human
    unsure = Classification(category="refill", confidence=0.5, rationale="")
    assert decide_human_review(unsure, safe_risk) is True

    # high severity -> human
    urgent = RiskAssessment(severity="P1", red_flags=["x"], clinical_reasoning="", requires_escalation=True)
    assert decide_human_review(safe_cls, urgent) is True

    # concrete red flag, but model labeled it routine and forgot to escalate -> still human
    leaky = RiskAssessment(severity="P2", red_flags=["blurred vision"], clinical_reasoning="", requires_escalation=False)
    assert decide_human_review(safe_cls, leaky) is True


def test_auto_path_no_interrupt():
    cls = Classification(category="refill", confidence=0.95, rationale="routine")
    risk = RiskAssessment(severity="P3", red_flags=[], clinical_reasoning="ok", requires_escalation=False)
    graph = build_graph(StubClient(cls, risk))
    result = graph.invoke({"raw_message": "refill my metformin", "trace": []}, _cfg())
    assert "__interrupt__" not in result
    assert result["final"]["human_reviewed"] is False
    assert result["final"]["final_assignee"] == "pharmacy_technician"


def test_dangerous_path_pauses_for_human():
    cls = Classification(category="adverse_event", confidence=0.95, rationale="reaction")
    risk = RiskAssessment(severity="P1", red_flags=["rash", "swelling"], clinical_reasoning="possible allergy", requires_escalation=True)
    graph = build_graph(StubClient(cls, risk))
    result = graph.invoke({"raw_message": "rash and swollen lips after amoxicillin", "trace": []}, _cfg())
    assert "__interrupt__" in result  # paused for pharmacist before finalize


def test_crisis_forced_to_p0_and_human():
    """Self-harm language is ALWAYS emergent (P0) and gated to a human, even when
    the model mis-rates severity (here the stub says P3 routine)."""
    cls = Classification(category="refill", confidence=0.95, rationale="routine refill")
    risk = RiskAssessment(severity="P3", red_flags=[], clinical_reasoning="looks routine", requires_escalation=False)
    graph = build_graph(StubClient(cls, risk))
    result = graph.invoke(
        {"raw_message": "refill my sertraline, honestly everyone would be better off without me", "trace": []},
        _cfg(),
    )
    assert "__interrupt__" in result  # crisis must pause for a human
    p = result["__interrupt__"][0].value
    assert p["severity"] == "P0"  # deterministic crisis floor overrode the model's P3
    assert any("suicid" in f.lower() or "self-harm" in f.lower() for f in p["red_flags"])


def test_critic_escalates_auto_path_to_human():
    """An auto-routed case (risk screen saw nothing) still gets pulled to a human
    if the downstream critic catches a miss — the critic has teeth."""
    cls = Classification(category="refill", confidence=0.95, rationale="routine")
    risk = RiskAssessment(severity="P3", red_flags=[], clinical_reasoning="ok", requires_escalation=False)
    critic = CriticResult(missed_red_flag=True, concern="possible hidden adverse event", recommend_escalate=True)
    graph = build_graph(StubClient(cls, risk, critic))
    result = graph.invoke({"raw_message": "refill, and by the way my heart's been racing", "trace": []}, _cfg())
    assert "__interrupt__" in result  # critic re-escalated a case the router auto-cleared


def test_critic_escalation_terminates_after_signoff():
    """The critic->HITL->critic path must not loop: once a human signs off, the
    critic can only annotate, so the graph reaches finalize."""
    from langgraph.types import Command

    cls = Classification(category="refill", confidence=0.95, rationale="routine")
    risk = RiskAssessment(severity="P3", red_flags=[], clinical_reasoning="ok", requires_escalation=False)
    critic = CriticResult(missed_red_flag=True, concern="x", recommend_escalate=True)  # flags every time
    graph = build_graph(StubClient(cls, risk, critic))
    cfg = _cfg()
    graph.invoke({"raw_message": "refill plus a symptom", "trace": []}, cfg)  # pauses at HITL
    final = graph.invoke(
        Command(resume={"action": "approve", "final_assignee": "pharmacist", "note": "ok"}), cfg
    )
    assert "__interrupt__" not in final  # terminated, no infinite loop
    assert final["final"]["human_reviewed"] is True
