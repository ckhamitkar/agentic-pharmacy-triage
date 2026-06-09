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

    def __init__(self, classification: Classification, risk: RiskAssessment):
        self._cls = classification
        self._risk = risk

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
            return CriticResult(missed_red_flag=False, concern="none", recommend_escalate=False)
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
