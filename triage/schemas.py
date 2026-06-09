"""Pydantic schemas for the pharmacy triage pipeline.

Every LLM node returns one of these via structured output, so the graph passes
validated objects between steps — no fragile string parsing.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# --- controlled vocabularies -------------------------------------------------

Category = Literal[
    "refill",
    "new_rx_transfer",
    "clinical_question",
    "interaction_check",
    "adverse_event",
    "admin_billing",
    "controlled_substance",
]

Severity = Literal["P0", "P1", "P2", "P3"]  # P0 = emergent, P3 = routine

Assignee = Literal["pharmacy_technician", "pharmacist", "prescriber"]

# Categories that are clinical in nature — never auto-handled without a licensed human.
CLINICAL_CATEGORIES = {
    "clinical_question",
    "interaction_check",
    "adverse_event",
    "controlled_substance",
}

CONFIDENCE_FLOOR = 0.75  # below this, a human reviews regardless of category


# --- per-node outputs --------------------------------------------------------


class Intake(BaseModel):
    """Structured extraction of a raw inbound pharmacy message."""

    patient_ref: str = Field(description="Patient name or identifier, if present; else 'unknown'.")
    medications: List[str] = Field(default_factory=list, description="Medications mentioned.")
    requested_action: str = Field(description="What the patient is asking for, in a short phrase.")
    summary: str = Field(description="One-sentence neutral summary of the message.")


class Classification(BaseModel):
    """Intent classification with calibrated confidence."""

    category: Category
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence in the category.")
    rationale: str = Field(description="Brief reason for the chosen category.")


class RiskAssessment(BaseModel):
    """Clinical safety screen — runs on the FULL message, independent of intent.

    This is what catches a red-flag symptom buried inside a routine-looking request.
    """

    severity: Severity = Field(description="Clinical urgency: P0 emergent ... P3 routine.")
    red_flags: List[str] = Field(
        default_factory=list,
        description="Concrete clinical red flags found (symptoms, dangerous interactions, etc.).",
    )
    clinical_reasoning: str = Field(description="Why this severity; cite the signals.")
    requires_escalation: bool = Field(
        description="True if a prescriber/physician should be looped in (adverse event, red-flag symptom, dose change)."
    )


class RoutingDecision(BaseModel):
    """Where the message should go, and a draft first response."""

    assignee: Assignee
    rationale: str = Field(description="Why this assignee.")
    draft_response: str = Field(
        description="Draft first reply for the assignee to review. Never final clinical advice."
    )


class CriticResult(BaseModel):
    """Adversarial safety review of the triage decision."""

    missed_red_flag: bool = Field(description="True if the triage plausibly missed a clinical danger.")
    concern: str = Field(description="The concern, or 'none'.")
    recommend_escalate: bool = Field(description="True if the critic thinks this should be escalated/human-reviewed.")


class HumanDecision(BaseModel):
    """The pharmacist's decision at the HITL checkpoint."""

    action: Literal["approve", "override"]
    final_assignee: Assignee
    note: str = ""


class TriageRecord(BaseModel):
    """The final, auditable triage output."""

    summary: str
    category: Category
    confidence: float
    severity: Severity
    red_flags: List[str]
    final_assignee: Assignee
    human_reviewed: bool
    human_note: str = ""
    draft_response: str
    audit_trail: List[str] = Field(default_factory=list)


def decide_human_review(classification: Classification, risk: RiskAssessment) -> bool:
    """The HITL trigger rule — recall-first.

    A human reviews if ANY of:
      - the category is clinical (incl. controlled substances),
      - severity is emergent/urgent (P0/P1),
      - the risk screen wants escalation,
      - the classifier was not confident.

    We bias toward over-escalation: a missed adverse event costs more than alarm fatigue.
    """
    return (
        classification.category in CLINICAL_CATEGORIES
        or risk.severity in {"P0", "P1"}
        or risk.requires_escalation
        or classification.confidence < CONFIDENCE_FLOOR
    )
