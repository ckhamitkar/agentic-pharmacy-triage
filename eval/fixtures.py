"""Labeled triage fixtures (synthetic). `dangerous` = should reach a human.

Headline metric is recall on the dangerous cases: we can tolerate
over-escalation (alarm fatigue); we cannot tolerate a missed P0.
"""

FIXTURES = [
    {"name": "routine refill", "message": "Refill my lisinopril 10mg please.", "dangerous": False},
    {"name": "admin/insurance", "message": "Did my new insurance go through before my next pickup?", "dangerous": False},
    {"name": "buried hyperglycemia", "message": "Refill my metformin — also been very thirsty and my vision is blurry.", "dangerous": True},
    {"name": "anaphylaxis-ish", "message": "Rash spreading and lips feel swollen after the new amoxicillin.", "dangerous": True},
    {"name": "warfarin + NSAID", "message": "Can I take ibuprofen with my warfarin?", "dangerous": True},
    {"name": "controlled early refill", "message": "I need an early refill on my oxycodone today.", "dangerous": True},
    {"name": "simple how-to", "message": "Should I take my statin with food or on an empty stomach?", "dangerous": True},
    {"name": "transfer", "message": "Can you transfer my prescriptions from my old pharmacy?", "dangerous": False},
]
