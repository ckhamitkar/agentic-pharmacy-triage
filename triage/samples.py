"""Synthetic sample messages — 100% fictional, no real patient data (no PHI).

These power the demo's "canned message" buttons so a public Space can run on a
fixed, bounded set of inputs instead of unbounded free text on the host's key.
"""

SAMPLES = {
    "Routine refill": (
        "Hi, this is Maria Gonzalez. Can I get a refill on my lisinopril 10mg? "
        "Same pharmacy as always. Thanks!"
    ),
    "Buried red flag (refill + symptoms)": (
        "Please refill my metformin 500mg. Also, I've been really thirsty all week "
        "and my vision has gone blurry the last couple of days — figured I'd mention it."
    ),
    "Drug interaction question": (
        "Quick question — I just started warfarin. Is it okay to take ibuprofen for "
        "a headache, or will that be a problem with the blood thinner?"
    ),
    "Adverse event": (
        "I started the new amoxicillin yesterday and now I have a rash spreading on my "
        "chest and my lips feel a little swollen. Should I keep taking it?"
    ),
    "Controlled substance early refill": (
        "I need an early refill on my oxycodone — I ran out sooner than expected this month. "
        "Can you fill it today?"
    ),
    "Insurance / admin": (
        "My copay was higher than usual last time. Can someone check whether my new "
        "insurance went through before my next pickup?"
    ),
    "Pregnancy contraindication": (
        "I just found out I'm 6 weeks pregnant. I take lisinopril for blood pressure "
        "and isotretinoin for acne — are those okay to keep taking?"
    ),
    "Pediatric dosing": (
        "My son is 4 years old and weighs about 35 lbs. He has a fever and the bottle "
        "of children's ibuprofen isn't clear — how much should I give him and how often?"
    ),
    "Crisis buried in a refill": (
        "Can I refill my sertraline? Honestly I stopped taking it a couple weeks ago "
        "and lately I just feel like there's no point and everyone would be better off "
        "without me."
    ),
}
