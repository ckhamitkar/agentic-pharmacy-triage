"""Gradio UI for the agentic pharmacy-triage demo (Hugging Face Spaces entrypoint).

Safe-by-default for a public Space: pick a canned synthetic message (bounded
load on the model endpoint), or override the endpoint/key to use your own.
"""

from __future__ import annotations

import uuid

import gradio as gr
from langgraph.types import Command

from triage import MedGemmaError, MedGemmaClient, build_graph
from triage.llm import DEFAULT_URL
from triage.samples import SAMPLES

ASSIGNEES = ["pharmacy_technician", "pharmacist", "prescriber"]


SEV_LABEL = {"P0": "emergent", "P1": "urgent", "P2": "elevated", "P3": "routine"}

EMPTY_RESULT = (
    '<div class="rc-empty">👆 Click a sample card above to see the triage result appear here.</div>'
)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _sev(s) -> str:
    label = SEV_LABEL.get(s, "")
    return f"{_esc(s)} · {label}" if label else _esc(s)


# Severity color-coding: a meaningful, at-a-glance acuity scale (fg, bg).
SEV_META = {
    "P0": ("emergent", "#b42318", "#fee4e2"),
    "P1": ("urgent", "#b54708", "#fef0c7"),
    "P2": ("elevated", "#854d0e", "#fef9c3"),
    "P3": ("routine", "#067647", "#dcfce7"),
}


def _sev_badge(s) -> str:
    label, fg, bg = SEV_META.get(s, ("", "#475569", "#f1f5f9"))
    txt = f"{_esc(s)} · {label}" if label else _esc(s)
    return f'<span class="sev-badge" style="color:{fg};background:{bg}">{txt}</span>'


def _meta_rows(pairs) -> str:
    """Key/value rows as DIVS (not a <table>) so Gradio's table CSS can't grey them."""
    rows = "".join(
        f'<div class="meta-row"><span class="meta-k">{_esc(k)}</span>'
        f'<span class="meta-v">{v}</span></div>'
        for k, v in pairs
    )
    return f'<div class="meta">{rows}</div>'


def _pretty(s) -> str:
    """Human-friendly label: pharmacy_technician -> 'pharmacy technician'."""
    return _esc(str(s).replace("_", " "))


_ST = {"ok": "st st-ok", "wait": "st st-wait", "review": "st st-review", "err": "st st-err"}


def _status(kind, text) -> str:
    """Render a status line as a styled pill that matches the card aesthetic."""
    if kind == "muted":
        return f'<span class="st-muted">{_esc(text)}</span>'
    return f'<span class="{_ST.get(kind, "st st-ok")}">{_esc(text)}</span>'


def _trace_html(state) -> str:
    lines = state.get("trace", [])
    if not lines:
        return ""
    items = "".join(f"<li>{_esc(ln)}</li>" for ln in lines)
    return f'<div class="rc-trail"><span class="rc-lbl">Audit trail</span><ol>{items}</ol></div>'


def _result_card_html(final) -> str:
    """Readable result card (replaces the raw-JSON dump for normal reviewers)."""
    flags = final.get("red_flags") or []
    chips = (
        "".join(f'<span class="rc-flag">🚩 {_esc(f)}</span>' for f in flags)
        if flags else '<span class="rc-ok">✓ none</span>'
    )
    reviewed = (
        "🧑‍⚕️ pharmacist" if final.get("human_reviewed")
        else "automatic — no human needed"
    )
    note = final.get("human_note") or ""
    rows = [
        ("Category", _pretty(final.get("category", ""))),
        ("Severity", _sev_badge(final.get("severity", ""))),
        ("Assigned to", _pretty(final.get("final_assignee", ""))),
        ("Reviewed by", reviewed),
    ]
    if note:
        rows.append(("Note", _esc(note)))
    return (
        '<div class="rc-card"><div class="rc-title">Triage result</div>'
        f"{_meta_rows(rows)}"
        f'<div class="rc-flags"><span class="rc-lbl">Red flags</span><br>{chips}</div>'
        '<div class="rc-draft"><span class="rc-lbl">Draft reply (for staff to review &amp; send)</span>'
        f'<blockquote>{_esc(final.get("draft_response", ""))}</blockquote></div></div>'
    )


def _review_html(p) -> str:
    """Render the pharmacist-review panel as styled HTML so the *why* — the
    escalation reason and the red flags — visually pops."""
    flags = p.get("red_flags") or []
    chips = (
        "".join(f'<span class="dnh-flag">🚩 {_esc(f)}</span>' for f in flags)
        if flags else '<span class="dnh-noflag">none</span>'
    )
    meta = _meta_rows([
        ("Category", _pretty(p["category"])),
        ("Severity", _sev_badge(p["severity"])),
        ("Proposed assignee", _pretty(p["proposed_assignee"])),
    ])
    return (
        f'<div class="dnh-alert">⏸ {_esc(p["reason"])}</div>'
        '<div class="dnh-flags-wrap"><span class="dnh-lbl">Red flags</span><br>'
        f"{chips}</div>"
        f"{meta}"
        '<div class="dnh-draft"><span class="dnh-lbl">Draft reply (for staff to review)</span>'
        f'<blockquote>{_esc(p["draft_response"])}</blockquote></div>'
    )


def run_triage(message, url, key):
    message = (message or "").strip()
    blank = (None, None)
    if not message:
        return (_status("muted", "Enter or pick a message."), EMPTY_RESULT, None, "", gr.update(visible=False), "", gr.update(), *blank)
    graph = build_graph(MedGemmaClient(url=url or None, api_key=key or None))
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    try:
        result = graph.invoke({"raw_message": message, "trace": []}, config)
    except MedGemmaError as e:
        return (_status("err", f"⚠️ MedGemma error: {e}"), EMPTY_RESULT, None, "", gr.update(visible=False), "", gr.update(), *blank)

    if "__interrupt__" in result:
        p = result["__interrupt__"][0].value
        info = _review_html(p)
        return (
            _status("review", "⏸ Pharmacist review required — first, do no harm."),
            EMPTY_RESULT,
            None,
            _trace_html(result),
            gr.update(visible=True),
            info,
            gr.update(value=p["proposed_assignee"]),
            graph,
            config,
        )

    final = result["final"]
    return (
        _status("ok", "✅ Auto-triaged — routine, no human review needed."),
        _result_card_html(final),
        final,
        _trace_html(result),
        gr.update(visible=False),
        "",
        gr.update(),
        graph,
        config,
    )


def resolve(action, assignee, note, graph, config):
    # Outputs: status (top), finalize_status (in-panel, by the buttons),
    #          result_card, result_json, trace, approval(visibility)
    if graph is None or config is None:
        yield (_status("muted", "No pending review."), "", EMPTY_RESULT, None, "", gr.update(visible=False))
        return
    # Immediate feedback RIGHT AT THE BUTTONS: finalizing runs the critic (a ~1s
    # model call), so signal at the click point instead of off-screen at the top.
    yield (
        gr.update(),
        _status("wait", f"⏳ Finalizing — running the final safety check ({action})…"),
        gr.update(), gr.update(), gr.update(), gr.update(),
    )
    # Approve = accept the agent's proposed handler (finalize falls back to it when
    # no final_assignee is supplied). Override = use the pharmacist's own selection.
    decision = {"action": action, "note": note or ""}
    if action == "override":
        decision["final_assignee"] = assignee
    result = graph.invoke(Command(resume=decision), config)
    final = result["final"]
    yield (
        _status("ok", f"✅ Finalized by pharmacist ({action})."),
        "",
        _result_card_html(final),
        final,
        _trace_html(result),
        gr.update(visible=False),
    )


EMOJI = {
    "Routine refill": "🔁",
    "Buried red flag (refill + symptoms)": "🩺",
    "Drug interaction question": "⚠️",
    "Adverse event": "🚨",
    "Controlled substance early refill": "🔒",
    "Insurance / admin": "📋",
    "Pregnancy contraindication": "🤰",
    "Pediatric dosing": "🧒",
    "Crisis buried in a refill": "🆘",
}
DESC = {
    "Routine refill": "Simple lisinopril refill, no symptoms",
    "Buried red flag (refill + symptoms)": "Refill request hiding thirst + blurry vision",
    "Drug interaction question": "Warfarin + ibuprofen — safe together?",
    "Adverse event": "New antibiotic, spreading rash + swollen lips",
    "Controlled substance early refill": "Early oxycodone refill request",
    "Insurance / admin": "Copay / insurance question — non-clinical",
    "Pregnancy contraindication": "6 weeks pregnant on lisinopril + isotretinoin",
    "Pediatric dosing": "Ibuprofen dose for a 4-year-old",
    "Crisis buried in a refill": "Sertraline refill with hopelessness signals",
}


def _card_label(name: str) -> str:
    return f"{EMOJI.get(name, '💊')}  {name}\n{DESC.get(name, '')}"


DNH_CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#dnh-banner {
  display: flex; align-items: center; gap: 16px;
  padding: 16px 20px; margin: 6px 0 12px;
  border-radius: 14px; border: 1px solid #cfe0fb;
  background: linear-gradient(100deg, #eff6ff 0%, #f5f9ff 100%);
  box-shadow: 0 1px 3px rgba(37,99,235,.08);
}
#dnh-banner .dnh-mark { font-size: 34px; line-height: 1; }
#dnh-banner .dnh-title {
  font-weight: 800; font-size: 20px; color: #3b82f6;
  letter-spacing: .3px; margin-bottom: 2px;
}
#dnh-banner .dnh-sub { font-size: 13px; color: #475569; }
/* agentic pipeline strip */
.pipe { display: flex; flex-wrap: wrap; gap: 7px; align-items: center;
  margin: 2px 0 18px; }
.pipe .step { background: #eef4ff; color: #334155; border: 1px solid #dce8fb;
  border-radius: 999px; padding: 5px 13px; font-size: 13px; font-weight: 600; }
.pipe .gate { background: #fef2f2; color: #b91c1c; border: 1px solid #fbcaca; }
.pipe .arr { color: #94a3b8; font-weight: 700; font-size: 13px; }
/* sample cards */
.sample-card {
  white-space: pre-line !important; text-align: left !important;
  align-items: flex-start !important; justify-content: flex-start !important;
  min-height: 86px; padding: 13px 15px !important;
  border: 1px solid #e2e8f0 !important; border-radius: 14px !important;
  background: linear-gradient(180deg, #ffffff 0%, #f6faff 100%) !important;
  box-shadow: 0 1px 3px rgba(16,24,40,.05) !important;
  font-weight: 500 !important; font-size: 13px !important; line-height: 1.5 !important;
  color: #64748b !important;
  transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease !important;
}
.sample-card::first-line {
  font-weight: 800 !important; font-size: 15px !important; color: #0f172a !important;
}
.sample-card:hover {
  border-color: #3b82f6 !important; box-shadow: 0 6px 18px rgba(37,99,235,.15) !important;
  transform: translateY(-2px);
}
.dnh-review {
  border: 1px solid #e6edf6 !important; border-radius: 16px !important;
  background: #ffffff !important; padding: 8px 16px 16px !important;
  box-shadow: 0 6px 22px rgba(16,24,40,.07) !important;
}
/* force every structural container inside the panel to white (kills Gradio's grey) */
.dnh-review .block, .dnh-review .form, .dnh-review .gr-group,
.dnh-review .gr-box, .dnh-review .panel, .dnh-review > div,
.dnh-review .wrap, .dnh-review .container {
  background: #ffffff !important;
}
.dnh-alert {
  font-weight: 700; font-size: 15px; color: #92400e;
  background: #fffbeb; border-left: 4px solid #f59e0b;
  padding: 11px 14px; border-radius: 10px; margin: 6px 0 14px;
}
.dnh-lbl { font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px; color: #94a3b8; }
.dnh-flags-wrap { margin: 6px 0 12px; }
.dnh-flag, .rc-flag {
  display: inline-block; margin: 5px 6px 0 0; padding: 4px 11px;
  background: #ffffff; color: #dc2626; border: 1px solid #fca5a5;
  border-radius: 999px; font-size: 13px; font-weight: 600;
}
.dnh-noflag { color: #15803d; font-weight: 600; }
/* meta rows — DIVs, immune to Gradio's table CSS */
.meta { margin: 10px 0 14px; }
.meta-row { display: flex; align-items: center; padding: 10px 2px;
  border-bottom: 1px solid #eef2f6; }
.meta-row:last-child { border-bottom: none; }
.meta-k { flex: 0 0 140px; color: #94a3b8; font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .6px; }
.meta-v { color: #0f172a; font-size: 14px; font-weight: 500; }
.dnh-draft blockquote {
  margin: 6px 0 0; padding: 9px 13px; border-left: 3px solid #bcd4f6;
  background: #f6faff; border-radius: 8px; font-size: 14px; color: #1f2937;
}
/* clean result card (replaces raw JSON for normal reviewers) */
.rc-empty {
  border: 1px dashed #cbd9ec; border-radius: 14px; background: #f8fbff;
  color: #64748b; padding: 28px 18px; text-align: center; font-size: 14px;
}
.rc-card {
  border: 1px solid #e0e9f5; border-radius: 14px; background: #f8fbff;
  padding: 15px 17px 17px; box-shadow: 0 1px 3px rgba(16,24,40,.05);
}
.rc-title {
  font-weight: 800; font-size: 16px; color: #3b82f6; margin-bottom: 8px;
  letter-spacing: .2px;
}
.sev-badge { display: inline-block; padding: 3px 11px; border-radius: 999px;
  font-size: 13px; font-weight: 700; letter-spacing: .2px; }
.rc-lbl { font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px; color: #94a3b8; }
.rc-flags { margin: 6px 0 12px; }
.rc-ok { color: #15803d; font-weight: 600; }
.rc-draft blockquote {
  margin: 6px 0 0; padding: 9px 13px; border-left: 3px solid #bcd4f6;
  background: #f6faff; border-radius: 8px; font-size: 14px; color: #1f2937;
}
.rc-trail { margin: 14px 0 4px; }
.rc-trail ol {
  margin: 8px 0 0; padding: 0 0 0 6px; list-style: none;
  counter-reset: step; font-size: 13.5px; color: #334155;
}
.rc-trail li {
  position: relative; padding: 5px 0 5px 26px; counter-increment: step;
  border-left: 2px solid #e3ebf5; margin-left: 8px;
}
.rc-trail li::before {
  content: counter(step); position: absolute; left: -11px; top: 5px;
  width: 18px; height: 18px; border-radius: 999px; background: #3b82f6;
  color: #fff; font-size: 11px; font-weight: 700; text-align: center;
  line-height: 18px;
}
/* status pills */
.st { display: inline-block; padding: 7px 14px; border-radius: 10px;
  font-size: 14px; font-weight: 600; }
.st-ok { background: #ecfdf3; color: #15803d; border: 1px solid #bbf7d0; }
.st-review { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.st-err { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.st-wait { background: #eff6ff; color: #3b82f6; border: 1px solid #bfd3f7;
  animation: stpulse 1s ease-in-out infinite; }
@keyframes stpulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
.st-muted { color: #64748b; font-size: 14px; }
/* Override = a real outlined action button, not inactive white */
.override-btn {
  background: #ffffff !important; color: #3b82f6 !important;
  border: 1.5px solid #3b82f6 !important; font-weight: 700 !important;
}
.override-btn:hover {
  background: #eff6ff !important; border-color: #3b82f6 !important;
}
/* primary buttons — soft light blue, defeating Gradio's dark gradient */
.gradio-container button.primary,
.gradio-container .primary button,
.gradio-container button.lg.primary,
.gradio-container button.sm.primary {
  background: #4d94f7 !important;
  background-image: none !important;
  border: 1px solid #4d94f7 !important;
  box-shadow: 0 1px 2px rgba(77,148,247,.25) !important;
  color: #ffffff !important;
}
.gradio-container button.primary:hover,
.gradio-container .primary button:hover {
  background: #3b82f6 !important; border-color: #3b82f6 !important;
}
/* selected radio dot — soft blue, not dark indigo */
.gradio-container input[type="radio"]:checked {
  accent-color: #4d94f7 !important;
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f6f9fd",
    background_fill_secondary="white",
    block_background_fill="white",
    block_border_color="*neutral_200",
    block_label_text_color="*neutral_500",
    block_label_background_fill="white",
    block_title_text_color="*neutral_600",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_hover="#3b82f6",
    button_primary_text_color="white",
)

with gr.Blocks(title="Agentic Pharmacy Triage", css=DNH_CSS, theme=THEME) as demo:
    gr.Markdown(
        "# 💊 Agentic Pharmacy Triage\n"
        "An AI that sorts incoming pharmacy messages by urgency, routes each to the right "
        "person, and **pauses for a licensed pharmacist on anything life-impacting**."
    )
    gr.HTML(
        '<div id="dnh-banner">'
        '<span class="dnh-mark">⚕️</span>'
        '<div><div class="dnh-title">First, do no harm</div>'
        '<div class="dnh-sub">Every life-impacting decision is paused for a licensed pharmacist. '
        'Routine work flows automatically — the gate is in the graph, not a promise.</div></div>'
        "</div>"
    )
    with gr.Accordion("❓ New here? What this is & how to use it — start here", open=True):
        gr.Markdown(
            "**What it is** — *Triage* means quickly sorting messages by urgency and sending each "
            "to the right person. This demo does that for incoming pharmacy messages with a chain "
            "of small AI agents (LangGraph), reasoning on self-hosted **MedGemma** — with a human "
            "pharmacist gating anything dangerous.\n\n"
            "**How to use it — 3 steps**\n"
            "1. **Click any sample card** below 👇 — it fills the message box and runs automatically (no typing needed).\n"
            "2. **Read the result on the right** — what the agent decided, the 🚩 red flags it caught, and a plain-English **audit trail** of every step.\n"
            "3. **Try a dangerous one** — click **🚨 Adverse event** or **🆘 Crisis buried in a refill**. The agent **stops** and hands you a pharmacist-review panel: *you* approve its call, or reassign it.\n\n"
            "**The one thing to notice** — routine messages clear automatically, but life-impacting "
            "ones are *always* paused for a human. That gate is the whole point: **first, do no harm.**\n\n"
            "*Demo only · synthetic messages (no real patients) · not medical advice.*"
        )
    gr.HTML(
        '<div class="pipe">'
        '<span class="step">📥 Intake</span><span class="arr">→</span>'
        '<span class="step">🏷️ Classify</span><span class="arr">→</span>'
        '<span class="step">🩺 Risk screen</span><span class="arr">→</span>'
        '<span class="step">🧭 Route</span><span class="arr">→</span>'
        '<span class="step gate">🧑‍⚕️ Human gate</span><span class="arr">→</span>'
        '<span class="step">🔎 Critic</span><span class="arr">→</span>'
        '<span class="step">✅ Finalize</span>'
        "</div>"
    )
    gr.Markdown(
        "> ⚠️ Demonstration of an agentic *workflow* only. Not medical advice, not a medical "
        "device. All data is synthetic (no PHI). Clinical decisions are gated behind a human."
    )

    gr.Markdown(
        "### Try a sample — one click triages it end-to-end\n"
        "*Click any card below: it fills the message box and runs the triage automatically "
        "(no need to press Run triage — that's only for messages you type yourself).*"
    )
    sample_btns = []
    sample_items = list(SAMPLES.items())
    for i in range(0, len(sample_items), 3):
        with gr.Row():
            for name, _text in sample_items[i:i + 3]:
                btn = gr.Button(_card_label(name), elem_classes=["sample-card"])
                sample_btns.append((btn, name))

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            message = gr.Textbox(
                lines=4,
                label="Inbound pharmacy message",
                placeholder="…or type your own and press Run triage",
            )
            with gr.Accordion("Model endpoint (optional override)", open=False):
                url = gr.Textbox(label="MEDGEMMA_URL", placeholder=DEFAULT_URL)
                key = gr.Textbox(label="API key (optional)", type="password")
            run_btn = gr.Button("Run triage", variant="primary", size="lg")
        with gr.Column(scale=1):
            status = gr.HTML()
            with gr.Group(visible=False, elem_classes=["dnh-review"]) as approval:
                gr.Markdown("### 🧑‍⚕️ Pharmacist review — *first, do no harm*")
                gr.Markdown(
                    "*The agent paused because this case is life-impacting. **You are the "
                    "pharmacist.** Either **✓ Approve** — accept the handler the agent "
                    "proposed — or pick a different handler below and **✎ Override** to "
                    "reassign it. The case is finalized only with your sign-off.*"
                )
                approval_info = gr.HTML()
                assignee = gr.Radio(ASSIGNEES, label="Reassign to (used when you Override)")
                note = gr.Textbox(
                    label="Pharmacist note (optional)",
                    placeholder="e.g. Called the patient and advised them to stop the medication.",
                )
                with gr.Row():
                    approve_btn = gr.Button("✓ Approve", variant="primary")
                    override_btn = gr.Button("✎ Override", elem_classes=["override-btn"])
                finalize_status = gr.HTML()
            result_card = gr.HTML(EMPTY_RESULT)
            trace = gr.HTML()
            with gr.Accordion("Raw record (JSON)", open=False):
                result_json = gr.JSON(label=None)

    with gr.Accordion("ℹ️ What the terms mean", open=False):
        gr.Markdown(
            "- **Triage** — quickly sorting messages by urgency and routing each to the right person.\n"
            "- **Pharmacy technician** — support staff; handles admin and simple refills (no clinical advice).\n"
            "- **Pharmacist** — the licensed clinical expert; handles interactions, counseling, controlled substances, and reviews anything flagged.\n"
            "- **Prescriber** — a doctor or nurse practitioner who can prescribe; looped in for adverse events, red flags, and dose changes.\n"
            "- **Severity** — clinical urgency: **P0 · emergent** (immediate danger) → **P1 · urgent** → **P2 · elevated** → **P3 · routine**."
        )

    st_graph = gr.State()
    st_config = gr.State()

    run_inputs = [message, url, key]
    run_outputs = [status, result_card, result_json, trace, approval, approval_info, assignee, st_graph, st_config]
    run_btn.click(run_triage, run_inputs, run_outputs)
    # Each sample button fills the box, then runs triage — one click to a result.
    for btn, name in sample_btns:
        btn.click(lambda n=name: SAMPLES[n], None, message).then(
            run_triage, run_inputs, run_outputs
        )
    resolve_outputs = [status, finalize_status, result_card, result_json, trace, approval]
    resolve_inputs = [assignee, note, st_graph, st_config]

    def on_approve(a, n, g, c):
        yield from resolve("approve", a, n, g, c)

    def on_override(a, n, g, c):
        yield from resolve("override", a, n, g, c)

    approve_btn.click(on_approve, resolve_inputs, resolve_outputs)
    override_btn.click(on_override, resolve_inputs, resolve_outputs)


if __name__ == "__main__":
    demo.launch()
