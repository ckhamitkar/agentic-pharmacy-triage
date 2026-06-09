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


def _trace_md(state) -> str:
    lines = state.get("trace", [])
    return "### Audit trail\n" + "\n".join(f"- {ln}" for ln in lines)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _review_html(p) -> str:
    """Render the pharmacist-review panel as styled HTML so the *why* — the
    escalation reason and the red flags — visually pops."""
    flags = p.get("red_flags") or []
    chips = (
        "".join(f'<span class="dnh-flag">🚩 {_esc(f)}</span>' for f in flags)
        if flags else '<span class="dnh-noflag">none</span>'
    )
    return (
        f'<div class="dnh-alert">⏸ {_esc(p["reason"])}</div>'
        '<div class="dnh-flags-wrap"><span class="dnh-lbl">Red flags</span><br>'
        f"{chips}</div>"
        '<table class="dnh-meta"><tr><td>Category</td><td>'
        f'{_esc(p["category"])}</td></tr>'
        f'<tr><td>Severity</td><td><b>{_esc(p["severity"])}</b></td></tr>'
        f'<tr><td>Proposed assignee</td><td>{_esc(p["proposed_assignee"])}</td></tr></table>'
        '<div class="dnh-draft"><span class="dnh-lbl">Draft reply (for review)</span>'
        f'<blockquote>{_esc(p["draft_response"])}</blockquote></div>'
    )


def run_triage(message, url, key):
    message = (message or "").strip()
    blank = (None, None)
    if not message:
        return ("Enter or pick a message.", None, "", gr.update(visible=False), "", gr.update(), *blank)
    graph = build_graph(MedGemmaClient(url=url or None, api_key=key or None))
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    try:
        result = graph.invoke({"raw_message": message, "trace": []}, config)
    except MedGemmaError as e:
        return (f"❌ MedGemma error: {e}", None, "", gr.update(visible=False), "", gr.update(), *blank)

    if "__interrupt__" in result:
        p = result["__interrupt__"][0].value
        info = _review_html(p)
        return (
            "⏸ **Pharmacist review required** — first, do no harm.",
            None,
            _trace_md(result),
            gr.update(visible=True),
            info,
            gr.update(value=p["proposed_assignee"]),
            graph,
            config,
        )

    return (
        "✅ Auto-triaged (routine — no human review needed).",
        result["final"],
        _trace_md(result),
        gr.update(visible=False),
        "",
        gr.update(),
        graph,
        config,
    )


def resolve(action, assignee, note, graph, config):
    if graph is None or config is None:
        return ("No pending review.", None, "", gr.update(visible=False))
    decision = {"action": action, "final_assignee": assignee, "note": note or ""}
    result = graph.invoke(Command(resume=decision), config)
    return (
        f"✅ Finalized by pharmacist ({action}).",
        result["final"],
        _trace_md(result),
        gr.update(visible=False),
    )


DNH_CSS = """
#dnh-banner {
  display: flex; align-items: center; gap: 16px;
  padding: 16px 20px; margin: 6px 0 10px;
  border-radius: 14px; border: 1px solid #f0b8bd;
  background: linear-gradient(100deg, #fff4f4 0%, #fff9f0 100%);
  box-shadow: 0 1px 3px rgba(176,42,55,.08);
}
#dnh-banner .dnh-mark { font-size: 34px; line-height: 1; }
#dnh-banner .dnh-title {
  font-weight: 800; font-size: 20px; color: #b02a37;
  letter-spacing: .3px; margin-bottom: 2px;
}
#dnh-banner .dnh-sub { font-size: 13px; color: #6e4b4b; }
.dnh-review {
  border: 2px solid #f0b8bd !important; border-radius: 14px !important;
  background: #fffafa !important; padding: 6px 14px 12px !important;
}
.dnh-alert {
  font-weight: 700; font-size: 15px; color: #842029;
  background: #fde8e8; border-left: 5px solid #d6303f;
  padding: 10px 14px; border-radius: 8px; margin: 4px 0 12px;
}
.dnh-lbl { font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px; color: #8a8a8a; }
.dnh-flags-wrap { margin: 6px 0 12px; }
.dnh-flag {
  display: inline-block; margin: 4px 6px 0 0; padding: 4px 10px;
  background: #fde2e2; color: #b02a37; border: 1px solid #f0b0b6;
  border-radius: 999px; font-size: 13px; font-weight: 600;
}
.dnh-noflag { color: #4a8a4a; font-weight: 600; }
.dnh-meta { border-collapse: collapse; margin: 2px 0 10px; font-size: 14px; }
.dnh-meta td { padding: 2px 14px 2px 0; }
.dnh-meta td:first-child { color: #8a8a8a; }
.dnh-draft blockquote {
  margin: 6px 0 0; padding: 8px 12px; border-left: 3px solid #cbd5e1;
  background: #f8fafc; border-radius: 6px; font-size: 14px;
}
"""

with gr.Blocks(title="Agentic Pharmacy Triage", css=DNH_CSS) as demo:
    gr.Markdown(
        "# 💊 Agentic Pharmacy Triage\n"
        "A multi-agent **LangGraph** pipeline that triages inbound pharmacy messages — "
        "classify intent, screen for clinical red flags, route to the right role, and "
        "**pause for a licensed pharmacist on anything life-impacting**. "
        "Reasoning runs on **MedGemma** (self-hosted)."
    )
    gr.HTML(
        '<div id="dnh-banner">'
        '<span class="dnh-mark">⚕️</span>'
        '<div><div class="dnh-title">First, do no harm</div>'
        '<div class="dnh-sub">Every life-impacting decision is paused for a licensed pharmacist. '
        'Routine work flows automatically — the gate is in the graph, not a promise.</div></div>'
        "</div>"
    )
    gr.Markdown(
        "> ⚠️ Demonstration of an agentic *workflow* only. Not medical advice, not a medical "
        "device. All data is synthetic (no PHI). Clinical decisions are gated behind a human."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Try a sample — click to triage")
            sample_btns = []
            sample_items = list(SAMPLES.items())
            for i in range(0, len(sample_items), 2):
                with gr.Row():
                    for name, _text in sample_items[i:i + 2]:
                        sample_btns.append((gr.Button(name, size="sm"), name))
            message = gr.Textbox(
                lines=4,
                label="Inbound pharmacy message",
                placeholder="…or type your own and press Run triage",
            )
            with gr.Accordion("Model endpoint (optional override)", open=False):
                url = gr.Textbox(label="MEDGEMMA_URL", placeholder=DEFAULT_URL)
                key = gr.Textbox(label="API key (optional)", type="password")
            run_btn = gr.Button("Run triage", variant="primary")
        with gr.Column():
            status = gr.Markdown()
            with gr.Group(visible=False, elem_classes=["dnh-review"]) as approval:
                gr.Markdown("### 🧑‍⚕️ Pharmacist review — *first, do no harm*")
                approval_info = gr.HTML()
                assignee = gr.Radio(ASSIGNEES, label="Final assignee")
                note = gr.Textbox(label="Pharmacist note (optional)")
                with gr.Row():
                    approve_btn = gr.Button("Approve", variant="primary")
                    override_btn = gr.Button("Override")
            result = gr.JSON(label="Triage record")
            trace = gr.Markdown()

    st_graph = gr.State()
    st_config = gr.State()

    run_inputs = [message, url, key]
    run_outputs = [status, result, trace, approval, approval_info, assignee, st_graph, st_config]
    run_btn.click(run_triage, run_inputs, run_outputs)
    # Each sample button fills the box, then runs triage — one click to a result.
    for btn, name in sample_btns:
        btn.click(lambda n=name: SAMPLES[n], None, message).then(
            run_triage, run_inputs, run_outputs
        )
    approve_btn.click(
        lambda a, n, g, c: resolve("approve", a, n, g, c),
        [assignee, note, st_graph, st_config],
        [status, result, trace, approval],
    )
    override_btn.click(
        lambda a, n, g, c: resolve("override", a, n, g, c),
        [assignee, note, st_graph, st_config],
        [status, result, trace, approval],
    )


if __name__ == "__main__":
    demo.launch()
