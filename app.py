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
        info = (
            f"**Category:** {p['category']}  \n"
            f"**Severity:** {p['severity']}  \n"
            f"**Red flags:** {', '.join(p['red_flags']) or 'none'}  \n"
            f"**Proposed assignee:** {p['proposed_assignee']}  \n\n"
            f"**Draft reply (for review):**\n\n> {p['draft_response']}"
        )
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


with gr.Blocks(title="Agentic Pharmacy Triage") as demo:
    gr.Markdown(
        "# 💊 Agentic Pharmacy Triage\n"
        "A multi-agent **LangGraph** pipeline that triages inbound pharmacy messages — "
        "classify intent, screen for clinical red flags, route to the right role, and "
        "**pause for a licensed pharmacist on anything life-impacting** (first, do no harm). "
        "Reasoning runs on **MedGemma** (self-hosted).\n\n"
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
            with gr.Group(visible=False) as approval:
                gr.Markdown("### 🧑‍⚕️ Pharmacist review")
                approval_info = gr.Markdown()
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
