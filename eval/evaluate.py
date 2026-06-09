"""Evaluate the triage pipeline against labeled fixtures.

Runs each message through the graph. If the graph pauses for a human
(`__interrupt__`) OR sets `requires_human`, we count it as a "flag for human"
prediction. We report overall accuracy AND — the metric that matters —
recall on the dangerous cases.

    python -m eval.evaluate          # uses MEDGEMMA_URL (default: live /ask)

Requires a reachable MedGemma endpoint.
"""

from __future__ import annotations

import uuid

from triage import build_graph
from eval.fixtures import FIXTURES


def predicted_human(graph, message: str) -> bool:
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = graph.invoke({"raw_message": message, "trace": []}, config)
    return "__interrupt__" in result or bool(result.get("requires_human"))


def main() -> None:
    graph = build_graph()
    tp = fp = tn = fn = 0
    print(f"{'case':32} {'expected':9} {'predicted':9}  ok")
    for fx in FIXTURES:
        pred = predicted_human(graph, fx["message"])
        exp = fx["dangerous"]
        ok = pred == exp
        tp += pred and exp
        fp += pred and not exp
        tn += (not pred) and (not exp)
        fn += (not pred) and exp
        print(f"{fx['name'][:32]:32} {str(exp):9} {str(pred):9}  {'✓' if ok else '✗'}")

    dangerous = tp + fn
    recall = tp / dangerous if dangerous else float("nan")
    acc = (tp + tn) / len(FIXTURES)
    print("\n--- summary ---")
    print(f"accuracy:                 {acc:.2f}")
    print(f"dangerous-case recall:    {recall:.2f}   (the metric that matters — missed danger = {fn})")
    print(f"confusion: tp={tp} fp={fp} tn={tn} fn={fn}")
    if fn:
        print("⚠️  FALSE NEGATIVES (dangerous cases auto-handled) — investigate before trusting.")


if __name__ == "__main__":
    main()
