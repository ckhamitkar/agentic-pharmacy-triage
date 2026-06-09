"""MedGemma client — the ONLY integration point with the model.

The triage pipeline is endpoint-agnostic: it speaks to MedGemma through this
small client. Point it at any compatible endpoint via env vars:

    MEDGEMMA_URL   default: https://api.axionaiapps.com/generate  (Mini-backed, live)
    MEDGEMMA_API_KEY  optional bearer token

Swap targets with zero code changes — the live /generate endpoint (raw MedGemma
completion, no retrieval), the MedGemma Docker container running locally, or a
Hugging Face Docker Space / Inference Endpoint. Only the URL changes.

MedGemma is a self-hosted open model, so we don't get native JSON-schema/tool
calling. Instead we instruct it to emit JSON, extract the JSON block, validate
it against a Pydantic schema, and retry once with the error fed back.
"""

from __future__ import annotations

import json
import os
import re
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

DEFAULT_URL = "https://api.axionaiapps.com/generate"

# ---- Endpoint contract (the one place to adjust if the endpoint differs) -----
# Request body key that carries the prompt. The default /generate endpoint
# expects "prompt"; override via MEDGEMMA_PROMPT_KEY for an endpoint (e.g. a RAG
# /ask route) that wants "question".
REQUEST_PROMPT_KEY = os.environ.get("MEDGEMMA_PROMPT_KEY", "prompt")
# Response keys tried, in order, to find the generated text:
RESPONSE_TEXT_KEYS = ("response", "answer", "text", "output", "completion", "result")

T = TypeVar("T", bound=BaseModel)


class MedGemmaError(RuntimeError):
    """Raised when MedGemma is unreachable or returns unusable output."""


class MedGemmaClient:
    def __init__(self, url: str | None = None, api_key: str | None = None, timeout: float = 90.0):
        self.url = url or os.environ.get("MEDGEMMA_URL", DEFAULT_URL)
        self.api_key = api_key or os.environ.get("MEDGEMMA_API_KEY")
        self.timeout = timeout

    def complete(self, prompt: str, system: str | None = None) -> str:
        full = f"{system.strip()}\n\n{prompt.strip()}" if system else prompt.strip()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(
                self.url, json={REQUEST_PROMPT_KEY: full}, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:  # network, timeout, non-2xx
            raise MedGemmaError(f"MedGemma request to {self.url} failed: {e}") from e
        return _extract_text(resp.json() if _is_json(resp) else resp.text)

    def complete_json(self, prompt: str, schema: Type[T], system: str | None = None, retries: int = 1) -> T:
        """Get a validated Pydantic object back from MedGemma."""
        json_schema = json.dumps(schema.model_json_schema(), indent=2)
        instruction = (
            "You are a careful clinical-operations assistant. "
            "Respond with ONLY a single JSON object — no prose, no markdown fences — "
            f"that conforms to this JSON schema:\n\n{json_schema}"
        )
        sys = f"{system.strip()}\n\n{instruction}" if system else instruction
        last_err = ""
        for attempt in range(retries + 1):
            ask = prompt if attempt == 0 else (
                f"{prompt}\n\nYour previous reply could not be parsed ({last_err}). "
                "Reply again with ONLY the JSON object."
            )
            raw = self.complete(ask, system=sys)
            try:
                return schema.model_validate(_extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:160]
        raise MedGemmaError(f"MedGemma did not return valid JSON for {schema.__name__}: {last_err}")


# ---- helpers ----------------------------------------------------------------


def _is_json(resp: httpx.Response) -> bool:
    return "application/json" in resp.headers.get("content-type", "")


def _extract_text(data) -> str:
    """Pull the generated text out of whatever shape the endpoint returns."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in RESPONSE_TEXT_KEYS:
            if isinstance(data.get(key), str):
                return data[key]
        # OpenAI-compatible shape: {"choices":[{"message":{"content": "..."}}]}
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        try:
            return data["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
    raise MedGemmaError(f"Could not find generated text in response: {str(data)[:200]}")


def _extract_json(text: str) -> dict:
    """Find and parse the first JSON object in a model reply."""
    text = text.strip()
    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)  # first {...} block
    if match:
        return json.loads(match.group(0))
    raise ValueError("no JSON object found in reply")
