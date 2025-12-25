from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Callable, Dict, List, Optional

from ..models import DatasetItem, FunctionCall


class LLMJudge:
    """
    Base judge abstraction. Subclass and override `score` method.
    The scorer should return a dict like: {"score": float, "passed": bool, "reason": str}.
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.0,
    ):
        self.name = name
        self.prompt = prompt
        self.model = model
        self.temperature = temperature

    def score(self, call: FunctionCall, item: DatasetItem) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement score()")


class EchoJudge(LLMJudge):
    """Useful for tests: returns 1.0 if expected substring is present."""

    def __init__(self):
        super().__init__(name="echo", prompt="echo judge", model="debug")

    def score(self, call: FunctionCall, item: DatasetItem) -> Dict[str, Any]:
        expected = item.expected
        text = str(call.output or "")
        passed = expected is not None and str(expected) in text
        return {"score": 1.0 if passed else 0.0, "passed": passed, "reason": "debug-echo", "raw": {"output": text}}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text, handling markdown code blocks."""
    text = (text or "").strip()
    if not text:
        return None

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Try to find JSON object by brace matching
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    parsed = json.loads(snippet)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
    return None


def _parse_passed(value: Any) -> Optional[bool]:
    """Parse various representations of pass/fail to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"pass", "passed", "true", "yes", "1"}:
            return True
        if low in {"fail", "failed", "false", "no", "0"}:
            return False
    return None


def _safe_trace_excerpt(call: FunctionCall, max_events: int = 20, max_chars: int = 2000) -> str:
    """Create a safe string excerpt of trace events for the judge prompt."""
    events = call.trace or []
    lines = []
    for ev in events[:max_events]:
        detail = ev.detail or {}
        try:
            detail_str = json.dumps(detail, default=str)
        except Exception:
            detail_str = str(detail)
        lines.append(f"- {ev.kind}: {detail_str[:300]}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


class GeminiJudge(LLMJudge):
    """
    LLM judge using Google Gemini API via direct HTTP calls.
    Requires GEMINI_API_KEY environment variable.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        name: str = "gemini-judge",
        prompt: str = "",
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        super().__init__(name=name, prompt=prompt, model=model, temperature=temperature)
        self._api_key = api_key

    def _get_api_key(self) -> str:
        key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set the environment variable or pass api_key to GeminiJudge."
            )
        return key

    def _call_api(self, prompt: str) -> str:
        """Make direct HTTP call to Gemini API."""
        api_key = self._get_api_key()
        url = self.API_URL.format(model=self.model) + f"?key={api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini API connection error: {e.reason}") from e

        # Extract text from response
        try:
            candidates = response_data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        except Exception:
            pass

        return ""

    def _build_evaluation_prompt(self, call: FunctionCall, item: DatasetItem) -> str:
        """Build the full prompt for evaluation."""
        trace_excerpt = _safe_trace_excerpt(call)

        # Use the new 4-column model: input, output, human_label, metadata
        evaluation_input = {
            "input": item.input or item.inputs,  # User input
            "output": item.output or call.output,  # Agent response
            "human_label": item.human_label,  # Human judgement (if any)
            "trace_excerpt": trace_excerpt if trace_excerpt else None,
        }

        full_prompt = f"""{self.prompt}

EVALUATION_INPUT:
{json.dumps(evaluation_input, default=str, ensure_ascii=False, indent=2)}

Evaluate the OUTPUT given the INPUT. Return ONLY a JSON object with:
- "passed": boolean (true if criteria met, false otherwise)
- "reason": string (brief explanation)
- "score": number 0-1 (optional, defaults to 1 if passed, 0 if failed)
"""
        return full_prompt

    def score(self, call: FunctionCall, item: DatasetItem) -> Dict[str, Any]:
        full_prompt = self._build_evaluation_prompt(call, item)

        try:
            raw_text = self._call_api(full_prompt)
        except Exception as e:
            return {
                "score": None,
                "passed": None,
                "reason": f"API call failed: {e}",
                "raw": {"error": str(e)},
            }

        parsed = _extract_json_object(raw_text) or {}

        # Extract passed status
        passed = None
        for key in ("passed", "pass", "verdict"):
            if key in parsed:
                passed = _parse_passed(parsed.get(key))
                if passed is not None:
                    break

        # Extract score
        score = parsed.get("score")
        if isinstance(score, str):
            try:
                score = float(score)
            except Exception:
                score = None

        # Default score based on passed status
        if score is None and passed is not None:
            score = 1.0 if passed else 0.0

        # Clamp score to 0-1
        if isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))

        return {
            "score": score,
            "passed": passed,
            "reason": parsed.get("reason"),
            "raw": {"response": parsed, "text": raw_text, "model": self.model},
        }


# Backwards compatibility aliases
OpenAIJudge = GeminiJudge  # Deprecated: use GeminiJudge


__all__ = [
    "LLMJudge",
    "EchoJudge",
    "GeminiJudge",
    "OpenAIJudge",  # Deprecated alias
]
