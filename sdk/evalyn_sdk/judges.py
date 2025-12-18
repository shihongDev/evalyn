from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Optional

from .models import DatasetItem, FunctionCall


class LLMJudge:
    """
    Simple judge abstraction. Provide a scorer callable or subclass and override `score`.
    The scorer should return a dict like: {"score": float, "reason": str, "raw": {...}}.
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        scorer: Optional[Callable[[FunctionCall, DatasetItem], Dict[str, Any]]] = None,
        model: str = "gpt-4.1",
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.model = model
        self.parameters = parameters or {}
        self._scorer = scorer

    def score(self, call: FunctionCall, item: DatasetItem) -> Dict[str, Any]:
        if self._scorer is None:
            raise NotImplementedError("Provide a scorer callable or override score()")
        return self._scorer(call, item)


class EchoJudge(LLMJudge):
    """Useful for tests: returns 1.0 if expected substring is present."""

    def __init__(self):
        super().__init__(name="echo", prompt="echo judge", scorer=None, model="debug")

    def score(self, call: FunctionCall, item: DatasetItem) -> Dict[str, Any]:
        expected = item.expected
        text = str(call.output or "")
        passed = expected is not None and str(expected) in text
        return {"score": 1.0 if passed else 0.0, "reason": "debug-echo", "raw": {"output": text}}


class OpenAIJudge(LLMJudge):
    """
    LLM judge using OpenAI Chat Completions. Requires `openai` package and `OPENAI_API_KEY`.
    """

    def __init__(
        self,
        name: str = "openai-judge",
        model: str = "gpt-4.1",
        prompt: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        prompt_text = prompt or (
            "You are an evaluator. Given the task input, expected behavior (if any), and the model output, "
            "return a JSON object with `score` between 0 and 1 and a brief `reason`."
        )
        super().__init__(name=name, prompt=prompt_text, scorer=None, model=model, parameters=parameters)
        self.system_prompt = system_prompt or "You are a strict evaluator."

    def score(self, call: FunctionCall, item: DatasetItem) -> Dict[str, Any]:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package not installed. Install with extras: pip install evalyn-sdk[llm]") from exc

        client = openai.OpenAI()
        user_prompt = (
            f"Task input: {item.inputs}\n"
            f"Expected (optional): {item.expected}\n"
            f"Model output: {call.output}\n"
        )

        params = {"temperature": 0.0, **self.parameters}
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{self.prompt}\n\n{user_prompt}"},
            ],
            response_format={"type": "json_object"},
            **params,
        )
        content = resp.choices[0].message.content
        import json

        parsed: Dict[str, Any] = json.loads(content)
        passed = parsed.get("passed") if isinstance(parsed, dict) else None
        if isinstance(passed, str):
            low = passed.strip().lower()
            if low in {"pass", "passed", "true", "yes"}:
                passed = True
            elif low in {"fail", "failed", "false", "no"}:
                passed = False
        score = parsed.get("score") if isinstance(parsed, dict) else None
        if isinstance(score, str):
            try:
                score = float(score)
            except Exception:
                score = None
        if passed is not None and score is None:
            score = 1.0 if bool(passed) else 0.0
        return {
            "score": score,
            "passed": passed,
            "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
            "raw": {"response": parsed, "model": self.model},
        }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Try to find the first JSON object by brace matching.
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


def _safe_trace_excerpt(call: FunctionCall, max_events: int = 20, max_chars: int = 2000) -> str:
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
    LLM judge using Google Gemini. Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and either
    `google-genai` (preferred) or `google-generativeai` installed.
    """

    def __init__(
        self,
        name: str = "gemini-judge",
        model: str = "gemini-2.5-flash-lite",
        prompt: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        prompt_text = prompt or (
            "You are an evaluator. Given the task input, expected behavior (if any), the model output, "
            "and any available trace/tool evidence, return a JSON object with `score` between 0 and 1 "
            "and a brief `reason`. Return ONLY JSON."
        )
        super().__init__(name=name, prompt=prompt_text, scorer=None, model=model, parameters=parameters)

    def score(self, call: FunctionCall, item: DatasetItem) -> Dict[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) for GeminiJudge.")

        trace_excerpt = _safe_trace_excerpt(call)
        user_payload = {
            "inputs": item.inputs,
            "expected": item.expected,
            "output": call.output,
            "trace_excerpt": trace_excerpt,
        }
        user_prompt = json.dumps(user_payload, default=str, ensure_ascii=False)
        full_prompt = f"{self.prompt}\n\nEVALUATION_INPUT:\n{user_prompt}\n"

        temperature = float(self.parameters.get("temperature", 0.0)) if self.parameters else 0.0

        text = None
        # Preferred: google-genai
        try:
            from google.genai import Client  # type: ignore

            client = Client(api_key=api_key)
            resp = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={"temperature": temperature, **(self.parameters or {})},
            )
            text = getattr(resp, "text", None)
        except Exception:
            text = None

        # Fallback: google-generativeai
        if text is None:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content(full_prompt, generation_config={"temperature": temperature})
                text = getattr(resp, "text", None)
            except Exception as exc:
                raise RuntimeError(f"GeminiJudge failed to call Gemini: {exc}") from exc

        raw_text = (text or "").strip()
        parsed = _extract_json_object(raw_text) or {}
        passed = parsed.get("passed") if isinstance(parsed, dict) else None
        if isinstance(passed, str):
            low = passed.strip().lower()
            if low in {"pass", "passed", "true", "yes"}:
                passed = True
            elif low in {"fail", "failed", "false", "no"}:
                passed = False
        if passed is None and isinstance(parsed, dict) and "verdict" in parsed:
            verdict = str(parsed.get("verdict") or "").strip().lower()
            if verdict in {"pass", "passed", "true", "yes"}:
                passed = True
            elif verdict in {"fail", "failed", "false", "no"}:
                passed = False

        score = parsed.get("score") if isinstance(parsed, dict) else None
        if isinstance(score, str):
            try:
                score = float(score)
            except Exception:
                score = None
        if passed is not None and score is None:
            score = 1.0 if bool(passed) else 0.0
        if isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))
        else:
            score = None

        return {
            "score": score,
            "passed": passed,
            "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
            "raw": {"response": parsed, "text": raw_text, "model": self.model},
        }
