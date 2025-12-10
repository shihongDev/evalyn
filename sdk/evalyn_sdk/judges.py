from __future__ import annotations

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
            "Return JSON with keys: score (0-1) and reason."
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
        return {
            "score": parsed.get("score"),
            "reason": parsed.get("reason"),
            "raw": {"response": parsed, "model": self.model},
        }
