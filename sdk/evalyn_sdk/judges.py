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
