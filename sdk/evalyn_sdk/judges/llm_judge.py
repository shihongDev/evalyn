"""
LLM Judge for subjective metric evaluation.

Usage:
    # From template
    metric = LLMJudge.from_template("toxicity").as_metric("safety_check")

    # Custom prompt
    judge = LLMJudge(prompt="You evaluate code quality...")
    metric = judge.as_metric("code_quality", threshold=0.7)

    # List available templates
    print(LLMJudge.list_templates())
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import canonical templates from metrics.subjective
from ..metrics.subjective import JUDGE_TEMPLATES
from ..models import DatasetItem, FunctionCall, Metric, MetricResult, MetricSpec
from ..parsing import _extract_json_object, _parse_passed, _safe_trace_excerpt
from ..utils.api_client import create_llm_client

# =============================================================================
# LLMJudge Class
# =============================================================================


class LLMJudge:
    """
    LLM-based judge for subjective evaluation.

    Supports multiple providers: gemini (default), openai, ollama.

    Usage:
        # From built-in template (default: Gemini)
        judge = LLMJudge.from_template("toxicity_safety")
        metric = judge.as_metric("safety_check")

        # With OpenAI
        judge = LLMJudge.from_template("toxicity_safety", provider="openai")

        # Custom prompt
        judge = LLMJudge(prompt="You evaluate code quality...", provider="openai")
        metric = judge.as_metric("code_quality", threshold=0.7)
    """

    TEMPLATES = JUDGE_TEMPLATES

    def __init__(
        self,
        prompt: str,
        name: str = "llm-judge",
        model: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
        provider: str = "gemini",
        rubric: list[str] | None = None,
    ):
        self.prompt = prompt
        self.name = name
        self.provider = provider
        self.temperature = temperature
        self._api_key = api_key
        self.rubric = rubric or []
        self._client = None

        # Default model depends on provider
        if model is None:
            from ..defaults import DEFAULT_EVAL_MODEL, DEFAULT_MODELS_BY_PROVIDER

            self.model = DEFAULT_MODELS_BY_PROVIDER.get(provider, DEFAULT_EVAL_MODEL)
        else:
            self.model = model

    @property
    def client(self):
        """Lazy-initialized API client based on provider."""
        if self._client is None:
            self._client = create_llm_client(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                api_key=self._api_key,
            )
        return self._client

    @classmethod
    def from_template(
        cls,
        template: str,
        model: str | None = None,
        api_key: str | None = None,
        provider: str = "gemini",
    ) -> LLMJudge:
        """Create judge from built-in template.

        Args:
            template: Template name (e.g., "toxicity_safety", "helpfulness_accuracy")
            model: LLM model to use (default depends on provider)
            api_key: Optional API key (default: from env var based on provider)
            provider: LLM provider - "gemini", "openai", or "ollama"

        Returns:
            Configured LLMJudge instance

        Example:
            judge = LLMJudge.from_template("toxicity_safety")
            judge = LLMJudge.from_template("toxicity_safety", provider="openai")
        """
        if template not in cls.TEMPLATES:
            available = ", ".join(cls.TEMPLATES.keys())
            raise ValueError(f"Unknown template '{template}'. Available: {available}")

        tpl = cls.TEMPLATES[template]
        # Rubric can be at top level (legacy) or nested under config (new format)
        rubric = tpl.get("rubric", [])
        if not rubric and "config" in tpl:
            rubric = tpl["config"].get("rubric", [])
        return cls(
            prompt=tpl["prompt"],
            name=template,
            model=model,
            api_key=api_key,
            provider=provider,
            rubric=rubric,
        )

    @classmethod
    def list_templates(cls) -> list[str]:
        """List available template names."""
        return list(cls.TEMPLATES.keys())

    def _build_evaluation_body(self, call: FunctionCall, item: DatasetItem) -> str:
        """Build the evaluation-specific part of the prompt (no rubric/system text).

        This contains the EVALUATION_INPUT data and response format instructions.
        The rubric and system prompt are handled separately via _get_system_instruction()
        for providers that support system messages (Gemini, OpenAI), or prepended
        by _build_evaluation_prompt() for providers that don't (Ollama).
        """
        trace_excerpt = _safe_trace_excerpt(call)

        evaluation_input = {
            "input": item.input or item.inputs,
            "output": item.output or call.output,
            "human_label": item.human_label,
            "trace_excerpt": trace_excerpt if trace_excerpt else None,
        }

        # Include span IDs for attribution when spans are available
        span_ids = []
        if call.spans:
            for s in call.spans:
                span_ids.append({"span_id": s.id, "name": s.name, "type": s.span_type})
            evaluation_input["spans"] = span_ids

        return f"""EVALUATION_INPUT:
{json.dumps(evaluation_input, default=str, ensure_ascii=False, indent=2)}

Evaluate the OUTPUT given the INPUT. Return ONLY a JSON object with:
- "passed": boolean (true if criteria met, false otherwise)
- "reason": string (brief explanation)
- "score": number 0-1 (optional, defaults to 1 if passed, 0 if failed)
- "span_attribution": array (optional, only if spans are provided in the input)
  Each element: {{"span_id": "...", "relevance": 0.0-1.0, "reason": "..."}}
  Identify which spans most influenced your verdict.
"""

    def _format_rubric(self) -> str:
        """Format rubric lines for inclusion in prompts. Returns empty string if no rubric."""
        if not self.rubric:
            return ""
        rubric_lines = "\n".join(f"- {r}" for r in self.rubric)
        return f"\n\nRUBRIC:\n{rubric_lines}"

    def _build_evaluation_prompt(self, call: FunctionCall, item: DatasetItem) -> str:
        """Build the full prompt with system text inlined.

        Used by score_with_confidence/score_with_deepconf. The main score()
        path uses _build_evaluation_body() + _get_system_instruction() instead,
        which enables prefix caching on Gemini/OpenAI.
        """
        body = self._build_evaluation_body(call, item)
        return f"""{self.prompt}{self._format_rubric()}

{body}"""

    def _parse_response(
        self, raw_text: str
    ) -> tuple[float | None, bool | None, str | None, dict]:
        """Parse LLM response text and return (score, passed, reason, parsed_dict)."""
        parsed = _extract_json_object(raw_text) or {}

        # Extract passed status
        passed = None
        for key in ("passed", "pass", "verdict"):
            if key in parsed:
                passed = _parse_passed(parsed.get(key))
                if passed is not None:
                    break

        # Extract and normalize score
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

        return score, passed, parsed.get("reason"), parsed

    def _get_system_instruction(self) -> str | None:
        """Build the stable system instruction (prompt + rubric) for prefix caching.

        Gemini/OpenAI automatically cache shared prefixes when sent as the
        system message, saving up to 50-90% on input token costs.
        """
        return f"{self.prompt}{self._format_rubric()}" if self.prompt else None

    def score(self, call: FunctionCall, item: DatasetItem) -> dict[str, Any]:
        """Evaluate and return {score, passed, reason, input_tokens, output_tokens, model}."""
        # Separate rubric (stable) from evaluation data (per-item) for prefix caching
        prompt = self._build_evaluation_body(call, item)
        kwargs: dict = {"system_instruction": self._get_system_instruction()}

        try:
            result = self.client.generate_with_usage(prompt, **kwargs)
            raw_text = result.text
            input_tokens = result.input_tokens
            output_tokens = result.output_tokens
        except Exception as e:
            return {
                "score": None,
                "passed": None,
                "reason": f"API call failed: {e}",
                "raw": {"error": str(e)},
                "input_tokens": 0,
                "output_tokens": 0,
                "model": self.model,
            }

        score, passed, reason, parsed = self._parse_response(raw_text)
        return {
            "score": score,
            "passed": passed,
            "reason": reason,
            "raw": {"response": parsed, "text": raw_text, "model": self.model},
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": self.model,
        }

    def score_with_confidence(
        self, call: FunctionCall, item: DatasetItem
    ) -> dict[str, Any]:
        """Evaluate with logprobs-based confidence (openai/ollama only).

        Returns {score, passed, reason, confidence, raw}.
        Requires provider to be 'openai' or 'ollama' (Gemini lacks logprobs).
        """
        if self.provider == "gemini":
            raise RuntimeError(
                "Logprobs confidence requires provider='openai' or 'ollama'. "
                "Gemini does not expose logprobs."
            )

        full_prompt = self._build_evaluation_prompt(call, item)

        try:
            raw_text, confidence = self.client.generate_with_confidence(full_prompt)
        except Exception as e:
            return {
                "score": None,
                "passed": None,
                "reason": f"API call failed: {e}",
                "confidence": 0.0,
                "raw": {"error": str(e)},
            }

        score, passed, reason, parsed = self._parse_response(raw_text)
        return {
            "score": score,
            "passed": passed,
            "reason": reason,
            "confidence": confidence,
            "raw": {
                "response": parsed,
                "text": raw_text,
                "model": self.model,
                "logprobs_confidence": confidence,
            },
        }

    def score_with_deepconf(
        self, call: FunctionCall, item: DatasetItem, strategy: str = "bottom10"
    ) -> dict[str, Any]:
        """Evaluate with DeepConf confidence (openai only, ollama limited).

        DeepConf uses specialized aggregation strategies that better distinguish
        correct from incorrect reasoning traces.

        Args:
            call: Function call to evaluate
            item: Dataset item with input/expected
            strategy: DeepConf strategy - "bottom10" (default), "tail", "average"

        Returns {score, passed, reason, confidence, raw}.
        """
        if self.provider == "gemini":
            raise RuntimeError(
                "DeepConf requires provider='openai' (or 'ollama' with limited support). "
                "Gemini does not expose logprobs."
            )

        full_prompt = self._build_evaluation_prompt(call, item)

        try:
            raw_text, logprobs = self.client.generate_with_logprobs(full_prompt)
        except Exception as e:
            return {
                "score": None,
                "passed": None,
                "reason": f"API call failed: {e}",
                "confidence": 0.0,
                "raw": {"error": str(e)},
            }

        # Calculate confidence using DeepConf
        if logprobs:
            from .confidence import DeepConfConfidence

            estimator = DeepConfConfidence(strategy=strategy)
            conf_result = estimator.estimate(logprobs=logprobs)
            confidence = conf_result.score
            conf_details = conf_result.details
        else:
            # Fallback for Ollama which doesn't return logprobs
            confidence = 0.5
            conf_details = {"reason": "no_logprobs_available"}

        score, passed, reason, parsed = self._parse_response(raw_text)
        return {
            "score": score,
            "passed": passed,
            "reason": reason,
            "confidence": confidence,
            "raw": {
                "response": parsed,
                "text": raw_text,
                "model": self.model,
                "deepconf_strategy": strategy,
                "deepconf_details": conf_details,
            },
        }

    def as_metric(
        self,
        metric_id: str | None = None,
        threshold: float | None = None,
        description: str | None = None,
    ) -> Metric:
        """Convert this judge to a Metric object.

        Args:
            metric_id: Metric identifier (default: judge name)
            threshold: Success threshold 0-1 (default: from template or 0.5)
            description: Metric description (default: from template)

        Returns:
            Metric object ready for evaluation

        Example:
            metric = LLMJudge.from_template("toxicity").as_metric("safety_check")
        """
        # Get defaults from template if available
        tpl = self.TEMPLATES.get(self.name, {})
        final_id = metric_id or self.name
        final_threshold = (
            threshold if threshold is not None else tpl.get("threshold", 0.5)
        )
        final_description = description or tpl.get(
            "description", "LLM judge evaluation"
        )

        spec = MetricSpec(
            id=final_id,
            name=f"Subjective - {self.name}",
            type="subjective",
            description=final_description,
            config={
                "success_threshold": final_threshold,
                "model": self.model,
            },
        )

        judge = self  # Capture for closure

        def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
            judge_raw = judge.score(call, item)

            # Prefer rubric-style boolean verdicts
            passed_val = None
            for key in ("passed", "pass", "verdict"):
                if key in judge_raw:
                    passed_val = judge_raw.get(key)
                    break
            if isinstance(passed_val, str):
                passed_val = _parse_passed(passed_val)

            if isinstance(passed_val, bool):
                passed = passed_val
                score = 1.0 if passed else 0.0
            else:
                score = judge_raw.get("score")
                passed = score is not None and score >= final_threshold

            # Detect parse failure: both score and passed_val are None means
            # the LLM response could not be parsed as JSON. Record as
            # indeterminate (passed=None) rather than false negative, and flag
            # it so downstream analysis can distinguish parse errors from real
            # failures.
            details: dict = {"judge": judge.name, "reason": judge_raw.get("reason")}
            if passed_val is None and judge_raw.get("score") is None:
                passed = None
                score = None
                details["parse_error"] = True
                logger.warning(
                    "Judge '%s' returned unparseable response for item '%s'",
                    judge.name, item.id,
                )

            return MetricResult(
                metric_id=spec.id,
                item_id=item.id,
                call_id=call.id,
                score=score,
                passed=passed,
                details=details,
                raw_judge=judge_raw,
                input_tokens=judge_raw.get("input_tokens"),
                output_tokens=judge_raw.get("output_tokens"),
                model=judge_raw.get("model"),
            )

        return Metric(spec, handler)


# =============================================================================
# EchoJudge (for testing)
# =============================================================================


class EchoJudge(LLMJudge):
    """Debug judge for tests: returns 1.0 if expected substring is present."""

    def __init__(self):
        super().__init__(name="echo", prompt="echo judge", model="debug")

    def score(self, call: FunctionCall, item: DatasetItem) -> dict[str, Any]:
        expected = item.expected
        text = str(call.output or "")
        passed = expected is not None and str(expected) in text
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "reason": "debug-echo",
            "raw": {"output": text},
        }


__all__ = ["LLMJudge", "EchoJudge", "JUDGE_TEMPLATES"]
