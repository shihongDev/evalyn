"""Metric score explanations: human-readable reasons for objective scores.

Provides explain() functions that describe WHY a metric scored the way
it did, beyond just the pass/fail result.
"""

from __future__ import annotations

from typing import Any


def explain_metric_result(result) -> str:
    """Generate a human-readable explanation for a metric result.

    Args:
        result: MetricResult object.

    Returns:
        Explanation string describing why the metric scored this way.
    """
    mid = result.metric_id

    # Try metric-specific explainers
    explainer = _EXPLAINERS.get(mid)
    if explainer:
        return explainer(result)

    # Generic explanation
    return _generic_explanation(result)


def _generic_explanation(result) -> str:
    """Generic explanation for metrics without a specific explainer."""
    mid = result.metric_id
    score = result.score
    passed = result.passed
    details = result.details or {}

    parts = [f"{mid}:"]

    if passed is True:
        parts.append("PASS")
    elif passed is False:
        parts.append("FAIL")
    else:
        parts.append("N/A")

    if score is not None:
        parts.append(f"(score={score:.2f})")

    # Extract useful details
    reason = details.get("reason")
    if reason:
        parts.append(f"- {reason}")

    error = details.get("error")
    if error:
        parts.append(f"- Error: {error}")

    return " ".join(parts)


def _explain_json_valid(result) -> str:
    score = result.score
    if score == 1.0:
        return "json_valid: PASS - output is valid JSON"
    return "json_valid: FAIL - output is not valid JSON (parse error)"


def _explain_nonempty_output(result) -> str:
    if result.passed:
        return "nonempty_output: PASS - output contains content"
    return "nonempty_output: FAIL - output is empty or whitespace-only"


def _explain_regex_match(result) -> str:
    details = result.details or {}
    pattern = details.get("pattern", "?")
    if result.passed:
        return f"regex_match: PASS - output matches pattern /{pattern}/"
    return f"regex_match: FAIL - output does not match pattern /{pattern}/"


def _explain_token_length(result) -> str:
    details = result.details or {}
    length = details.get("token_length", details.get("length", "?"))
    return f"token_length: output is {length} tokens (score={result.score})"


def _explain_latency(result) -> str:
    score = result.score
    if score is None:
        return "latency_ms: no latency recorded"
    return f"latency_ms: execution took {score:.0f}ms"


def _explain_cost(result) -> str:
    score = result.score
    if score is not None:
        return f"cost: evaluation cost ${score:.6f}"
    return "cost: no cost data available"


def _explain_bleu(result) -> str:
    score = result.score or 0.0
    if score >= 0.5:
        return f"bleu: GOOD overlap with reference (score={score:.2f})"
    if score >= 0.2:
        return f"bleu: MODERATE overlap with reference (score={score:.2f})"
    return f"bleu: LOW overlap with reference (score={score:.2f})"


def _explain_prompt_injection(result) -> str:
    details = result.details or {}
    matches = details.get("matches", [])
    if result.passed:
        return "prompt_injection_check: PASS - no injection patterns detected"
    return f"prompt_injection_check: FAIL - detected: {', '.join(matches[:3])}"


def _explain_tool_call_accuracy(result) -> str:
    details = result.details or {}
    matched = details.get("matched", 0)
    total = details.get("total_expected", 0)
    if result.score is None:
        return "tool_call_accuracy: N/A - no expected_tools defined"
    return f"tool_call_accuracy: {matched}/{total} expected tools called correctly (score={result.score:.2f})"


def _explain_tool_call_sequence(result) -> str:
    details = result.details or {}
    lcs = details.get("lcs_length", 0)
    if result.score is None:
        return "tool_call_sequence: N/A - no expected_tool_sequence defined"
    return f"tool_call_sequence: longest common subsequence = {lcs} (score={result.score:.2f})"


# Registry of metric-specific explainers
_EXPLAINERS: dict[str, Any] = {
    "json_valid": _explain_json_valid,
    "nonempty_output": _explain_nonempty_output,
    "regex_match": _explain_regex_match,
    "token_length": _explain_token_length,
    "latency_ms": _explain_latency,
    "cost": _explain_cost,
    "bleu": _explain_bleu,
    "prompt_injection_check": _explain_prompt_injection,
    "tool_call_accuracy": _explain_tool_call_accuracy,
    "tool_call_sequence": _explain_tool_call_sequence,
}


def register_explainer(metric_id: str, explainer_fn) -> None:
    """Register a custom explanation function for a metric.

    Args:
        metric_id: Metric to explain.
        explainer_fn: Function(MetricResult) -> str.
    """
    _EXPLAINERS[metric_id] = explainer_fn


def list_explainable_metrics() -> list:
    """Return list of metrics that have custom explainers."""
    return sorted(_EXPLAINERS.keys())
