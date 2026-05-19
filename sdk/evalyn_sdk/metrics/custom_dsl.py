"""Custom metric DSL: define metrics via YAML/dict without writing Python.

Supports two metric types:
- objective: evaluated via safe expression on item/output data
- subjective: evaluated via LLM judge with template variables

Example YAML:
    custom_metrics:
      - id: short_response
        type: objective
        expression: "len(output) < 500"
        description: "Output must be under 500 characters"
      - id: domain_accuracy
        type: subjective
        prompt: "Evaluate if {{output}} correctly answers {{input}}"
        rubric:
          - "Response must be factually accurate"
          - "Response must address the specific question"
        threshold: 0.5
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..models import (
    DatasetItem,
    FunctionCall,
    Metric,
    MetricResult,
    MetricSpec,
)


def build_custom_objective_metric(
    metric_id: str,
    expression: str,
    description: str = "",
    threshold: float = 0.5,
) -> Metric:
    """Build an objective metric from a safe expression string.

    Supported expressions (restricted for safety):
    - len(output) < N, len(input) > N
    - "keyword" in output, "keyword" not in output
    - output == "exact", output != "exact"
    - output_length > N, input_length > N

    Args:
        metric_id: Unique metric identifier.
        expression: Safe expression string to evaluate.
        description: Human-readable description.
        threshold: Pass/fail threshold (not used for boolean expressions).

    Returns:
        Metric instance.

    Raises:
        ValueError: If expression is not safe or cannot be parsed.
    """
    evaluator = _parse_expression(expression)

    spec = MetricSpec(
        id=metric_id,
        name=metric_id,
        type="objective",
        description=description or f"Custom: {expression}",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        try:
            input_text = _to_text(item.input)
            output_text = _to_text(call.output)
            result = evaluator(input_text, output_text)

            if isinstance(result, bool):
                score = 1.0 if result else 0.0
                passed = result
            elif isinstance(result, (int, float)):
                score = float(result)
                passed = score >= threshold
            else:
                score = 0.0
                passed = False
        except Exception as e:
            score = 0.0
            passed = False
            return MetricResult(
                metric_id=spec.id, item_id=item.id, call_id=call.id,
                score=score, passed=passed,
                details={"error": str(e), "expression": expression},
            )

        return MetricResult(
            metric_id=spec.id, item_id=item.id, call_id=call.id,
            score=score, passed=passed,
            details={"expression": expression},
        )

    return Metric(spec, handler)


def build_custom_subjective_spec(
    metric_id: str,
    prompt: str,
    rubric: list[str] | None = None,
    description: str = "",
    threshold: float = 0.5,
) -> MetricSpec:
    """Build a MetricSpec for a custom subjective metric.

    Template variables {{input}}, {{output}}, {{expected}} will be
    resolved at evaluation time by the LLM judge.

    Args:
        metric_id: Unique metric identifier.
        prompt: Judge prompt template with {{variable}} placeholders.
        rubric: List of evaluation criteria.
        description: Human-readable description.
        threshold: Pass/fail threshold.

    Returns:
        MetricSpec ready for build_subjective_metric().
    """
    config: dict[str, Any] = {
        "prompt": prompt,
        "threshold": threshold,
    }
    if rubric:
        config["rubric"] = rubric

    return MetricSpec(
        id=metric_id,
        name=metric_id,
        type="subjective",
        description=description or f"Custom judge: {metric_id}",
        config=config,
    )


def load_custom_metrics_from_config(
    config: dict[str, Any],
) -> list[Metric]:
    """Load custom metrics from evalyn.yaml config.

    Expected format:
        custom_metrics:
          - id: my_check
            type: objective
            expression: "len(output) < 500"
          - id: my_judge
            type: subjective
            prompt: "Evaluate ..."
            rubric: ["criteria 1", "criteria 2"]

    Args:
        config: Parsed evalyn.yaml config dict.

    Returns:
        List of built Metric instances (objective only; subjective returns specs).
    """
    custom_defs = config.get("custom_metrics", [])
    metrics = []

    for defn in custom_defs:
        metric_id = defn.get("id")
        metric_type = defn.get("type", "objective")

        if not metric_id:
            continue

        if metric_type == "objective":
            expression = defn.get("expression", "")
            if expression:
                try:
                    m = build_custom_objective_metric(
                        metric_id=metric_id,
                        expression=expression,
                        description=defn.get("description", ""),
                        threshold=float(defn.get("threshold", 0.5)),
                    )
                    metrics.append(m)
                except ValueError:
                    pass  # skip unparseable expressions

    return metrics


# =============================================================================
# Safe expression parser (NO eval/exec)
# =============================================================================

def _parse_expression(expr: str):
    """Parse a safe expression into a callable (input_text, output_text) -> result.

    Raises ValueError if expression cannot be safely parsed.
    """
    expr = expr.strip()

    # len(output) op N
    m = re.match(r'^len\(output\)\s*([<>=!]+)\s*(\d+)$', expr)
    if m:
        op, val = m.group(1), int(m.group(2))
        return lambda inp, out: _compare(len(out), op, val)

    # len(input) op N
    m = re.match(r'^len\(input\)\s*([<>=!]+)\s*(\d+)$', expr)
    if m:
        op, val = m.group(1), int(m.group(2))
        return lambda inp, out: _compare(len(inp), op, val)

    # output_length op N
    m = re.match(r'^output_length\s*([<>=!]+)\s*(\d+)$', expr)
    if m:
        op, val = m.group(1), int(m.group(2))
        return lambda inp, out: _compare(len(out), op, val)

    # input_length op N
    m = re.match(r'^input_length\s*([<>=!]+)\s*(\d+)$', expr)
    if m:
        op, val = m.group(1), int(m.group(2))
        return lambda inp, out: _compare(len(inp), op, val)

    # "keyword" in output
    m = re.match(r'^"([^"]+)"\s+in\s+output$', expr)
    if m:
        keyword = m.group(1).lower()
        return lambda inp, out: keyword in out.lower()

    # "keyword" not in output
    m = re.match(r'^"([^"]+)"\s+not\s+in\s+output$', expr)
    if m:
        keyword = m.group(1).lower()
        return lambda inp, out: keyword not in out.lower()

    # "keyword" in input
    m = re.match(r'^"([^"]+)"\s+in\s+input$', expr)
    if m:
        keyword = m.group(1).lower()
        return lambda inp, out: keyword in inp.lower()

    # output == "exact"
    m = re.match(r'^output\s*==\s*"([^"]*)"$', expr)
    if m:
        expected = m.group(1)
        return lambda inp, out: out.strip() == expected

    # output != "exact"
    m = re.match(r'^output\s*!=\s*"([^"]*)"$', expr)
    if m:
        expected = m.group(1)
        return lambda inp, out: out.strip() != expected

    raise ValueError(f"Cannot parse expression: {expr!r}. "
                     f"Supported: len(output) < N, \"keyword\" in output, output == \"value\"")


def _compare(actual: int, op: str, expected: int) -> bool:
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    return False


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    import json
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)
