"""
EvalView projection for normalizing units into evaluatable views.

Transforms EvalUnits into EvalViews that provide a consistent interface
for metric evaluation regardless of unit type.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...models import EvalUnit, EvalView, FunctionCall, Span


def project_unit(unit: EvalUnit, call: FunctionCall) -> EvalView:
    """
    Project an EvalUnit into an EvalView for metric evaluation.

    The projection extracts input/output based on unit type:
    - outcome: Full call inputs/output
    - single_turn: LLM messages from span attributes
    - tool_use: Tool request/result from span attributes
    - multi_turn: Concatenated conversation turns
    - custom: User-defined content from span attributes
    """
    projector = _PROJECTORS.get(unit.unit_type, _project_default)
    return projector(unit, call)


def _project_outcome(unit: EvalUnit, call: FunctionCall) -> EvalView:
    """Project full-trace outcome unit."""
    return EvalView(
        unit_id=unit.id,
        unit_type=unit.unit_type,
        input=call.inputs,
        output=call.output,
        context={
            "function_name": call.function_name,
            "error": call.error,
            "duration_ms": call.duration_ms,
            **unit.context,
        },
    )


def _project_single_turn(unit: EvalUnit, call: FunctionCall) -> EvalView:
    """Project single LLM call unit."""
    span = _get_span_by_id(call.spans, unit.span_ids[0]) if unit.span_ids else None

    if span:
        # Extract input/output from span attributes
        input_data = span.attributes.get("input") or span.attributes.get("messages")
        output_data = span.attributes.get("output") or span.attributes.get("response")
    else:
        input_data = None
        output_data = None

    return EvalView(
        unit_id=unit.id,
        unit_type=unit.unit_type,
        input=input_data,
        output=output_data,
        context={
            "model": span.attributes.get("model") if span else None,
            "span_name": span.name if span else None,
            "span_id": span.id if span else None,
            **unit.context,
        },
    )


def _project_tool_use(unit: EvalUnit, call: FunctionCall) -> EvalView:
    """Project tool use unit."""
    spans = [s for sid in unit.span_ids if (s := _get_span_by_id(call.spans, sid))]

    # Find tool_call and tool_result spans
    tool_call_span = next((s for s in spans if s.span_type == "tool_call"), None)
    tool_result_span = next((s for s in spans if s.span_type == "tool_result"), None)

    # Extract input from tool_call span
    if tool_call_span:
        input_data = {
            "tool_name": tool_call_span.name,
            "arguments": tool_call_span.attributes.get("arguments")
            or tool_call_span.attributes.get("input"),
        }
    else:
        input_data = None

    # Extract output: prefer tool_result, fallback to tool_call
    if tool_result_span:
        output_data = tool_result_span.attributes.get("result") or tool_result_span.attributes.get("output")
    elif tool_call_span:
        output_data = tool_call_span.attributes.get("result") or tool_call_span.attributes.get("output")
    else:
        output_data = None

    return EvalView(
        unit_id=unit.id,
        unit_type=unit.unit_type,
        input=input_data,
        output=output_data,
        context={
            "tool_name": tool_call_span.name if tool_call_span else None,
            "status": tool_call_span.status if tool_call_span else None,
            **unit.context,
        },
    )


def _project_multi_turn(unit: EvalUnit, call: FunctionCall) -> EvalView:
    """Project multi-turn conversation unit."""
    spans = [s for sid in unit.span_ids if (s := _get_span_by_id(call.spans, sid))]
    spans = sorted(spans, key=lambda s: s.start_time)

    # Build conversation as list of turns
    turns: List[Dict[str, Any]] = []
    for span in spans:
        turn = {
            "input": span.attributes.get("input") or span.attributes.get("messages"),
            "output": span.attributes.get("output") or span.attributes.get("response"),
            "model": span.attributes.get("model"),
        }
        turns.append(turn)

    # Input is all inputs, output is final output
    all_inputs = [t["input"] for t in turns]
    final_output = turns[-1]["output"] if turns else None

    return EvalView(
        unit_id=unit.id,
        unit_type=unit.unit_type,
        input=all_inputs,
        output=final_output,
        context={
            "turns": turns,
            "turn_count": len(turns),
            **unit.context,
        },
    )


def _project_custom(unit: EvalUnit, call: FunctionCall) -> EvalView:
    """Project custom user-defined unit."""
    span = _get_span_by_id(call.spans, unit.span_ids[0]) if unit.span_ids else None

    if span:
        input_data = span.attributes.get("eval_input") or span.attributes.get("input")
        output_data = span.attributes.get("eval_output") or span.attributes.get("output")
    else:
        input_data = unit.context.get("input")
        output_data = unit.context.get("output")

    return EvalView(
        unit_id=unit.id,
        unit_type=unit.unit_type,
        input=input_data,
        output=output_data,
        context=unit.context,
    )


def _project_default(unit: EvalUnit, call: FunctionCall) -> EvalView:
    """Default projection for unknown unit types."""
    return EvalView(
        unit_id=unit.id,
        unit_type=unit.unit_type,
        input=unit.context.get("input"),
        output=unit.context.get("output"),
        context=unit.context,
    )


def _get_span_by_id(spans: List[Span], span_id: str) -> Optional[Span]:
    """Find a span by ID."""
    for span in spans:
        if span.id == span_id:
            return span
    return None


# Projector registry
_PROJECTORS = {
    "outcome": _project_outcome,
    "single_turn": _project_single_turn,
    "tool_use": _project_tool_use,
    "multi_turn": _project_multi_turn,
    "custom": _project_custom,
}
