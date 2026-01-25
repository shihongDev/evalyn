"""
EvalUnit builders that discover evaluatable units from trace structure.

Each builder implements a strategy for finding units of a specific type
within a FunctionCall's span tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import uuid

from ..models import EvalUnit, FunctionCall, Span


class EvalUnitBuilder(ABC):
    """Protocol for discovering evaluatable units from trace structure."""

    @property
    @abstractmethod
    def unit_type(self) -> str:
        """The type of units this builder creates."""
        pass

    @abstractmethod
    def discover(self, call: FunctionCall) -> List[EvalUnit]:
        """Discover all evaluatable units of this type from a FunctionCall."""
        pass


class OutcomeBuilder(EvalUnitBuilder):
    """
    Discovers full-trace outcome units (backward-compatible default).

    Creates exactly one unit per FunctionCall representing the entire trace.
    This is the default behavior that preserves existing evaluation semantics.
    """

    @property
    def unit_type(self) -> str:
        return "outcome"

    def discover(self, call: FunctionCall) -> List[EvalUnit]:
        return [
            EvalUnit(
                id=f"outcome-{call.id}",
                unit_type="outcome",
                call_id=call.id,
                span_ids=[s.id for s in call.spans],
                context={
                    "function_name": call.function_name,
                    "has_error": call.error is not None,
                },
            )
        ]


class SingleTurnBuilder(EvalUnitBuilder):
    """
    Discovers single-turn LLM call units.

    Creates one unit per llm_call span, capturing the input->output
    of individual LLM invocations within the trace.
    """

    @property
    def unit_type(self) -> str:
        return "single_turn"

    def discover(self, call: FunctionCall) -> List[EvalUnit]:
        units = []
        for span in call.spans:
            if span.span_type == "llm_call":
                units.append(
                    EvalUnit(
                        id=f"single_turn-{span.id}",
                        unit_type="single_turn",
                        call_id=call.id,
                        span_ids=[span.id],
                        context={
                            "model": span.attributes.get("model"),
                            "span_name": span.name,
                            "duration_ms": span.duration_ms,
                        },
                    )
                )
        return units


class ToolUseBuilder(EvalUnitBuilder):
    """
    Discovers tool use units.

    Creates one unit per tool_call span, capturing tool invocations
    and their results for evaluation.
    """

    @property
    def unit_type(self) -> str:
        return "tool_use"

    def discover(self, call: FunctionCall) -> List[EvalUnit]:
        units = []
        for span in call.spans:
            if span.span_type == "tool_call":
                # Find associated tool_result span if any
                result_span = self._find_result_span(call.spans, span)
                span_ids = [span.id]
                if result_span:
                    span_ids.append(result_span.id)

                units.append(
                    EvalUnit(
                        id=f"tool_use-{span.id}",
                        unit_type="tool_use",
                        call_id=call.id,
                        span_ids=span_ids,
                        context={
                            "tool_name": span.name,
                            "has_result": result_span is not None,
                            "status": span.status,
                        },
                    )
                )
        return units

    def _find_result_span(
        self, spans: List[Span], tool_span: Span
    ) -> Optional[Span]:
        """Find the tool_result span associated with a tool_call span."""
        for span in spans:
            if span.span_type == "tool_result" and span.parent_id == tool_span.id:
                return span
        return None


class MultiTurnBuilder(EvalUnitBuilder):
    """
    Discovers multi-turn conversation units.

    Groups consecutive llm_call spans that share a parent into
    conversation units for evaluating multi-turn interactions.
    """

    @property
    def unit_type(self) -> str:
        return "multi_turn"

    def discover(self, call: FunctionCall) -> List[EvalUnit]:
        # Group llm_calls by their parent span
        by_parent: Dict[Optional[str], List[Span]] = {}
        for span in call.spans:
            if span.span_type == "llm_call":
                parent = span.parent_id
                if parent not in by_parent:
                    by_parent[parent] = []
                by_parent[parent].append(span)

        units = []
        for parent_id, spans in by_parent.items():
            if len(spans) > 1:
                # Sort by start_time
                sorted_spans = sorted(spans, key=lambda s: s.start_time)
                units.append(
                    EvalUnit(
                        id=f"multi_turn-{uuid.uuid4().hex[:8]}",
                        unit_type="multi_turn",
                        call_id=call.id,
                        span_ids=[s.id for s in sorted_spans],
                        context={
                            "turn_count": len(sorted_spans),
                            "parent_span_id": parent_id,
                        },
                    )
                )
        return units


class CustomBuilder(EvalUnitBuilder):
    """
    Discovers custom user-defined units.

    Looks for spans marked with custom evaluation boundaries via
    the 'eval_boundary' attribute.
    """

    @property
    def unit_type(self) -> str:
        return "custom"

    def discover(self, call: FunctionCall) -> List[EvalUnit]:
        units = []
        for span in call.spans:
            if span.attributes.get("eval_boundary"):
                units.append(
                    EvalUnit(
                        id=f"custom-{span.id}",
                        unit_type="custom",
                        call_id=call.id,
                        span_ids=[span.id],
                        context=span.attributes,
                    )
                )
        return units


# Registry of all builders
_BUILDERS: Dict[str, Type[EvalUnitBuilder]] = {
    "outcome": OutcomeBuilder,
    "single_turn": SingleTurnBuilder,
    "tool_use": ToolUseBuilder,
    "multi_turn": MultiTurnBuilder,
    "custom": CustomBuilder,
}


def get_default_builders() -> List[EvalUnitBuilder]:
    """Get the default builder (OutcomeBuilder only for backward compat)."""
    return [OutcomeBuilder()]


def get_builder_for_type(unit_type: str) -> Optional[EvalUnitBuilder]:
    """Get a builder instance for a specific unit type."""
    builder_cls = _BUILDERS.get(unit_type)
    return builder_cls() if builder_cls else None


def get_builders_for_types(unit_types: List[str]) -> List[EvalUnitBuilder]:
    """Get builder instances for multiple unit types."""
    builders = []
    for ut in unit_types:
        builder = get_builder_for_type(ut)
        if builder:
            builders.append(builder)
    return builders if builders else get_default_builders()
