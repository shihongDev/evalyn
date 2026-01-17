"""
Span converter for normalizing spans from different sources.

Converts spans from:
- OpenTelemetry ReadableSpan (for OTEL-native SDKs like Google ADK)
- Hook events (for hook-based SDKs like Anthropic Agent SDK)

To Evalyn's Span model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ...models import Span
from .conventions import (
    LLMAttributes,
    ToolAttributes,
    IOAttributes,
    get_span_type,
)


class SpanConverter:
    """
    Converts spans from various sources to Evalyn's Span model.

    Supports:
    - OTEL ReadableSpan (from opentelemetry-sdk)
    - Hook event data (from SDK callbacks)
    """

    @classmethod
    def from_otel_span(
        cls,
        otel_span: Any,  # ReadableSpan from opentelemetry-sdk
        parent_evalyn_id: Optional[str] = None,
    ) -> Span:
        """
        Convert an OpenTelemetry ReadableSpan to an Evalyn Span.

        Args:
            otel_span: The OTEL span to convert
            parent_evalyn_id: Override parent ID with Evalyn span ID

        Returns:
            Evalyn Span object
        """
        # Extract basic info
        name = otel_span.name
        attributes = dict(otel_span.attributes) if otel_span.attributes else {}

        # Determine span type from OpenInference conventions
        openinference_kind = attributes.get("openinference.span.kind", "")
        span_type = (
            get_span_type(openinference_kind) if openinference_kind else "custom"
        )

        # Override span type based on name heuristics if not set
        if span_type == "custom":
            name_lower = name.lower()
            if "llm" in name_lower or "generate" in name_lower or "chat" in name_lower:
                span_type = "llm_call"
            elif "tool" in name_lower:
                span_type = "tool_call"
            elif "agent" in name_lower:
                span_type = "agent"

        # Convert timestamps
        start_time = cls._ns_to_datetime(otel_span.start_time)
        end_time = (
            cls._ns_to_datetime(otel_span.end_time) if otel_span.end_time else None
        )

        # Determine status
        status = "ok"
        if otel_span.status:
            from opentelemetry.trace import StatusCode

            if otel_span.status.status_code == StatusCode.ERROR:
                status = "error"

        # Convert span ID and parent ID
        span_id = cls._hex_to_uuid(format(otel_span.context.span_id, "016x"))

        parent_id = parent_evalyn_id
        if parent_id is None and otel_span.parent:
            parent_id = cls._hex_to_uuid(format(otel_span.parent.span_id, "016x"))

        # Extract Evalyn-relevant attributes
        evalyn_attrs = cls._extract_evalyn_attributes(attributes)

        return Span(
            id=span_id,
            name=name,
            span_type=span_type,
            parent_id=parent_id,
            start_time=start_time,
            end_time=end_time,
            status=status,
            attributes=evalyn_attrs,
        )

    @classmethod
    def from_hook_event(
        cls,
        event_type: str,
        event_data: Dict[str, Any],
        parent_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Span:
        """
        Create an Evalyn Span from hook event data.

        Args:
            event_type: Type of hook event (e.g., 'tool_start', 'llm_start')
            event_data: Data from the hook callback
            parent_id: Parent span ID
            span_id: Optional span ID (auto-generated if not provided)

        Returns:
            Evalyn Span object (in running state, call finish() when done)
        """
        # Determine span type and name from event type
        if "tool" in event_type.lower():
            span_type = "tool_call"
            name = event_data.get("tool_name", event_data.get("name", "tool"))
        elif "llm" in event_type.lower() or "model" in event_type.lower():
            span_type = "llm_call"
            model = event_data.get("model", event_data.get("model_name", "llm"))
            provider = event_data.get("provider", "")
            name = f"{provider}:{model}" if provider else model
        elif "agent" in event_type.lower():
            span_type = "agent"
            name = event_data.get("agent_name", event_data.get("name", "agent"))
        else:
            span_type = "custom"
            name = event_data.get("name", event_type)

        # Create span
        return Span.new(
            name=name,
            span_type=span_type,
            parent_id=parent_id,
            **event_data,
        )

    @classmethod
    def _ns_to_datetime(cls, ns: Optional[int]) -> datetime:
        """Convert nanoseconds since epoch to datetime."""
        if ns is None:
            return datetime.now(timezone.utc)
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)

    @classmethod
    def _hex_to_uuid(cls, hex_id: str) -> str:
        """Convert hex span ID to UUID format for consistency."""
        # Pad to 32 chars and format as UUID
        hex_padded = hex_id.zfill(32)
        return f"{hex_padded[:8]}-{hex_padded[8:12]}-{hex_padded[12:16]}-{hex_padded[16:20]}-{hex_padded[20:32]}"

    @classmethod
    def _extract_evalyn_attributes(cls, otel_attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Evalyn-relevant attributes from OTEL attributes."""
        evalyn_attrs = {}

        # LLM attributes
        if LLMAttributes.MODEL_NAME in otel_attrs:
            evalyn_attrs["model"] = otel_attrs[LLMAttributes.MODEL_NAME]
        if LLMAttributes.PROVIDER in otel_attrs:
            evalyn_attrs["provider"] = otel_attrs[LLMAttributes.PROVIDER]
        if LLMAttributes.TOKEN_COUNT_PROMPT in otel_attrs:
            evalyn_attrs["input_tokens"] = otel_attrs[LLMAttributes.TOKEN_COUNT_PROMPT]
        if LLMAttributes.TOKEN_COUNT_COMPLETION in otel_attrs:
            evalyn_attrs["output_tokens"] = otel_attrs[
                LLMAttributes.TOKEN_COUNT_COMPLETION
            ]
        if LLMAttributes.TOKEN_COUNT_TOTAL in otel_attrs:
            evalyn_attrs["total_tokens"] = otel_attrs[LLMAttributes.TOKEN_COUNT_TOTAL]

        # Tool attributes
        if ToolAttributes.NAME in otel_attrs:
            evalyn_attrs["tool_name"] = otel_attrs[ToolAttributes.NAME]
        if ToolAttributes.PARAMETERS in otel_attrs:
            evalyn_attrs["tool_params"] = otel_attrs[ToolAttributes.PARAMETERS]

        # I/O attributes
        if IOAttributes.INPUT_VALUE in otel_attrs:
            evalyn_attrs["input"] = cls._truncate(
                str(otel_attrs[IOAttributes.INPUT_VALUE]), 1000
            )
        if IOAttributes.OUTPUT_VALUE in otel_attrs:
            evalyn_attrs["output"] = cls._truncate(
                str(otel_attrs[IOAttributes.OUTPUT_VALUE]), 1000
            )

        # Pass through any evalyn.* attributes
        for key, value in otel_attrs.items():
            if key.startswith("evalyn."):
                evalyn_attrs[key.replace("evalyn.", "")] = value

        return evalyn_attrs

    @classmethod
    def _truncate(cls, s: str, max_len: int) -> str:
        """Truncate string to max length."""
        return s[:max_len] + "..." if len(s) > max_len else s
