"""
Native export to popular LLM observability platforms.

Supports Phoenix, Langfuse, and LangSmith trace formats.
Does NOT hardcode API keys - callers handle authentication.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models import Span


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PhoenixSpan:
    """A span in Phoenix-compatible format."""

    name: str
    span_kind: str = "LLM"
    context: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    start_time_ns: int = 0
    end_time_ns: int = 0
    status: str = "OK"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "span_kind": self.span_kind,
            "context": dict(self.context),
            "attributes": dict(self.attributes),
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhoenixSpan:
        return cls(
            name=data["name"],
            span_kind=data.get("span_kind", "LLM"),
            context=data.get("context", {}),
            attributes=data.get("attributes", {}),
            start_time_ns=data.get("start_time_ns", 0),
            end_time_ns=data.get("end_time_ns", 0),
            status=data.get("status", "OK"),
        )


@dataclass
class ExportConfig:
    """Configuration for exporting spans to an observability platform."""

    platform: str = "phoenix"  # "phoenix" / "langfuse" / "langsmith"
    project_name: str = "evalyn"
    include_inputs: bool = True
    include_outputs: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "project_name": self.project_name,
            "include_inputs": self.include_inputs,
            "include_outputs": self.include_outputs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExportConfig:
        return cls(
            platform=data.get("platform", "phoenix"),
            project_name=data.get("project_name", "evalyn"),
            include_inputs=data.get("include_inputs", True),
            include_outputs=data.get("include_outputs", True),
        )


@dataclass
class PlatformExportResult:
    """Result of exporting spans to a platform."""

    platform: str
    spans_exported: int = 0
    format: str = ""
    file_path: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "spans_exported": self.spans_exported,
            "format": self.format,
            "file_path": self.file_path,
        }

    def format_text(self) -> str:
        lines = [
            f"Export: {self.platform}",
            f"Spans exported: {self.spans_exported}",
            f"Format: {self.format}",
        ]
        if self.file_path:
            lines.append(f"File: {self.file_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversion Helpers
# ---------------------------------------------------------------------------

_SPAN_TYPE_TO_KIND: Dict[str, str] = {
    "llm_call": "LLM",
    "tool_call": "TOOL",
    "retrieval": "RETRIEVER",
    "agent": "AGENT",
    "graph": "CHAIN",
    "node": "CHAIN",
    "session": "CHAIN",
    "scorer": "EVALUATOR",
    "custom": "CHAIN",
}


def _datetime_to_ns(dt) -> int:
    """Convert datetime to nanoseconds since epoch."""
    if dt is None:
        return 0
    return int(dt.timestamp() * 1_000_000_000)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def convert_to_phoenix_span(span: Span) -> PhoenixSpan:
    """Convert an evalyn Span to Phoenix format."""
    span_kind = _SPAN_TYPE_TO_KIND.get(span.span_type, "CHAIN")
    context: Dict[str, str] = {
        "trace_id": span.parent_id or span.id,
        "span_id": span.id,
    }
    attributes: Dict[str, Any] = dict(span.attributes)
    attributes["evalyn.span_type"] = span.span_type

    return PhoenixSpan(
        name=span.name,
        span_kind=span_kind,
        context=context,
        attributes=attributes,
        start_time_ns=_datetime_to_ns(span.start_time),
        end_time_ns=_datetime_to_ns(span.end_time),
        status="OK" if span.status == "ok" else "ERROR",
    )


def convert_to_langfuse_format(span: Span) -> Dict[str, Any]:
    """Convert an evalyn Span to Langfuse-compatible dict."""
    result: Dict[str, Any] = {
        "id": span.id,
        "name": span.name,
        "type": "generation" if span.span_type == "llm_call" else "span",
        "traceId": span.parent_id or span.id,
        "startTime": span.start_time.isoformat() if span.start_time else None,
        "endTime": span.end_time.isoformat() if span.end_time else None,
        "metadata": dict(span.attributes),
        "level": "DEFAULT" if span.status == "ok" else "ERROR",
    }
    if span.parent_id:
        result["parentObservationId"] = span.parent_id
    return result


def convert_to_langsmith_format(span: Span) -> Dict[str, Any]:
    """Convert an evalyn Span to LangSmith-compatible dict."""
    result: Dict[str, Any] = {
        "id": span.id,
        "name": span.name,
        "run_type": "llm" if span.span_type == "llm_call" else "chain",
        "start_time": span.start_time.isoformat() if span.start_time else None,
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "extra": dict(span.attributes),
        "error": None if span.status == "ok" else span.status,
    }
    if span.parent_id:
        result["parent_run_id"] = span.parent_id
    return result


def export_spans(spans: List[Span], config: ExportConfig) -> PlatformExportResult:
    """Export all spans in the format specified by config."""
    if config.platform == "phoenix":
        content = format_as_phoenix_json([convert_to_phoenix_span(s) for s in spans])
        fmt = "phoenix_json"
    elif config.platform == "langfuse":
        content = format_as_langfuse_json(spans)
        fmt = "langfuse_json"
    elif config.platform == "langsmith":
        converted = [convert_to_langsmith_format(s) for s in spans]
        content = json.dumps(converted, indent=2, default=str)
        fmt = "langsmith_json"
    else:
        raise ValueError(f"Unsupported platform: {config.platform!r}")

    return PlatformExportResult(
        platform=config.platform,
        spans_exported=len(spans),
        format=fmt,
        file_path="",
    )


def format_as_phoenix_json(phoenix_spans: List[PhoenixSpan]) -> str:
    """Serialize Phoenix spans to a JSON string for import."""
    return json.dumps([s.as_dict() for s in phoenix_spans], indent=2, default=str)


def format_as_langfuse_json(spans: List[Span]) -> str:
    """Serialize evalyn spans to Langfuse-compatible JSON string."""
    converted = [convert_to_langfuse_format(s) for s in spans]
    return json.dumps(converted, indent=2, default=str)
