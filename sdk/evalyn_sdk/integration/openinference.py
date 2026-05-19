"""
OpenInference full compliance module.

Converts evalyn spans/traces to the OpenInference semantic convention format,
validates compliance, and exports as JSON.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from ..models import Span

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENINFERENCE_SPAN_KINDS = {
    "LLM",
    "CHAIN",
    "TOOL",
    "RETRIEVER",
    "EMBEDDING",
    "AGENT",
    "RERANKER",
    "GUARDRAIL",
}

SPAN_TYPE_TO_OI: dict[str, str] = {
    "llm_call": "LLM",
    "tool_call": "TOOL",
    "retrieval": "RETRIEVER",
    "node": "CHAIN",
    "agent": "AGENT",
    "embedding": "EMBEDDING",
    "reranker": "RERANKER",
    "scorer": "CHAIN",
    "custom": "CHAIN",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class OpenInferenceSpan:
    """A span in OpenInference-compatible format."""

    name: str
    span_kind: str
    trace_id: str = ""
    span_id: str = ""
    parent_id: str = ""
    start_time: str = ""
    end_time: str = ""
    status_code: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_kind": self.span_kind,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status_code": self.status_code,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenInferenceSpan:
        return cls(
            name=data.get("name", ""),
            span_kind=data.get("span_kind", "CHAIN"),
            trace_id=data.get("trace_id", ""),
            span_id=data.get("span_id", ""),
            parent_id=data.get("parent_id", ""),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            status_code=data.get("status_code", "OK"),
            attributes=data.get("attributes", {}),
        )


@dataclass
class OpenInferenceTrace:
    """A collection of OpenInference spans forming a trace."""

    trace_id: str
    spans: list[OpenInferenceSpan] = field(default_factory=list)
    project_name: str = "evalyn"

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [s.as_dict() for s in self.spans],
            "project_name": self.project_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenInferenceTrace:
        return cls(
            trace_id=data.get("trace_id", ""),
            spans=[
                OpenInferenceSpan.from_dict(s) for s in data.get("spans", [])
            ],
            project_name=data.get("project_name", "evalyn"),
        )


@dataclass
class ComplianceReport:
    """Result of checking OpenInference compliance for a trace."""

    total_spans: int = 0
    compliant_spans: int = 0
    issues: list[str] = field(default_factory=list)
    compliance_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_spans": self.total_spans,
            "compliant_spans": self.compliant_spans,
            "issues": list(self.issues),
            "compliance_rate": self.compliance_rate,
        }

    def format_text(self) -> str:
        lines = [
            f"Total spans: {self.total_spans}",
            f"Compliant: {self.compliant_spans}",
            f"Compliance rate: {self.compliance_rate:.1%}",
        ]
        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def convert_span_to_oi(span: Span) -> OpenInferenceSpan:
    """Convert an evalyn Span to OpenInference format."""
    span_kind = SPAN_TYPE_TO_OI.get(span.span_type, "CHAIN")
    start_time = span.start_time.isoformat() if span.start_time else ""
    end_time = span.end_time.isoformat() if span.end_time else ""
    status_code = "OK" if span.status == "ok" else "ERROR"

    attributes: dict[str, Any] = dict(span.attributes)
    attributes["openinference.span.kind"] = span_kind

    return OpenInferenceSpan(
        name=span.name,
        span_kind=span_kind,
        trace_id=span.parent_id or span.id,
        span_id=span.id,
        parent_id=span.parent_id or "",
        start_time=start_time,
        end_time=end_time,
        status_code=status_code,
        attributes=attributes,
    )


def convert_trace_to_oi(
    spans: list[Span], project_name: str = "evalyn"
) -> OpenInferenceTrace:
    """Convert a list of evalyn Spans into a full OpenInference trace."""
    if not spans:
        return OpenInferenceTrace(trace_id="", project_name=project_name)

    # Use the first root span's id as the trace id, or fall back to first span
    root_spans = [s for s in spans if s.parent_id is None]
    trace_id = root_spans[0].id if root_spans else spans[0].id

    oi_spans = []
    for span in spans:
        oi_span = convert_span_to_oi(span)
        # Override trace_id to be consistent across the trace
        oi_span.trace_id = trace_id
        oi_spans.append(oi_span)

    return OpenInferenceTrace(
        trace_id=trace_id,
        spans=oi_spans,
        project_name=project_name,
    )


def check_compliance(oi_span: OpenInferenceSpan) -> list[str]:
    """Validate a single OpenInference span against the spec.

    Returns a list of issue strings. Empty list means fully compliant.
    """
    issues: list[str] = []

    if not oi_span.name:
        issues.append("missing span name")

    if oi_span.span_kind not in OPENINFERENCE_SPAN_KINDS:
        issues.append(
            f"invalid span_kind: {oi_span.span_kind!r}"
        )

    if not oi_span.span_id:
        issues.append("missing span_id")

    if not oi_span.trace_id:
        issues.append("missing trace_id")

    return issues


def check_trace_compliance(trace: OpenInferenceTrace) -> ComplianceReport:
    """Run compliance checks on all spans in a trace."""
    all_issues: list[str] = []
    compliant = 0

    for span in trace.spans:
        span_issues = check_compliance(span)
        if span_issues:
            for issue in span_issues:
                all_issues.append(f"span {span.span_id or '(no id)'}: {issue}")
        else:
            compliant += 1

    total = len(trace.spans)
    rate = compliant / total if total > 0 else 0.0

    return ComplianceReport(
        total_spans=total,
        compliant_spans=compliant,
        issues=all_issues,
        compliance_rate=rate,
    )


def add_oi_attributes(span: Span) -> Span:
    """Add OpenInference semantic attributes to a span.

    Returns a NEW span instance - does not mutate the original.
    """
    new_span = copy.deepcopy(span)
    span_kind = SPAN_TYPE_TO_OI.get(new_span.span_type, "CHAIN")
    new_span.attributes["openinference.span.kind"] = span_kind
    new_span.attributes["openinference.version"] = "1.0"
    return new_span


def export_as_openinference_json(trace: OpenInferenceTrace) -> str:
    """Serialize an OpenInference trace to a JSON string."""
    return json.dumps(trace.as_dict(), indent=2, default=str)
