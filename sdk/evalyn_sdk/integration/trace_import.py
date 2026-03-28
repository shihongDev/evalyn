"""
Trace import from external platforms.

Bring existing traces from Phoenix, Langfuse, LangSmith, or generic
sources into evalyn for evaluation. Provides dataclasses and pure
functions for parsing, mapping, and validating imported traces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ImportedTrace:
    """A trace imported from an external platform."""

    trace_id: str
    spans: List[Dict[str, Any]] = field(default_factory=list)
    source_platform: str = ""
    imported_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": list(self.spans),
            "source_platform": self.source_platform,
            "imported_at": self.imported_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImportedTrace:
        return cls(
            trace_id=data.get("trace_id", ""),
            spans=data.get("spans", []),
            source_platform=data.get("source_platform", ""),
            imported_at=data.get("imported_at", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TraceImportConfig:
    """Configuration for trace import."""

    source_platform: str = "phoenix"
    map_span_types: bool = True
    preserve_ids: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_platform": self.source_platform,
            "map_span_types": self.map_span_types,
            "preserve_ids": self.preserve_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceImportConfig:
        return cls(
            source_platform=data.get("source_platform", "phoenix"),
            map_span_types=data.get("map_span_types", True),
            preserve_ids=data.get("preserve_ids", True),
        )


@dataclass
class TraceImportResult:
    """Result of a trace import operation."""

    traces: List[ImportedTrace] = field(default_factory=list)
    total_imported: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "traces": [t.as_dict() for t in self.traces],
            "total_imported": self.total_imported,
            "skipped": self.skipped,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceImportResult:
        return cls(
            traces=[ImportedTrace.from_dict(t) for t in data.get("traces", [])],
            total_imported=data.get("total_imported", 0),
            skipped=data.get("skipped", 0),
            errors=data.get("errors", []),
        )

    def format_text(self) -> str:
        lines = [
            f"Imported: {self.total_imported} trace(s)",
            f"Skipped: {self.skipped}",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for err in self.errors:
                lines.append(f"  - {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Span Type Mapping
# ---------------------------------------------------------------------------

_PHOENIX_SPAN_MAP: Dict[str, str] = {
    "LLM": "llm_call",
    "TOOL": "tool_call",
    "RETRIEVER": "retrieval",
    "CHAIN": "graph",
    "AGENT": "agent",
    "EMBEDDING": "custom",
}

_LANGFUSE_SPAN_MAP: Dict[str, str] = {
    "GENERATION": "llm_call",
    "SPAN": "custom",
    "EVENT": "custom",
    "TOOL": "tool_call",
    "RETRIEVAL": "retrieval",
    "AGENT": "agent",
}

_LANGSMITH_SPAN_MAP: Dict[str, str] = {
    "llm": "llm_call",
    "tool": "tool_call",
    "retriever": "retrieval",
    "chain": "graph",
    "agent": "agent",
}


def map_span_type(external_type: str, platform: str) -> str:
    """Map an external span type to an evalyn span type.

    Args:
        external_type: The span type string from the external platform.
        platform: One of "phoenix", "langfuse", "langsmith", "generic".

    Returns:
        The corresponding evalyn span type, or "custom" if unmapped.
    """
    if platform == "phoenix":
        return _PHOENIX_SPAN_MAP.get(external_type, "custom")
    if platform == "langfuse":
        return _LANGFUSE_SPAN_MAP.get(external_type, "custom")
    if platform == "langsmith":
        return _LANGSMITH_SPAN_MAP.get(external_type, "custom")
    return "custom"


# ---------------------------------------------------------------------------
# Import Functions
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_from_phoenix(
    data: List[Dict[str, Any]], config: TraceImportConfig
) -> TraceImportResult:
    """Parse Phoenix OTEL format traces.

    Expects each item in data to have at minimum a "trace_id" or "context"
    dict with "trace_id", and optionally "spans" or top-level span fields
    like "name", "span_kind", "attributes".
    """
    traces: List[ImportedTrace] = []
    errors: List[str] = []
    skipped = 0

    for idx, item in enumerate(data):
        try:
            trace_id = item.get("trace_id") or item.get("context", {}).get(
                "trace_id", ""
            )
            if not trace_id:
                skipped += 1
                errors.append(f"Item {idx}: missing trace_id")
                continue

            raw_spans = item.get("spans", [])
            if not raw_spans and "name" in item:
                raw_spans = [item]

            mapped_spans = []
            for span in raw_spans:
                mapped = dict(span)
                if config.map_span_types:
                    span_kind = span.get("span_kind", span.get("kind", ""))
                    mapped["span_type"] = map_span_type(str(span_kind), "phoenix")
                if not config.preserve_ids:
                    mapped.pop("span_id", None)
                mapped_spans.append(mapped)

            traces.append(
                ImportedTrace(
                    trace_id=str(trace_id),
                    spans=mapped_spans,
                    source_platform="phoenix",
                    imported_at=_now_iso(),
                    metadata=item.get("metadata", {}),
                )
            )
        except Exception as exc:
            skipped += 1
            errors.append(f"Item {idx}: {exc}")

    return TraceImportResult(
        traces=traces,
        total_imported=len(traces),
        skipped=skipped,
        errors=errors,
    )


def import_from_langfuse(
    data: List[Dict[str, Any]], config: TraceImportConfig
) -> TraceImportResult:
    """Parse Langfuse format traces.

    Expects each item to have "id" or "trace_id" and optionally
    "observations" (Langfuse's span equivalent).
    """
    traces: List[ImportedTrace] = []
    errors: List[str] = []
    skipped = 0

    for idx, item in enumerate(data):
        try:
            trace_id = item.get("id") or item.get("trace_id", "")
            if not trace_id:
                skipped += 1
                errors.append(f"Item {idx}: missing trace id")
                continue

            observations = item.get("observations", [])
            mapped_spans = []
            for obs in observations:
                mapped = dict(obs)
                if config.map_span_types:
                    obs_type = obs.get("type", "SPAN")
                    mapped["span_type"] = map_span_type(str(obs_type), "langfuse")
                if not config.preserve_ids:
                    mapped.pop("id", None)
                mapped_spans.append(mapped)

            traces.append(
                ImportedTrace(
                    trace_id=str(trace_id),
                    spans=mapped_spans,
                    source_platform="langfuse",
                    imported_at=_now_iso(),
                    metadata=item.get("metadata", {}),
                )
            )
        except Exception as exc:
            skipped += 1
            errors.append(f"Item {idx}: {exc}")

    return TraceImportResult(
        traces=traces,
        total_imported=len(traces),
        skipped=skipped,
        errors=errors,
    )


def import_from_generic(
    data: List[Dict[str, Any]], config: TraceImportConfig
) -> TraceImportResult:
    """Generic trace import with minimal mapping.

    Expects each item to have "trace_id" and optionally "spans".
    """
    traces: List[ImportedTrace] = []
    errors: List[str] = []
    skipped = 0

    for idx, item in enumerate(data):
        try:
            trace_id = item.get("trace_id", "")
            if not trace_id:
                skipped += 1
                errors.append(f"Item {idx}: missing trace_id")
                continue

            raw_spans = item.get("spans", [])
            mapped_spans = []
            for span in raw_spans:
                mapped = dict(span)
                if config.map_span_types:
                    mapped["span_type"] = mapped.get("span_type", "custom")
                mapped_spans.append(mapped)

            traces.append(
                ImportedTrace(
                    trace_id=str(trace_id),
                    spans=mapped_spans,
                    source_platform="generic",
                    imported_at=_now_iso(),
                    metadata=item.get("metadata", {}),
                )
            )
        except Exception as exc:
            skipped += 1
            errors.append(f"Item {idx}: {exc}")

    return TraceImportResult(
        traces=traces,
        total_imported=len(traces),
        skipped=skipped,
        errors=errors,
    )


def import_traces(
    data: List[Dict[str, Any]], config: TraceImportConfig
) -> TraceImportResult:
    """Route to the appropriate platform-specific importer.

    Args:
        data: List of raw trace dicts from the external platform.
        config: Import configuration specifying the source platform.

    Returns:
        A TraceImportResult with imported traces and any errors.
    """
    if config.source_platform == "phoenix":
        return import_from_phoenix(data, config)
    if config.source_platform == "langfuse":
        return import_from_langfuse(data, config)
    if config.source_platform in ("langsmith",):
        # LangSmith uses a similar structure to generic with span type mapping
        return import_from_generic(data, config)
    return import_from_generic(data, config)


def validate_imported_traces(
    result: TraceImportResult,
) -> Tuple[bool, List[str]]:
    """Validate imported trace data.

    Checks that each trace has a non-empty trace_id and that spans
    are well-formed (list of dicts).

    Returns:
        A tuple of (is_valid, list_of_issues).
    """
    issues: List[str] = []

    if not result.traces:
        issues.append("No traces imported")

    for i, trace in enumerate(result.traces):
        if not trace.trace_id:
            issues.append(f"Trace {i}: empty trace_id")
        if not isinstance(trace.spans, list):
            issues.append(f"Trace {i}: spans is not a list")
        for j, span in enumerate(trace.spans):
            if not isinstance(span, dict):
                issues.append(f"Trace {i}, span {j}: span is not a dict")

    return (len(issues) == 0, issues)
