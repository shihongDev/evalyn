from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models import Span


@dataclass
class FlattenedRecord:
    """A flattened representation of a span for query performance."""

    record_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    span_name: str = ""
    span_type: str = ""
    parent_name: str = ""
    depth: int = 0
    duration_ms: float = 0.0
    cost: float = 0.0
    tokens: int = 0
    input_text: str = ""
    output_text: str = ""
    status: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "span_name": self.span_name,
            "span_type": self.span_type,
            "parent_name": self.parent_name,
            "depth": self.depth,
            "duration_ms": self.duration_ms,
            "cost": self.cost,
            "tokens": self.tokens,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlattenedRecord:
        return cls(
            record_id=data.get("record_id", ""),
            trace_id=data.get("trace_id", ""),
            span_id=data.get("span_id", ""),
            span_name=data.get("span_name", ""),
            span_type=data.get("span_type", ""),
            parent_name=data.get("parent_name", ""),
            depth=data.get("depth", 0),
            duration_ms=data.get("duration_ms", 0.0),
            cost=data.get("cost", 0.0),
            tokens=data.get("tokens", 0),
            input_text=data.get("input_text", ""),
            output_text=data.get("output_text", ""),
            status=data.get("status", "ok"),
        )


@dataclass
class DenormalizationConfig:
    """Configuration for span denormalization."""

    include_input: bool = True
    include_output: bool = True
    max_text_length: int = 500
    flatten_attributes: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "include_input": self.include_input,
            "include_output": self.include_output,
            "max_text_length": self.max_text_length,
            "flatten_attributes": self.flatten_attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DenormalizationConfig:
        return cls(
            include_input=data.get("include_input", True),
            include_output=data.get("include_output", True),
            max_text_length=data.get("max_text_length", 500),
            flatten_attributes=data.get("flatten_attributes", True),
        )


@dataclass
class DenormalizationReport:
    """Statistics from a denormalization operation."""

    total_spans: int = 0
    flattened_records: int = 0
    avg_depth: float = 0.0
    total_size_reduction_pct: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_spans": self.total_spans,
            "flattened_records": self.flattened_records,
            "avg_depth": self.avg_depth,
            "total_size_reduction_pct": self.total_size_reduction_pct,
        }

    def format_text(self) -> str:
        lines = [
            "Denormalization Report",
            "-" * 40,
            f"  Total spans: {self.total_spans}",
            f"  Flattened records: {self.flattened_records}",
            f"  Avg depth: {self.avg_depth:.1f}",
            f"  Size reduction: {self.total_size_reduction_pct:.1f}%",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length."""
    if len(text) <= max_length:
        return text
    return text[:max_length]


def flatten_span(
    span: Span,
    parent_name: str = "",
    depth: int = 0,
    config: DenormalizationConfig | None = None,
) -> FlattenedRecord:
    """Convert a Span to a flat record.

    Truncates text fields to config.max_text_length.
    Extracts cost/tokens from span attributes.
    """
    if config is None:
        config = DenormalizationConfig()

    attrs = span.attributes or {}

    input_text = ""
    if config.include_input:
        raw = attrs.get("input", "")
        input_text = _truncate(str(raw), config.max_text_length)

    output_text = ""
    if config.include_output:
        raw = attrs.get("output", "")
        output_text = _truncate(str(raw), config.max_text_length)

    cost = float(attrs.get("cost", 0.0))
    tokens = int(attrs.get("tokens", 0))

    return FlattenedRecord(
        record_id=f"{span.id}-flat",
        trace_id=attrs.get("trace_id", ""),
        span_id=span.id,
        span_name=span.name,
        span_type=span.span_type,
        parent_name=parent_name,
        depth=depth,
        duration_ms=span.duration_ms or 0.0,
        cost=cost,
        tokens=tokens,
        input_text=input_text,
        output_text=output_text,
        status=span.status,
    )


def flatten_trace(
    spans: list[Span],
    config: DenormalizationConfig | None = None,
) -> list[FlattenedRecord]:
    """Flatten all spans preserving hierarchy info (depth, parent_name).

    Builds parent-child tree first, then walks it depth-first.
    """
    if config is None:
        config = DenormalizationConfig()

    if not spans:
        return []

    # Build lookup
    by_id: dict[str, Span] = {s.id: s for s in spans}
    children: dict[str, list[str]] = {s.id: [] for s in spans}
    roots: list[str] = []

    for s in spans:
        if s.parent_id and s.parent_id in by_id:
            children[s.parent_id].append(s.id)
        else:
            roots.append(s.id)

    # Walk depth-first
    records: list[FlattenedRecord] = []

    def walk(span_id: str, parent_name: str, depth: int) -> None:
        s = by_id[span_id]
        rec = flatten_span(s, parent_name=parent_name, depth=depth, config=config)
        records.append(rec)
        for child_id in children[span_id]:
            walk(child_id, parent_name=s.name, depth=depth + 1)

    for root_id in roots:
        walk(root_id, parent_name="", depth=0)

    return records


def build_denormalization_report(
    records: list[FlattenedRecord],
    original_spans: list[Span],
) -> DenormalizationReport:
    """Build statistics about the denormalization."""
    total_spans = len(original_spans)
    flattened = len(records)

    avg_depth = 0.0
    if records:
        avg_depth = sum(r.depth for r in records) / len(records)

    # Estimate size reduction: flat records drop nested structure overhead.
    # Use a simple heuristic based on depth.
    reduction = 0.0
    if total_spans > 0 and avg_depth > 0:
        reduction = min(avg_depth * 10.0, 50.0)

    return DenormalizationReport(
        total_spans=total_spans,
        flattened_records=flattened,
        avg_depth=avg_depth,
        total_size_reduction_pct=reduction,
    )


def query_flat_records(
    records: list[FlattenedRecord],
    filters: dict[str, Any],
) -> list[FlattenedRecord]:
    """Filter records by field values.

    Supported filter keys:
    - span_type: exact match on span_type
    - status: exact match on status
    - span_name: exact match on span_name
    - min_duration: minimum duration_ms (inclusive)
    - max_duration: maximum duration_ms (inclusive)
    - min_depth: minimum depth (inclusive)
    - max_depth: maximum depth (inclusive)
    """
    result: list[FlattenedRecord] = []

    for rec in records:
        if "span_type" in filters and rec.span_type != filters["span_type"]:
            continue
        if "status" in filters and rec.status != filters["status"]:
            continue
        if "span_name" in filters and rec.span_name != filters["span_name"]:
            continue
        if "min_duration" in filters and rec.duration_ms < filters["min_duration"]:
            continue
        if "max_duration" in filters and rec.duration_ms > filters["max_duration"]:
            continue
        if "min_depth" in filters and rec.depth < filters["min_depth"]:
            continue
        if "max_depth" in filters and rec.depth > filters["max_depth"]:
            continue
        result.append(rec)

    return result


def export_flat_csv(records: list[FlattenedRecord]) -> str:
    """Export records as a CSV string."""
    if not records:
        return ""

    output = io.StringIO()
    fieldnames = [
        "record_id",
        "trace_id",
        "span_id",
        "span_name",
        "span_type",
        "parent_name",
        "depth",
        "duration_ms",
        "cost",
        "tokens",
        "input_text",
        "output_text",
        "status",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec.as_dict())

    return output.getvalue()
