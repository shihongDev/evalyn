"""
Trace summary generation: LLM prompt construction and heuristic summarization.

Provides dataclasses for configuration and output, prompt building for LLM-based
summarization, and pure-heuristic fallback summarization without LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_STYLES = ("brief", "detailed", "technical")
SLOW_SPAN_THRESHOLD_MS = 1000.0


@dataclass
class TraceSummaryConfig:
    """Configuration for trace summary generation."""

    max_spans: int = 20
    include_tool_calls: bool = True
    include_latencies: bool = True
    summary_style: str = "brief"  # "brief", "detailed", "technical"

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_spans": self.max_spans,
            "include_tool_calls": self.include_tool_calls,
            "include_latencies": self.include_latencies,
            "summary_style": self.summary_style,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceSummaryConfig:
        return cls(
            max_spans=data.get("max_spans", 20),
            include_tool_calls=data.get("include_tool_calls", True),
            include_latencies=data.get("include_latencies", True),
            summary_style=data.get("summary_style", "brief"),
        )


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


@dataclass
class TraceSummary:
    """Summary of a single trace."""

    trace_id: str
    summary_text: str
    key_findings: list[str] = field(default_factory=list)
    span_count: int = 0
    total_latency_ms: float = 0.0
    generated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "summary_text": self.summary_text,
            "key_findings": list(self.key_findings),
            "span_count": self.span_count,
            "total_latency_ms": self.total_latency_ms,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceSummary:
        return cls(
            trace_id=data.get("trace_id", ""),
            summary_text=data.get("summary_text", ""),
            key_findings=data.get("key_findings", []),
            span_count=data.get("span_count", 0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            generated_at=data.get("generated_at", ""),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_spans(trace_data: dict) -> list[dict]:
    """Extract span list from trace data, capped at a reasonable default."""
    return trace_data.get("spans", [])


def _span_latency_ms(span: dict) -> float | None:
    """Return duration_ms from a span dict, computing from timestamps if needed."""
    if "duration_ms" in span and span["duration_ms"] is not None:
        return float(span["duration_ms"])
    start = span.get("start_time")
    end = span.get("end_time")
    if start and end:
        try:
            t0 = datetime.fromisoformat(str(start))
            t1 = datetime.fromisoformat(str(end))
            return (t1 - t0).total_seconds() * 1000
        except (ValueError, TypeError):
            return None
    return None


def _total_latency(spans: list[dict]) -> float:
    """Sum latencies across all spans."""
    total = 0.0
    for span in spans:
        lat = _span_latency_ms(span)
        if lat is not None:
            total += lat
    return total


def _count_by_type(spans: list[dict]) -> dict[str, int]:
    """Count spans grouped by span_type."""
    counts: dict[str, int] = {}
    for span in spans:
        st = span.get("span_type", "unknown")
        counts[st] = counts.get(st, 0) + 1
    return counts


def _collect_tool_calls(spans: list[dict]) -> list[str]:
    """Collect tool/function names from tool_call spans."""
    names: list[str] = []
    for span in spans:
        if span.get("span_type") in ("tool_call", "tool_use"):
            name = span.get("name", "unknown_tool")
            names.append(name)
    return names


def _collect_errors(spans: list[dict]) -> list[str]:
    """Collect error messages from spans with error status."""
    errors: list[str] = []
    for span in spans:
        if span.get("status") == "error":
            name = span.get("name", "unknown")
            attrs = span.get("attributes", {})
            msg = attrs.get("error", attrs.get("error_message", ""))
            if msg:
                errors.append(f"{name}: {msg}")
            else:
                errors.append(f"{name} (error)")
    return errors


def _collect_slow_spans(spans: list[dict], threshold_ms: float = SLOW_SPAN_THRESHOLD_MS) -> list[str]:
    """Identify spans slower than threshold."""
    slow: list[str] = []
    for span in spans:
        lat = _span_latency_ms(span)
        if lat is not None and lat > threshold_ms:
            name = span.get("name", "unknown")
            slow.append(f"{name} ({lat:.0f}ms)")
    return slow


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_summary_prompt(trace_data: dict, config: TraceSummaryConfig) -> str:
    """Build an LLM prompt to summarize a trace.

    Includes span types, tool calls, latencies, and errors based on config.
    The prompt instructs the LLM to generate a natural language summary.
    """
    spans = _get_spans(trace_data)[:config.max_spans]
    trace_id = trace_data.get("trace_id", trace_data.get("id", "unknown"))
    type_counts = _count_by_type(spans)
    errors = _collect_errors(spans)

    parts: list[str] = []
    parts.append(f"Summarize this trace (ID: {trace_id}) with {len(spans)} spans.")
    parts.append("")

    # Span type breakdown
    parts.append("Span types:")
    for stype, count in sorted(type_counts.items()):
        parts.append(f"  - {stype}: {count}")
    parts.append("")

    # Tool calls
    if config.include_tool_calls:
        tool_names = _collect_tool_calls(spans)
        if tool_names:
            parts.append("Tool calls: " + ", ".join(tool_names))
        else:
            parts.append("Tool calls: none")
        parts.append("")

    # Latencies
    if config.include_latencies:
        total_lat = _total_latency(spans)
        parts.append(f"Total latency: {total_lat:.0f}ms")
        slow = _collect_slow_spans(spans)
        if slow:
            parts.append("Slow spans (>1s): " + ", ".join(slow))
        parts.append("")

    # Errors
    if errors:
        parts.append("Errors:")
        for err in errors:
            parts.append(f"  - {err}")
        parts.append("")

    # Style instruction
    style_instructions = {
        "brief": "Provide a 2-3 sentence summary focusing on what happened and any issues.",
        "detailed": "Provide a detailed paragraph covering the trace flow, tool usage, latencies, and any errors or anomalies.",
        "technical": "Provide a technical summary with span-level detail, latency breakdown, and root cause analysis for any errors.",
    }
    instruction = style_instructions.get(config.summary_style, style_instructions["brief"])
    parts.append(instruction)
    parts.append("")
    parts.append("Format key findings as bullet points starting with '- '.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Heuristic extraction
# ---------------------------------------------------------------------------


def extract_trace_highlights(trace_data: dict) -> list[str]:
    """Extract notable events from trace data using heuristics.

    Identifies: errors, slow spans (>1s), tool failures, unusual patterns.
    No LLM required.
    """
    spans = _get_spans(trace_data)
    highlights: list[str] = []

    # Errors
    errors = _collect_errors(spans)
    for err in errors:
        highlights.append(f"Error: {err}")

    # Slow spans
    slow = _collect_slow_spans(spans)
    for s in slow:
        highlights.append(f"Slow span: {s}")

    # Tool failures: tool_call/tool_use spans with error status
    for span in spans:
        if span.get("span_type") in ("tool_call", "tool_use") and span.get("status") == "error":
            name = span.get("name", "unknown_tool")
            highlights.append(f"Tool failure: {name}")

    # Unusual patterns: retries (multiple spans with same name)
    name_counts: dict[str, int] = {}
    for span in spans:
        n = span.get("name", "")
        if n:
            name_counts[n] = name_counts.get(n, 0) + 1
    for name, count in name_counts.items():
        if count >= 3:
            highlights.append(f"Repeated span: {name} ({count} times, possible retry)")

    # No spans at all
    if not spans:
        highlights.append("Empty trace: no spans recorded")

    return highlights


# ---------------------------------------------------------------------------
# Heuristic summary (no LLM)
# ---------------------------------------------------------------------------


def build_heuristic_summary(trace_data: dict, config: TraceSummaryConfig) -> TraceSummary:
    """Generate a trace summary using heuristics only - no LLM call.

    Counts spans by type, identifies errors, computes total latency,
    lists tool calls. Returns a TraceSummary with template-generated text.
    """
    spans = _get_spans(trace_data)[:config.max_spans]
    trace_id = trace_data.get("trace_id", trace_data.get("id", "unknown"))
    type_counts = _count_by_type(spans)
    errors = _collect_errors(spans)
    total_lat = _total_latency(spans)
    tool_names = _collect_tool_calls(spans) if config.include_tool_calls else []

    # Build summary text
    text_parts: list[str] = []
    text_parts.append(f"Trace {trace_id} contains {len(spans)} spans")

    if type_counts:
        type_desc = ", ".join(f"{count} {stype}" for stype, count in sorted(type_counts.items()))
        text_parts.append(f" ({type_desc})")

    text_parts.append(".")

    if config.include_latencies and total_lat > 0:
        text_parts.append(f" Total latency: {total_lat:.0f}ms.")

    if tool_names:
        unique_tools = sorted(set(tool_names))
        text_parts.append(f" Tools used: {', '.join(unique_tools)}.")

    if errors:
        text_parts.append(f" {len(errors)} error(s) detected.")

    summary_text = "".join(text_parts)

    # Build key findings
    key_findings: list[str] = []
    highlights = extract_trace_highlights(trace_data)
    key_findings.extend(highlights)

    if not errors and spans:
        key_findings.append("No errors detected")

    if config.include_latencies:
        slow = _collect_slow_spans(spans)
        if not slow and spans:
            key_findings.append("All spans completed within 1s threshold")

    generated_at = datetime.now(timezone.utc).isoformat()

    return TraceSummary(
        trace_id=trace_id,
        summary_text=summary_text,
        key_findings=key_findings,
        span_count=len(spans),
        total_latency_ms=total_lat,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def parse_llm_summary_response(response: str, trace_id: str) -> TraceSummary:
    """Parse an LLM's text response into a TraceSummary.

    Extracts key_findings from lines starting with '- ' (bullet points).
    The remaining text becomes the summary_text.
    """
    lines = response.strip().split("\n")
    summary_lines: list[str] = []
    key_findings: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            key_findings.append(stripped[2:].strip())
        else:
            summary_lines.append(line)

    summary_text = "\n".join(summary_lines).strip()

    return TraceSummary(
        trace_id=trace_id,
        summary_text=summary_text,
        key_findings=key_findings,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


def format_trace_summary(summary: TraceSummary) -> str:
    """Format a TraceSummary for human-readable display."""
    lines: list[str] = []
    lines.append(f"Trace Summary: {summary.trace_id}")
    lines.append("=" * (len(lines[0])))
    lines.append("")
    lines.append(summary.summary_text)
    lines.append("")

    if summary.key_findings:
        lines.append("Key Findings:")
        for finding in summary.key_findings:
            lines.append(f"  - {finding}")
        lines.append("")

    lines.append(f"Spans: {summary.span_count}")
    if summary.total_latency_ms > 0:
        lines.append(f"Total Latency: {summary.total_latency_ms:.0f}ms")
    if summary.generated_at:
        lines.append(f"Generated: {summary.generated_at}")

    return "\n".join(lines)
