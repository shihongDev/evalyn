from __future__ import annotations

import difflib
import html as _html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import Span


@dataclass
class SpanDiff:
    """Comparison of two matched spans."""

    span_name: str
    span_type: str
    status: str  # "added", "removed", "changed", "unchanged"
    left_id: str | None = None
    right_id: str | None = None
    duration_delta_ms: float | None = None
    output_diff: str = ""
    cost_delta: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_name": self.span_name,
            "span_type": self.span_type,
            "status": self.status,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "duration_delta_ms": self.duration_delta_ms,
            "output_diff": self.output_diff,
            "cost_delta": self.cost_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanDiff:
        return cls(
            span_name=data["span_name"],
            span_type=data["span_type"],
            status=data["status"],
            left_id=data.get("left_id"),
            right_id=data.get("right_id"),
            duration_delta_ms=data.get("duration_delta_ms"),
            output_diff=data.get("output_diff", ""),
            cost_delta=data.get("cost_delta"),
        )


@dataclass
class TraceDiff:
    """Full comparison of two traces."""

    diffs: list[SpanDiff] = field(default_factory=list)
    added_count: int = 0
    removed_count: int = 0
    changed_count: int = 0
    unchanged_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "diffs": [d.as_dict() for d in self.diffs],
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "changed_count": self.changed_count,
            "unchanged_count": self.unchanged_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceDiff:
        return cls(
            diffs=[SpanDiff.from_dict(d) for d in data.get("diffs", [])],
            added_count=data.get("added_count", 0),
            removed_count=data.get("removed_count", 0),
            changed_count=data.get("changed_count", 0),
            unchanged_count=data.get("unchanged_count", 0),
        )

    def format_text(self) -> str:
        """Plain text summary of the diff."""
        lines: list[str] = []
        lines.append(
            f"Trace diff: {self.added_count} added, {self.removed_count} removed, "
            f"{self.changed_count} changed, {self.unchanged_count} unchanged"
        )
        for d in self.diffs:
            prefix = {"added": "+", "removed": "-", "changed": "~", "unchanged": " "}.get(
                d.status, "?"
            )
            line = f"  {prefix} {d.span_name} ({d.span_type}) [{d.status}]"
            if d.duration_delta_ms is not None and d.status == "changed":
                line += f" dt={d.duration_delta_ms:+.1f}ms"
            if d.cost_delta is not None and d.cost_delta != 0 and d.status == "changed":
                line += f" cost={d.cost_delta:+.4f}"
            lines.append(line)
        return "\n".join(lines)


def _span_key(span: Span) -> tuple[str, str]:
    return (span.name, span.span_type)


def _compute_output_diff(left_output: str, right_output: str) -> str:
    if left_output == right_output:
        return ""
    left_lines = left_output.splitlines(keepends=True)
    right_lines = right_output.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(left_lines, right_lines, fromfile="left", tofile="right")
    )
    return "".join(diff_lines)


def diff_traces(left: list[Span], right: list[Span]) -> TraceDiff:
    """Align spans by (name, span_type) pairs and compute diffs."""
    left_by_key: dict[tuple[str, str], Span] = {}
    for span in left:
        key = _span_key(span)
        # First occurrence wins
        if key not in left_by_key:
            left_by_key[key] = span

    right_by_key: dict[tuple[str, str], Span] = {}
    for span in right:
        key = _span_key(span)
        if key not in right_by_key:
            right_by_key[key] = span

    all_keys: list[tuple[str, str]] = []
    seen: set = set()
    for span in left:
        key = _span_key(span)
        if key not in seen:
            all_keys.append(key)
            seen.add(key)
    for span in right:
        key = _span_key(span)
        if key not in seen:
            all_keys.append(key)
            seen.add(key)

    diffs: list[SpanDiff] = []
    added = 0
    removed = 0
    changed = 0
    unchanged = 0

    for key in all_keys:
        name, span_type = key
        l_span = left_by_key.get(key)
        r_span = right_by_key.get(key)

        if l_span and not r_span:
            diffs.append(
                SpanDiff(
                    span_name=name,
                    span_type=span_type,
                    status="removed",
                    left_id=l_span.id,
                )
            )
            removed += 1
        elif r_span and not l_span:
            diffs.append(
                SpanDiff(
                    span_name=name,
                    span_type=span_type,
                    status="added",
                    right_id=r_span.id,
                )
            )
            added += 1
        else:
            # Both present - compare
            if l_span is None or r_span is None:
                continue  # defensive: should not happen given prior branches

            l_dur = l_span.duration_ms
            r_dur = r_span.duration_ms
            duration_delta: float | None = None
            if l_dur is not None and r_dur is not None:
                duration_delta = r_dur - l_dur

            l_output = str(l_span.attributes.get("output", ""))
            r_output = str(r_span.attributes.get("output", ""))
            output_diff = _compute_output_diff(l_output, r_output)

            l_cost = l_span.attributes.get("cost", 0) or 0
            r_cost = r_span.attributes.get("cost", 0) or 0
            cost_delta: float | None = None
            if l_cost or r_cost:
                cost_delta = float(r_cost) - float(l_cost)

            has_changes = bool(output_diff) or (
                duration_delta is not None and duration_delta != 0
            ) or (cost_delta is not None and cost_delta != 0)

            status = "changed" if has_changes else "unchanged"

            diffs.append(
                SpanDiff(
                    span_name=name,
                    span_type=span_type,
                    status=status,
                    left_id=l_span.id,
                    right_id=r_span.id,
                    duration_delta_ms=duration_delta,
                    output_diff=output_diff,
                    cost_delta=cost_delta,
                )
            )
            if has_changes:
                changed += 1
            else:
                unchanged += 1

    return TraceDiff(
        diffs=diffs,
        added_count=added,
        removed_count=removed,
        changed_count=changed,
        unchanged_count=unchanged,
    )


def format_diff_ascii(diff: TraceDiff) -> str:
    """ASCII table format showing +/- for added/removed spans."""
    lines: list[str] = []

    # Header
    header = f"{'Status':<10} {'Span Name':<30} {'Type':<15} {'Duration Delta':<18} {'Cost Delta':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    for d in diff.diffs:
        prefix = {"added": "+", "removed": "-", "changed": "~", "unchanged": " "}.get(
            d.status, "?"
        )
        status_str = f"[{prefix}]"

        dur_str = ""
        if d.duration_delta_ms is not None:
            dur_str = f"{d.duration_delta_ms:+.1f}ms"

        cost_str = ""
        if d.cost_delta is not None and d.cost_delta != 0:
            cost_str = f"{d.cost_delta:+.4f}"

        line = f"{status_str:<10} {d.span_name:<30} {d.span_type:<15} {dur_str:<18} {cost_str:<12}"
        lines.append(line)

    # Summary
    lines.append("-" * len(header))
    lines.append(
        f"Total: {diff.added_count} added, {diff.removed_count} removed, "
        f"{diff.changed_count} changed, {diff.unchanged_count} unchanged"
    )
    return "\n".join(lines)


def format_diff_html(diff: TraceDiff) -> str:
    """Simple HTML table with green/red highlighting."""
    rows: list[str] = []
    rows.append("<table border='1' cellpadding='4' cellspacing='0'>")
    rows.append(
        "<tr><th>Status</th><th>Span Name</th><th>Type</th>"
        "<th>Duration Delta</th><th>Cost Delta</th></tr>"
    )

    color_map = {
        "added": "#d4edda",
        "removed": "#f8d7da",
        "changed": "#fff3cd",
        "unchanged": "#ffffff",
    }

    for d in diff.diffs:
        bg = color_map.get(d.status, "#ffffff")
        dur_str = ""
        if d.duration_delta_ms is not None:
            dur_str = f"{d.duration_delta_ms:+.1f}ms"
        cost_str = ""
        if d.cost_delta is not None and d.cost_delta != 0:
            cost_str = f"{d.cost_delta:+.4f}"

        rows.append(
            f"<tr style='background-color:{bg}'>"
            f"<td>{_html.escape(d.status)}</td>"
            f"<td>{_html.escape(d.span_name)}</td>"
            f"<td>{_html.escape(d.span_type)}</td>"
            f"<td>{_html.escape(dur_str)}</td>"
            f"<td>{_html.escape(cost_str)}</td>"
            f"</tr>"
        )

    rows.append("</table>")
    return "\n".join(rows)
