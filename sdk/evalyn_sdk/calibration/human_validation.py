"""Calibration human validation.

Present calibrated prompts to a human reviewer for approval before committing.
Tracks validation requests, decisions, and produces summary reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ValidationRequest:
    """A request for human validation of a calibrated prompt."""

    metric_id: str
    original_prompt: str
    calibrated_prompt: str
    sample_items: list[dict[str, Any]] = field(default_factory=list)
    alignment_before: float = 0.0
    alignment_after: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "original_prompt": self.original_prompt,
            "calibrated_prompt": self.calibrated_prompt,
            "sample_items": list(self.sample_items),
            "alignment_before": self.alignment_before,
            "alignment_after": self.alignment_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationRequest:
        return cls(
            metric_id=data["metric_id"],
            original_prompt=data["original_prompt"],
            calibrated_prompt=data["calibrated_prompt"],
            sample_items=data.get("sample_items", []),
            alignment_before=data.get("alignment_before", 0.0),
            alignment_after=data.get("alignment_after", 0.0),
        )


@dataclass
class ValidationDecision:
    """A human reviewer's decision on a validation request."""

    approved: bool
    reviewer: str = ""
    notes: str = ""
    timestamp: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reviewer": self.reviewer,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationDecision:
        ts = data.get("timestamp")
        if ts is not None and isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            approved=data["approved"],
            reviewer=data.get("reviewer", ""),
            notes=data.get("notes", ""),
            timestamp=ts,
        )


@dataclass
class ValidationReport:
    """Summary report of validation requests and decisions."""

    requests: list[ValidationRequest] = field(default_factory=list)
    decisions: list[ValidationDecision] = field(default_factory=list)
    approved_count: int = 0
    rejected_count: int = 0
    pending_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": [r.as_dict() for r in self.requests],
            "decisions": [d.as_dict() for d in self.decisions],
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "pending_count": self.pending_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        return cls(
            requests=[
                ValidationRequest.from_dict(r) for r in data.get("requests", [])
            ],
            decisions=[
                ValidationDecision.from_dict(d) for d in data.get("decisions", [])
            ],
            approved_count=data.get("approved_count", 0),
            rejected_count=data.get("rejected_count", 0),
            pending_count=data.get("pending_count", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Validation Report",
            f"  Requests: {len(self.requests)}",
            f"  Decisions: {len(self.decisions)}",
            f"  Approved: {self.approved_count}",
            f"  Rejected: {self.rejected_count}",
            f"  Pending: {self.pending_count}",
        ]
        for i, req in enumerate(self.requests):
            lines.append(f"  [{i + 1}] {req.metric_id} - "
                         f"alignment {req.alignment_before:.3f} -> {req.alignment_after:.3f}")
        return "\n".join(lines)


def create_validation_request(
    metric_id: str,
    original_prompt: str,
    calibrated_prompt: str,
    sample_items: list[dict[str, Any]] | None = None,
    alignment_before: float = 0.0,
    alignment_after: float = 0.0,
) -> ValidationRequest:
    """Factory for creating a validation request."""
    if sample_items is None:
        sample_items = []
    return ValidationRequest(
        metric_id=metric_id,
        original_prompt=original_prompt,
        calibrated_prompt=calibrated_prompt,
        sample_items=list(sample_items),
        alignment_before=alignment_before,
        alignment_after=alignment_after,
    )


def render_validation_summary(request: ValidationRequest) -> str:
    """Human-readable summary showing original vs calibrated prompt, alignment delta, and sample items."""
    delta = request.alignment_after - request.alignment_before
    sign = "+" if delta >= 0 else ""
    lines = [
        f"Metric: {request.metric_id}",
        "",
        "Original prompt:",
        request.original_prompt,
        "",
        "Calibrated prompt:",
        request.calibrated_prompt,
        "",
        f"Alignment: {request.alignment_before:.3f} -> {request.alignment_after:.3f} ({sign}{delta:.3f})",
    ]
    if request.sample_items:
        lines.append("")
        lines.append(f"Sample items ({len(request.sample_items)}):")
        for i, item in enumerate(request.sample_items, 1):
            lines.append(f"  {i}. {item}")
    return "\n".join(lines)


def record_decision(
    approved: bool,
    reviewer: str = "",
    notes: str = "",
) -> ValidationDecision:
    """Record a human decision with current timestamp."""
    return ValidationDecision(
        approved=approved,
        reviewer=reviewer,
        notes=notes,
        timestamp=datetime.now(timezone.utc),
    )


def build_validation_report(
    requests: list[ValidationRequest],
    decisions: list[ValidationDecision],
) -> ValidationReport:
    """Build report matching requests to decisions.

    Counts approved, rejected, and pending (requests without decisions).
    """
    approved = sum(1 for d in decisions if d.approved)
    rejected = sum(1 for d in decisions if not d.approved)
    pending = max(len(requests) - len(decisions), 0)

    return ValidationReport(
        requests=list(requests),
        decisions=list(decisions),
        approved_count=approved,
        rejected_count=rejected,
        pending_count=pending,
    )
