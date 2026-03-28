"""Calibration diagnostic report.

Detailed analysis of why calibration improved or degraded alignment,
with per-item delta tracking and human-readable findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticFinding:
    """A single finding from calibration diagnostics."""

    category: str  # "improvement", "degradation", "unchanged"
    description: str
    severity: str = "info"  # "info", "warning", "critical"
    affected_items: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "affected_items": list(self.affected_items),
        }


@dataclass
class CalibrationDiagnostic:
    """Full diagnostic for a single metric's calibration."""

    metric_id: str
    before_score: float
    after_score: float
    delta: float
    findings: List[DiagnosticFinding] = field(default_factory=list)
    improved_items: List[str] = field(default_factory=list)
    degraded_items: List[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "before_score": self.before_score,
            "after_score": self.after_score,
            "delta": self.delta,
            "findings": [f.as_dict() for f in self.findings],
            "improved_items": list(self.improved_items),
            "degraded_items": list(self.degraded_items),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationDiagnostic:
        findings = []
        for f in data.get("findings", []):
            findings.append(
                DiagnosticFinding(
                    category=f["category"],
                    description=f["description"],
                    severity=f.get("severity", "info"),
                    affected_items=f.get("affected_items", []),
                )
            )
        return cls(
            metric_id=data["metric_id"],
            before_score=data["before_score"],
            after_score=data["after_score"],
            delta=data["delta"],
            findings=findings,
            improved_items=data.get("improved_items", []),
            degraded_items=data.get("degraded_items", []),
            summary=data.get("summary", ""),
        )

    def format_text(self) -> str:
        lines = [f"Calibration Diagnostic: {self.metric_id}"]
        lines.append(
            f"  Score: {self.before_score:.4f} -> {self.after_score:.4f}"
            f" (delta {self.delta:+.4f})"
        )
        if self.improved_items:
            lines.append(f"  Improved items: {len(self.improved_items)}")
        if self.degraded_items:
            lines.append(f"  Degraded items: {len(self.degraded_items)}")
        for f in self.findings:
            lines.append(f"  [{f.severity}] {f.category}: {f.description}")
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_calibration_delta(
    before: Dict[str, float],
    after: Dict[str, float],
) -> Dict[str, float]:
    """Per-item delta (after - before) for items present in both dicts."""
    common = sorted(set(before.keys()) & set(after.keys()))
    return {item: after[item] - before[item] for item in common}


def identify_regression_items(
    before_scores: Dict[str, float],
    after_scores: Dict[str, float],
    threshold: float = 0.01,
) -> List[str]:
    """Items where after_score < before_score - threshold."""
    regressions: List[str] = []
    for item in sorted(set(before_scores.keys()) & set(after_scores.keys())):
        if after_scores[item] < before_scores[item] - threshold:
            regressions.append(item)
    return regressions


def diagnose_calibration(
    metric_id: str,
    before_scores: Dict[str, float],
    after_scores: Dict[str, float],
    threshold: float = 0.5,
) -> CalibrationDiagnostic:
    """Compare before/after scores per item and generate diagnostic findings.

    Items with delta > threshold are improved; items with delta < -threshold
    are degraded; the rest are unchanged.
    """
    deltas = compute_calibration_delta(before_scores, after_scores)
    common_items = sorted(deltas.keys())

    improved: List[str] = []
    degraded: List[str] = []
    unchanged: List[str] = []

    for item in common_items:
        d = deltas[item]
        if d > threshold:
            improved.append(item)
        elif d < -threshold:
            degraded.append(item)
        else:
            unchanged.append(item)

    # Compute aggregate scores (mean over common items)
    if common_items:
        before_mean = sum(before_scores[i] for i in common_items) / len(common_items)
        after_mean = sum(after_scores[i] for i in common_items) / len(common_items)
    else:
        before_mean = 0.0
        after_mean = 0.0
    delta = after_mean - before_mean

    findings: List[DiagnosticFinding] = []

    # Improvement finding
    if improved:
        findings.append(
            DiagnosticFinding(
                category="improvement",
                description=f"{len(improved)} items improved",
                severity="info",
                affected_items=improved,
            )
        )

    # Degradation finding
    if degraded:
        severity = "critical" if len(degraded) > len(improved) else "warning"
        findings.append(
            DiagnosticFinding(
                category="degradation",
                description=f"{len(degraded)} items degraded",
                severity=severity,
                affected_items=degraded,
            )
        )

    # Unchanged finding
    if unchanged:
        findings.append(
            DiagnosticFinding(
                category="unchanged",
                description=f"{len(unchanged)} items unchanged",
                severity="info",
                affected_items=unchanged,
            )
        )

    # Largest improvement
    if deltas:
        best_item = max(deltas, key=lambda k: deltas[k])
        if deltas[best_item] > 0:
            findings.append(
                DiagnosticFinding(
                    category="improvement",
                    description=f"largest improvement on item {best_item} ({deltas[best_item]:+.4f})",
                    severity="info",
                    affected_items=[best_item],
                )
            )

    diagnostic = CalibrationDiagnostic(
        metric_id=metric_id,
        before_score=before_mean,
        after_score=after_mean,
        delta=delta,
        findings=findings,
        improved_items=improved,
        degraded_items=degraded,
    )
    diagnostic.summary = generate_summary(diagnostic)
    return diagnostic


def generate_summary(diagnostic: CalibrationDiagnostic) -> str:
    """Human-readable summary of a calibration diagnostic."""
    parts: List[str] = []
    parts.append(
        f"Metric {diagnostic.metric_id}: {diagnostic.before_score:.4f} -> {diagnostic.after_score:.4f}"
        f" (delta {diagnostic.delta:+.4f})"
    )
    n_improved = len(diagnostic.improved_items)
    n_degraded = len(diagnostic.degraded_items)
    if n_improved and not n_degraded:
        parts.append(f"{n_improved} items improved, none degraded")
    elif n_degraded and not n_improved:
        parts.append(f"{n_degraded} items degraded, none improved")
    elif n_improved or n_degraded:
        parts.append(f"{n_improved} improved, {n_degraded} degraded")
    else:
        parts.append("no significant changes")
    return ". ".join(parts)
