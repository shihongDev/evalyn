"""Calibration staleness detection.

Monitors calibration freshness by checking age and dataset drift,
producing alerts when recalibration may be needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMetadata:
    """Metadata for a calibration run."""

    metric_id: str
    calibrated_at: datetime
    dataset_hash: str
    num_items: int
    alignment_score: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "calibrated_at": self.calibrated_at.isoformat(),
            "dataset_hash": self.dataset_hash,
            "num_items": self.num_items,
            "alignment_score": self.alignment_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationMetadata:
        calibrated_at = data["calibrated_at"]
        if isinstance(calibrated_at, str):
            calibrated_at = datetime.fromisoformat(calibrated_at)
        return cls(
            metric_id=data["metric_id"],
            calibrated_at=calibrated_at,
            dataset_hash=data["dataset_hash"],
            num_items=data["num_items"],
            alignment_score=data["alignment_score"],
        )


@dataclass
class StalenessAlert:
    """An alert about calibration staleness."""

    metric_id: str
    reason: str
    severity: str  # "info" / "warning" / "critical"
    days_since_calibration: int = 0
    dataset_drift: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "reason": self.reason,
            "severity": self.severity,
            "days_since_calibration": self.days_since_calibration,
            "dataset_drift": self.dataset_drift,
        }


@dataclass
class StalenessReport:
    """Aggregated staleness report across all calibrations."""

    alerts: List[StalenessAlert] = field(default_factory=list)
    stale_metrics: List[str] = field(default_factory=list)
    healthy_metrics: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alerts": [a.as_dict() for a in self.alerts],
            "stale_metrics": self.stale_metrics,
            "healthy_metrics": self.healthy_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StalenessReport:
        alerts = []
        for a in data.get("alerts", []):
            alerts.append(
                StalenessAlert(
                    metric_id=a["metric_id"],
                    reason=a["reason"],
                    severity=a["severity"],
                    days_since_calibration=a.get("days_since_calibration", 0),
                    dataset_drift=a.get("dataset_drift", 0.0),
                )
            )
        return cls(
            alerts=alerts,
            stale_metrics=data.get("stale_metrics", []),
            healthy_metrics=data.get("healthy_metrics", []),
        )

    def format_text(self) -> str:
        lines = ["Staleness Report"]
        if self.healthy_metrics:
            lines.append(f"  Healthy: {', '.join(self.healthy_metrics)}")
        if self.stale_metrics:
            lines.append(f"  Stale:   {', '.join(self.stale_metrics)}")
        if self.alerts:
            lines.append("  Alerts:")
            for alert in self.alerts:
                lines.append(
                    f"    [{alert.severity.upper()}] {alert.metric_id}: {alert.reason}"
                )
        else:
            lines.append("  No alerts.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_dataset_drift(
    old_hash: str, new_hash: str, old_items: int, new_items: int
) -> float:
    """Simple drift metric based on hash comparison and item count change.

    If hashes are the same, drift is 0.0.
    If hashes differ, drift = abs(new_items - old_items) / max(old_items, new_items, 1).
    """
    if old_hash == new_hash:
        return 0.0
    return abs(new_items - old_items) / max(old_items, new_items, 1)


def check_staleness(
    metadata: CalibrationMetadata,
    current_hash: str,
    current_items: int,
    max_age_days: int = 30,
    drift_threshold: float = 0.2,
) -> List[StalenessAlert]:
    """Check age and drift for a single calibration.

    Age > max_age_days produces a warning.
    Drift > drift_threshold produces a warning.
    Both conditions together produce a critical alert.
    """
    now = datetime.now(timezone.utc)
    age_days = (now - metadata.calibrated_at).days
    drift = compute_dataset_drift(
        metadata.dataset_hash, current_hash, metadata.num_items, current_items
    )

    is_old = age_days > max_age_days
    is_drifted = drift > drift_threshold

    alerts: List[StalenessAlert] = []

    if is_old and is_drifted:
        alerts.append(
            StalenessAlert(
                metric_id=metadata.metric_id,
                reason=f"Calibration is {age_days} days old and dataset has drifted ({drift:.2f})",
                severity="critical",
                days_since_calibration=age_days,
                dataset_drift=drift,
            )
        )
    elif is_old:
        alerts.append(
            StalenessAlert(
                metric_id=metadata.metric_id,
                reason=f"Calibration is {age_days} days old (max {max_age_days})",
                severity="warning",
                days_since_calibration=age_days,
                dataset_drift=drift,
            )
        )
    elif is_drifted:
        alerts.append(
            StalenessAlert(
                metric_id=metadata.metric_id,
                reason=f"Dataset has drifted ({drift:.2f}) beyond threshold ({drift_threshold})",
                severity="warning",
                days_since_calibration=age_days,
                dataset_drift=drift,
            )
        )

    return alerts


def check_all_staleness(
    calibrations: List[CalibrationMetadata],
    current_datasets: Dict[str, Tuple[str, int]],
    max_age_days: int = 30,
) -> StalenessReport:
    """Check all calibrations and produce an aggregated report.

    current_datasets maps metric_id to (current_hash, current_items).
    """
    all_alerts: List[StalenessAlert] = []
    stale: List[str] = []
    healthy: List[str] = []

    for meta in calibrations:
        current_hash, current_items = current_datasets.get(
            meta.metric_id, (meta.dataset_hash, meta.num_items)
        )
        alerts = check_staleness(
            meta, current_hash, current_items, max_age_days=max_age_days
        )
        if alerts:
            all_alerts.extend(alerts)
            if meta.metric_id not in stale:
                stale.append(meta.metric_id)
        else:
            if meta.metric_id not in healthy:
                healthy.append(meta.metric_id)

    return StalenessReport(
        alerts=all_alerts,
        stale_metrics=stale,
        healthy_metrics=healthy,
    )
