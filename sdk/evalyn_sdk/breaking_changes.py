from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricFingerprint:
    """Hash fingerprint of a metric's implementation and parameters."""

    metric_id: str
    version_hash: str
    parameters_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "version_hash": self.version_hash,
            "parameters_hash": self.parameters_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricFingerprint:
        return cls(
            metric_id=data["metric_id"],
            version_hash=data["version_hash"],
            parameters_hash=data["parameters_hash"],
        )


@dataclass
class RunManifest:
    """Manifest capturing the metric fingerprints for a specific eval run."""

    run_id: str
    evalyn_version: str
    fingerprints: list[MetricFingerprint]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evalyn_version": self.evalyn_version,
            "fingerprints": [fp.as_dict() for fp in self.fingerprints],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(
            run_id=data["run_id"],
            evalyn_version=data["evalyn_version"],
            fingerprints=[
                MetricFingerprint.from_dict(fp) for fp in data.get("fingerprints", [])
            ],
            created_at=data["created_at"],
        )


@dataclass
class BreakingChange:
    """A single breaking change detected between two manifests."""

    metric_id: str
    old_hash: str
    new_hash: str
    change_type: str  # "removed", "modified", "parameters_changed"
    migration_hint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "change_type": self.change_type,
            "migration_hint": self.migration_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BreakingChange:
        return cls(
            metric_id=data["metric_id"],
            old_hash=data["old_hash"],
            new_hash=data["new_hash"],
            change_type=data["change_type"],
            migration_hint=data["migration_hint"],
        )


@dataclass
class CompatibilityReport:
    """Report of compatibility between two manifests."""

    compatible: bool
    changes: list[BreakingChange]
    old_version: str
    new_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "changes": [c.as_dict() for c in self.changes],
            "old_version": self.old_version,
            "new_version": self.new_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompatibilityReport:
        return cls(
            compatible=data["compatible"],
            changes=[
                BreakingChange.from_dict(c) for c in data.get("changes", [])
            ],
            old_version=data["old_version"],
            new_version=data["new_version"],
        )


def _stable_hash(data: str) -> str:
    """Compute a stable SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _stable_json(obj: Any) -> str:
    """Produce a deterministic JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_fingerprint(
    metric_id: str, implementation: str, parameters: dict
) -> MetricFingerprint:
    """Hash a metric's code and parameters into a fingerprint.

    The version_hash covers the implementation source, while the
    parameters_hash covers the parameter dict.
    """
    version_hash = _stable_hash(implementation)
    parameters_hash = _stable_hash(_stable_json(parameters))
    return MetricFingerprint(
        metric_id=metric_id,
        version_hash=version_hash,
        parameters_hash=parameters_hash,
    )


def create_manifest(
    run_id: str, version: str, fingerprints: list[MetricFingerprint]
) -> RunManifest:
    """Create a run manifest with the given fingerprints."""
    from datetime import datetime, timezone

    created_at = datetime.now(timezone.utc).isoformat()
    return RunManifest(
        run_id=run_id,
        evalyn_version=version,
        fingerprints=list(fingerprints),
        created_at=created_at,
    )


def compare_manifests(
    old: RunManifest, new: RunManifest
) -> CompatibilityReport:
    """Compare two manifests and detect breaking changes.

    Detects three kinds of changes:
    - removed: metric present in old but absent in new
    - modified: metric implementation hash changed
    - parameters_changed: metric parameters hash changed
    """
    old_fps: dict[str, MetricFingerprint] = {
        fp.metric_id: fp for fp in old.fingerprints
    }
    new_fps: dict[str, MetricFingerprint] = {
        fp.metric_id: fp for fp in new.fingerprints
    }

    changes: list[BreakingChange] = []

    # Check for removed and modified metrics
    for metric_id, old_fp in old_fps.items():
        if metric_id not in new_fps:
            changes.append(
                BreakingChange(
                    metric_id=metric_id,
                    old_hash=old_fp.version_hash,
                    new_hash="",
                    change_type="removed",
                    migration_hint=f"Metric '{metric_id}' was removed. "
                    "Check release notes for a replacement.",
                )
            )
            continue

        new_fp = new_fps[metric_id]

        if old_fp.version_hash != new_fp.version_hash:
            changes.append(
                BreakingChange(
                    metric_id=metric_id,
                    old_hash=old_fp.version_hash,
                    new_hash=new_fp.version_hash,
                    change_type="modified",
                    migration_hint=f"Metric '{metric_id}' implementation changed. "
                    "Re-run evaluation to get comparable results.",
                )
            )
        elif old_fp.parameters_hash != new_fp.parameters_hash:
            changes.append(
                BreakingChange(
                    metric_id=metric_id,
                    old_hash=old_fp.parameters_hash,
                    new_hash=new_fp.parameters_hash,
                    change_type="parameters_changed",
                    migration_hint=f"Metric '{metric_id}' parameters changed. "
                    "Review parameter differences and re-run if needed.",
                )
            )

    compatible = len(changes) == 0
    return CompatibilityReport(
        compatible=compatible,
        changes=changes,
        old_version=old.evalyn_version,
        new_version=new.evalyn_version,
    )


def format_compatibility_report(report: CompatibilityReport) -> str:
    """Format a compatibility report as human-readable text."""
    lines: list[str] = []
    lines.append(
        f"Compatibility: {report.old_version} -> {report.new_version}"
    )
    if report.compatible:
        lines.append("Status: COMPATIBLE - no breaking changes detected.")
        return "\n".join(lines)

    lines.append(
        f"Status: INCOMPATIBLE - {len(report.changes)} breaking change(s) detected."
    )
    for change in report.changes:
        lines.append(f"  [{change.change_type}] {change.metric_id}")
        lines.append(f"    {change.migration_hint}")
    return "\n".join(lines)


def suggest_migration(change: BreakingChange) -> str:
    """Return the migration hint for a breaking change."""
    return change.migration_hint
