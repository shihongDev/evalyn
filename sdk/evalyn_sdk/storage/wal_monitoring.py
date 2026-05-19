"""
WAL monitoring: check Write-Ahead Log size and checkpoint frequency.

Provides pure functions to inspect WAL/SHM files, estimate pending writes,
and recommend checkpoint actions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class WALStatus:
    """Current state of the WAL and SHM files for a database."""

    wal_exists: bool = False
    wal_size_bytes: int = 0
    shm_exists: bool = False
    shm_size_bytes: int = 0
    estimated_pending_writes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "wal_exists": self.wal_exists,
            "wal_size_bytes": self.wal_size_bytes,
            "shm_exists": self.shm_exists,
            "shm_size_bytes": self.shm_size_bytes,
            "estimated_pending_writes": self.estimated_pending_writes,
        }

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("WAL Status")
        lines.append("-" * 40)
        lines.append(f"  WAL exists: {self.wal_exists}")
        lines.append(f"  WAL size: {self.wal_size_bytes} bytes")
        lines.append(f"  SHM exists: {self.shm_exists}")
        lines.append(f"  SHM size: {self.shm_size_bytes} bytes")
        lines.append(f"  Estimated pending writes: {self.estimated_pending_writes}")
        return "\n".join(lines)


@dataclass
class WALConfig:
    """Configuration thresholds for WAL monitoring."""

    max_wal_size_bytes: int = 10_000_000
    checkpoint_threshold_bytes: int = 5_000_000
    monitor_interval_seconds: float = 60.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_wal_size_bytes": self.max_wal_size_bytes,
            "checkpoint_threshold_bytes": self.checkpoint_threshold_bytes,
            "monitor_interval_seconds": self.monitor_interval_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WALConfig:
        return cls(
            max_wal_size_bytes=data.get("max_wal_size_bytes", 10_000_000),
            checkpoint_threshold_bytes=data.get("checkpoint_threshold_bytes", 5_000_000),
            monitor_interval_seconds=data.get("monitor_interval_seconds", 60.0),
        )


@dataclass
class WALReport:
    """Full WAL analysis report with status, checkpoint need, and recommendations."""

    status: WALStatus = field(default_factory=WALStatus)
    needs_checkpoint: bool = False
    recommendations: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.as_dict(),
            "needs_checkpoint": self.needs_checkpoint,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WALReport:
        status_data = data.get("status", {})
        status = WALStatus(
            wal_exists=status_data.get("wal_exists", False),
            wal_size_bytes=status_data.get("wal_size_bytes", 0),
            shm_exists=status_data.get("shm_exists", False),
            shm_size_bytes=status_data.get("shm_size_bytes", 0),
            estimated_pending_writes=status_data.get("estimated_pending_writes", 0),
        )
        return cls(
            status=status,
            needs_checkpoint=data.get("needs_checkpoint", False),
            recommendations=list(data.get("recommendations", [])),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("WAL Report")
        lines.append("=" * 40)
        lines.append(self.status.format_text())
        lines.append(f"  Needs checkpoint: {self.needs_checkpoint}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    - {rec}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def check_wal_status(db_path: str) -> WALStatus:
    """Check for WAL and SHM files alongside the database.

    Looks for <db_path>-wal and <db_path>-shm, reads their sizes,
    and estimates pending writes.
    """
    wal_path = db_path + "-wal"
    shm_path = db_path + "-shm"

    wal_exists = os.path.exists(wal_path)
    shm_exists = os.path.exists(shm_path)

    wal_size = os.path.getsize(wal_path) if wal_exists else 0
    shm_size = os.path.getsize(shm_path) if shm_exists else 0

    pending = estimate_pending_writes(wal_size)

    return WALStatus(
        wal_exists=wal_exists,
        wal_size_bytes=wal_size,
        shm_exists=shm_exists,
        shm_size_bytes=shm_size,
        estimated_pending_writes=pending,
    )


def needs_checkpoint(status: WALStatus, config: WALConfig) -> bool:
    """True if WAL size exceeds the checkpoint threshold."""
    return status.wal_size_bytes >= config.checkpoint_threshold_bytes


def estimate_pending_writes(wal_size_bytes: int, avg_write_bytes: int = 4096) -> int:
    """Estimate the number of uncommitted writes based on WAL size.

    Divides WAL size by average write size. Returns 0 if WAL is empty
    or avg_write_bytes is non-positive.
    """
    if wal_size_bytes <= 0 or avg_write_bytes <= 0:
        return 0
    return wal_size_bytes // avg_write_bytes


def build_wal_report(db_path: str, config: Optional[WALConfig] = None) -> WALReport:
    """Build a full WAL analysis report for the given database path."""
    if config is None:
        config = WALConfig()

    status = check_wal_status(db_path)
    checkpoint_needed = needs_checkpoint(status, config)
    recommendations = suggest_wal_actions(
        WALReport(
            status=status,
            needs_checkpoint=checkpoint_needed,
            recommendations=[],
        )
    )

    return WALReport(
        status=status,
        needs_checkpoint=checkpoint_needed,
        recommendations=recommendations,
    )


def suggest_wal_actions(report: WALReport) -> List[str]:
    """Generate action suggestions based on WAL state."""
    actions: List[str] = []

    if not report.status.wal_exists:
        actions.append("No WAL file found - database may not be in WAL mode")
        return actions

    if report.needs_checkpoint:
        actions.append("Run PRAGMA wal_checkpoint(TRUNCATE) to reduce WAL size")

    if report.status.wal_size_bytes > 10_000_000:
        actions.append("WAL size exceeds 10 MB - consider increasing checkpoint frequency")

    if report.status.estimated_pending_writes > 1000:
        actions.append("High number of pending writes - checkpoint recommended")

    if report.status.shm_exists and report.status.shm_size_bytes > 1_000_000:
        actions.append("SHM file is large - may indicate many concurrent readers")

    if not actions:
        actions.append("WAL is healthy - no action needed")

    return actions
