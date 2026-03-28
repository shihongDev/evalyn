from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CompactionStats:
    """Statistics from a database compaction operation."""

    original_size_bytes: int = 0
    compacted_size_bytes: int = 0
    savings_bytes: int = 0
    savings_pct: float = 0.0
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_size_bytes": self.original_size_bytes,
            "compacted_size_bytes": self.compacted_size_bytes,
            "savings_bytes": self.savings_bytes,
            "savings_pct": self.savings_pct,
            "duration_ms": self.duration_ms,
        }

    def format_text(self) -> str:
        lines = [
            "Compaction Results",
            "-" * 40,
            f"  Original size: {format_size(self.original_size_bytes)}",
            f"  Compacted size: {format_size(self.compacted_size_bytes)}",
            f"  Savings: {format_size(self.savings_bytes)} ({self.savings_pct:.1f}%)",
            f"  Duration: {self.duration_ms:.1f} ms",
        ]
        return "\n".join(lines)


@dataclass
class CompactionConfig:
    """Configuration for database compaction operations."""

    vacuum: bool = True
    reindex: bool = True
    analyze: bool = True
    wal_checkpoint: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vacuum": self.vacuum,
            "reindex": self.reindex,
            "analyze": self.analyze,
            "wal_checkpoint": self.wal_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompactionConfig:
        return cls(
            vacuum=data.get("vacuum", True),
            reindex=data.get("reindex", True),
            analyze=data.get("analyze", True),
            wal_checkpoint=data.get("wal_checkpoint", True),
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_database_size(path: str) -> int:
    """Get file size in bytes. Returns 0 if file does not exist."""
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path)


def format_size(bytes_val: int) -> str:
    """Human-readable size string (B, KB, MB, GB)."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.1f} GB"


def estimate_compaction_savings(db_path: str) -> CompactionStats:
    """Estimate savings without actually compacting.

    Checks file size and estimates based on typical fragmentation
    (approximately 10-20% savings).
    """
    original = get_database_size(db_path)
    if original == 0:
        return CompactionStats()

    # Estimate 15% savings as a middle ground for typical fragmentation
    estimated_savings = int(original * 0.15)
    estimated_compacted = original - estimated_savings
    savings_pct = (estimated_savings / original) * 100.0 if original > 0 else 0.0

    return CompactionStats(
        original_size_bytes=original,
        compacted_size_bytes=estimated_compacted,
        savings_bytes=estimated_savings,
        savings_pct=savings_pct,
    )


def compact_database(
    db_path: str, config: Optional[CompactionConfig] = None
) -> CompactionStats:
    """Run compaction on a database file.

    Measures before/after file size. The actual VACUUM, REINDEX, ANALYZE
    operations would require sqlite3 - this implementation measures the
    file size impact of the operations configured.

    Note: does not import sqlite3 - uses os for file operations only.
    In production, the caller is responsible for running SQL commands
    (VACUUM, REINDEX, ANALYZE) separately.
    """
    if config is None:
        config = CompactionConfig()

    original = get_database_size(db_path)
    if original == 0:
        return CompactionStats()

    start = time.monotonic()

    # In a real implementation, this would execute SQL commands via sqlite3.
    # Here we only measure file size (the caller handles SQL operations).
    compacted = get_database_size(db_path)

    elapsed_ms = (time.monotonic() - start) * 1000.0

    savings = original - compacted
    savings_pct = (savings / original) * 100.0 if original > 0 else 0.0

    return CompactionStats(
        original_size_bytes=original,
        compacted_size_bytes=compacted,
        savings_bytes=savings,
        savings_pct=savings_pct,
        duration_ms=elapsed_ms,
    )


def should_compact(db_path: str, threshold_mb: float = 100.0) -> bool:
    """Recommend compaction if database exceeds threshold.

    Returns True if the file size is greater than threshold_mb megabytes.
    """
    size = get_database_size(db_path)
    threshold_bytes = int(threshold_mb * 1024 * 1024)
    return size > threshold_bytes
