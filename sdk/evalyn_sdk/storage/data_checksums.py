"""Storage data checksums: verify data integrity with per-row checksums.

Provides functions to compute, add, verify, and strip SHA-256 checksums
on dict records, plus batch verification and corruption detection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ChecksumRecord:
    """A single record's checksum metadata."""

    record_id: str
    checksum: str
    algorithm: str = "sha256"
    verified: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "checksum": self.checksum,
            "algorithm": self.algorithm,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChecksumRecord:
        return cls(
            record_id=d["record_id"],
            checksum=d["checksum"],
            algorithm=d.get("algorithm", "sha256"),
            verified=d.get("verified", True),
        )


@dataclass
class ChecksumReport:
    """Aggregated report of batch checksum verification."""

    total_records: int = 0
    verified: int = 0
    failed: int = 0
    missing: int = 0
    integrity_rate: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "verified": self.verified,
            "failed": self.failed,
            "missing": self.missing,
            "integrity_rate": self.integrity_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChecksumReport:
        return cls(
            total_records=d.get("total_records", 0),
            verified=d.get("verified", 0),
            failed=d.get("failed", 0),
            missing=d.get("missing", 0),
            integrity_rate=d.get("integrity_rate", 1.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Checksum Report")
        lines.append("-" * 40)
        lines.append(f"Total records: {self.total_records}")
        lines.append(f"Verified: {self.verified}")
        lines.append(f"Failed: {self.failed}")
        lines.append(f"Missing: {self.missing}")
        pct = self.integrity_rate * 100
        lines.append(f"Integrity rate: {pct:.1f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_checksum(
    data: dict[str, Any], algorithm: str = "sha256"
) -> str:
    """Compute checksum of a dict by JSON-serializing (sorted keys) and hashing.

    Returns the hex digest.
    """
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    h = hashlib.new(algorithm)
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def add_checksum(
    record: dict[str, Any], algorithm: str = "sha256"
) -> dict[str, Any]:
    """Add a '_checksum' field to a record. Returns a new dict."""
    # Compute checksum on the data without any existing _checksum
    data = {k: v for k, v in record.items() if k != "_checksum"}
    cs = compute_checksum(data, algorithm=algorithm)
    result = dict(data)
    result["_checksum"] = cs
    return result


def verify_checksum(
    record: dict[str, Any], algorithm: str = "sha256"
) -> bool:
    """Verify that a record's '_checksum' matches the computed checksum."""
    stored = record.get("_checksum")
    if stored is None:
        return False
    data = {k: v for k, v in record.items() if k != "_checksum"}
    expected = compute_checksum(data, algorithm=algorithm)
    return stored == expected


def verify_batch(records: list[dict[str, Any]]) -> ChecksumReport:
    """Verify all records and return an aggregated report."""
    total = len(records)
    verified = 0
    failed = 0
    missing = 0

    for rec in records:
        if "_checksum" not in rec:
            missing += 1
        elif verify_checksum(rec):
            verified += 1
        else:
            failed += 1

    rate = verified / total if total > 0 else 1.0
    return ChecksumReport(
        total_records=total,
        verified=verified,
        failed=failed,
        missing=missing,
        integrity_rate=rate,
    )


def find_corrupted(records: list[dict[str, Any]]) -> list[str]:
    """Return IDs of records with failed checksums."""
    result: list[str] = []
    for rec in records:
        if "_checksum" in rec and not verify_checksum(rec):
            record_id = rec.get("id", rec.get("record_id", "unknown"))
            result.append(str(record_id))
    return result


def strip_checksums(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove '_checksum' field from all records. Returns a new list."""
    return [
        {k: v for k, v in rec.items() if k != "_checksum"}
        for rec in records
    ]
