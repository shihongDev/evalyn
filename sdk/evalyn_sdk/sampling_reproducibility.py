"""Sampling reproducibility: log which items were selected and why.

Provides dataclasses and pure functions for recording, saving, loading,
and verifying sampling selections. Pure Python, no external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SamplingParams:
    """Parameters that controlled a sampling operation."""

    mode: str
    seed: int | None = None
    sample_size: int = 0
    extra_params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "sample_size": self.sample_size,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingParams:
        return cls(
            mode=data["mode"],
            seed=data.get("seed"),
            sample_size=data.get("sample_size", 0),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class SamplingRecord:
    """Immutable record of a sampling selection for audit and reproducibility."""

    record_id: str
    params: SamplingParams
    selected_ids: list[str]
    total_pool: int
    timestamp: str
    checksum: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "params": self.params.as_dict(),
            "selected_ids": self.selected_ids,
            "total_pool": self.total_pool,
            "timestamp": self.timestamp,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingRecord:
        return cls(
            record_id=data["record_id"],
            params=SamplingParams.from_dict(data["params"]),
            selected_ids=data["selected_ids"],
            total_pool=data["total_pool"],
            timestamp=data["timestamp"],
            checksum=data["checksum"],
        )


def compute_selection_checksum(selected_ids: list[str]) -> str:
    """SHA256 of sorted, JSON-serialized selected IDs."""
    payload = json.dumps(sorted(selected_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_sampling_record(
    mode: str,
    seed: int | None,
    sample_size: int,
    selected_ids: list[str],
    total_pool: int,
    extra_params: dict[str, Any] | None = None,
) -> SamplingRecord:
    """Factory: build a SamplingRecord with auto-generated UUID and timestamp."""
    params = SamplingParams(
        mode=mode,
        seed=seed,
        sample_size=sample_size,
        extra_params=extra_params or {},
    )
    return SamplingRecord(
        record_id=str(uuid.uuid4()),
        params=params,
        selected_ids=list(selected_ids),
        total_pool=total_pool,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checksum=compute_selection_checksum(selected_ids),
    )


def save_sampling_record(record: SamplingRecord, path: str) -> None:
    """Persist a SamplingRecord as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record.as_dict(), f, indent=2)


def load_sampling_record(path: str) -> SamplingRecord | None:
    """Load a SamplingRecord from JSON. Returns None if file is missing."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return SamplingRecord.from_dict(data)
    except FileNotFoundError:
        return None


def verify_reproducibility(record: SamplingRecord, new_selected_ids: list[str]) -> bool:
    """Check whether new_selected_ids reproduce the same checksum as record."""
    return compute_selection_checksum(new_selected_ids) == record.checksum


def format_reproducibility_report(record: SamplingRecord) -> str:
    """Human-readable audit trail for a sampling record."""
    lines = [
        "Sampling Reproducibility Report",
        "=" * 40,
        f"Record ID : {record.record_id}",
        f"Timestamp : {record.timestamp}",
        f"Mode      : {record.params.mode}",
        f"Seed      : {record.params.seed}",
        f"Sample    : {record.params.sample_size}",
        f"Pool      : {record.total_pool}",
        f"Selected  : {len(record.selected_ids)}",
        f"Checksum  : {record.checksum}",
    ]
    if record.params.extra_params:
        lines.append(f"Extra     : {json.dumps(record.params.extra_params)}")
    lines.append("-" * 40)
    lines.append("Selected IDs:")
    for sid in record.selected_ids:
        lines.append(f"  {sid}")
    return "\n".join(lines)
