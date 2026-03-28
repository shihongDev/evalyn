"""Calibration freeze.

Lock calibration records to prevent accidental overwriting.
Provides a manager for freezing/unfreezing metric calibrations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FreezeRecord:
    """Record of a frozen calibration."""

    metric_id: str
    frozen_at: datetime
    frozen_by: str = ""
    reason: str = ""
    prompt_hash: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "frozen_at": self.frozen_at.isoformat(),
            "frozen_by": self.frozen_by,
            "reason": self.reason,
            "prompt_hash": self.prompt_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FreezeRecord:
        frozen_at = data["frozen_at"]
        if isinstance(frozen_at, str):
            frozen_at = datetime.fromisoformat(frozen_at)
        return cls(
            metric_id=data["metric_id"],
            frozen_at=frozen_at,
            frozen_by=data.get("frozen_by", ""),
            reason=data.get("reason", ""),
            prompt_hash=data.get("prompt_hash", ""),
        )


@dataclass
class FreezeStatus:
    """Status of a metric's freeze state."""

    is_frozen: bool
    record: Optional[FreezeRecord] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_frozen": self.is_frozen,
            "record": self.record.as_dict() if self.record else None,
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class CalibrationFreezeManager:
    """Manages freeze state for metric calibrations."""

    def __init__(self) -> None:
        self._frozen: Dict[str, FreezeRecord] = {}

    def freeze(
        self,
        metric_id: str,
        reason: str = "",
        frozen_by: str = "",
    ) -> FreezeRecord:
        """Lock a metric's calibration. Raises ValueError if already frozen."""
        if metric_id in self._frozen:
            raise ValueError(
                f"Metric '{metric_id}' is already frozen"
            )
        record = FreezeRecord(
            metric_id=metric_id,
            frozen_at=datetime.now(timezone.utc),
            frozen_by=frozen_by,
            reason=reason,
        )
        self._frozen[metric_id] = record
        return record

    def unfreeze(self, metric_id: str) -> bool:
        """Unlock a metric. Return True if it was frozen, False otherwise."""
        if metric_id in self._frozen:
            del self._frozen[metric_id]
            return True
        return False

    def is_frozen(self, metric_id: str) -> bool:
        """Check if a metric is frozen."""
        return metric_id in self._frozen

    def get_status(self, metric_id: str) -> FreezeStatus:
        """Get detailed freeze status for a metric."""
        record = self._frozen.get(metric_id)
        return FreezeStatus(
            is_frozen=record is not None,
            record=record,
        )

    def list_frozen(self) -> List[FreezeRecord]:
        """Return all frozen metric records."""
        return list(self._frozen.values())

    def check_can_calibrate(self, metric_id: str) -> Tuple[bool, str]:
        """Check if a metric can be calibrated.

        Returns (can_calibrate, reason). False if frozen.
        """
        if metric_id in self._frozen:
            record = self._frozen[metric_id]
            reason = f"Metric '{metric_id}' is frozen"
            if record.reason:
                reason += f": {record.reason}"
            return False, reason
        return True, ""

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the manager state."""
        return {
            "frozen": {
                mid: rec.as_dict() for mid, rec in self._frozen.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationFreezeManager:
        """Deserialize manager state."""
        manager = cls()
        for mid, rec_data in data.get("frozen", {}).items():
            manager._frozen[mid] = FreezeRecord.from_dict(rec_data)
        return manager
