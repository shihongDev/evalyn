"""
Storage migration versioning: formal migration version tracking with up/down support.

Provides dataclasses and a MigrationManager class to register, apply, and
rollback numbered migrations, plus utility functions for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MigrationStep:
    """A single versioned migration step."""

    version: int
    name: str
    up_description: str = ""
    down_description: str = ""
    applied_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "up_description": self.up_description,
            "down_description": self.down_description,
            "applied_at": self.applied_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationStep:
        return cls(
            version=data.get("version", 0),
            name=data.get("name", ""),
            up_description=data.get("up_description", ""),
            down_description=data.get("down_description", ""),
            applied_at=data.get("applied_at", ""),
        )


@dataclass
class MigrationHistory:
    """Full history of applied migrations."""

    steps: list[MigrationStep] = field(default_factory=list)
    current_version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "current_version": self.current_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationHistory:
        steps = [MigrationStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            steps=steps,
            current_version=data.get("current_version", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Migration History")
        lines.append("=" * 40)
        lines.append(f"Current version: {self.current_version}")
        lines.append(f"Total steps: {len(self.steps)}")
        if self.steps:
            lines.append("")
            for step in self.steps:
                applied = f" (applied: {step.applied_at})" if step.applied_at else ""
                lines.append(f"  v{step.version}: {step.name}{applied}")
        return "\n".join(lines)

    def is_up_to_date(self, target: int) -> bool:
        """Check if current version matches the target."""
        return self.current_version >= target

    def pending_count(self, target: int) -> int:
        """Count how many versions remain to reach target."""
        if self.current_version >= target:
            return 0
        return target - self.current_version


# ---------------------------------------------------------------------------
# MigrationManager
# ---------------------------------------------------------------------------


class MigrationManager:
    """Manages migration registration, application, and rollback."""

    def __init__(self) -> None:
        self._registered: list[MigrationStep] = []
        self._applied: list[MigrationStep] = []

    def register(
        self,
        version: int,
        name: str,
        up_desc: str = "",
        down_desc: str = "",
    ) -> None:
        """Register a migration step."""
        step = MigrationStep(
            version=version,
            name=name,
            up_description=up_desc,
            down_description=down_desc,
        )
        self._registered.append(step)
        self._registered.sort(key=lambda s: s.version)

    def apply(self, version: int) -> MigrationStep:
        """Mark a migration as applied. Raises ValueError if not registered."""
        step = self._find_registered(version)
        if step is None:
            raise ValueError(f"Migration version {version} is not registered")

        # Check if already applied
        for applied in self._applied:
            if applied.version == version:
                return applied

        applied_step = MigrationStep(
            version=step.version,
            name=step.name,
            up_description=step.up_description,
            down_description=step.down_description,
            applied_at=datetime.now(timezone.utc).isoformat(),
        )
        self._applied.append(applied_step)
        self._applied.sort(key=lambda s: s.version)
        return applied_step

    def rollback(self, version: int) -> MigrationStep | None:
        """Remove a migration from applied list. Returns None if not applied."""
        for i, step in enumerate(self._applied):
            if step.version == version:
                return self._applied.pop(i)
        return None

    def get_current_version(self) -> int:
        """Return the highest applied version, or 0 if none."""
        if not self._applied:
            return 0
        return max(s.version for s in self._applied)

    def get_pending(self) -> list[MigrationStep]:
        """Return registered migrations that have not been applied."""
        applied_versions = {s.version for s in self._applied}
        return [s for s in self._registered if s.version not in applied_versions]

    def get_history(self) -> MigrationHistory:
        """Build a full MigrationHistory from applied steps."""
        return MigrationHistory(
            steps=list(self._applied),
            current_version=self.get_current_version(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "registered": [s.as_dict() for s in self._registered],
            "applied": [s.as_dict() for s in self._applied],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationManager:
        mgr = cls()
        for s in data.get("registered", []):
            step = MigrationStep.from_dict(s)
            mgr._registered.append(step)
        mgr._registered.sort(key=lambda s: s.version)
        for s in data.get("applied", []):
            step = MigrationStep.from_dict(s)
            mgr._applied.append(step)
        mgr._applied.sort(key=lambda s: s.version)
        return mgr

    def _find_registered(self, version: int) -> MigrationStep | None:
        for step in self._registered:
            if step.version == version:
                return step
        return None


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_migration(
    version: int,
    name: str,
    up_desc: str = "",
    down_desc: str = "",
) -> MigrationStep:
    """Factory function to create a MigrationStep."""
    return MigrationStep(
        version=version,
        name=name,
        up_description=up_desc,
        down_description=down_desc,
    )


def validate_migration_order(
    steps: list[MigrationStep],
) -> tuple[bool, list[str]]:
    """Validate that migration versions are sequential starting from 1.

    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []
    if not steps:
        return (True, errors)

    sorted_steps = sorted(steps, key=lambda s: s.version)

    for i, step in enumerate(sorted_steps):
        expected = i + 1
        if step.version != expected:
            errors.append(
                f"Expected version {expected} but got {step.version}"
                f" for '{step.name}'"
            )

    # Check for duplicates
    versions = [s.version for s in steps]
    seen: set[int] = set()
    for v in versions:
        if v in seen:
            errors.append(f"Duplicate version: {v}")
        seen.add(v)

    return (len(errors) == 0, errors)
