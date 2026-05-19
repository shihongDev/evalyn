"""
Data retention policies: auto-delete traces and runs older than a threshold.

Provides pure functions to identify candidates for deletion, apply retention
policies, compute savings, and suggest policies based on storage size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class RetentionPolicy:
    """Configurable retention policy for stored data."""

    max_age_days: int = 90
    max_runs: int | None = None
    max_size_mb: float | None = None
    exclude_pinned: bool = True
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_age_days": self.max_age_days,
            "max_runs": self.max_runs,
            "max_size_mb": self.max_size_mb,
            "exclude_pinned": self.exclude_pinned,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetentionPolicy:
        return cls(
            max_age_days=data.get("max_age_days", 90),
            max_runs=data.get("max_runs"),
            max_size_mb=data.get("max_size_mb"),
            exclude_pinned=data.get("exclude_pinned", True),
            dry_run=data.get("dry_run", False),
        )


@dataclass
class RetentionCandidate:
    """An item identified for potential deletion."""

    item_id: str
    item_type: str  # "run", "trace", or "span"
    age_days: int
    size_bytes: int = 0
    pinned: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "age_days": self.age_days,
            "size_bytes": self.size_bytes,
            "pinned": self.pinned,
            "reason": self.reason,
        }


@dataclass
class RetentionResult:
    """Result of applying a retention policy."""

    candidates: list[RetentionCandidate] = field(default_factory=list)
    deleted_count: int = 0
    freed_bytes: int = 0
    skipped_pinned: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.as_dict() for c in self.candidates],
            "deleted_count": self.deleted_count,
            "freed_bytes": self.freed_bytes,
            "skipped_pinned": self.skipped_pinned,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetentionResult:
        candidates = [
            RetentionCandidate(
                item_id=c["item_id"],
                item_type=c["item_type"],
                age_days=c["age_days"],
                size_bytes=c.get("size_bytes", 0),
                pinned=c.get("pinned", False),
                reason=c.get("reason", ""),
            )
            for c in data.get("candidates", [])
        ]
        return cls(
            candidates=candidates,
            deleted_count=data.get("deleted_count", 0),
            freed_bytes=data.get("freed_bytes", 0),
            skipped_pinned=data.get("skipped_pinned", 0),
            dry_run=data.get("dry_run", False),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        mode = "DRY RUN" if self.dry_run else "APPLIED"
        lines.append(f"Retention Result ({mode})")
        lines.append("=" * 40)
        lines.append(f"Candidates found: {len(self.candidates)}")
        lines.append(f"Deleted: {self.deleted_count}")
        lines.append(f"Freed: {self.freed_bytes} bytes")
        if self.skipped_pinned > 0:
            lines.append(f"Skipped (pinned): {self.skipped_pinned}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def identify_candidates(
    items: list[dict[str, Any]], policy: RetentionPolicy
) -> list[RetentionCandidate]:
    """Find items exceeding the retention policy.

    Each item dict should have: "id", "type", "age_days", "size_bytes", "pinned".
    """
    candidates: list[RetentionCandidate] = []
    for item in items:
        age_days = item.get("age_days", 0)
        pinned = item.get("pinned", False)

        if age_days <= policy.max_age_days:
            continue

        reason = f"age {age_days}d exceeds max {policy.max_age_days}d"
        candidates.append(
            RetentionCandidate(
                item_id=item.get("id", ""),
                item_type=item.get("type", "run"),
                age_days=age_days,
                size_bytes=item.get("size_bytes", 0),
                pinned=pinned,
                reason=reason,
            )
        )

    return candidates


def apply_retention(
    candidates: list[RetentionCandidate], policy: RetentionPolicy
) -> RetentionResult:
    """Apply retention policy to candidates.

    In dry_run mode, counts what would be deleted but does not actually delete.
    Skips pinned items when exclude_pinned is True.
    """
    deleted_count = 0
    freed_bytes = 0
    skipped_pinned = 0

    for c in candidates:
        if policy.exclude_pinned and c.pinned:
            skipped_pinned += 1
            continue
        deleted_count += 1
        freed_bytes += c.size_bytes

    return RetentionResult(
        candidates=candidates,
        deleted_count=deleted_count,
        freed_bytes=freed_bytes,
        skipped_pinned=skipped_pinned,
        dry_run=policy.dry_run,
    )


def compute_retention_savings(candidates: list[RetentionCandidate]) -> int:
    """Total bytes that would be freed by deleting all candidates."""
    return sum(c.size_bytes for c in candidates)


def validate_policy(policy: RetentionPolicy) -> tuple[bool, list[str]]:
    """Validate a retention policy. Returns (valid, list of errors)."""
    errors: list[str] = []

    if policy.max_age_days <= 0:
        errors.append("max_age_days must be positive")

    if policy.max_runs is not None and policy.max_runs <= 0:
        errors.append("max_runs must be positive if set")

    if policy.max_size_mb is not None and policy.max_size_mb <= 0:
        errors.append("max_size_mb must be positive if set")

    return (len(errors) == 0, errors)


def suggest_retention_policy(
    total_size_mb: float, total_items: int
) -> RetentionPolicy:
    """Auto-suggest a retention policy based on storage size and item count."""
    # Aggressive retention for large storage
    if total_size_mb > 1000:
        max_age = 30
    elif total_size_mb > 500:
        max_age = 60
    elif total_size_mb > 100:
        max_age = 90
    else:
        max_age = 180

    max_runs: int | None = None
    if total_items > 10_000:
        max_runs = 5000
    elif total_items > 1000:
        max_runs = 1000

    max_size: float | None = None
    if total_size_mb > 500:
        max_size = total_size_mb * 0.8

    return RetentionPolicy(
        max_age_days=max_age,
        max_runs=max_runs,
        max_size_mb=max_size,
        exclude_pinned=True,
        dry_run=True,  # default to dry run for safety
    )
