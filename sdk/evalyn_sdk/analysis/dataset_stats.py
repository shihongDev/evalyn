"""Dataset statistics and health check utilities.

Provides auto-computed statistics and quality validation for datasets
before evaluation. No external dependencies.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetStats:
    """Auto-computed statistics for a dataset."""

    item_count: int = 0
    input_lengths: list[int] = field(default_factory=list)
    output_lengths: list[int] = field(default_factory=list)
    has_reference_count: int = 0  # items with human_label
    metadata_fields: dict[str, int] = field(default_factory=dict)  # field -> count
    duplicate_input_count: int = 0
    empty_output_count: int = 0
    empty_input_count: int = 0

    @property
    def avg_input_length(self) -> float:
        return sum(self.input_lengths) / len(self.input_lengths) if self.input_lengths else 0.0

    @property
    def avg_output_length(self) -> float:
        return sum(self.output_lengths) / len(self.output_lengths) if self.output_lengths else 0.0

    @property
    def max_input_length(self) -> int:
        return max(self.input_lengths) if self.input_lengths else 0

    @property
    def max_output_length(self) -> int:
        return max(self.output_lengths) if self.output_lengths else 0

    @property
    def reference_coverage(self) -> float:
        """Fraction of items with human labels/references."""
        return self.has_reference_count / self.item_count if self.item_count > 0 else 0.0

    @property
    def duplicate_rate(self) -> float:
        return self.duplicate_input_count / self.item_count if self.item_count > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_count": self.item_count,
            "avg_input_length": round(self.avg_input_length, 1),
            "avg_output_length": round(self.avg_output_length, 1),
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "has_reference_count": self.has_reference_count,
            "reference_coverage": round(self.reference_coverage, 4),
            "duplicate_input_count": self.duplicate_input_count,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "empty_input_count": self.empty_input_count,
            "empty_output_count": self.empty_output_count,
            "metadata_fields": dict(self.metadata_fields),
        }

    def format_text(self) -> str:
        lines = [
            "Dataset Statistics",
            "=" * 40,
            f"Items:              {self.item_count}",
            f"Avg input length:   {self.avg_input_length:.0f} chars",
            f"Avg output length:  {self.avg_output_length:.0f} chars",
            f"Max input length:   {self.max_input_length} chars",
            f"Max output length:  {self.max_output_length} chars",
            f"Reference coverage: {self.reference_coverage:.0%} ({self.has_reference_count}/{self.item_count})",
            f"Duplicate inputs:   {self.duplicate_input_count} ({self.duplicate_rate:.1%})",
            f"Empty inputs:       {self.empty_input_count}",
            f"Empty outputs:      {self.empty_output_count}",
        ]
        if self.metadata_fields:
            lines.append("")
            lines.append("Metadata fields:")
            for field_name, count in sorted(self.metadata_fields.items()):
                lines.append(f"  {field_name}: {count}/{self.item_count} items")
        return "\n".join(lines)


@dataclass
class HealthCheckResult:
    """Result of a dataset health check."""

    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    def format_text(self) -> str:
        lines = [f"Health Check: {self.status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        if not self.errors and not self.warnings:
            lines.append("  All checks passed.")
        return "\n".join(lines)


def compute_dataset_stats(items: list) -> DatasetStats:
    """Compute statistics for a dataset.

    Args:
        items: List of DatasetItem objects.

    Returns:
        DatasetStats with computed values.
    """
    stats = DatasetStats(item_count=len(items))

    seen_hashes: set[str] = set()
    metadata_counter: Counter = Counter()

    for item in items:
        # Input/output lengths
        input_text = _to_text(item.input)
        output_text = _to_text(item.output)
        stats.input_lengths.append(len(input_text))
        stats.output_lengths.append(len(output_text))

        # Empty checks
        if not input_text.strip():
            stats.empty_input_count += 1
        if not output_text.strip():
            stats.empty_output_count += 1

        # Reference coverage
        if item.human_label:
            stats.has_reference_count += 1

        # Duplicate detection via input hash
        import hashlib

        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
        if input_hash in seen_hashes:
            stats.duplicate_input_count += 1
        seen_hashes.add(input_hash)

        # Metadata field tracking
        if item.metadata:
            for key in item.metadata:
                metadata_counter[key] += 1

    stats.metadata_fields = dict(metadata_counter)
    return stats


def check_dataset_health(
    items: list,
    *,
    min_items: int = 5,
    max_duplicate_rate: float = 0.20,
    max_empty_rate: float = 0.10,
    require_output: bool = True,
) -> HealthCheckResult:
    """Validate dataset quality before evaluation.

    Args:
        items: List of DatasetItem objects.
        min_items: Minimum required items (below = error).
        max_duplicate_rate: Maximum acceptable duplicate input rate.
        max_empty_rate: Maximum acceptable empty input/output rate.
        require_output: If True, warn when items have no output.

    Returns:
        HealthCheckResult with pass/fail status, warnings, and errors.
    """
    errors = []
    warnings = []

    if len(items) == 0:
        return HealthCheckResult(passed=False, errors=["Dataset is empty"])

    stats = compute_dataset_stats(items)

    # Errors (block evaluation)
    if stats.item_count < min_items:
        errors.append(f"Too few items: {stats.item_count} (minimum: {min_items})")

    # Warnings (allow evaluation but flag issues)
    if stats.duplicate_rate > max_duplicate_rate:
        warnings.append(
            f"High duplicate rate: {stats.duplicate_rate:.1%} ({stats.duplicate_input_count} duplicates)"
        )

    empty_input_rate = stats.empty_input_count / stats.item_count
    if empty_input_rate > max_empty_rate:
        warnings.append(
            f"Many empty inputs: {stats.empty_input_count}/{stats.item_count} ({empty_input_rate:.1%})"
        )

    if require_output:
        empty_output_rate = stats.empty_output_count / stats.item_count
        if empty_output_rate > max_empty_rate:
            warnings.append(
                f"Many empty outputs: {stats.empty_output_count}/{stats.item_count} ({empty_output_rate:.1%})"
            )

    if stats.reference_coverage == 0:
        warnings.append(
            "No items have human labels/references (subjective calibration requires labels)"
        )

    passed = len(errors) == 0
    return HealthCheckResult(passed=passed, warnings=warnings, errors=errors)


def _to_text(value: Any) -> str:
    """Convert value to text for length calculation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)
