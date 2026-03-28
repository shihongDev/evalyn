"""Metric deprecation lifecycle: formal deprecation with migration hints.

Track deprecated metrics, warn users, and provide migration guidance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeprecationEntry:
    """A deprecated metric with migration information."""

    metric_id: str
    replacement: Optional[str] = None   # replacement metric ID
    deprecated_since: str = ""          # version string
    sunset_date: Optional[str] = None   # date when metric will be removed
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "replacement": self.replacement,
            "deprecated_since": self.deprecated_since,
            "sunset_date": self.sunset_date,
            "reason": self.reason,
        }

    def format_warning(self) -> str:
        msg = f"Metric '{self.metric_id}' is deprecated"
        if self.deprecated_since:
            msg += f" (since {self.deprecated_since})"
        if self.replacement:
            msg += f". Use '{self.replacement}' instead"
        if self.sunset_date:
            msg += f". Will be removed after {self.sunset_date}"
        if self.reason:
            msg += f". Reason: {self.reason}"
        return msg


class DeprecationRegistry:
    """Registry of deprecated metrics with warnings and migration hints.

    Usage:
        registry = DeprecationRegistry()
        registry.deprecate("old_metric", replacement="new_metric", since="0.15.0")
        warnings = registry.check_metrics(["old_metric", "current_metric"])
    """

    def __init__(self):
        self._entries: Dict[str, DeprecationEntry] = {}
        self._warned: set = set()  # track which warnings have been issued

    def deprecate(
        self,
        metric_id: str,
        replacement: Optional[str] = None,
        since: str = "",
        sunset_date: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """Register a metric as deprecated.

        Args:
            metric_id: The deprecated metric.
            replacement: ID of the replacement metric.
            since: Version when deprecation was introduced.
            sunset_date: Date when metric will be removed.
            reason: Why the metric is deprecated.
        """
        self._entries[metric_id] = DeprecationEntry(
            metric_id=metric_id,
            replacement=replacement,
            deprecated_since=since,
            sunset_date=sunset_date,
            reason=reason,
        )

    def is_deprecated(self, metric_id: str) -> bool:
        """Check if a metric is deprecated."""
        return metric_id in self._entries

    def get_entry(self, metric_id: str) -> Optional[DeprecationEntry]:
        """Get deprecation info for a metric."""
        return self._entries.get(metric_id)

    def check_metrics(self, metric_ids: List[str]) -> List[str]:
        """Check a list of metric IDs for deprecations.

        Returns list of warning messages for any deprecated metrics found.
        Each warning is issued at most once per session.
        """
        warnings = []
        for mid in metric_ids:
            entry = self._entries.get(mid)
            if entry and mid not in self._warned:
                warnings.append(entry.format_warning())
                self._warned.add(mid)
                logger.warning(entry.format_warning())
        return warnings

    def list_deprecated(self) -> List[DeprecationEntry]:
        """List all deprecated metrics."""
        return sorted(self._entries.values(), key=lambda e: e.metric_id)

    def migration_guide(self) -> str:
        """Generate a migration guide for all deprecated metrics."""
        if not self._entries:
            return "No deprecated metrics."
        lines = ["Metric Migration Guide:", ""]
        for entry in self.list_deprecated():
            lines.append(f"  {entry.metric_id}:")
            if entry.replacement:
                lines.append(f"    Replace with: {entry.replacement}")
            if entry.reason:
                lines.append(f"    Reason: {entry.reason}")
            if entry.sunset_date:
                lines.append(f"    Sunset: {entry.sunset_date}")
            lines.append("")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._entries)
