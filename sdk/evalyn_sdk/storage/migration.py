from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class MigrationRecord:
    """Record of a completed migration."""

    source: str
    destination: str
    item_type: str
    item_count: int = 0
    timestamp: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "item_type": self.item_type,
            "item_count": self.item_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationRecord:
        return cls(
            source=data.get("source", ""),
            destination=data.get("destination", ""),
            item_type=data.get("item_type", ""),
            item_count=data.get("item_count", 0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class MigrationPlan:
    """Plan describing a pending migration."""

    source_path: str
    dest_path: str
    format: str = "json"  # "json" / "sqlite" / "csv"
    tables: List[str] = field(default_factory=list)
    estimated_items: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "dest_path": self.dest_path,
            "format": self.format,
            "tables": list(self.tables),
            "estimated_items": self.estimated_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationPlan:
        return cls(
            source_path=data.get("source_path", ""),
            dest_path=data.get("dest_path", ""),
            format=data.get("format", "json"),
            tables=data.get("tables", []),
            estimated_items=data.get("estimated_items", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Migration plan: {self.source_path} -> {self.dest_path}",
            f"  Format: {self.format}",
        ]
        if self.tables:
            lines.append(f"  Tables: {', '.join(self.tables)}")
        lines.append(f"  Estimated items: {self.estimated_items}")
        return "\n".join(lines)


@dataclass
class MigrationResult:
    """Result of a completed migration."""

    plan: MigrationPlan
    exported_count: int = 0
    imported_count: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "exported_count": self.exported_count,
            "imported_count": self.imported_count,
            "errors": list(self.errors),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MigrationResult:
        return cls(
            plan=MigrationPlan.from_dict(data.get("plan", {})),
            exported_count=data.get("exported_count", 0),
            imported_count=data.get("imported_count", 0),
            errors=data.get("errors", []),
            duration_ms=data.get("duration_ms", 0.0),
        )

    def format_text(self) -> str:
        status = "OK" if not self.errors else f"ERRORS ({len(self.errors)})"
        lines = [
            f"Migration result: {status}",
            f"  Source: {self.plan.source_path}",
            f"  Destination: {self.plan.dest_path}",
            f"  Exported: {self.exported_count}",
            f"  Imported: {self.imported_count}",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        for err in self.errors:
            lines.append(f"  Error: {err}")
        return "\n".join(lines)


_SUPPORTED_FORMATS = {"json", "sqlite", "csv"}


def plan_migration(
    source: str,
    dest: str,
    format: str = "json",
    tables: List[str] | None = None,
) -> MigrationPlan:
    """Create a migration plan."""
    estimated = 0
    if os.path.isfile(source):
        estimated = max(1, os.path.getsize(source) // 100)
    elif os.path.isdir(source):
        count = 0
        for entry in os.listdir(source):
            if entry.endswith(".json"):
                count += 1
        estimated = count

    return MigrationPlan(
        source_path=source,
        dest_path=dest,
        format=format,
        tables=tables if tables is not None else [],
        estimated_items=estimated,
    )


def export_to_json(data: List[Dict[str, Any]], file_path: str) -> int:
    """Write data to a JSON file. Returns the number of items written."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    return len(data)


def import_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Read data from a JSON file."""
    with open(file_path) as f:
        return json.load(f)


def validate_migration_plan(plan: MigrationPlan) -> Tuple[bool, List[str]]:
    """Validate a migration plan.

    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    if not plan.source_path:
        errors.append("source_path is empty")
    elif not os.path.exists(plan.source_path):
        errors.append(f"source does not exist: {plan.source_path}")

    if not plan.dest_path:
        errors.append("dest_path is empty")

    if plan.format not in _SUPPORTED_FORMATS:
        errors.append(f"unsupported format: {plan.format}")

    return (len(errors) == 0, errors)


def compute_migration_size(items: List[Dict[str, Any]]) -> int:
    """Estimate serialized size in bytes."""
    return len(json.dumps(items).encode("utf-8"))
