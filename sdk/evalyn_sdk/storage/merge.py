"""Storage merge: merge two databases with conflict resolution.

Provides dataclasses and pure functions to detect conflicts, resolve them
according to a strategy, and merge record sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MergeConflict:
    """A single merge conflict between two records."""

    table: str
    record_id: str
    field: str = ""
    value_a: str = ""
    value_b: str = ""
    resolution: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "record_id": self.record_id,
            "field": self.field,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "resolution": self.resolution,
        }


@dataclass
class MergeConfig:
    """Configuration for a merge operation."""

    strategy: str = "keep_newest"
    tables: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "tables": list(self.tables),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeConfig:
        return cls(
            strategy=data.get("strategy", "keep_newest"),
            tables=data.get("tables", []),
        )


@dataclass
class MergeResult:
    """Result summary of a merge operation."""

    merged_count: int = 0
    conflict_count: int = 0
    conflicts: list[MergeConflict] = field(default_factory=list)
    skipped_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "merged_count": self.merged_count,
            "conflict_count": self.conflict_count,
            "conflicts": [c.as_dict() for c in self.conflicts],
            "skipped_count": self.skipped_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeResult:
        conflicts = [
            MergeConflict(**c) for c in data.get("conflicts", [])
        ]
        return cls(
            merged_count=data.get("merged_count", 0),
            conflict_count=data.get("conflict_count", 0),
            conflicts=conflicts,
            skipped_count=data.get("skipped_count", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Merge Result")
        lines.append("-" * 40)
        lines.append(f"Merged records: {self.merged_count}")
        lines.append(f"Conflicts: {self.conflict_count}")
        lines.append(f"Skipped: {self.skipped_count}")
        if self.conflicts:
            lines.append("")
            for c in self.conflicts:
                res = f" -> {c.resolution}" if c.resolution else ""
                lines.append(
                    f"  [{c.table}] {c.record_id}.{c.field}: "
                    f"'{c.value_a}' vs '{c.value_b}'{res}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def detect_conflicts(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    id_field: str = "id",
) -> list[MergeConflict]:
    """Find records with the same ID but different field values."""
    index_a: dict[str, dict[str, Any]] = {}
    for rec in records_a:
        rid = str(rec.get(id_field, ""))
        if rid:
            index_a[rid] = rec

    conflicts: list[MergeConflict] = []
    for rec in records_b:
        rid = str(rec.get(id_field, ""))
        if rid and rid in index_a:
            rec_a = index_a[rid]
            all_keys = sorted(set(rec_a.keys()) | set(rec.keys()))
            for key in all_keys:
                if key == id_field:
                    continue
                val_a = rec_a.get(key)
                val_b = rec.get(key)
                if val_a != val_b:
                    conflicts.append(
                        MergeConflict(
                            table="",
                            record_id=rid,
                            field=key,
                            value_a=str(val_a) if val_a is not None else "",
                            value_b=str(val_b) if val_b is not None else "",
                        )
                    )
    return conflicts


def resolve_conflict(
    conflict: MergeConflict,
    strategy: str = "keep_newest",
) -> MergeConflict:
    """Apply a resolution strategy to a conflict.

    Strategies: "keep_a", "keep_b", "keep_newest" (defaults to keep_a).
    """
    if strategy == "keep_b":
        resolution = "keep_b"
    elif strategy == "keep_a":
        resolution = "keep_a"
    else:
        # keep_newest defaults to keep_a (caller should set timestamps)
        resolution = "keep_a"
    return MergeConflict(
        table=conflict.table,
        record_id=conflict.record_id,
        field=conflict.field,
        value_a=conflict.value_a,
        value_b=conflict.value_b,
        resolution=resolution,
    )


def merge_records(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    config: MergeConfig,
    id_field: str = "id",
) -> tuple[list[dict[str, Any]], MergeResult]:
    """Merge two record sets. Return (merged_records, result)."""
    index_a: dict[str, dict[str, Any]] = {}
    for rec in records_a:
        rid = str(rec.get(id_field, ""))
        if rid:
            index_a[rid] = dict(rec)

    conflicts: list[MergeConflict] = []
    skipped = 0

    for rec in records_b:
        rid = str(rec.get(id_field, ""))
        if not rid:
            skipped += 1
            continue

        if rid not in index_a:
            index_a[rid] = dict(rec)
            continue

        # Conflict: same ID exists in both sets
        rec_a = index_a[rid]
        all_keys = sorted(set(rec_a.keys()) | set(rec.keys()))
        has_diff = False
        for key in all_keys:
            if key == id_field:
                continue
            val_a = rec_a.get(key)
            val_b = rec.get(key)
            if val_a != val_b:
                has_diff = True
                conflicts.append(
                    MergeConflict(
                        table="",
                        record_id=rid,
                        field=key,
                        value_a=str(val_a) if val_a is not None else "",
                        value_b=str(val_b) if val_b is not None else "",
                    )
                )

        if not has_diff:
            # Identical records - no conflict
            continue

        if config.strategy == "fail_on_conflict":
            skipped += 1
            continue
        elif config.strategy == "keep_b":
            index_a[rid] = dict(rec)
        # keep_a and keep_newest: keep what is already in index_a

    resolved_conflicts = [
        resolve_conflict(c, config.strategy) for c in conflicts
    ]

    merged = list(index_a.values())
    result = MergeResult(
        merged_count=len(merged),
        conflict_count=len(resolved_conflicts),
        conflicts=resolved_conflicts,
        skipped_count=skipped,
    )
    return merged, result


def plan_merge(
    tables_a: dict[str, int],
    tables_b: dict[str, int],
) -> dict[str, Any]:
    """Plan showing which tables will be merged and estimated sizes."""
    all_tables = sorted(set(tables_a.keys()) | set(tables_b.keys()))
    plan: list[dict[str, Any]] = []
    for name in all_tables:
        count_a = tables_a.get(name, 0)
        count_b = tables_b.get(name, 0)
        plan.append({
            "table": name,
            "rows_a": count_a,
            "rows_b": count_b,
            "estimated_merged": count_a + count_b,
            "only_in_a": name not in tables_b,
            "only_in_b": name not in tables_a,
        })
    return {
        "tables": plan,
        "total_tables": len(all_tables),
    }


def validate_merge(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    id_field: str = "id",
) -> tuple[bool, list[str]]:
    """Validate that a merge is possible (same schema / compatible fields)."""
    errors: list[str] = []

    if not records_a and not records_b:
        return True, []

    if not records_a or not records_b:
        return True, []

    keys_a = set(records_a[0].keys())
    keys_b = set(records_b[0].keys())

    if id_field not in keys_a:
        errors.append(f"ID field '{id_field}' missing from first record set")
    if id_field not in keys_b:
        errors.append(f"ID field '{id_field}' missing from second record set")

    extra_in_a = sorted(keys_a - keys_b)
    extra_in_b = sorted(keys_b - keys_a)

    if extra_in_a:
        errors.append(f"Fields only in A: {', '.join(extra_in_a)}")
    if extra_in_b:
        errors.append(f"Fields only in B: {', '.join(extra_in_b)}")

    valid = len(errors) == 0
    return valid, errors
