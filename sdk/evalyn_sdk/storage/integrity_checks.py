"""Storage integrity checks: verify referential integrity between tables.

Provides functions to detect orphan references, duplicate IDs, missing
required fields, and type mismatches in record sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class IntegrityCheck:
    """Result of a single integrity check."""

    name: str
    passed: bool
    details: str = ""
    affected_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "affected_rows": self.affected_rows,
        }


@dataclass
class IntegrityReport:
    """Aggregated report of all integrity checks."""

    checks: list[IntegrityCheck] = field(default_factory=list)
    all_passed: bool = True
    total_checks: int = 0
    failed_checks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "all_passed": self.all_passed,
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntegrityReport:
        checks = [
            IntegrityCheck(**c) for c in d.get("checks", [])
        ]
        return cls(
            checks=checks,
            all_passed=d.get("all_passed", True),
            total_checks=d.get("total_checks", 0),
            failed_checks=d.get("failed_checks", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Integrity Report")
        lines.append("-" * 40)
        lines.append(f"Total checks: {self.total_checks}")
        lines.append(f"Failed checks: {self.failed_checks}")
        status = "PASS" if self.all_passed else "FAIL"
        lines.append(f"Overall: {status}")
        if self.checks:
            lines.append("")
            for check in self.checks:
                mark = "OK" if check.passed else "FAIL"
                lines.append(f"  [{mark}] {check.name}")
                if check.details:
                    lines.append(f"       {check.details}")
                if check.affected_rows > 0:
                    lines.append(
                        f"       affected rows: {check.affected_rows}"
                    )
        return "\n".join(lines)


def check_orphan_references(
    records: list[dict[str, Any]],
    valid_ids: set,
    id_field: str = "parent_id",
) -> IntegrityCheck:
    """Find records referencing non-existent IDs."""
    orphans = []
    for i, rec in enumerate(records):
        ref = rec.get(id_field)
        if ref is not None and ref not in valid_ids:
            orphans.append(i)

    if orphans:
        return IntegrityCheck(
            name="orphan_references",
            passed=False,
            details=f"Found {len(orphans)} orphan reference(s) in '{id_field}'",
            affected_rows=len(orphans),
        )
    return IntegrityCheck(
        name="orphan_references",
        passed=True,
        details="No orphan references found",
    )


def check_duplicate_ids(
    records: list[dict[str, Any]],
    id_field: str = "id",
) -> IntegrityCheck:
    """Find duplicate IDs in a record set."""
    seen: dict[Any, int] = {}
    duplicates = 0
    for rec in records:
        val = rec.get(id_field)
        if val is None:
            continue
        count = seen.get(val, 0)
        if count == 1:
            # First time seeing a duplicate for this value
            duplicates += 1
        seen[val] = count + 1

    if duplicates:
        return IntegrityCheck(
            name="duplicate_ids",
            passed=False,
            details=f"Found {duplicates} duplicate ID(s) in '{id_field}'",
            affected_rows=duplicates,
        )
    return IntegrityCheck(
        name="duplicate_ids",
        passed=True,
        details="No duplicate IDs found",
    )


def check_required_fields(
    records: list[dict[str, Any]],
    required: list[str],
) -> IntegrityCheck:
    """Check all records have required fields with non-None values."""
    missing_count = 0
    missing_fields: set = set()
    for rec in records:
        for f in required:
            if f not in rec or rec[f] is None:
                missing_count += 1
                missing_fields.add(f)

    if missing_count:
        fields_str = ", ".join(sorted(missing_fields))
        return IntegrityCheck(
            name="required_fields",
            passed=False,
            details=f"Missing required field(s): {fields_str} ({missing_count} violation(s))",
            affected_rows=missing_count,
        )
    return IntegrityCheck(
        name="required_fields",
        passed=True,
        details="All required fields present",
    )


def check_data_types(
    records: list[dict[str, Any]],
    field_types: dict[str, type],
) -> IntegrityCheck:
    """Verify field types match expected types."""
    violations = 0
    bad_fields: set = set()
    for rec in records:
        for fld, expected_type in field_types.items():
            val = rec.get(fld)
            if val is not None and not isinstance(val, expected_type):
                violations += 1
                bad_fields.add(fld)

    if violations:
        fields_str = ", ".join(sorted(bad_fields))
        return IntegrityCheck(
            name="data_types",
            passed=False,
            details=f"Type mismatch in field(s): {fields_str} ({violations} violation(s))",
            affected_rows=violations,
        )
    return IntegrityCheck(
        name="data_types",
        passed=True,
        details="All field types correct",
    )


def run_integrity_checks(
    records: list[dict[str, Any]],
    valid_parent_ids: set = None,
    required_fields: list[str] = None,
    field_types: dict[str, type] = None,
) -> IntegrityReport:
    """Run all applicable integrity checks on a record set.

    Always runs duplicate_ids check. Other checks run when their
    configuration parameters are provided (non-empty).
    """
    if valid_parent_ids is None:
        valid_parent_ids = set()
    if required_fields is None:
        required_fields = []
    if field_types is None:
        field_types = {}

    checks: list[IntegrityCheck] = []

    # Always check for duplicate IDs
    checks.append(check_duplicate_ids(records))

    # Orphan references - only if valid_parent_ids provided
    if valid_parent_ids:
        checks.append(
            check_orphan_references(records, valid_parent_ids)
        )

    # Required fields - only if specified
    if required_fields:
        checks.append(check_required_fields(records, required_fields))

    # Data types - only if specified
    if field_types:
        checks.append(check_data_types(records, field_types))

    failed = sum(1 for c in checks if not c.passed)
    return IntegrityReport(
        checks=checks,
        all_passed=(failed == 0),
        total_checks=len(checks),
        failed_checks=failed,
    )


def suggest_fixes(report: IntegrityReport) -> list[str]:
    """Generate fix suggestions for failed integrity checks."""
    suggestions: list[str] = []
    for check in report.checks:
        if check.passed:
            continue
        if check.name == "orphan_references":
            suggestions.append(
                "Remove or update records with orphan references "
                "to point to valid parent IDs"
            )
        elif check.name == "duplicate_ids":
            suggestions.append(
                "Deduplicate records by removing or renaming entries "
                "with duplicate IDs"
            )
        elif check.name == "required_fields":
            suggestions.append(
                "Fill in missing required fields or remove incomplete records"
            )
        elif check.name == "data_types":
            suggestions.append(
                "Cast or correct field values to match expected types"
            )
        else:
            suggestions.append(
                f"Review and fix issues reported by '{check.name}'"
            )
    return suggestions
