"""Data governance metadata for tracking provenance and compliance.

Pure Python, no external dependencies. Provides governance tagging,
compliance checking, and reporting for evaluation datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


class GovernanceTag:
    """Known governance tag constants."""

    PII_PRESENT = "pii_present"
    INTERNAL_ONLY = "internal_only"
    CUSTOMER_DATA = "customer_data"
    SYNTHETIC = "synthetic"
    PUBLIC = "public"
    ANONYMIZED = "anonymized"

    ALL = frozenset(
        {
            PII_PRESENT,
            INTERNAL_ONLY,
            CUSTOMER_DATA,
            SYNTHETIC,
            PUBLIC,
            ANONYMIZED,
        }
    )


@dataclass
class ComplianceFlag:
    """Result of a single compliance check."""

    flag_name: str
    passed: bool
    checked_at: str
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "flag_name": self.flag_name,
            "passed": self.passed,
            "checked_at": self.checked_at,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComplianceFlag:
        return cls(
            flag_name=data["flag_name"],
            passed=data["passed"],
            checked_at=data["checked_at"],
            details=data.get("details", ""),
        )


@dataclass
class DataGovernanceRecord:
    """Governance metadata for a single dataset."""

    dataset_id: str
    tags: list[str]
    compliance_flags: list[ComplianceFlag]
    owner: str
    created_at: str
    retention_days: int = 365
    classification: str = "internal"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "tags": list(self.tags),
            "compliance_flags": [f.as_dict() for f in self.compliance_flags],
            "owner": self.owner,
            "created_at": self.created_at,
            "retention_days": self.retention_days,
            "classification": self.classification,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataGovernanceRecord:
        return cls(
            dataset_id=data["dataset_id"],
            tags=list(data.get("tags", [])),
            compliance_flags=[
                ComplianceFlag.from_dict(f) for f in data.get("compliance_flags", [])
            ],
            owner=data.get("owner", ""),
            created_at=data["created_at"],
            retention_days=data.get("retention_days", 365),
            classification=data.get("classification", "internal"),
        )


@dataclass
class GovernanceReport:
    """Aggregated governance report across multiple datasets."""

    records: list[DataGovernanceRecord]
    summary: dict[str, Any]
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": [r.as_dict() for r in self.records],
            "summary": dict(self.summary),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernanceReport:
        return cls(
            records=[DataGovernanceRecord.from_dict(r) for r in data.get("records", [])],
            summary=dict(data.get("summary", {})),
            generated_at=data["generated_at"],
        )


def create_governance_record(
    dataset_id: str,
    tags: list[str],
    owner: str = "",
    classification: str = "internal",
) -> DataGovernanceRecord:
    """Create a new governance record with auto-timestamp."""
    return DataGovernanceRecord(
        dataset_id=dataset_id,
        tags=list(tags),
        compliance_flags=[],
        owner=owner,
        created_at=_now_iso(),
        classification=classification,
    )


def add_compliance_check(
    record: DataGovernanceRecord,
    flag_name: str,
    passed: bool,
    details: str = "",
) -> DataGovernanceRecord:
    """Return a new record with an additional compliance flag appended."""
    new_flag = ComplianceFlag(
        flag_name=flag_name,
        passed=passed,
        checked_at=_now_iso(),
        details=details,
    )
    new_flags = list(record.compliance_flags) + [new_flag]
    return DataGovernanceRecord(
        dataset_id=record.dataset_id,
        tags=list(record.tags),
        compliance_flags=new_flags,
        owner=record.owner,
        created_at=record.created_at,
        retention_days=record.retention_days,
        classification=record.classification,
    )


def validate_tags(tags: list[str]) -> list[str]:
    """Return list of tags not in the known governance tag set."""
    return [t for t in tags if t not in GovernanceTag.ALL]


def check_pii_compliance(record: DataGovernanceRecord) -> ComplianceFlag:
    """Check if PII-tagged data has proper handling.

    PII data must not have classification 'public'.
    """
    has_pii = GovernanceTag.PII_PRESENT in record.tags
    if not has_pii:
        return ComplianceFlag(
            flag_name="pii_compliance",
            passed=True,
            checked_at=_now_iso(),
            details="No PII tag present",
        )
    if record.classification == "public":
        return ComplianceFlag(
            flag_name="pii_compliance",
            passed=False,
            checked_at=_now_iso(),
            details="PII data must not be classified as public",
        )
    return ComplianceFlag(
        flag_name="pii_compliance",
        passed=True,
        checked_at=_now_iso(),
        details="PII data properly classified as " + record.classification,
    )


def generate_governance_report(
    records: list[DataGovernanceRecord],
) -> GovernanceReport:
    """Generate an aggregated governance report."""
    total = len(records)
    tag_counts: dict[str, int] = {}
    classification_counts: dict[str, int] = {}
    total_flags = 0
    passed_flags = 0
    failed_flags = 0

    for rec in records:
        for tag in rec.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        cls = rec.classification
        classification_counts[cls] = classification_counts.get(cls, 0) + 1
        for flag in rec.compliance_flags:
            total_flags += 1
            if flag.passed:
                passed_flags += 1
            else:
                failed_flags += 1

    summary: dict[str, Any] = {
        "total_records": total,
        "tag_counts": tag_counts,
        "classification_counts": classification_counts,
        "total_compliance_checks": total_flags,
        "passed_checks": passed_flags,
        "failed_checks": failed_flags,
    }
    return GovernanceReport(
        records=list(records),
        summary=summary,
        generated_at=_now_iso(),
    )


def format_governance_report(report: GovernanceReport) -> str:
    """Format a governance report as human-readable text."""
    lines: list[str] = []
    lines.append("Data Governance Report")
    lines.append("=" * 40)
    lines.append(f"Generated: {report.generated_at}")
    lines.append("")

    s = report.summary
    lines.append(f"Total records: {s.get('total_records', 0)}")
    lines.append(
        f"Compliance checks: {s.get('total_compliance_checks', 0)} "
        f"(passed: {s.get('passed_checks', 0)}, "
        f"failed: {s.get('failed_checks', 0)})"
    )
    lines.append("")

    tag_counts = s.get("tag_counts", {})
    if tag_counts:
        lines.append("Tags:")
        for tag, count in sorted(tag_counts.items()):
            lines.append(f"  {tag}: {count}")
        lines.append("")

    cls_counts = s.get("classification_counts", {})
    if cls_counts:
        lines.append("Classifications:")
        for cls, count in sorted(cls_counts.items()):
            lines.append(f"  {cls}: {count}")
        lines.append("")

    for rec in report.records:
        lines.append(f"Dataset: {rec.dataset_id}")
        lines.append(f"  Owner: {rec.owner}")
        lines.append(f"  Classification: {rec.classification}")
        lines.append(f"  Tags: {', '.join(rec.tags) if rec.tags else 'none'}")
        lines.append(f"  Retention: {rec.retention_days} days")
        if rec.compliance_flags:
            for flag in rec.compliance_flags:
                status = "PASS" if flag.passed else "FAIL"
                lines.append(f"  [{status}] {flag.flag_name}: {flag.details}")
        lines.append("")

    return "\n".join(lines)


def export_governance_report_json(report: GovernanceReport) -> str:
    """Export a governance report as a JSON string."""
    return json.dumps(report.as_dict(), indent=2)
