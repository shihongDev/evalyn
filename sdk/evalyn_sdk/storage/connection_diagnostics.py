"""Storage connection diagnostics: report SQLite configuration and health.

Provides functions to check database file existence, permissions, size,
journal mode, and WAL status, then aggregate into a health report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DiagnosticItem:
    """Result of a single diagnostic check."""

    name: str
    value: str
    status: str = "ok"
    recommendation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "recommendation": self.recommendation,
        }


@dataclass
class ConnectionHealth:
    """Aggregated health report for a database connection."""

    items: List[DiagnosticItem] = field(default_factory=list)
    overall_status: str = "ok"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "overall_status": self.overall_status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConnectionHealth:
        items = [
            DiagnosticItem(**item) for item in d.get("items", [])
        ]
        return cls(
            items=items,
            overall_status=d.get("overall_status", "ok"),
            warnings=d.get("warnings", []),
            errors=d.get("errors", []),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Connection Health Report")
        lines.append("-" * 40)
        lines.append(f"Overall status: {self.overall_status}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        if self.items:
            lines.append("")
            for item in self.items:
                lines.append(f"  [{item.status.upper()}] {item.name}: {item.value}")
                if item.recommendation:
                    lines.append(f"       {item.recommendation}")
        return "\n".join(lines)


def check_file_exists(db_path: str) -> DiagnosticItem:
    """Check whether the database file exists."""
    if os.path.isfile(db_path):
        return DiagnosticItem(
            name="file_exists",
            value="true",
            status="ok",
        )
    return DiagnosticItem(
        name="file_exists",
        value="false",
        status="error",
        recommendation="Database file not found - verify the path is correct",
    )


def check_file_permissions(db_path: str) -> DiagnosticItem:
    """Check read/write permissions on the database file."""
    readable = os.access(db_path, os.R_OK)
    writable = os.access(db_path, os.W_OK)

    if readable and writable:
        return DiagnosticItem(
            name="file_permissions",
            value="read/write",
            status="ok",
        )
    if readable:
        return DiagnosticItem(
            name="file_permissions",
            value="read-only",
            status="warning",
            recommendation="Database is read-only - check file permissions",
        )
    return DiagnosticItem(
        name="file_permissions",
        value="no access",
        status="error",
        recommendation="Cannot access database file - check ownership and permissions",
    )


def check_file_size(db_path: str, max_size_mb: float = 1000.0) -> DiagnosticItem:
    """Check database file size, warning if approaching max."""
    if not os.path.isfile(db_path):
        return DiagnosticItem(
            name="file_size",
            value="unknown",
            status="error",
            recommendation="File does not exist",
        )
    size_bytes = os.path.getsize(db_path)
    size_mb = size_bytes / (1024 * 1024)
    value = f"{size_mb:.2f} MB"

    if size_mb >= max_size_mb:
        return DiagnosticItem(
            name="file_size",
            value=value,
            status="error",
            recommendation=f"Database exceeds {max_size_mb} MB limit - consider archiving old data",
        )
    if size_mb >= max_size_mb * 0.8:
        return DiagnosticItem(
            name="file_size",
            value=value,
            status="warning",
            recommendation=f"Database approaching {max_size_mb} MB limit - consider cleanup",
        )
    return DiagnosticItem(
        name="file_size",
        value=value,
        status="ok",
    )


def check_wal_mode(db_path: str) -> DiagnosticItem:
    """Check if WAL mode is enabled by looking for a -wal file."""
    wal_path = db_path + "-wal"
    if os.path.isfile(wal_path):
        return DiagnosticItem(
            name="wal_mode",
            value="enabled",
            status="ok",
        )
    return DiagnosticItem(
        name="wal_mode",
        value="not detected",
        status="ok",
        recommendation="WAL mode not detected - consider enabling for better concurrency",
    )


def check_journal_file(db_path: str) -> DiagnosticItem:
    """Check for journal or WAL files alongside the database."""
    journal_path = db_path + "-journal"
    wal_path = db_path + "-wal"
    found: List[str] = []

    if os.path.isfile(journal_path):
        found.append("journal")
    if os.path.isfile(wal_path):
        found.append("wal")

    if found:
        return DiagnosticItem(
            name="journal_files",
            value=", ".join(found),
            status="ok",
        )
    return DiagnosticItem(
        name="journal_files",
        value="none",
        status="ok",
    )


def run_connection_diagnostics(db_path: str) -> ConnectionHealth:
    """Run all diagnostic checks and aggregate into a health report."""
    items: List[DiagnosticItem] = []
    warnings: List[str] = []
    errors: List[str] = []

    items.append(check_file_exists(db_path))
    if os.path.isfile(db_path):
        items.append(check_file_permissions(db_path))
        items.append(check_file_size(db_path))
        items.append(check_wal_mode(db_path))
        items.append(check_journal_file(db_path))

    for item in items:
        if item.status == "warning":
            warnings.append(f"{item.name}: {item.recommendation}")
        elif item.status == "error":
            errors.append(f"{item.name}: {item.recommendation}")

    if errors:
        overall = "error"
    elif warnings:
        overall = "warning"
    else:
        overall = "ok"

    return ConnectionHealth(
        items=items,
        overall_status=overall,
        warnings=warnings,
        errors=errors,
    )


def suggest_connection_fixes(health: ConnectionHealth) -> List[str]:
    """Generate fix suggestions from a health report."""
    suggestions: List[str] = []
    for item in health.items:
        if item.status in ("warning", "error") and item.recommendation:
            suggestions.append(item.recommendation)
    return suggestions
