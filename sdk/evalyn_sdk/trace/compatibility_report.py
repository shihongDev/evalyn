"""Instrumentation compatibility report.

Tracks which provider SDK versions have been tested and reports
compatibility status for installed packages.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Dict, List

KNOWN_COMPATIBLE_VERSIONS: dict[str, list[str]] = {
    "google-generativeai": ["0.8.0", "0.8.1", "0.8.2", "0.8.3"],
    "openai": ["1.50.0", "1.51.0", "1.52.0", "1.55.0", "1.57.0"],
    "anthropic": ["0.39.0", "0.40.0", "0.41.0", "0.42.0"],
}


@dataclass
class TestedVersion:
    """A specific provider version that has been tested."""

    provider: str
    version: str
    tested_at: str = ""
    status: str = "compatible"  # "compatible" / "untested" / "incompatible"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "version": self.version,
            "tested_at": self.tested_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestedVersion:
        return cls(
            provider=data["provider"],
            version=data["version"],
            tested_at=data.get("tested_at", ""),
            status=data.get("status", "compatible"),
        )


@dataclass
class CompatibilityEntry:
    """Compatibility info for a single provider."""

    provider: str
    current_version: str
    tested_versions: list[str] = field(default_factory=list)
    is_tested: bool = False
    warning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "current_version": self.current_version,
            "tested_versions": list(self.tested_versions),
            "is_tested": self.is_tested,
            "warning": self.warning,
        }


@dataclass
class CompatibilityReport:
    """Full compatibility report across all known providers."""

    entries: list[CompatibilityEntry] = field(default_factory=list)
    all_compatible: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "all_compatible": self.all_compatible,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompatibilityReport:
        entries = [
            CompatibilityEntry(
                provider=e["provider"],
                current_version=e.get("current_version", ""),
                tested_versions=e.get("tested_versions", []),
                is_tested=e.get("is_tested", False),
                warning=e.get("warning", ""),
            )
            for e in data.get("entries", [])
        ]
        return cls(
            entries=entries,
            all_compatible=data.get("all_compatible", True),
            warnings=data.get("warnings", []),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Compatibility Report")
        lines.append("-" * 40)
        for entry in self.entries:
            if entry.current_version:
                status = "tested" if entry.is_tested else "UNTESTED"
                lines.append(f"  {entry.provider}: {entry.current_version} [{status}]")
            else:
                lines.append(f"  {entry.provider}: not installed")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.all_compatible:
            lines.append("")
            lines.append("All installed providers are compatible.")
        return "\n".join(lines)


def get_installed_version(package: str) -> str:
    """Return the installed version of a package, or '' if not installed."""
    try:
        return importlib.metadata.version(package)
    except (importlib.metadata.PackageNotFoundError, ImportError):
        return ""


def is_version_tested(provider: str, version: str) -> bool:
    """Check whether a provider version is in the known compatible list."""
    known = KNOWN_COMPATIBLE_VERSIONS.get(provider, [])
    return version in known


def check_compatibility(provider: str) -> CompatibilityEntry:
    """Check compatibility for a single provider."""
    version = get_installed_version(provider)
    known = KNOWN_COMPATIBLE_VERSIONS.get(provider, [])

    if not version:
        return CompatibilityEntry(
            provider=provider,
            current_version="",
            tested_versions=known,
            is_tested=False,
            warning="",
        )

    tested = version in known
    warning = ""
    if not tested and version:
        warning = f"{provider}=={version} has not been tested"

    return CompatibilityEntry(
        provider=provider,
        current_version=version,
        tested_versions=known,
        is_tested=tested,
        warning=warning,
    )


def generate_compatibility_report() -> CompatibilityReport:
    """Check all known providers and return a full compatibility report."""
    entries: list[CompatibilityEntry] = []
    warnings: list[str] = []
    all_compatible = True

    for provider in KNOWN_COMPATIBLE_VERSIONS:
        entry = check_compatibility(provider)
        entries.append(entry)
        if entry.warning:
            warnings.append(entry.warning)
        if entry.current_version and not entry.is_tested:
            all_compatible = False

    return CompatibilityReport(
        entries=entries,
        all_compatible=all_compatible,
        warnings=warnings,
    )


def record_tested_version(provider: str, version: str) -> None:
    """Add a version to the known compatible list for dynamic updates."""
    if provider not in KNOWN_COMPATIBLE_VERSIONS:
        KNOWN_COMPATIBLE_VERSIONS[provider] = []
    if version not in KNOWN_COMPATIBLE_VERSIONS[provider]:
        KNOWN_COMPATIBLE_VERSIONS[provider].append(version)
