from __future__ import annotations

import copy
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models import Span

TRACKED_PACKAGES = [
    "google-generativeai",
    "openai",
    "anthropic",
    "langchain",
    "llama-index",
    "cohere",
    "mistralai",
]


@dataclass
class SDKVersionInfo:
    """Version info for a single provider SDK package."""

    package_name: str
    version: str = "unknown"
    installed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "version": self.version,
            "installed": self.installed,
        }


@dataclass
class VersionReport:
    """Report of all tracked SDK versions and any mismatches."""

    versions: list[SDKVersionInfo] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "versions": [v.as_dict() for v in self.versions],
            "mismatches": list(self.mismatches),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionReport:
        versions = [
            SDKVersionInfo(
                package_name=v["package_name"],
                version=v.get("version", "unknown"),
                installed=v.get("installed", False),
            )
            for v in data.get("versions", [])
        ]
        return cls(
            versions=versions,
            mismatches=data.get("mismatches", []),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("SDK Version Report")
        lines.append("-" * 40)
        for v in self.versions:
            status = v.version if v.installed else "not installed"
            lines.append(f"  {v.package_name}: {status}")
        if self.mismatches:
            lines.append("")
            lines.append("Mismatches:")
            for m in self.mismatches:
                lines.append(f"  - {m}")
        return "\n".join(lines)


def get_package_version(package_name: str) -> SDKVersionInfo:
    """Get the installed version of a package.

    Returns SDKVersionInfo with installed=False if the package is not found.
    """
    try:
        version = importlib.metadata.version(package_name)
        return SDKVersionInfo(
            package_name=package_name,
            version=version,
            installed=True,
        )
    except (importlib.metadata.PackageNotFoundError, ImportError):
        return SDKVersionInfo(
            package_name=package_name,
            version="unknown",
            installed=False,
        )


def detect_all_versions() -> VersionReport:
    """Check all tracked packages and return a version report."""
    versions = [get_package_version(pkg) for pkg in TRACKED_PACKAGES]
    return VersionReport(versions=versions)


def check_version_mismatch(spans: list[Span]) -> list[str]:
    """Compare span attribute 'evalyn.provider_sdk_version' across spans.

    Reports if different versions are found for the same provider.
    Returns a list of mismatch description strings.
    """
    # Collect provider -> set of versions
    provider_versions: dict[str, set] = {}
    for s in spans:
        sdk_ver = s.attributes.get("evalyn.provider_sdk_version")
        if sdk_ver and isinstance(sdk_ver, str):
            # Expected format: "provider_name==version"
            parts = sdk_ver.split("==", 1)
            if len(parts) == 2:
                provider, ver = parts
                provider_versions.setdefault(provider, set()).add(ver)

    mismatches: list[str] = []
    for provider, vers in sorted(provider_versions.items()):
        if len(vers) > 1:
            ver_list = ", ".join(sorted(vers))
            mismatches.append(
                f"{provider}: multiple versions found ({ver_list})"
            )
    return mismatches


def inject_version_info(span: Span) -> Span:
    """Add installed SDK versions to span attributes under 'evalyn.sdk_versions'.

    Returns a NEW span (does not mutate the original).
    """
    new_span = copy.deepcopy(span)
    report = detect_all_versions()
    installed = {
        v.package_name: v.version
        for v in report.versions
        if v.installed
    }
    new_span.attributes["evalyn.sdk_versions"] = installed
    return new_span
