"""Instrumentation dry-run.

Show what would be patched without actually applying instrumentation.
Useful for verifying which providers are detected and what methods
would be wrapped before committing to live patching.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PatchTarget:
    """A single method that would be patched during instrumentation."""

    module_name: str
    class_name: str = ""
    method_name: str = ""
    provider: str = ""
    strategy: str = "wrap"

    @property
    def full_path(self) -> str:
        """Return 'module.class.method' path."""
        parts = [self.module_name]
        if self.class_name:
            parts.append(self.class_name)
        if self.method_name:
            parts.append(self.method_name)
        return ".".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "provider": self.provider,
            "strategy": self.strategy,
            "full_path": self.full_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatchTarget:
        return cls(
            module_name=data.get("module_name", ""),
            class_name=data.get("class_name", ""),
            method_name=data.get("method_name", ""),
            provider=data.get("provider", ""),
            strategy=data.get("strategy", "wrap"),
        )


@dataclass
class DryRunReport:
    """Report of what instrumentation would do."""

    targets: list[PatchTarget] = field(default_factory=list)
    total_patches: int = 0
    providers_detected: list[str] = field(default_factory=list)
    sdk_versions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "targets": [t.as_dict() for t in self.targets],
            "total_patches": self.total_patches,
            "providers_detected": list(self.providers_detected),
            "sdk_versions": dict(self.sdk_versions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DryRunReport:
        targets = [PatchTarget.from_dict(t) for t in data.get("targets", [])]
        return cls(
            targets=targets,
            total_patches=data.get("total_patches", 0),
            providers_detected=list(data.get("providers_detected", [])),
            sdk_versions=dict(data.get("sdk_versions", {})),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Instrumentation Dry Run")
        lines.append("-" * 40)
        lines.append(f"Providers detected: {len(self.providers_detected)}")
        for p in self.providers_detected:
            version = self.sdk_versions.get(p, "unknown")
            lines.append(f"  {p} ({version})")
        lines.append(f"Total patches: {self.total_patches}")
        if self.targets:
            lines.append("")
            lines.append(format_dry_run_table(self))
        return "\n".join(lines)


# -------------------------------------------------------------------------
# Known patch targets per provider
# -------------------------------------------------------------------------

KNOWN_PATCH_TARGETS: dict[str, list[PatchTarget]] = {
    "google-generativeai": [
        PatchTarget(
            "google.generativeai",
            "GenerativeModel",
            "generate_content",
            "gemini",
            "wrap",
        ),
        PatchTarget(
            "google.generativeai",
            "GenerativeModel",
            "generate_content_async",
            "gemini",
            "wrap",
        ),
    ],
    "openai": [
        PatchTarget(
            "openai.resources.chat",
            "Completions",
            "create",
            "openai",
            "wrap",
        ),
    ],
    "anthropic": [
        PatchTarget(
            "anthropic.resources",
            "Messages",
            "create",
            "anthropic",
            "wrap",
        ),
    ],
}


# -------------------------------------------------------------------------
# Detection and dry-run functions
# -------------------------------------------------------------------------


def detect_installed_providers() -> list[str]:
    """Check which provider packages are installed using importlib.metadata."""
    installed: list[str] = []
    for pkg_name in KNOWN_PATCH_TARGETS:
        try:
            importlib.metadata.version(pkg_name)
            installed.append(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return installed


def get_patch_targets(provider: str) -> list[PatchTarget]:
    """Return known patch targets for a provider."""
    return list(KNOWN_PATCH_TARGETS.get(provider, []))


def run_dry_run() -> DryRunReport:
    """Detect installed providers and list all targets that would be patched."""
    providers = detect_installed_providers()
    all_targets: list[PatchTarget] = []
    sdk_versions: dict[str, str] = {}

    for pkg in providers:
        targets = get_patch_targets(pkg)
        all_targets.extend(targets)
        try:
            sdk_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            sdk_versions[pkg] = "unknown"

    return DryRunReport(
        targets=all_targets,
        total_patches=len(all_targets),
        providers_detected=providers,
        sdk_versions=sdk_versions,
    )


def format_dry_run_table(report: DryRunReport) -> str:
    """ASCII table showing provider, module, class, method, strategy."""
    headers = ["Provider", "Module", "Class", "Method", "Strategy"]
    rows: list[list[str]] = []
    for t in report.targets:
        rows.append(
            [
                t.provider,
                t.module_name,
                t.class_name,
                t.method_name,
                t.strategy,
            ]
        )

    if not rows:
        return "No patch targets detected."

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Build table
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)
