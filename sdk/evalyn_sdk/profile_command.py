"""Profile command: storage size, run counts, disk usage, and system health."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class StorageProfile:
    """Storage statistics for a data directory."""

    data_dir: str
    total_size_bytes: int
    file_count: int
    run_count: int
    trace_count: int
    annotation_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "total_size_bytes": self.total_size_bytes,
            "file_count": self.file_count,
            "run_count": self.run_count,
            "trace_count": self.trace_count,
            "annotation_count": self.annotation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StorageProfile:
        return cls(
            data_dir=data["data_dir"],
            total_size_bytes=data["total_size_bytes"],
            file_count=data["file_count"],
            run_count=data["run_count"],
            trace_count=data["trace_count"],
            annotation_count=data["annotation_count"],
        )


@dataclass
class EnvironmentProfile:
    """Environment and system information."""

    python_version: str
    evalyn_version: str
    platform: str
    installed_providers: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "evalyn_version": self.evalyn_version,
            "platform": self.platform,
            "installed_providers": list(self.installed_providers),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentProfile:
        return cls(
            python_version=data["python_version"],
            evalyn_version=data["evalyn_version"],
            platform=data["platform"],
            installed_providers=data.get("installed_providers", []),
        )


@dataclass
class ProfileReport:
    """Combined storage and environment profile."""

    storage: StorageProfile
    environment: EnvironmentProfile
    generated_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "storage": self.storage.as_dict(),
            "environment": self.environment.as_dict(),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProfileReport:
        return cls(
            storage=StorageProfile.from_dict(data["storage"]),
            environment=EnvironmentProfile.from_dict(data["environment"]),
            generated_at=data["generated_at"],
        )


# ---------------------------------------------------------------------------
# Provider env var mapping
# ---------------------------------------------------------------------------

_PROVIDER_ENV_VARS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def scan_data_directory(path: str) -> StorageProfile:
    """Walk a data directory and collect storage statistics.

    Counts:
    - total file size in bytes
    - total file count
    - run directories (matching run_* pattern)
    - trace files (.jsonl files)
    - annotation files (filenames starting with "annotation")
    """
    total_size = 0
    file_count = 0
    run_count = 0
    trace_count = 0
    annotation_count = 0

    if not os.path.isdir(path):
        return StorageProfile(
            data_dir=path,
            total_size_bytes=0,
            file_count=0,
            run_count=0,
            trace_count=0,
            annotation_count=0,
        )

    seen_run_dirs: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(path):
        # Count run directories
        for d in dirnames:
            if d.startswith("run_") and d not in seen_run_dirs:
                seen_run_dirs.add(d)
                run_count += 1

        for fname in filenames:
            filepath = os.path.join(dirpath, fname)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                pass
            file_count += 1

            if fname.endswith(".jsonl"):
                trace_count += 1

            if fname.startswith("annotation"):
                annotation_count += 1

    return StorageProfile(
        data_dir=path,
        total_size_bytes=total_size,
        file_count=file_count,
        run_count=run_count,
        trace_count=trace_count,
        annotation_count=annotation_count,
    )


def get_environment_profile() -> EnvironmentProfile:
    """Collect Python version, platform, evalyn version, and active providers."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    try:
        from evalyn_sdk import __version__ as evalyn_version
    except ImportError:
        evalyn_version = "unknown"

    plat = platform.platform()

    installed_providers: List[str] = []
    for provider_name, env_var in sorted(_PROVIDER_ENV_VARS.items()):
        if os.environ.get(env_var):
            installed_providers.append(provider_name)

    return EnvironmentProfile(
        python_version=python_version,
        evalyn_version=evalyn_version,
        platform=plat,
        installed_providers=installed_providers,
    )


def build_profile(data_dir: str = "data") -> ProfileReport:
    """Build a complete profile report combining storage and environment info."""
    storage = scan_data_directory(data_dir)
    environment = get_environment_profile()
    generated_at = datetime.now(timezone.utc).isoformat()
    return ProfileReport(
        storage=storage,
        environment=environment,
        generated_at=generated_at,
    )


def format_size_human(size_bytes: int) -> str:
    """Convert byte count to human-readable string (B/KB/MB/GB)."""
    if size_bytes < 0:
        return "0 B"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_profile_report(report: ProfileReport) -> str:
    """Format a ProfileReport as a human-readable string."""
    s = report.storage
    e = report.environment

    lines = [
        "PROFILE REPORT",
        f"Generated: {report.generated_at}",
        "",
        "Storage",
        f"  Directory:    {s.data_dir}",
        f"  Total size:   {format_size_human(s.total_size_bytes)}",
        f"  Files:        {s.file_count}",
        f"  Runs:         {s.run_count}",
        f"  Traces:       {s.trace_count}",
        f"  Annotations:  {s.annotation_count}",
        "",
        "Environment",
        f"  Python:       {e.python_version}",
        f"  Evalyn:       {e.evalyn_version}",
        f"  Platform:     {e.platform}",
        f"  Providers:    {', '.join(e.installed_providers) if e.installed_providers else 'none'}",
    ]

    return "\n".join(lines)
