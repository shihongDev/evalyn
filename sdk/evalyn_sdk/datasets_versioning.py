"""Dataset versioning: track changes with snapshots and diff.

Each build-dataset operation creates a versioned snapshot. Supports
viewing version history and rolling back to previous versions.

Storage layout:
    data/prod/datasets/my_project/
      dataset.jsonl           # current version
      meta.json               # includes version_hash
      .versions/
        v_<hash>.jsonl        # snapshots
        changelog.json        # [{version, timestamp, item_count, hash}]
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetVersion:
    """A recorded version of a dataset."""

    version_hash: str
    item_count: int
    created_at: str  # ISO timestamp
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version_hash": self.version_hash,
            "item_count": self.item_count,
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetVersion:
        return cls(
            version_hash=data["version_hash"],
            item_count=data["item_count"],
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )


def compute_version_hash(dataset_path: Path) -> str:
    """Compute a hash of the dataset file content.

    Args:
        dataset_path: Path to dataset.jsonl file.

    Returns:
        SHA-256 hash (16 chars) of the file content.
    """
    if not dataset_path.exists():
        return ""
    content = dataset_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def create_snapshot(
    dataset_dir: Path,
    description: str = "",
) -> DatasetVersion | None:
    """Create a versioned snapshot of the current dataset.

    Copies current dataset.jsonl to .versions/v_<hash>.jsonl and
    records the version in changelog.json.

    Args:
        dataset_dir: Directory containing dataset.jsonl.
        description: Optional description for this version.

    Returns:
        DatasetVersion if snapshot created, None if no dataset exists.
    """
    dataset_file = dataset_dir / "dataset.jsonl"
    if not dataset_file.exists():
        return None

    versions_dir = dataset_dir / ".versions"
    versions_dir.mkdir(exist_ok=True)

    # Compute hash
    version_hash = compute_version_hash(dataset_file)

    # Check if this version already exists
    snapshot_path = versions_dir / f"v_{version_hash}.jsonl"
    if snapshot_path.exists():
        # Already have this version, no need to copy
        return None

    # Copy current dataset to snapshot
    shutil.copy2(dataset_file, snapshot_path)

    # Count items
    item_count = sum(1 for line in dataset_file.read_text(encoding="utf-8").splitlines() if line.strip())

    # Record in changelog
    version = DatasetVersion(
        version_hash=version_hash,
        item_count=item_count,
        created_at=datetime.now(timezone.utc).isoformat(),
        description=description,
    )
    _append_changelog(versions_dir, version)

    return version


def list_versions(dataset_dir: Path) -> list[DatasetVersion]:
    """List all versioned snapshots for a dataset.

    Args:
        dataset_dir: Directory containing the dataset.

    Returns:
        List of DatasetVersion objects, newest first.
    """
    changelog_path = dataset_dir / ".versions" / "changelog.json"
    if not changelog_path.exists():
        return []

    try:
        with open(changelog_path, encoding="utf-8") as f:
            entries = json.load(f)
        versions = [DatasetVersion.from_dict(e) for e in entries]
        return list(reversed(versions))  # newest first
    except (json.JSONDecodeError, KeyError):
        return []


def rollback_to_version(
    dataset_dir: Path,
    version_hash: str,
) -> bool:
    """Restore a previous dataset version.

    Creates a snapshot of the current version first (safety), then
    replaces dataset.jsonl with the requested version.

    Args:
        dataset_dir: Directory containing the dataset.
        version_hash: Hash of the version to restore.

    Returns:
        True if rollback succeeded, False if version not found.
    """
    snapshot_path = dataset_dir / ".versions" / f"v_{version_hash}.jsonl"
    if not snapshot_path.exists():
        return False

    dataset_file = dataset_dir / "dataset.jsonl"

    # Safety: snapshot current version before overwriting
    if dataset_file.exists():
        create_snapshot(dataset_dir, description="auto-snapshot before rollback")

    # Restore
    shutil.copy2(snapshot_path, dataset_file)
    return True


def _append_changelog(versions_dir: Path, version: DatasetVersion) -> None:
    """Append a version entry to the changelog."""
    changelog_path = versions_dir / "changelog.json"
    entries = []
    if changelog_path.exists():
        try:
            with open(changelog_path, encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append(version.as_dict())

    with open(changelog_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
