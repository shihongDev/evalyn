"""Garbage collection for orphaned data files.

Finds and removes orphaned checkpoints, runs, and temp files
that are no longer referenced by active datasets or sessions.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OrphanedItem:
    """An orphaned file or directory identified for cleanup."""

    path: str
    item_type: str  # "checkpoint", "run", "temp"
    size_bytes: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "item_type": self.item_type,
            "size_bytes": self.size_bytes,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrphanedItem:
        return cls(
            path=data.get("path", ""),
            item_type=data.get("item_type", "temp"),
            size_bytes=data.get("size_bytes", 0),
            reason=data.get("reason", ""),
        )


@dataclass
class GCConfig:
    """Configuration for garbage collection."""

    data_dir: str = "data"
    temp_dir: str = ".evalyn"
    dry_run: bool = True
    max_age_days: float = 30.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "temp_dir": self.temp_dir,
            "dry_run": self.dry_run,
            "max_age_days": self.max_age_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCConfig:
        return cls(
            data_dir=data.get("data_dir", "data"),
            temp_dir=data.get("temp_dir", ".evalyn"),
            dry_run=data.get("dry_run", True),
            max_age_days=data.get("max_age_days", 30.0),
        )


@dataclass
class GCResult:
    """Result of a garbage collection run."""

    items_found: list[OrphanedItem] = field(default_factory=list)
    items_removed: int = 0
    bytes_freed: int = 0
    dry_run: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "items_found": [item.as_dict() for item in self.items_found],
            "items_removed": self.items_removed,
            "bytes_freed": self.bytes_freed,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCResult:
        return cls(
            items_found=[OrphanedItem.from_dict(item) for item in data.get("items_found", [])],
            items_removed=data.get("items_removed", 0),
            bytes_freed=data.get("bytes_freed", 0),
            dry_run=data.get("dry_run", True),
        )


def _get_size(path: str) -> int:
    """Get size in bytes for a file or directory."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        total = 0
        for entry in p.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total
    return 0


def find_orphaned_checkpoints(data_dir: str) -> list[OrphanedItem]:
    """Find checkpoint*.json files without a corresponding run directory.

    Scans data_dir recursively for checkpoint*.json files. A checkpoint is
    orphaned if no run_* directory exists in the same parent directory.
    """
    results: list[OrphanedItem] = []
    root = Path(data_dir)
    if not root.is_dir():
        return results

    for checkpoint_file in root.rglob("checkpoint*.json"):
        parent = checkpoint_file.parent
        has_run_dir = any(d.is_dir() and d.name.startswith("run_") for d in parent.iterdir())
        if not has_run_dir:
            results.append(
                OrphanedItem(
                    path=str(checkpoint_file),
                    item_type="checkpoint",
                    size_bytes=_get_size(str(checkpoint_file)),
                    reason="no corresponding run directory in parent",
                )
            )

    return results


def find_orphaned_runs(data_dir: str) -> list[OrphanedItem]:
    """Find run_* directories without a dataset.jsonl in their parent.

    A run directory is orphaned if its parent directory does not contain
    a dataset.jsonl file, meaning the dataset it belonged to is gone.
    """
    results: list[OrphanedItem] = []
    root = Path(data_dir)
    if not root.is_dir():
        return results

    for run_dir in root.rglob("run_*"):
        if not run_dir.is_dir():
            continue
        parent = run_dir.parent
        dataset_file = parent / "dataset.jsonl"
        if not dataset_file.exists():
            results.append(
                OrphanedItem(
                    path=str(run_dir),
                    item_type="run",
                    size_bytes=_get_size(str(run_dir)),
                    reason="no dataset.jsonl in parent directory",
                )
            )

    return results


def find_temp_files(temp_dir: str, max_age_days: float = 30.0) -> list[OrphanedItem]:
    """Find old temporary files in the .evalyn/ directory.

    Files older than max_age_days are considered orphaned.
    """
    results: list[OrphanedItem] = []
    root = Path(temp_dir)
    if not root.is_dir():
        return results

    now = time.time()
    max_age_seconds = max_age_days * 86400

    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        age = now - mtime
        if age > max_age_seconds:
            age_days = age / 86400
            results.append(
                OrphanedItem(
                    path=str(entry),
                    item_type="temp",
                    size_bytes=_get_size(str(entry)),
                    reason=f"temp file is {age_days:.1f} days old (max: {max_age_days})",
                )
            )

    return results


def remove_item(item: OrphanedItem) -> bool:
    """Delete a file or directory. Returns True if successful."""
    p = Path(item.path)
    try:
        if p.is_file():
            p.unlink()
            return True
        if p.is_dir():
            shutil.rmtree(str(p))
            return True
    except OSError:
        return False
    return False


def collect_garbage(config: GCConfig) -> GCResult:
    """Find all orphaned items and remove them if not dry_run.

    Combines results from find_orphaned_checkpoints, find_orphaned_runs,
    and find_temp_files. When dry_run is False, removes each item.
    """
    all_items: list[OrphanedItem] = []
    all_items.extend(find_orphaned_checkpoints(config.data_dir))
    all_items.extend(find_orphaned_runs(config.data_dir))
    all_items.extend(find_temp_files(config.temp_dir, config.max_age_days))

    result = GCResult(
        items_found=all_items,
        items_removed=0,
        bytes_freed=0,
        dry_run=config.dry_run,
    )

    if not config.dry_run:
        for item in all_items:
            if remove_item(item):
                result.items_removed += 1
                result.bytes_freed += item.size_bytes

    return result


def format_gc_report(result: GCResult) -> str:
    """Format a human-readable garbage collection report."""
    lines: list[str] = []
    mode = "DRY RUN" if result.dry_run else "CLEANUP"
    lines.append(f"Garbage Collection Report ({mode})")
    lines.append("=" * 40)

    if not result.items_found:
        lines.append("No orphaned items found.")
        return "\n".join(lines)

    total_bytes = sum(item.size_bytes for item in result.items_found)
    lines.append(f"Items found: {len(result.items_found)}")
    lines.append(f"Total size: {_format_bytes(total_bytes)}")
    lines.append("")

    # Group by type
    by_type: dict[str, list[OrphanedItem]] = {}
    for item in result.items_found:
        by_type.setdefault(item.item_type, []).append(item)

    for item_type, items in sorted(by_type.items()):
        type_bytes = sum(i.size_bytes for i in items)
        lines.append(f"  {item_type} ({len(items)} items, {_format_bytes(type_bytes)}):")
        for item in items:
            lines.append(f"    {item.path}")
            lines.append(f"      {item.reason}")
        lines.append("")

    if not result.dry_run:
        lines.append(f"Removed: {result.items_removed} items")
        lines.append(f"Freed: {_format_bytes(result.bytes_freed)}")
    else:
        lines.append(f"Would remove: {len(result.items_found)} items")
        lines.append(f"Would free: {_format_bytes(total_bytes)}")

    return "\n".join(lines)


def _format_bytes(size: int) -> str:
    """Format byte count as human-readable string."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"
