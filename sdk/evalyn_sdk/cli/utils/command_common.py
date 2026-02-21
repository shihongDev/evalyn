"""Shared CLI command helpers.

These helpers consolidate repeated argument-resolution flows used by
multiple command modules while keeping command-specific behavior intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .config import load_config, resolve_dataset_path
from .errors import fatal_error


def resolve_dataset_dir_and_file(
    dataset_arg: Optional[str],
    use_latest: bool,
    *,
    config: Optional[dict] = None,
) -> tuple[Path, Path]:
    """Resolve dataset input into (dataset_dir, dataset_file)."""
    cfg = config if config is not None else load_config()
    resolved_path = resolve_dataset_path(dataset_arg, use_latest, cfg)

    if not resolved_path:
        fatal_error("No dataset specified", "Use --dataset <path> or --latest")

    dataset_path = Path(resolved_path)
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        dataset_file = dataset_path / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = dataset_path / "dataset.json"
    else:
        dataset_dir = dataset_path.parent
        dataset_file = dataset_path

    if not dataset_file.exists():
        fatal_error(f"Dataset file not found: {dataset_file}")

    return dataset_dir, dataset_file


def try_resolve_dataset_dir_and_file(
    dataset_arg: Optional[str],
    use_latest: bool,
    *,
    config: Optional[dict] = None,
) -> Optional[tuple[Path, Path]]:
    """Resolve dataset input into (dataset_dir, dataset_file) without failing."""
    cfg = config if config is not None else load_config()
    resolved_path = resolve_dataset_path(dataset_arg, use_latest, cfg)
    if not resolved_path:
        return None

    dataset_path = Path(resolved_path)
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        dataset_file = dataset_path / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = dataset_path / "dataset.json"
    else:
        dataset_dir = dataset_path.parent
        dataset_file = dataset_path

    if not dataset_file.exists():
        return None

    return dataset_dir, dataset_file


def resolve_call_id(
    storage: Any,
    input_id: str,
    *,
    strict_prefix_match: bool = False,
) -> str:
    """Resolve a short call ID prefix to full ID when supported by storage."""
    if hasattr(storage, "resolve_call_id"):
        resolved = storage.resolve_call_id(input_id)
        if resolved:
            return resolved
        if strict_prefix_match:
            fatal_error(
                f"No call found matching '{input_id}'",
                "Use more characters for a unique match",
            )
    return input_id


def try_resolve_call_id(storage: Any, input_id: str) -> Optional[str]:
    """Resolve a call ID and return None when prefix resolution fails."""
    if hasattr(storage, "resolve_call_id"):
        return storage.resolve_call_id(input_id)
    return input_id


def resolve_call_id_or_last(
    storage: Any,
    *,
    input_id: Optional[str],
    use_last: bool,
) -> str:
    """Resolve call ID from --id/--last semantics used by trace commands."""
    if use_last:
        calls = storage.list_calls(limit=1)
        if not calls:
            fatal_error("No calls found")
        return calls[0].id

    if input_id:
        return resolve_call_id(storage, input_id, strict_prefix_match=True)

    fatal_error("Must specify --id or --last")
