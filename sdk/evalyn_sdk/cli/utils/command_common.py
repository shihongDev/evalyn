"""Shared CLI command helpers.

These helpers consolidate repeated argument-resolution flows used by
multiple command modules while keeping command-specific behavior intact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .config import load_config, resolve_dataset_path
from .errors import fatal_error

if TYPE_CHECKING:
    from ...models import EvalRun
    from ...storage.base import StorageBackend


@dataclass
class LoadedRun:
    """Result of loading an eval run, with metadata about the source."""
    run: EvalRun
    run_file_path: Optional[Path] = None


def load_eval_run_for_command(
    *,
    run_id: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    metric_id: Optional[str] = None,
    fallback_to_storage: bool = False,
    storage: Optional[StorageBackend] = None,
    error_message: str = "No eval runs found",
    error_hint: str = "Run 'evalyn run-eval' first",
) -> LoadedRun:
    """Load an EvalRun from explicit ID, dataset directory, or storage fallback.

    Consolidates the run-loading pattern used across CLI commands:
    1. If run_id given, load from storage
    2. If dataset_path given, find latest run in eval_runs/ dir
    3. If fallback_to_storage, try latest run in storage
    4. fatal_error if nothing found

    When *storage* is None, falls back to creating a default SQLiteStorage.
    """
    from ...models import EvalRun
    from ...decorators import get_default_tracer
    from ...analysis.core import find_eval_runs

    def _get_storage():
        if storage is not None:
            return storage
        tracer = get_default_tracer()
        return tracer.storage

    run = None
    run_file_path = None

    if run_id:
        s = _get_storage()
        # Resolve short-ID prefix via SQL (avoids loading all runs into memory)
        if hasattr(s, "resolve_eval_run_id"):
            resolved = s.resolve_eval_run_id(run_id)
            if resolved:
                run = s.get_eval_run(resolved)
            else:
                fatal_error(
                    f"No eval run found with ID '{run_id}'",
                    "Use more characters for a unique match, or check 'evalyn list-runs'",
                )
        else:
            run = s.get_eval_run(run_id)
        if not run:
            fatal_error(f"No eval run found with ID '{run_id}'")
        return LoadedRun(run=run)

    if dataset_path:
        run_files = find_eval_runs(dataset_path)
        for run_file in run_files:
            with open(run_file, encoding="utf-8") as f:
                candidate = EvalRun.from_dict(json.load(f))
            if not metric_id or any(
                r.metric_id == metric_id for r in candidate.metric_results
            ):
                run = candidate
                run_file_path = run_file
                break

    if run is None and fallback_to_storage:
        s = _get_storage()
        runs = s.list_eval_runs(limit=1)
        if runs:
            run = runs[0]

    if run is None:
        fatal_error(error_message, error_hint)

    return LoadedRun(run=run, run_file_path=run_file_path)


def _normalize_dataset_path(path: Path) -> Optional[tuple[Path, Path]]:
    """Normalize a path to (dataset_dir, dataset_file).

    Handles both directory paths (looks for dataset.jsonl/dataset.json inside)
    and direct file paths. Returns None if the resolved file doesn't exist.
    """
    if path.is_dir():
        dataset_dir = path
        dataset_file = path / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = path / "dataset.json"
    else:
        dataset_dir = path.parent
        dataset_file = path

    if not dataset_file.exists():
        return None
    return dataset_dir, dataset_file


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

    result = _normalize_dataset_path(Path(resolved_path))
    if result is None:
        fatal_error(f"Dataset file not found in: {resolved_path}")
    return result


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

    return _normalize_dataset_path(Path(resolved_path))


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
            # Differentiate ambiguous from not-found
            if hasattr(storage, "count_call_prefix_matches"):
                count = storage.count_call_prefix_matches(input_id)
                if count > 1:
                    fatal_error(
                        f"Ambiguous ID '{input_id}' matches {count} calls",
                        "Use more characters for a unique match",
                    )
            fatal_error(
                f"No call found matching '{input_id}'",
                "Check the ID and try again",
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
        calls = storage.list_calls(limit=1, lightweight=True)
        if not calls:
            fatal_error("No calls found")
        return calls[0].id

    if input_id:
        return resolve_call_id(storage, input_id, strict_prefix_match=True)

    fatal_error("Must specify --id or --last")
