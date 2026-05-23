"""Dataset diff command: compare two datasets and show added / removed / modified items.

Thin CLI wrapper over `evalyn_sdk.datasets.merge.diff_datasets`. The library does
all the matching by item ID and equality work; this module is responsible for:

1. Resolving each input path - a dataset directory containing dataset.jsonl, or a
   direct dataset.jsonl / *.json / *.jsonl file.
2. Loading both datasets through the standard `load_dataset` helper.
3. Rendering the resulting `DatasetDiff` as either a human-readable table or a
   JSON object suitable for scripting.

Usage:
    evalyn dataset-diff <dataset_a> <dataset_b>
    evalyn dataset-diff <a> <b> --format json
    evalyn dataset-diff <a> <b> --show-items
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Dataset"

# Both positional paths are required, so there is nothing optional worth
# highlighting in the default dashboard form.
ESSENTIAL: set[str] = set()

import argparse
import json
from pathlib import Path

from ..utils.errors import fatal_error

# Cap on how many added/removed item IDs we surface when --show-items is set.
# Matches the cap used inside DatasetDiff.format_text so the CLI stays
# predictable regardless of dataset size.
_SHOW_ITEMS_LIMIT = 10


def _resolve_dataset_file(path_str: str) -> Path:
    """Resolve a CLI-provided dataset path to an actual jsonl/json file.

    Accepts either:
    - A path to a dataset.jsonl / *.jsonl / *.json file directly
    - A path to a dataset directory containing dataset.jsonl
    """
    p = Path(path_str)
    if not p.exists():
        fatal_error(
            f"Dataset path not found: {path_str}",
            "Pass either a dataset directory or a dataset.jsonl file",
        )
    if p.is_file():
        return p
    # Directory: prefer dataset.jsonl, fall back to a single *.jsonl, then *.json.
    candidate = p / "dataset.jsonl"
    if candidate.exists():
        return candidate
    jsonl_files = sorted(p.glob("*.jsonl"))
    if len(jsonl_files) == 1:
        return jsonl_files[0]
    json_files = sorted(p.glob("*.json"))
    if not jsonl_files and len(json_files) == 1:
        return json_files[0]
    fatal_error(
        f"Could not find a dataset file under: {path_str}",
        "Expected dataset.jsonl in the directory, or pass the file path directly",
    )


def _render_section(lines: list[str], heading: str, marker: str, items: list) -> None:
    """Append a section of item IDs (capped at _SHOW_ITEMS_LIMIT) to lines."""
    if not items:
        return
    lines.append("")
    lines.append(f"  {heading}:")
    for item in items[:_SHOW_ITEMS_LIMIT]:
        lines.append(f"    {marker} {item.id}")
    if len(items) > _SHOW_ITEMS_LIMIT:
        lines.append(f"    ... and {len(items) - _SHOW_ITEMS_LIMIT} more")


def _render_table(diff, *, label_a: str, label_b: str, show_items: bool) -> str:
    """Render a DatasetDiff as a human-readable table string."""
    lines = [
        "Dataset Diff",
        f"  A: {label_a}",
        f"  B: {label_b}",
        "",
        f"  Added:     {len(diff.added)}",
        f"  Removed:   {len(diff.removed)}",
        f"  Modified:  {len(diff.modified)}",
        f"  Unchanged: {diff.unchanged}",
    ]

    if show_items:
        _render_section(lines, "Added items", "+", diff.added)
        _render_section(lines, "Removed items", "-", diff.removed)
        _render_section(lines, "Modified items", "~", diff.modified)

    if not diff.has_changes:
        lines.append("")
        lines.append("  Datasets are identical.")
    return "\n".join(lines)


def _render_json(diff, *, label_a: str, label_b: str, show_items: bool) -> str:
    """Render a DatasetDiff as a JSON string."""
    payload: dict = {
        "a": label_a,
        "b": label_b,
        "summary": diff.summary,
        "has_changes": diff.has_changes,
    }
    if show_items:
        payload["added_ids"] = [item.id for item in diff.added[:_SHOW_ITEMS_LIMIT]]
        payload["removed_ids"] = [item.id for item in diff.removed[:_SHOW_ITEMS_LIMIT]]
        payload["modified_ids"] = [item.id for item in diff.modified[:_SHOW_ITEMS_LIMIT]]
        payload["truncated"] = {
            "added": max(0, len(diff.added) - _SHOW_ITEMS_LIMIT),
            "removed": max(0, len(diff.removed) - _SHOW_ITEMS_LIMIT),
            "modified": max(0, len(diff.modified) - _SHOW_ITEMS_LIMIT),
        }
    return json.dumps(payload, indent=2)


def cmd_dataset_diff(args: argparse.Namespace) -> None:
    """Compute and render the diff between two datasets."""
    from ...datasets import load_dataset
    from ...datasets.merge import diff_datasets

    file_a = _resolve_dataset_file(args.dataset_a)
    file_b = _resolve_dataset_file(args.dataset_b)

    items_a = load_dataset(file_a)
    items_b = load_dataset(file_b)

    diff = diff_datasets(items_a, items_b)

    output_format = getattr(args, "format", "table")
    show_items = bool(getattr(args, "show_items", False))

    if output_format == "json":
        print(
            _render_json(
                diff,
                label_a=str(file_a),
                label_b=str(file_b),
                show_items=show_items,
            )
        )
    else:
        print(
            _render_table(
                diff,
                label_a=str(file_a),
                label_b=str(file_b),
                show_items=show_items,
            )
        )


def register_commands(subparsers) -> None:
    """Register the dataset-diff command."""
    p = subparsers.add_parser(
        "dataset-diff",
        help="Compare two datasets and show added / removed / modified items",
    )
    p.add_argument(
        "dataset_a",
        help="First dataset (directory containing dataset.jsonl, or a .jsonl/.json file)",
    )
    p.add_argument(
        "dataset_b",
        help="Second dataset (directory containing dataset.jsonl, or a .jsonl/.json file)",
    )
    p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.add_argument(
        "--show-items",
        action="store_true",
        help="Also list the first added / removed / modified item IDs (max 10 each)",
    )
    p.set_defaults(func=cmd_dataset_diff)


__all__ = ["cmd_dataset_diff", "register_commands"]
