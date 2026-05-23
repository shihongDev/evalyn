"""Dataset-stats command: quick summary statistics over a dataset.

Useful before running a costly eval - see item count, input/output length
distribution, label coverage, and metadata-tag frequencies.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from typing import Any

from ...datasets import load_dataset

GROUP = "Dataset"
ESSENTIAL = {"dataset"}


def _to_text(value: Any) -> str:
    """Coerce any value to a string for length measurement."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _length_stats(lengths: list[int]) -> dict[str, float]:
    """Compute min/max/mean/median/p95 for a list of lengths."""
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0}
    sorted_lengths = sorted(lengths)
    p95_idx = max(0, int(len(sorted_lengths) * 0.95) - 1)
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p95": sorted_lengths[p95_idx],
    }


def compute_stats(items: list) -> dict[str, Any]:
    """Compute summary statistics over a list of DatasetItem objects."""
    input_lengths = [len(_to_text(item.input)) for item in items]
    output_lengths = [len(_to_text(item.output)) for item in items]
    labeled_count = sum(1 for item in items if item.human_label)

    metadata_keys: Counter = Counter()
    for item in items:
        if isinstance(item.metadata, dict):
            metadata_keys.update(item.metadata.keys())

    return {
        "item_count": len(items),
        "labeled_count": labeled_count,
        "labeled_pct": (labeled_count / len(items) * 100) if items else 0.0,
        "input_length": _length_stats(input_lengths),
        "output_length": _length_stats(output_lengths),
        "metadata_key_freq": dict(metadata_keys.most_common(10)),
    }


def _render_table(stats: dict[str, Any], dataset_path: str) -> str:
    lines = []
    lines.append(f"\nDataset stats: {dataset_path}")
    lines.append("=" * 70)
    lines.append(f"  Items:           {stats['item_count']}")
    lines.append(
        f"  Human-labeled:   {stats['labeled_count']} ({stats['labeled_pct']:.1f}%)"
    )
    lines.append("")
    lines.append("  Input length (chars):")
    il = stats["input_length"]
    lines.append(
        f"    min={il['min']}  max={il['max']}  mean={il['mean']:.0f}  median={il['median']:.0f}  p95={il['p95']}"
    )
    lines.append("  Output length (chars):")
    ol = stats["output_length"]
    lines.append(
        f"    min={ol['min']}  max={ol['max']}  mean={ol['mean']:.0f}  median={ol['median']:.0f}  p95={ol['p95']}"
    )
    if stats["metadata_key_freq"]:
        lines.append("")
        lines.append("  Top metadata keys (frequency):")
        for key, count in stats["metadata_key_freq"].items():
            lines.append(f"    {key:<24} {count:>5}")
    lines.append("")
    return "\n".join(lines)


def cmd_dataset_stats(args: argparse.Namespace) -> None:
    """Compute and render dataset statistics."""
    try:
        items = load_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(2)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise SystemExit(2)

    if not items:
        print(f"Dataset is empty: {args.dataset}")
        return

    stats = compute_stats(items)

    if args.format == "json":
        print(json.dumps({"path": args.dataset, **stats}, indent=2, default=float))
        return

    print(_render_table(stats, args.dataset))


def register_commands(subparsers) -> None:
    parser = subparsers.add_parser(
        "dataset-stats",
        help="Summary statistics over a dataset",
        description=(
            "Compute item count, input/output length distribution, label "
            "coverage, and metadata-tag frequencies for a dataset."
        ),
    )
    parser.add_argument("dataset", help="Path to dataset file or directory")
    parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.set_defaults(func=cmd_dataset_stats)
