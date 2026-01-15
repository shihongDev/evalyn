"""Dataset commands: build-dataset.

This module provides CLI commands for building datasets from stored traces.
Datasets are the foundation for evaluation - they contain input/output pairs
from real agent executions that can be evaluated against metrics.

Commands:
- build-dataset: Build a dataset.jsonl file from stored traces

Key concepts:
- Traces are grouped by project (--project) and optionally by version (--version)
- Datasets can include only production or simulation traces
- The output is a JSONL file with one item per line, plus meta.json with schema info

Typical workflow:
1. First run 'evalyn show-projects' to see available projects
2. Then run 'evalyn build-dataset --project <name>' to create a dataset
3. The dataset is saved to data/<project>-<version>-<timestamp>/dataset.jsonl
4. Next step: 'evalyn suggest-metrics --dataset <path>' to suggest metrics
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Optional

from ...datasets import build_dataset_from_storage, save_dataset_with_meta
from ...decorators import get_default_tracer
from ..utils.hints import print_hint


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Build dataset from stored traces."""
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return

    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    items = build_dataset_from_storage(
        tracer.storage,
        function_name=None,  # prefer project-based grouping
        project_id=args.project,
        project_name=args.project,
        version=args.version,
        simulation_only=getattr(args, "simulation", False),
        production_only=getattr(args, "production", False),
        since=_parse_dt(args.since),
        until=_parse_dt(args.until),
        limit=args.limit,
        success_only=not args.include_errors,
        include_metadata=True,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    proj = args.project or "all"
    ver = args.version or "v0"
    dataset_name = f"{proj}-{ver}-{ts}"

    dataset_dir = args.output or os.path.join("data", dataset_name)
    dataset_file = "dataset.jsonl"
    if args.output and args.output.endswith(".jsonl"):
        dataset_dir = os.path.dirname(args.output) or "."
        dataset_file = os.path.basename(args.output)

    functions = sorted(
        {
            item.metadata.get("function")
            for item in items
            if isinstance(item.metadata, dict) and item.metadata.get("function")
        }
    )
    function_name = functions[0] if len(functions) == 1 else None

    input_keys = set()
    meta_keys = set()
    for item in items:
        if isinstance(item.inputs, dict):
            input_keys.update(item.inputs.keys())
        if isinstance(item.metadata, dict):
            meta_keys.update(item.metadata.keys())

    meta = {
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": args.project,
        "version": args.version,
        "function": function_name,
        "filters": {
            "since": args.since,
            "until": args.until,
            "limit": args.limit,
            "include_errors": bool(args.include_errors),
        },
        "counts": {"items": len(items)},
        "schema": {
            "inputs_keys": sorted(input_keys),
            "metadata_keys": sorted(meta_keys),
        },
    }

    dataset_path = save_dataset_with_meta(
        items, dataset_dir, meta, dataset_filename=dataset_file
    )
    print(f"Wrote {len(items)} items to {dataset_path}")
    print_hint(
        f"To suggest metrics, run: evalyn suggest-metrics --dataset {dataset_dir} --mode basic",
        quiet=getattr(args, "quiet", False),
    )


def register_commands(subparsers) -> None:
    """Register dataset commands."""
    # build-dataset
    p = subparsers.add_parser("build-dataset", help="Build dataset from stored traces")
    p.add_argument(
        "--output",
        help="Path to write dataset JSONL (default: data/<project>-<version>-<timestamp>.jsonl)",
    )
    p.add_argument(
        "--project",
        help="Filter by metadata.project_id or project_name (recommended grouping)",
    )
    p.add_argument("--version", help="Filter by metadata.version")
    p.add_argument(
        "--simulation", action="store_true", help="Include only simulation traces"
    )
    p.add_argument(
        "--production", action="store_true", help="Include only production traces"
    )
    p.add_argument("--since", help="ISO timestamp lower bound for started_at")
    p.add_argument("--until", help="ISO timestamp upper bound for started_at")
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max number of items to include (after filtering)",
    )
    p.add_argument(
        "--include-errors",
        action="store_true",
        help="Include errored calls (default: skip)",
    )
    p.set_defaults(func=cmd_build_dataset)


__all__ = ["cmd_build_dataset", "register_commands"]
