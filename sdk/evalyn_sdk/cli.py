from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, List

from .decorators import get_default_tracer
from .metrics.objective import exact_match_metric, latency_metric, register_builtin_metrics
from .metrics.registry import MetricRegistry
from .models import DatasetItem
from .runner import EvalRunner


def _load_callable(target: str) -> Callable[..., Any]:
    """Load a function from a string like module:function_name."""
    if ":" not in target:
        raise ValueError("Target must be in the form 'module:function'")
    module_name, func_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _load_dataset(path: str) -> List[DatasetItem]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    rows: List[Any] = []
    if text.startswith("["):
        rows = json.loads(text)
    else:
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [DatasetItem.from_payload(row) for row in rows]


def cmd_list_calls(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    calls = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []
    for call in calls:
        status = "error" if call.error else "ok"
        print(f"{call.id} | {call.function_name} | {status} | {call.duration_ms:.2f} ms")


def cmd_run_dataset(args: argparse.Namespace) -> None:
    target_fn = _load_callable(args.target)
    dataset = _load_dataset(args.dataset)

    registry = MetricRegistry()
    register_builtin_metrics(registry)

    runner = EvalRunner(
        target_fn=target_fn,
        metrics=registry.list(),
        dataset_name=args.dataset_name,
    )
    run = runner.run_dataset(dataset)

    print(f"Eval run {run.id} over dataset '{run.dataset_name}'")
    for metric_id, stats in run.summary.get("metrics", {}).items():
        print(f"- {metric_id}: count={stats['count']}, avg_score={stats['avg_score']}, pass_rate={stats['pass_rate']}")
    if run.summary.get("failed_items"):
        print(f"- failed items: {run.summary['failed_items']}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evalyn SDK CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-calls", help="List recent traced calls")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of calls to display")
    list_parser.set_defaults(func=cmd_list_calls)

    run_parser = subparsers.add_parser("run-dataset", help="Execute a dataset against a target function")
    run_parser.add_argument("--target", required=True, help="Callable to run in the form module:function")
    run_parser.add_argument("--dataset", required=True, help="Path to JSON/JSONL dataset")
    run_parser.add_argument("--dataset-name", default="dataset", help="Name for the eval run")
    run_parser.set_defaults(func=cmd_run_dataset)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
