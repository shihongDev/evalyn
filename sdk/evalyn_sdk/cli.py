from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Callable, List, Optional

from .annotations import import_annotations
from .calibration import CalibrationEngine
from .datasets import load_dataset
from .decorators import get_default_tracer
from .metrics.objective import register_builtin_metrics
from .metrics.registry import MetricRegistry
from .runner import EvalRunner
from .suggester import HeuristicSuggester, LLMSuggester, LLMRegistrySelector
from .metrics.registry import MetricRegistry
from .tracing import EvalTracer


def _load_callable(target: str) -> Callable[..., Any]:
    """Load a function from a string like module:function_name."""
    if ":" not in target:
        raise ValueError("Target must be in the form 'module:function'")
    module_name, func_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _extract_code_meta(tracer: EvalTracer, fn: Callable[..., Any]) -> Optional[dict]:
    """
    Try to reuse cached code metadata from the tracer; fall back to inspect if not present.
    """
    meta = getattr(tracer, "_function_meta_cache", {}).get(id(fn))  # type: ignore[attr-defined]
    if meta:
        return meta
    try:
        import inspect
        import hashlib

        source = inspect.getsource(fn)
        return {
            "module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None),
            "doc": inspect.getdoc(fn),
            "signature": str(inspect.signature(fn)),
            "source": source,
            "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "file_path": inspect.getsourcefile(fn),
        }
    except Exception:
        return None


def cmd_list_calls(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    calls = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []
    for call in calls:
        status = "error" if call.error else "ok"
        print(f"{call.id} | {call.function_name} | {status} | {call.duration_ms:.2f} ms")


def cmd_show_call(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    call = tracer.storage.get_call(args.id)
    if not call:
        print(f"No call found with id={args.id}")
        return
    status = "error" if call.error else "ok"
    print(f"\n=== Call {call.id} ===")
    print(f"Function : {call.function_name}")
    print(f"Status   : {status}")
    print(f"Session  : {call.session_id}")
    print(f"Started  : {call.started_at}")
    print(f"Duration : {call.duration_ms:.2f} ms")

    print("\nInputs:")
    print(json.dumps(call.inputs, indent=2) or "<empty>")

    if call.error:
        print("\nError:")
        print(call.error)
    else:
        print("\nOutput:")
        print(str(call.output) or "<empty>")

    if call.metadata:
        print("\nMetadata (incl. code info):")
        print(json.dumps(call.metadata, indent=2))

    if call.trace:
        print("\nEvents (Evalyn + instrumented SDK/tool events):")
        for ev in call.trace:
            print(f" - {ev.timestamp} | {ev.kind} | {ev.detail}")

    print("\nOTel:")
    print(" Spans are emitted alongside this call (exporter-configured). See your OTel backend/console for span view.")


def cmd_run_dataset(args: argparse.Namespace) -> None:
    target_fn = _load_callable(args.target)
    dataset = load_dataset(args.dataset)

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


def cmd_suggest_metrics(args: argparse.Namespace) -> None:
    target_fn = _load_callable(args.target)
    tracer = get_default_tracer()
    traces = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []

    if args.llm_caller:
        caller = _load_callable(args.llm_caller)
        suggester = LLMSuggester(caller=caller)
    else:
        suggester = HeuristicSuggester()

    specs = suggester.suggest(target_fn, traces)
    for spec in specs:
        print(f"- {spec.name} [{spec.type}] :: {spec.description}")


def cmd_list_runs(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    runs = tracer.storage.list_eval_runs(limit=args.limit)
    for run in runs:
        print(f"{run.id} | dataset={run.dataset_name} | metrics={len(run.metrics)} | results={len(run.metric_results)}")


def cmd_show_run(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    run = tracer.storage.get_eval_run(args.id)
    if not run:
        print(f"No eval run found with id={args.id}")
        return
    print(f"\n=== Eval Run {run.id} ===")
    print(f"Dataset: {run.dataset_name}")
    print("Metrics summary:")
    for mid, stats in run.summary.get("metrics", {}).items():
        print(
            f" - {mid:<18} count={stats['count']:<3} "
            f"avg_score={stats['avg_score']!s:<10} pass_rate={stats['pass_rate']!s:<10}"
        )
    if run.summary.get("failed_items"):
        print(f"Failed items: {run.summary['failed_items']}")
    print("\nMetric results:")
    for res in run.metric_results:
        print(
            f" [{res.metric_id}] item={res.item_id} call={res.call_id} "
            f"score={res.score} passed={res.passed} details={res.details}"
        )


def cmd_import_annotations(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    anns = import_annotations(args.path)
    tracer.storage.store_annotations(anns)
    print(f"Imported {len(anns)} annotations into storage.")


def cmd_calibrate(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return

    run = tracer.storage.get_eval_run(args.run_id) if args.run_id else None
    if run is None:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None
    if run is None:
        print("No eval runs available.")
        return

    metric_results = [r for r in run.metric_results if r.metric_id == args.metric_id]
    if not metric_results:
        print(f"No metric results found for metric_id={args.metric_id} in run {run.id}")
        return

    anns = import_annotations(args.annotations)
    engine = CalibrationEngine(judge_name=args.metric_id, current_threshold=args.threshold)
    record = engine.calibrate(metric_results, anns)

    print(f"Calibration for metric '{args.metric_id}' on run {run.id}")
    print(f"- disagreement_rate: {record.adjustments['disagreement_rate']:.3f}")
    print(f"- suggested_threshold: {record.adjustments['suggested_threshold']:.3f}")
    print(f"- current_threshold: {record.adjustments['current_threshold']:.3f}")


def cmd_select_metrics(args: argparse.Namespace) -> None:
    if not args.llm_caller:
        print("Please provide --llm-caller to enable LLM-based selection.")
        return

    target_fn = _load_callable(args.target)
    caller = _load_callable(args.llm_caller)

    registry = MetricRegistry()
    register_builtin_metrics(registry)

    tracer = get_default_tracer()
    traces = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []

    selector = LLMRegistrySelector(caller)
    selected = selector.select(target_fn, registry, traces=traces, code_meta=_extract_code_meta(tracer, target_fn))

    print("Selected metrics:")
    for spec in selected:
        print(f"- {spec.id}: {spec.name} [{spec.type}]")


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

    suggest_parser = subparsers.add_parser("suggest-metrics", help="Suggest metrics for a target function")
    suggest_parser.add_argument("--target", required=True, help="Callable to analyze in the form module:function")
    suggest_parser.add_argument("--limit", type=int, default=5, help="How many recent traces to include as examples")
    suggest_parser.add_argument(
        "--llm-caller",
        help="Optional callable path that accepts a prompt string and returns a list of metric dicts",
    )
    suggest_parser.set_defaults(func=cmd_suggest_metrics)

    runs_parser = subparsers.add_parser("list-runs", help="List stored eval runs")
    runs_parser.add_argument("--limit", type=int, default=10)
    runs_parser.set_defaults(func=cmd_list_runs)

    show_call = subparsers.add_parser("show-call", help="Show details for a traced call")
    show_call.add_argument("--id", required=True, help="Call id to display")
    show_call.set_defaults(func=cmd_show_call)

    show_run = subparsers.add_parser("show-run", help="Show details for an eval run")
    show_run.add_argument("--id", required=True, help="Eval run id to display")
    show_run.set_defaults(func=cmd_show_run)

    import_ann = subparsers.add_parser("import-annotations", help="Import annotations from a JSONL file")
    import_ann.add_argument("--path", required=True, help="Path to annotations JSONL")
    import_ann.set_defaults(func=cmd_import_annotations)

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate a subjective metric using human annotations")
    calibrate_parser.add_argument("--metric-id", required=True, help="Metric ID to calibrate (usually the judge metric id)")
    calibrate_parser.add_argument("--annotations", required=True, help="Path to annotations JSONL (target_id must match call_id)")
    calibrate_parser.add_argument("--run-id", help="Eval run id to calibrate; defaults to latest run")
    calibrate_parser.add_argument("--threshold", type=float, default=0.5, help="Current threshold for pass/fail")
    calibrate_parser.set_defaults(func=cmd_calibrate)

    select_parser = subparsers.add_parser("select-metrics", help="LLM-guided selection from metric registry")
    select_parser.add_argument("--target", required=True, help="Callable to analyze in the form module:function")
    select_parser.add_argument("--llm-caller", required=True, help="Callable that accepts a prompt and returns metric ids or dicts")
    select_parser.add_argument("--limit", type=int, default=5, help="Recent traces to include as examples")
    select_parser.set_defaults(func=cmd_select_metrics)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
