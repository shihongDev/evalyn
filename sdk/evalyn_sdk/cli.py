from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from typing import Any, Callable, List, Optional

from .annotations import import_annotations
from .calibration import CalibrationEngine
from .datasets import load_dataset
from .decorators import get_default_tracer
from .metrics.objective import register_builtin_metrics
from .metrics.registry import MetricRegistry
from .metrics.templates import OBJECTIVE_TEMPLATES, SUBJECTIVE_TEMPLATES
from .runner import EvalRunner
from .suggester import HeuristicSuggester, LLMSuggester, LLMRegistrySelector, TemplateSelector, render_selection_prompt_with_templates
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
    if not calls:
        print("No calls found.")
        return
    headers = ["id", "function", "status", "start", "end", "duration_ms"]
    print(" | ".join(headers))
    print("-" * 120)
    for call in calls:
        status = "ERROR" if call.error else "OK"
        row = [
            call.id,
            call.function_name,
            status,
            str(call.started_at),
            str(call.ended_at),
            f"{call.duration_ms:.2f}",
        ]
        print(" | ".join(row))


def cmd_show_call(args: argparse.Namespace) -> None:
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    call = tracer.storage.get_call(args.id)
    if not call:
        print(f"No call found with id={args.id}")
        return
    status = "ERROR" if call.error else "OK"

    def _format_value(value, max_len=300):
        if isinstance(value, str):
            return value if len(value) <= max_len else value[:max_len] + "..."
        try:
            text = json.dumps(value, indent=2)
            return text if len(text) <= max_len else text[:max_len] + "..."
        except Exception:
            text = str(value)
            return text if len(text) <= max_len else text[:max_len] + "..."

    def _detect_turns(inputs) -> tuple[str, int]:
        if isinstance(inputs, dict):
            kwargs = inputs.get("kwargs", {})
            for key in ("messages", "history", "conversation", "turns"):
                val = kwargs.get(key)
                if isinstance(val, list):
                    return ("multi" if len(val) > 1 else "single"), len(val)
            for arg in inputs.get("args", []):
                if isinstance(arg, list):
                    return ("multi" if len(arg) > 1 else "single"), len(arg)
        return "single", 1

    def _count_events(kinds: list[str]) -> int:
        return sum(1 for ev in call.trace if any(k in ev.kind.lower() for k in kinds))

    turn_label, turns = _detect_turns(call.inputs)
    llm_calls = _count_events(["gemini.request", "openai.request", "llm.request"])
    tool_events = _count_events(["tool"])

    print("\n================ Call Details ================")
    print(f"id       : {call.id}")
    print(f"function : {call.function_name}")
    print(f"status   : {status}")
    print(f"session  : {call.session_id}")
    print(f"started  : {call.started_at}")
    print(f"ended    : {call.ended_at}")
    print(f"duration : {call.duration_ms:.2f} ms")
    print(f"turns    : {turn_label} ({turns})")
    print(f"llm_calls: {llm_calls} | tool_events: {tool_events}")

    print("\nInputs:")
    args = call.inputs.get("args", [])
    kwargs = call.inputs.get("kwargs", {})
    if args:
        print("  args:")
        for idx, arg in enumerate(args):
            print(f"    - [{idx}] {_format_value(arg)}")
    if kwargs:
        print("  kwargs:")
        for key, value in kwargs.items():
            print(f"    - {key}: {_format_value(value)}")
    if not args and not kwargs:
        print("  <empty>")

    if call.error:
        print("\nError:")
        print(call.error)
    else:
        print("\nOutput:")
        output_text = str(call.output or "")
        print(f"  type   : {type(call.output).__name__}")
        print(f"  length : {len(output_text)} chars")
        print(f"  preview: {_format_value(output_text, max_len=1000)}")

    if call.metadata:
        def _print_metadata(meta: dict) -> None:
            print("\nMetadata:")
            for key, value in meta.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_val in value.items():
                        if sub_key == "source":
                            src = sub_val or ""
                            src_preview = src if len(src) <= 1200 else src[:1200] + "..."
                            print("    source:")
                            for line in src_preview.splitlines()[:40]:
                                print(f"      {line}")
                        else:
                            print(f"    - {sub_key}: {_format_value(sub_val, max_len=400)}")
                else:
                    print(f"  - {key}: {_format_value(value, max_len=400)}")

        _print_metadata(call.metadata)

    def _normalize_span_time(raw):
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            value = float(raw)
            if value > 1e12:
                return value / 1e9
            if value > 1e10:
                return value / 1e9
            if value > 1e6:
                return value / 1e3
            return value
        if isinstance(raw, str):
            raw = raw.strip()
            try:
                if raw.isdigit():
                    return _normalize_span_time(int(raw))
                return _normalize_span_time(float(raw))
            except ValueError:
                pass
            try:
                import datetime as _dt

                return _dt.datetime.fromisoformat(raw).timestamp()
            except Exception:
                return None
        return None

    def _span_duration_ms(span):
        start_ts = _normalize_span_time(span.get("start_time"))
        end_ts = _normalize_span_time(span.get("end_time"))
        if start_ts is None or end_ts is None:
            return None
        return max(0.0, (end_ts - start_ts) * 1000)

    def _span_status(span):
        status = span.get("status")
        if status is None:
            return "UNSET"
        text = str(status).upper()
        if "ERROR" in text:
            return "ERROR"
        if "OK" in text:
            return "OK"
        return text

    def _span_attr_summary(attrs, max_items=3):
        if not isinstance(attrs, dict) or not attrs:
            return ""
        preferred = [
            "model",
            "llm.model",
            "tool",
            "tool.name",
            "http.method",
            "http.url",
            "rpc.system",
        ]
        parts = []
        seen = set()

        def _add(key, value):
            if key in seen:
                return
            seen.add(key)
            if value is None:
                return
            text = str(value)
            text = text if len(text) <= 60 else text[:60] + "..."
            parts.append(f"{key}={text}")

        for key in preferred:
            if key in attrs:
                _add(key, attrs.get(key))
                if len(parts) >= max_items:
                    return " ".join(parts)

        for key in sorted(attrs.keys()):
            if key.startswith("evalyn."):
                continue
            _add(key, attrs.get(key))
            if len(parts) >= max_items:
                break
        return " ".join(parts)

    def _print_span_tree(spans, call_start_ts):
        by_id = {s.get("span_id"): s for s in spans if s.get("span_id")}
        children = {span_id: [] for span_id in by_id}
        for span in spans:
            span_id = span.get("span_id")
            parent_id = span.get("parent_span_id")
            if parent_id in by_id and span_id in by_id:
                children[parent_id].append(span_id)

        def _sort_key(span_id):
            span = by_id.get(span_id, {})
            return _normalize_span_time(span.get("start_time")) or 0.0

        for parent_id in children:
            children[parent_id].sort(key=_sort_key)

        roots = [sid for sid, span in by_id.items() if span.get("parent_span_id") not in by_id]
        roots.sort(key=_sort_key)

        def _render(span_id, prefix, is_last):
            span = by_id[span_id]
            dur_ms = _span_duration_ms(span)
            status = _span_status(span)
            attr_summary = _span_attr_summary(span.get("attributes", {}))
            start_ts = _normalize_span_time(span.get("start_time"))
            rel_ms = None
            if start_ts is not None and call_start_ts is not None:
                rel_ms = (start_ts - call_start_ts) * 1000
            dur_text = f"{dur_ms:.1f}ms" if dur_ms is not None else "n/a"
            rel_text = f"+{rel_ms:.1f}ms" if rel_ms is not None else "n/a"
            branch = "`- " if is_last else "|- "
            line = f"{prefix}{branch}{span.get('name')} ({status}, {dur_text}, {rel_text})"
            if attr_summary:
                line += f" {attr_summary}"
            print(line)
            next_prefix = prefix + ("   " if is_last else "|  ")
            child_ids = children.get(span_id, [])
            for idx, child_id in enumerate(child_ids):
                _render(child_id, next_prefix, idx == len(child_ids) - 1)

        for idx, root_id in enumerate(roots):
            _render(root_id, "", idx == len(roots) - 1)

    spans = tracer.storage.list_spans(call.id) if tracer.storage else []
    if spans:
        print("\nSpan tree (OTel):")
        call_start_ts = call.started_at.timestamp() if call.started_at else None
        _print_span_tree(spans, call_start_ts)

    if call.trace:
        def _format_time(ev):
            try:
                delta = (ev.timestamp - call.started_at).total_seconds()
                return f"+{delta:0.3f}s"
            except Exception:
                return str(ev.timestamp)

        def _truncate(text, max_len=120):
            if text is None:
                return ""
            text = str(text)
            return text if len(text) <= max_len else text[:max_len] + "..."

        def _format_inline(value, max_len=100):
            try:
                if isinstance(value, (dict, list)):
                    text = json.dumps(value, separators=(",", ":"))
                else:
                    text = str(value)
            except Exception:
                text = str(value)
            return text if len(text) <= max_len else text[:max_len] + "..."

        def _summarize_detail(detail, max_items=4):
            if not detail:
                return ""
            parts = []
            if isinstance(detail, dict):
                model = detail.get("model")
                if model:
                    parts.append(f"model={_truncate(model, 40)}")
                tool = detail.get("tool") or detail.get("name")
                if tool:
                    parts.append(f"tool={_truncate(tool, 40)}")
                if "config" in detail and len(parts) < max_items:
                    cfg = detail.get("config") or {}
                    tools = []
                    raw_tools = cfg.get("tools")
                    if isinstance(raw_tools, list):
                        for t in raw_tools:
                            if isinstance(t, dict):
                                tools.extend(list(t.keys()))
                            else:
                                tools.append(str(t))
                    if tools:
                        parts.append(f"tools={','.join(tools)}")
                    else:
                        parts.append(f"config_keys={list(cfg.keys())}")
                for key in ("status", "status_code", "elapsed_ms", "duration_ms", "count", "length"):
                    if key in detail and len(parts) < max_items:
                        parts.append(f"{key}={_format_inline(detail.get(key), 40)}")
                for key in ("error", "url"):
                    if key in detail and len(parts) < max_items:
                        parts.append(f"{key}={_truncate(detail.get(key), 60)}")
                for key in ("contents", "messages", "prompt", "prompt_excerpt"):
                    if key in detail and len(parts) < max_items:
                        parts.append(f"{key}={_truncate(detail.get(key), 80)}")
            if not parts:
                parts.append(_truncate(_format_inline(detail, 120), 120))
            return " ".join(parts[:max_items])

        print("\nEvents summary:")
        total = len(call.trace)
        reqs = sum(1 for ev in call.trace if ev.kind.lower().endswith(".request"))
        resps = sum(1 for ev in call.trace if ev.kind.lower().endswith(".response"))
        tool_cnt = sum(1 for ev in call.trace if "tool" in ev.kind.lower())
        print(f"  total={total} | requests={reqs} | responses={resps} | tool_events={tool_cnt}")
        kind_counts = {}
        for ev in call.trace:
            kind_counts[ev.kind] = kind_counts.get(ev.kind, 0) + 1
        for kind, count in sorted(kind_counts.items()):
            print(f"  - {kind}: {count}")

        print("\nEvents timeline:")
        header = ["idx", "t+ms", "delta_ms", "kind", "summary"]
        print(" | ".join(header))
        print("-" * 140)
        prev_ts = None
        for idx, ev in enumerate(call.trace, start=1):
            elapsed_ms = (ev.timestamp - call.started_at).total_seconds() * 1000 if call.started_at else 0.0
            delta_ms = (
                (ev.timestamp - prev_ts).total_seconds() * 1000 if prev_ts else 0.0
            )
            summary = _summarize_detail(ev.detail or {})
            print(f"{idx} | {elapsed_ms:7.1f} | {delta_ms:7.1f} | {ev.kind} | {summary}")
            prev_ts = ev.timestamp

    if not spans:
        print("\nSpan tree (OTel):")
        print("  <no spans found>")
        print("  Tip: set EVALYN_OTEL_EXPORTER=sqlite and re-run to capture span data locally.")

    print("\nOTel:")
    print(" Spans are emitted alongside this call (exporter-configured).")
    print("=============================================\n")


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
    if not runs:
        print("No eval runs found.")
        return
    headers = ["id", "dataset", "created_at", "metrics", "results"]
    print(" | ".join(headers))
    print("-" * 120)
    for run in runs:
        row = [
            run.id,
            run.dataset_name,
            str(run.created_at),
            str(len(run.metrics)),
            str(len(run.metric_results)),
        ]
        print(" | ".join(row))


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
    target_fn = _load_callable(args.target)
    tracer = get_default_tracer()
    traces = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []

    if args.llm_caller:
        caller = _load_callable(args.llm_caller)
        selector = TemplateSelector(caller, OBJECTIVE_TEMPLATES + SUBJECTIVE_TEMPLATES)
        selected = selector.select(target_fn, traces=traces, code_meta=_extract_code_meta(tracer, target_fn))
    else:
        # fallback to heuristic/built-in registry selection
        registry = MetricRegistry()
        register_builtin_metrics(registry)
        selector = LLMRegistrySelector(lambda prompt: [])
        selected = registry.list()

    print("Selected metrics:")
    for spec in selected:
        print(f"- {spec.id}: [{spec.type}] config={getattr(spec, 'config', {})}")


def cmd_list_metrics(args: argparse.Namespace) -> None:
    def _compact(value, max_len: int = 60) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _compact_text(value: Any, max_len: int = 55) -> str:
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _config_summary(cfg: Any) -> str:
        if not cfg:
            return "{}"
        if not isinstance(cfg, dict):
            return _compact(cfg, 55)
        parts: list[str] = []
        for key in sorted(cfg.keys()):
            val = cfg.get(key)
            if key == "rubric" and isinstance(val, list):
                parts.append(f"rubric[{len(val)}]")
                continue
            if key == "policy" and isinstance(val, str):
                parts.append(f"policy[{len(val)}]")
                continue
            if key == "schema" and isinstance(val, dict):
                parts.append(f"schema[{len(val)}]")
                continue
            if isinstance(val, (str, int, float, bool)) or val is None:
                parts.append(f"{key}={_compact_text(val, 18)}")
            else:
                parts.append(f"{key}=…")
        text = ", ".join(parts)
        return text if text else "{}"

    def _print_table(title: str, templates: list[dict]) -> None:
        print(title)
        headers = ["id", "category", "inputs", "config", "desc"]
        rows = []
        for tpl in templates:
            inputs = tpl.get("inputs") or tpl.get("signals") or []
            rows.append(
                [
                    tpl.get("id", ""),
                    tpl.get("category", ""),
                    _compact(inputs, 45),
                    _config_summary(tpl.get("config", {})),
                    _compact_text(tpl.get("description", ""), 55),
                ]
            )

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    _print_table("\nObjective metrics:", OBJECTIVE_TEMPLATES)
    _print_table("\nSubjective metrics:", SUBJECTIVE_TEMPLATES)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=None, add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _print_ascii_help():
        art = r"""
   _______      __      ___   ___  _   _
  |  ____ \     \ \    / / \ / / \| \ | |
  | |____| |     \ \  / /|  V /|  \  \| |
  |  _____/       \ \/ / | | | | |\   / |
  | |             / /\ \ | | | | | \  | |
  |_|            /_/  \_\|_| |_|_|  \_|_|
        """
        print("================================================================================")
        print(art)
        print("Evalyn CLI - Shihong Liu (https://github.com/shihongDev)")
        print("================================================================================")
        parser.print_help()

    help_parser = subparsers.add_parser("help", help="Show available commands and examples", add_help=False)
    help_parser.set_defaults(func=lambda args: _print_ascii_help())

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

    list_metrics = subparsers.add_parser("list-metrics", help="List available metric templates (objective + subjective)")
    list_metrics.set_defaults(func=cmd_list_metrics)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:
        # Allow piping to tools that close stdout early (e.g., `head`, `Select-Object -First`).
        try:
            sys.stdout.close()
        finally:
            return


if __name__ == "__main__":
    main()
