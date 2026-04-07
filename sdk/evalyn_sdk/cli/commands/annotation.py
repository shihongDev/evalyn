"""Annotation commands: annotate, import-annotations, annotation-stats.

This module provides CLI commands for human annotation of dataset items.
Annotations are used to calibrate LLM judges - by comparing LLM judgments
to human labels, we can identify where the LLM is too strict or too lenient.

Commands:
- annotate: Interactive CLI for annotating dataset items (simple pass/fail or per-metric)
- import-annotations: Import annotations from a JSONL file (from external tools)
- annotation-stats: Show annotation coverage statistics and human-LLM agreement

Annotation modes:
- Simple mode: Mark each item as pass/fail overall
- Per-metric mode (--per-metric): Agree/disagree with each LLM metric judgment
- Span mode (--spans): Annotate individual spans (LLM calls, tool calls, etc.)

Key concepts:
- Annotations have confidence scores (1-5) to indicate certainty
- Per-metric annotations track agreement with LLM judges
- Disagreements are used by calibration to improve prompts

Typical workflow:
1. Run evaluation: 'evalyn run-eval --dataset <path>'
2. Annotate items: 'evalyn annotate --dataset <path> --per-metric'
3. Check coverage: 'evalyn annotation-stats --dataset <path>'
4. Calibrate: 'evalyn calibrate --metric-id <id> --annotations <path>'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..utils.errors import fatal_error
from ..utils.input_helpers import (
    truncate_text,
    get_bool_input,
    get_int_input,
    get_str_input,
    get_confidence,
)
from ..utils.command_common import resolve_dataset_dir_and_file
from ..utils.hints import HintCollector


def cmd_import_annotations(args: argparse.Namespace) -> None:
    """Import annotations from a JSONL file."""
    from ...annotation import import_annotations
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    anns = import_annotations(args.path)
    tracer.storage.store_annotations(anns)
    print(f"Imported {len(anns)} annotations into storage.")

    hints = HintCollector(quiet=getattr(args, "quiet", False))
    hints.add("evalyn annotation-stats --dataset <path>", "View annotation statistics")
    hints.render()


def cmd_annotation_stats(args: argparse.Namespace) -> None:
    """Show annotation coverage statistics."""
    from ...models import AnnotationItem

    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        # Look for annotation file or dataset file
        ann_file = dataset_path / "annotations.jsonl"
        if ann_file.exists():
            data_file = ann_file
        else:
            data_file = dataset_path / "dataset.jsonl"
            if not data_file.exists():
                data_file = dataset_path / "dataset.json"
    else:
        data_file = dataset_path

    if not data_file.exists():
        fatal_error(f"File not found: {data_file}")

    # Load data
    items = []
    raw = data_file.read_text(encoding="utf-8").strip()
    if not raw:
        print("Empty file.")
        return

    for line in raw.splitlines():
        if line.strip():
            data = json.loads(line)
            items.append(AnnotationItem.from_dict(data))

    # Also check storage for imported annotations
    try:
        from ...decorators import get_default_tracer
        tracer = get_default_tracer()
        if tracer.storage:
            stored_anns = tracer.storage.list_annotations(limit=10000)
            stored_by_target = {a.target_id: a for a in stored_anns}
            for item in items:
                item_id = getattr(item, "id", "") or ""
                if item_id in stored_by_target and not item.human_label:
                    item.human_label = stored_by_target[item_id]
    except Exception as e:
        logger.debug("Failed to load annotations from storage: %s", e)

    if not items:
        print("No items found.")
        return

    # Calculate stats
    total = len(items)
    with_labels = sum(1 for item in items if item.human_label)
    with_evals = sum(1 for item in items if item.eval_results)
    coverage = with_labels / total if total > 0 else 0

    # Per-metric stats
    metric_stats: Dict[str, Dict[str, int]] = {}
    for item in items:
        for metric_id, result in item.eval_results.items():
            if metric_id not in metric_stats:
                metric_stats[metric_id] = {"total": 0, "passed": 0, "failed": 0}
            metric_stats[metric_id]["total"] += 1
            if result.get("passed") is True:
                metric_stats[metric_id]["passed"] += 1
            elif result.get("passed") is False:
                metric_stats[metric_id]["failed"] += 1

    # Human agreement with LLM (if we have both)
    agreement_stats: Dict[str, Dict[str, int]] = {}
    for item in items:
        if not item.human_label or not item.eval_results:
            continue
        human_passed = getattr(item.human_label, 'passed', None)
        if human_passed is None and hasattr(item.human_label, 'label'):
            human_passed = item.human_label.label
        for metric_id, result in item.eval_results.items():
            llm_passed = result.get("passed")
            if llm_passed is None:
                continue
            if metric_id not in agreement_stats:
                agreement_stats[metric_id] = {
                    "agree": 0,
                    "disagree": 0,
                    "fp": 0,
                    "fn": 0,
                }
            if human_passed == llm_passed:
                agreement_stats[metric_id]["agree"] += 1
            else:
                agreement_stats[metric_id]["disagree"] += 1
                if llm_passed and not human_passed:
                    agreement_stats[metric_id]["fp"] += 1  # LLM=PASS, Human=FAIL
                else:
                    agreement_stats[metric_id]["fn"] += 1  # LLM=FAIL, Human=PASS

    # Print report
    print("\n" + "=" * 60)
    print("ANNOTATION COVERAGE REPORT")
    print("=" * 60)
    print(f"\nTotal items:        {total}")
    print(f"With human labels:  {with_labels} ({coverage * 100:.1f}%)")
    print(f"With eval results:  {with_evals}")
    print(f"Awaiting annotation: {total - with_labels}")

    if metric_stats:
        print("\n" + "-" * 60)
        print("EVAL RESULTS BY METRIC")
        print("-" * 60)
        print(f"{'Metric':<25} {'Total':<8} {'Pass':<8} {'Fail':<8} {'Pass %':<8}")
        print("-" * 60)
        for metric_id, stats in sorted(metric_stats.items()):
            pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            print(
                f"{metric_id:<25} {stats['total']:<8} {stats['passed']:<8} {stats['failed']:<8} {pass_rate * 100:.1f}%"
            )

    if agreement_stats:
        print("\n" + "-" * 60)
        print("HUMAN vs LLM AGREEMENT")
        print("-" * 60)
        print(
            f"{'Metric':<25} {'Agree':<8} {'Disagree':<8} {'FP':<8} {'FN':<8} {'Agr %':<8}"
        )
        print("-" * 60)
        for metric_id, stats in sorted(agreement_stats.items()):
            total_compared = stats["agree"] + stats["disagree"]
            agr_rate = stats["agree"] / total_compared if total_compared > 0 else 0
            print(
                f"{metric_id:<25} {stats['agree']:<8} {stats['disagree']:<8} {stats['fp']:<8} {stats['fn']:<8} {agr_rate * 100:.1f}%"
            )
        print("\nFP = False Positive (LLM=PASS, Human=FAIL)")
        print("FN = False Negative (LLM=FAIL, Human=PASS)")

    print("\n" + "=" * 60)

    # Show hint for next step
    if agreement_stats:
        # If there are disagreements, suggest calibration
        total_disagree = sum(s["disagree"] for s in agreement_stats.values())
        if total_disagree > 0:
            annotations_path = Path(dataset_path) / "annotations.jsonl"
            hints = HintCollector(quiet=getattr(args, "quiet", False))
            hints.add(
                f"evalyn calibrate --metric-id <metric> --annotations {annotations_path} --dataset {dataset_path}",
                "Calibrate LLM judges",
            )
            hints.render()


def _display_span_detail(span_type: str, detail: dict) -> None:
    """Display span-specific details based on type."""
    if span_type == "llm_call":
        print(f"  Model: {detail.get('model', 'unknown')}")
        print(
            f"  Tokens: {detail.get('input_tokens', '?')} in / {detail.get('output_tokens', '?')} out"
        )
        if detail.get("cost"):
            print(f"  Cost: ${detail.get('cost', 0):.6f}")
        if detail.get("response_excerpt"):
            print(
                f"  Response: {truncate_text(detail.get('response_excerpt', ''), 200)}"
            )
    elif span_type == "tool_call":
        print(f"  Tool: {detail.get('tool_name', 'unknown')}")
        if detail.get("args"):
            print(
                f"  Args: {truncate_text(json.dumps(detail.get('args', {}), ensure_ascii=False), 150)}"
            )
        if detail.get("result"):
            print(f"  Result: {truncate_text(str(detail.get('result', '')), 150)}")
    elif span_type == "reasoning":
        if detail.get("content"):
            print(f"  Content: {truncate_text(str(detail.get('content', '')), 200)}")
    elif span_type == "retrieval":
        if detail.get("query"):
            print(f"  Query: {truncate_text(str(detail.get('query', '')), 100)}")
        if detail.get("results_count"):
            print(f"  Results: {detail.get('results_count')} documents")


def _load_existing_span_annotations(
    output_path: Path, *, restart: bool
) -> Dict[str, SpanAnnotation]:
    """Load existing span annotations indexed by span_id."""
    from ...annotation import SpanAnnotation

    existing_annotations: Dict[str, SpanAnnotation] = {}
    if not output_path.exists() or restart:
        return existing_annotations

    try:
        for line in output_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            ann = SpanAnnotation.from_dict(data)
            existing_annotations[ann.span_id] = ann
    except Exception as e:
        logger.debug("Failed to load existing span annotations: %s", e)
    return existing_annotations


def _collect_spans_for_annotation(
    dataset_items: List[DatasetItem],
    tracer,
    *,
    span_type_filter: Optional[str],
    existing_annotations: Dict[str, SpanAnnotation],
) -> List[Dict[str, Any]]:
    """Collect unannotated spans with call context attached."""
    from ...annotation import SpanAnnotation, extract_spans_from_trace
    from ...models import DatasetItem

    all_spans: List[Dict[str, Any]] = []
    for item in dataset_items:
        call_id = item.metadata.get("call_id", item.id)
        call = tracer.storage.get_call(call_id)
        if not call:
            continue

        spans = extract_spans_from_trace(call)
        if span_type_filter:
            spans = [s for s in spans if s["span_type"] == span_type_filter]
        spans = [s for s in spans if s["span_id"] not in existing_annotations]

        for span in spans:
            span["call"] = call
            all_spans.append(span)
    return all_spans


def _summarize_span_types(spans: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count spans by span_type for the intro summary."""
    by_type: Dict[str, int] = {}
    for span in spans:
        st = span["span_type"]
        by_type[st] = by_type.get(st, 0) + 1
    return by_type


def _print_span_annotation_intro(
    *,
    data_file: Path,
    output_path: Path,
    all_spans: List[Dict[str, Any]],
    existing_annotations: Dict[str, SpanAnnotation],
) -> None:
    """Print span annotation session summary."""
    by_type = _summarize_span_types(all_spans)

    print("\n" + "=" * 70)
    print("SPAN ANNOTATION MODE")
    print("=" * 70)
    print(f"Dataset: {data_file}")
    print(f"Spans to annotate: {len(all_spans)}")
    print(f"Already annotated: {len(existing_annotations)}")
    print(f"Output: {output_path}")
    print("\nSpan types:")
    for st, count in sorted(by_type.items()):
        print(f"  {st}: {count}")
    print("\nCommands: [y/n/1-5] answer prompts  [s]kip span  [q]uit")
    print("=" * 70)


def _save_span_annotations(output_path: Path, annotations: List[SpanAnnotation]) -> None:
    """Persist span annotations to disk (full rewrite)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")


def _append_span_annotation(output_path: Path, ann: SpanAnnotation) -> None:
    """Append a single span annotation to disk (avoids full rewrite)."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")


def _display_span_annotation_context(span: Dict[str, Any], idx: int, total: int) -> None:
    """Display span context before prompting."""
    call = span["call"]
    span_type = span["span_type"]

    print(f"\n{'---' * 23}")
    print(f"Span {idx + 1}/{total} [{span_type.upper()}]")
    print(f"Call: {call.function_name} [{call.id[:12]}...]")
    print("---" * 23)

    print(f"\n{span['summary']}")

    if span_type == "overall":
        print(
            f"\nINPUT: {truncate_text(json.dumps(span.get('input', {}), ensure_ascii=False), 200)}"
        )
        print(f"OUTPUT: {truncate_text(str(span.get('output', '')), 300)}")
    elif span.get("detail"):
        _display_span_detail(span_type, span["detail"])

    print("---" * 23)


def _prompt_span_annotation_values(span_type: str) -> tuple[dict, bool, bool]:
    """Prompt for annotation values. Returns (values, quit_requested, skip_span)."""
    from ...annotation import get_annotation_prompts

    prompts = get_annotation_prompts(span_type)
    annotation_values: dict = {}
    quit_requested = False
    skip_span = False

    print(f"\nAnnotate this {span_type}:")

    for prompt_info in prompts:
        field = prompt_info["field"]
        question = prompt_info["question"]
        field_type = prompt_info["type"]

        if field_type == "bool":
            val = get_bool_input(question)
            if val is None:
                try:
                    check = input("  Skip this span? [y/n/q]: ").strip().lower()
                    if check in ("q", "quit"):
                        quit_requested = True
                        break
                    if check in ("y", "yes"):
                        skip_span = True
                        break
                except (EOFError, KeyboardInterrupt):
                    quit_requested = True
                    break
                continue
            annotation_values[field] = val
        elif field_type == "int":
            range_info = prompt_info.get("range", (1, 5))
            val = get_int_input(question, range_info[0], range_info[1])
            if val is not None:
                annotation_values[field] = val
        elif field_type == "str":
            val = get_str_input(question)
            if val:
                annotation_values[field] = val

    return annotation_values, quit_requested, skip_span


def _build_span_annotation_record(
    *,
    span: Dict[str, Any],
    args: argparse.Namespace,
    annotation_values: dict,
) -> SpanAnnotation:
    """Build a SpanAnnotation model from collected form values."""
    from ...annotation import SpanAnnotation, ANNOTATION_SCHEMAS

    call = span["call"]
    span_type = span["span_type"]
    span_id = span["span_id"]
    schema_cls = ANNOTATION_SCHEMAS[span_type]
    annotation_schema = schema_cls(**annotation_values)
    return SpanAnnotation(
        id=f"span-ann-{uuid.uuid4().hex[:8]}",
        call_id=call.id,
        span_id=span_id,
        span_type=span_type,
        annotation=annotation_schema,
        annotator=getattr(args, "annotator", None) or "human",
    )


def _handle_single_span_annotation(
    span: Dict[str, Any],
    idx: int,
    total: int,
    args: argparse.Namespace,
) -> tuple[str, Optional[SpanAnnotation]]:
    """Run one interactive span annotation step.

    Returns:
        ("quit"|"skip"|"saved"|"empty", annotation_or_none)
    """
    span_type = span["span_type"]
    _display_span_annotation_context(span, idx, total)
    annotation_values, quit_requested, skip_span = _prompt_span_annotation_values(
        span_type
    )

    if quit_requested:
        return "quit", None
    if skip_span:
        return "skip", None
    if not annotation_values:
        return "empty", None
    return "saved", _build_span_annotation_record(
        span=span, args=args, annotation_values=annotation_values
    )


def cmd_annotate_spans(args: argparse.Namespace) -> None:
    """Interactive span-level annotation interface."""
    from ...annotation import SpanAnnotation
    from ...datasets import load_dataset
    from ...decorators import get_default_tracer

    dataset_dir, data_file = resolve_dataset_dir_and_file(
        getattr(args, "dataset", None), getattr(args, "latest", False)
    )

    # Load dataset items to get call_ids
    dataset_items = load_dataset(data_file)
    if not dataset_items:
        print("No items in dataset.")
        return

    # Get storage to fetch full calls
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured", "Cannot retrieve call traces")

    # Get span_type filter
    span_type_filter = getattr(args, "span_type", "all")
    if span_type_filter == "all":
        span_type_filter = None

    # Output path for span annotations
    output_path = (
        Path(args.output) if args.output else dataset_dir / "span_annotations.jsonl"
    )

    existing_annotations = _load_existing_span_annotations(
        output_path,
        restart=getattr(args, "restart", False),
    )
    all_spans = _collect_spans_for_annotation(
        dataset_items,
        tracer,
        span_type_filter=span_type_filter,
        existing_annotations=existing_annotations,
    )

    if not all_spans:
        print(
            "No spans to annotate. All spans already annotated or no matching spans found."
        )
        if span_type_filter:
            print(f"Tip: Filter was set to '{span_type_filter}'. Try --span-type all")
        return

    _print_span_annotation_intro(
        data_file=data_file,
        output_path=output_path,
        all_spans=all_spans,
        existing_annotations=existing_annotations,
    )

    annotations: List[SpanAnnotation] = list(existing_annotations.values())

    idx = 0
    total = len(all_spans)

    while idx < total:
        span = all_spans[idx]
        result, span_annotation = _handle_single_span_annotation(span, idx, total, args)

        if result == "quit":
            _save_span_annotations(output_path, annotations)
            print(f"\nSaved {len(annotations)} span annotations to {output_path}")
            return

        if result == "skip":
            print("Skipped.")
            idx += 1
            continue

        if result == "saved" and span_annotation is not None:
            annotations.append(span_annotation)
            _append_span_annotation(output_path, span_annotation)
            print(f"Saved annotation for {span['span_type']}")
        else:
            print("No annotation recorded (all fields skipped).")

        idx += 1

    # Final save
    _save_span_annotations(output_path, annotations)
    print("\n" + "=" * 70)
    print("SPAN ANNOTATION COMPLETE")
    print(f"Total annotated: {len(annotations)}")
    print(f"Saved to: {output_path}")
    print("=" * 70)


def _annotation_truncate(text: str, max_len: int = 500) -> str:
    """Truncate annotation UI text for terminal display."""
    text = str(text) if text else ""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def _display_annotation_item(
    idx: int,
    total: int,
    item: DatasetItem,
    eval_results_by_call: Dict[str, Dict[str, Any]],
    *,
    show_metric_numbers: bool = False,
) -> List[tuple]:
    """Display item and return subjective metric tuples (metric_id, passed, reason)."""
    call_id = item.metadata.get("call_id", item.id)
    print(f"\n{'---' * 23}")
    print(f"Item {idx + 1}/{total} [{call_id[:12]}...]")
    print("---" * 23)

    input_text = (
        json.dumps(item.input, ensure_ascii=False, indent=2) if item.input else "(no input)"
    )
    print("\nINPUT:")
    if len(input_text) > 300:
        print(f"   {_annotation_truncate(input_text, 300)}")
    else:
        for line in input_text.split("\n")[:5]:
            print(f"   {line}")

    output_text = str(item.output) if item.output else "(no output)"
    print("\nOUTPUT:")
    print(f"   {_annotation_truncate(output_text, 500)}")

    eval_data = eval_results_by_call.get(call_id, {})
    subjective_metrics: List[tuple] = []
    if eval_data:
        print("\nLLM JUDGE RESULTS:")
        metric_num = 1
        for metric_id, result in eval_data.items():
            passed = result.get("passed")
            if passed is None:
                continue
            status = "PASS" if passed else "FAIL"
            reason = result.get("reason", "")
            if show_metric_numbers:
                print(f"   [{metric_num}] {metric_id}: {status}")
            else:
                print(f"   {metric_id}: {status}")
            if reason:
                print(f"       Reason: {_annotation_truncate(reason, 200)}")
            subjective_metrics.append((metric_id, passed, reason))
            metric_num += 1

    print("---" * 23)
    return subjective_metrics



def _save_single_annotation(output_path: Path, ann: Annotation) -> bool:
    """Append a single annotation atomically with fsync."""
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True
    except IOError as e:
        print(f"Warning: Failed to save annotation: {e}")
        return False


def _save_all_annotations_atomic(output_path: Path, annotations: List[Annotation]) -> bool:
    """Save all annotations atomically for recovery/fallback."""
    import tempfile

    try:
        temp_fd, temp_path = tempfile.mkstemp(
            dir=output_path.parent, prefix=".annotations_", suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                for ann in annotations:
                    f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, output_path)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except IOError as e:
        print(f"Warning: Failed to save annotations: {e}")
        return False


def _annotate_per_metric(
    item: DatasetItem,
    subjective_metrics: List[tuple],
    *,
    annotator: str,
    existing_count: int,
) -> Optional[Annotation]:
    """Per-metric annotation flow."""
    from ...models import Annotation, MetricLabel

    call_id = item.metadata.get("call_id", item.id)
    metric_labels: Dict[str, MetricLabel] = {}

    if not subjective_metrics:
        print("No subjective metrics to annotate for this item.")
        return None

    print("\nAnnotate each metric ([a]gree, [d]isagree/flip, [s]kip, [q]uit):")
    for i, (metric_id, llm_passed, reason) in enumerate(subjective_metrics, 1):
        status = "PASS" if llm_passed else "FAIL"
        print(f"\n  [{i}/{len(subjective_metrics)}] {metric_id}: LLM says {status}")
        if reason:
            print(f"      Reason: {_annotation_truncate(reason, 150)}")

        while True:
            try:
                choice = input("  Your verdict [a/d/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None

            if choice in ("q", "quit"):
                return None
            if choice in ("s", "skip"):
                break

            if choice in ("a", "agree", "y", "yes"):
                metric_labels[metric_id] = MetricLabel(
                    metric_id=metric_id,
                    agree_with_llm=True,
                    human_label=llm_passed,
                    notes="",
                )
                print(f"      -> Agreed: {status}")
                break

            if choice in ("d", "disagree", "n", "no", "flip"):
                human_label = not llm_passed
                human_status = "PASS" if human_label else "FAIL"
                try:
                    notes = input("      Notes (why disagree?): ").strip()
                except (EOFError, KeyboardInterrupt):
                    notes = ""
                metric_labels[metric_id] = MetricLabel(
                    metric_id=metric_id,
                    agree_with_llm=False,
                    human_label=human_label,
                    notes=notes,
                )
                print(f"      -> Disagreed: Human says {human_status}")
                break

            print("      Invalid. Use: a(gree), d(isagree), s(kip), q(uit)")

    if not metric_labels:
        print("No metrics annotated.")
        return None

    human_passes = [ml.human_label for ml in metric_labels.values()]
    overall_passed = all(human_passes) if human_passes else True
    confidence = get_confidence()

    try:
        overall_notes = input("Overall notes (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        overall_notes = ""

    return Annotation(
        id=f"ann-{call_id[:8]}-{existing_count}",
        target_id=call_id,
        label=overall_passed,
        rationale=overall_notes if overall_notes else None,
        annotator=annotator,
        source="human",
        confidence=confidence,
        metric_labels=metric_labels,
    )


def _annotate_simple(
    item: DatasetItem,
    *,
    annotator: str,
    existing_count: int,
) -> Optional[Annotation | str]:
    """Simple overall pass/fail annotation flow."""
    from ...models import Annotation

    call_id = item.metadata.get("call_id", item.id)
    while True:
        try:
            user_input = input("\nPass? [y/n/s/v/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if user_input in ("q", "quit"):
            return None
        if user_input in ("s", "skip"):
            return "skip"
        if user_input in ("v", "view"):
            print("\n" + "=" * 70)
            print("FULL OUTPUT:")
            print("=" * 70)
            print(str(item.output) if item.output else "(no output)")
            print("=" * 70)
            continue
        if user_input in ("y", "yes", "1", "p", "pass"):
            passed = True
        elif user_input in ("n", "no", "0", "f", "fail"):
            passed = False
        else:
            print("Invalid input. Use: y(es), n(o), s(kip), v(iew), q(uit)")
            continue

        confidence = get_confidence()
        try:
            notes = input("Notes (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""

        return Annotation(
            id=f"ann-{call_id[:8]}-{existing_count}",
            target_id=call_id,
            label=passed,
            rationale=notes if notes else None,
            annotator=annotator,
            source="human",
            confidence=confidence,
        )


def run_annotation(
    dataset_dir: str,
    *,
    output: Optional[str] = None,
    per_metric: bool = False,
    run_id: Optional[str] = None,
    annotator: str = "human",
    restart: bool = False,
    only_disagreements: bool = False,
) -> None:
    """Run interactive annotation - typed API callable without argparse."""
    args = argparse.Namespace(
        dataset=dataset_dir,
        latest=False,
        run_id=run_id,
        output=output,
        annotator=annotator,
        restart=restart,
        per_metric=per_metric,
        only_disagreements=only_disagreements,
        spans=False,
        span_type="all",
    )
    cmd_annotate(args)


def cmd_annotate(args: argparse.Namespace) -> None:
    """Interactive CLI annotation interface with per-metric support."""
    from ...datasets import load_dataset
    from ...decorators import get_default_tracer
    from ...models import Annotation

    # Check for span annotation mode
    if getattr(args, "spans", False):
        return cmd_annotate_spans(args)

    dataset_dir, data_file = resolve_dataset_dir_and_file(
        getattr(args, "dataset", None), getattr(args, "latest", False)
    )

    # Load dataset items
    dataset_items = load_dataset(data_file)
    if not dataset_items:
        print("No items in dataset.")
        return

    # Get eval run for LLM judge results
    tracer = get_default_tracer()
    run = None
    if args.run_id and tracer.storage:
        run = tracer.storage.get_eval_run(args.run_id)
    elif tracer.storage:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None

    # Build eval results lookup by call_id
    eval_results_by_call: Dict[str, Dict[str, Any]] = {}
    if run:
        for result in run.metric_results:
            if result.call_id not in eval_results_by_call:
                eval_results_by_call[result.call_id] = {}
            eval_results_by_call[result.call_id][result.metric_id] = {
                "score": result.score,
                "passed": result.passed,
                "reason": result.details.get("reason", "") if result.details else "",
            }

    # Load existing annotations if any
    output_path = (
        Path(args.output) if args.output else dataset_dir / "annotations.jsonl"
    )
    existing_annotations: Dict[str, Annotation] = {}
    if output_path.exists() and not args.restart:
        try:
            for line in output_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    data = json.loads(line)
                    ann = Annotation.from_dict(data)
                    existing_annotations[ann.target_id] = ann
        except Exception as e:
            logger.debug("Failed to load existing annotations file: %s", e)

    # Filter items based on options
    items_to_annotate = []
    for item in dataset_items:
        call_id = item.metadata.get("call_id", item.id)

        # Skip if already annotated (unless --restart)
        if call_id in existing_annotations and not args.restart:
            continue

        items_to_annotate.append(item)

    if not items_to_annotate:
        print("No items to annotate. All items already have annotations.")
        print(f"Use --restart to re-annotate, or check {output_path}")
        return

    total = len(items_to_annotate)
    annotated_count = len(existing_annotations)
    annotations: List[Annotation] = list(existing_annotations.values())
    new_annotation_count = 0  # Track new annotations for progress display
    per_metric_mode = args.per_metric

    print("\n" + "=" * 70)
    print("INTERACTIVE ANNOTATION" + (" (Per-Metric Mode)" if per_metric_mode else ""))
    print("=" * 70)
    print(f"Dataset: {data_file}")
    print(f"Items to annotate: {total}")
    print(f"Already annotated: {annotated_count}")
    if run:
        print(f"Using eval run: {run.id[:8]}...")
    print(f"Output: {output_path}")
    print("\nEach annotation is saved immediately - safe to quit anytime")
    if per_metric_mode:
        print("\nPer-metric commands:")
        print("  [a]gree with LLM  [d]isagree (flip)  [s]kip metric")
    else:
        print("\nCommands: [y]es/pass  [n]o/fail  [s]kip  [v]iew full  [q]uit")
    print("=" * 70)

    # Main annotation loop
    idx = 0
    while idx < total:
        item = items_to_annotate[idx]

        subjective_metrics = _display_annotation_item(
            idx,
            total,
            item,
            eval_results_by_call,
            show_metric_numbers=per_metric_mode,
        )

        if per_metric_mode:
            ann = _annotate_per_metric(
                item,
                subjective_metrics,
                annotator=args.annotator or "human",
                existing_count=len(annotations),
            )
            if ann is None:
                # Check if user wants to quit
                try:
                    quit_check = input("\nQuit? [y/n]: ").strip().lower()
                    if quit_check in ("y", "yes"):
                        print(
                            f"\nAll annotations already saved ({new_annotation_count} new this session)"
                        )
                        print(f"  Output: {output_path}")
                        return
                except (EOFError, KeyboardInterrupt):
                    print(
                        f"\nAll annotations already saved ({new_annotation_count} new this session)"
                    )
                    print(f"  Output: {output_path}")
                    return
                idx += 1
                continue
        else:
            ann = _annotate_simple(
                item,
                annotator=args.annotator or "human",
                existing_count=len(annotations),
            )
            if ann is None:
                print(
                    f"\nAll annotations already saved ({new_annotation_count} new this session)"
                )
                print(f"  Output: {output_path}")
                return
            if ann == "skip":
                print("Skipped.")
                idx += 1
                continue

        # Save annotation immediately (append mode with fsync)
        annotations.append(ann)
        if _save_single_annotation(output_path, ann):
            new_annotation_count += 1
        else:
            # Failed to save - try full atomic save as fallback
            print("  Attempting full save as fallback...")
            if _save_all_annotations_atomic(output_path, annotations):
                new_annotation_count += 1
            else:
                print("  Could not save annotation. Please check disk space.")

        # Show summary
        if per_metric_mode:
            agrees = sum(1 for ml in ann.metric_labels.values() if ml.agree_with_llm)
            total_metrics = len(ann.metric_labels)
            print(
                f"\nSaved: {agrees}/{total_metrics} agree with LLM, overall={('PASS' if ann.label else 'FAIL')}"
            )
        else:
            status = "PASS" if ann.label else "FAIL"
            conf_str = f", confidence={ann.confidence}" if ann.confidence else ""
            print(
                f"Saved: {status}{conf_str}"
                + (f" - {ann.rationale}" if ann.rationale else "")
            )

        idx += 1

    # All annotations already saved incrementally - just show summary
    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print(
        f"Total annotated: {len(annotations)} ({new_annotation_count} new this session)"
    )
    print(f"Saved to: {output_path}")
    print("=" * 70)

    hints = HintCollector(quiet=getattr(args, "quiet", False))
    hints.add(
        f"evalyn calibrate --metric-id <metric> --annotations {output_path} --dataset {dataset_dir}",
        "Calibrate LLM judges",
    )
    hints.render()


def register_commands(subparsers) -> None:
    """Register annotation commands."""
    # annotate
    annotate_parser = subparsers.add_parser(
        "annotate", help="Interactive CLI for annotating dataset items"
    )
    annotate_parser.add_argument(
        "--dataset", help="Path to dataset directory or dataset.jsonl file"
    )
    annotate_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    annotate_parser.add_argument(
        "--run-id", help="Eval run ID to show LLM judge results (defaults to latest)"
    )
    annotate_parser.add_argument(
        "--output",
        help="Output path for annotations (defaults to <dataset>/annotations.jsonl)",
    )
    annotate_parser.add_argument(
        "--annotator", default="human", help="Annotator name/id (default: human)"
    )
    annotate_parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart annotation from scratch (ignore existing)",
    )
    annotate_parser.add_argument(
        "--per-metric",
        action="store_true",
        help="Annotate each metric separately (agree/disagree with LLM)",
    )
    annotate_parser.add_argument(
        "--spans",
        action="store_true",
        help="Annotate individual spans (LLM calls, tool calls, etc.)",
    )
    annotate_parser.add_argument(
        "--span-type",
        choices=["all", "llm_call", "tool_call", "reasoning", "retrieval"],
        default="all",
        help="Filter span types to annotate",
    )
    annotate_parser.set_defaults(func=cmd_annotate)

    # import-annotations
    import_ann = subparsers.add_parser(
        "import-annotations", help="Import annotations from a JSONL file"
    )
    import_ann.add_argument("--path", required=True, help="Path to annotations JSONL")
    import_ann.set_defaults(func=cmd_import_annotations)

    # annotation-stats
    ann_stats = subparsers.add_parser(
        "annotation-stats", help="Show annotation coverage statistics"
    )
    ann_stats.add_argument(
        "--dataset",
        required=True,
        help="Path to annotations.jsonl or dataset directory",
    )
    ann_stats.set_defaults(func=cmd_annotation_stats)


__all__ = [
    "cmd_annotate",
    "cmd_import_annotations",
    "cmd_annotation_stats",
    "cmd_annotate_spans",
    "register_commands",
]
