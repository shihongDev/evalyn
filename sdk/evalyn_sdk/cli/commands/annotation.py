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
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...annotation import import_annotations
from ..utils.errors import fatal_error
from ..utils.input_helpers import (
    truncate_text,
    get_bool_input,
    get_int_input,
    get_str_input,
)
from ...annotation import (
    SpanAnnotation,
    ANNOTATION_SCHEMAS,
    extract_spans_from_trace,
    get_annotation_prompts,
)
from ...datasets import load_dataset
from ...decorators import get_default_tracer
from ...models import AnnotationItem, Annotation, MetricLabel, DatasetItem
from ..utils.config import load_config, resolve_dataset_path
from ..utils.hints import print_hint


def cmd_import_annotations(args: argparse.Namespace) -> None:
    """Import annotations from a JSONL file."""
    tracer = get_default_tracer()
    if not tracer.storage:
        print("No storage configured.")
        return
    anns = import_annotations(args.path)
    tracer.storage.store_annotations(anns)
    print(f"Imported {len(anns)} annotations into storage.")

    print_hint(
        "To view annotation statistics, run: evalyn annotation-stats --dataset <path>",
        quiet=getattr(args, "quiet", False),
    )


def cmd_annotation_stats(args: argparse.Namespace) -> None:
    """Show annotation coverage statistics."""
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
        human_passed = item.human_label.passed
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
            print_hint(
                f"To calibrate LLM judges, run: evalyn calibrate --metric-id <metric> --annotations {annotations_path} --dataset {dataset_path}",
                quiet=getattr(args, "quiet", False),
            )


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


def cmd_annotate_spans(args: argparse.Namespace) -> None:
    """Interactive span-level annotation interface."""
    config = load_config()

    # Resolve dataset path
    dataset_arg = getattr(args, "dataset", None)
    use_latest = getattr(args, "latest", False)
    resolved_path = resolve_dataset_path(dataset_arg, use_latest, config)

    if not resolved_path:
        fatal_error("No dataset specified", "Use --dataset <path> or --latest")

    dataset_path = Path(resolved_path)
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        data_file = dataset_path / "dataset.jsonl"
        if not data_file.exists():
            data_file = dataset_path / "dataset.json"
    else:
        dataset_dir = dataset_path.parent
        data_file = dataset_path

    if not data_file.exists():
        fatal_error(f"Dataset file not found: {data_file}")

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

    # Load existing span annotations
    existing_annotations: Dict[str, SpanAnnotation] = {}
    if output_path.exists() and not getattr(args, "restart", False):
        try:
            for line in output_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    data = json.loads(line)
                    ann = SpanAnnotation.from_dict(data)
                    existing_annotations[ann.span_id] = ann
        except Exception:
            pass

    # Collect all spans to annotate
    all_spans = []
    for item in dataset_items:
        call_id = item.metadata.get("call_id", item.id)

        # Fetch the full call from storage
        call = tracer.storage.get_call(call_id)
        if not call:
            continue

        # Extract spans from the call
        spans = extract_spans_from_trace(call)

        # Filter by span_type if specified
        if span_type_filter:
            spans = [s for s in spans if s["span_type"] == span_type_filter]

        # Skip already annotated spans
        spans = [s for s in spans if s["span_id"] not in existing_annotations]

        for span in spans:
            span["call"] = call  # Attach call for context
            all_spans.append(span)

    if not all_spans:
        print(
            "No spans to annotate. All spans already annotated or no matching spans found."
        )
        if span_type_filter:
            print(f"Tip: Filter was set to '{span_type_filter}'. Try --span-type all")
        return

    # Group spans by type for summary
    by_type = {}
    for span in all_spans:
        st = span["span_type"]
        by_type[st] = by_type.get(st, 0) + 1

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

    annotations: List[SpanAnnotation] = list(existing_annotations.values())

    def save_annotations() -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")

    idx = 0
    total = len(all_spans)

    while idx < total:
        span = all_spans[idx]
        call = span["call"]
        span_type = span["span_type"]
        span_id = span["span_id"]

        print(f"\n{'---' * 23}")
        print(f"Span {idx + 1}/{total} [{span_type.upper()}]")
        print(f"Call: {call.function_name} [{call.id[:12]}...]")
        print("---" * 23)

        # Show span context
        print(f"\n{span['summary']}")

        if span_type == "overall":
            # Show input/output for overall
            print(
                f"\nINPUT: {truncate_text(json.dumps(span.get('input', {}), ensure_ascii=False), 200)}"
            )
            print(f"OUTPUT: {truncate_text(str(span.get('output', '')), 300)}")
        elif span.get("detail"):
            _display_span_detail(span_type, span["detail"])

        print("---" * 23)

        # Get annotation prompts for this span type
        prompts = get_annotation_prompts(span_type)
        schema_cls = ANNOTATION_SCHEMAS[span_type]
        annotation_values = {}

        print(f"\nAnnotate this {span_type}:")

        quit_requested = False
        skip_span = False

        for prompt_info in prompts:
            field = prompt_info["field"]
            question = prompt_info["question"]
            field_type = prompt_info["type"]

            if field_type == "bool":
                val = get_bool_input(question)
                if val is None:
                    # Check for quit
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
                    continue  # Skip this field
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

        if quit_requested:
            save_annotations()
            print(f"\nSaved {len(annotations)} span annotations to {output_path}")
            return

        if skip_span:
            print("Skipped.")
            idx += 1
            continue

        # Create annotation if we have any values
        if annotation_values:
            annotation_schema = schema_cls(**annotation_values)
            span_ann = SpanAnnotation(
                id=f"span-ann-{uuid.uuid4().hex[:8]}",
                call_id=call.id,
                span_id=span_id,
                span_type=span_type,
                annotation=annotation_schema,
                annotator=getattr(args, "annotator", None) or "human",
            )
            annotations.append(span_ann)
            save_annotations()
            print(f"Saved annotation for {span_type}")
        else:
            print("No annotation recorded (all fields skipped).")

        idx += 1

    # Final save
    save_annotations()
    print("\n" + "=" * 70)
    print("SPAN ANNOTATION COMPLETE")
    print(f"Total annotated: {len(annotations)}")
    print(f"Saved to: {output_path}")
    print("=" * 70)


def cmd_annotate(args: argparse.Namespace) -> None:
    """Interactive CLI annotation interface with per-metric support."""
    # Check for span annotation mode
    if getattr(args, "spans", False):
        return cmd_annotate_spans(args)

    config = load_config()

    # Resolve dataset path using --dataset, --latest, or config
    dataset_arg = getattr(args, "dataset", None)
    use_latest = getattr(args, "latest", False)
    resolved_path = resolve_dataset_path(dataset_arg, use_latest, config)

    if not resolved_path:
        fatal_error("No dataset specified", "Use --dataset <path> or --latest")

    # Resolve dataset path
    dataset_path = Path(resolved_path)
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        data_file = dataset_path / "dataset.jsonl"
        if not data_file.exists():
            data_file = dataset_path / "dataset.json"
    else:
        dataset_dir = dataset_path.parent
        data_file = dataset_path

    if not data_file.exists():
        fatal_error(f"Dataset file not found: {data_file}")

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
        except Exception:
            pass

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

    def truncate_text(text: str, max_len: int = 500) -> str:
        text = str(text) if text else ""
        text = text.replace("\n", " ").strip()
        return text if len(text) <= max_len else text[:max_len] + "..."

    def display_item(
        idx: int, item: DatasetItem, show_metric_numbers: bool = False
    ) -> List[tuple]:
        """Display item and return list of (metric_id, llm_passed, reason) for subjective metrics."""
        call_id = item.metadata.get("call_id", item.id)
        print(f"\n{'---' * 23}")
        print(f"Item {idx + 1}/{total} [{call_id[:12]}...]")
        print("---" * 23)

        # Input
        input_text = (
            json.dumps(item.input, ensure_ascii=False, indent=2)
            if item.input
            else "(no input)"
        )
        print("\nINPUT:")
        if len(input_text) > 300:
            print(f"   {truncate_text(input_text, 300)}")
        else:
            for line in input_text.split("\n")[:5]:
                print(f"   {line}")

        # Output
        output_text = str(item.output) if item.output else "(no output)"
        print("\nOUTPUT:")
        print(f"   {truncate_text(output_text, 500)}")

        # LLM Judge results
        eval_data = eval_results_by_call.get(call_id, {})
        subjective_metrics = []
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
                    print(f"       Reason: {truncate_text(reason, 200)}")
                subjective_metrics.append((metric_id, passed, reason))
                metric_num += 1

        print("---" * 23)
        return subjective_metrics

    def get_confidence() -> Optional[int]:
        """Get confidence score 1-5 from user."""
        try:
            conf_input = input("Confidence (1-5, Enter to skip): ").strip()
            if not conf_input:
                return None
            conf = int(conf_input)
            if 1 <= conf <= 5:
                return conf
            print("Invalid. Use 1-5.")
            return get_confidence()
        except ValueError:
            print("Invalid. Use 1-5 or Enter to skip.")
            return get_confidence()
        except (EOFError, KeyboardInterrupt):
            return None

    def save_single_annotation(ann: Annotation) -> bool:
        """Append a single annotation atomically with fsync.

        Returns True if saved successfully, False otherwise.
        """
        try:
            # Append mode - each annotation is written immediately
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return True
        except IOError as e:
            print(f"Warning: Failed to save annotation: {e}")
            return False

    def save_all_annotations_atomic() -> bool:
        """Save all annotations atomically (for final save or recovery).

        Writes to temp file, then renames for atomic operation.
        Returns True if saved successfully, False otherwise.
        """
        import tempfile

        try:
            # Write to temp file in same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=output_path.parent, prefix=".annotations_", suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    for ann in annotations:
                        f.write(json.dumps(ann.as_dict(), ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                # Atomic rename
                os.replace(temp_path, output_path)
                return True
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except IOError as e:
            print(f"Warning: Failed to save annotations: {e}")
            return False

    def annotate_per_metric(
        item: DatasetItem, subjective_metrics: List[tuple]
    ) -> Optional[Annotation]:
        """Per-metric annotation flow."""
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
                print(f"      Reason: {truncate_text(reason, 150)}")

            while True:
                try:
                    choice = input("  Your verdict [a/d/s/q]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return None

                if choice in ("q", "quit"):
                    return None  # Signal to quit

                if choice in ("s", "skip"):
                    break  # Skip this metric

                if choice in ("a", "agree", "y", "yes"):
                    # Agree with LLM
                    metric_labels[metric_id] = MetricLabel(
                        metric_id=metric_id,
                        agree_with_llm=True,
                        human_label=llm_passed,
                        notes="",
                    )
                    print(f"      -> Agreed: {status}")
                    break

                if choice in ("d", "disagree", "n", "no", "flip"):
                    # Disagree - flip the label
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

        # Calculate overall label from metric labels
        human_passes = [ml.human_label for ml in metric_labels.values()]
        overall_passed = all(human_passes) if human_passes else True

        # Get confidence
        confidence = get_confidence()

        # Get overall notes
        try:
            overall_notes = input("Overall notes (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            overall_notes = ""

        return Annotation(
            id=f"ann-{call_id[:8]}-{len(annotations)}",
            target_id=call_id,
            label=overall_passed,
            rationale=overall_notes if overall_notes else None,
            annotator=args.annotator or "human",
            source="human",
            confidence=confidence,
            metric_labels=metric_labels,
        )

    def annotate_simple(item: DatasetItem) -> Optional[Annotation]:
        """Simple overall pass/fail annotation flow."""
        call_id = item.metadata.get("call_id", item.id)

        while True:
            try:
                user_input = input("\nPass? [y/n/s/v/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None

            if user_input in ("q", "quit"):
                return None  # Signal to quit

            if user_input in ("s", "skip"):
                return "skip"  # Signal to skip

            if user_input in ("v", "view"):
                # Show full output
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

            # Get confidence
            confidence = get_confidence()

            # Get optional notes
            try:
                notes = input("Notes (optional): ").strip()
            except (EOFError, KeyboardInterrupt):
                notes = ""

            return Annotation(
                id=f"ann-{call_id[:8]}-{len(annotations)}",
                target_id=call_id,
                label=passed,
                rationale=notes if notes else None,
                annotator=args.annotator or "human",
                source="human",
                confidence=confidence,
            )

    # Main annotation loop
    idx = 0
    while idx < total:
        item = items_to_annotate[idx]
        call_id = item.metadata.get("call_id", item.id)

        subjective_metrics = display_item(
            idx, item, show_metric_numbers=per_metric_mode
        )

        if per_metric_mode:
            ann = annotate_per_metric(item, subjective_metrics)
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
            ann = annotate_simple(item)
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
        if save_single_annotation(ann):
            new_annotation_count += 1
        else:
            # Failed to save - try full atomic save as fallback
            print("  Attempting full save as fallback...")
            if save_all_annotations_atomic():
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

    print_hint(
        f"To calibrate LLM judges, run: evalyn calibrate --metric-id <metric> --annotations {output_path} --dataset {dataset_dir}",
        quiet=getattr(args, "quiet", False),
    )


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
