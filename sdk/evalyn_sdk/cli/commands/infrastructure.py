"""Infrastructure commands: init, one-click.

This module provides CLI commands for setup and running the full evaluation pipeline.

Commands:
- init: Initialize configuration file (evalyn.yaml) from template
- one-click: Run complete evaluation pipeline in one command

The one-click pipeline:
1. Build Dataset: Collect traces from storage by project/version
2. Suggest Metrics: Select metrics based on mode (basic/llm-registry/llm-brainstorm/bundle)
3. Run Initial Evaluation: Execute metrics on dataset items
4. Human Annotation: Interactive annotation session (optional)
5. Calibrate LLM Judges: Optimize prompts using annotations (optional)
6. Re-evaluate with Calibrated Prompts: Run again with improved judges (optional)
7. Generate Simulations: Create synthetic test data (optional)

Pipeline state:
- Progress is saved to pipeline_state.json for resume capability
- Use --resume to continue from where you left off
- Each step outputs to numbered folders (1_dataset, 2_metrics, etc.)

Typical workflow:
1. Initialize config: 'evalyn init'
2. Set API key: 'export GEMINI_API_KEY=...'
3. Run pipeline: 'evalyn one-click --project <name>'
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ...datasets import load_dataset, build_dataset_from_storage, save_dataset_with_meta
from ...models import MetricSpec
from ...storage import SQLiteStorage
from ..utils.config import load_config, get_config_default
from ..utils.loaders import _load_callable


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize configuration file by copying from evalyn.yaml.example."""
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        print(
            f"Error: {output_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find the example file - check multiple locations
    example_paths = [
        Path("evalyn.yaml.example"),  # Current directory
        Path(__file__).parent.parent.parent.parent
        / "evalyn.yaml.example",  # Project root
    ]

    example_path = None
    for p in example_paths:
        if p.exists():
            example_path = p
            break

    if example_path:
        shutil.copy(example_path, output_path)
        print(f"Created {output_path} (from {example_path})")
    else:
        # Fallback: create minimal config if example not found
        minimal = """# Evalyn Configuration
# See evalyn.yaml.example for all available options

# API Keys - only set what you need
api_keys:
  gemini: "your-gemini-api-key-here"  # Required for example agent
  # openai: "your-openai-key"         # Optional

llm:
  model: "gemini-2.5-flash-lite"

defaults:
  project: null
  version: null
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(minimal)
        print(f"Created {output_path} (minimal config)")
        print("Note: evalyn.yaml.example not found for full template")

    print("\nSet your API key:")
    print("  export GEMINI_API_KEY='your-key'")
    print(f"  # or edit {output_path} directly")


def cmd_one_click(args: argparse.Namespace) -> None:
    """Run the complete evaluation pipeline from dataset building to calibrated evaluation.

    Pipeline steps:
    [1/7] Build Dataset: Collect traces by project, save to 1_dataset/
    [2/7] Suggest Metrics: Select metrics based on mode, save to 2_metrics/
    [3/7] Run Initial Eval: Execute metrics, save to 3_initial_eval/
    [4/7] Human Annotation: Interactive session, save to 4_annotations/ (optional)
    [5/7] Calibrate Judges: Optimize prompts, save to 5_calibrations/ (optional)
    [6/7] Calibrated Eval: Re-run with improved prompts, save to 6_calibrated_eval/
    [7/7] Simulate: Generate synthetic data, save to 7_simulations/ (optional)

    State is persisted to pipeline_state.json for resume capability.
    """
    # Import here to avoid circular imports
    from .annotation import cmd_annotate
    from .calibration import cmd_calibrate
    from .simulate import cmd_simulate

    # Load config and apply defaults (CLI args take precedence)
    config = load_config()

    # Apply config defaults for unset args
    if not args.version:
        args.version = get_config_default(config, "defaults", "version")
    if not args.metric_mode or args.metric_mode == "basic":
        args.metric_mode = get_config_default(
            config, "metrics", "mode", default="basic"
        )
    if not args.model:
        args.model = get_config_default(
            config, "llm", "model", default="gemini-2.5-flash-lite"
        )
    if not args.llm_mode:
        args.llm_mode = get_config_default(config, "llm", "mode", default="api")
    if not args.bundle:
        args.bundle = get_config_default(config, "metrics", "bundle")
    if args.dataset_limit == 100:  # default value
        args.dataset_limit = get_config_default(config, "dataset", "limit", default=100)
    if not args.skip_annotation:
        args.skip_annotation = get_config_default(
            config, "annotation", "skip", default=False
        )
    if args.annotation_limit == 20:  # default value
        args.annotation_limit = get_config_default(
            config, "annotation", "limit", default=20
        )
    if not args.per_metric:
        args.per_metric = get_config_default(
            config, "annotation", "per_metric", default=False
        )
    if not args.skip_calibration:
        args.skip_calibration = get_config_default(
            config, "calibration", "skip", default=False
        )
    if not args.optimizer or args.optimizer == "llm":
        args.optimizer = get_config_default(
            config, "calibration", "optimizer", default="llm"
        )
    if not args.enable_simulation:
        args.enable_simulation = get_config_default(
            config, "simulation", "enable", default=False
        )
    if not args.simulation_modes or args.simulation_modes == "similar":
        args.simulation_modes = get_config_default(
            config, "simulation", "modes", default="similar"
        )
    if args.num_similar == 3:  # default value
        args.num_similar = get_config_default(
            config, "simulation", "num_similar", default=3
        )
    if args.num_outlier == 2:  # default value
        args.num_outlier = get_config_default(
            config, "simulation", "num_outlier", default=1
        )
    if args.max_sim_seeds == 10:  # default value
        args.max_sim_seeds = get_config_default(
            config, "simulation", "max_seeds", default=50
        )
    if not args.auto_yes:
        args.auto_yes = get_config_default(
            config, "pipeline", "auto_yes", default=False
        )
    if not args.verbose:
        args.verbose = get_config_default(config, "pipeline", "verbose", default=False)

    # If no version specified, query available versions and prompt
    if not args.version:
        storage = SQLiteStorage()
        calls = storage.list_calls(limit=500)
        versions = set()
        for call in calls:
            meta = call.metadata or {}
            if (
                meta.get("project_name") == args.project
                or meta.get("project_id") == args.project
            ):
                v = meta.get("version")
                if v:
                    versions.add(v)

        if len(versions) == 0:
            print(f"No versions found for project '{args.project}'")
            print("Proceeding without version filter (all traces)...")
        elif len(versions) == 1:
            args.version = list(versions)[0]
            print(f"Auto-selected version: {args.version}")
        else:
            print(f"\nAvailable versions for project '{args.project}':")
            version_list = sorted(versions)
            for i, v in enumerate(version_list, 1):
                print(f"  {i}. {v}")
            print(f"  {len(version_list) + 1}. [all versions]")

            try:
                choice = input("\nSelect version (number): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(version_list):
                    args.version = version_list[idx]
                elif idx == len(version_list):
                    args.version = None  # all versions
                else:
                    print("Invalid selection, using all versions")
            except (ValueError, EOFError):
                print("Invalid input, using all versions")

    # Print header
    print("\n" + "=" * 70)
    print(" " * 15 + "EVALYN ONE-CLICK EVALUATION PIPELINE")
    print("=" * 70)
    print(f"\nProject:  {args.project}")
    if args.target:
        print(f"Target:   {args.target}")
    if args.version:
        print(f"Version:  {args.version}")
    print(
        f"Mode:     {args.metric_mode}"
        + (f" ({args.model})" if args.metric_mode != "basic" else "")
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        version_str = f"-{args.version}" if args.version else ""
        output_dir = Path("data") / f"{args.project}{version_str}-{timestamp}-oneclick"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:   {output_dir}")
    print("\n" + "-" * 70 + "\n")

    if args.dry_run:
        print("DRY RUN MODE - showing what would be done:\n")

    # State file path for persistence
    state_path = output_dir / "pipeline_state.json"

    def save_state_atomic(state: dict) -> None:
        """Save state atomically after each step."""
        import tempfile

        try:
            state["updated_at"] = datetime.now().isoformat()
            temp_fd, temp_path = tempfile.mkstemp(
                dir=output_dir,
                prefix=".state_",
                suffix=".tmp",
            )
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, state_path)
        except Exception:
            # Non-fatal - continue even if state save fails
            pass

    # Check for resume mode
    resumed = False
    if state_path.exists() and getattr(args, "resume", False):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            print("  Resuming from previous run...")
            print(f"  Completed steps: {', '.join(state.get('steps', {}).keys())}\n")
            resumed = True
        except Exception:
            state = None

    # Initialize state tracking if not resuming
    if not resumed or state is None:
        # Filter out non-serializable items (like func) from args
        args_dict = {k: v for k, v in vars(args).items() if not callable(v)}
        state = {
            "started_at": datetime.now().isoformat(),
            "config": args_dict,
            "steps": {},
            "output_dir": str(output_dir),
        }

    # Helper to check if a step is already done
    def step_done(step_name: str) -> bool:
        return (
            resumed
            and step_name in state.get("steps", {})
            and state["steps"][step_name].get("status") == "success"
        )

    # Variables to track across steps
    dataset_path = None
    metrics_path = None
    target_fn = None
    metric_specs = []

    try:
        # Step 1: Build Dataset
        print("[1/7] Building Dataset")
        dataset_dir = output_dir / "1_dataset"
        dataset_dir.mkdir(exist_ok=True)

        if step_done("1_dataset"):
            print("  Already completed (skipping)\n")
            dataset_path = Path(state["steps"]["1_dataset"]["output"])
        elif args.dry_run:
            print(
                f"  -> Would build dataset with: project={args.project}, limit={args.dataset_limit}"
            )
            print(f"  -> Would save to: {dataset_dir}/dataset.jsonl\n")
        else:
            from datetime import datetime as dt

            # Parse date filters
            since = dt.fromisoformat(args.since) if args.since else None
            until = dt.fromisoformat(args.until) if args.until else None

            storage = SQLiteStorage()
            items = build_dataset_from_storage(
                storage,
                project_name=args.project,
                version=args.version,
                production_only=args.production_only,
                simulation_only=args.simulation_only,
                since=since,
                until=until,
                limit=args.dataset_limit,
                success_only=True,
                include_metadata=True,
            )

            if not items:
                print("  No traces found matching filters")
                return

            meta = {
                "project": args.project,
                "version": args.version or "all",
                "created_at": datetime.now().isoformat(),
                "filters": {
                    "production_only": args.production_only,
                    "simulation_only": args.simulation_only,
                    "since": args.since,
                    "until": args.until,
                },
                "item_count": len(items),
            }

            dataset_path = save_dataset_with_meta(items, dataset_dir, meta)
            print(f"  Found {len(items)} items")
            print(f"  Saved to: {dataset_path}\n")

            state["steps"]["1_dataset"] = {
                "status": "success",
                "output": str(dataset_path),
                "item_count": len(items),
            }
            save_state_atomic(state)

        # Step 2: Suggest Metrics
        print("[2/7] Suggesting Metrics")
        metrics_dir = output_dir / "2_metrics"
        metrics_dir.mkdir(exist_ok=True)

        if step_done("2_metrics"):
            print("  Already completed (skipping)\n")
            metrics_path = Path(state["steps"]["2_metrics"]["output"])
        elif args.dry_run:
            print(f"  -> Would suggest metrics with mode={args.metric_mode}")
            print(f"  -> Would save to: {metrics_dir}/metrics.json\n")
        else:
            # Call suggest-metrics logic
            metrics_path = metrics_dir / "metrics.json"

            # Load target function if provided
            target_fn = None
            if args.target:
                try:
                    target_fn = _load_callable(args.target)
                except Exception as e:
                    print(f"  Could not load target function: {args.target}")
                    print(f"    Error: {e}")
                    print("  -> Using trace-based suggestions only")

            # Get sample traces
            storage = SQLiteStorage()
            calls = storage.list_calls(limit=5)

            # Import suggesters
            from ...metrics.suggester import (
                HeuristicSuggester,
                LLMSuggester,
                LLMRegistrySelector,
            )

            # Suggest metrics based on mode
            if args.metric_mode == "all":
                # Comprehensive mode: run all modes and merge
                all_metrics = []
                seen_ids = set()

                # 1. Basic mode (fast, offline)
                print("    -> Running basic mode...")
                suggester = HeuristicSuggester()
                basic_specs = suggester.suggest(target_fn, calls)
                for spec in basic_specs:
                    if spec.id not in seen_ids:
                        all_metrics.append(spec)
                        seen_ids.add(spec.id)

                # 2. LLM-registry mode (LLM picks from templates)
                print("    -> Running llm-registry mode...")
                try:
                    selector = LLMRegistrySelector(
                        model=args.model, llm_mode=args.llm_mode
                    )
                    llm_specs = selector.select_metrics(target_fn, calls)
                    for spec in llm_specs:
                        if spec.id not in seen_ids:
                            all_metrics.append(spec)
                            seen_ids.add(spec.id)
                except Exception as e:
                    print(f"    LLM-registry failed: {e}")

                # 3. Bundle mode (if bundle specified)
                if args.bundle:
                    print(f"    -> Adding bundle: {args.bundle}...")
                    from ...metrics.bundles import get_bundle_metrics

                    bundle_specs = get_bundle_metrics(args.bundle)
                    for spec in bundle_specs:
                        if spec.id not in seen_ids:
                            all_metrics.append(spec)
                            seen_ids.add(spec.id)

                metric_specs = all_metrics
                print(f"    -> Merged {len(metric_specs)} unique metrics")

            elif args.metric_mode == "basic":
                suggester = HeuristicSuggester()
                metric_specs = suggester.suggest(target_fn, calls)
            elif args.metric_mode == "llm-registry":
                selector = LLMRegistrySelector(model=args.model, llm_mode=args.llm_mode)
                metric_specs = selector.select_metrics(target_fn, calls)
            elif args.metric_mode == "llm-brainstorm":
                suggester = LLMSuggester(model=args.model, llm_mode=args.llm_mode)
                metric_specs = suggester.suggest(target_fn, calls)
            else:  # bundle
                from ...metrics.bundles import get_bundle_metrics

                metric_specs = get_bundle_metrics(args.bundle) if args.bundle else []

            # Save metrics
            payload = []
            for spec in metric_specs:
                payload.append(
                    {
                        "id": spec.id,
                        "name": getattr(spec, "name", spec.id),
                        "type": spec.type,
                        "description": spec.description,
                        "config": spec.config,
                        "why": getattr(spec, "why", ""),
                    }
                )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            obj_count = sum(1 for spec in metric_specs if spec.type == "objective")
            subj_count = sum(1 for spec in metric_specs if spec.type == "subjective")

            print(
                f"  Selected {len(metric_specs)} metrics ({obj_count} objective, {subj_count} subjective)"
            )
            for spec in metric_specs:
                print(f"    - {spec.id} ({spec.type})")
            print(f"  Saved to: {metrics_path}\n")

            state["steps"]["2_metrics"] = {
                "status": "success",
                "output": str(metrics_path),
                "total": len(metric_specs),
                "objective": obj_count,
                "subjective": subj_count,
            }
            save_state_atomic(state)

        # Step 3: Run Initial Evaluation
        print("[3/7] Running Initial Evaluation")
        eval_dir = output_dir / "3_initial_eval"
        eval_dir.mkdir(exist_ok=True)

        if step_done("3_initial_eval"):
            print("  Already completed (skipping)\n")
        elif args.dry_run:
            print(f"  -> Would run evaluation on {args.dataset_limit} items")
            print(f"  -> Would save to: {eval_dir}/\n")
        else:
            # Run evaluation
            from ...runner import EvalRunner
            from ...metrics.factory import (
                build_objective_metric,
                build_subjective_metric,
            )

            # Load metrics from file
            with open(metrics_path, encoding="utf-8") as f:
                metrics_data = json.load(f)

            metrics = []
            for spec_data in metrics_data:
                spec = MetricSpec(
                    id=spec_data["id"],
                    name=spec_data.get("name", spec_data["id"]),
                    type=spec_data["type"],
                    description=spec_data.get("description", ""),
                    config=spec_data.get("config", {}),
                )
                # Get API key from config
                pipeline_gemini_key = get_config_default(config, "api_keys", "gemini")
                try:
                    if spec.type == "objective":
                        m = build_objective_metric(spec.id, spec.config)
                    else:
                        m = build_subjective_metric(
                            spec.id, spec.config, api_key=pipeline_gemini_key
                        )
                    if m:
                        metrics.append(m)
                except Exception:
                    pass

            items = list(load_dataset(dataset_path))
            runner = EvalRunner(
                target_fn=target_fn or (lambda: None),
                metrics=metrics,
                dataset_name=args.project,
                instrument=False,  # Don't re-run, use existing outputs
                max_workers=getattr(args, "workers", 4),
            )
            eval_run = runner.run_dataset(items, use_synthetic=True)

            # Save run
            run_path = eval_dir / f"run_{timestamp}_{eval_run.id[:8]}.json"
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(eval_run.as_dict(), f, indent=2)

            print(f"  Evaluated {len(items)} items")
            print("  RESULTS:")
            for metric_id, summary in eval_run.summary.items():
                if "pass_rate" in summary:
                    print(f"    {metric_id}: pass_rate={summary['pass_rate']:.2f}")
                elif "avg" in summary:
                    print(f"    {metric_id}: avg={summary['avg']:.1f}")
            print(f"  Saved to: {run_path}\n")

            state["steps"]["3_initial_eval"] = {
                "status": "success",
                "output": str(run_path),
                "run_id": eval_run.id,
            }
            save_state_atomic(state)

        # Step 4: Human Annotation (optional)
        if step_done("4_annotation"):
            print("[4/7] Human Annotation")
            print("  Already completed (skipping)\n")
        elif args.skip_annotation:
            print("[4/7] Human Annotation")
            print("  SKIPPED (--skip-annotation)\n")
            state["steps"]["4_annotation"] = {"status": "skipped"}
            save_state_atomic(state)
        else:
            print("[4/7] Human Annotation")
            if args.dry_run:
                print(f"  -> Would annotate {args.annotation_limit} items")
                print(f"  -> Mode: {'per-metric' if args.per_metric else 'overall'}\n")
            else:
                ann_dir = output_dir / "4_annotations"
                ann_dir.mkdir(exist_ok=True)
                ann_path = ann_dir / "annotations.jsonl"

                print(f"  -> Annotating {args.annotation_limit} items...")
                print("  -> Interactive annotation mode")
                print("  -> Press Ctrl+C to skip this step\n")

                try:
                    # Call interactive annotation
                    ann_args = argparse.Namespace(
                        dataset=str(dataset_dir),
                        latest=False,
                        run_id=None,
                        output=str(ann_path),
                        annotator="human",
                        restart=False,
                        per_metric=args.per_metric,
                        only_disagreements=False,
                        spans=False,
                        span_type="all",
                    )
                    cmd_annotate(ann_args)

                    # Count annotations
                    if ann_path.exists():
                        with open(ann_path, encoding="utf-8") as f:
                            ann_count = sum(1 for _ in f)
                        print(f"  Completed {ann_count} annotations")
                        print(f"  Saved to: {ann_path}\n")
                        state["steps"]["4_annotation"] = {
                            "status": "success",
                            "output": str(ann_path),
                            "count": ann_count,
                        }
                        save_state_atomic(state)
                    else:
                        print("  No annotations created\n")
                        state["steps"]["4_annotation"] = {"status": "skipped"}
                        save_state_atomic(state)
                except KeyboardInterrupt:
                    print("\n  Annotation interrupted by user\n")
                    state["steps"]["4_annotation"] = {"status": "interrupted"}
                    save_state_atomic(state)

        # Step 5: Calibrate LLM Judges (optional)
        has_annotations = (output_dir / "4_annotations" / "annotations.jsonl").exists()

        if step_done("5_calibration"):
            print("[5/7] Calibrating LLM Judges")
            print("  Already completed (skipping)\n")
        elif args.skip_calibration or not has_annotations:
            print("[5/7] Calibrating LLM Judges")
            if args.skip_calibration:
                print("  SKIPPED (--skip-calibration)\n")
            else:
                print("  SKIPPED (requires annotations)\n")
            state["steps"]["5_calibration"] = {"status": "skipped"}
            save_state_atomic(state)
        else:
            print("[5/7] Calibrating LLM Judges")
            if args.dry_run:
                print("  -> Would calibrate subjective metrics")
                print(f"  -> Optimizer: {args.optimizer}\n")
            else:
                cal_dir = output_dir / "5_calibrations"
                cal_dir.mkdir(exist_ok=True)

                ann_path = output_dir / "4_annotations" / "annotations.jsonl"

                # Get subjective metrics
                subj_metrics = [
                    spec for spec in metric_specs if spec.type == "subjective"
                ]

                print(f"  -> Calibrating {len(subj_metrics)} subjective metrics...\n")

                for spec in subj_metrics:
                    print(f"  [{spec.id}]")

                    # Run calibration
                    cal_args = argparse.Namespace(
                        metric_id=spec.id,
                        annotations=str(ann_path),
                        run_id=None,
                        threshold=0.5,
                        dataset=str(dataset_dir),
                        latest=False,
                        no_optimize=False,
                        optimizer=args.optimizer,
                        model=args.model,
                        gepa_task_lm="gemini/gemini-2.5-flash",
                        gepa_reflection_lm="gemini/gemini-2.5-flash",
                        gepa_max_calls=150,
                        show_examples=False,
                        output=None,
                        format="table",
                    )

                    try:
                        cmd_calibrate(cal_args)
                        print()
                    except Exception as e:
                        print(f"    Calibration failed: {e}\n")

                state["steps"]["5_calibration"] = {
                    "status": "success",
                    "metrics_calibrated": len(subj_metrics),
                }
                save_state_atomic(state)

        # Step 6: Re-evaluate with Calibrated Prompts (optional)
        has_calibrations = (output_dir / "5_calibrations").exists()

        if step_done("6_calibrated_eval"):
            print("[6/7] Re-evaluating with Calibrated Prompts")
            print("  Already completed (skipping)\n")
        elif not has_calibrations:
            print("[6/7] Re-evaluating with Calibrated Prompts")
            print("  SKIPPED (no calibrations)\n")
            state["steps"]["6_calibrated_eval"] = {"status": "skipped"}
            save_state_atomic(state)
        else:
            print("[6/7] Re-evaluating with Calibrated Prompts")
            if args.dry_run:
                print("  -> Would re-run evaluation with calibrated prompts\n")
            else:
                eval2_dir = output_dir / "6_calibrated_eval"
                eval2_dir.mkdir(exist_ok=True)

                # Re-run evaluation with calibrated prompts
                from ...runner import EvalRunner
                from ...metrics.factory import (
                    build_objective_metric,
                    build_subjective_metric,
                )
                from ...annotation import load_optimized_prompt

                # Load metrics from file
                with open(metrics_path, encoding="utf-8") as f:
                    metrics_data = json.load(f)

                metrics = []
                calibrated_count = 0
                for spec_data in metrics_data:
                    spec = MetricSpec(
                        id=spec_data["id"],
                        name=spec_data.get("name", spec_data["id"]),
                        type=spec_data["type"],
                        description=spec_data.get("description", ""),
                        config=spec_data.get("config", {}),
                    )
                    # Load calibrated prompt for subjective metrics
                    if spec.type == "subjective":
                        optimized_prompt = load_optimized_prompt(
                            str(dataset_dir), spec.id
                        )
                        if optimized_prompt:
                            spec.config = dict(spec.config or {})
                            spec.config["prompt"] = optimized_prompt
                            calibrated_count += 1
                    # Get API key from config
                    pipeline_gemini_key2 = get_config_default(
                        config, "api_keys", "gemini"
                    )
                    try:
                        if spec.type == "objective":
                            m = build_objective_metric(spec.id, spec.config)
                        else:
                            m = build_subjective_metric(
                                spec.id, spec.config, api_key=pipeline_gemini_key2
                            )
                        if m:
                            metrics.append(m)
                    except Exception:
                        pass

                items = list(load_dataset(dataset_path))
                runner = EvalRunner(
                    target_fn=target_fn or (lambda: None),
                    metrics=metrics,
                    dataset_name=args.project,
                    instrument=False,
                    max_workers=getattr(args, "workers", 4),
                )
                eval_run2 = runner.run_dataset(items, use_synthetic=True)

                # Save run
                run_path2 = eval2_dir / f"run_{timestamp}_{eval_run2.id[:8]}.json"
                with open(run_path2, "w", encoding="utf-8") as f:
                    json.dump(eval_run2.as_dict(), f, indent=2)

                print(f"  Used {calibrated_count} calibrated prompts")
                print(f"  Evaluated {len(items)} items")
                print("  RESULTS:")
                for metric_id, summary in eval_run2.summary.items():
                    if "pass_rate" in summary:
                        print(f"    {metric_id}: pass_rate={summary['pass_rate']:.2f}")
                print(f"  Saved to: {run_path2}\n")

                state["steps"]["6_calibrated_eval"] = {
                    "status": "success",
                    "output": str(run_path2),
                    "calibrated_count": calibrated_count,
                }
                save_state_atomic(state)

        # Step 7: Generate Simulations (optional)
        if step_done("7_simulation"):
            print("[7/7] Generating Simulations")
            print("  Already completed (skipping)\n")
        elif not args.enable_simulation:
            print("[7/7] Generating Simulations")
            print("  SKIPPED (use --enable-simulation to enable)\n")
            state["steps"]["7_simulation"] = {"status": "skipped"}
            save_state_atomic(state)
        elif not args.target:
            print("[7/7] Generating Simulations")
            print("  SKIPPED (--target required for simulation)\n")
            state["steps"]["7_simulation"] = {
                "status": "skipped",
                "reason": "no target",
            }
            save_state_atomic(state)
        else:
            print("[7/7] Generating Simulations")
            if args.dry_run:
                print(
                    f"  -> Would generate simulations: modes={args.simulation_modes}\n"
                )
            else:
                sim_dir = output_dir / "7_simulations"
                sim_dir.mkdir(exist_ok=True)

                # Run simulation
                sim_args = argparse.Namespace(
                    dataset=str(dataset_dir),
                    target=args.target,
                    output=str(sim_dir),
                    modes=args.simulation_modes,
                    num_similar=args.num_similar,
                    num_outlier=args.num_outlier,
                    max_seeds=args.max_sim_seeds,
                    model=args.model,
                    temp_similar=0.3,
                    temp_outlier=0.8,
                )

                try:
                    cmd_simulate(sim_args)
                    print("  Simulations generated")
                    print(f"  Saved to: {sim_dir}/\n")
                    state["steps"]["7_simulation"] = {
                        "status": "success",
                        "output": str(sim_dir),
                    }
                    save_state_atomic(state)
                except Exception as e:
                    print(f"  Simulation failed: {e}\n")
                    state["steps"]["7_simulation"] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    save_state_atomic(state)

        # Save final pipeline state (also saved as pipeline_summary.json for backwards compat)
        state["completed_at"] = datetime.now().isoformat()
        summary_path = output_dir / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Final summary
        print("\n" + "=" * 70)
        print(" " * 20 + "PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        print("\nSummary:")
        for step_name, step_info in state["steps"].items():
            status = step_info.get("status", "unknown")
            status_icon = (
                "[OK]"
                if status == "success"
                else ("[SKIP]" if status == "skipped" else "[FAIL]")
            )
            print(f"  {status_icon} {step_name}: {status}")

        print("\nNext steps:")
        if (
            "6_calibrated_eval" in state["steps"]
            and state["steps"]["6_calibrated_eval"]["status"] == "success"
        ):
            print(
                f"  1. Review results: cat {state['steps']['6_calibrated_eval']['output']}"
            )
        elif "3_initial_eval" in state["steps"]:
            print(
                f"  1. Review results: cat {state['steps']['3_initial_eval']['output']}"
            )
        print(f"  2. View full summary: cat {summary_path}")
        print()

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        save_state_atomic(state)  # Save current state
        print(f"Progress saved to: {state_path}")
        print(
            f"Resume with: evalyn one-click --project {args.project} --output-dir {output_dir} --resume\n"
        )
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        save_state_atomic(state)  # Save current state
        print(f"Progress saved to: {state_path}")
        print(
            f"Resume with: evalyn one-click --project {args.project} --output-dir {output_dir} --resume\n"
        )
        import traceback

        if args.verbose:
            traceback.print_exc()


def cmd_workflow(args: argparse.Namespace) -> None:
    """Show the evaluation workflow and next steps."""
    workflow = """
EVALYN WORKFLOW
===============

The evaluation pipeline has 3 phases:

PHASE 1: COLLECT
----------------
  1. Add @eval decorator to your agent function
  2. Run your agent to collect traces
  3. Build a dataset from traces

  Commands:
    evalyn list-calls              # View captured traces
    evalyn show-projects           # See project summary
    evalyn build-dataset --project <name>  # Create dataset

PHASE 2: EVALUATE
-----------------
  4. Select metrics for evaluation
  5. Run evaluation
  6. Analyze results

  Commands:
    evalyn suggest-metrics --dataset <path> --mode basic
    evalyn run-eval --dataset <path>
    evalyn analyze --dataset <path>

PHASE 3: CALIBRATE (optional)
-----------------------------
  7. Annotate results (human feedback)
  8. Calibrate LLM judges
  9. Re-evaluate with calibrated prompts

  Commands:
    evalyn annotate --dataset <path>
    evalyn calibrate --dataset <path> --metric-id <id>
    evalyn run-eval --dataset <path> --use-calibrated

ONE-CLICK OPTION
----------------
  Run the entire pipeline automatically:
    evalyn one-click --project <name>

NEXT STEPS
----------
"""
    print(workflow)

    # Show context-aware next steps
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    if tracer.storage:
        calls = tracer.storage.list_calls(limit=10)
        if calls:
            projects = set()
            for call in calls:
                if isinstance(call.metadata, dict):
                    proj = call.metadata.get("project_id") or call.metadata.get(
                        "project"
                    )
                    if proj:
                        projects.add(proj)
            if projects:
                print(f"  You have traces for: {', '.join(sorted(projects))}")
                print(f"  Try: evalyn build-dataset --project {sorted(projects)[0]}")
            else:
                print("  You have traces. Try: evalyn build-dataset")
        else:
            print("  No traces yet. Add @eval decorator to your agent and run it.")
    else:
        print("  No storage configured. Run your @eval-decorated agent first.")


def register_commands(subparsers) -> None:
    """Register infrastructure commands."""
    # workflow
    workflow_parser = subparsers.add_parser(
        "workflow", help="Show evaluation workflow and next steps"
    )
    workflow_parser.set_defaults(func=cmd_workflow)

    # init
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--output",
        default="evalyn.yaml",
        help="Output path for config file (default: evalyn.yaml)",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing config file"
    )
    init_parser.set_defaults(func=cmd_init)

    # one-click
    oneclick_parser = subparsers.add_parser(
        "one-click",
        help="Run complete evaluation pipeline (dataset -> metrics -> eval -> annotate -> calibrate)",
    )
    oneclick_parser.add_argument(
        "--project", required=True, help="Project name to filter traces"
    )
    oneclick_parser.add_argument(
        "--target",
        help="Target function (file.py:func or module:func). Optional - if not provided, uses existing trace outputs",
    )
    oneclick_parser.add_argument(
        "--version", help="Version filter (default: all versions)"
    )
    oneclick_parser.add_argument(
        "--production-only", action="store_true", help="Use only production traces"
    )
    oneclick_parser.add_argument(
        "--simulation-only", action="store_true", help="Use only simulation traces"
    )
    oneclick_parser.add_argument(
        "--output-dir", help="Custom output directory (default: auto-generated)"
    )
    oneclick_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous incomplete run (auto-detects from output-dir)",
    )
    oneclick_parser.add_argument(
        "--dataset-limit",
        type=int,
        default=100,
        help="Max dataset items (default: 100)",
    )
    oneclick_parser.add_argument(
        "--since", help="Filter traces since date (ISO format)"
    )
    oneclick_parser.add_argument(
        "--until", help="Filter traces until date (ISO format)"
    )
    oneclick_parser.add_argument(
        "--metric-mode",
        choices=["basic", "llm-registry", "llm-brainstorm", "bundle", "all"],
        default="basic",
        help="Metric selection mode: basic (fast), llm-registry (LLM picks from templates), "
        "llm-brainstorm (LLM generates custom), bundle (preset), all (comprehensive - runs all modes)",
    )
    oneclick_parser.add_argument(
        "--llm-mode",
        choices=["api", "local"],
        default="api",
        help="LLM mode for metric selection (default: api)",
    )
    oneclick_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model name (default: gemini-2.5-flash-lite)",
    )
    oneclick_parser.add_argument("--bundle", help="Bundle name (if metric-mode=bundle)")
    oneclick_parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip annotation step (default: false)",
    )
    oneclick_parser.add_argument(
        "--annotation-limit",
        type=int,
        default=20,
        help="Max items to annotate (default: 20)",
    )
    oneclick_parser.add_argument(
        "--per-metric", action="store_true", help="Use per-metric annotation mode"
    )
    oneclick_parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration step (default: false)",
    )
    oneclick_parser.add_argument(
        "--optimizer",
        choices=["llm", "gepa"],
        default="llm",
        help="Prompt optimization method (default: llm)",
    )
    oneclick_parser.add_argument(
        "--calibrate-all-metrics",
        action="store_true",
        help="Calibrate all subjective metrics (default: only poorly-aligned)",
    )
    oneclick_parser.add_argument(
        "--enable-simulation",
        action="store_true",
        help="Enable simulation step (default: false)",
    )
    oneclick_parser.add_argument(
        "--simulation-modes",
        default="similar",
        help="Simulation modes: similar,outlier (default: similar)",
    )
    oneclick_parser.add_argument(
        "--num-similar",
        type=int,
        default=3,
        help="Similar queries per seed (default: 3)",
    )
    oneclick_parser.add_argument(
        "--num-outlier",
        type=int,
        default=2,
        help="Outlier queries per seed (default: 2)",
    )
    oneclick_parser.add_argument(
        "--max-sim-seeds",
        type=int,
        default=10,
        help="Max seeds for simulation (default: 10)",
    )
    oneclick_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Parallel workers for LLM evaluation (default: 4, max: 16)",
    )
    oneclick_parser.add_argument(
        "--auto-yes",
        action="store_true",
        help="Skip all confirmation prompts (default: false)",
    )
    oneclick_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed logs (default: false)"
    )
    oneclick_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing (default: false)",
    )
    oneclick_parser.set_defaults(func=cmd_one_click)


__all__ = ["cmd_init", "cmd_one_click", "register_commands"]
