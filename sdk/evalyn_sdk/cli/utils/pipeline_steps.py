"""Pipeline step implementations for one-click evaluation.

Contains the 7 steps of the evaluation pipeline:
1. BuildDatasetStep - Collect traces and build dataset
2. SuggestMetricsStep - Select metrics based on mode
3. InitialEvalStep - Run initial evaluation
4. AnnotationStep - Human annotation (optional)
5. CalibrationStep - Calibrate LLM judges (optional)
6. CalibratedEvalStep - Re-evaluate with calibrated prompts
7. SimulationStep - Generate synthetic data (optional)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pipeline import PipelineStep, StepResult
from .config import get_config_default
from .loaders import _load_callable


def _build_metrics_from_specs(
    metrics_data: List[Dict],
    gemini_key: Optional[str],
    calibrated_prompts: Optional[Dict[str, str]] = None,
) -> Tuple[List, int]:
    """Build metric instances from spec data.

    Args:
        metrics_data: List of metric spec dictionaries
        gemini_key: API key for Gemini LLM judges
        calibrated_prompts: Optional dict mapping metric_id to optimized prompt

    Returns:
        Tuple of (metrics list, calibrated count)
    """
    from ...models import MetricSpec
    from ...metrics.factory import build_objective_metric, build_subjective_metric

    metrics = []
    calibrated_count = 0
    calibrated_prompts = calibrated_prompts or {}

    for spec_data in metrics_data:
        spec = MetricSpec(
            id=spec_data["id"],
            name=spec_data.get("name", spec_data["id"]),
            type=spec_data["type"],
            description=spec_data.get("description", ""),
            config=spec_data.get("config", {}),
        )

        # Apply calibrated prompt if available
        if spec.type == "subjective" and spec.id in calibrated_prompts:
            spec.config = dict(spec.config or {})
            spec.config["prompt"] = calibrated_prompts[spec.id]
            calibrated_count += 1

        try:
            if spec.type == "objective":
                m = build_objective_metric(spec.id, spec.config)
            else:
                m = build_subjective_metric(spec.id, spec.config, api_key=gemini_key)
            if m:
                metrics.append(m)
        except Exception:
            pass

    return metrics, calibrated_count


class BuildDatasetStep(PipelineStep):
    """Step 1: Build dataset from traces."""

    name = "dataset"
    display_name = "Building Dataset"
    step_number = 1

    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> Tuple[StepResult, Dict[str, Any]]:
        from datetime import datetime as dt

        from ...datasets import build_dataset_from_storage, save_dataset_with_meta
        from ...storage import SQLiteStorage

        dataset_dir = output_dir / self.name
        dataset_dir.mkdir(exist_ok=True)

        # Parse date filters
        since = dt.fromisoformat(self.args.since) if self.args.since else None
        until = dt.fromisoformat(self.args.until) if self.args.until else None

        storage = SQLiteStorage()
        items = build_dataset_from_storage(
            storage,
            project_name=self.args.project,
            version=self.args.version,
            production_only=getattr(self.args, "production_only", False),
            simulation_only=getattr(self.args, "simulation_only", False),
            since=since,
            until=until,
            limit=self.args.dataset_limit,
            success_only=True,
            include_metadata=True,
        )

        if not items:
            print("  No traces found matching filters")
            return StepResult(status="failed", error="No traces found"), {}

        meta = {
            "project": self.args.project,
            "version": self.args.version or "all",
            "created_at": datetime.now().isoformat(),
            "filters": {
                "production_only": getattr(self.args, "production_only", False),
                "simulation_only": getattr(self.args, "simulation_only", False),
                "since": self.args.since,
                "until": self.args.until,
            },
            "item_count": len(items),
        }

        dataset_path = save_dataset_with_meta(items, dataset_dir, meta)

        # Validate output exists
        if not dataset_path.exists():
            return StepResult(status="failed", error="Dataset file was not created"), {}

        print(f"  Found {len(items)} items")
        print(f"  Saved to: {dataset_path}\n")

        return (
            StepResult(
                status="success",
                output=str(dataset_path),
                details={"item_count": len(items)},
            ),
            {"dataset_path": dataset_path, "dataset_dir": dataset_dir},
        )

    def dry_run_message(self, output_dir: Path) -> str:
        dataset_dir = output_dir / self.name
        return (
            f"  -> Would build dataset with: project={self.args.project}, "
            f"limit={self.args.dataset_limit}\n"
            f"  -> Would save to: {dataset_dir}/dataset.jsonl\n"
        )


class SuggestMetricsStep(PipelineStep):
    """Step 2: Suggest metrics based on mode."""

    name = "metrics"
    display_name = "Suggesting Metrics"
    step_number = 2

    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> Tuple[StepResult, Dict[str, Any]]:
        from ...storage import SQLiteStorage

        metrics_dir = output_dir / self.name
        metrics_dir.mkdir(exist_ok=True)
        metrics_path = metrics_dir / "metrics.json"

        # Load target function if provided
        target_fn = None
        if self.args.target:
            try:
                target_fn = _load_callable(self.args.target)
            except Exception as e:
                print(f"  Could not load target function: {self.args.target}")
                print(f"    Error: {e}")
                print("  -> Using trace-based suggestions only")

        # Get sample traces
        storage = SQLiteStorage()
        calls = storage.list_calls(limit=5)

        # Suggest metrics based on mode
        mode = getattr(self.args, "metric_mode", "basic")
        model = getattr(self.args, "model", "gemini-2.5-flash-lite")
        llm_mode = getattr(self.args, "llm_mode", "api")
        bundle = getattr(self.args, "bundle", None)

        metric_specs = self._suggest_metrics(
            mode, model, llm_mode, bundle, target_fn, calls
        )

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

        # Validate output exists
        if not metrics_path.exists():
            return StepResult(status="failed", error="Metrics file was not created"), {}

        obj_count = sum(1 for spec in metric_specs if spec.type == "objective")
        subj_count = sum(1 for spec in metric_specs if spec.type == "subjective")

        print(
            f"  Selected {len(metric_specs)} metrics "
            f"({obj_count} objective, {subj_count} subjective)"
        )
        for spec in metric_specs:
            print(f"    - {spec.id} ({spec.type})")
        print(f"  Saved to: {metrics_path}\n")

        return (
            StepResult(
                status="success",
                output=str(metrics_path),
                details={
                    "total": len(metric_specs),
                    "objective": obj_count,
                    "subjective": subj_count,
                },
            ),
            {
                "metrics_path": metrics_path,
                "metric_specs": metric_specs,
                "target_fn": target_fn,
            },
        )

    def _suggest_metrics(
        self,
        mode: str,
        model: str,
        llm_mode: str,
        bundle: Optional[str],
        target_fn: Any,
        calls: List,
    ) -> List:
        """Suggest metrics based on mode."""
        from ...metrics.suggester import (
            HeuristicSuggester,
            LLMSuggester,
            LLMRegistrySelector,
        )

        if mode == "all":
            return self._suggest_all_modes(model, llm_mode, bundle, target_fn, calls)
        elif mode == "basic":
            suggester = HeuristicSuggester()
            return suggester.suggest(target_fn, calls)
        elif mode == "llm-registry":
            selector = LLMRegistrySelector(model=model, llm_mode=llm_mode)
            return selector.select_metrics(target_fn, calls)
        elif mode == "llm-brainstorm":
            suggester = LLMSuggester(model=model, llm_mode=llm_mode)
            return suggester.suggest(target_fn, calls)
        else:  # bundle
            return self._get_bundle_metrics(bundle) if bundle else []

    def _get_bundle_metrics(self, bundle_name: str) -> List:
        """Get metrics for a bundle by name."""
        from ...models import MetricSpec
        from ...metrics.objective import OBJECTIVE_REGISTRY
        from ...metrics.subjective import SUBJECTIVE_REGISTRY
        from ..constants import BUNDLES

        bundle = bundle_name.lower()
        ids = BUNDLES.get(bundle, [])
        if not ids:
            return []

        all_templates = OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY
        tpl_map = {t["id"]: t for t in all_templates}
        specs = []
        for mid in ids:
            tpl = tpl_map.get(mid)
            if tpl:
                specs.append(
                    MetricSpec(
                        id=tpl["id"],
                        name=tpl["id"],
                        type=tpl["type"],
                        description=tpl.get("description", ""),
                        config=tpl.get("config", {}),
                    )
                )
        return specs

    def _suggest_all_modes(
        self,
        model: str,
        llm_mode: str,
        bundle: Optional[str],
        target_fn: Any,
        calls: List,
    ) -> List:
        """Comprehensive mode: run all modes and merge."""
        from ...metrics.suggester import (
            HeuristicSuggester,
            LLMRegistrySelector,
        )

        all_metrics = []
        seen_ids = set()

        # 1. Basic mode
        print("    -> Running basic mode...")
        suggester = HeuristicSuggester()
        for spec in suggester.suggest(target_fn, calls):
            if spec.id not in seen_ids:
                all_metrics.append(spec)
                seen_ids.add(spec.id)

        # 2. LLM-registry mode
        print("    -> Running llm-registry mode...")
        try:
            selector = LLMRegistrySelector(model=model, llm_mode=llm_mode)
            for spec in selector.select_metrics(target_fn, calls):
                if spec.id not in seen_ids:
                    all_metrics.append(spec)
                    seen_ids.add(spec.id)
        except Exception as e:
            print(f"    LLM-registry failed: {e}")

        # 3. Bundle mode
        if bundle:
            print(f"    -> Adding bundle: {bundle}...")
            for spec in self._get_bundle_metrics(bundle):
                if spec.id not in seen_ids:
                    all_metrics.append(spec)
                    seen_ids.add(spec.id)

        print(f"    -> Merged {len(all_metrics)} unique metrics")
        return all_metrics

    def dry_run_message(self, output_dir: Path) -> str:
        metrics_dir = output_dir / self.name
        mode = getattr(self.args, "metric_mode", "basic")
        return (
            f"  -> Would suggest metrics with mode={mode}\n"
            f"  -> Would save to: {metrics_dir}/metrics.json\n"
        )


class InitialEvalStep(PipelineStep):
    """Step 3: Run initial evaluation."""

    name = "initial_eval"
    display_name = "Running Initial Evaluation"
    step_number = 3

    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> Tuple[StepResult, Dict[str, Any]]:
        from ...datasets import load_dataset
        from ...runner import EvalRunner

        eval_dir = output_dir / self.name
        eval_dir.mkdir(exist_ok=True)

        metrics_path = context.get("metrics_path")
        dataset_path = context.get("dataset_path")
        target_fn = context.get("target_fn")

        if not metrics_path or not dataset_path:
            return StepResult(status="failed", error="Missing metrics or dataset"), {}

        # Load metrics
        with open(metrics_path, encoding="utf-8") as f:
            metrics_data = json.load(f)

        gemini_key = get_config_default(self.config, "api_keys", "gemini")
        metrics, _ = _build_metrics_from_specs(metrics_data, gemini_key)

        # Run evaluation
        items = list(load_dataset(dataset_path))
        runner = EvalRunner(
            target_fn=target_fn or (lambda: None),
            metrics=metrics,
            dataset_name=self.args.project,
            instrument=False,
            max_workers=getattr(self.args, "workers", 4),
        )
        eval_run = runner.run_dataset(items, use_synthetic=True)

        # Save run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = eval_dir / f"run_{timestamp}_{eval_run.id[:8]}.json"
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(eval_run.as_dict(), f, indent=2)

        # Validate output exists
        if not run_path.exists():
            return StepResult(
                status="failed", error="Eval run file was not created"
            ), {}

        print(f"  Evaluated {len(items)} items")
        print("  RESULTS:")
        for metric_id, summary in eval_run.summary.items():
            if "pass_rate" in summary:
                print(f"    {metric_id}: pass_rate={summary['pass_rate']:.2f}")
            elif "avg" in summary:
                print(f"    {metric_id}: avg={summary['avg']:.1f}")
        print(f"  Saved to: {run_path}\n")

        return (
            StepResult(
                status="success",
                output=str(run_path),
                details={"run_id": eval_run.id},
            ),
            {"eval_run_path": run_path},
        )

    def dry_run_message(self, output_dir: Path) -> str:
        eval_dir = output_dir / self.name
        return (
            f"  -> Would run evaluation on {self.args.dataset_limit} items\n"
            f"  -> Would save to: {eval_dir}/\n"
        )


class AnnotationStep(PipelineStep):
    """Step 4: Human annotation (optional)."""

    name = "annotation"
    display_name = "Human Annotation"
    step_number = 4

    def should_skip(self) -> Optional[str]:
        if getattr(self.args, "skip_annotation", False):
            return "--skip-annotation"
        return None

    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> Tuple[StepResult, Dict[str, Any]]:
        from ..commands.annotation import cmd_annotate

        ann_dir = output_dir / "annotations"
        ann_dir.mkdir(exist_ok=True)
        ann_path = ann_dir / "annotations.jsonl"
        dataset_dir = context.get("dataset_dir", output_dir / "dataset")

        limit = getattr(self.args, "annotation_limit", 20)
        per_metric = getattr(self.args, "per_metric", False)

        print(f"  -> Annotating {limit} items...")
        print("  -> Interactive annotation mode")
        print("  -> Press Ctrl+C to skip this step\n")

        try:
            ann_args = argparse.Namespace(
                dataset=str(dataset_dir),
                latest=False,
                run_id=None,
                output=str(ann_path),
                annotator="human",
                restart=False,
                per_metric=per_metric,
                only_disagreements=False,
                spans=False,
                span_type="all",
            )
            cmd_annotate(ann_args)

            if ann_path.exists():
                with open(ann_path, encoding="utf-8") as f:
                    ann_count = sum(1 for _ in f)
                print(f"  Completed {ann_count} annotations")
                print(f"  Saved to: {ann_path}\n")
                return (
                    StepResult(
                        status="success",
                        output=str(ann_path),
                        details={"count": ann_count},
                    ),
                    {"annotation_path": ann_path},
                )
            else:
                print("  No annotations created\n")
                return StepResult(status="skipped"), {}

        except KeyboardInterrupt:
            print("\n  Annotation interrupted by user\n")
            return StepResult(status="interrupted"), {}

    def dry_run_message(self, output_dir: Path) -> str:
        limit = getattr(self.args, "annotation_limit", 20)
        per_metric = getattr(self.args, "per_metric", False)
        mode = "per-metric" if per_metric else "overall"
        return f"  -> Would annotate {limit} items\n  -> Mode: {mode}\n"


class CalibrationStep(PipelineStep):
    """Step 5: Calibrate LLM judges (optional)."""

    name = "calibration"
    display_name = "Calibrating LLM Judges"
    step_number = 5

    def should_skip(self) -> Optional[str]:
        if getattr(self.args, "skip_calibration", False):
            return "--skip-calibration"
        return None

    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> Tuple[StepResult, Dict[str, Any]]:
        from ..commands.calibration import cmd_calibrate

        ann_path = output_dir / "annotations" / "annotations.jsonl"
        if not ann_path.exists():
            print("  SKIPPED (requires annotations)\n")
            return StepResult(
                status="skipped", details={"reason": "no annotations"}
            ), {}

        cal_dir = output_dir / "calibrations"
        cal_dir.mkdir(exist_ok=True)
        dataset_dir = context.get("dataset_dir", output_dir / "dataset")
        metric_specs = context.get("metric_specs", [])

        # Get subjective metrics
        subj_metrics = [spec for spec in metric_specs if spec.type == "subjective"]

        if not subj_metrics:
            print("  SKIPPED (no subjective metrics)\n")
            return StepResult(
                status="skipped", details={"reason": "no subjective metrics"}
            ), {}

        print(f"  -> Calibrating {len(subj_metrics)} subjective metrics...\n")

        optimizer = getattr(self.args, "optimizer", "basic")
        model = getattr(self.args, "model", "gemini-2.5-flash-lite")

        calibrated_metrics = []
        failed_metrics = []

        for spec in subj_metrics:
            print(f"  [{spec.id}]")
            cal_args = argparse.Namespace(
                metric_id=spec.id,
                annotations=str(ann_path),
                run_id=None,
                threshold=0.5,
                dataset=str(dataset_dir),
                latest=False,
                no_optimize=False,
                optimizer=optimizer,
                model=model,
                # GEPA settings
                gepa_task_lm="gemini/gemini-2.5-flash",
                gepa_reflection_lm="gemini/gemini-2.5-flash",
                gepa_max_calls=150,
                # OPRO settings
                opro_optimizer_model="gemini-2.5-flash",
                opro_scorer_model="gemini-2.5-flash-lite",
                opro_iterations=10,
                opro_candidates=4,
                # APE settings
                ape_candidates=10,
                ape_rounds=5,
                ape_samples=5,
                show_examples=False,
                output=None,
                format="table",
            )
            try:
                cmd_calibrate(cal_args)
                calibrated_metrics.append(spec.id)
                print()
            except Exception as e:
                failed_metrics.append({"id": spec.id, "error": str(e)})
                print(f"    Calibration failed: {e}\n")

        # Report summary
        if calibrated_metrics:
            print(f"  Successfully calibrated: {', '.join(calibrated_metrics)}")
        if failed_metrics:
            print(
                f"  Failed to calibrate: {', '.join(m['id'] for m in failed_metrics)}"
            )

        # Determine status based on results
        if not calibrated_metrics and failed_metrics:
            return (
                StepResult(
                    status="failed",
                    error=f"All {len(failed_metrics)} calibrations failed",
                    details={
                        "calibrated_metrics": [],
                        "failed_metrics": failed_metrics,
                    },
                ),
                {"calibrated_metrics": []},
            )

        return (
            StepResult(
                status="success",
                details={
                    "metrics_calibrated": len(calibrated_metrics),
                    "calibrated_metrics": calibrated_metrics,
                    "failed_metrics": failed_metrics,
                },
            ),
            {"calibrated_metrics": calibrated_metrics},
        )

    def dry_run_message(self, output_dir: Path) -> str:
        optimizer = getattr(self.args, "optimizer", "basic")
        return f"  -> Would calibrate subjective metrics\n  -> Optimizer: {optimizer}\n"


class CalibratedEvalStep(PipelineStep):
    """Step 6: Re-evaluate with calibrated prompts."""

    name = "calibrated_eval"
    display_name = "Re-evaluating with Calibrated Prompts"
    step_number = 6

    def should_skip(self) -> Optional[str]:
        return None  # Will check for calibrations in execute

    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> Tuple[StepResult, Dict[str, Any]]:
        from ...datasets import load_dataset
        from ...runner import EvalRunner

        cal_dir = output_dir / "calibrations"
        if not cal_dir.exists():
            print("  SKIPPED (no calibrations)\n")
            return StepResult(
                status="skipped", details={"reason": "no calibrations"}
            ), {}

        eval_dir = output_dir / self.name
        eval_dir.mkdir(exist_ok=True)

        metrics_path = context.get("metrics_path")
        dataset_path = context.get("dataset_path")
        dataset_dir = context.get("dataset_dir", output_dir / "dataset")
        target_fn = context.get("target_fn")

        if not metrics_path or not dataset_path:
            return StepResult(status="failed", error="Missing metrics or dataset"), {}

        # Load metrics with calibrated prompts
        from ...annotation import load_optimized_prompt

        with open(metrics_path, encoding="utf-8") as f:
            metrics_data = json.load(f)

        # Build calibrated prompts dict
        calibrated_prompts = {}
        for spec_data in metrics_data:
            if spec_data.get("type") == "subjective":
                prompt = load_optimized_prompt(str(dataset_dir), spec_data["id"])
                if prompt:
                    calibrated_prompts[spec_data["id"]] = prompt

        gemini_key = get_config_default(self.config, "api_keys", "gemini")
        metrics, calibrated_count = _build_metrics_from_specs(
            metrics_data, gemini_key, calibrated_prompts
        )

        # Run evaluation
        items = list(load_dataset(dataset_path))
        runner = EvalRunner(
            target_fn=target_fn or (lambda: None),
            metrics=metrics,
            dataset_name=self.args.project,
            instrument=False,
            max_workers=getattr(self.args, "workers", 4),
        )
        eval_run = runner.run_dataset(items, use_synthetic=True)

        # Save run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = eval_dir / f"run_{timestamp}_{eval_run.id[:8]}.json"
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(eval_run.as_dict(), f, indent=2)

        # Validate output exists
        if not run_path.exists():
            return StepResult(
                status="failed", error="Calibrated eval run file was not created"
            ), {}

        print(f"  Used {calibrated_count} calibrated prompts")
        print(f"  Evaluated {len(items)} items")
        print("  RESULTS:")
        for metric_id, summary in eval_run.summary.items():
            if "pass_rate" in summary:
                print(f"    {metric_id}: pass_rate={summary['pass_rate']:.2f}")
        print(f"  Saved to: {run_path}\n")

        return (
            StepResult(
                status="success",
                output=str(run_path),
                details={"calibrated_count": calibrated_count},
            ),
            {},
        )

    def dry_run_message(self, output_dir: Path) -> str:
        return "  -> Would re-run evaluation with calibrated prompts\n"


class SimulationStep(PipelineStep):
    """Step 7: Generate simulations (optional)."""

    name = "simulation"
    display_name = "Generating Simulations"
    step_number = 7

    def should_skip(self) -> Optional[str]:
        if not getattr(self.args, "enable_simulation", False):
            return "use --enable-simulation to enable"
        if not self.args.target:
            return "--target required for simulation"
        return None

    def execute(
        self, output_dir: Path, context: Dict[str, Any]
    ) -> Tuple[StepResult, Dict[str, Any]]:
        from ..commands.simulate import cmd_simulate

        sim_dir = output_dir / "simulations"
        sim_dir.mkdir(exist_ok=True)
        dataset_dir = context.get("dataset_dir", output_dir / "dataset")

        sim_args = argparse.Namespace(
            dataset=str(dataset_dir),
            target=self.args.target,
            output=str(sim_dir),
            modes=getattr(self.args, "simulation_modes", "similar"),
            num_similar=getattr(self.args, "num_similar", 3),
            num_outlier=getattr(self.args, "num_outlier", 2),
            max_seeds=getattr(self.args, "max_sim_seeds", 10),
            model=getattr(self.args, "model", "gemini-2.5-flash-lite"),
            temp_similar=0.3,
            temp_outlier=0.8,
        )

        try:
            cmd_simulate(sim_args)
            print("  Simulations generated")
            print(f"  Saved to: {sim_dir}/\n")
            return (
                StepResult(status="success", output=str(sim_dir)),
                {},
            )
        except Exception as e:
            print(f"  Simulation failed: {e}\n")
            return (
                StepResult(status="failed", error=str(e)),
                {},
            )

    def dry_run_message(self, output_dir: Path) -> str:
        modes = getattr(self.args, "simulation_modes", "similar")
        return f"  -> Would generate simulations: modes={modes}\n"


def create_pipeline_steps(
    args: argparse.Namespace, config: Dict[str, Any]
) -> List[PipelineStep]:
    """Create all pipeline steps."""
    return [
        BuildDatasetStep(args, config),
        SuggestMetricsStep(args, config),
        InitialEvalStep(args, config),
        AnnotationStep(args, config),
        CalibrationStep(args, config),
        CalibratedEvalStep(args, config),
        SimulationStep(args, config),
    ]


__all__ = [
    "BuildDatasetStep",
    "SuggestMetricsStep",
    "InitialEvalStep",
    "AnnotationStep",
    "CalibrationStep",
    "CalibratedEvalStep",
    "SimulationStep",
    "create_pipeline_steps",
]
