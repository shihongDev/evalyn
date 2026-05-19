"""Public Python API for running evaluations programmatically.

Usage:
    import evalyn_sdk as evalyn

    # Quick evaluation
    results = evalyn.api.evaluate(
        dataset="data/prod/datasets/my_project/",
        metrics=["helpfulness_accuracy", "json_valid"],
    )
    print(results.overall_pass_rate)
    print(results.metric_summary)

    # Analyze an existing run
    analysis = evalyn.api.analyze(run)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .analysis.core import RunAnalysis, analyze_run
from .datasets import load_dataset
from .models import DatasetItem, EvalRun


@dataclass
class EvalResult:
    """Result from a programmatic evaluation."""

    run: EvalRun
    analysis: RunAnalysis

    @property
    def overall_pass_rate(self) -> float:
        return self.analysis.overall_pass_rate

    @property
    def metric_summary(self) -> dict[str, dict[str, Any]]:
        """Per-metric summary: {metric_id: {pass_rate, avg_score, count}}."""
        summary = {}
        for mid, stats in self.analysis.metric_stats.items():
            summary[mid] = {
                "pass_rate": stats.pass_rate,
                "avg_score": stats.avg_score,
                "count": stats.count,
                "passed": stats.passed,
                "failed": stats.failed,
            }
        return summary

    @property
    def failed_items(self) -> list[str]:
        return self.analysis.failed_items

    @property
    def total_items(self) -> int:
        return self.analysis.total_items

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run.id,
            "overall_pass_rate": round(self.overall_pass_rate, 4),
            "total_items": self.total_items,
            "failed_items_count": len(self.failed_items),
            "metrics": self.metric_summary,
        }


def evaluate(
    dataset: str | Path | list[DatasetItem],
    metrics: str | Path | list[str] | list = None,
    *,
    provider: str = "gemini",
    max_workers: int = 1,
    name: str | None = None,
    tags: list[str] | None = None,
) -> EvalResult:
    """Run evaluation on a dataset with specified metrics.

    This is the primary programmatic API. It loads data, builds metrics,
    runs evaluation, and returns structured results. No sys.exit, no print.

    Args:
        dataset: Path to dataset directory/file, or list of DatasetItem objects.
        metrics: Metric specifications. Can be:
            - Path to metrics JSON file
            - List of metric ID strings (looked up from registry)
            - List of Metric objects
            - None (uses all available objective metrics)
        provider: LLM provider for subjective metrics.
        max_workers: Parallel evaluation workers.
        name: Optional run name.
        tags: Optional run tags.

    Returns:
        EvalResult with run data and analysis.

    Raises:
        FileNotFoundError: If dataset path doesn't exist.
        ValueError: If metrics can't be resolved.
    """
    # Load dataset
    items = _resolve_dataset(dataset)
    if not items:
        raise ValueError("Dataset is empty")

    # Build metrics
    metric_objects = _resolve_metrics(metrics, provider)
    if not metric_objects:
        raise ValueError("No metrics to evaluate")

    # Run evaluation
    from .evaluation.runner import EvalRunner

    runner = EvalRunner(
        target_fn=lambda x: x,  # not used - items already have outputs
        metrics=metric_objects,
        max_workers=max_workers,
        instrument=False,
    )
    run = runner.run_dataset(items, use_synthetic=True)

    # Apply name and tags
    if name:
        run.name = name
    if tags:
        run.tags = tags

    # Analyze
    analysis = analyze_run(run.as_dict())

    return EvalResult(run=run, analysis=analysis)


def analyze(run: EvalRun | dict[str, Any]) -> RunAnalysis:
    """Analyze an evaluation run.

    Args:
        run: EvalRun object or dict from run.as_dict().

    Returns:
        RunAnalysis with per-metric and per-item statistics.
    """
    if isinstance(run, EvalRun):
        return analyze_run(run.as_dict())
    return analyze_run(run)


def _resolve_dataset(dataset) -> list[DatasetItem]:
    """Resolve dataset argument to a list of DatasetItem."""
    if isinstance(dataset, list):
        if all(isinstance(d, DatasetItem) for d in dataset):
            return dataset
        # List of dicts
        return [DatasetItem.from_payload(d) if isinstance(d, dict) else d for d in dataset]

    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.is_dir():
        dataset_file = path / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = path / "dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"No dataset.jsonl or dataset.json in {path}")
        return load_dataset(dataset_file)

    return load_dataset(path)


def _resolve_metrics(metrics, provider: str) -> list:
    """Resolve metrics argument to a list of Metric objects."""
    from .metrics.factory import build_metrics_from_specs, build_objective_metric

    if metrics is None:
        # Default: all objective metrics (no LLM cost)
        from .metrics.objective import OBJECTIVE_REGISTRY
        specs = [{"id": m["id"], "type": "objective"} for m in OBJECTIVE_REGISTRY[:10]]
        return build_metrics_from_specs(specs)

    if isinstance(metrics, (str, Path)):
        # Path to metrics JSON
        import json
        path = Path(metrics)
        with open(path, encoding="utf-8") as f:
            specs_data = json.load(f)
        return build_metrics_from_specs(specs_data, provider=provider)

    if isinstance(metrics, list):
        if not metrics:
            return []
        # Check if already Metric objects
        if all(hasattr(m, "spec") for m in metrics):
            return metrics
        # List of metric ID strings
        if all(isinstance(m, str) for m in metrics):
            specs = [{"id": m, "type": "objective"} for m in metrics]
            # Try objective first, then subjective
            built = []
            for mid in metrics:
                try:
                    m = build_objective_metric(mid, {})
                    if m:
                        built.append(m)
                        continue
                except (KeyError, Exception):
                    pass
                # Try as subjective
                specs_data = [{"id": mid, "type": "subjective"}]
                subj = build_metrics_from_specs(specs_data, provider=provider)
                built.extend(subj)
            return built

    return []
