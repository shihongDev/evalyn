#!/usr/bin/env python
"""Test calibration with different optimizers using the test fixtures."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "sdk"))

from evalyn_sdk.models import EvalRun, MetricResult, MetricSpec
from evalyn_sdk.calibration import CalibrationEngine, CalibrationConfig
from evalyn_sdk.annotation import import_annotations
from evalyn_sdk.datasets import load_dataset


def load_metric_results(path: Path) -> list[MetricResult]:
    """Load metric results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(MetricResult.from_dict(data))
    return results


def load_metric_spec(path: Path) -> list[MetricSpec]:
    """Load metric specs from JSON file."""
    with open(path) as f:
        data = json.load(f)
    specs = []
    for item in data:
        specs.append(MetricSpec(
            id=item["id"],
            name=item["id"],
            type="subjective",
            config=item.get("config", {}),
        ))
    return specs


def main():
    fixture_dir = Path(__file__).parent

    # Load test data
    print("Loading test fixtures...")
    dataset = load_dataset(fixture_dir / "dataset.jsonl")
    annotations = import_annotations(fixture_dir / "annotations.jsonl")
    metric_results = load_metric_results(fixture_dir / "metric_results.jsonl")
    metric_specs = load_metric_spec(fixture_dir / "metrics.json")

    print(f"  Dataset: {len(dataset)} items")
    print(f"  Annotations: {len(annotations)} items (PASS: {sum(1 for a in annotations if a.label)}, FAIL: {sum(1 for a in annotations if not a.label)})")
    print(f"  Metric results: {len(metric_results)} items")

    # Create mock eval run
    run = EvalRun(
        id="test-calibration-run",
        dataset_name="calibration-test",
        created_at=datetime.now(timezone.utc),
        metric_results=metric_results,
        metrics=metric_specs,
    )

    # Get rubric from metric spec
    rubric = metric_specs[0].config.get("rubric", [])
    preamble = metric_specs[0].config.get("prompt", "")

    print(f"\nMetric: {metric_specs[0].id}")
    print(f"Rubric: {rubric}")

    # Test CalibrationEngine directly (no LLM calls for alignment metrics)
    print("\n" + "="*60)
    print("TESTING CALIBRATION ENGINE (metrics only)")
    print("="*60)

    config = CalibrationConfig(
        judge_name="helpfulness",
        current_rubric=rubric,
        current_preamble=preamble,
        optimize_prompts=False,  # Start with metrics only
    )
    engine = CalibrationEngine(config)

    # Compute alignment metrics
    alignment = engine.compute_alignment(metric_results, annotations)
    print(f"\nAlignment Metrics:")
    print(f"  Total samples: {alignment.total}")
    print(f"  Accuracy: {alignment.accuracy:.1%}")
    print(f"  Precision: {alignment.precision:.1%}")
    print(f"  Recall: {alignment.recall:.1%}")
    print(f"  F1 Score: {alignment.f1:.1%}")
    print(f"  Cohen's Kappa: {alignment.cohens_kappa:.3f}")

    # Analyze disagreements
    disagreements = engine.analyze_disagreements(metric_results, annotations, dataset)
    print(f"\nDisagreements:")
    print(f"  False Positives (judge PASS, human FAIL): {len(disagreements.false_positives)}")
    print(f"  False Negatives (judge FAIL, human PASS): {len(disagreements.false_negatives)}")

    if disagreements.false_positives:
        print("\n  FP Examples:")
        for fp in disagreements.false_positives[:2]:
            print(f"    - {fp.input_text[:50]}...")

    if disagreements.false_negatives:
        print("\n  FN Examples:")
        for fn in disagreements.false_negatives[:2]:
            print(f"    - {fn.input_text[:50]}...")

    # Test optimizers if --test-optimizers flag is passed
    if len(sys.argv) > 1 and sys.argv[1] == "--test-optimizers":
        run_optimizer_tests(metric_specs[0].id, rubric, preamble, disagreements, alignment, dataset, metric_results, annotations)
    else:
        print("\n" + "="*60)
        print("OPTIMIZER TESTS")
        print("="*60)
        print("\nRun with --test-optimizers to test LLM-based optimizers:")
        print(f"  uv run python {__file__} --test-optimizers")


def run_optimizer_tests(metric_id, rubric, preamble, disagreements, alignment, dataset, metric_results, annotations):
    """Test each optimizer with actual LLM calls."""
    from evalyn_sdk.calibration import BasicOptimizer, OPROOptimizer, OPROConfig, APEOptimizer, APEConfig

    print("\n" + "="*60)
    print("TESTING BASIC OPTIMIZER")
    print("="*60)

    try:
        basic_opt = BasicOptimizer(model="gemini-2.5-flash-lite")
        result = basic_opt.optimize(metric_id, rubric, disagreements, alignment)
        print(f"\nOriginal rubric: {result.original_rubric}")
        print(f"Improved rubric: {result.improved_rubric}")
        print(f"Reasoning: {result.improvement_reasoning[:200]}...")
        print(f"Estimated improvement: {result.estimated_improvement}")
        print("Basic Optimizer: PASS")
    except Exception as e:
        print(f"Basic Optimizer: FAIL - {e}")

    print("\n" + "="*60)
    print("TESTING OPRO OPTIMIZER")
    print("="*60)

    try:
        opro_config = OPROConfig(
            max_iterations=2,  # Quick test
            candidates_per_step=2,
            trajectory_length=5,
        )
        opro_opt = OPROOptimizer(opro_config)
        result = opro_opt.optimize(
            metric_id=metric_id,
            current_rubric=rubric,
            metric_results=metric_results,
            annotations=annotations,
            dataset_items=dataset,
            current_preamble=preamble,
        )
        print(f"\nReasoning: {result.improvement_reasoning}")
        print(f"Estimated improvement: {result.estimated_improvement}")
        print("OPRO Optimizer: PASS")
    except Exception as e:
        print(f"OPRO Optimizer: FAIL - {e}")

    print("\n" + "="*60)
    print("TESTING APE OPTIMIZER")
    print("="*60)

    try:
        ape_config = APEConfig(
            num_candidates=3,  # Quick test
            eval_rounds=2,
            eval_samples_per_round=3,
        )
        ape_opt = APEOptimizer(ape_config)
        result = ape_opt.optimize(
            metric_id=metric_id,
            current_rubric=rubric,
            metric_results=metric_results,
            annotations=annotations,
            disagreements=disagreements,
            dataset_items=dataset,
            current_preamble=preamble,
        )
        print(f"\nReasoning: {result.improvement_reasoning}")
        print(f"Estimated improvement: {result.estimated_improvement}")
        print("APE Optimizer: PASS")
    except Exception as e:
        print(f"APE Optimizer: FAIL - {e}")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
