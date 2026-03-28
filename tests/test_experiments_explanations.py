"""Tests for experiment tracking and metric score explanations."""

import pytest
from datetime import datetime, timezone, timedelta
from evalyn_sdk.models import EvalRun, MetricResult, MetricSpec
from evalyn_sdk.experiments import (
    group_runs_by_experiment,
    compare_experiments,
    ExperimentComparison,
    _get_experiment_id,
)
from evalyn_sdk.metrics.explanations import (
    explain_metric_result,
    register_explainer,
    list_explainable_metrics,
)


def _run(run_id, tags=None, name=None, results=None, created_offset_hours=0):
    now = datetime.now(timezone.utc) + timedelta(hours=created_offset_hours)
    return EvalRun(
        id=run_id,
        dataset_name="test",
        created_at=now,
        metric_results=results or [],
        tags=tags or [],
        name=name,
    )


def _result(metric_id, passed=True, score=1.0, details=None):
    return MetricResult(
        metric_id=metric_id, item_id="i1", call_id="c1",
        score=score, passed=passed, details=details or {},
    )


# =============================================================================
# Experiment tracking tests
# =============================================================================

class TestGetExperimentId:
    def test_from_tag(self):
        run = _run("r1", tags=["experiment:prompt-v2"])
        assert _get_experiment_id(run) == "prompt-v2"

    def test_from_exp_tag(self):
        run = _run("r1", tags=["exp:baseline"])
        assert _get_experiment_id(run) == "baseline"

    def test_from_name(self):
        run = _run("r1", name="experiment-prompt-v2")
        assert _get_experiment_id(run) == "experiment-prompt-v2"

    def test_no_experiment(self):
        run = _run("r1", tags=["nightly", "ci"])
        assert _get_experiment_id(run) is None

    def test_empty(self):
        run = _run("r1")
        assert _get_experiment_id(run) is None


class TestGroupByExperiment:
    def test_basic_grouping(self):
        runs = [
            _run("r1", tags=["experiment:v1"]),
            _run("r2", tags=["experiment:v2"]),
            _run("r3", tags=["experiment:v1"]),
        ]
        groups = group_runs_by_experiment(runs)
        assert len(groups) == 2
        assert len(groups["v1"]) == 2
        assert len(groups["v2"]) == 1

    def test_no_experiments(self):
        runs = [_run("r1", tags=["other"])]
        assert group_runs_by_experiment(runs) == {}

    def test_empty(self):
        assert group_runs_by_experiment([]) == {}


class TestCompareExperiments:
    def test_b_wins(self):
        results_a = [_result("help", True, 1.0), _result("help", False, 0.0)]
        results_b = [_result("help", True, 1.0), _result("help", True, 1.0)]
        comp = compare_experiments(
            [_run("r1", results=results_a)],
            [_run("r2", results=results_b)],
            "baseline", "improved",
        )
        assert comp.winner == "improved"
        assert comp.metric_deltas["help"] > 0

    def test_a_wins(self):
        results_a = [_result("help", True)] * 5
        results_b = [_result("help", False)] * 5
        comp = compare_experiments(
            [_run("r1", results=results_a)],
            [_run("r2", results=results_b)],
            "good", "bad",
        )
        assert comp.winner == "good"

    def test_tie(self):
        results = [_result("help", True)]
        comp = compare_experiments(
            [_run("r1", results=results)],
            [_run("r2", results=results)],
            "a", "b",
        )
        assert comp.winner is None

    def test_empty_runs(self):
        comp = compare_experiments([], [], "a", "b")
        assert comp.metric_deltas == {}
        assert comp.winner is None

    def test_format_text(self):
        results_a = [_result("help", True)]
        results_b = [_result("help", False)]
        comp = compare_experiments(
            [_run("r1", results=results_a)],
            [_run("r2", results=results_b)],
            "v1", "v2",
        )
        text = comp.format_text()
        assert "v1 vs v2" in text
        assert "help" in text

    def test_as_dict(self):
        comp = compare_experiments([], [], "a", "b")
        d = comp.as_dict()
        assert d["experiment_a"] == "a"
        assert d["experiment_b"] == "b"


# =============================================================================
# Metric explanation tests
# =============================================================================

class TestExplainMetricResult:
    def test_json_valid_pass(self):
        r = _result("json_valid", True, 1.0)
        explanation = explain_metric_result(r)
        assert "PASS" in explanation
        assert "valid JSON" in explanation

    def test_json_valid_fail(self):
        r = _result("json_valid", False, 0.0)
        explanation = explain_metric_result(r)
        assert "FAIL" in explanation

    def test_nonempty_pass(self):
        r = _result("nonempty_output", True, 1.0)
        assert "PASS" in explain_metric_result(r)

    def test_nonempty_fail(self):
        r = _result("nonempty_output", False, 0.0)
        assert "FAIL" in explain_metric_result(r)
        assert "empty" in explain_metric_result(r).lower()

    def test_latency(self):
        r = _result("latency_ms", None, 250.0)
        assert "250" in explain_metric_result(r)

    def test_bleu_good(self):
        r = _result("bleu", True, 0.75)
        assert "GOOD" in explain_metric_result(r)

    def test_bleu_low(self):
        r = _result("bleu", False, 0.1)
        assert "LOW" in explain_metric_result(r)

    def test_prompt_injection_pass(self):
        r = _result("prompt_injection_check", True, 1.0)
        assert "PASS" in explain_metric_result(r)

    def test_prompt_injection_fail(self):
        r = _result("prompt_injection_check", False, 0.0,
                    details={"matches": ["ignore all previous"]})
        explanation = explain_metric_result(r)
        assert "FAIL" in explanation
        assert "ignore" in explanation

    def test_tool_call_accuracy(self):
        r = _result("tool_call_accuracy", True, 0.75,
                    details={"matched": 3, "total_expected": 4})
        explanation = explain_metric_result(r)
        assert "3/4" in explanation

    def test_tool_call_accuracy_na(self):
        r = _result("tool_call_accuracy", None, None)
        assert "N/A" in explain_metric_result(r)

    def test_generic_fallback(self):
        r = _result("custom_metric_xyz", True, 0.85)
        explanation = explain_metric_result(r)
        assert "custom_metric_xyz" in explanation
        assert "PASS" in explanation

    def test_generic_with_reason(self):
        r = _result("some_metric", False, 0.3, details={"reason": "output too short"})
        assert "output too short" in explain_metric_result(r)


class TestRegisterExplainer:
    def test_custom_explainer(self):
        register_explainer("my_custom", lambda r: f"Custom: {r.score}")
        r = _result("my_custom", True, 0.9)
        assert "Custom: 0.9" in explain_metric_result(r)

    def test_list_explainable(self):
        metrics = list_explainable_metrics()
        assert "json_valid" in metrics
        assert "prompt_injection_check" in metrics
        assert len(metrics) >= 10
