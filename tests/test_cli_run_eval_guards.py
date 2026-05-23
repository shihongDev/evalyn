"""Tests for run-eval production guards: --max-cost and --fail-on.

Covers:
- Threshold expression parsing (valid + invalid)
- BudgetTracker state machine (under, warning, hard-stop)
- evaluate_thresholds against a run-like object (pass + breach + missing)
- CLI smoke tests: flags appear in --help, --fail-on against an actual run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# conftest already inserts sdk/ into sys.path.
from evalyn_sdk.evaluation.guards import (
    BudgetExceededError,
    BudgetTracker,
    MissingMetricWarning,
    ThresholdBreach,
    ThresholdExpression,
    ThresholdReport,
    evaluate_thresholds,
    parse_threshold_expression,
)

from conftest import run_cli


# ===========================================================================
# Threshold expression parser
# ===========================================================================


class TestParseThresholdExpression:
    """parse_threshold_expression should accept all six operators and reject junk."""

    @pytest.mark.parametrize(
        "expr,expected_metric,expected_op,expected_value",
        [
            ("metric<0.8", "metric", "<", 0.8),
            ("metric<=0.8", "metric", "<=", 0.8),
            ("metric>0.5", "metric", ">", 0.5),
            ("metric>=0.5", "metric", ">=", 0.5),
            ("metric==1.0", "metric", "==", 1.0),
            ("metric!=0", "metric", "!=", 0.0),
            ("helpfulness_accuracy<0.8", "helpfulness_accuracy", "<", 0.8),
            ("toxicity.safety<=0.95", "toxicity.safety", "<=", 0.95),
            ("metric-with-dashes>0", "metric-with-dashes", ">", 0.0),
            ("  metric < 0.8  ", "metric", "<", 0.8),
        ],
    )
    def test_valid_expressions(self, expr, expected_metric, expected_op, expected_value):
        parsed = parse_threshold_expression(expr)
        assert parsed.metric_id == expected_metric
        assert parsed.op == expected_op
        assert parsed.threshold == pytest.approx(expected_value)

    @pytest.mark.parametrize(
        "expr",
        [
            "",
            "   ",
            "no_operator",
            "metric@0.5",          # bogus operator
            "<0.8",                # missing metric
            "metric<",             # missing value
            "metric<abc",          # non-numeric value
            "1metric<0.5",         # metric must start with letter/underscore
            "metric < 0.5 extra",  # trailing junk
        ],
    )
    def test_invalid_expressions(self, expr):
        with pytest.raises(ValueError):
            parse_threshold_expression(expr)

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            parse_threshold_expression(None)  # type: ignore[arg-type]

    def test_evaluate_returns_true_when_failure_condition_holds(self):
        # Expression describes the failure condition; evaluate is True when
        # the condition holds (meaning: the gate is breached).
        assert parse_threshold_expression("m<0.8").evaluate(0.5) is True
        assert parse_threshold_expression("m<0.8").evaluate(0.8) is False
        assert parse_threshold_expression("m<=0.8").evaluate(0.8) is True
        assert parse_threshold_expression("m>0.5").evaluate(0.6) is True
        assert parse_threshold_expression("m>=0.5").evaluate(0.5) is True
        assert parse_threshold_expression("m==1.0").evaluate(1.0) is True
        assert parse_threshold_expression("m!=0").evaluate(0.1) is True

    def test_is_breached_matches_failure_condition(self):
        # --fail-on metric<0.8 fires when the metric drops below 0.8.
        expr = parse_threshold_expression("m<0.8")
        assert expr.is_breached(0.5) is True   # 0.5 < 0.8 -> breach
        assert expr.is_breached(0.8) is False  # not strictly less -> ok
        assert expr.is_breached(0.9) is False  # well above floor -> ok


# ===========================================================================
# Budget tracker
# ===========================================================================


class TestBudgetTracker:
    def test_disabled_when_max_cost_is_none(self):
        tracker = BudgetTracker(max_cost=None)
        assert tracker.enabled is False
        status = tracker.record_cost(100.0)
        assert status.exceeded is False
        assert status.warning_triggered is False

    def test_invalid_max_cost(self):
        with pytest.raises(ValueError):
            BudgetTracker(max_cost=0)
        with pytest.raises(ValueError):
            BudgetTracker(max_cost=-1.0)

    def test_invalid_warning_fraction(self):
        with pytest.raises(ValueError):
            BudgetTracker(max_cost=1.0, warning_fraction=0)
        with pytest.raises(ValueError):
            BudgetTracker(max_cost=1.0, warning_fraction=1.5)

    def test_under_budget_no_warning(self):
        tracker = BudgetTracker(max_cost=10.0)
        status = tracker.record_cost(1.0)
        assert status.cumulative_cost == 1.0
        assert status.fraction_used == pytest.approx(0.1)
        assert status.warning_triggered is False
        assert status.exceeded is False

    def test_warning_fires_once_at_80_percent(self):
        tracker = BudgetTracker(max_cost=10.0)
        tracker.record_cost(7.0)  # 70% - under
        status_at_warning = tracker.record_cost(1.5)  # 85% - crosses warning
        assert status_at_warning.warning_triggered is True
        assert status_at_warning.exceeded is False

        status_after_warning = tracker.record_cost(0.1)  # still warning band
        assert status_after_warning.warning_triggered is False  # only once

    def test_hard_stop_at_100_percent(self):
        tracker = BudgetTracker(max_cost=10.0)
        status = tracker.record_cost(10.0)
        assert status.exceeded is True

    def test_exact_warning_threshold(self):
        tracker = BudgetTracker(max_cost=10.0)
        status = tracker.record_cost(8.0)  # exactly 80%
        assert status.warning_triggered is True

    def test_negative_cost_clamped(self):
        tracker = BudgetTracker(max_cost=10.0)
        status = tracker.record_cost(-5.0)
        assert status.cumulative_cost == 0.0

    def test_record_result_without_model_is_noop(self):
        tracker = BudgetTracker(max_cost=10.0)
        result = SimpleNamespace(model=None, input_tokens=100, output_tokens=50)
        status = tracker.record_result(result)
        assert status.cumulative_cost == 0.0

    def test_record_result_with_known_model(self):
        # Use a known pricing entry to avoid the warning path.
        tracker = BudgetTracker(max_cost=10.0)
        result = SimpleNamespace(
            model="gemini-2.5-flash", input_tokens=1_000_000, output_tokens=0
        )
        status = tracker.record_result(result)
        assert status.cumulative_cost > 0


class TestBudgetExceededError:
    def test_carries_costs(self):
        err = BudgetExceededError(current_cost=5.5, max_cost=5.0)
        assert err.current_cost == 5.5
        assert err.max_cost == 5.0
        assert err.partial_results == []
        assert "5.0" in str(err)


# ===========================================================================
# evaluate_thresholds against a run summary
# ===========================================================================


def _fake_run(metric_means: dict[str, float]):
    """Build a minimal run-like object with summary.metrics[id].avg_score."""
    return SimpleNamespace(
        summary={"metrics": {mid: {"avg_score": v} for mid, v in metric_means.items()}}
    )


class TestEvaluateThresholds:
    def test_all_pass(self):
        # "acc<0.8" fires when acc < 0.8; acc=0.9 -> safe.
        run = _fake_run({"acc": 0.9, "safety": 0.99})
        exprs = [
            parse_threshold_expression("acc<0.8"),
            parse_threshold_expression("safety<0.95"),
        ]
        report = evaluate_thresholds(exprs, run)
        assert report.any_breached is False
        assert report.breaches == []
        assert report.missing == []

    def test_breach_recorded(self):
        run = _fake_run({"acc": 0.5})
        # "acc<0.8" -> 0.5 < 0.8 -> breach
        exprs = [parse_threshold_expression("acc<0.8")]
        report = evaluate_thresholds(exprs, run)
        assert report.any_breached is True
        assert len(report.breaches) == 1
        breach = report.breaches[0]
        assert breach.expression.metric_id == "acc"
        assert breach.actual_score == pytest.approx(0.5)
        assert "acc" in breach.format()

    def test_missing_metric_is_warning_not_breach(self):
        run = _fake_run({"acc": 0.9})
        exprs = [parse_threshold_expression("nonexistent<0.5")]
        report = evaluate_thresholds(exprs, run)
        assert report.any_breached is False
        assert len(report.missing) == 1
        assert "nonexistent" in report.missing[0].format()

    def test_mixed_pass_breach_missing(self):
        run = _fake_run({"good": 0.99, "bad": 0.10})
        exprs = [
            parse_threshold_expression("good<0.9"),    # 0.99 < 0.9? no -> pass
            parse_threshold_expression("bad<0.8"),     # 0.10 < 0.8 -> breach
            parse_threshold_expression("missing<1"),   # missing
        ]
        report = evaluate_thresholds(exprs, run)
        assert len(report.breaches) == 1
        assert report.breaches[0].expression.metric_id == "bad"
        assert len(report.missing) == 1
        assert report.missing[0].expression.metric_id == "missing"

    def test_accepts_raw_summary_dict(self):
        summary = {"metrics": {"m": {"avg_score": 0.3}}}
        # "m<0.5" -> 0.3 < 0.5 -> breach
        exprs = [parse_threshold_expression("m<0.5")]
        report = evaluate_thresholds(exprs, summary)
        assert report.any_breached is True

    def test_metric_with_no_avg_score_is_missing(self):
        run = SimpleNamespace(summary={"metrics": {"m": {"count": 0, "avg_score": None}}})
        exprs = [parse_threshold_expression("m<0.5")]
        report = evaluate_thresholds(exprs, run)
        # avg_score=None is filtered out by _extract_metric_means -> treated missing
        assert len(report.missing) == 1


# ===========================================================================
# CLI smoke tests
# ===========================================================================


class TestCLIFlagsAppearInHelp:
    """Confirm the new flags show up in evalyn run-eval --help."""

    def test_max_cost_in_help(self):
        result = run_cli("run-eval", "--help")
        result.assert_success()
        result.assert_output_contains("--max-cost")
        result.assert_output_contains("cumulative")

    def test_fail_on_in_help(self):
        result = run_cli("run-eval", "--help")
        result.assert_success()
        result.assert_output_contains("--fail-on")
        result.assert_output_contains("Quality gate")


class TestCLIInvalidFlags:
    def test_invalid_max_cost_negative(self, sample_dataset):
        result = run_cli(
            "run-eval",
            "--dataset",
            str(sample_dataset),
            "--max-cost",
            "-1",
        )
        result.assert_failure()
        # argparse error goes to stderr
        assert "max-cost" in (result.stderr + result.stdout).lower()

    def test_invalid_max_cost_zero(self, sample_dataset):
        result = run_cli(
            "run-eval",
            "--dataset",
            str(sample_dataset),
            "--max-cost",
            "0",
        )
        result.assert_failure()

    def test_invalid_fail_on_expression(self, sample_dataset):
        result = run_cli(
            "run-eval",
            "--dataset",
            str(sample_dataset),
            "--fail-on",
            "garbage",
        )
        result.assert_failure()


class TestCLIFailOnAgainstRealRun:
    """Run-eval with --fail-on against a tiny objective-only dataset.

    Uses output_nonempty (a registered objective metric) so we avoid LLM API
    calls. The metric will pass for all items (mean=1.0), so a breach
    expression should exit 3 and a passing expression should exit 0.
    """

    def _dataset_with_passing_metric(self, tmp_path: Path) -> Path:
        from datetime import datetime, timezone
        from uuid import uuid4

        dataset_dir = tmp_path / "guard-dataset"
        dataset_dir.mkdir()
        items = [
            {
                "id": f"item-{i}",
                "input": {"q": f"question {i}"},
                "output": f"non-empty answer {i}",
                "metadata": {"source": "test"},
            }
            for i in range(3)
        ]
        (dataset_dir / "dataset.jsonl").write_text(
            "\n".join(json.dumps(it) for it in items) + "\n",
            encoding="utf-8",
        )
        metrics_dir = dataset_dir / "metrics"
        metrics_dir.mkdir()
        (metrics_dir / "default.json").write_text(
            json.dumps(
                [
                    {
                        "id": "output_nonempty",
                        "type": "objective",
                        "name": "Output Non-Empty",
                        "description": "Check output is not empty",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (dataset_dir / "meta.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "project": "guard-test",
                    "version": "v1",
                    "item_count": len(items),
                    "active_metric_set": "default",
                    "metric_sets": [
                        {
                            "name": "default",
                            "file": "metrics/default.json",
                            "mode": "manual",
                            "num_metrics": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return dataset_dir

    def test_fail_on_passing_threshold_exits_zero(self, tmp_path):
        dataset = self._dataset_with_passing_metric(tmp_path)
        # mean(output_nonempty) is 1.0. "<0.5" fires when score<0.5, which
        # is false here -> no breach -> exit 0.
        result = run_cli(
            "run-eval",
            "--dataset",
            str(dataset),
            "--fail-on",
            "output_nonempty<0.5",
        )
        assert result.returncode == 0, (
            f"expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_fail_on_breach_exits_three(self, tmp_path):
        dataset = self._dataset_with_passing_metric(tmp_path)
        # mean(output_nonempty) is 1.0. "<5.0" fires when score<5.0, which
        # is true -> breach -> exit 3.
        result = run_cli(
            "run-eval",
            "--dataset",
            str(dataset),
            "--fail-on",
            "output_nonempty<5.0",
        )
        assert result.returncode == 3, (
            f"expected exit 3, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "output_nonempty" in result.stderr
        assert "FAIL" in result.stderr

    def test_fail_on_missing_metric_warns_but_passes(self, tmp_path):
        dataset = self._dataset_with_passing_metric(tmp_path)
        result = run_cli(
            "run-eval",
            "--dataset",
            str(dataset),
            "--fail-on",
            "nonexistent_metric<0.5",
        )
        assert result.returncode == 0, (
            f"expected exit 0 for missing metric, got {result.returncode}\n"
            f"stderr: {result.stderr}"
        )
        assert "nonexistent_metric" in result.stderr
        assert "WARNING" in result.stderr
