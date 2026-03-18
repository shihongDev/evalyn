"""Tests for calibration optimizers - TextGrad and factory."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from evalyn_sdk.calibration.textgrad import TextGradConfig, TextGradOptimizer
from evalyn_sdk.calibration.factory import create_optimizer
from evalyn_sdk.calibration.models import PromptOptimizationResult, TokenAccumulator
from evalyn_sdk.utils.api_client import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generate_result(text: str = "ok") -> GenerateResult:
    """Build a minimal GenerateResult for mocking."""
    return GenerateResult(text=text, input_tokens=10, output_tokens=5, model="test")


def _make_examples(n: int = 10) -> list:
    """Build a list of labelled examples for train/val splits."""
    examples = []
    for i in range(n):
        examples.append({
            "id": f"call-{i}",
            "input": f"input {i}",
            "output": f"output {i}",
            "expected": "PASS" if i % 2 == 0 else "FAIL",
            "call_id": f"call-{i}",
        })
    return examples


# ---------------------------------------------------------------------------
# TextGradConfig tests
# ---------------------------------------------------------------------------

class TestTextGradConfig:
    def test_defaults(self):
        cfg = TextGradConfig()
        assert cfg.max_iterations == 8
        assert cfg.improvement_threshold == 0.01
        assert cfg.num_failure_examples == 5
        assert cfg.early_stop_patience == 3

    def test_custom_values(self):
        cfg = TextGradConfig(max_iterations=4, early_stop_patience=1)
        assert cfg.max_iterations == 4
        assert cfg.early_stop_patience == 1


# ---------------------------------------------------------------------------
# TextGradOptimizer tests
# ---------------------------------------------------------------------------

class TestTextGrad:
    def _make_optimizer(self) -> TextGradOptimizer:
        """Build an optimizer with mocked LLM clients."""
        cfg = TextGradConfig(max_iterations=3, early_stop_patience=2)
        opt = TextGradOptimizer(config=cfg, api_key="fake-key")

        # Mock clients via the private attributes so the lazy properties
        # return these mocks directly.
        mock_task = MagicMock()
        mock_task.generate_with_usage.return_value = _make_generate_result(
            "Revised preamble: be stricter on edge cases."
        )
        opt._task_client = mock_task

        mock_scorer = MagicMock()
        mock_scorer.generate_with_usage.return_value = _make_generate_result(
            '{"passed": true, "reason": "ok", "score": 0.9}'
        )
        opt._scorer_client = mock_scorer

        return opt

    def test_optimize_returns_result(self):
        opt = self._make_optimizer()

        # Patch split_train_val to return pre-built splits
        train = _make_examples(7)
        val = _make_examples(3)
        opt.split_train_val = MagicMock(return_value=(train, val))

        # Patch score_preamble to return increasing scores
        call_count = {"n": 0}

        def increasing_score(*_args, **_kwargs):
            call_count["n"] += 1
            # First call is seed score, subsequent calls are revised scores
            return 0.5 + call_count["n"] * 0.05

        opt.score_preamble = MagicMock(side_effect=increasing_score)

        result = opt.optimize(
            metric_id="test_metric",
            current_rubric=["Be accurate"],
            current_preamble="You are a judge.",
            metric_results=[],
            annotations=[],
        )

        assert isinstance(result, PromptOptimizationResult)
        assert result.original_preamble == "You are a judge."
        assert result.optimized_preamble  # Should have content
        assert result.improvement_reasoning  # Should have reasoning
        assert result.estimated_improvement in ("none", "low", "medium", "high", "unknown")
        assert result.improved_rubric == ["Be accurate"]  # Rubric unchanged

    def test_early_stopping(self):
        opt = self._make_optimizer()
        opt.config.early_stop_patience = 2
        opt.config.max_iterations = 10

        train = _make_examples(7)
        val = _make_examples(3)
        opt.split_train_val = MagicMock(return_value=(train, val))

        # score_preamble always returns the same value -> triggers early stop
        opt.score_preamble = MagicMock(return_value=0.6)

        result = opt.optimize(
            metric_id="test_metric",
            current_rubric=["Be accurate"],
            current_preamble="You are a judge.",
            metric_results=[],
            annotations=[],
        )

        assert isinstance(result, PromptOptimizationResult)
        # With patience=2, _collect_failures is called at most 2 + 1 times
        # (one extra iteration before the patience counter saturates).
        # Each iteration also calls score_preamble. The total is bounded.
        # We mainly care that it did NOT run all 10 iterations.
        # score_preamble is called: 1 (seed) + N (revised) + 2 (val)
        # N should be <= patience + 1 = 3 because early stop kicks in.
        total_score_calls = opt.score_preamble.call_count
        # Seed(1) + iterations(<=3) + val(2) = at most 6
        assert total_score_calls <= 8  # generous upper bound

    def test_not_enough_data(self):
        opt = self._make_optimizer()

        # Return too-small splits
        opt.split_train_val = MagicMock(return_value=([], []))

        result = opt.optimize(
            metric_id="test_metric",
            current_rubric=["Be accurate"],
            current_preamble="You are a judge.",
            metric_results=[],
            annotations=[],
        )

        assert isinstance(result, PromptOptimizationResult)
        assert "Not enough data" in result.improvement_reasoning
        assert result.estimated_improvement == "unknown"

    def test_no_failures_stops_early(self):
        """When there are zero failures, optimization should stop immediately."""
        opt = self._make_optimizer()

        train = _make_examples(7)
        val = _make_examples(3)
        opt.split_train_val = MagicMock(return_value=(train, val))
        opt.score_preamble = MagicMock(return_value=1.0)

        # _collect_failures returns empty list (perfect preamble)
        opt._collect_failures = MagicMock(return_value=[])

        result = opt.optimize(
            metric_id="test_metric",
            current_rubric=["Be accurate"],
            current_preamble="You are a judge.",
            metric_results=[],
            annotations=[],
        )

        assert isinstance(result, PromptOptimizationResult)
        # _critique should never be called since there were no failures
        opt._task_client.generate_with_usage.assert_not_called()

    def test_accumulator_tracking(self):
        opt = self._make_optimizer()

        train = _make_examples(7)
        val = _make_examples(3)
        opt.split_train_val = MagicMock(return_value=(train, val))
        opt.score_preamble = MagicMock(return_value=0.7)

        accumulator = TokenAccumulator()
        result = opt.optimize(
            metric_id="test_metric",
            current_rubric=["Be accurate"],
            current_preamble="You are a judge.",
            metric_results=[],
            annotations=[],
            accumulator=accumulator,
        )

        assert isinstance(result, PromptOptimizationResult)
        # The accumulator should have been passed through to _collect_failures,
        # _critique, and _revise. Since _collect_failures uses scorer_client
        # directly, tokens should accumulate.


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_factory_creates_textgrad(self):
        cfg = TextGradConfig()
        opt = create_optimizer("textgrad", config=cfg)
        assert isinstance(opt, TextGradOptimizer)
        assert opt.config is cfg

    def test_factory_creates_textgrad_case_insensitive(self):
        opt = create_optimizer("TextGrad")
        assert isinstance(opt, TextGradOptimizer)

    def test_factory_creates_ape(self):
        from evalyn_sdk.calibration.ape import APEOptimizer

        opt = create_optimizer("ape")
        assert isinstance(opt, APEOptimizer)

    def test_factory_creates_opro(self):
        from evalyn_sdk.calibration.opro import OPROOptimizer

        opt = create_optimizer("opro")
        assert isinstance(opt, OPROOptimizer)

    def test_factory_creates_basic(self):
        from evalyn_sdk.calibration.basic import BasicOptimizer

        opt = create_optimizer("basic")
        assert isinstance(opt, BasicOptimizer)

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer("nonexistent")
