"""Tests for new optimizer infrastructure and implementations."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from evalyn_sdk.calibration.base_optimizer import BaseOptimizer
from evalyn_sdk.calibration.models import PromptOptimizationResult


@dataclass
class DummyConfig:
    task_model: str = "test-model"
    scorer_model: str = "test-model"


class ConcreteOptimizer(BaseOptimizer):
    """Concrete subclass for testing."""

    def optimize(self, *, metric_id, current_rubric, current_preamble, **kwargs):
        return self.build_result(
            original_preamble=current_preamble,
            optimized_preamble="improved " + current_preamble,
            rubric=current_rubric,
            reasoning="test",
            estimated_improvement="medium",
        )


class TestBaseOptimizer:
    def test_init_stores_config(self):
        cfg = DummyConfig()
        opt = ConcreteOptimizer(config=cfg)
        assert opt.config is cfg
        assert opt._api_key is None

    def test_init_with_api_key(self):
        opt = ConcreteOptimizer(config=DummyConfig(), api_key="sk-test")
        assert opt._api_key == "sk-test"

    def test_build_result_returns_prompt_optimization_result(self):
        opt = ConcreteOptimizer(config=DummyConfig())
        result = opt.build_result(
            original_preamble="original",
            optimized_preamble="improved",
            rubric=["criterion 1", "criterion 2"],
            reasoning="fixed false positives",
            estimated_improvement="high",
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.original_preamble == "original"
        assert result.optimized_preamble == "improved"
        assert result.original_rubric == ["criterion 1", "criterion 2"]
        assert result.improved_rubric == ["criterion 1", "criterion 2"]
        assert result.estimated_improvement == "high"
        assert "fixed false positives" in result.improvement_reasoning
        assert result.full_prompt  # non-empty

    def test_build_result_rubric_always_unchanged(self):
        opt = ConcreteOptimizer(config=DummyConfig())
        result = opt.build_result(
            original_preamble="a",
            optimized_preamble="b",
            rubric=["r1", "r2"],
            reasoning="x",
            estimated_improvement="low",
        )
        assert result.original_rubric == result.improved_rubric

    def test_optimize_abstract_raises(self):
        with pytest.raises(NotImplementedError):
            BaseOptimizer(config=DummyConfig()).optimize(
                metric_id="test",
                current_rubric=[],
                current_preamble="",
                disagreements=None,
                metric_results=[],
                annotations=[],
                dataset_items=[],
                accumulator=None,
            )

    def test_concrete_optimize_works(self):
        opt = ConcreteOptimizer(config=DummyConfig())
        result = opt.optimize(
            metric_id="test",
            current_rubric=["be good"],
            current_preamble="judge prompt",
            metric_results=[],
            annotations=[],
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.optimized_preamble == "improved judge prompt"
