"""
Tests for APE (Automatic Prompt Engineer) optimizer.

These tests verify the APE optimizer functionality without requiring API calls.
Run with: pytest tests/test_ape.py -v
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from evalyn_sdk.calibration import (
    APEConfig,
    APEOptimizer,
    AlignmentMetrics,
    DisagreementCase,
    DisagreementAnalysis,
    build_full_prompt,
    build_dataset_from_annotations,
    parse_candidates_response,
    parse_judge_response,
)
from evalyn_sdk.defaults import DEFAULT_EVAL_MODEL, DEFAULT_GENERATOR_MODEL
from evalyn_sdk.models import Annotation, DatasetItem, MetricResult


class TestAPEConfig:
    """Test APEConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = APEConfig()
        assert config.num_candidates == 10
        assert config.eval_rounds == 5
        assert config.eval_samples_per_round == 5
        assert config.generator_model == DEFAULT_GENERATOR_MODEL
        assert config.scorer_model == DEFAULT_EVAL_MODEL
        assert config.exploration_weight == 1.0
        assert config.train_split == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = APEConfig(
            num_candidates=5,
            eval_rounds=3,
            eval_samples_per_round=3,
            exploration_weight=2.0,
        )
        assert config.num_candidates == 5
        assert config.eval_rounds == 3
        assert config.eval_samples_per_round == 3
        assert config.exploration_weight == 2.0


class TestOptimizerUtilsParsing:
    """Test optimizer_utils parsing functions (no API calls)."""

    def test_parse_candidates_response_valid_json(self):
        """Test parsing valid JSON array of candidates."""
        response = '["Candidate 1 text", "Candidate 2 text", "Candidate 3 text"]'
        candidates = parse_candidates_response(response)
        assert len(candidates) == 3
        assert candidates[0] == "Candidate 1 text"
        assert candidates[1] == "Candidate 2 text"

    def test_parse_candidates_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
["First prompt", "Second prompt"]
```"""
        candidates = parse_candidates_response(response)
        assert len(candidates) == 2
        assert candidates[0] == "First prompt"

    def test_parse_candidates_response_with_extra_text(self):
        """Test parsing JSON with surrounding text."""
        response = """Here are the candidates:
["Prompt A", "Prompt B"]
Hope these help!"""
        candidates = parse_candidates_response(response)
        assert len(candidates) == 2

    def test_parse_candidates_response_invalid(self):
        """Test parsing invalid response returns empty list."""
        response = "This is not valid JSON"
        candidates = parse_candidates_response(response)
        assert candidates == []

    def test_parse_judge_response_json(self):
        """Test parsing judge response with JSON."""
        response = '{"passed": true, "reason": "Good output"}'
        result = parse_judge_response(response)
        assert result is True

    def test_parse_judge_response_false(self):
        """Test parsing judge response with passed=false."""
        response = '{"passed": false, "reason": "Bad output"}'
        result = parse_judge_response(response)
        assert result is False

    def test_parse_judge_response_fallback_true(self):
        """Test parsing judge response using fallback detection."""
        response = "The output is good. true"
        result = parse_judge_response(response)
        assert result is True

    def test_parse_judge_response_fallback_false(self):
        """Test parsing judge response defaults to False."""
        response = "Unable to determine"
        result = parse_judge_response(response)
        assert result is False


class TestAPEOptimizerBuildMethods:
    """Test APE prompt building methods."""

    def test_build_full_prompt_with_rubric(self):
        """Test building full prompt with rubric."""
        preamble = "You are an expert evaluator."
        rubric = ["Check for accuracy", "Check for completeness"]

        full_prompt = build_full_prompt(preamble, rubric)

        assert "You are an expert evaluator." in full_prompt
        assert "Check for accuracy" in full_prompt
        assert "Check for completeness" in full_prompt
        assert "PASS only if all criteria met" in full_prompt
        assert '{"passed": true/false' in full_prompt

    def test_build_full_prompt_without_rubric(self):
        """Test building full prompt without rubric."""
        preamble = "You are an expert evaluator."

        full_prompt = build_full_prompt(preamble, [])

        assert "You are an expert evaluator." in full_prompt
        assert '{"passed": true/false' in full_prompt
        assert "rubric" not in full_prompt.lower() or "no rubric" in full_prompt.lower()


class TestOptimizerUtilsDataset:
    """Test optimizer_utils dataset building functions."""

    def test_build_dataset_from_annotations(self):
        """Test building train/val datasets from annotations."""
        # Create mock data
        annotations = [
            Annotation(
                id=f"ann{i}",
                target_id=f"call{i}",
                label=i % 2 == 0,  # Alternating PASS/FAIL
                rationale=f"Reason {i}",
                annotator="test_user",
            )
            for i in range(10)
        ]

        metric_results = [
            MetricResult(
                metric_id="test_metric",
                item_id=f"item{i}",
                call_id=f"call{i}",
                passed=True,
                score=0.8,
            )
            for i in range(10)
        ]

        dataset_items = [
            DatasetItem(
                id=f"item{i}",
                input={"query": f"Question {i}"},
                output=f"Answer {i}",
                metadata={"call_id": f"call{i}"},
            )
            for i in range(10)
        ]

        trainset, valset = build_dataset_from_annotations(
            metric_results, annotations, dataset_items, train_split=0.5
        )

        # With 50% split and 10 items, expect roughly 5 each
        assert len(trainset) + len(valset) == 10
        assert len(trainset) >= 3
        assert len(valset) >= 2

        # Check structure
        for item in trainset + valset:
            assert "input" in item
            assert "output" in item
            assert "expected" in item
            assert item["expected"] in ["PASS", "FAIL"]


class TestAPEUCBSelect:
    """Test UCB selection algorithm."""

    def test_ucb_select_explores_unvisited(self):
        """Test that UCB explores unvisited candidates first."""
        config = APEConfig(
            eval_rounds=2,
            eval_samples_per_round=1,
            exploration_weight=1.0,
        )
        optimizer = APEOptimizer(config=config)

        # Mock the scorer to return fixed scores
        with patch.object(optimizer, "_score_candidate") as mock_score:
            mock_score.return_value = 0.5

            candidates = ["Candidate A", "Candidate B", "Candidate C"]
            val_examples = [{"input": "test", "output": "test", "expected": "PASS"}]

            best, score = optimizer._ucb_select(candidates, [], val_examples)

            # Should have evaluated at least 2 different candidates (due to exploration)
            assert mock_score.call_count >= 2

    def test_ucb_select_empty_candidates(self):
        """Test UCB select with empty candidates."""
        optimizer = APEOptimizer()
        best, score = optimizer._ucb_select([], [], [])
        assert best == ""
        assert score == 0.0


class TestAPEOptimizerIntegration:
    """Integration tests with mocked API calls."""

    def test_optimize_not_enough_data(self):
        """Test optimization fails gracefully with insufficient data."""
        optimizer = APEOptimizer()

        # Create minimal data that won't split well
        annotations = [
            Annotation(
                id="ann1",
                target_id="call1",
                label=True,
                rationale="Good",
                annotator="test_user",
            ),
        ]
        metric_results = [
            MetricResult(
                metric_id="test",
                item_id="item1",
                call_id="call1",
                passed=True,
                score=1.0,
            ),
        ]
        disagreements = DisagreementAnalysis()

        result = optimizer.optimize(
            metric_id="test_metric",
            current_rubric=["Be accurate"],
            metric_results=metric_results,
            annotations=annotations,
            disagreements=disagreements,
            current_preamble="You are an expert.",
        )

        assert "Not enough data" in result.improvement_reasoning
        assert result.estimated_improvement == "unknown"

    def test_optimize_no_candidates_generated(self):
        """Test optimization handles failure to generate candidates."""
        optimizer = APEOptimizer(config=APEConfig(train_split=0.5))

        # Create enough data
        annotations = [
            Annotation(
                id=f"ann{i}",
                target_id=f"call{i}",
                label=True,
                rationale="Good",
                annotator="test_user",
            )
            for i in range(10)
        ]
        metric_results = [
            MetricResult(
                metric_id="test",
                item_id=f"item{i}",
                call_id=f"call{i}",
                passed=True,
                score=1.0,
            )
            for i in range(10)
        ]
        dataset_items = [
            DatasetItem(
                id=f"item{i}",
                input={"query": f"Q{i}"},
                output=f"A{i}",
                metadata={"call_id": f"call{i}"},
            )
            for i in range(10)
        ]
        disagreements = DisagreementAnalysis()

        # Mock the generator to return empty candidates
        with patch.object(optimizer, "_propose_candidates", return_value=[]):
            result = optimizer.optimize(
                metric_id="test_metric",
                current_rubric=["Be accurate"],
                metric_results=metric_results,
                annotations=annotations,
                disagreements=disagreements,
                dataset_items=dataset_items,
                current_preamble="You are an expert.",
            )

        assert "failed to generate" in result.improvement_reasoning.lower()
        assert result.estimated_improvement == "unknown"

    def test_optimize_success(self):
        """Test successful optimization with mocked API."""
        config = APEConfig(
            num_candidates=3,
            eval_rounds=2,
            eval_samples_per_round=2,
            train_split=0.5,
        )
        optimizer = APEOptimizer(config=config)

        # Create data
        annotations = [
            Annotation(
                id=f"ann{i}",
                target_id=f"call{i}",
                label=i % 2 == 0,
                rationale=f"Reason {i}",
                annotator="test_user",
            )
            for i in range(10)
        ]
        metric_results = [
            MetricResult(
                metric_id="test",
                item_id=f"item{i}",
                call_id=f"call{i}",
                passed=True,
                score=0.8,
            )
            for i in range(10)
        ]
        dataset_items = [
            DatasetItem(
                id=f"item{i}",
                input={"query": f"Q{i}"},
                output=f"A{i}",
                metadata={"call_id": f"call{i}"},
            )
            for i in range(10)
        ]
        disagreements = DisagreementAnalysis(
            false_positives=[
                DisagreementCase(
                    call_id="call0",
                    input_text="Input 1",
                    output_text="Output 1",
                    judge_passed=True,
                    judge_reason="Looked good",
                    human_passed=False,
                    human_notes="Actually bad",
                    disagreement_type="false_positive",
                )
            ],
            false_negatives=[],
        )

        # Mock the generator and scorer
        with (
            patch.object(
                optimizer,
                "_propose_candidates",
                return_value=["Better prompt 1", "Better prompt 2", "Better prompt 3"],
            ),
            patch.object(optimizer, "_score_candidate", return_value=0.8),
        ):
            result = optimizer.optimize(
                metric_id="test_metric",
                current_rubric=["Be accurate", "Be complete"],
                metric_results=metric_results,
                annotations=annotations,
                disagreements=disagreements,
                dataset_items=dataset_items,
                current_preamble="You are an expert.",
            )

        assert "APE optimization completed" in result.improvement_reasoning
        assert result.optimized_preamble is not None
        assert result.full_prompt is not None
        assert "Be accurate" in result.full_prompt
        assert result.original_rubric == ["Be accurate", "Be complete"]
        assert result.improved_rubric == [
            "Be accurate",
            "Be complete",
        ]  # Rubric unchanged


class TestAPEImports:
    """Test APE imports work correctly."""

    def test_import_from_annotation(self):
        """Test importing APE from annotation module."""
        from evalyn_sdk.annotation import APEConfig, APEOptimizer

        assert APEConfig is not None
        assert APEOptimizer is not None

    def test_import_from_top_level(self):
        """Test importing APE from top-level module."""
        from evalyn_sdk import APEConfig, APEOptimizer

        assert APEConfig is not None
        assert APEOptimizer is not None
