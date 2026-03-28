"""Tests for integration.eval_result_export module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from evalyn_sdk.integration.eval_result_export import (
    EvalExportConfig,
    EvalExportResult,
    EvalScore,
    batch_scores,
    export_eval_results,
    format_eval_for_langfuse,
    format_eval_for_langsmith,
    format_eval_for_phoenix,
    validate_scores,
)
from evalyn_sdk.models import Span


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _span(sid, name="test", span_type="llm_call", duration_ms=100, **attrs):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        attributes=attrs,
    )


def _score(
    span_id="s1", metric_id="helpfulness", score=0.9, label="", explanation=""
):
    return EvalScore(
        span_id=span_id,
        metric_id=metric_id,
        score=score,
        label=label,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# format_eval_for_phoenix
# ---------------------------------------------------------------------------


class TestFormatPhoenix:
    def test_basic(self):
        scores = [_score()]
        result = format_eval_for_phoenix(scores)
        assert len(result) == 1
        assert result[0]["span_id"] == "s1"
        assert result[0]["name"] == "helpfulness"
        assert result[0]["score"] == 0.9

    def test_includes_label(self):
        scores = [_score(label="good")]
        result = format_eval_for_phoenix(scores)
        assert result[0]["label"] == "good"

    def test_includes_explanation(self):
        scores = [_score(explanation="detailed reasoning")]
        result = format_eval_for_phoenix(scores)
        assert result[0]["explanation"] == "detailed reasoning"

    def test_omits_empty_label(self):
        scores = [_score()]
        result = format_eval_for_phoenix(scores)
        assert "label" not in result[0]

    def test_empty_scores(self):
        result = format_eval_for_phoenix([])
        assert result == []

    def test_multiple_scores(self):
        scores = [_score(span_id="s1"), _score(span_id="s2")]
        result = format_eval_for_phoenix(scores)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# format_eval_for_langfuse
# ---------------------------------------------------------------------------


class TestFormatLangfuse:
    def test_basic(self):
        scores = [_score()]
        result = format_eval_for_langfuse(scores)
        assert len(result) == 1
        assert result[0]["observationId"] == "s1"
        assert result[0]["name"] == "helpfulness"
        assert result[0]["value"] == 0.9

    def test_includes_comment_from_explanation(self):
        scores = [_score(explanation="good output")]
        result = format_eval_for_langfuse(scores)
        assert result[0]["comment"] == "good output"

    def test_empty_scores(self):
        result = format_eval_for_langfuse([])
        assert result == []


# ---------------------------------------------------------------------------
# format_eval_for_langsmith
# ---------------------------------------------------------------------------


class TestFormatLangsmith:
    def test_basic(self):
        scores = [_score()]
        result = format_eval_for_langsmith(scores)
        assert len(result) == 1
        assert result[0]["run_id"] == "s1"
        assert result[0]["key"] == "helpfulness"
        assert result[0]["score"] == 0.9

    def test_includes_value_from_label(self):
        scores = [_score(label="pass")]
        result = format_eval_for_langsmith(scores)
        assert result[0]["value"] == "pass"

    def test_includes_comment_from_explanation(self):
        scores = [_score(explanation="detailed")]
        result = format_eval_for_langsmith(scores)
        assert result[0]["comment"] == "detailed"

    def test_empty_scores(self):
        result = format_eval_for_langsmith([])
        assert result == []


# ---------------------------------------------------------------------------
# export_eval_results
# ---------------------------------------------------------------------------


class TestExportEvalResults:
    def test_phoenix_routing(self):
        scores = [_score()]
        config = EvalExportConfig(platform="phoenix")
        result = export_eval_results(scores, config)
        assert result.platform == "phoenix"
        assert result.exported == 1
        assert result.failed == 0

    def test_langfuse_routing(self):
        scores = [_score()]
        config = EvalExportConfig(platform="langfuse")
        result = export_eval_results(scores, config)
        assert result.platform == "langfuse"
        assert result.exported == 1

    def test_langsmith_routing(self):
        scores = [_score()]
        config = EvalExportConfig(platform="langsmith")
        result = export_eval_results(scores, config)
        assert result.platform == "langsmith"
        assert result.exported == 1

    def test_unsupported_platform(self):
        scores = [_score()]
        config = EvalExportConfig(platform="unknown")
        result = export_eval_results(scores, config)
        assert result.failed == 1
        assert result.exported == 0

    def test_invalid_scores_fail(self):
        scores = [EvalScore(span_id="", metric_id="", score=0.0)]
        config = EvalExportConfig(platform="phoenix")
        result = export_eval_results(scores, config)
        assert result.failed == 1
        assert result.exported == 0

    def test_empty_scores(self):
        config = EvalExportConfig(platform="phoenix")
        result = export_eval_results([], config)
        assert result.total_scores == 0
        assert result.exported == 0


# ---------------------------------------------------------------------------
# batch_scores
# ---------------------------------------------------------------------------


class TestBatchScores:
    def test_correct_sizes(self):
        scores = [_score(span_id=f"s{i}") for i in range(5)]
        batches = batch_scores(scores, batch_size=2)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_single_batch(self):
        scores = [_score()]
        batches = batch_scores(scores, batch_size=100)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_exact_batch_size(self):
        scores = [_score(span_id=f"s{i}") for i in range(4)]
        batches = batch_scores(scores, batch_size=2)
        assert len(batches) == 2

    def test_empty_scores(self):
        batches = batch_scores([], batch_size=10)
        assert batches == []


# ---------------------------------------------------------------------------
# validate_scores
# ---------------------------------------------------------------------------


class TestValidateScores:
    def test_valid(self):
        scores = [_score()]
        valid, issues = validate_scores(scores)
        assert valid is True
        assert issues == []

    def test_missing_span_id(self):
        scores = [EvalScore(span_id="", metric_id="m1", score=1.0)]
        valid, issues = validate_scores(scores)
        assert valid is False
        assert any("missing span_id" in i for i in issues)

    def test_missing_metric_id(self):
        scores = [EvalScore(span_id="s1", metric_id="", score=1.0)]
        valid, issues = validate_scores(scores)
        assert valid is False
        assert any("missing metric_id" in i for i in issues)

    def test_multiple_invalid(self):
        scores = [
            EvalScore(span_id="", metric_id="", score=0.0),
            EvalScore(span_id="", metric_id="m1", score=0.5),
        ]
        valid, issues = validate_scores(scores)
        assert valid is False
        assert len(issues) == 3

    def test_empty_scores(self):
        valid, issues = validate_scores([])
        assert valid is True
        assert issues == []


# ---------------------------------------------------------------------------
# EvalExportConfig
# ---------------------------------------------------------------------------


class TestEvalExportConfig:
    def test_from_dict(self):
        data = {"platform": "langfuse", "project_name": "myproj", "batch_size": 50}
        config = EvalExportConfig.from_dict(data)
        assert config.platform == "langfuse"
        assert config.project_name == "myproj"
        assert config.batch_size == 50

    def test_as_dict(self):
        config = EvalExportConfig(platform="phoenix", project_name="evalyn")
        d = config.as_dict()
        assert d["platform"] == "phoenix"
        assert d["project_name"] == "evalyn"

    def test_defaults(self):
        config = EvalExportConfig()
        assert config.platform == "phoenix"
        assert config.batch_size == 100


# ---------------------------------------------------------------------------
# EvalExportResult
# ---------------------------------------------------------------------------


class TestEvalExportResult:
    def test_format_text(self):
        result = EvalExportResult(
            total_scores=10, exported=8, failed=2, platform="phoenix"
        )
        text = result.format_text()
        assert "Platform: phoenix" in text
        assert "Total scores: 10" in text
        assert "Exported: 8" in text
        assert "Failed: 2" in text

    def test_as_dict(self):
        result = EvalExportResult(
            total_scores=5, exported=5, failed=0, platform="langfuse"
        )
        d = result.as_dict()
        assert d["total_scores"] == 5
        assert d["platform"] == "langfuse"


# ---------------------------------------------------------------------------
# EvalScore
# ---------------------------------------------------------------------------


class TestEvalScore:
    def test_as_dict(self):
        s = _score(label="good", explanation="well done")
        d = s.as_dict()
        assert d["span_id"] == "s1"
        assert d["metric_id"] == "helpfulness"
        assert d["score"] == 0.9
        assert d["label"] == "good"
        assert d["explanation"] == "well done"

    def test_from_dict(self):
        data = {
            "span_id": "s1",
            "metric_id": "accuracy",
            "score": 0.75,
            "label": "ok",
            "explanation": "reasonable",
        }
        s = EvalScore.from_dict(data)
        assert s.span_id == "s1"
        assert s.metric_id == "accuracy"
        assert s.score == 0.75
        assert s.label == "ok"

    def test_from_dict_defaults(self):
        data = {"span_id": "s1", "metric_id": "m1", "score": 0.5}
        s = EvalScore.from_dict(data)
        assert s.label == ""
        assert s.explanation == ""

    def test_round_trip(self):
        original = _score(label="pass", explanation="looks good")
        restored = EvalScore.from_dict(original.as_dict())
        assert restored.span_id == original.span_id
        assert restored.score == original.score
        assert restored.label == original.label
        assert restored.explanation == original.explanation
