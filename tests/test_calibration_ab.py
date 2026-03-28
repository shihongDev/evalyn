"""Tests for calibration A/B testing module."""

from evalyn_sdk.calibration.ab_testing import (
    ABComparison,
    ABTestConfig,
    ABTestResult,
    compare_scores,
    run_ab_test,
    summarize_ab_tests,
)


# ---------------------------------------------------------------------------
# compare_scores
# ---------------------------------------------------------------------------


def test_compare_scores_positive_delta():
    comp = compare_scores("item1", "metric1", 0.9, 0.5)
    assert comp.delta == 0.9 - 0.5
    assert comp.delta > 0


def test_compare_scores_negative_delta():
    comp = compare_scores("item1", "metric1", 0.3, 0.8)
    assert comp.delta == 0.3 - 0.8
    assert comp.delta < 0


def test_compare_scores_tie():
    comp = compare_scores("item1", "metric1", 0.5, 0.5)
    assert comp.delta == 0.0


def test_compare_scores_fields():
    comp = compare_scores("id1", "m1", 0.7, 0.3)
    assert comp.item_id == "id1"
    assert comp.metric_id == "m1"
    assert comp.score_a == 0.7
    assert comp.score_b == 0.3


def test_compare_scores_custom_tie_margin():
    # tie_margin does not affect the delta computation itself
    comp = compare_scores("id1", "m1", 0.5, 0.5, tie_margin=0.1)
    assert comp.delta == 0.0


# ---------------------------------------------------------------------------
# run_ab_test
# ---------------------------------------------------------------------------


def test_run_ab_test_a_wins():
    scores_a = {"i1": 0.9, "i2": 0.8, "i3": 0.7}
    scores_b = {"i1": 0.1, "i2": 0.2, "i3": 0.3}
    result = run_ab_test("m1", scores_a, scores_b)
    assert result.a_wins == 3
    assert result.b_wins == 0
    assert result.ties == 0
    assert result.mean_delta > 0


def test_run_ab_test_b_wins():
    scores_a = {"i1": 0.1, "i2": 0.2}
    scores_b = {"i1": 0.9, "i2": 0.8}
    result = run_ab_test("m1", scores_a, scores_b)
    assert result.b_wins == 2
    assert result.a_wins == 0
    assert result.mean_delta < 0


def test_run_ab_test_all_ties():
    scores_a = {"i1": 0.5, "i2": 0.5}
    scores_b = {"i1": 0.5, "i2": 0.5}
    result = run_ab_test("m1", scores_a, scores_b)
    assert result.ties == 2
    assert result.a_wins == 0
    assert result.b_wins == 0
    assert result.mean_delta == 0.0


def test_run_ab_test_mixed():
    scores_a = {"i1": 0.9, "i2": 0.1, "i3": 0.5}
    scores_b = {"i1": 0.1, "i2": 0.9, "i3": 0.5}
    result = run_ab_test("m1", scores_a, scores_b)
    assert result.a_wins == 1
    assert result.b_wins == 1
    assert result.ties == 1


def test_run_ab_test_only_common_items():
    scores_a = {"i1": 0.9, "i2": 0.8, "extra_a": 1.0}
    scores_b = {"i1": 0.1, "i2": 0.2, "extra_b": 0.0}
    result = run_ab_test("m1", scores_a, scores_b)
    assert len(result.comparisons) == 2


def test_run_ab_test_is_significant():
    scores_a = {"i1": 0.9, "i2": 0.8}
    scores_b = {"i1": 0.1, "i2": 0.2}
    result = run_ab_test("m1", scores_a, scores_b, significance_threshold=0.05)
    assert result.is_significant is True


def test_run_ab_test_not_significant():
    scores_a = {"i1": 0.50, "i2": 0.50}
    scores_b = {"i1": 0.49, "i2": 0.49}
    # mean_delta = 0.01, threshold = 0.05
    result = run_ab_test("m1", scores_a, scores_b, significance_threshold=0.05)
    assert result.is_significant is False


def test_run_ab_test_empty():
    result = run_ab_test("m1", {}, {})
    assert result.comparisons == []
    assert result.mean_score_a == 0.0
    assert result.mean_score_b == 0.0
    assert result.mean_delta == 0.0
    assert result.a_wins == 0
    assert result.b_wins == 0
    assert result.ties == 0


# ---------------------------------------------------------------------------
# summarize_ab_tests
# ---------------------------------------------------------------------------


def test_summarize_ab_tests_counts():
    r1 = run_ab_test("m1", {"i1": 0.9}, {"i1": 0.1})  # improved
    r2 = run_ab_test("m2", {"i1": 0.1}, {"i1": 0.9})  # degraded
    r3 = run_ab_test("m3", {"i1": 0.5}, {"i1": 0.5})  # unchanged
    summary = summarize_ab_tests([r1, r2, r3])
    assert summary["total_metrics"] == 3
    assert summary["improved"] == 1
    assert summary["degraded"] == 1
    assert summary["unchanged"] == 1


def test_summarize_ab_tests_empty():
    summary = summarize_ab_tests([])
    assert summary["total_metrics"] == 0
    assert summary["improved"] == 0


def test_summarize_ab_tests_all_improved():
    r1 = run_ab_test("m1", {"i1": 0.9}, {"i1": 0.1})
    r2 = run_ab_test("m2", {"i1": 0.8}, {"i1": 0.2})
    summary = summarize_ab_tests([r1, r2])
    assert summary["improved"] == 2
    assert summary["degraded"] == 0


# ---------------------------------------------------------------------------
# ABTestResult format_text
# ---------------------------------------------------------------------------


def test_ab_test_result_format_text():
    result = run_ab_test("quality", {"i1": 0.9, "i2": 0.8}, {"i1": 0.3, "i2": 0.2})
    text = result.format_text()
    assert "quality" in text
    assert "Mean A" in text
    assert "A wins" in text
    assert "Significant" in text


def test_ab_test_result_format_text_not_significant():
    result = run_ab_test("quality", {"i1": 0.5}, {"i1": 0.5})
    text = result.format_text()
    assert "no" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


def test_ab_comparison_as_dict():
    comp = compare_scores("id1", "m1", 0.7, 0.3)
    d = comp.as_dict()
    assert d["item_id"] == "id1"
    assert d["metric_id"] == "m1"
    assert d["score_a"] == 0.7
    assert d["score_b"] == 0.3
    assert d["delta"] == 0.7 - 0.3


def test_ab_test_result_roundtrip():
    original = run_ab_test("m1", {"i1": 0.9, "i2": 0.5}, {"i1": 0.3, "i2": 0.4})
    d = original.as_dict()
    restored = ABTestResult.from_dict(d)
    assert restored.metric_id == original.metric_id
    assert restored.mean_score_a == original.mean_score_a
    assert restored.mean_score_b == original.mean_score_b
    assert restored.mean_delta == original.mean_delta
    assert restored.a_wins == original.a_wins
    assert restored.b_wins == original.b_wins
    assert restored.ties == original.ties
    assert restored.is_significant == original.is_significant
    assert len(restored.comparisons) == len(original.comparisons)


def test_ab_test_result_roundtrip_empty():
    original = run_ab_test("m1", {}, {})
    d = original.as_dict()
    restored = ABTestResult.from_dict(d)
    assert restored.comparisons == []
    assert restored.mean_delta == 0.0


def test_ab_test_config_roundtrip():
    config = ABTestConfig(
        metric_id="quality",
        prompt_a="calibrated prompt",
        prompt_b="original prompt",
        label_a="cal",
        label_b="orig",
        significance_threshold=0.1,
    )
    d = config.as_dict()
    restored = ABTestConfig.from_dict(d)
    assert restored.metric_id == config.metric_id
    assert restored.prompt_a == config.prompt_a
    assert restored.prompt_b == config.prompt_b
    assert restored.label_a == config.label_a
    assert restored.label_b == config.label_b
    assert restored.significance_threshold == config.significance_threshold


def test_ab_test_config_defaults():
    config = ABTestConfig(metric_id="m1")
    assert config.prompt_a == ""
    assert config.prompt_b == ""
    assert config.label_a == "calibrated"
    assert config.label_b == "uncalibrated"
    assert config.significance_threshold == 0.05


def test_ab_test_config_roundtrip_defaults():
    config = ABTestConfig(metric_id="m1")
    d = config.as_dict()
    restored = ABTestConfig.from_dict(d)
    assert restored.label_a == "calibrated"
    assert restored.label_b == "uncalibrated"
