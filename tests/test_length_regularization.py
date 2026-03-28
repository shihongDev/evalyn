"""Tests for prompt length regularization module."""

from evalyn_sdk.calibration.length_regularization import (
    LengthPenalty,
    RegularizationConfig,
    RegularizationReport,
    apply_regularization,
    compute_length_penalty,
    evaluate_prompt_lengths,
    suggest_compression,
)


# ---------------------------------------------------------------------------
# compute_length_penalty - linear
# ---------------------------------------------------------------------------


def test_linear_penalty_proportional_to_length():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=1000)
    short = compute_length_penalty("a" * 100, config)
    long = compute_length_penalty("a" * 500, config)
    assert long.penalty > short.penalty


def test_linear_penalty_exact_value():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=1000)
    result = compute_length_penalty("a" * 500, config)
    # penalty = 0.1 * 500/1000 = 0.05
    assert abs(result.penalty - 0.05) < 1e-9


def test_linear_penalty_clamped_to_weight():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=100)
    result = compute_length_penalty("a" * 200, config)
    # penalty = 0.1 * 200/100 = 0.2, clamped to 0.1
    assert abs(result.penalty - 0.1) < 1e-9


def test_linear_penalty_empty_text():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=1000)
    result = compute_length_penalty("", config)
    assert result.penalty == 0.0
    assert result.original_length == 0


# ---------------------------------------------------------------------------
# compute_length_penalty - quadratic
# ---------------------------------------------------------------------------


def test_quadratic_penalty_grows_faster():
    config_lin = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=1000)
    config_quad = RegularizationConfig(method="quadratic", penalty_weight=0.1, max_length=1000)
    text = "a" * 800
    lin = compute_length_penalty(text, config_lin)
    quad = compute_length_penalty(text, config_quad)
    # For ratio < 1, quadratic < linear
    # For ratio = 0.8: linear=0.08, quadratic=0.064
    assert quad.penalty < lin.penalty


def test_quadratic_penalty_exact_value():
    config = RegularizationConfig(method="quadratic", penalty_weight=0.1, max_length=1000)
    result = compute_length_penalty("a" * 500, config)
    # penalty = 0.1 * (500/1000)^2 = 0.1 * 0.25 = 0.025
    assert abs(result.penalty - 0.025) < 1e-9


def test_quadratic_penalty_at_max_length():
    config = RegularizationConfig(method="quadratic", penalty_weight=0.1, max_length=100)
    result = compute_length_penalty("a" * 100, config)
    # penalty = 0.1 * (100/100)^2 = 0.1
    assert abs(result.penalty - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# compute_length_penalty - threshold
# ---------------------------------------------------------------------------


def test_threshold_penalty_above():
    config = RegularizationConfig(method="threshold", penalty_weight=0.1, threshold_length=100)
    result = compute_length_penalty("a" * 150, config)
    assert result.penalty == 0.1


def test_threshold_penalty_below():
    config = RegularizationConfig(method="threshold", penalty_weight=0.1, threshold_length=100)
    result = compute_length_penalty("a" * 50, config)
    assert result.penalty == 0.0


def test_threshold_penalty_at_boundary():
    config = RegularizationConfig(method="threshold", penalty_weight=0.1, threshold_length=100)
    result = compute_length_penalty("a" * 100, config)
    # exactly at threshold - not over
    assert result.penalty == 0.0


# ---------------------------------------------------------------------------
# apply_regularization
# ---------------------------------------------------------------------------


def test_apply_regularization_reduces_score():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=1000)
    result = apply_regularization(0.8, "a" * 500, config)
    assert result < 0.8


def test_apply_regularization_clamps_to_zero():
    config = RegularizationConfig(method="linear", penalty_weight=1.0, max_length=100)
    result = apply_regularization(0.05, "a" * 100, config)
    # penalty = 1.0 * 100/100 = 1.0, score = 0.05 - 1.0 = -0.95 -> clamped to 0
    assert result == 0.0


def test_apply_regularization_clamps_to_one():
    config = RegularizationConfig(method="linear", penalty_weight=0.0, max_length=1000)
    result = apply_regularization(1.0, "", config)
    assert result == 1.0


def test_apply_regularization_empty_text():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=1000)
    result = apply_regularization(0.9, "", config)
    assert result == 0.9


# ---------------------------------------------------------------------------
# evaluate_prompt_lengths
# ---------------------------------------------------------------------------


def test_evaluate_prompt_lengths_report():
    config = RegularizationConfig(
        method="threshold", penalty_weight=0.1, threshold_length=50
    )
    prompts = ["a" * 30, "b" * 60, "c" * 100]
    report = evaluate_prompt_lengths(prompts, config)
    assert report.prompts_evaluated == 3
    assert report.prompts_over_threshold == 2  # 60 and 100 are over 50


def test_evaluate_prompt_lengths_avg_length():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=1000)
    prompts = ["a" * 100, "b" * 200, "c" * 300]
    report = evaluate_prompt_lengths(prompts, config)
    assert abs(report.avg_length - 200.0) < 1e-9


def test_evaluate_prompt_lengths_empty():
    config = RegularizationConfig()
    report = evaluate_prompt_lengths([], config)
    assert report.prompts_evaluated == 0
    assert report.avg_length == 0.0
    assert report.avg_penalty == 0.0


# ---------------------------------------------------------------------------
# suggest_compression
# ---------------------------------------------------------------------------


def test_suggest_compression_over_budget():
    result = suggest_compression("a" * 1500, max_length=1000)
    assert result["over_budget"] is True
    assert result["excess_chars"] == 500
    assert result["current_length"] == 1500
    assert "exceeds" in result["suggestion"].lower()


def test_suggest_compression_under_budget():
    result = suggest_compression("a" * 500, max_length=1000)
    assert result["over_budget"] is False
    assert result["excess_chars"] == 0
    assert "within budget" in result["suggestion"].lower()


def test_suggest_compression_exact_budget():
    result = suggest_compression("a" * 1000, max_length=1000)
    assert result["over_budget"] is False
    assert result["excess_chars"] == 0


def test_suggest_compression_empty_text():
    result = suggest_compression("", max_length=1000)
    assert result["current_length"] == 0
    assert result["over_budget"] is False


# ---------------------------------------------------------------------------
# RegularizationConfig from_dict
# ---------------------------------------------------------------------------


def test_regularization_config_from_dict():
    data = {
        "method": "quadratic",
        "penalty_weight": 0.2,
        "max_length": 3000,
        "threshold_length": 1500,
    }
    config = RegularizationConfig.from_dict(data)
    assert config.method == "quadratic"
    assert config.penalty_weight == 0.2
    assert config.max_length == 3000
    assert config.threshold_length == 1500


def test_regularization_config_from_dict_defaults():
    config = RegularizationConfig.from_dict({})
    assert config.method == "linear"
    assert config.penalty_weight == 0.1
    assert config.max_length == 2000


def test_regularization_config_roundtrip():
    config = RegularizationConfig(
        method="threshold", penalty_weight=0.5, max_length=500, threshold_length=200
    )
    restored = RegularizationConfig.from_dict(config.as_dict())
    assert restored.method == config.method
    assert restored.penalty_weight == config.penalty_weight
    assert restored.max_length == config.max_length
    assert restored.threshold_length == config.threshold_length


# ---------------------------------------------------------------------------
# RegularizationReport format_text
# ---------------------------------------------------------------------------


def test_regularization_report_format_text():
    report = RegularizationReport(
        prompts_evaluated=10,
        avg_length=500.0,
        avg_penalty=0.05,
        prompts_over_threshold=3,
    )
    text = report.format_text()
    assert "10 prompts" in text
    assert "500.0" in text
    assert "0.0500" in text
    assert "3" in text


# ---------------------------------------------------------------------------
# LengthPenalty as_dict
# ---------------------------------------------------------------------------


def test_length_penalty_as_dict():
    lp = LengthPenalty(original_length=100, penalized_score=0.9, penalty=0.01, method="linear")
    data = lp.as_dict()
    assert data["original_length"] == 100
    assert data["penalty"] == 0.01
    assert data["method"] == "linear"


# ---------------------------------------------------------------------------
# Edge: zero max_length
# ---------------------------------------------------------------------------


def test_linear_zero_max_length():
    config = RegularizationConfig(method="linear", penalty_weight=0.1, max_length=0)
    result = compute_length_penalty("hello", config)
    assert result.penalty == 0.0


def test_quadratic_zero_max_length():
    config = RegularizationConfig(method="quadratic", penalty_weight=0.1, max_length=0)
    result = compute_length_penalty("hello", config)
    assert result.penalty == 0.0
