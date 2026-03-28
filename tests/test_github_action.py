from __future__ import annotations

import pytest

from evalyn_sdk.integration.github_action import (
    ActionInput,
    ActionOutput,
    ActionYaml,
    create_status_badge,
    format_pr_summary,
    generate_action_yaml,
    generate_workflow_yaml,
    parse_action_inputs,
)


# ---------------------------------------------------------------------------
# ActionInput
# ---------------------------------------------------------------------------


class TestActionInput:
    def test_defaults(self):
        ai = ActionInput()
        assert ai.dataset_path == "data/"
        assert ai.metrics == ["output_nonempty"]
        assert ai.provider == "gemini"
        assert ai.profile == "smoke-test"
        assert ai.fail_on_regression is True
        assert ai.regression_threshold == 0.05
        assert ai.compare_with_baseline is True

    def test_as_dict(self):
        ai = ActionInput(provider="openai", metrics=["a", "b"])
        d = ai.as_dict()
        assert d["provider"] == "openai"
        assert d["metrics"] == ["a", "b"]
        assert d["dataset_path"] == "data/"

    def test_from_dict_roundtrip(self):
        ai = ActionInput(
            dataset_path="eval/",
            metrics=["m1", "m2"],
            provider="anthropic",
            profile="full",
            fail_on_regression=False,
            regression_threshold=0.1,
            compare_with_baseline=False,
        )
        restored = ActionInput.from_dict(ai.as_dict())
        assert restored == ai

    def test_from_dict_missing_keys(self):
        ai = ActionInput.from_dict({})
        assert ai.dataset_path == "data/"
        assert ai.metrics == ["output_nonempty"]
        assert ai.provider == "gemini"

    def test_from_dict_partial(self):
        ai = ActionInput.from_dict({"provider": "openai", "profile": "full"})
        assert ai.provider == "openai"
        assert ai.profile == "full"
        assert ai.metrics == ["output_nonempty"]

    def test_empty_metrics(self):
        ai = ActionInput(metrics=[])
        assert ai.metrics == []
        d = ai.as_dict()
        assert d["metrics"] == []


# ---------------------------------------------------------------------------
# ActionOutput
# ---------------------------------------------------------------------------


class TestActionOutput:
    def test_defaults(self):
        ao = ActionOutput()
        assert ao.passed is False
        assert ao.pass_rate == 0.0
        assert ao.regressions == []
        assert ao.summary_markdown == ""
        assert ao.badge_url == ""

    def test_as_dict(self):
        ao = ActionOutput(
            passed=True, pass_rate=0.95, regressions=["metric_a"],
            summary_markdown="all good", badge_url="https://example.com/badge",
        )
        d = ao.as_dict()
        assert d["passed"] is True
        assert d["pass_rate"] == 0.95
        assert d["regressions"] == ["metric_a"]
        assert d["badge_url"] == "https://example.com/badge"

    def test_format_text_passed(self):
        ao = ActionOutput(passed=True, pass_rate=0.92)
        text = ao.format_text()
        assert "PASSED" in text
        assert "92.0%" in text
        assert "none" in text

    def test_format_text_failed_with_regressions(self):
        ao = ActionOutput(
            passed=False, pass_rate=0.5, regressions=["helpfulness", "accuracy"],
        )
        text = ao.format_text()
        assert "FAILED" in text
        assert "helpfulness" in text
        assert "accuracy" in text

    def test_format_text_with_badge(self):
        ao = ActionOutput(badge_url="https://example.com/badge")
        text = ao.format_text()
        assert "https://example.com/badge" in text

    def test_format_text_with_summary(self):
        ao = ActionOutput(summary_markdown="details here")
        text = ao.format_text()
        assert "details here" in text


# ---------------------------------------------------------------------------
# ActionYaml
# ---------------------------------------------------------------------------


class TestActionYaml:
    def test_defaults(self):
        ay = ActionYaml()
        assert ay.name == "Evalyn Evaluation"
        assert ay.description == "Run evalyn evaluation on PR"
        assert ay.inputs == {}
        assert ay.runs_steps == []

    def test_as_dict(self):
        ay = ActionYaml(
            name="Custom Action",
            inputs={"dataset": {"description": "path", "default": "data/"}},
            runs_steps=[{"name": "step1", "run": "echo hello"}],
        )
        d = ay.as_dict()
        assert d["name"] == "Custom Action"
        assert "dataset" in d["inputs"]
        assert len(d["runs_steps"]) == 1

    def test_from_dict_roundtrip(self):
        ay = ActionYaml(
            name="My Action",
            description="test action",
            inputs={"key": {"default": "val"}},
            runs_steps=[{"run": "echo ok"}],
        )
        restored = ActionYaml.from_dict(ay.as_dict())
        assert restored == ay

    def test_from_dict_missing_keys(self):
        ay = ActionYaml.from_dict({})
        assert ay.name == "Evalyn Evaluation"
        assert ay.description == "Run evalyn evaluation on PR"
        assert ay.inputs == {}

    def test_from_dict_partial(self):
        ay = ActionYaml.from_dict({"name": "X"})
        assert ay.name == "X"
        assert ay.description == "Run evalyn evaluation on PR"


# ---------------------------------------------------------------------------
# generate_action_yaml
# ---------------------------------------------------------------------------


class TestGenerateActionYaml:
    def test_contains_steps(self):
        yaml = generate_action_yaml(ActionInput())
        assert "steps:" in yaml
        assert "Checkout" in yaml
        assert "Set up Python" in yaml
        assert "Install evalyn" in yaml
        assert "Run evaluation" in yaml
        assert "Post results" in yaml

    def test_contains_provider(self):
        yaml = generate_action_yaml(ActionInput(provider="anthropic"))
        assert "anthropic" in yaml

    def test_contains_metrics(self):
        yaml = generate_action_yaml(ActionInput(metrics=["helpfulness", "accuracy"]))
        assert "helpfulness" in yaml
        assert "accuracy" in yaml

    def test_contains_gemini_api_key(self):
        yaml = generate_action_yaml(ActionInput())
        assert "GEMINI_API_KEY" in yaml

    def test_custom_dataset_path(self):
        yaml = generate_action_yaml(ActionInput(dataset_path="eval_data/"))
        assert "eval_data/" in yaml


# ---------------------------------------------------------------------------
# generate_workflow_yaml
# ---------------------------------------------------------------------------


class TestGenerateWorkflowYaml:
    def test_valid_structure(self):
        yaml = generate_workflow_yaml(ActionInput())
        assert "name:" in yaml
        assert "on:" in yaml
        assert "jobs:" in yaml
        assert "steps:" in yaml
        assert "pull_request:" in yaml

    def test_contains_provider(self):
        yaml = generate_workflow_yaml(ActionInput(provider="openai"))
        assert "openai" in yaml

    def test_compare_with_baseline(self):
        yaml = generate_workflow_yaml(ActionInput(compare_with_baseline=True))
        assert "Compare with baseline" in yaml
        assert "evalyn compare" in yaml

    def test_no_compare_when_disabled(self):
        yaml = generate_workflow_yaml(ActionInput(compare_with_baseline=False))
        assert "Compare with baseline" not in yaml

    def test_contains_gemini_api_key(self):
        yaml = generate_workflow_yaml(ActionInput())
        assert "GEMINI_API_KEY" in yaml

    def test_custom_threshold(self):
        yaml = generate_workflow_yaml(
            ActionInput(compare_with_baseline=True, regression_threshold=0.1)
        )
        assert "0.1" in yaml


# ---------------------------------------------------------------------------
# create_status_badge
# ---------------------------------------------------------------------------


class TestCreateStatusBadge:
    def test_green_badge(self):
        url = create_status_badge(0.95)
        assert "green" in url
        assert "shields.io" in url

    def test_yellow_badge(self):
        url = create_status_badge(0.7)
        assert "yellow" in url

    def test_red_badge(self):
        url = create_status_badge(0.4)
        assert "red" in url

    def test_boundary_green(self):
        # > 0.8 is green, exactly 0.8 is yellow
        url_above = create_status_badge(0.81)
        url_at = create_status_badge(0.8)
        assert "green" in url_above
        assert "yellow" in url_at

    def test_boundary_yellow(self):
        # > 0.6 is yellow, exactly 0.6 is red
        url_above = create_status_badge(0.61)
        url_at = create_status_badge(0.6)
        assert "yellow" in url_above
        assert "red" in url_at

    def test_zero_pass_rate(self):
        url = create_status_badge(0.0)
        assert "red" in url

    def test_perfect_pass_rate(self):
        url = create_status_badge(1.0)
        assert "green" in url


# ---------------------------------------------------------------------------
# format_pr_summary
# ---------------------------------------------------------------------------


class TestFormatPrSummary:
    def test_passed_summary(self):
        output = ActionOutput(passed=True, pass_rate=0.95)
        md = format_pr_summary(output)
        assert "Passed" in md
        assert "95.0%" in md

    def test_failed_summary(self):
        output = ActionOutput(passed=False, pass_rate=0.3)
        md = format_pr_summary(output)
        assert "Failed" in md

    def test_with_regressions(self):
        output = ActionOutput(
            passed=False, regressions=["metric_a", "metric_b"],
        )
        md = format_pr_summary(output)
        assert "metric_a" in md
        assert "metric_b" in md
        assert "Regressions" in md

    def test_with_badge(self):
        output = ActionOutput(badge_url="https://img.shields.io/badge/evalyn-95%25-green")
        md = format_pr_summary(output)
        assert "![badge]" in md
        assert "shields.io" in md

    def test_with_summary_markdown(self):
        output = ActionOutput(summary_markdown="Additional details here")
        md = format_pr_summary(output)
        assert "Additional details here" in md

    def test_no_regressions_section_when_empty(self):
        output = ActionOutput(passed=True, pass_rate=1.0, regressions=[])
        md = format_pr_summary(output)
        assert "Regressions" not in md


# ---------------------------------------------------------------------------
# parse_action_inputs
# ---------------------------------------------------------------------------


class TestParseActionInputs:
    def test_from_env_vars(self):
        env = {
            "INPUT_DATASET_PATH": "my_data/",
            "INPUT_METRICS": "helpfulness,accuracy",
            "INPUT_PROVIDER": "openai",
            "INPUT_PROFILE": "full",
            "INPUT_FAIL_ON_REGRESSION": "false",
            "INPUT_REGRESSION_THRESHOLD": "0.1",
            "INPUT_COMPARE_WITH_BASELINE": "false",
        }
        ai = parse_action_inputs(env)
        assert ai.dataset_path == "my_data/"
        assert ai.metrics == ["helpfulness", "accuracy"]
        assert ai.provider == "openai"
        assert ai.profile == "full"
        assert ai.fail_on_regression is False
        assert ai.regression_threshold == 0.1
        assert ai.compare_with_baseline is False

    def test_defaults_from_empty_env(self):
        ai = parse_action_inputs({})
        assert ai.dataset_path == "data/"
        assert ai.metrics == ["output_nonempty"]
        assert ai.provider == "gemini"

    def test_single_metric(self):
        env = {"INPUT_METRICS": "helpfulness"}
        ai = parse_action_inputs(env)
        assert ai.metrics == ["helpfulness"]

    def test_metrics_with_spaces(self):
        env = {"INPUT_METRICS": " helpfulness , accuracy "}
        ai = parse_action_inputs(env)
        assert ai.metrics == ["helpfulness", "accuracy"]

    def test_fail_on_regression_true(self):
        env = {"INPUT_FAIL_ON_REGRESSION": "true"}
        ai = parse_action_inputs(env)
        assert ai.fail_on_regression is True

    def test_fail_on_regression_capitalized(self):
        env = {"INPUT_FAIL_ON_REGRESSION": "True"}
        ai = parse_action_inputs(env)
        assert ai.fail_on_regression is True
