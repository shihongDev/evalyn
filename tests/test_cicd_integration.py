from __future__ import annotations

import os

import pytest

from evalyn_sdk.integration.cicd import (
    CIConfig,
    CIResult,
    GitHubActionConfig,
    check_ci_result,
    generate_github_action_yaml,
    generate_pr_comment,
    parse_ci_environment,
    should_run_evaluation,
)


# ---------------------------------------------------------------------------
# CIConfig
# ---------------------------------------------------------------------------


class TestCIConfig:
    def test_defaults(self):
        c = CIConfig()
        assert c.trigger == "pull_request"
        assert c.dataset_path == "data/"
        assert c.metrics == ["output_nonempty"]
        assert c.fail_on_regression is True
        assert c.regression_threshold == 0.05
        assert c.provider == "gemini"

    def test_as_dict(self):
        c = CIConfig(trigger="push", metrics=["a", "b"], provider="openai")
        d = c.as_dict()
        assert d["trigger"] == "push"
        assert d["metrics"] == ["a", "b"]
        assert d["provider"] == "openai"

    def test_from_dict_roundtrip(self):
        c = CIConfig(
            trigger="schedule", dataset_path="eval/",
            metrics=["m1", "m2"], fail_on_regression=False,
            regression_threshold=0.1, provider="anthropic",
        )
        restored = CIConfig.from_dict(c.as_dict())
        assert restored == c

    def test_from_dict_missing_keys(self):
        c = CIConfig.from_dict({})
        assert c.trigger == "pull_request"
        assert c.metrics == ["output_nonempty"]

    def test_from_dict_custom_values(self):
        c = CIConfig.from_dict({
            "trigger": "push",
            "regression_threshold": 0.2,
        })
        assert c.trigger == "push"
        assert c.regression_threshold == 0.2


# ---------------------------------------------------------------------------
# CIResult
# ---------------------------------------------------------------------------


class TestCIResult:
    def test_defaults(self):
        r = CIResult()
        assert r.passed is False
        assert r.pass_rate == 0.0
        assert r.regressions == []
        assert r.summary == ""
        assert r.exit_code == 0

    def test_as_dict(self):
        r = CIResult(passed=True, pass_rate=0.95, regressions=["metric_a"], summary="ok", exit_code=0)
        d = r.as_dict()
        assert d["passed"] is True
        assert d["pass_rate"] == 0.95
        assert d["regressions"] == ["metric_a"]
        assert d["summary"] == "ok"

    def test_format_text_passed(self):
        r = CIResult(passed=True, pass_rate=1.0, summary="All good")
        text = r.format_text()
        assert "PASSED" in text
        assert "100.0%" in text
        assert "none" in text

    def test_format_text_failed_with_regressions(self):
        r = CIResult(passed=False, pass_rate=0.8, regressions=["helpfulness"])
        text = r.format_text()
        assert "FAILED" in text
        assert "helpfulness" in text

    def test_format_text_exit_code(self):
        r = CIResult(exit_code=1)
        text = r.format_text()
        assert "Exit code: 1" in text


# ---------------------------------------------------------------------------
# GitHubActionConfig
# ---------------------------------------------------------------------------


class TestGitHubActionConfig:
    def test_defaults(self):
        g = GitHubActionConfig()
        assert g.name == "evalyn-eval"
        assert g.runs_on == "ubuntu-latest"
        assert g.python_version == "3.13"
        assert g.evalyn_version == "latest"

    def test_as_dict(self):
        g = GitHubActionConfig(name="my-eval", python_version="3.12")
        d = g.as_dict()
        assert d["name"] == "my-eval"
        assert d["python_version"] == "3.12"

    def test_from_dict_roundtrip(self):
        g = GitHubActionConfig(
            name="custom", runs_on="macos-latest",
            python_version="3.11", evalyn_version="0.5.0",
        )
        restored = GitHubActionConfig.from_dict(g.as_dict())
        assert restored == g

    def test_from_dict_missing_keys(self):
        g = GitHubActionConfig.from_dict({})
        assert g.name == "evalyn-eval"
        assert g.evalyn_version == "latest"


# ---------------------------------------------------------------------------
# generate_github_action_yaml
# ---------------------------------------------------------------------------


class TestGenerateGitHubActionYaml:
    def test_contains_name(self):
        yaml = generate_github_action_yaml(CIConfig())
        assert "name: evalyn-eval" in yaml

    def test_contains_trigger(self):
        yaml = generate_github_action_yaml(CIConfig(trigger="push"))
        assert "push:" in yaml

    def test_contains_python_setup(self):
        yaml = generate_github_action_yaml(CIConfig())
        assert "setup-python" in yaml
        assert "3.13" in yaml

    def test_contains_evalyn_install(self):
        yaml = generate_github_action_yaml(CIConfig())
        assert "pip install evalyn" in yaml

    def test_contains_gemini_api_key_env(self):
        yaml = generate_github_action_yaml(CIConfig())
        assert "GEMINI_API_KEY" in yaml

    def test_contains_dataset_path(self):
        yaml = generate_github_action_yaml(CIConfig(dataset_path="my_data/"))
        assert "my_data/" in yaml

    def test_contains_metrics(self):
        yaml = generate_github_action_yaml(CIConfig(metrics=["metric_a", "metric_b"]))
        assert "metric_a metric_b" in yaml

    def test_contains_compare_when_fail_on_regression(self):
        yaml = generate_github_action_yaml(CIConfig(fail_on_regression=True))
        assert "evalyn compare" in yaml

    def test_no_compare_when_no_fail_on_regression(self):
        yaml = generate_github_action_yaml(CIConfig(fail_on_regression=False))
        assert "evalyn compare" not in yaml

    def test_custom_action_config(self):
        ac = GitHubActionConfig(name="my-job", runs_on="macos-latest", python_version="3.12")
        yaml = generate_github_action_yaml(CIConfig(), action_config=ac)
        assert "name: my-job" in yaml
        assert "macos-latest" in yaml
        assert "3.12" in yaml

    def test_specific_evalyn_version(self):
        ac = GitHubActionConfig(evalyn_version="1.2.3")
        yaml = generate_github_action_yaml(CIConfig(), action_config=ac)
        assert "pip install evalyn==1.2.3" in yaml


# ---------------------------------------------------------------------------
# check_ci_result
# ---------------------------------------------------------------------------


class TestCheckCIResult:
    def test_pass(self):
        result = check_ci_result(1.0, [], CIConfig())
        assert result.passed is True
        assert result.exit_code == 0

    def test_fail_low_pass_rate(self):
        result = check_ci_result(0.5, [], CIConfig(regression_threshold=0.05))
        assert result.passed is False
        assert result.exit_code == 1

    def test_fail_on_regression(self):
        result = check_ci_result(1.0, ["metric_a"], CIConfig(fail_on_regression=True))
        assert result.passed is False
        assert result.exit_code == 1

    def test_regressions_ignored_when_disabled(self):
        result = check_ci_result(1.0, ["metric_a"], CIConfig(fail_on_regression=False))
        assert result.passed is True
        assert result.exit_code == 0

    def test_empty_regressions_pass(self):
        result = check_ci_result(1.0, [], CIConfig())
        assert result.regressions == []
        assert result.passed is True

    def test_summary_contains_info(self):
        result = check_ci_result(0.5, ["x"], CIConfig())
        assert "regression" in result.summary.lower() or "threshold" in result.summary.lower()


# ---------------------------------------------------------------------------
# generate_pr_comment
# ---------------------------------------------------------------------------


class TestGeneratePRComment:
    def test_passed_comment(self):
        result = CIResult(passed=True, pass_rate=1.0, summary="All good")
        md = generate_pr_comment(result)
        assert "Passed" in md
        assert "100.0%" in md

    def test_failed_comment(self):
        result = CIResult(passed=False, pass_rate=0.7, regressions=["metric_a"])
        md = generate_pr_comment(result)
        assert "Failed" in md
        assert "metric_a" in md

    def test_markdown_header(self):
        result = CIResult(passed=True, pass_rate=1.0)
        md = generate_pr_comment(result)
        assert md.startswith("## ")

    def test_no_regressions_section_when_empty(self):
        result = CIResult(passed=True, pass_rate=1.0, regressions=[])
        md = generate_pr_comment(result)
        assert "Regressions" not in md


# ---------------------------------------------------------------------------
# parse_ci_environment
# ---------------------------------------------------------------------------


class TestParseCIEnvironment:
    def test_reads_set_vars(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SHA", "abc123")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
        env = parse_ci_environment()
        assert env["GITHUB_SHA"] == "abc123"
        assert env["GITHUB_REF"] == "refs/heads/main"
        assert env["GITHUB_EVENT_NAME"] == "push"

    def test_skips_missing_vars(self, monkeypatch):
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        monkeypatch.delenv("GITHUB_REF", raising=False)
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        env = parse_ci_environment()
        assert "GITHUB_SHA" not in env

    def test_partial_vars(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SHA", "def456")
        monkeypatch.delenv("GITHUB_REF", raising=False)
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        env = parse_ci_environment()
        assert env["GITHUB_SHA"] == "def456"
        assert "GITHUB_REF" not in env


# ---------------------------------------------------------------------------
# should_run_evaluation
# ---------------------------------------------------------------------------


class TestShouldRunEvaluation:
    def test_matching_trigger(self):
        assert should_run_evaluation(CIConfig(trigger="push"), "push") is True

    def test_non_matching_trigger(self):
        assert should_run_evaluation(CIConfig(trigger="push"), "pull_request") is False

    def test_pull_request_default(self):
        assert should_run_evaluation(CIConfig(), "pull_request") is True

    def test_schedule_trigger(self):
        assert should_run_evaluation(CIConfig(trigger="schedule"), "schedule") is True

    def test_schedule_vs_push(self):
        assert should_run_evaluation(CIConfig(trigger="schedule"), "push") is False
