from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CIConfig:
    """Configuration for CI/CD evaluation runs."""

    trigger: str = "pull_request"  # "pull_request" / "push" / "schedule"
    dataset_path: str = "data/"
    metrics: list[str] = field(default_factory=lambda: ["output_nonempty"])
    fail_on_regression: bool = True
    regression_threshold: float = 0.05
    provider: str = "gemini"

    def as_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "dataset_path": self.dataset_path,
            "metrics": list(self.metrics),
            "fail_on_regression": self.fail_on_regression,
            "regression_threshold": self.regression_threshold,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CIConfig:
        return cls(
            trigger=data.get("trigger", "pull_request"),
            dataset_path=data.get("dataset_path", "data/"),
            metrics=data.get("metrics", ["output_nonempty"]),
            fail_on_regression=data.get("fail_on_regression", True),
            regression_threshold=data.get("regression_threshold", 0.05),
            provider=data.get("provider", "gemini"),
        )


@dataclass
class CIResult:
    """Result of a CI evaluation run."""

    passed: bool = False
    pass_rate: float = 0.0
    regressions: list[str] = field(default_factory=list)
    summary: str = ""
    exit_code: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "regressions": list(self.regressions),
            "summary": self.summary,
            "exit_code": self.exit_code,
        }

    def format_text(self) -> str:
        lines: list[str] = []
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"CI Evaluation: {status}")
        lines.append("-" * 40)
        lines.append(f"  Pass rate: {self.pass_rate:.1%}")
        if self.regressions:
            lines.append(f"  Regressions: {', '.join(self.regressions)}")
        else:
            lines.append("  Regressions: none")
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        lines.append(f"  Exit code: {self.exit_code}")
        return "\n".join(lines)


@dataclass
class GitHubActionConfig:
    """Configuration for GitHub Actions workflow generation."""

    name: str = "evalyn-eval"
    runs_on: str = "ubuntu-latest"
    python_version: str = "3.13"
    evalyn_version: str = "latest"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runs_on": self.runs_on,
            "python_version": self.python_version,
            "evalyn_version": self.evalyn_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GitHubActionConfig:
        return cls(
            name=data.get("name", "evalyn-eval"),
            runs_on=data.get("runs_on", "ubuntu-latest"),
            python_version=data.get("python_version", "3.13"),
            evalyn_version=data.get("evalyn_version", "latest"),
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def generate_github_action_yaml(
    config: CIConfig,
    action_config: GitHubActionConfig | None = None,
) -> str:
    """Generate a complete .github/workflows/evalyn.yml file content."""
    if action_config is None:
        action_config = GitHubActionConfig()

    install_line = "pip install evalyn"
    if action_config.evalyn_version != "latest":
        install_line = f"pip install evalyn=={action_config.evalyn_version}"

    metrics_str = " ".join(config.metrics)

    lines = [
        f"name: {action_config.name}",
        "",
        "on:",
        f"  {config.trigger}:",
        "",
        "jobs:",
        "  evaluate:",
        f"    runs-on: {action_config.runs_on}",
        "    steps:",
        "      - uses: actions/checkout@v4",
        "",
        f"      - name: Set up Python {action_config.python_version}",
        "        uses: actions/setup-python@v5",
        "        with:",
        f"          python-version: '{action_config.python_version}'",
        "",
        "      - name: Install evalyn",
        f"        run: {install_line}",
        "",
        "      - name: Run evaluation",
        "        env:",
        "          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}",
        "        run: |",
        f"          evalyn run {config.dataset_path} --metrics {metrics_str}",
    ]

    if config.fail_on_regression:
        lines.append(
            f"          evalyn compare --threshold {config.regression_threshold}"
        )

    return "\n".join(lines) + "\n"


def check_ci_result(
    pass_rate: float,
    regressions: list[str],
    config: CIConfig,
) -> CIResult:
    """Determine pass/fail based on config thresholds."""
    has_regressions = len(regressions) > 0 and config.fail_on_regression
    passed = pass_rate >= (1.0 - config.regression_threshold) and not has_regressions

    if passed:
        summary = f"All checks passed with {pass_rate:.1%} pass rate."
        exit_code = 0
    else:
        parts: list[str] = []
        if pass_rate < (1.0 - config.regression_threshold):
            parts.append(f"pass rate {pass_rate:.1%} below threshold")
        if has_regressions:
            parts.append(f"{len(regressions)} regression(s) detected")
        summary = "Failed: " + "; ".join(parts) + "."
        exit_code = 1

    return CIResult(
        passed=passed,
        pass_rate=pass_rate,
        regressions=list(regressions),
        summary=summary,
        exit_code=exit_code,
    )


def generate_pr_comment(result: CIResult) -> str:
    """Generate markdown comment for a GitHub PR."""
    status = "Passed" if result.passed else "Failed"
    lines = [
        f"## Evalyn Evaluation: {status}",
        "",
        f"**Pass rate:** {result.pass_rate:.1%}",
        "",
    ]

    if result.regressions:
        lines.append("**Regressions:**")
        for r in result.regressions:
            lines.append(f"- {r}")
        lines.append("")

    if result.summary:
        lines.append(f"> {result.summary}")
        lines.append("")

    return "\n".join(lines)


def parse_ci_environment() -> dict[str, str]:
    """Read CI environment variables. Return dict of available vars."""
    keys = ["GITHUB_SHA", "GITHUB_REF", "GITHUB_EVENT_NAME"]
    result: dict[str, str] = {}
    for key in keys:
        val = os.environ.get(key)
        if val is not None:
            result[key] = val
    return result


def should_run_evaluation(config: CIConfig, event_name: str) -> bool:
    """Check if evaluation should run for this event."""
    return config.trigger == event_name
