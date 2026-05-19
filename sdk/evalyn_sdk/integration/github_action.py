from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionInput:
    """Input configuration for the Evalyn GitHub Action."""

    dataset_path: str = "data/"
    metrics: list[str] = field(default_factory=lambda: ["output_nonempty"])
    provider: str = "gemini"
    profile: str = "smoke-test"
    fail_on_regression: bool = True
    regression_threshold: float = 0.05
    compare_with_baseline: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "metrics": list(self.metrics),
            "provider": self.provider,
            "profile": self.profile,
            "fail_on_regression": self.fail_on_regression,
            "regression_threshold": self.regression_threshold,
            "compare_with_baseline": self.compare_with_baseline,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionInput:
        return cls(
            dataset_path=data.get("dataset_path", "data/"),
            metrics=data.get("metrics", ["output_nonempty"]),
            provider=data.get("provider", "gemini"),
            profile=data.get("profile", "smoke-test"),
            fail_on_regression=data.get("fail_on_regression", True),
            regression_threshold=data.get("regression_threshold", 0.05),
            compare_with_baseline=data.get("compare_with_baseline", True),
        )


@dataclass
class ActionOutput:
    """Output from an Evalyn GitHub Action run."""

    passed: bool = False
    pass_rate: float = 0.0
    regressions: list[str] = field(default_factory=list)
    summary_markdown: str = ""
    badge_url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "regressions": list(self.regressions),
            "summary_markdown": self.summary_markdown,
            "badge_url": self.badge_url,
        }

    def format_text(self) -> str:
        lines: list[str] = []
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"Evalyn Action: {status}")
        lines.append("-" * 40)
        lines.append(f"  Pass rate: {self.pass_rate:.1%}")
        if self.regressions:
            lines.append(f"  Regressions: {', '.join(self.regressions)}")
        else:
            lines.append("  Regressions: none")
        if self.summary_markdown:
            lines.append(f"  Summary: {self.summary_markdown}")
        if self.badge_url:
            lines.append(f"  Badge: {self.badge_url}")
        return "\n".join(lines)


@dataclass
class ActionYaml:
    """Representation of an action.yml file."""

    name: str = "Evalyn Evaluation"
    description: str = "Run evalyn evaluation on PR"
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    runs_steps: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": dict(self.inputs),
            "runs_steps": list(self.runs_steps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionYaml:
        return cls(
            name=data.get("name", "Evalyn Evaluation"),
            description=data.get("description", "Run evalyn evaluation on PR"),
            inputs=data.get("inputs", {}),
            runs_steps=data.get("runs_steps", []),
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def generate_action_yaml(config: ActionInput) -> str:
    """Generate action.yml content for a reusable GitHub Action.

    Includes steps for: checkout, setup-python, install evalyn, run
    evaluation, and post results.
    """
    metrics_str = " ".join(config.metrics)

    lines = [
        "name: Evalyn Evaluation",
        "description: Run evalyn evaluation on PR",
        "",
        "inputs:",
        "  dataset_path:",
        "    description: Path to evaluation dataset",
        f"    default: '{config.dataset_path}'",
        "  provider:",
        "    description: LLM provider to use",
        f"    default: '{config.provider}'",
        "  profile:",
        "    description: Evaluation profile",
        f"    default: '{config.profile}'",
        "",
        "runs:",
        "  using: composite",
        "  steps:",
        "    - name: Checkout",
        "      uses: actions/checkout@v4",
        "",
        "    - name: Set up Python",
        "      uses: actions/setup-python@v5",
        "      with:",
        "        python-version: '3.13'",
        "",
        "    - name: Install evalyn",
        "      shell: bash",
        "      run: pip install evalyn",
        "",
        "    - name: Run evaluation",
        "      shell: bash",
        "      env:",
        "        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}",
        "      run: |",
        f"        evalyn run ${{{{ inputs.dataset_path }}}} --metrics {metrics_str} --provider ${{{{ inputs.provider }}}}",
        "",
        "    - name: Post results",
        "      shell: bash",
        "      run: evalyn analyze",
    ]

    return "\n".join(lines) + "\n"


def generate_workflow_yaml(config: ActionInput) -> str:
    """Generate .github/workflows/evalyn.yml that uses the action."""
    metrics_str = " ".join(config.metrics)

    lines = [
        "name: Evalyn Evaluation",
        "",
        "on:",
        "  pull_request:",
        "",
        "jobs:",
        "  evaluate:",
        "    runs-on: ubuntu-latest",
        "    steps:",
        "      - uses: actions/checkout@v4",
        "",
        "      - name: Set up Python",
        "        uses: actions/setup-python@v5",
        "        with:",
        "          python-version: '3.13'",
        "",
        "      - name: Install evalyn",
        "        run: pip install evalyn",
        "",
        "      - name: Run evaluation",
        "        env:",
        "          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}",
        "        run: |",
        f"          evalyn run {config.dataset_path} --metrics {metrics_str} --provider {config.provider}",
    ]

    if config.compare_with_baseline:
        lines.append("")
        lines.append("      - name: Compare with baseline")
        lines.append("        run: |")
        lines.append(
            f"          evalyn compare --threshold {config.regression_threshold}"
        )

    return "\n".join(lines) + "\n"


def create_status_badge(pass_rate: float) -> str:
    """Generate shields.io badge URL based on pass rate.

    Green if > 0.8, yellow if > 0.6, red otherwise.
    """
    pct = f"{pass_rate:.0%}"
    if pass_rate > 0.8:
        color = "green"
    elif pass_rate > 0.6:
        color = "yellow"
    else:
        color = "red"
    return f"https://img.shields.io/badge/evalyn-{pct}-{color}"


def format_pr_summary(output: ActionOutput) -> str:
    """Markdown summary for PR comment with pass rate, regressions, badge."""
    status = "Passed" if output.passed else "Failed"
    lines = [
        f"## Evalyn Evaluation: {status}",
        "",
    ]

    if output.badge_url:
        lines.append(f"![badge]({output.badge_url})")
        lines.append("")

    lines.append(f"**Pass rate:** {output.pass_rate:.1%}")
    lines.append("")

    if output.regressions:
        lines.append("**Regressions:**")
        for r in output.regressions:
            lines.append(f"- {r}")
        lines.append("")

    if output.summary_markdown:
        lines.append(output.summary_markdown)
        lines.append("")

    return "\n".join(lines)


def parse_action_inputs(env_vars: dict[str, str]) -> ActionInput:
    """Parse INPUT_* environment variables that GitHub Actions sets.

    GitHub Actions sets inputs as INPUT_<NAME> environment variables
    (uppercased, with hyphens replaced by underscores).
    """
    dataset_path = env_vars.get("INPUT_DATASET_PATH", "data/")
    metrics_raw = env_vars.get("INPUT_METRICS", "output_nonempty")
    metrics = [m.strip() for m in metrics_raw.split(",") if m.strip()]
    provider = env_vars.get("INPUT_PROVIDER", "gemini")
    profile = env_vars.get("INPUT_PROFILE", "smoke-test")
    fail_on_regression = env_vars.get("INPUT_FAIL_ON_REGRESSION", "true").lower() == "true"
    regression_threshold = float(
        env_vars.get("INPUT_REGRESSION_THRESHOLD", "0.05")
    )
    compare_with_baseline = (
        env_vars.get("INPUT_COMPARE_WITH_BASELINE", "true").lower() == "true"
    )

    return ActionInput(
        dataset_path=dataset_path,
        metrics=metrics,
        provider=provider,
        profile=profile,
        fail_on_regression=fail_on_regression,
        regression_threshold=regression_threshold,
        compare_with_baseline=compare_with_baseline,
    )
