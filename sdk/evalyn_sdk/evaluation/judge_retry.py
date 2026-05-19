"""Judge output retry: automatically retry judge calls on parse failure.

Provides retry simulation, prompt augmentation, and aggregate statistics
for tracking parse reliability across evaluation runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RetryConfig:
    """Configuration for judge output retry behavior."""

    max_retries: int = 2
    retry_prompt_suffix: str = "Please respond with valid JSON."
    backoff_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "retry_prompt_suffix": self.retry_prompt_suffix,
            "backoff_ms": self.backoff_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryConfig:
        return cls(
            max_retries=data.get("max_retries", 2),
            retry_prompt_suffix=data.get(
                "retry_prompt_suffix", "Please respond with valid JSON."
            ),
            backoff_ms=data.get("backoff_ms", 0.0),
        )


@dataclass
class RetryAttempt:
    """Record of a single parse attempt."""

    attempt_number: int
    raw_output: str
    parse_success: bool
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "raw_output": self.raw_output,
            "parse_success": self.parse_success,
            "error": self.error,
        }


@dataclass
class RetryResult:
    """Outcome of retry logic for a single judge call."""

    final_output: dict[str, Any] | None = None
    attempts: list[RetryAttempt] = field(default_factory=list)
    total_attempts: int = 0
    success: bool = False
    parse_failure_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_output": self.final_output,
            "attempts": [a.as_dict() for a in self.attempts],
            "total_attempts": self.total_attempts,
            "success": self.success,
            "parse_failure_rate": round(self.parse_failure_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryResult:
        attempts = [
            RetryAttempt(
                attempt_number=a["attempt_number"],
                raw_output=a["raw_output"],
                parse_success=a["parse_success"],
                error=a.get("error", ""),
            )
            for a in data.get("attempts", [])
        ]
        return cls(
            final_output=data.get("final_output"),
            attempts=attempts,
            total_attempts=data.get("total_attempts", 0),
            success=data.get("success", False),
            parse_failure_rate=data.get("parse_failure_rate", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            f"Success: {self.success}",
            f"Total attempts: {self.total_attempts}",
            f"Parse failure rate: {self.parse_failure_rate:.1%}",
        ]
        if self.final_output is not None:
            lines.append(f"Final output: {self.final_output}")
        for a in self.attempts:
            status = "ok" if a.parse_success else f"fail ({a.error})"
            lines.append(f"  Attempt {a.attempt_number}: {status}")
        return "\n".join(lines)


@dataclass
class RetryStats:
    """Aggregate retry statistics across multiple judge calls."""

    total_calls: int = 0
    total_retries: int = 0
    success_on_first: int = 0
    success_on_retry: int = 0
    total_failures: int = 0

    @property
    def retry_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_retries / self.total_calls

    @property
    def first_attempt_success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.success_on_first / self.total_calls

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_retries": self.total_retries,
            "success_on_first": self.success_on_first,
            "success_on_retry": self.success_on_retry,
            "total_failures": self.total_failures,
            "retry_rate": round(self.retry_rate, 4),
            "first_attempt_success_rate": round(self.first_attempt_success_rate, 4),
        }

    def format_text(self) -> str:
        return "\n".join([
            f"Total calls: {self.total_calls}",
            f"Total retries: {self.total_retries}",
            f"Success on first: {self.success_on_first}",
            f"Success on retry: {self.success_on_retry}",
            f"Total failures: {self.total_failures}",
            f"Retry rate: {self.retry_rate:.1%}",
            f"First attempt success rate: {self.first_attempt_success_rate:.1%}",
        ])


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def attempt_parse(output: str) -> tuple[bool, dict[str, Any] | None, str]:
    """Try to parse JSON from output.

    Returns (success, data, error_msg).
    """
    if not output or not output.strip():
        return (False, None, "empty output")

    stripped = output.strip()

    # Direct parse
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return (True, data, "")
    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if code_block:
        try:
            data = json.loads(code_block.group(1).strip())
            if isinstance(data, dict):
                return (True, data, "")
        except (json.JSONDecodeError, ValueError):
            pass

    # Embedded JSON object
    brace_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return (True, data, "")
        except (json.JSONDecodeError, ValueError):
            pass

    return (False, None, "no valid JSON found")


def build_retry_prompt(original_prompt: str, config: RetryConfig) -> str:
    """Append retry suffix to the original prompt."""
    return f"{original_prompt}\n{config.retry_prompt_suffix}"


def simulate_retry(outputs: list[str], config: RetryConfig) -> RetryResult:
    """Simulate retry logic on a list of outputs (one per attempt).

    Parses each output in order, stops on the first success. At most
    config.max_retries + 1 attempts are used (1 initial + max_retries).
    """
    attempts: list[RetryAttempt] = []
    max_total = config.max_retries + 1
    final_output: dict[str, Any] | None = None
    success = False
    failures = 0

    for i, text in enumerate(outputs):
        if i >= max_total:
            break
        ok, data, error = attempt_parse(text)
        attempt = RetryAttempt(
            attempt_number=i + 1,
            raw_output=text,
            parse_success=ok,
            error=error,
        )
        attempts.append(attempt)
        if ok:
            final_output = data
            success = True
            break
        failures += 1

    total = len(attempts)
    failure_rate = failures / total if total > 0 else 0.0

    return RetryResult(
        final_output=final_output,
        attempts=attempts,
        total_attempts=total,
        success=success,
        parse_failure_rate=failure_rate,
    )


def compute_retry_stats(results: list[RetryResult]) -> RetryStats:
    """Aggregate statistics across multiple retry results."""
    stats = RetryStats()
    for r in results:
        stats.total_calls += 1
        retries = r.total_attempts - 1 if r.total_attempts > 0 else 0
        stats.total_retries += retries
        if r.success:
            if r.total_attempts == 1:
                stats.success_on_first += 1
            else:
                stats.success_on_retry += 1
        else:
            stats.total_failures += 1
    return stats


def should_retry(attempt: RetryAttempt, config: RetryConfig) -> bool:
    """True if parse failed and attempt number has not exceeded max_retries."""
    if attempt.parse_success:
        return False
    return attempt.attempt_number <= config.max_retries
