"""Tests for the judge output retry module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.evaluation.judge_retry import (
    RetryConfig,
    RetryAttempt,
    RetryResult,
    RetryStats,
    attempt_parse,
    build_retry_prompt,
    simulate_retry,
    compute_retry_stats,
    should_retry,
)


# ---------------------------------------------------------------------------
# attempt_parse
# ---------------------------------------------------------------------------


class TestAttemptParse:
    def test_valid_json(self):
        ok, data, error = attempt_parse('{"score": 0.9}')
        assert ok is True
        assert data == {"score": 0.9}
        assert error == ""

    def test_invalid_json(self):
        ok, data, error = attempt_parse("not json")
        assert ok is False
        assert data is None
        assert error != ""

    def test_empty_string(self):
        ok, data, error = attempt_parse("")
        assert ok is False
        assert data is None
        assert error == "empty output"

    def test_json_in_code_block(self):
        ok, data, _ = attempt_parse('```json\n{"x": 1}\n```')
        assert ok is True
        assert data == {"x": 1}

    def test_json_embedded_in_text(self):
        ok, data, _ = attempt_parse('Result: {"score": 0.5} end')
        assert ok is True
        assert data == {"score": 0.5}

    def test_array_not_accepted(self):
        ok, data, _ = attempt_parse("[1, 2]")
        assert ok is False


# ---------------------------------------------------------------------------
# build_retry_prompt
# ---------------------------------------------------------------------------


class TestBuildRetryPrompt:
    def test_appends_suffix(self):
        config = RetryConfig(retry_prompt_suffix="Return JSON only.")
        result = build_retry_prompt("Evaluate this", config)
        assert result == "Evaluate this\nReturn JSON only."

    def test_default_suffix(self):
        config = RetryConfig()
        result = build_retry_prompt("Prompt", config)
        assert "Please respond with valid JSON." in result

    def test_preserves_original(self):
        config = RetryConfig()
        result = build_retry_prompt("original text", config)
        assert result.startswith("original text")


# ---------------------------------------------------------------------------
# simulate_retry
# ---------------------------------------------------------------------------


class TestSimulateRetry:
    def test_success_first_attempt(self):
        outputs = ['{"score": 0.8}', '{"score": 0.9}']
        config = RetryConfig(max_retries=2)
        result = simulate_retry(outputs, config)
        assert result.success is True
        assert result.total_attempts == 1
        assert result.final_output == {"score": 0.8}

    def test_success_on_retry(self):
        outputs = ["bad output", '{"score": 0.7}']
        config = RetryConfig(max_retries=2)
        result = simulate_retry(outputs, config)
        assert result.success is True
        assert result.total_attempts == 2
        assert result.final_output == {"score": 0.7}

    def test_all_fail(self):
        outputs = ["bad", "also bad", "still bad"]
        config = RetryConfig(max_retries=2)
        result = simulate_retry(outputs, config)
        assert result.success is False
        assert result.total_attempts == 3
        assert result.final_output is None

    def test_respects_max_retries(self):
        outputs = ["fail"] * 10
        config = RetryConfig(max_retries=1)
        result = simulate_retry(outputs, config)
        assert result.total_attempts == 2  # 1 initial + 1 retry

    def test_empty_outputs(self):
        result = simulate_retry([], RetryConfig())
        assert result.success is False
        assert result.total_attempts == 0
        assert result.attempts == []

    def test_single_attempt_success(self):
        outputs = ['{"ok": true}']
        config = RetryConfig(max_retries=0)
        result = simulate_retry(outputs, config)
        assert result.success is True
        assert result.total_attempts == 1

    def test_single_attempt_failure(self):
        outputs = ["nope"]
        config = RetryConfig(max_retries=0)
        result = simulate_retry(outputs, config)
        assert result.success is False
        assert result.total_attempts == 1

    def test_parse_failure_rate(self):
        outputs = ["bad", '{"x": 1}']
        config = RetryConfig(max_retries=2)
        result = simulate_retry(outputs, config)
        assert result.parse_failure_rate == 0.5


# ---------------------------------------------------------------------------
# compute_retry_stats
# ---------------------------------------------------------------------------


class TestComputeRetryStats:
    def test_all_first_attempt(self):
        results = [
            RetryResult(
                final_output={"a": 1},
                attempts=[RetryAttempt(1, '{"a":1}', True)],
                total_attempts=1,
                success=True,
            ),
            RetryResult(
                final_output={"b": 2},
                attempts=[RetryAttempt(1, '{"b":2}', True)],
                total_attempts=1,
                success=True,
            ),
        ]
        stats = compute_retry_stats(results)
        assert stats.total_calls == 2
        assert stats.success_on_first == 2
        assert stats.total_retries == 0
        assert stats.total_failures == 0

    def test_mixed_results(self):
        results = [
            RetryResult(total_attempts=1, success=True,
                        attempts=[RetryAttempt(1, "", True)]),
            RetryResult(total_attempts=2, success=True,
                        attempts=[RetryAttempt(1, "", False), RetryAttempt(2, "", True)]),
            RetryResult(total_attempts=3, success=False,
                        attempts=[RetryAttempt(1, "", False), RetryAttempt(2, "", False),
                                  RetryAttempt(3, "", False)]),
        ]
        stats = compute_retry_stats(results)
        assert stats.total_calls == 3
        assert stats.success_on_first == 1
        assert stats.success_on_retry == 1
        assert stats.total_failures == 1
        assert stats.total_retries == 3  # 0 + 1 + 2

    def test_empty_results(self):
        stats = compute_retry_stats([])
        assert stats.total_calls == 0
        assert stats.retry_rate == 0.0


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    def test_true_when_failed_and_under_limit(self):
        attempt = RetryAttempt(1, "bad", False, "parse error")
        config = RetryConfig(max_retries=2)
        assert should_retry(attempt, config) is True

    def test_false_when_succeeded(self):
        attempt = RetryAttempt(1, '{"ok": true}', True)
        config = RetryConfig(max_retries=2)
        assert should_retry(attempt, config) is False

    def test_false_when_max_reached(self):
        attempt = RetryAttempt(3, "bad", False, "parse error")
        config = RetryConfig(max_retries=2)
        assert should_retry(attempt, config) is False

    def test_true_at_boundary(self):
        attempt = RetryAttempt(2, "bad", False, "err")
        config = RetryConfig(max_retries=2)
        assert should_retry(attempt, config) is True


# ---------------------------------------------------------------------------
# RetryResult format_text
# ---------------------------------------------------------------------------


class TestRetryResultFormatText:
    def test_contains_key_info(self):
        result = RetryResult(
            final_output={"score": 0.8},
            attempts=[
                RetryAttempt(1, "bad", False, "no JSON"),
                RetryAttempt(2, '{"score":0.8}', True),
            ],
            total_attempts=2,
            success=True,
            parse_failure_rate=0.5,
        )
        text = result.format_text()
        assert "Success: True" in text
        assert "Total attempts: 2" in text
        assert "50.0%" in text
        assert "Attempt 1: fail" in text
        assert "Attempt 2: ok" in text


# ---------------------------------------------------------------------------
# RetryStats properties
# ---------------------------------------------------------------------------


class TestRetryStatsProperties:
    def test_retry_rate(self):
        stats = RetryStats(total_calls=10, total_retries=3)
        assert abs(stats.retry_rate - 0.3) < 1e-9

    def test_first_attempt_success_rate(self):
        stats = RetryStats(total_calls=10, success_on_first=8)
        assert abs(stats.first_attempt_success_rate - 0.8) < 1e-9

    def test_zero_calls(self):
        stats = RetryStats()
        assert stats.retry_rate == 0.0
        assert stats.first_attempt_success_rate == 0.0

    def test_format_text(self):
        stats = RetryStats(
            total_calls=5, total_retries=2, success_on_first=3,
            success_on_retry=1, total_failures=1,
        )
        text = stats.format_text()
        assert "Total calls: 5" in text
        assert "Total retries: 2" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_retry_config_roundtrip(self):
        original = RetryConfig(max_retries=5, retry_prompt_suffix="JSON!", backoff_ms=100.0)
        data = original.as_dict()
        restored = RetryConfig.from_dict(data)
        assert restored.max_retries == 5
        assert restored.retry_prompt_suffix == "JSON!"
        assert restored.backoff_ms == 100.0

    def test_retry_config_defaults(self):
        restored = RetryConfig.from_dict({})
        assert restored.max_retries == 2
        assert restored.backoff_ms == 0.0

    def test_retry_result_roundtrip(self):
        original = RetryResult(
            final_output={"score": 1.0},
            attempts=[RetryAttempt(1, '{"score":1.0}', True)],
            total_attempts=1,
            success=True,
            parse_failure_rate=0.0,
        )
        data = original.as_dict()
        restored = RetryResult.from_dict(data)
        assert restored.success is True
        assert restored.final_output == {"score": 1.0}
        assert len(restored.attempts) == 1
        assert restored.attempts[0].attempt_number == 1

    def test_retry_stats_as_dict(self):
        stats = RetryStats(total_calls=10, total_retries=3, success_on_first=7,
                           success_on_retry=2, total_failures=1)
        data = stats.as_dict()
        assert data["total_calls"] == 10
        assert data["retry_rate"] == 0.3
        assert data["first_attempt_success_rate"] == 0.7
