"""Tests for circuit_breaker module."""

from __future__ import annotations

import time
from unittest.mock import patch

from evalyn_sdk.evaluation.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStatus,
    create_circuit_breaker,
    should_skip_provider,
)


# -- CircuitBreakerConfig --


def test_config_defaults():
    cfg = CircuitBreakerConfig()
    assert cfg.failure_threshold == 5
    assert cfg.reset_timeout_seconds == 60.0
    assert cfg.half_open_max_calls == 1


def test_config_as_dict():
    cfg = CircuitBreakerConfig(failure_threshold=3, reset_timeout_seconds=30.0)
    d = cfg.as_dict()
    assert d["failure_threshold"] == 3
    assert d["reset_timeout_seconds"] == 30.0
    assert d["half_open_max_calls"] == 1


def test_config_from_dict():
    d = {"failure_threshold": 10, "reset_timeout_seconds": 120.0, "half_open_max_calls": 2}
    cfg = CircuitBreakerConfig.from_dict(d)
    assert cfg.failure_threshold == 10
    assert cfg.reset_timeout_seconds == 120.0
    assert cfg.half_open_max_calls == 2


def test_config_from_dict_defaults():
    cfg = CircuitBreakerConfig.from_dict({})
    assert cfg.failure_threshold == 5
    assert cfg.reset_timeout_seconds == 60.0


def test_config_roundtrip():
    cfg = CircuitBreakerConfig(failure_threshold=7, reset_timeout_seconds=45.0, half_open_max_calls=3)
    restored = CircuitBreakerConfig.from_dict(cfg.as_dict())
    assert restored.failure_threshold == cfg.failure_threshold
    assert restored.reset_timeout_seconds == cfg.reset_timeout_seconds
    assert restored.half_open_max_calls == cfg.half_open_max_calls


# -- CircuitBreakerStatus --


def test_status_defaults():
    s = CircuitBreakerStatus()
    assert s.state == "closed"
    assert s.consecutive_failures == 0
    assert s.total_failures == 0
    assert s.total_successes == 0
    assert s.last_failure_time is None
    assert s.opened_at is None


def test_status_as_dict():
    s = CircuitBreakerStatus(state="open", consecutive_failures=5, total_failures=8)
    d = s.as_dict()
    assert d["state"] == "open"
    assert d["consecutive_failures"] == 5
    assert d["total_failures"] == 8


def test_status_format_text_closed():
    s = CircuitBreakerStatus(state="closed", total_successes=10)
    text = s.format_text()
    assert "closed" in text
    assert "10" in text
    assert "Opened at" not in text


def test_status_format_text_open():
    s = CircuitBreakerStatus(state="open", last_failure_time=100.5, opened_at=99.0)
    text = s.format_text()
    assert "open" in text
    assert "100.5" in text
    assert "99.0" in text


# -- CircuitBreaker: basic states --


def test_breaker_starts_closed():
    b = CircuitBreaker("openai", CircuitBreakerConfig())
    assert b.is_closed()
    assert not b.is_open()
    assert not b.is_half_open()
    assert b.can_execute()


def test_breaker_opens_after_threshold():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=3))
    for _ in range(3):
        b.record_failure()
    assert b.is_open()
    assert not b.can_execute()


def test_breaker_stays_closed_below_threshold():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=3))
    b.record_failure()
    b.record_failure()
    assert b.is_closed()
    assert b.can_execute()


def test_success_resets_consecutive_failures():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=3))
    b.record_failure()
    b.record_failure()
    b.record_success()
    # Should still be closed, consecutive failures reset
    assert b.is_closed()
    b.record_failure()
    b.record_failure()
    # Still below threshold (2 consecutive, not 3)
    assert b.is_closed()


def test_can_execute_false_when_open():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=1))
    b.record_failure()
    assert b.is_open()
    assert not b.can_execute()


# -- half_open transitions --


def test_transitions_to_half_open_after_timeout():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=1, reset_timeout_seconds=0.01))
    b.record_failure()
    assert b.is_open()
    time.sleep(0.02)
    assert b.can_execute()
    assert b.is_half_open()


def test_success_in_half_open_closes():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=1, reset_timeout_seconds=0.01))
    b.record_failure()
    time.sleep(0.02)
    b.can_execute()  # triggers transition to half_open
    assert b.is_half_open()
    b.record_success()
    assert b.is_closed()


def test_failure_in_half_open_reopens():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=1, reset_timeout_seconds=0.01))
    b.record_failure()
    time.sleep(0.02)
    b.can_execute()  # triggers transition to half_open
    assert b.is_half_open()
    b.record_failure()
    assert b.is_open()


def test_half_open_max_calls():
    b = CircuitBreaker(
        "openai",
        CircuitBreakerConfig(failure_threshold=1, reset_timeout_seconds=0.01, half_open_max_calls=2),
    )
    b.record_failure()
    time.sleep(0.02)
    # First call: transitions to half_open, allowed
    assert b.can_execute()
    assert b.is_half_open()
    # Second call: still under max
    assert b.can_execute()
    # We haven't recorded success/failure yet, so _half_open_calls is still 0
    # The max_calls limit restricts how many calls can happen before a success/failure verdict


# -- reset --


def test_reset_forces_closed():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=1))
    b.record_failure()
    assert b.is_open()
    b.reset()
    assert b.is_closed()
    assert b.can_execute()
    status = b.get_status()
    assert status.consecutive_failures == 0


# -- get_status --


def test_get_status_reflects_state():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=3))
    b.record_success()
    b.record_failure()
    status = b.get_status()
    assert status.state == "closed"
    assert status.total_successes == 1
    assert status.total_failures == 1
    assert status.consecutive_failures == 1


def test_get_status_open():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=2))
    b.record_failure()
    b.record_failure()
    status = b.get_status()
    assert status.state == "open"
    assert status.opened_at is not None
    assert status.last_failure_time is not None


# -- factory --


def test_create_circuit_breaker():
    b = create_circuit_breaker("anthropic", failure_threshold=3)
    assert b.provider == "anthropic"
    assert b.config.failure_threshold == 3
    assert b.is_closed()


def test_create_circuit_breaker_defaults():
    b = create_circuit_breaker("gemini")
    assert b.config.failure_threshold == 5


# -- should_skip_provider --


def test_should_skip_provider_closed():
    b = create_circuit_breaker("openai", failure_threshold=3)
    skip, reason = should_skip_provider(b)
    assert not skip
    assert reason == ""


def test_should_skip_provider_open():
    b = create_circuit_breaker("openai", failure_threshold=2)
    b.record_failure()
    b.record_failure()
    skip, reason = should_skip_provider(b)
    assert skip
    assert "openai" in reason
    assert "consecutive failures" in reason


# -- edge cases --


def test_zero_threshold_opens_immediately():
    """A threshold of 0 means no failures allowed - but we need at least 1 to open."""
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=0))
    # With threshold 0, any failure opens (0 >= 0)
    b.record_failure()
    assert b.is_open()


def test_single_failure_threshold():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=1))
    assert b.is_closed()
    b.record_failure()
    assert b.is_open()


def test_many_successes_no_open():
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=2))
    for _ in range(100):
        b.record_success()
    assert b.is_closed()
    status = b.get_status()
    assert status.total_successes == 100
    assert status.total_failures == 0


def test_total_failures_accumulate():
    """Total failures keep counting even after resets via success."""
    b = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=5))
    b.record_failure()
    b.record_failure()
    b.record_success()  # resets consecutive but not total
    b.record_failure()
    status = b.get_status()
    assert status.total_failures == 3
    assert status.consecutive_failures == 1


def test_provider_name_preserved():
    b = CircuitBreaker("my-custom-provider", CircuitBreakerConfig())
    assert b.provider == "my-custom-provider"
