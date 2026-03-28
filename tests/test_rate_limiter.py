"""Tests for rate_limiter module."""

from __future__ import annotations

import time

import pytest

from evalyn_sdk.evaluation.rate_limiter import (
    PROVIDER_DEFAULTS,
    RateLimitConfig,
    RateLimitStatus,
    RateLimiter,
    create_rate_limiter,
    get_provider_limits,
)


# --- RateLimitConfig ---


def test_config_defaults():
    cfg = RateLimitConfig()
    assert cfg.requests_per_minute == 60
    assert cfg.tokens_per_minute == 100000
    assert cfg.burst_limit == 10
    assert cfg.provider == ""


def test_config_as_dict():
    cfg = RateLimitConfig(requests_per_minute=30, provider="openai")
    d = cfg.as_dict()
    assert d["requests_per_minute"] == 30
    assert d["provider"] == "openai"


def test_config_from_dict():
    d = {"requests_per_minute": 100, "tokens_per_minute": 500000, "burst_limit": 20}
    cfg = RateLimitConfig.from_dict(d)
    assert cfg.requests_per_minute == 100
    assert cfg.tokens_per_minute == 500000
    assert cfg.burst_limit == 20


def test_config_roundtrip():
    original = RateLimitConfig(
        requests_per_minute=45, tokens_per_minute=200000, burst_limit=5, provider="gemini"
    )
    restored = RateLimitConfig.from_dict(original.as_dict())
    assert restored.requests_per_minute == original.requests_per_minute
    assert restored.tokens_per_minute == original.tokens_per_minute
    assert restored.burst_limit == original.burst_limit
    assert restored.provider == original.provider


def test_config_from_dict_defaults():
    cfg = RateLimitConfig.from_dict({})
    assert cfg.requests_per_minute == 60
    assert cfg.tokens_per_minute == 100000


# --- RateLimitStatus ---


def test_status_defaults():
    status = RateLimitStatus()
    assert status.current_rpm == 0
    assert status.throttled is False
    assert status.wait_ms == 0.0


def test_status_as_dict():
    status = RateLimitStatus(current_rpm=10, throttled=True, wait_ms=500.0)
    d = status.as_dict()
    assert d["current_rpm"] == 10
    assert d["throttled"] is True
    assert d["wait_ms"] == 500.0


def test_status_format_text_not_throttled():
    status = RateLimitStatus(current_rpm=5, remaining_rpm=55, current_tpm=1000, remaining_tpm=99000)
    text = status.format_text()
    assert "RPM: 5" in text
    assert "TPM: 1000" in text
    assert "THROTTLED" not in text


def test_status_format_text_throttled():
    status = RateLimitStatus(throttled=True, wait_ms=1234.0)
    text = status.format_text()
    assert "THROTTLED" in text
    assert "1234" in text


# --- PROVIDER_DEFAULTS ---


def test_provider_defaults_has_three_providers():
    assert len(PROVIDER_DEFAULTS) == 3
    assert "gemini" in PROVIDER_DEFAULTS
    assert "openai" in PROVIDER_DEFAULTS
    assert "anthropic" in PROVIDER_DEFAULTS


def test_provider_defaults_gemini():
    cfg = PROVIDER_DEFAULTS["gemini"]
    assert cfg.requests_per_minute == 60
    assert cfg.tokens_per_minute == 1000000


def test_provider_defaults_openai():
    cfg = PROVIDER_DEFAULTS["openai"]
    assert cfg.requests_per_minute == 60
    assert cfg.tokens_per_minute == 150000


def test_provider_defaults_anthropic():
    cfg = PROVIDER_DEFAULTS["anthropic"]
    assert cfg.requests_per_minute == 50
    assert cfg.tokens_per_minute == 100000


# --- RateLimiter ---


def test_limiter_check_initial_status():
    limiter = RateLimiter(RateLimitConfig())
    status = limiter.check()
    assert status.current_rpm == 0
    assert status.current_tpm == 0
    assert status.throttled is False
    assert status.wait_ms == 0.0


def test_limiter_record_request_increments():
    limiter = RateLimiter(RateLimitConfig())
    limiter.record_request(tokens=100)
    status = limiter.check()
    assert status.current_rpm == 1
    assert status.current_tpm == 100


def test_limiter_record_multiple_requests():
    limiter = RateLimiter(RateLimitConfig())
    limiter.record_request(tokens=50)
    limiter.record_request(tokens=75)
    limiter.record_request()  # no tokens
    status = limiter.check()
    assert status.current_rpm == 3
    assert status.current_tpm == 125


def test_limiter_should_wait_zero_when_under_limit():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=100))
    limiter.record_request()
    assert limiter.should_wait() == 0.0


def test_limiter_should_wait_positive_when_at_rpm_limit():
    cfg = RateLimitConfig(requests_per_minute=3, burst_limit=100)
    limiter = RateLimiter(cfg)
    for _ in range(3):
        limiter.record_request()
    wait = limiter.should_wait()
    assert wait > 0


def test_limiter_should_wait_positive_when_at_tpm_limit():
    cfg = RateLimitConfig(tokens_per_minute=100, requests_per_minute=10000, burst_limit=10000)
    limiter = RateLimiter(cfg)
    limiter.record_request(tokens=100)
    wait = limiter.should_wait()
    assert wait > 0


def test_limiter_reset_clears():
    limiter = RateLimiter(RateLimitConfig())
    limiter.record_request(tokens=500)
    limiter.record_request(tokens=500)
    limiter.reset()
    status = limiter.check()
    assert status.current_rpm == 0
    assert status.current_tpm == 0


def test_limiter_get_stats_returns_dict():
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, provider="openai"))
    limiter.record_request(tokens=200)
    stats = limiter.get_stats()
    assert isinstance(stats, dict)
    assert stats["requests_in_window"] == 1
    assert stats["tokens_in_window"] == 200
    assert stats["provider"] == "openai"
    assert stats["requests_per_minute_limit"] == 60


def test_limiter_get_stats_utilization():
    cfg = RateLimitConfig(requests_per_minute=100, tokens_per_minute=1000)
    limiter = RateLimiter(cfg)
    for _ in range(10):
        limiter.record_request(tokens=50)
    stats = limiter.get_stats()
    assert abs(stats["utilization_rpm"] - 0.1) < 0.01
    assert abs(stats["utilization_tpm"] - 0.5) < 0.01


def test_limiter_remaining_decreases():
    cfg = RateLimitConfig(requests_per_minute=10, tokens_per_minute=1000)
    limiter = RateLimiter(cfg)
    limiter.record_request(tokens=100)
    status = limiter.check()
    assert status.remaining_rpm == 9
    assert status.remaining_tpm == 900


def test_limiter_burst_limit():
    cfg = RateLimitConfig(requests_per_minute=10000, burst_limit=3)
    limiter = RateLimiter(cfg)
    for _ in range(3):
        limiter.record_request()
    wait = limiter.should_wait()
    # Should trigger burst throttle
    assert wait > 0


def test_limiter_check_throttled_flag():
    cfg = RateLimitConfig(requests_per_minute=2, burst_limit=100)
    limiter = RateLimiter(cfg)
    limiter.record_request()
    limiter.record_request()
    status = limiter.check()
    assert status.throttled is True


# --- Edge cases ---


def test_limiter_zero_rpm_limit():
    cfg = RateLimitConfig(requests_per_minute=0, tokens_per_minute=100, burst_limit=100)
    limiter = RateLimiter(cfg)
    limiter.record_request()
    status = limiter.check()
    # With 0 limit, remaining should be 0 (clamped)
    assert status.remaining_rpm == 0


def test_limiter_zero_tpm_limit():
    cfg = RateLimitConfig(requests_per_minute=100, tokens_per_minute=0, burst_limit=100)
    limiter = RateLimiter(cfg)
    limiter.record_request(tokens=10)
    status = limiter.check()
    assert status.remaining_tpm == 0


def test_limiter_no_tokens_recorded():
    limiter = RateLimiter(RateLimitConfig())
    limiter.record_request()
    limiter.record_request()
    status = limiter.check()
    assert status.current_tpm == 0
    assert status.current_rpm == 2


# --- get_provider_limits ---


def test_get_provider_limits_known():
    cfg = get_provider_limits("openai")
    assert cfg.requests_per_minute == 60
    assert cfg.tokens_per_minute == 150000
    assert cfg.provider == "openai"


def test_get_provider_limits_unknown():
    cfg = get_provider_limits("unknown_provider")
    assert cfg.provider == "unknown_provider"
    # Falls back to defaults
    assert cfg.requests_per_minute == 60
    assert cfg.tokens_per_minute == 100000


# --- create_rate_limiter ---


def test_create_rate_limiter_with_provider():
    limiter = create_rate_limiter(provider="anthropic")
    assert isinstance(limiter, RateLimiter)
    assert limiter.config.requests_per_minute == 50


def test_create_rate_limiter_with_config():
    cfg = RateLimitConfig(requests_per_minute=999)
    limiter = create_rate_limiter(config=cfg)
    assert limiter.config.requests_per_minute == 999


def test_create_rate_limiter_default():
    limiter = create_rate_limiter()
    assert isinstance(limiter, RateLimiter)
    assert limiter.config.requests_per_minute == 60


def test_create_rate_limiter_config_overrides_provider():
    cfg = RateLimitConfig(requests_per_minute=123)
    limiter = create_rate_limiter(provider="openai", config=cfg)
    # config takes precedence over provider
    assert limiter.config.requests_per_minute == 123
