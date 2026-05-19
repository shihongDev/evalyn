"""Rate limit awareness for LLM provider API calls.

Respects provider rate limits with automatic throttling. Tracks requests
per minute and tokens per minute, returning wait times when limits are
approached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    burst_limit: int = 10
    provider: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "burst_limit": self.burst_limit,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RateLimitConfig:
        return cls(
            requests_per_minute=data.get("requests_per_minute", 60),
            tokens_per_minute=data.get("tokens_per_minute", 100000),
            burst_limit=data.get("burst_limit", 10),
            provider=data.get("provider", ""),
        )


@dataclass
class RateLimitStatus:
    """Current rate limit status snapshot."""

    current_rpm: int = 0
    current_tpm: int = 0
    remaining_rpm: int = 0
    remaining_tpm: int = 0
    throttled: bool = False
    wait_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_rpm": self.current_rpm,
            "current_tpm": self.current_tpm,
            "remaining_rpm": self.remaining_rpm,
            "remaining_tpm": self.remaining_tpm,
            "throttled": self.throttled,
            "wait_ms": self.wait_ms,
        }

    def format_text(self) -> str:
        """Human-readable status summary."""
        lines = [
            f"RPM: {self.current_rpm} (remaining: {self.remaining_rpm})",
            f"TPM: {self.current_tpm} (remaining: {self.remaining_tpm})",
        ]
        if self.throttled:
            lines.append(f"THROTTLED - wait {self.wait_ms:.0f}ms")
        return "\n".join(lines)


PROVIDER_DEFAULTS: dict[str, RateLimitConfig] = {
    "gemini": RateLimitConfig(requests_per_minute=60, tokens_per_minute=1000000),
    "openai": RateLimitConfig(requests_per_minute=60, tokens_per_minute=150000),
    "anthropic": RateLimitConfig(requests_per_minute=50, tokens_per_minute=100000),
}


class RateLimiter:
    """Tracks request and token usage to enforce rate limits.

    Usage:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))
        wait = limiter.should_wait()
        if wait > 0:
            time.sleep(wait / 1000)
        limiter.record_request(tokens=500)
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._requests: list[float] = []
        self._tokens: list[tuple[float, int]] = []

    def _prune(self, now: float) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = now - 60.0
        self._requests = [t for t in self._requests if t > cutoff]
        self._tokens = [(t, c) for t, c in self._tokens if t > cutoff]

    def check(self) -> RateLimitStatus:
        """Check current rate limit status without recording anything."""
        now = time.monotonic()
        self._prune(now)

        current_rpm = len(self._requests)
        current_tpm = sum(c for _, c in self._tokens)
        remaining_rpm = max(0, self.config.requests_per_minute - current_rpm)
        remaining_tpm = max(0, self.config.tokens_per_minute - current_tpm)

        wait_ms = self._compute_wait(now, current_rpm, current_tpm)
        throttled = wait_ms > 0

        return RateLimitStatus(
            current_rpm=current_rpm,
            current_tpm=current_tpm,
            remaining_rpm=remaining_rpm,
            remaining_tpm=remaining_tpm,
            throttled=throttled,
            wait_ms=wait_ms,
        )

    def _compute_wait(self, now: float, current_rpm: int, current_tpm: int) -> float:
        """Compute milliseconds to wait before the next request is safe."""
        wait_ms = 0.0

        # Check RPM limit
        if current_rpm >= self.config.requests_per_minute and self._requests:
            oldest = min(self._requests)
            expire_at = oldest + 60.0
            rpm_wait = max(0.0, (expire_at - now) * 1000)
            wait_ms = max(wait_ms, rpm_wait)

        # Check TPM limit
        if current_tpm >= self.config.tokens_per_minute and self._tokens:
            oldest_t = min(t for t, _ in self._tokens)
            expire_at = oldest_t + 60.0
            tpm_wait = max(0.0, (expire_at - now) * 1000)
            wait_ms = max(wait_ms, tpm_wait)

        # Check burst limit: count requests in the last second
        one_sec_ago = now - 1.0
        recent_burst = sum(1 for t in self._requests if t > one_sec_ago)
        if recent_burst >= self.config.burst_limit and self._requests:
            burst_times = [t for t in self._requests if t > one_sec_ago]
            oldest_burst = min(burst_times)
            burst_wait = max(0.0, (oldest_burst + 1.0 - now) * 1000)
            wait_ms = max(wait_ms, burst_wait)

        return wait_ms

    def record_request(self, tokens: int = 0) -> None:
        """Record a request with optional token count."""
        now = time.monotonic()
        self._requests.append(now)
        if tokens > 0:
            self._tokens.append((now, tokens))

    def should_wait(self) -> float:
        """Return milliseconds to wait before next request. 0 if ok."""
        status = self.check()
        return status.wait_ms

    def reset(self) -> None:
        """Clear all recorded history."""
        self._requests.clear()
        self._tokens.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return current usage statistics."""
        now = time.monotonic()
        self._prune(now)
        current_rpm = len(self._requests)
        current_tpm = sum(c for _, c in self._tokens)
        return {
            "requests_in_window": current_rpm,
            "tokens_in_window": current_tpm,
            "requests_per_minute_limit": self.config.requests_per_minute,
            "tokens_per_minute_limit": self.config.tokens_per_minute,
            "burst_limit": self.config.burst_limit,
            "provider": self.config.provider,
            "utilization_rpm": (
                current_rpm / self.config.requests_per_minute
                if self.config.requests_per_minute > 0
                else 0.0
            ),
            "utilization_tpm": (
                current_tpm / self.config.tokens_per_minute
                if self.config.tokens_per_minute > 0
                else 0.0
            ),
        }


def get_provider_limits(provider: str) -> RateLimitConfig:
    """Look up default rate limits for a known provider.

    Returns the provider-specific defaults if known, otherwise a generic config.
    """
    if provider in PROVIDER_DEFAULTS:
        cfg = PROVIDER_DEFAULTS[provider]
        return RateLimitConfig(
            requests_per_minute=cfg.requests_per_minute,
            tokens_per_minute=cfg.tokens_per_minute,
            burst_limit=cfg.burst_limit,
            provider=provider,
        )
    return RateLimitConfig(provider=provider)


def create_rate_limiter(
    provider: str = "",
    config: RateLimitConfig | None = None,
) -> RateLimiter:
    """Factory for creating a RateLimiter.

    If config is provided, uses it directly. Otherwise looks up provider defaults.
    """
    if config is not None:
        return RateLimiter(config)
    if provider:
        return RateLimiter(get_provider_limits(provider))
    return RateLimiter(RateLimitConfig())
