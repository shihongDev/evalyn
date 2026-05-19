"""Circuit breaker for LLM providers.

Stop calling a provider after N consecutive failures. Implements the
standard closed -> open -> half_open -> closed state machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5
    reset_timeout_seconds: float = 60.0
    half_open_max_calls: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "reset_timeout_seconds": self.reset_timeout_seconds,
            "half_open_max_calls": self.half_open_max_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CircuitBreakerConfig:
        return cls(
            failure_threshold=data.get("failure_threshold", 5),
            reset_timeout_seconds=data.get("reset_timeout_seconds", 60.0),
            half_open_max_calls=data.get("half_open_max_calls", 1),
        )


@dataclass
class CircuitBreakerStatus:
    """Snapshot of a circuit breaker's current state."""

    state: str = "closed"
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float | None = None
    opened_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_failure_time": self.last_failure_time,
            "opened_at": self.opened_at,
        }

    def format_text(self) -> str:
        lines = [
            f"State: {self.state}",
            f"Consecutive failures: {self.consecutive_failures}",
            f"Total failures: {self.total_failures}",
            f"Total successes: {self.total_successes}",
        ]
        if self.last_failure_time is not None:
            lines.append(f"Last failure: {self.last_failure_time:.1f}")
        if self.opened_at is not None:
            lines.append(f"Opened at: {self.opened_at:.1f}")
        return "\n".join(lines)


class CircuitBreaker:
    """Circuit breaker for a single provider.

    Usage:
        breaker = CircuitBreaker("openai", CircuitBreakerConfig(failure_threshold=3))

        if breaker.can_execute():
            try:
                result = call_provider()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
        else:
            # skip this provider
            pass
    """

    def __init__(self, provider: str, config: CircuitBreakerConfig) -> None:
        self.provider = provider
        self.config = config
        self._state: str = "closed"
        self._consecutive_failures: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None
        self._half_open_calls: int = 0

    def record_success(self) -> None:
        """Record a successful call. Resets consecutive failures."""
        self._total_successes += 1
        self._consecutive_failures = 0
        if self._state == "half_open":
            self._state = "closed"
            self._opened_at = None
            self._half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed call. Opens circuit if threshold reached."""
        self._consecutive_failures += 1
        self._total_failures += 1
        self._last_failure_time = time.monotonic()

        if self._state == "half_open":
            # Failure during half-open: reopen immediately
            self._state = "open"
            self._opened_at = time.monotonic()
            self._half_open_calls = 0
        elif self._consecutive_failures >= self.config.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    def can_execute(self) -> bool:
        """Check whether a call to this provider is allowed.

        Returns True if the circuit is closed, or if the circuit is open
        and the reset timeout has expired (transitioning to half_open).
        """
        if self._state == "closed":
            return True

        if self._state == "half_open":
            return self._half_open_calls < self.config.half_open_max_calls

        # state == "open"
        if self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.config.reset_timeout_seconds:
                self._state = "half_open"
                self._half_open_calls = 0
                return True

        return False

    def get_status(self) -> CircuitBreakerStatus:
        """Return a snapshot of the current breaker state."""
        return CircuitBreakerStatus(
            state=self._state,
            consecutive_failures=self._consecutive_failures,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            last_failure_time=self._last_failure_time,
            opened_at=self._opened_at,
        )

    def reset(self) -> None:
        """Force-reset the circuit breaker to closed state."""
        self._state = "closed"
        self._consecutive_failures = 0
        self._last_failure_time = None
        self._opened_at = None
        self._half_open_calls = 0

    def is_open(self) -> bool:
        return self._state == "open"

    def is_closed(self) -> bool:
        return self._state == "closed"

    def is_half_open(self) -> bool:
        return self._state == "half_open"


def create_circuit_breaker(
    provider: str, failure_threshold: int = 5
) -> CircuitBreaker:
    """Factory: create a circuit breaker with a given failure threshold."""
    config = CircuitBreakerConfig(failure_threshold=failure_threshold)
    return CircuitBreaker(provider, config)


def should_skip_provider(breaker: CircuitBreaker) -> tuple[bool, str]:
    """Check whether a provider should be skipped.

    Returns:
        (should_skip, reason) tuple. reason is empty when should_skip is False.
    """
    if breaker.can_execute():
        return (False, "")
    status = breaker.get_status()
    reason = (
        f"Circuit open for provider '{breaker.provider}': "
        f"{status.consecutive_failures} consecutive failures"
    )
    return (True, reason)
