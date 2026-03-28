"""Provider fallback chain.

Automatically try alternative providers on failure. Providers are tried
in priority order; the first successful call wins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class ProviderConfig:
    """Configuration for a single provider in the fallback chain."""

    name: str
    model: str = ""
    priority: int = 0
    max_retries: int = 1
    enabled: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProviderConfig:
        return cls(
            name=data["name"],
            model=data.get("model", ""),
            priority=data.get("priority", 0),
            max_retries=data.get("max_retries", 1),
            enabled=data.get("enabled", True),
        )


@dataclass
class FallbackAttempt:
    """Record of a single provider attempt."""

    provider: str
    success: bool
    error: str = ""
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class FallbackResult:
    """Result of executing through the fallback chain."""

    final_provider: str = ""
    attempts: List[FallbackAttempt] = field(default_factory=list)
    success: bool = False
    total_attempts: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "final_provider": self.final_provider,
            "attempts": [a.as_dict() for a in self.attempts],
            "success": self.success,
            "total_attempts": self.total_attempts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FallbackResult:
        attempts = [
            FallbackAttempt(**a) for a in data.get("attempts", [])
        ]
        return cls(
            final_provider=data.get("final_provider", ""),
            attempts=attempts,
            success=data.get("success", False),
            total_attempts=data.get("total_attempts", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Success: {self.success}",
            f"Final provider: {self.final_provider}",
            f"Total attempts: {self.total_attempts}",
        ]
        if self.attempts:
            lines.append("Attempts:")
            for a in self.attempts:
                status = "ok" if a.success else "FAIL"
                parts = [f"  - {a.provider} [{status}]"]
                if a.duration_ms > 0:
                    parts.append(f"{a.duration_ms:.1f}ms")
                if a.error:
                    parts.append(f": {a.error}")
                lines.append(" ".join(parts))
        return "\n".join(lines)


@dataclass
class FallbackChainConfig:
    """Configuration for the full fallback chain."""

    providers: List[ProviderConfig] = field(default_factory=list)
    stop_on_first_success: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "providers": [p.as_dict() for p in self.providers],
            "stop_on_first_success": self.stop_on_first_success,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FallbackChainConfig:
        providers = [
            ProviderConfig.from_dict(p) for p in data.get("providers", [])
        ]
        return cls(
            providers=providers,
            stop_on_first_success=data.get("stop_on_first_success", True),
        )


def create_fallback_chain(providers: List[Dict[str, Any]]) -> FallbackChainConfig:
    """Factory: build a FallbackChainConfig from a list of dicts."""
    configs = [ProviderConfig.from_dict(p) for p in providers]
    return FallbackChainConfig(providers=configs)


def get_ordered_providers(config: FallbackChainConfig) -> List[ProviderConfig]:
    """Return enabled providers sorted by priority (lowest first)."""
    enabled = [p for p in config.providers if p.enabled]
    return sorted(enabled, key=lambda p: p.priority)


def execute_with_fallback(
    config: FallbackChainConfig,
    execute_fn: Callable[[str, str], Any],
) -> FallbackResult:
    """Try providers in priority order until one succeeds.

    Args:
        config: The fallback chain configuration.
        execute_fn: Callable(provider_name, model) -> result. Raises on failure.

    Returns:
        FallbackResult with attempts and final outcome.
    """
    ordered = get_ordered_providers(config)
    attempts: List[FallbackAttempt] = []

    for provider in ordered:
        for retry in range(provider.max_retries):
            start = time.monotonic()
            try:
                execute_fn(provider.name, provider.model)
                elapsed_ms = (time.monotonic() - start) * 1000
                attempts.append(FallbackAttempt(
                    provider=provider.name,
                    success=True,
                    duration_ms=elapsed_ms,
                ))
                if config.stop_on_first_success:
                    return FallbackResult(
                        final_provider=provider.name,
                        attempts=attempts,
                        success=True,
                        total_attempts=len(attempts),
                    )
            except Exception as exc:
                elapsed_ms = (time.monotonic() - start) * 1000
                attempts.append(FallbackAttempt(
                    provider=provider.name,
                    success=False,
                    error=str(exc),
                    duration_ms=elapsed_ms,
                ))

    # All providers exhausted
    final = attempts[-1].provider if attempts else ""
    any_success = any(a.success for a in attempts)
    if any_success:
        # Find the last successful attempt
        for a in reversed(attempts):
            if a.success:
                final = a.provider
                break
    return FallbackResult(
        final_provider=final,
        attempts=attempts,
        success=any_success,
        total_attempts=len(attempts),
    )


def build_fallback_report(results: List[FallbackResult]) -> Dict[str, Any]:
    """Aggregate multiple fallback results into a summary report.

    Returns dict with provider success rates and average fallback depth.
    """
    provider_attempts: Dict[str, int] = {}
    provider_successes: Dict[str, int] = {}
    total_depth = 0

    for result in results:
        total_depth += result.total_attempts
        for attempt in result.attempts:
            provider_attempts[attempt.provider] = (
                provider_attempts.get(attempt.provider, 0) + 1
            )
            if attempt.success:
                provider_successes[attempt.provider] = (
                    provider_successes.get(attempt.provider, 0) + 1
                )

    provider_success_rates: Dict[str, float] = {}
    for name, total in provider_attempts.items():
        successes = provider_successes.get(name, 0)
        provider_success_rates[name] = successes / total if total > 0 else 0.0

    avg_depth = total_depth / len(results) if results else 0.0

    return {
        "total_executions": len(results),
        "provider_attempts": provider_attempts,
        "provider_success_rates": provider_success_rates,
        "avg_fallback_depth": avg_depth,
    }


def suggest_chain_optimization(results: List[FallbackResult]) -> List[str]:
    """Suggest optimizations based on fallback results.

    Looks at first-attempt success rates and suggests reordering.
    """
    if not results:
        return []

    suggestions: List[str] = []

    # Count first-attempt successes per provider
    first_attempt_success: Dict[str, int] = {}
    first_attempt_total: Dict[str, int] = {}
    for result in results:
        if result.attempts:
            first = result.attempts[0]
            first_attempt_total[first.provider] = (
                first_attempt_total.get(first.provider, 0) + 1
            )
            if first.success:
                first_attempt_success[first.provider] = (
                    first_attempt_success.get(first.provider, 0) + 1
                )

    # Count overall success rates per provider
    provider_successes: Dict[str, int] = {}
    provider_totals: Dict[str, int] = {}
    for result in results:
        for attempt in result.attempts:
            provider_totals[attempt.provider] = (
                provider_totals.get(attempt.provider, 0) + 1
            )
            if attempt.success:
                provider_successes[attempt.provider] = (
                    provider_successes.get(attempt.provider, 0) + 1
                )

    # Suggest moving high-success providers to the front
    for name, total in provider_totals.items():
        successes = provider_successes.get(name, 0)
        rate = successes / total if total > 0 else 0.0
        if rate >= 0.9:
            first_total = first_attempt_total.get(name, 0)
            first_success = first_attempt_success.get(name, 0)
            first_rate = first_success / first_total if first_total > 0 else 0.0
            if first_rate < 0.5 or first_total == 0:
                suggestions.append(
                    f"Move provider {name} to front - {rate:.0%} success rate"
                )

    # Flag providers that always fail
    for name, total in provider_totals.items():
        successes = provider_successes.get(name, 0)
        if successes == 0 and total >= 2:
            suggestions.append(
                f"Consider disabling provider {name} - 0% success across {total} attempts"
            )

    # Suggest reducing retries if first attempt usually works
    total_first_success = sum(first_attempt_success.values())
    if results and total_first_success / len(results) >= 0.9:
        suggestions.append(
            "First-attempt success rate is high - consider reducing max_retries to 1"
        )

    return suggestions
