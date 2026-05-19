"""Evaluation profiles: named configurations for common evaluation patterns.

Profiles bundle evaluation settings (workers, provider, metric types,
confidence) into reusable named configs. Three built-in profiles plus
support for user-defined profiles in evalyn.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalProfile:
    """A named evaluation configuration."""

    name: str
    description: str = ""
    max_workers: int = 1
    provider: str = "gemini"
    metric_types: list[str] = field(default_factory=lambda: ["objective", "subjective"])
    confidence_method: str = "none"
    confidence_samples: int = 3
    checkpoint_enabled: bool = True
    batch_mode: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_workers": self.max_workers,
            "provider": self.provider,
            "metric_types": list(self.metric_types),
            "confidence_method": self.confidence_method,
            "confidence_samples": self.confidence_samples,
            "checkpoint_enabled": self.checkpoint_enabled,
            "batch_mode": self.batch_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalProfile:
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            max_workers=data.get("max_workers", 1),
            provider=data.get("provider", "gemini"),
            metric_types=data.get("metric_types", ["objective", "subjective"]),
            confidence_method=data.get("confidence_method", "none"),
            confidence_samples=data.get("confidence_samples", 3),
            checkpoint_enabled=data.get("checkpoint_enabled", True),
            batch_mode=data.get("batch_mode", False),
        )

    def filter_metrics(self, metrics: list) -> list:
        """Filter metrics to those allowed by this profile's metric_types."""
        return [m for m in metrics if m.spec.type in self.metric_types]


# Built-in profiles
SMOKE_TEST = EvalProfile(
    name="smoke-test",
    description="Quick check with objective metrics only, no LLM calls",
    max_workers=4,
    metric_types=["objective"],
    confidence_method="none",
    checkpoint_enabled=False,
)

STANDARD = EvalProfile(
    name="standard",
    description="Balanced evaluation with all metrics and moderate parallelism",
    max_workers=2,
    metric_types=["objective", "subjective"],
    confidence_method="none",
    checkpoint_enabled=True,
)

THOROUGH = EvalProfile(
    name="thorough",
    description="Comprehensive evaluation with confidence estimation",
    max_workers=1,
    metric_types=["objective", "subjective"],
    confidence_method="consistency",
    confidence_samples=5,
    checkpoint_enabled=True,
)

COST_OPTIMIZED = EvalProfile(
    name="cost-optimized",
    description="Minimize LLM costs using batch mode and local provider",
    max_workers=1,
    provider="ollama",
    metric_types=["objective", "subjective"],
    confidence_method="none",
    batch_mode=True,
    checkpoint_enabled=True,
)

# Registry of built-in profiles
BUILTIN_PROFILES: dict[str, EvalProfile] = {
    "smoke-test": SMOKE_TEST,
    "standard": STANDARD,
    "thorough": THOROUGH,
    "cost-optimized": COST_OPTIMIZED,
}


def get_profile(name: str, custom_profiles: dict[str, dict] | None = None) -> EvalProfile:
    """Get an evaluation profile by name.

    Checks custom profiles first, then built-in profiles.

    Args:
        name: Profile name.
        custom_profiles: Optional dict of custom profile definitions from config.

    Returns:
        EvalProfile instance.

    Raises:
        ValueError: If profile name not found.
    """
    # Check custom profiles first
    if custom_profiles and name in custom_profiles:
        data = custom_profiles[name]
        data["name"] = name
        return EvalProfile.from_dict(data)

    # Check built-in profiles
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]

    available = list(BUILTIN_PROFILES.keys())
    if custom_profiles:
        available.extend(custom_profiles.keys())
    raise ValueError(f"Unknown profile '{name}'. Available: {sorted(available)}")


def list_profiles(custom_profiles: dict[str, dict] | None = None) -> list[EvalProfile]:
    """List all available profiles (built-in + custom).

    Args:
        custom_profiles: Optional dict of custom profile definitions.

    Returns:
        List of all available EvalProfile instances.
    """
    profiles = list(BUILTIN_PROFILES.values())
    if custom_profiles:
        for name, data in custom_profiles.items():
            if name not in BUILTIN_PROFILES:
                data["name"] = name
                profiles.append(EvalProfile.from_dict(data))
    return profiles
