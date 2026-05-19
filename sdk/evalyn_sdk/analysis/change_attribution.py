"""
Analysis change attribution: attribute metric changes to dataset, model, prompt, or config factors.

Given two evaluation runs and the context of what changed between them,
determines which factor most likely caused each metric's delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ChangeFactor:
    """A single factor that may have caused a metric change."""

    factor: str  # "dataset" / "model" / "prompt" / "config" / "unknown"
    confidence: float = 0.0  # 0-1
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class ChangeAttribution:
    """Attribution of a metric change to one or more factors."""

    metric_id: str
    delta: float
    primary_factor: str = ""
    factors: list[ChangeFactor] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "delta": self.delta,
            "primary_factor": self.primary_factor,
            "factors": [f.as_dict() for f in self.factors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeAttribution:
        factors = [
            ChangeFactor(
                factor=f["factor"],
                confidence=f.get("confidence", 0.0),
                evidence=f.get("evidence", ""),
            )
            for f in data.get("factors", [])
        ]
        return cls(
            metric_id=data["metric_id"],
            delta=data["delta"],
            primary_factor=data.get("primary_factor", ""),
            factors=factors,
        )


@dataclass
class AttributionReport:
    """Aggregated attribution report across all metrics."""

    attributions: list[ChangeAttribution] = field(default_factory=list)
    dominant_factor: str = ""
    factor_distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attributions": [a.as_dict() for a in self.attributions],
            "dominant_factor": self.dominant_factor,
            "factor_distribution": dict(self.factor_distribution),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionReport:
        attributions = [
            ChangeAttribution.from_dict(a) for a in data.get("attributions", [])
        ]
        return cls(
            attributions=attributions,
            dominant_factor=data.get("dominant_factor", ""),
            factor_distribution=dict(data.get("factor_distribution", {})),
        )

    def format_text(self) -> str:
        """Format the report as human-readable text."""
        lines: list[str] = []
        lines.append("CHANGE ATTRIBUTION REPORT")
        lines.append("-" * 40)
        if not self.attributions:
            lines.append("No attributions to report.")
            return "\n".join(lines)

        lines.append(f"Dominant factor: {self.dominant_factor}")
        lines.append("")
        lines.append("Factor distribution:")
        for factor, count in sorted(
            self.factor_distribution.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {factor}: {count}")
        lines.append("")
        lines.append(f"{'Metric':<30} {'Delta':>8} {'Primary Factor':<15}")
        lines.append("-" * 55)
        for a in self.attributions:
            lines.append(f"{a.metric_id:<30} {a.delta:>+8.3f} {a.primary_factor:<15}")
            for f in a.factors:
                lines.append(
                    f"  - {f.factor} (confidence: {f.confidence:.2f}) {f.evidence}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection Functions
# ---------------------------------------------------------------------------


def detect_dataset_change(
    items_a: int, items_b: int, hash_a: str = "", hash_b: str = ""
) -> ChangeFactor:
    """Detect if dataset changed between runs based on item count and hash."""
    if hash_a and hash_b and hash_a != hash_b:
        return ChangeFactor(
            factor="dataset",
            confidence=0.9,
            evidence=f"Dataset hash changed: {hash_a[:8]}.. -> {hash_b[:8]}..",
        )
    if items_a != items_b:
        return ChangeFactor(
            factor="dataset",
            confidence=0.8,
            evidence=f"Item count changed: {items_a} -> {items_b}",
        )
    return ChangeFactor(factor="dataset", confidence=0.0, evidence="No dataset change detected")


def detect_model_change(model_a: str, model_b: str) -> ChangeFactor:
    """Detect if the model changed between runs."""
    if model_a != model_b:
        return ChangeFactor(
            factor="model",
            confidence=0.95,
            evidence=f"Model changed: {model_a} -> {model_b}",
        )
    return ChangeFactor(factor="model", confidence=0.0, evidence="No model change detected")


def _edit_distance_ratio(a: str, b: str) -> float:
    """Compute normalized edit distance ratio (0 = identical, 1 = completely different).

    Uses a simple character-level comparison for efficiency.
    """
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    max_len = max(len(a), len(b))
    # Simple diff: count mismatched characters at each position
    common = min(len(a), len(b))
    mismatches = sum(1 for i in range(common) if a[i] != b[i])
    mismatches += abs(len(a) - len(b))
    return mismatches / max_len


def detect_prompt_change(prompt_a: str, prompt_b: str) -> ChangeFactor:
    """Detect if the prompt changed. Confidence based on edit distance ratio."""
    if prompt_a == prompt_b:
        return ChangeFactor(
            factor="prompt", confidence=0.0, evidence="No prompt change detected"
        )
    ratio = _edit_distance_ratio(prompt_a, prompt_b)
    confidence = min(0.5 + ratio * 0.5, 1.0)
    return ChangeFactor(
        factor="prompt",
        confidence=confidence,
        evidence=f"Prompt changed (edit distance ratio: {ratio:.2f})",
    )


def detect_config_change(
    config_a: dict[str, Any], config_b: dict[str, Any]
) -> ChangeFactor:
    """Detect if any config values differ between runs."""
    all_keys = set(config_a.keys()) | set(config_b.keys())
    changed_keys = []
    for key in all_keys:
        if config_a.get(key) != config_b.get(key):
            changed_keys.append(key)
    if not changed_keys:
        return ChangeFactor(
            factor="config", confidence=0.0, evidence="No config change detected"
        )
    confidence = min(0.6 + len(changed_keys) * 0.1, 1.0)
    return ChangeFactor(
        factor="config",
        confidence=confidence,
        evidence=f"Config keys changed: {', '.join(sorted(changed_keys))}",
    )


# ---------------------------------------------------------------------------
# Attribution Logic
# ---------------------------------------------------------------------------


def attribute_change(
    metric_id: str, delta: float, context: dict[str, Any]
) -> ChangeAttribution:
    """Determine which factor most likely caused a metric change.

    Context keys:
        items_a, items_b: int - item counts
        hash_a, hash_b: str - dataset hashes (optional)
        model_a, model_b: str - model names
        prompt_a, prompt_b: str - prompt texts
        config_a, config_b: Dict - config dicts
    """
    factors: list[ChangeFactor] = []

    dataset_factor = detect_dataset_change(
        context.get("items_a", 0),
        context.get("items_b", 0),
        context.get("hash_a", ""),
        context.get("hash_b", ""),
    )
    if dataset_factor.confidence > 0:
        factors.append(dataset_factor)

    model_factor = detect_model_change(
        context.get("model_a", ""),
        context.get("model_b", ""),
    )
    if model_factor.confidence > 0:
        factors.append(model_factor)

    prompt_factor = detect_prompt_change(
        context.get("prompt_a", ""),
        context.get("prompt_b", ""),
    )
    if prompt_factor.confidence > 0:
        factors.append(prompt_factor)

    config_factor = detect_config_change(
        context.get("config_a", {}),
        context.get("config_b", {}),
    )
    if config_factor.confidence > 0:
        factors.append(config_factor)

    # Determine primary factor: highest confidence, or "unknown" if none
    if factors:
        primary = max(factors, key=lambda f: f.confidence)
        primary_factor = primary.factor
    else:
        primary_factor = "unknown"
        factors.append(
            ChangeFactor(factor="unknown", confidence=1.0, evidence="No changes detected in any tracked factor")
        )

    return ChangeAttribution(
        metric_id=metric_id,
        delta=delta,
        primary_factor=primary_factor,
        factors=factors,
    )


def build_attribution_report(
    attributions: list[ChangeAttribution],
) -> AttributionReport:
    """Aggregate attributions into a report with factor distribution."""
    if not attributions:
        return AttributionReport()

    distribution: dict[str, int] = {}
    for a in attributions:
        distribution[a.primary_factor] = distribution.get(a.primary_factor, 0) + 1

    dominant = max(distribution, key=lambda k: distribution[k]) if distribution else ""

    return AttributionReport(
        attributions=attributions,
        dominant_factor=dominant,
        factor_distribution=distribution,
    )
