"""
Offline evaluation support for running evaluations without internet.

Provides capability checking, metric filtering, and offline alternative
suggestions. Pure Python with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class OfflineCapability:
    """Describes whether a metric can run offline."""

    metric_id: str
    offline_supported: bool
    requires: List[str]  # "llm_api", "embeddings_api", "internet"
    fallback: str  # offline alternative, empty if none

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "offline_supported": self.offline_supported,
            "requires": list(self.requires),
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OfflineCapability:
        return cls(
            metric_id=data["metric_id"],
            offline_supported=data["offline_supported"],
            requires=data.get("requires", []),
            fallback=data.get("fallback", ""),
        )


@dataclass
class OfflineConfig:
    """Configuration for offline evaluation runs."""

    strict: bool = True  # error if any metric needs internet
    allow_local_models: bool = True
    local_model_provider: str = "ollama"
    cached_artifacts_dir: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strict": self.strict,
            "allow_local_models": self.allow_local_models,
            "local_model_provider": self.local_model_provider,
            "cached_artifacts_dir": self.cached_artifacts_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OfflineConfig:
        return cls(
            strict=data.get("strict", True),
            allow_local_models=data.get("allow_local_models", True),
            local_model_provider=data.get("local_model_provider", "ollama"),
            cached_artifacts_dir=data.get("cached_artifacts_dir", ""),
        )


@dataclass
class OfflineCheckResult:
    """Result of an offline compatibility check."""

    all_offline: bool
    online_metrics: List[str]
    offline_metrics: List[str]
    warnings: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "all_offline": self.all_offline,
            "online_metrics": list(self.online_metrics),
            "offline_metrics": list(self.offline_metrics),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OfflineCheckResult:
        return cls(
            all_offline=data["all_offline"],
            online_metrics=data.get("online_metrics", []),
            offline_metrics=data.get("offline_metrics", []),
            warnings=data.get("warnings", []),
        )


# ---------------------------------------------------------------------------
# Metric Capabilities Registry
# ---------------------------------------------------------------------------


METRIC_CAPABILITIES: Dict[str, OfflineCapability] = {
    # Objective metrics - fully offline
    "exact_match": OfflineCapability(
        metric_id="exact_match",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "regex_match": OfflineCapability(
        metric_id="regex_match",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "json_valid": OfflineCapability(
        metric_id="json_valid",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "latency": OfflineCapability(
        metric_id="latency",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "token_length": OfflineCapability(
        metric_id="token_length",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "tool_call_count": OfflineCapability(
        metric_id="tool_call_count",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "bleu": OfflineCapability(
        metric_id="bleu",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "rouge": OfflineCapability(
        metric_id="rouge",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "pass_at_k": OfflineCapability(
        metric_id="pass_at_k",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "cost": OfflineCapability(
        metric_id="cost",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "string_distance": OfflineCapability(
        metric_id="string_distance",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    "output_length": OfflineCapability(
        metric_id="output_length",
        offline_supported=True,
        requires=[],
        fallback="",
    ),
    # Subjective / LLM-based metrics - need API
    "helpfulness": OfflineCapability(
        metric_id="helpfulness",
        offline_supported=False,
        requires=["llm_api"],
        fallback="token_length",
    ),
    "coherence": OfflineCapability(
        metric_id="coherence",
        offline_supported=False,
        requires=["llm_api"],
        fallback="bleu",
    ),
    "safety": OfflineCapability(
        metric_id="safety",
        offline_supported=False,
        requires=["llm_api"],
        fallback="regex_match",
    ),
    "hallucination": OfflineCapability(
        metric_id="hallucination",
        offline_supported=False,
        requires=["llm_api"],
        fallback="exact_match",
    ),
    "relevance": OfflineCapability(
        metric_id="relevance",
        offline_supported=False,
        requires=["llm_api"],
        fallback="bleu",
    ),
    # Embedding-based metrics - need embeddings API
    "semantic_similarity": OfflineCapability(
        metric_id="semantic_similarity",
        offline_supported=False,
        requires=["embeddings_api"],
        fallback="bleu",
    ),
    "embedding_distance": OfflineCapability(
        metric_id="embedding_distance",
        offline_supported=False,
        requires=["embeddings_api"],
        fallback="string_distance",
    ),
    # Internet-dependent metrics
    "fact_check": OfflineCapability(
        metric_id="fact_check",
        offline_supported=False,
        requires=["internet", "llm_api"],
        fallback="exact_match",
    ),
}


# Offline alternatives mapping: groups metrics by what they approximate
_OFFLINE_ALTERNATIVES: Dict[str, List[str]] = {
    "helpfulness": ["token_length", "output_length", "bleu"],
    "coherence": ["bleu", "rouge", "string_distance"],
    "safety": ["regex_match", "exact_match"],
    "hallucination": ["exact_match", "bleu", "rouge"],
    "relevance": ["bleu", "rouge", "exact_match"],
    "semantic_similarity": ["bleu", "rouge", "string_distance"],
    "embedding_distance": ["string_distance", "bleu"],
    "fact_check": ["exact_match", "regex_match"],
}


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def check_offline_compatibility(
    metric_ids: List[str],
    config: OfflineConfig,
) -> OfflineCheckResult:
    """Check which metrics can run offline given a configuration.

    Args:
        metric_ids: List of metric IDs to check.
        config: Offline configuration settings.

    Returns:
        OfflineCheckResult with categorized metrics and warnings.
    """
    online: List[str] = []
    offline: List[str] = []
    warnings: List[str] = []

    for mid in metric_ids:
        cap = METRIC_CAPABILITIES.get(mid)
        if cap is None:
            # Unknown metric - assume offline in non-strict, warn always
            warnings.append(f"Unknown metric '{mid}' - offline capability unknown.")
            if config.strict:
                online.append(mid)
            else:
                offline.append(mid)
            continue

        if cap.offline_supported:
            offline.append(mid)
            continue

        # Check if local models can substitute
        needs_llm = "llm_api" in cap.requires
        needs_embeddings = "embeddings_api" in cap.requires
        needs_internet = "internet" in cap.requires

        if needs_internet:
            online.append(mid)
            warnings.append(
                f"Metric '{mid}' requires internet access and cannot run offline."
            )
        elif (needs_llm or needs_embeddings) and config.allow_local_models:
            offline.append(mid)
            warnings.append(
                f"Metric '{mid}' will use local model provider "
                f"'{config.local_model_provider}'."
            )
        else:
            online.append(mid)
            if cap.fallback:
                warnings.append(
                    f"Metric '{mid}' requires {', '.join(cap.requires)}. "
                    f"Consider fallback: '{cap.fallback}'."
                )
            else:
                warnings.append(
                    f"Metric '{mid}' requires {', '.join(cap.requires)} "
                    f"and has no offline fallback."
                )

    all_offline = len(online) == 0
    return OfflineCheckResult(
        all_offline=all_offline,
        online_metrics=online,
        offline_metrics=offline,
        warnings=warnings,
    )


def filter_offline_metrics(metric_ids: List[str]) -> List[str]:
    """Return only metrics that work fully offline (no API needed).

    Args:
        metric_ids: List of metric IDs to filter.

    Returns:
        List of metric IDs that are offline-capable.
    """
    result = []
    for mid in metric_ids:
        cap = METRIC_CAPABILITIES.get(mid)
        if cap is not None and cap.offline_supported:
            result.append(mid)
    return result


def suggest_offline_alternatives(metric_id: str) -> List[str]:
    """Suggest offline-capable alternatives for an online metric.

    Args:
        metric_id: The metric to find alternatives for.

    Returns:
        List of offline metric IDs that approximate the given metric.
        Empty list if the metric is already offline or has no alternatives.
    """
    # If already offline, no alternatives needed
    cap = METRIC_CAPABILITIES.get(metric_id)
    if cap is not None and cap.offline_supported:
        return []

    # Check the alternatives mapping first
    alternatives = _OFFLINE_ALTERNATIVES.get(metric_id, [])
    if alternatives:
        return list(alternatives)

    # Fall back to the capability's fallback field
    if cap is not None and cap.fallback:
        return [cap.fallback]

    return []


def validate_offline_run(
    metric_ids: List[str],
    config: OfflineConfig,
) -> Tuple[bool, List[str]]:
    """Validate whether a set of metrics can run offline.

    Args:
        metric_ids: Metrics to validate.
        config: Offline configuration.

    Returns:
        Tuple of (can_run: bool, error_messages: list of str).
        In strict mode, any online metric is an error.
        In non-strict mode, online metrics are warnings but run is allowed.
    """
    check = check_offline_compatibility(metric_ids, config)
    errors: List[str] = []

    if not metric_ids:
        errors.append("No metrics specified.")
        return False, errors

    if config.strict and check.online_metrics:
        for mid in check.online_metrics:
            cap = METRIC_CAPABILITIES.get(mid)
            if cap is not None:
                errors.append(
                    f"Metric '{mid}' cannot run offline "
                    f"(requires: {', '.join(cap.requires)})."
                )
            else:
                errors.append(
                    f"Metric '{mid}' has unknown offline capability."
                )
        return False, errors

    return True, errors


def format_offline_check(result: OfflineCheckResult) -> str:
    """Format an offline check result as a human-readable report.

    Args:
        result: The check result to format.

    Returns:
        Multi-line string report.
    """
    lines: List[str] = []

    if result.all_offline:
        lines.append("Offline Status: ALL METRICS OFFLINE-COMPATIBLE")
    else:
        lines.append("Offline Status: SOME METRICS REQUIRE ONLINE ACCESS")

    lines.append("")

    if result.offline_metrics:
        lines.append(f"Offline metrics ({len(result.offline_metrics)}):")
        for mid in result.offline_metrics:
            lines.append(f"  - {mid}")
        lines.append("")

    if result.online_metrics:
        lines.append(f"Online-only metrics ({len(result.online_metrics)}):")
        for mid in result.online_metrics:
            cap = METRIC_CAPABILITIES.get(mid)
            if cap and cap.fallback:
                lines.append(f"  - {mid} (fallback: {cap.fallback})")
            else:
                lines.append(f"  - {mid}")
        lines.append("")

    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def estimate_offline_coverage(metric_ids: List[str]) -> float:
    """Estimate what fraction of metrics work fully offline.

    Args:
        metric_ids: List of metric IDs to check.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 for empty input.
    """
    if not metric_ids:
        return 0.0

    offline_count = 0
    for mid in metric_ids:
        cap = METRIC_CAPABILITIES.get(mid)
        if cap is not None and cap.offline_supported:
            offline_count += 1

    return offline_count / len(metric_ids)
