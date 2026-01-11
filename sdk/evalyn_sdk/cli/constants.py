"""Constants for CLI commands."""

from __future__ import annotations

from typing import Dict, List

# Config file paths to search
DEFAULT_CONFIG_PATHS = [".evalynrc", "evalyn.yaml", "evalyn.yml", ".evalyn.yaml"]

# Light bundles for quick manual selection (no LLM).
BUNDLES: Dict[str, List[str]] = {
    "summarization": [
        "latency_ms",
        "token_overlap_f1",
        "rouge_l",
        "rouge_1",
        "hallucination_risk",
        "clarity_readability",
        "conciseness",
        "toxicity_safety",
    ],
    "orchestrator": [
        "latency_ms",
        "tool_call_count",
        "llm_call_count",
        "llm_error_rate",
        "tool_success_ratio",
        "hallucination_risk",
        "instruction_following",
        "policy_guardrails",
    ],
    "research-agent": [
        "latency_ms",
        "url_count",
        "hallucination_risk",
        "factual_consistency",
        "helpfulness_accuracy",
        "tool_success_ratio",
        "clarity_readability",
        "tone_alignment",
    ],
}

__all__ = ["DEFAULT_CONFIG_PATHS", "BUNDLES"]
