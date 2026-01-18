"""Constants for CLI commands."""

from __future__ import annotations

from typing import Dict, List

# Config file paths to search
DEFAULT_CONFIG_PATHS = [".evalynrc", "evalyn.yaml", "evalyn.yml", ".evalyn.yaml"]

# =============================================================================
# Metric Bundles for Common GenAI Use Cases
# =============================================================================
#
# Each bundle is a curated set of metrics optimized for a specific use case.
# Bundles include both objective (automated) and subjective (LLM judge) metrics.
#
# Usage: evalyn suggest-metrics --mode bundle --bundle <name>
#        evalyn one-click --metric-mode bundle --bundle <name>
#
# Bundle Design Principles:
# 1. Start with safety metrics for all user-facing applications
# 2. Include efficiency metrics (latency) for production monitoring
# 3. Add domain-specific quality metrics based on use case
# 4. Keep bundles focused (6-10 metrics) to avoid evaluation fatigue
#
# =============================================================================

BUNDLES: Dict[str, List[str]] = {
    # -------------------------------------------------------------------------
    # CONVERSATIONAL AI
    # -------------------------------------------------------------------------
    "chatbot": [
        # Safety first
        "toxicity_safety",
        "pii_safety",
        # Core quality
        "helpfulness_accuracy",
        "tone_alignment",
        "coherence_clarity",
        # Efficiency
        "latency_ms",
        "output_nonempty",
    ],
    "customer-support": [
        # Safety
        "toxicity_safety",
        "pii_safety",
        # Quality
        "helpfulness_accuracy",
        "tone_alignment",
        "completeness",
        "relevance",
        # Efficiency
        "latency_ms",
    ],
    # -------------------------------------------------------------------------
    # CONTENT GENERATION
    # -------------------------------------------------------------------------
    "content-writer": [
        # Quality
        "coherence_clarity",
        "conciseness",
        "tone_alignment",
        "instruction_following",
        "completeness",
        # Safety
        "toxicity_safety",
        # Structure
        "output_nonempty",
        "output_length_range",
    ],
    "summarization": [
        # Reference-based (needs human_label.reference)
        "rouge_l",
        "rouge_1",
        "token_overlap_f1",
        # Quality
        "coherence_clarity",
        "conciseness",
        "completeness",
        # Safety
        "hallucination_risk",
        # Efficiency
        "latency_ms",
    ],
    "creative-writer": [
        # Quality
        "coherence_clarity",
        "completeness",
        "tone_alignment",
        "instruction_following",
        # Safety
        "toxicity_safety",
        "cultural_sensitivity",
        # Structure
        "output_nonempty",
    ],
    # -------------------------------------------------------------------------
    # KNOWLEDGE & RESEARCH
    # -------------------------------------------------------------------------
    "rag-qa": [
        # Grounding (critical for RAG)
        "hallucination_risk",
        "source_attribution",
        "url_count",
        # Quality
        "factual_accuracy",
        "relevance",
        "completeness",
        # Efficiency
        "latency_ms",
    ],
    "research-agent": [
        # Grounding
        "hallucination_risk",
        "source_attribution",
        "url_count",
        # Quality
        "factual_accuracy",
        "helpfulness_accuracy",
        "coherence_clarity",
        # Agent
        "tool_success_ratio",
        # Efficiency
        "latency_ms",
    ],
    "tutor": [
        # Quality
        "helpfulness_accuracy",
        "factual_accuracy",
        "coherence_clarity",
        "reasoning_quality",
        "completeness",
        # Safety
        "toxicity_safety",
        # Efficiency
        "latency_ms",
    ],
    # -------------------------------------------------------------------------
    # CODE & TECHNICAL
    # -------------------------------------------------------------------------
    "code-assistant": [
        # Correctness
        "technical_accuracy",
        "instruction_following",
        # Quality
        "reasoning_quality",
        "helpfulness_accuracy",
        "completeness",
        # Structure (for JSON/structured output)
        "json_valid",
        # Efficiency
        "latency_ms",
    ],
    "data-extraction": [
        # Structure (critical)
        "json_valid",
        "json_schema_keys",
        "json_path_present",
        # Quality
        "instruction_following",
        "completeness",
        # Basic
        "output_nonempty",
        # Efficiency
        "latency_ms",
    ],
    # -------------------------------------------------------------------------
    # AGENTS & ORCHESTRATION
    # -------------------------------------------------------------------------
    "orchestrator": [
        # Agent behavior
        "tool_success_ratio",
        "tool_call_count",
        "tool_error_count",
        "planning_quality",
        "error_recovery",
        # Quality
        "instruction_following",
        "hallucination_risk",
        # Efficiency
        "latency_ms",
        "llm_call_count",
    ],
    "multi-step-agent": [
        # Agent
        "planning_quality",
        "reasoning_quality",
        "tool_use_appropriateness",
        "error_recovery",
        "context_utilization",
        # Quality
        "completeness",
        "hallucination_risk",
        # Efficiency
        "latency_ms",
    ],
    # -------------------------------------------------------------------------
    # HIGH-STAKES DOMAINS
    # -------------------------------------------------------------------------
    "medical-advisor": [
        # Safety (critical)
        "toxicity_safety",
        "pii_safety",
        "ethical_reasoning",
        # Accuracy (critical)
        "factual_accuracy",
        "hallucination_risk",
        "source_attribution",
        # Quality
        "completeness",
        "coherence_clarity",
    ],
    "legal-assistant": [
        # Accuracy (critical)
        "factual_accuracy",
        "technical_accuracy",
        "hallucination_risk",
        "source_attribution",
        # Quality
        "completeness",
        "relevance",
        # Safety
        "pii_safety",
        # Efficiency
        "latency_ms",
    ],
    "financial-advisor": [
        # Safety (critical)
        "pii_safety",
        "ethical_reasoning",
        # Accuracy (critical)
        "factual_accuracy",
        "hallucination_risk",
        # Quality
        "completeness",
        "coherence_clarity",
        "relevance",
        # Efficiency
        "latency_ms",
    ],
    # -------------------------------------------------------------------------
    # SAFETY & MODERATION
    # -------------------------------------------------------------------------
    "moderator": [
        # Safety (all)
        "toxicity_safety",
        "bias_detection",
        "pii_safety",
        "manipulation_resistance",
        "cultural_sensitivity",
        # Quality
        "relevance",
        # Efficiency
        "latency_ms",
    ],
    # -------------------------------------------------------------------------
    # TRANSLATION
    # -------------------------------------------------------------------------
    "translator": [
        # Reference-based (needs human_label.reference)
        "bleu",
        "rouge_l",
        "token_overlap_f1",
        # Quality
        "completeness",
        "cultural_sensitivity",
        # Basic
        "output_nonempty",
        # Efficiency
        "latency_ms",
    ],
}

# Bundle descriptions for CLI help
BUNDLE_DESCRIPTIONS: Dict[str, str] = {
    "chatbot": "General conversational AI - safety, helpfulness, tone",
    "customer-support": "Support ticket handling - safety, helpfulness, completeness",
    "content-writer": "Marketing copy, blog posts - style, clarity, instructions",
    "summarization": "Text summarization - reference overlap, conciseness, grounding",
    "creative-writer": "Storytelling, brainstorming - coherence, tone, creativity",
    "rag-qa": "RAG/Question answering - grounding, accuracy, citations",
    "research-agent": "Research tasks - citations, grounding, tool use",
    "tutor": "Educational explanations - clarity, accuracy, reasoning",
    "code-assistant": "Code generation/review - correctness, reasoning, structure",
    "data-extraction": "Structured output from text - JSON validity, schema compliance",
    "orchestrator": "Tool orchestration - tool success, planning, error handling",
    "multi-step-agent": "Complex multi-step tasks - planning, reasoning, context",
    "medical-advisor": "Healthcare (high-stakes) - safety, accuracy, ethics",
    "legal-assistant": "Legal research (high-stakes) - accuracy, citations, completeness",
    "financial-advisor": "Financial advice (high-stakes) - safety, accuracy, ethics",
    "moderator": "Content moderation - toxicity, bias, PII, manipulation",
    "translator": "Language translation - BLEU, completeness, cultural sensitivity",
}

__all__ = ["DEFAULT_CONFIG_PATHS", "BUNDLES", "BUNDLE_DESCRIPTIONS"]
