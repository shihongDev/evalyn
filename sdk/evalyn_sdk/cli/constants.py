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
# 4. Keep bundles focused (8-12 metrics) to balance coverage and evaluation cost
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
        # Multi-turn conversation
        "context_retention",
        "memory_consistency",
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
        # UX - critical for support
        "empathy",
        "patience",
        "escalation_appropriateness",
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
        "engagement",
        # Readability
        "flesch_kincaid",
        # Safety
        "toxicity_safety",
        # Structure
        "output_nonempty",
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
        # Compression
        "compression_ratio",
        # Safety
        "hallucination_risk",
        # Efficiency
        "latency_ms",
    ],
    "creative-writer": [
        # Creativity - key for this use case
        "originality",
        "engagement",
        # Quality
        "coherence_clarity",
        "completeness",
        "tone_alignment",
        "instruction_following",
        # Diversity
        "vocabulary_richness",
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
        "citation_count",
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
        "citation_count",
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
        # Explanation quality - key for education
        "pedagogical_clarity",
        "example_appropriateness",
        # Quality
        "helpfulness_accuracy",
        "factual_accuracy",
        "coherence_clarity",
        "reasoning_quality",
        "completeness",
        # UX
        "patience",
        # Safety
        "toxicity_safety",
        # Efficiency
        "latency_ms",
    ],
    # -------------------------------------------------------------------------
    # CODE & TECHNICAL
    # -------------------------------------------------------------------------
    "code-assistant": [
        # Code correctness
        "technical_accuracy",
        "syntax_valid",
        # Quality
        "instruction_following",
        "reasoning_quality",
        "helpfulness_accuracy",
        "completeness",
        # Code quality
        "code_complexity",
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
        "json_types_match",
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
        # Multi-turn consistency
        "context_retention",
        "memory_consistency",
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
        # Domain-specific accuracy - critical
        "medical_accuracy",
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
        # Domain-specific accuracy - critical
        "legal_compliance",
        # Accuracy (critical)
        "factual_accuracy",
        "technical_accuracy",
        "hallucination_risk",
        "source_attribution",
        "citation_count",
        # Quality
        "completeness",
        "relevance",
        # Safety
        "pii_safety",
        # Efficiency
        "latency_ms",
    ],
    "financial-advisor": [
        # Domain-specific accuracy - critical
        "financial_prudence",
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
        "levenshtein_similarity",
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
    "chatbot": "Conversational AI - safety, helpfulness, multi-turn memory",
    "customer-support": "Support tickets - empathy, patience, escalation handling",
    "content-writer": "Marketing/blogs - style, engagement, readability",
    "summarization": "Text summarization - compression, reference overlap, grounding",
    "creative-writer": "Storytelling - originality, engagement, vocabulary diversity",
    "rag-qa": "RAG/QA - grounding, citations, factual accuracy",
    "research-agent": "Research - citations, grounding, tool use",
    "tutor": "Education - pedagogical clarity, examples, patience",
    "code-assistant": "Coding - syntax validity, complexity, technical accuracy",
    "data-extraction": "Structured output - JSON validity, schema compliance",
    "orchestrator": "Tool orchestration - tool success, planning, error handling",
    "multi-step-agent": "Multi-step tasks - planning, context retention, memory",
    "medical-advisor": "Healthcare - medical accuracy, safety, ethics",
    "legal-assistant": "Legal - legal compliance, citations, accuracy",
    "financial-advisor": "Finance - financial prudence, safety, ethics",
    "moderator": "Content moderation - toxicity, bias, PII, manipulation",
    "translator": "Translation - BLEU, Levenshtein, cultural sensitivity",
}

__all__ = ["DEFAULT_CONFIG_PATHS", "BUNDLES", "BUNDLE_DESCRIPTIONS"]
