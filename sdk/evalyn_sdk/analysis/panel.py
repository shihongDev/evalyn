"""
Expert panel: LLM-powered multi-perspective analysis of evaluation results.

Provides sequential expert consultation where each expert builds on prior opinions,
producing qualitative reasoning and synthesis beyond rule-based insights.

Expert roles:
- quality_analyst: focuses on metric health, pass/fail patterns, rubric quality
- metric_critic: questions metric design, thresholds, and measurement validity
- data_scientist: examines statistical patterns, distributions, correlations
- strategist: synthesizes into actionable improvement roadmap
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .core import RunAnalysis
from .insights import InsightsReport

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

EXPERT_ROLES = ["quality_analyst", "metric_critic", "data_scientist", "strategist"]


@dataclass
class ExpertOpinion:
    """One expert's analysis of the evaluation results."""

    role: str
    summary: str
    findings: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "summary": self.summary,
            "findings": self.findings,
            "concerns": self.concerns,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpertOpinion:
        return cls(
            role=data.get("role", "unknown"),
            summary=data.get("summary", ""),
            findings=data.get("findings", []),
            concerns=data.get("concerns", []),
            suggestions=data.get("suggestions", []),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class PanelDiscussion:
    """Complete expert panel discussion with synthesis."""

    experts: list[ExpertOpinion] = field(default_factory=list)
    synthesis: str = ""
    action_plan: list[str] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "experts": [e.as_dict() for e in self.experts],
            "synthesis": self.synthesis,
            "action_plan": self.action_plan,
            "dissenting_views": self.dissenting_views,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PanelDiscussion:
        return cls(
            experts=[ExpertOpinion.from_dict(e) for e in data.get("experts", [])],
            synthesis=data.get("synthesis", ""),
            action_plan=data.get("action_plan", []),
            dissenting_views=data.get("dissenting_views", []),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
        )


# ---------------------------------------------------------------------------
# Context preparation
# ---------------------------------------------------------------------------


def prepare_panel_context(
    run_analysis: RunAnalysis,
    insights_report: InsightsReport,
    dataset_items: list[dict[str, Any]] | None = None,
    sample_failures: int = 5,
) -> str:
    """Condense RunAnalysis + InsightsReport into compact text for LLM context.

    Targets ~1500 tokens. Includes metric stats, top failures, and insight findings.
    """
    lines = []

    # Header
    lines.append(f"Dataset: {run_analysis.dataset_name}")
    lines.append(f"Items: {run_analysis.total_items}, Metrics: {run_analysis.total_metrics}")
    lines.append(f"Overall pass rate: {run_analysis.overall_pass_rate:.1%}")
    lines.append("")

    # Metric stats table
    lines.append("METRIC STATS:")
    for mid, ms in sorted(run_analysis.metric_stats.items()):
        pr = f"{ms.pass_rate:.0%}" if ms.pass_rate is not None else "N/A"
        lines.append(
            f"  {mid}: pass_rate={pr}, avg={ms.avg_score:.2f}, "
            f"std={ms.std_dev:.2f}, n={ms.count}"
        )
    lines.append("")

    # Top failed items with reasons
    if run_analysis.failed_items:
        lines.append(f"TOP FAILED ITEMS ({min(sample_failures, len(run_analysis.failed_items))}):")
        for item_id in run_analysis.failed_items[:sample_failures]:
            ist = run_analysis.item_stats[item_id]
            failed_metrics = [
                m for m, r in ist.metric_results.items() if r["passed"] is False
            ]
            reasons = []
            for m in failed_metrics[:3]:
                reason = ist.metric_results[m].get("reason") or ""
                if reason:
                    reasons.append(f"{m}: {reason[:80]}")
                else:
                    reasons.append(m)
            lines.append(f"  {item_id[:16]}... failed: {', '.join(reasons)}")
        lines.append("")

    # Insight findings
    if insights_report.correlations:
        lines.append("CORRELATIONS:")
        for c in insights_report.correlations[:5]:
            lines.append(f"  {c.metric_a} <-> {c.metric_b}: r={c.pearson} ({c.relationship})")
        lines.append("")

    if insights_report.regressions:
        lines.append("REGRESSIONS:")
        for r in insights_report.regressions[:5]:
            lines.append(
                f"  [{r.severity}] {r.metric_id}: {r.previous_pass_rate:.0%} -> {r.current_pass_rate:.0%}"
            )
        lines.append("")

    if insights_report.distribution_insights:
        lines.append("DISTRIBUTIONS:")
        for d in insights_report.distribution_insights[:5]:
            lines.append(f"  {d.metric_id}: {d.shape} - {d.finding}")
        lines.append("")

    if insights_report.feature_insights:
        lines.append("FEATURE INSIGHTS:")
        for f in insights_report.feature_insights[:3]:
            lines.append(f"  {f.finding}")
        lines.append("")

    if insights_report.recommendations:
        lines.append("RECOMMENDATIONS:")
        for rec in insights_report.recommendations[:5]:
            lines.append(f"  {rec.priority}. [{rec.category}] {rec.message}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_ROLE_INSTRUCTIONS = {
    "quality_analyst": (
        "You are a Quality Analyst reviewing LLM evaluation results. "
        "Focus on: metric health, pass/fail patterns, which items struggle and why, "
        "rubric quality indicators. Identify the most impactful quality issues."
    ),
    "metric_critic": (
        "You are a Metric Critic reviewing LLM evaluation design. "
        "Focus on: are the metrics measuring what matters? Are thresholds appropriate? "
        "Are any metrics redundant or missing? Question the measurement validity."
    ),
    "data_scientist": (
        "You are a Data Scientist reviewing LLM evaluation statistics. "
        "Focus on: distribution shapes, correlations, statistical significance, "
        "sample sizes, potential confounders. Flag where data is insufficient for conclusions."
    ),
    "strategist": (
        "You are a Strategy Lead reviewing LLM evaluation results. "
        "Focus on: what should the team do next? Prioritize improvements by impact. "
        "Consider resource constraints. Provide a concrete action roadmap."
    ),
}

_OUTPUT_SCHEMA = """\
Respond in JSON with this exact structure:
{
  "summary": "2-3 sentence overview of your analysis",
  "findings": ["finding 1", "finding 2", ...],
  "concerns": ["concern 1", ...],
  "suggestions": ["suggestion 1", ...],
  "confidence": 0.8
}
Rules:
- Max 5 findings, 3 concerns, 3 suggestions
- confidence: 0.0-1.0 based on data quality
- Be specific, reference metric names and numbers
- No markdown formatting in the JSON values"""


def build_expert_prompt(
    role: str,
    context: str,
    prior_opinions: list[ExpertOpinion] | None = None,
) -> str:
    """Build prompt for one expert, including prior opinions for sequential reasoning."""
    parts = []

    # Role instruction
    instruction = _ROLE_INSTRUCTIONS.get(role, _ROLE_INSTRUCTIONS["quality_analyst"])
    parts.append(instruction)
    parts.append("")

    # Evaluation data
    parts.append("=== EVALUATION DATA ===")
    parts.append(context)

    # Prior expert opinions
    if prior_opinions:
        parts.append("=== PRIOR EXPERT OPINIONS ===")
        for op in prior_opinions:
            parts.append(f"\n[{op.role}] {op.summary}")
            if op.findings:
                parts.append(f"  Key findings: {'; '.join(op.findings[:3])}")
            if op.concerns:
                parts.append(f"  Concerns: {'; '.join(op.concerns[:2])}")
        parts.append("")
        parts.append("Build on or challenge the above opinions where appropriate.")
        parts.append("")

    parts.append(_OUTPUT_SCHEMA)

    return "\n".join(parts)


def build_moderator_prompt(
    context: str,
    opinions: list[ExpertOpinion],
) -> str:
    """Build the moderator synthesis prompt."""
    parts = [
        "You are a Moderator synthesizing expert opinions on LLM evaluation results.",
        "",
        "=== EVALUATION DATA ===",
        context,
        "",
        "=== EXPERT OPINIONS ===",
    ]

    for op in opinions:
        parts.append(f"\n[{op.role}] (confidence: {op.confidence})")
        parts.append(f"Summary: {op.summary}")
        if op.findings:
            for f in op.findings:
                parts.append(f"  - {f}")
        if op.concerns:
            parts.append(f"  Concerns: {'; '.join(op.concerns)}")
        if op.suggestions:
            parts.append(f"  Suggestions: {'; '.join(op.suggestions)}")

    parts.append("")
    parts.append(
        "Synthesize these opinions into a unified analysis. "
        "Identify where experts agree and disagree."
    )
    parts.append("")
    parts.append("""\
Respond in JSON with this exact structure:
{
  "synthesis": "3-5 sentence unified analysis",
  "action_plan": ["step 1", "step 2", ...],
  "dissenting_views": ["disagreement 1", ...]
}
Rules:
- Max 7 action steps, ordered by priority
- Max 3 dissenting views
- Reference specific expert roles when noting disagreements
- Be concrete and actionable""")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_json(raw_text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM response text.

    Handles JSON wrapped in markdown code blocks, bare JSON, or JSON
    embedded in surrounding text. Returns None if no valid JSON found.
    """
    text = raw_text.strip()

    # Strip markdown code block wrappers
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_expert_response(role: str, raw_text: str) -> ExpertOpinion:
    """Extract ExpertOpinion from LLM response text.

    Handles JSON wrapped in markdown code blocks or bare JSON.
    Falls back to a basic opinion if parsing fails.
    """
    data = _extract_json(raw_text)
    if data is None:
        return ExpertOpinion(
            role=role,
            summary=raw_text[:200],
            confidence=0.3,
        )

    return ExpertOpinion(
        role=role,
        summary=data.get("summary", ""),
        findings=data.get("findings", [])[:5],
        concerns=data.get("concerns", [])[:3],
        suggestions=data.get("suggestions", [])[:3],
        confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
    )


def _parse_moderator_response(raw_text: str) -> dict[str, Any]:
    """Parse moderator synthesis response."""
    data = _extract_json(raw_text)
    if data is not None:
        return data
    return {"synthesis": raw_text[:300], "action_plan": [], "dissenting_views": []}


# ---------------------------------------------------------------------------
# Client creation
# ---------------------------------------------------------------------------


def create_api_client(
    provider: str = "gemini",
    model: str | None = None,
    temperature: float = 0.3,
) -> Any:
    """Create an API client based on provider.

    Args:
        provider: "gemini", "openai", or "ollama"
        model: Model name override (uses provider defaults if None)
        temperature: Generation temperature

    Returns:
        GeminiClient, OpenAIClient, or OllamaClient instance
    """
    from ..utils.api_client import GeminiClient, OllamaClient, OpenAIClient

    default_models = {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "ollama": "llama3.2",
    }

    resolved_model = model or default_models.get(provider, "gemini-2.5-flash")

    if provider == "openai":
        return OpenAIClient(model=resolved_model, temperature=temperature)
    elif provider == "ollama":
        return OllamaClient(model=resolved_model, temperature=temperature)
    else:
        return GeminiClient(model=resolved_model, temperature=temperature)


# ---------------------------------------------------------------------------
# Expert panel orchestration
# ---------------------------------------------------------------------------


def run_expert_panel(
    run_analysis: RunAnalysis,
    insights_report: InsightsReport,
    client: Any,
    dataset_items: list[dict[str, Any]] | None = None,
    experts: list[str] | None = None,
    verbose: bool = False,
) -> PanelDiscussion:
    """Run the expert panel: sequential experts + moderator synthesis.

    Each expert sees prior experts' summaries, enabling progressive deepening.
    The moderator synthesizes all opinions into an action plan.

    Args:
        run_analysis: Analyzed run data
        insights_report: Deterministic insights report
        client: API client with generate_with_usage() method
        dataset_items: Optional raw dataset items for richer context
        experts: Subset of roles to run (default: all 4)
        verbose: Print progress to stdout

    Returns:
        PanelDiscussion with all expert opinions and synthesis
    """
    expert_roles = experts or EXPERT_ROLES
    # Validate roles
    expert_roles = [r for r in expert_roles if r in EXPERT_ROLES]
    if not expert_roles:
        expert_roles = EXPERT_ROLES

    context = prepare_panel_context(
        run_analysis, insights_report, dataset_items
    )

    opinions: list[ExpertOpinion] = []
    total_input = 0
    total_output = 0

    for role in expert_roles:
        if verbose:
            print(f"  Consulting {role}...")

        prompt = build_expert_prompt(role, context, prior_opinions=opinions or None)
        result = client.generate_with_usage(prompt)
        total_input += result.input_tokens
        total_output += result.output_tokens

        opinion = parse_expert_response(role, result.text)
        opinions.append(opinion)

        if verbose:
            print(f"    -> {opinion.summary[:80]}...")

    # Moderator synthesis
    if verbose:
        print("  Synthesizing expert opinions...")

    moderator_prompt = build_moderator_prompt(context, opinions)
    mod_result = client.generate_with_usage(moderator_prompt)
    total_input += mod_result.input_tokens
    total_output += mod_result.output_tokens

    mod_data = _parse_moderator_response(mod_result.text)

    discussion = PanelDiscussion(
        experts=opinions,
        synthesis=mod_data.get("synthesis", ""),
        action_plan=mod_data.get("action_plan", [])[:7],
        dissenting_views=mod_data.get("dissenting_views", [])[:3],
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )

    if verbose:
        print(f"  Done. Tokens used: {total_input} in, {total_output} out")

    return discussion


__all__ = [
    "ExpertOpinion",
    "PanelDiscussion",
    "EXPERT_ROLES",
    "prepare_panel_context",
    "build_expert_prompt",
    "build_moderator_prompt",
    "parse_expert_response",
    "create_api_client",
    "run_expert_panel",
]
