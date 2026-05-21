"""
Pre-built simulation configs for common evaluation use cases.

Pure Python, no external dependencies. Provides a library of templates
with persona mixes, edge case types, and topic lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PersonaMix:
    """A persona with its weight in the simulation mix."""

    persona_id: str
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaMix:
        return cls(
            persona_id=data["persona_id"],
            weight=data["weight"],
        )


@dataclass
class SimulationTemplate:
    """A complete simulation configuration template."""

    template_id: str
    name: str
    description: str
    persona_mix: list[PersonaMix] = field(default_factory=list)
    edge_case_types: list[str] = field(default_factory=list)
    output_format: str = "text"
    topics: list[str] = field(default_factory=list)
    min_turns: int = 1
    max_turns: int = 1
    constraints: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "persona_mix": [p.as_dict() for p in self.persona_mix],
            "edge_case_types": list(self.edge_case_types),
            "output_format": self.output_format,
            "topics": list(self.topics),
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationTemplate:
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            persona_mix=[PersonaMix.from_dict(p) for p in data.get("persona_mix", [])],
            edge_case_types=data.get("edge_case_types", []),
            output_format=data.get("output_format", "text"),
            topics=data.get("topics", []),
            min_turns=data.get("min_turns", 1),
            max_turns=data.get("max_turns", 1),
            constraints=data.get("constraints", []),
        )


@dataclass
class TemplateLibrary:
    """Collection of simulation templates keyed by template_id."""

    templates: dict[str, SimulationTemplate] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "templates": {k: v.as_dict() for k, v in self.templates.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateLibrary:
        return cls(
            templates={
                k: SimulationTemplate.from_dict(v) for k, v in data.get("templates", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATES = TemplateLibrary(
    templates={
        "customer_support": SimulationTemplate(
            template_id="customer_support",
            name="Customer Support",
            description="Simulate customer support interactions across skill levels",
            persona_mix=[
                PersonaMix(persona_id="novice", weight=0.4),
                PersonaMix(persona_id="power_user", weight=0.3),
                PersonaMix(persona_id="adversarial", weight=0.2),
                PersonaMix(persona_id="non_native", weight=0.1),
            ],
            edge_case_types=["empty", "special_chars"],
            output_format="text",
            topics=["refund", "billing", "account", "password reset"],
            min_turns=1,
            max_turns=1,
            constraints=[],
        ),
        "rag_qa": SimulationTemplate(
            template_id="rag_qa",
            name="RAG QA",
            description="Simulate retrieval-augmented generation question answering",
            persona_mix=[
                PersonaMix(persona_id="novice", weight=0.3),
                PersonaMix(persona_id="expert", weight=0.5),
                PersonaMix(persona_id="adversarial", weight=0.2),
            ],
            edge_case_types=["unicode", "max_length"],
            output_format="text",
            topics=["knowledge retrieval", "fact checking", "source citation"],
            min_turns=1,
            max_turns=1,
            constraints=[],
        ),
        "code_review": SimulationTemplate(
            template_id="code_review",
            name="Code Review",
            description="Simulate code review interactions for various quality aspects",
            persona_mix=[
                PersonaMix(persona_id="expert", weight=0.6),
                PersonaMix(persona_id="novice", weight=0.2),
                PersonaMix(persona_id="adversarial", weight=0.2),
            ],
            edge_case_types=["special_chars", "max_length"],
            output_format="text",
            topics=["bug detection", "style", "security", "performance"],
            min_turns=1,
            max_turns=1,
            constraints=[],
        ),
        "multi_step_agent": SimulationTemplate(
            template_id="multi_step_agent",
            name="Multi-Step Agent",
            description="Simulate multi-turn agent interactions requiring planning",
            persona_mix=[
                PersonaMix(persona_id="power_user", weight=0.5),
                PersonaMix(persona_id="expert", weight=0.3),
                PersonaMix(persona_id="adversarial", weight=0.2),
            ],
            edge_case_types=["boundary", "contradiction"],
            output_format="text",
            topics=["research", "planning", "tool use"],
            min_turns=2,
            max_turns=6,
            constraints=[],
        ),
    }
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_template(template_id: str) -> SimulationTemplate | None:
    """Get a built-in template by ID. Returns None if not found."""
    return BUILTIN_TEMPLATES.templates.get(template_id)


def list_templates() -> list[str]:
    """Return sorted list of built-in template IDs."""
    return sorted(BUILTIN_TEMPLATES.templates.keys())


def format_template_catalog(library: TemplateLibrary) -> str:
    """Format a human-readable catalog of all templates in a library."""
    lines: list[str] = []
    lines.append("SIMULATION TEMPLATE CATALOG")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Templates: {len(library.templates)}")
    lines.append("")

    for tid in sorted(library.templates.keys()):
        tmpl = library.templates[tid]
        lines.append(f"  {tid}")
        lines.append(f"    {tmpl.name} - {tmpl.description}")
        persona_strs = [f"{p.persona_id}({p.weight})" for p in tmpl.persona_mix]
        lines.append(f"    Personas: {', '.join(persona_strs)}")
        lines.append(f"    Topics: {', '.join(tmpl.topics)}")
        if tmpl.max_turns > 1:
            lines.append(f"    Turns: {tmpl.min_turns}-{tmpl.max_turns}")
        lines.append("")

    return "\n".join(lines)


def format_template_detail(template: SimulationTemplate) -> str:
    """Format a detailed view of a single template."""
    lines: list[str] = []
    lines.append(f"Template: {template.name} ({template.template_id})")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Description: {template.description}")
    lines.append(f"Output format: {template.output_format}")
    lines.append(f"Turns: {template.min_turns}-{template.max_turns}")
    lines.append("")

    lines.append("Persona Mix:")
    for p in template.persona_mix:
        lines.append(f"  {p.persona_id}: {p.weight:.0%}")
    lines.append("")

    if template.edge_case_types:
        lines.append(f"Edge Cases: {', '.join(template.edge_case_types)}")

    if template.topics:
        lines.append(f"Topics: {', '.join(template.topics)}")

    if template.constraints:
        lines.append(f"Constraints: {', '.join(template.constraints)}")

    lines.append("")
    return "\n".join(lines)


def customize_template(
    base: SimulationTemplate,
    overrides: dict,
) -> SimulationTemplate:
    """Create a modified copy of a template with the given overrides.

    Supported override keys match SimulationTemplate fields. The persona_mix
    override should be a list of dicts with persona_id and weight.

    Args:
        base: the template to copy
        overrides: dict of field names to new values

    Returns:
        New SimulationTemplate with overrides applied.
    """
    data = base.as_dict()
    for key, value in overrides.items():
        if key == "persona_mix" and isinstance(value, list):
            # Accept list of dicts or list of PersonaMix
            data["persona_mix"] = [v.as_dict() if isinstance(v, PersonaMix) else v for v in value]
        else:
            data[key] = value
    return SimulationTemplate.from_dict(data)
