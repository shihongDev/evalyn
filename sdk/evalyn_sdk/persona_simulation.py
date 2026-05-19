"""
Persona-based simulation: generate inputs as specific user personas.

Provides a framework for defining personas with traits, language styles,
and expertise levels, then generating styled inputs for evaluation.
Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Persona:
    """A user persona with traits, style, and expertise level."""

    persona_id: str
    name: str
    description: str
    traits: list[str]
    language_style: str  # "formal" / "casual" / "technical" / "simple"
    expertise_level: str  # "novice" / "intermediate" / "expert" / "adversarial"
    custom_instructions: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "description": self.description,
            "traits": list(self.traits),
            "language_style": self.language_style,
            "expertise_level": self.expertise_level,
            "custom_instructions": self.custom_instructions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        return cls(
            persona_id=data["persona_id"],
            name=data["name"],
            description=data["description"],
            traits=data.get("traits", []),
            language_style=data.get("language_style", "casual"),
            expertise_level=data.get("expertise_level", "intermediate"),
            custom_instructions=data.get("custom_instructions", ""),
        )


@dataclass
class PersonaTemplate:
    """A text template for generating persona-styled prompts."""

    template_id: str
    text_template: str  # Python format string: {persona_name}, {trait_list}, {style}, {topic}
    persona_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "text_template": self.text_template,
            "persona_id": self.persona_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaTemplate:
        return cls(
            template_id=data["template_id"],
            text_template=data["text_template"],
            persona_id=data["persona_id"],
        )


@dataclass
class SimulatedInput:
    """A single generated input tied to a persona."""

    input_text: str
    persona_id: str
    persona_name: str
    tags: list[str]
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "tags": list(self.tags),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulatedInput:
        return cls(
            input_text=data["input_text"],
            persona_id=data["persona_id"],
            persona_name=data["persona_name"],
            tags=data.get("tags", []),
            generated_at=data.get("generated_at", ""),
        )


# ---------------------------------------------------------------------------
# Built-in Personas
# ---------------------------------------------------------------------------

BUILTIN_PERSONAS: dict[str, Persona] = {
    "novice": Persona(
        persona_id="novice",
        name="Novice User",
        description="A beginner who asks basic questions and needs simple explanations.",
        traits=["curious", "unfamiliar with jargon", "asks clarifying questions"],
        language_style="simple",
        expertise_level="novice",
    ),
    "power_user": Persona(
        persona_id="power_user",
        name="Power User",
        description="A technical user who makes specific, detailed requests.",
        traits=["precise", "uses technical terms", "expects detailed answers"],
        language_style="technical",
        expertise_level="expert",
    ),
    "adversarial": Persona(
        persona_id="adversarial",
        name="Adversarial User",
        description="A user who attempts to break the system or find edge cases.",
        traits=["provocative", "tests boundaries", "uses unusual inputs"],
        language_style="casual",
        expertise_level="adversarial",
    ),
    "non_native": Persona(
        persona_id="non_native",
        name="Non-Native Speaker",
        description="A user with grammatical quirks and simpler vocabulary.",
        traits=["sometimes omits articles", "shorter sentences", "direct phrasing"],
        language_style="simple",
        expertise_level="intermediate",
    ),
    "enterprise": Persona(
        persona_id="enterprise",
        name="Enterprise User",
        description="A formal, compliance-focused business user.",
        traits=["formal tone", "references policies", "risk-aware"],
        language_style="formal",
        expertise_level="expert",
    ),
}


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def generate_persona_prompt(persona: Persona, topic: str) -> str:
    """Generate a system-instruction-style prompt describing this persona's behavior.

    Returns a multi-line string suitable for use as an LLM system prompt
    that instructs the model to behave as the given persona on the topic.
    """
    trait_str = ", ".join(persona.traits) if persona.traits else "none specified"
    lines = [
        f"You are {persona.name}.",
        f"Description: {persona.description}",
        f"Traits: {trait_str}",
        f"Language style: {persona.language_style}",
        f"Expertise level: {persona.expertise_level}",
        f"Topic: {topic}",
    ]
    if persona.custom_instructions:
        lines.append(f"Additional instructions: {persona.custom_instructions}")
    return "\n".join(lines)


def apply_persona_style(text: str, persona: Persona) -> str:
    """Transform text to match a persona's language style and expertise.

    Applies simple heuristic transformations:
    - adversarial: uppercase the text
    - novice/simple: prefix with "Can you explain" and add question mark
    - enterprise/formal: add qualifiers and formal framing
    - technical/expert: no change (already assumed precise)
    - other: return text unchanged
    """
    if persona.expertise_level == "adversarial":
        return text.upper()

    if persona.expertise_level == "novice" or persona.language_style == "simple":
        clean = text.rstrip("?. ")
        return f"Can you explain {clean}?"

    if persona.language_style == "formal":
        return f"Per our requirements, please provide information regarding: {text}"

    # technical / intermediate / default: return as-is
    return text


def generate_inputs_for_persona(
    persona: Persona,
    topics: list[str],
    template: PersonaTemplate | None = None,
) -> list[SimulatedInput]:
    """Generate one SimulatedInput per topic for a persona.

    If a template is provided, uses it to format the input text.
    Otherwise, uses apply_persona_style on each topic.
    """
    results: list[SimulatedInput] = []
    now = _now_iso()
    for topic in topics:
        if template is not None:
            trait_list = ", ".join(persona.traits)
            input_text = template.text_template.format(
                persona_name=persona.name,
                trait_list=trait_list,
                style=persona.language_style,
                topic=topic,
            )
        else:
            input_text = apply_persona_style(topic, persona)
        results.append(
            SimulatedInput(
                input_text=input_text,
                persona_id=persona.persona_id,
                persona_name=persona.name,
                tags=[persona.expertise_level, persona.language_style],
                generated_at=now,
            )
        )
    return results


def generate_cohort(
    personas: list[Persona],
    topics: list[str],
) -> list[SimulatedInput]:
    """Generate inputs for all persona x topic combinations."""
    results: list[SimulatedInput] = []
    for persona in personas:
        results.extend(generate_inputs_for_persona(persona, topics))
    return results


def filter_by_persona(
    inputs: list[SimulatedInput],
    persona_id: str,
) -> list[SimulatedInput]:
    """Filter simulated inputs to only those matching the given persona_id."""
    return [inp for inp in inputs if inp.persona_id == persona_id]


def format_persona_catalog(personas: dict[str, Persona]) -> str:
    """List all personas with descriptions, one per line.

    Returns a newline-separated string with persona_id, name, and description.
    """
    lines: list[str] = []
    for pid, persona in personas.items():
        lines.append(f"{pid}: {persona.name} - {persona.description}")
    return "\n".join(lines)
