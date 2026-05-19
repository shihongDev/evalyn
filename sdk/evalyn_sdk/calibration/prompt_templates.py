"""Calibration prompt templates.

Reusable preamble templates for common calibration patterns.
Provides dataclasses for template management and a library of
built-in evaluation templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable placeholders."""

    name: str
    template: str
    variables: list[str] = field(default_factory=list)
    description: str = ""
    category: str = "general"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        return cls(
            name=data["name"],
            template=data["template"],
            variables=data.get("variables", []),
            description=data.get("description", ""),
            category=data.get("category", "general"),
        )


@dataclass
class TemplateLibrary:
    """A collection of named prompt templates."""

    templates: dict[str, PromptTemplate] = field(default_factory=dict)

    def add(self, template: PromptTemplate) -> None:
        """Add a template to the library."""
        self.templates[template.name] = template

    def get(self, name: str) -> PromptTemplate | None:
        """Get a template by name, or None if not found."""
        return self.templates.get(name)

    def list_by_category(self, category: str) -> list[PromptTemplate]:
        """List all templates in a given category."""
        return [
            t for t in self.templates.values() if t.category == category
        ]

    def list_all(self) -> list[PromptTemplate]:
        """List all templates."""
        return list(self.templates.values())

    def remove(self, name: str) -> bool:
        """Remove a template by name. Returns True if it existed."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "templates": {
                name: t.as_dict() for name, t in self.templates.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateLibrary:
        templates = {
            name: PromptTemplate.from_dict(t_data)
            for name, t_data in data.get("templates", {}).items()
        }
        return cls(templates=templates)


# ---------------------------------------------------------------------------
# Built-in Templates
# ---------------------------------------------------------------------------


BUILTIN_TEMPLATES = {
    "binary_judge": PromptTemplate(
        name="binary_judge",
        template=(
            "Evaluate if the following output meets the {{metric_name}} criteria."
            " Respond with PASS or FAIL.\n\nOutput: {{output}}"
        ),
        variables=["metric_name", "output"],
        category="evaluation",
    ),
    "likert_judge": PromptTemplate(
        name="likert_judge",
        template=(
            "Rate the {{metric_name}} of the following output on a scale of 1-5."
            "\n\nOutput: {{output}}"
        ),
        variables=["metric_name", "output"],
        category="evaluation",
    ),
    "comparison_judge": PromptTemplate(
        name="comparison_judge",
        template=(
            "Compare outputs A and B for {{metric_name}}. Which is better?"
            "\n\nA: {{output_a}}\nB: {{output_b}}"
        ),
        variables=["metric_name", "output_a", "output_b"],
        category="comparison",
    ),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def render_template(
    template: PromptTemplate,
    variables: dict[str, str],
) -> str:
    """Replace {{variable}} placeholders with values.

    Raises ValueError if a required variable is missing from variables dict.
    """
    required = set(_PLACEHOLDER_RE.findall(template.template))
    missing = required - set(variables.keys())
    if missing:
        raise ValueError(f"Missing variables: {', '.join(sorted(missing))}")

    result = template.template
    for var_name, value in variables.items():
        result = result.replace("{{" + var_name + "}}", value)
    return result


def get_builtin_templates() -> TemplateLibrary:
    """Return a TemplateLibrary pre-loaded with built-in templates."""
    library = TemplateLibrary()
    for tmpl in BUILTIN_TEMPLATES.values():
        library.add(tmpl)
    return library
