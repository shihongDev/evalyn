from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemplateConfig:
    """Configuration for a pipeline template."""

    name: str
    goal: str = ""
    metrics: list[str] = field(default_factory=list)
    provider: str = "gemini"
    profile: str = "standard"
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal,
            "metrics": list(self.metrics),
            "provider": self.provider,
            "profile": self.profile,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemplateConfig:
        return cls(
            name=data["name"],
            goal=data.get("goal", ""),
            metrics=data.get("metrics", []),
            provider=data.get("provider", "gemini"),
            profile=data.get("profile", "standard"),
            description=data.get("description", ""),
        )


@dataclass
class TemplateLibrary:
    """Collection of pipeline templates."""

    templates: dict[str, TemplateConfig] = field(default_factory=dict)

    def add(self, config: TemplateConfig) -> None:
        self.templates[config.name] = config

    def get(self, name: str) -> TemplateConfig | None:
        return self.templates.get(name)

    def list_all(self) -> list[TemplateConfig]:
        return list(self.templates.values())

    def list_by_goal(self, goal: str) -> list[TemplateConfig]:
        return [t for t in self.templates.values() if t.goal == goal]

    def remove(self, name: str) -> bool:
        if name in self.templates:
            del self.templates[name]
            return True
        return False

    def as_dict(self) -> dict:
        return {
            "templates": {k: v.as_dict() for k, v in self.templates.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemplateLibrary:
        templates = {k: TemplateConfig.from_dict(v) for k, v in data.get("templates", {}).items()}
        return cls(templates=templates)


BUILTIN_TEMPLATES: dict[str, TemplateConfig] = {
    "safety-gate": TemplateConfig(
        "safety-gate",
        "safety",
        ["toxicity_safety", "prompt_injection"],
        "gemini",
        "smoke-test",
        "Quick safety check before deployment",
    ),
    "quality-check": TemplateConfig(
        "quality-check",
        "quality",
        ["helpfulness", "output_nonempty"],
        "gemini",
        "standard",
        "Standard quality evaluation",
    ),
    "regression-test": TemplateConfig(
        "regression-test",
        "regression",
        ["output_nonempty"],
        "gemini",
        "thorough",
        "Thorough regression testing with confidence",
    ),
    "cost-audit": TemplateConfig(
        "cost-audit",
        "cost",
        ["output_nonempty"],
        "gemini",
        "cost-optimized",
        "Cost-optimized evaluation run",
    ),
}


def get_template(name: str) -> TemplateConfig:
    """Look up a built-in template by name. Raise ValueError if not found."""
    if name in BUILTIN_TEMPLATES:
        return BUILTIN_TEMPLATES[name]
    raise ValueError(f"Template not found: {name}")


def list_templates() -> list[TemplateConfig]:
    """Return all built-in templates."""
    return list(BUILTIN_TEMPLATES.values())


def get_default_library() -> TemplateLibrary:
    """Return a TemplateLibrary pre-populated with built-in templates."""
    return TemplateLibrary(templates=dict(BUILTIN_TEMPLATES))


def suggest_template(goal: str) -> TemplateConfig | None:
    """Suggest the best template for a given goal. Returns None if no match."""
    for t in BUILTIN_TEMPLATES.values():
        if t.goal == goal:
            return t
    return None
