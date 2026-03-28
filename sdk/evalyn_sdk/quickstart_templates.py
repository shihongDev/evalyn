"""
Quickstart templates for framework-specific guided setup.

Provides built-in templates for LangChain, LangGraph, CrewAI, and
vanilla Python projects, along with functions for listing, planning,
rendering, and suggesting templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class QuickstartTemplate:
    """A framework-specific quickstart template."""

    name: str
    framework: str
    description: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    setup_commands: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "framework": self.framework,
            "description": self.description,
            "files": dict(self.files),
            "dependencies": list(self.dependencies),
            "setup_commands": list(self.setup_commands),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuickstartTemplate:
        return cls(
            name=data.get("name", ""),
            framework=data.get("framework", ""),
            description=data.get("description", ""),
            files=data.get("files", {}),
            dependencies=data.get("dependencies", []),
            setup_commands=data.get("setup_commands", []),
        )


@dataclass
class QuickstartResult:
    """Result of planning a quickstart (dry run)."""

    template_name: str
    files_created: List[str] = field(default_factory=list)
    commands_to_run: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "files_created": list(self.files_created),
            "commands_to_run": list(self.commands_to_run),
        }

    def format_text(self) -> str:
        lines = [f"Template: {self.template_name}"]
        if self.files_created:
            lines.append("Files to create:")
            for f in self.files_created:
                lines.append(f"  - {f}")
        if self.commands_to_run:
            lines.append("Commands to run:")
            for cmd in self.commands_to_run:
                lines.append(f"  $ {cmd}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in Templates
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, QuickstartTemplate] = {
    "langchain": QuickstartTemplate(
        "langchain",
        "langchain",
        "LangChain agent evaluation",
        {
            "evalyn.yaml": (
                "provider: gemini\n"
                "metrics:\n"
                "  - helpfulness\n"
                "  - tool_call_accuracy\n"
            ),
            "agent.py": "# LangChain agent template\n",
        },
        ["langchain", "evalyn-sdk"],
        ["uv pip install langchain evalyn-sdk"],
    ),
    "langgraph": QuickstartTemplate(
        "langgraph",
        "langgraph",
        "LangGraph workflow evaluation",
        {
            "evalyn.yaml": (
                "provider: gemini\n"
                "metrics:\n"
                "  - helpfulness\n"
                "  - goal_completion\n"
            ),
            "graph.py": "# LangGraph workflow template\n",
        },
        ["langgraph", "evalyn-sdk"],
        ["uv pip install langgraph evalyn-sdk"],
    ),
    "crewai": QuickstartTemplate(
        "crewai",
        "crewai",
        "CrewAI multi-agent evaluation",
        {
            "evalyn.yaml": (
                "provider: gemini\n"
                "metrics:\n"
                "  - helpfulness\n"
                "  - multi_agent_communication\n"
            ),
            "crew.py": "# CrewAI template\n",
        },
        ["crewai", "evalyn-sdk"],
        ["uv pip install crewai evalyn-sdk"],
    ),
    "vanilla": QuickstartTemplate(
        "vanilla",
        "python",
        "Plain Python with Gemini API",
        {
            "evalyn.yaml": (
                "provider: gemini\n"
                "metrics:\n"
                "  - output_nonempty\n"
                "  - helpfulness\n"
            ),
            "main.py": "# Vanilla Python agent template\n",
        },
        ["google-generativeai", "evalyn-sdk"],
        ["uv pip install google-generativeai evalyn-sdk"],
    ),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_quickstart(name: str) -> QuickstartTemplate:
    """Look up a quickstart template by name.

    Raises ValueError if the name is not found.
    """
    if name not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise ValueError(
            f"Unknown quickstart template: {name!r}. Available: {available}"
        )
    return TEMPLATES[name]


def list_quickstarts() -> List[QuickstartTemplate]:
    """Return all available quickstart templates."""
    return list(TEMPLATES.values())


def plan_quickstart(name: str) -> QuickstartResult:
    """Plan what a quickstart would create (dry run).

    Returns a QuickstartResult describing files and commands without
    actually creating anything.
    """
    template = get_quickstart(name)
    return QuickstartResult(
        template_name=template.name,
        files_created=list(template.files.keys()),
        commands_to_run=list(template.setup_commands),
    )


def render_quickstart_guide(template: QuickstartTemplate) -> str:
    """Render a formatted getting-started guide for a template.

    Uses GEMINI_API_KEY from environment in the generated guide text.
    """
    lines = [
        f"Quickstart: {template.name}",
        f"Framework: {template.framework}",
        "",
    ]
    if template.description:
        lines.append(template.description)
        lines.append("")

    lines.append("Step 1: Install dependencies")
    for cmd in template.setup_commands:
        lines.append(f"  $ {cmd}")
    lines.append("")

    lines.append("Step 2: Set your API key")
    lines.append("  $ export GEMINI_API_KEY=<your-key>")
    lines.append("")

    lines.append("Step 3: Files created")
    for filename, content in template.files.items():
        lines.append(f"  {filename}:")
        for content_line in content.strip().splitlines():
            lines.append(f"    {content_line}")
    lines.append("")

    lines.append("Step 4: Run evaluation")
    lines.append("  $ evalyn run")

    return "\n".join(lines)


def suggest_quickstart(framework: str) -> Optional[QuickstartTemplate]:
    """Suggest the best template for a given framework name.

    Performs case-insensitive matching against template names and
    framework fields. Returns None if no match is found.
    """
    lower = framework.lower().strip()
    if not lower:
        return None

    # Exact match on template name
    if lower in TEMPLATES:
        return TEMPLATES[lower]

    # Match on framework field
    for template in TEMPLATES.values():
        if template.framework.lower() == lower:
            return template

    # Substring match on name or framework
    for template in TEMPLATES.values():
        if lower in template.name.lower() or lower in template.framework.lower():
            return template

    return None
