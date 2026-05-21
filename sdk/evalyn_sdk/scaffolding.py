from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectTemplate:
    """A project template with file contents and directory structure."""

    name: str
    description: str = ""
    files: dict[str, str] = field(default_factory=dict)  # relative_path -> content
    directories: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "files": dict(self.files),
            "directories": list(self.directories),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectTemplate:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            files=data.get("files", {}),
            directories=data.get("directories", []),
        )


@dataclass
class ScaffoldConfig:
    """Configuration for scaffolding a new project."""

    project_name: str = ""
    template: str = "basic"
    output_dir: str = "."
    provider: str = "gemini"
    include_tests: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "template": self.template,
            "output_dir": self.output_dir,
            "provider": self.provider,
            "include_tests": self.include_tests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScaffoldConfig:
        return cls(
            project_name=data.get("project_name", ""),
            template=data.get("template", "basic"),
            output_dir=data.get("output_dir", "."),
            provider=data.get("provider", "gemini"),
            include_tests=data.get("include_tests", True),
        )


@dataclass
class ScaffoldResult:
    """Result of a scaffolding operation."""

    created_files: list[str] = field(default_factory=list)
    created_dirs: list[str] = field(default_factory=list)
    project_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "created_files": list(self.created_files),
            "created_dirs": list(self.created_dirs),
            "project_path": self.project_path,
        }

    def format_text(self) -> str:
        lines = [f"Project: {self.project_path}"]
        if self.created_dirs:
            lines.append(f"Directories: {len(self.created_dirs)}")
            for d in self.created_dirs:
                lines.append(f"  {d}/")
        if self.created_files:
            lines.append(f"Files: {len(self.created_files)}")
            for f in self.created_files:
                lines.append(f"  {f}")
        return "\n".join(lines)


# -- Built-in templates -------------------------------------------------------

BASIC_TEMPLATE = ProjectTemplate(
    name="basic",
    description="Minimal evalyn project",
    files={
        "evalyn.yaml": "provider: {{provider}}\ndataset: data/\nmetrics:\n  - output_nonempty\n",
        "data/.gitkeep": "",
    },
    directories=["data", "results"],
)

ADVANCED_TEMPLATE = ProjectTemplate(
    name="advanced",
    description="Full evalyn project with calibration and CI",
    files={
        "evalyn.yaml": (
            "provider: {{provider}}\ndataset: data/\nmetrics:\n"
            "  - output_nonempty\n  - helpfulness\ncalibration:\n  enabled: true\n"
        ),
        "data/.gitkeep": "",
        "tests/test_eval.py": (
            "# Evaluation tests\nimport pytest\n\ndef test_placeholder():\n    pass\n"
        ),
    },
    directories=["data", "results", "tests", "calibration"],
)

_TEMPLATES: dict[str, ProjectTemplate] = {
    "basic": BASIC_TEMPLATE,
    "advanced": ADVANCED_TEMPLATE,
}


# -- Public functions ----------------------------------------------------------


def get_template(name: str) -> ProjectTemplate:
    """Look up a built-in template by name. Raises KeyError if not found."""
    if name not in _TEMPLATES:
        raise KeyError(f"Unknown template: {name!r}")
    return _TEMPLATES[name]


def list_templates() -> list[ProjectTemplate]:
    """Return all available templates."""
    return list(_TEMPLATES.values())


def render_template_file(content: str, variables: dict[str, str]) -> str:
    """Replace {{variable}} placeholders in content."""
    result = content
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def validate_project_name(name: str) -> tuple[bool, str]:
    """Check if the project name is valid.

    Valid names contain only alphanumeric characters, hyphens, and underscores.
    Must be non-empty and not start with a hyphen.
    """
    if not name:
        return False, "Project name must not be empty"
    if name.startswith("-"):
        return False, "Project name must not start with a hyphen"
    if not re.match(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$", name):
        return False, "Project name may only contain letters, digits, hyphens, and underscores"
    return True, ""


def plan_scaffold(config: ScaffoldConfig) -> ScaffoldResult:
    """Plan what would be created (dry run). Does not write to disk."""
    template = get_template(config.template)

    import os

    project_path = os.path.join(config.output_dir, config.project_name)

    # Collect directories
    created_dirs = []
    for d in template.directories:
        created_dirs.append(os.path.join(project_path, d))

    # Collect files
    created_files = []
    for rel_path in template.files:
        # Skip test files if include_tests is False
        if not config.include_tests and rel_path.startswith("tests/"):
            continue
        created_files.append(os.path.join(project_path, rel_path))

    return ScaffoldResult(
        created_files=created_files,
        created_dirs=created_dirs,
        project_path=project_path,
    )
