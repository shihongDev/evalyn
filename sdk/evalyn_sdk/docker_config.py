"""Dockerfile and docker-compose generation for evalyn.

Pure Python, no external dependencies. Generates Dockerfile, .dockerignore,
and docker-compose.yml content as strings - never makes actual Docker calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DockerConfig:
    """Configuration for Docker image generation."""

    base_image: str = "python:3.11-slim"
    python_version: str = "3.11"
    include_dev_deps: bool = False
    port: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_image": self.base_image,
            "python_version": self.python_version,
            "include_dev_deps": self.include_dev_deps,
            "port": self.port,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DockerConfig:
        return cls(
            base_image=data.get("base_image", "python:3.11-slim"),
            python_version=data.get("python_version", "3.11"),
            include_dev_deps=data.get("include_dev_deps", False),
            port=data.get("port", 0),
        )


def generate_dockerfile(config: DockerConfig) -> str:
    """Generate a Dockerfile string for evalyn.

    Produces a multi-stage-friendly Dockerfile with proper layer caching:
    dependencies installed first, then application code copied.
    """
    lines = [
        f"FROM {config.base_image}",
        "",
        "WORKDIR /app",
        "",
        "# Install system dependencies",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\",
        "    gcc \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
        "# Copy dependency files first for layer caching",
        "COPY pyproject.toml .",
        "COPY README.md .",
    ]

    if config.include_dev_deps:
        lines.append("")
        lines.append("# Install all dependencies including dev")
        lines.append("RUN pip install --no-cache-dir -e '.[dev]'")
    else:
        lines.append("")
        lines.append("# Install production dependencies")
        lines.append("RUN pip install --no-cache-dir -e '.'")

    lines.extend(
        [
            "",
            "# Copy application code",
            "COPY . .",
            "",
            "# Create non-root user",
            "RUN useradd --create-home evalyn",
            "USER evalyn",
        ]
    )

    if config.port > 0:
        lines.extend(
            [
                "",
                f"EXPOSE {config.port}",
            ]
        )

    lines.extend(
        [
            "",
            'ENTRYPOINT ["python", "-m", "evalyn_sdk.cli.main"]',
            "",
        ]
    )

    return "\n".join(lines)


def generate_dockerignore() -> str:
    """Generate .dockerignore content for evalyn projects."""
    entries = [
        "# Python",
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "*.egg-info/",
        "dist/",
        "build/",
        ".eggs/",
        "",
        "# Virtual environments",
        ".venv/",
        "venv/",
        "env/",
        "",
        "# Testing",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        "",
        "# IDE",
        ".idea/",
        ".vscode/",
        "*.swp",
        "*.swo",
        "",
        "# Git",
        ".git/",
        ".gitignore",
        "",
        "# Docker",
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "",
        "# Data and logs",
        "data/",
        "*.sqlite",
        "*.log",
        "",
        "# Documentation",
        "docs/",
        "*.md",
        "",
    ]
    return "\n".join(entries)


def generate_docker_compose(config: DockerConfig) -> str:
    """Generate docker-compose.yml content for evalyn."""
    lines = [
        "version: '3.8'",
        "",
        "services:",
        "  evalyn:",
        "    build:",
        "      context: .",
        "      dockerfile: Dockerfile",
        "    image: evalyn:latest",
        "    container_name: evalyn",
    ]

    if config.port > 0:
        lines.extend(
            [
                "    ports:",
                f'      - "{config.port}:{config.port}"',
            ]
        )

    lines.extend(
        [
            "    volumes:",
            "      - evalyn-data:/app/data",
            "    environment:",
            "      - EVALYN_ENV=production",
        ]
    )

    if config.include_dev_deps:
        lines.extend(
            [
                "",
                "  evalyn-dev:",
                "    build:",
                "      context: .",
                "      dockerfile: Dockerfile",
                "    image: evalyn:dev",
                "    container_name: evalyn-dev",
                "    volumes:",
                "      - .:/app",
                "    environment:",
                "      - EVALYN_ENV=development",
            ]
        )

    lines.extend(
        [
            "",
            "volumes:",
            "  evalyn-data:",
            "",
        ]
    )

    return "\n".join(lines)


def validate_docker_config(config: DockerConfig) -> list[str]:
    """Return a list of validation errors for the given DockerConfig.

    Returns an empty list if the config is valid.
    """
    errors: list[str] = []

    if not config.base_image:
        errors.append("base_image must not be empty")

    if not config.python_version:
        errors.append("python_version must not be empty")

    # Validate python version format
    parts = config.python_version.split(".")
    if len(parts) < 2:
        errors.append("python_version must be in major.minor format (e.g. '3.11')")
    else:
        try:
            major = int(parts[0])
            minor = int(parts[1])
            if major < 3 or (major == 3 and minor < 8):
                errors.append("python_version must be 3.8 or higher")
        except ValueError:
            errors.append("python_version must contain numeric major.minor")

    if config.port < 0 or config.port > 65535:
        errors.append("port must be between 0 and 65535")

    return errors


def format_docker_instructions(config: DockerConfig) -> str:
    """Return human-readable build and run instructions."""
    lines = [
        "Docker Build and Run Instructions",
        "=" * 35,
        "",
        "1. Build the image:",
        "   docker build -t evalyn:latest .",
        "",
        "2. Run the container:",
    ]

    if config.port > 0:
        lines.append(f"   docker run -p {config.port}:{config.port} evalyn:latest")
    else:
        lines.append("   docker run evalyn:latest")

    lines.extend(
        [
            "",
            "3. Run with docker-compose:",
            "   docker-compose up",
            "",
            "4. Run interactively:",
            "   docker run -it evalyn:latest /bin/bash",
            "",
            f"Base image: {config.base_image}",
            f"Python version: {config.python_version}",
            f"Dev dependencies: {'yes' if config.include_dev_deps else 'no'}",
        ]
    )

    if config.port > 0:
        lines.append(f"Exposed port: {config.port}")

    return "\n".join(lines)
