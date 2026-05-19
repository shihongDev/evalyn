"""Sandboxed agent evaluation framework.

Safe execution environment configuration and management for agent evals.
Pure Python - no actual Docker calls, just command building and validation.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class SandboxConfig:
    """Configuration for a sandbox execution environment."""

    timeout_seconds: int = 30
    max_memory_mb: int = 256
    network_enabled: bool = False
    image: str = "python:3.11-slim"
    working_dir: str = "/workspace"

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_memory_mb": self.max_memory_mb,
            "network_enabled": self.network_enabled,
            "image": self.image,
            "working_dir": self.working_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxConfig:
        return cls(
            timeout_seconds=data.get("timeout_seconds", 30),
            max_memory_mb=data.get("max_memory_mb", 256),
            network_enabled=data.get("network_enabled", False),
            image=data.get("image", "python:3.11-slim"),
            working_dir=data.get("working_dir", "/workspace"),
        )


@dataclass
class SandboxResult:
    """Result from a sandbox execution."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxResult:
        return cls(
            exit_code=data["exit_code"],
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            timed_out=data.get("timed_out", False),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


@dataclass
class SandboxSpec:
    """Specification for a sandbox instance."""

    sandbox_id: str
    config: SandboxConfig
    status: str = "ready"  # ready / running / completed / failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "config": self.config.as_dict(),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxSpec:
        return cls(
            sandbox_id=data["sandbox_id"],
            config=SandboxConfig.from_dict(data.get("config", {})),
            status=data.get("status", "ready"),
        )


def build_docker_command(code: str, config: SandboxConfig) -> list[str]:
    """Build a docker run command list (does not execute it).

    Includes timeout, memory limit, network flag, and working directory.
    """
    cmd: list[str] = ["docker", "run", "--rm"]

    # Memory limit
    cmd.extend(["--memory", f"{config.max_memory_mb}m"])

    # Network
    if not config.network_enabled:
        cmd.extend(["--network", "none"])

    # Working directory
    cmd.extend(["--workdir", config.working_dir])

    # Image
    cmd.append(config.image)

    # Timeout wrapper + code execution
    cmd.extend([
        "timeout",
        str(config.timeout_seconds),
        "python",
        "-c",
        code,
    ])

    return cmd


def validate_sandbox_config(config: SandboxConfig) -> list[str]:
    """Return a list of validation errors for the given config.

    Checks for unreasonable timeout, memory, and image values.
    """
    errors: list[str] = []

    if config.timeout_seconds <= 0:
        errors.append("timeout_seconds must be positive")
    elif config.timeout_seconds > 3600:
        errors.append("timeout_seconds exceeds maximum of 3600")

    if config.max_memory_mb < 16:
        errors.append("max_memory_mb must be at least 16")
    elif config.max_memory_mb > 65536:
        errors.append("max_memory_mb exceeds maximum of 65536")

    if not config.image:
        errors.append("image must not be empty")

    if not config.working_dir:
        errors.append("working_dir must not be empty")

    return errors


def estimate_sandbox_cost(
    n_items: int, avg_duration: float, config: SandboxConfig
) -> dict[str, Any]:
    """Estimate total time and resource usage for sandboxed evaluation.

    Returns a dict with estimated totals.
    """
    total_seconds = n_items * avg_duration
    total_memory_mb_seconds = total_seconds * config.max_memory_mb
    # Estimate wall-clock time assuming serial execution
    wall_clock_seconds = total_seconds
    # Cap each item at config timeout
    capped_duration = min(avg_duration, config.timeout_seconds)
    capped_total = n_items * capped_duration

    return {
        "n_items": n_items,
        "avg_duration": avg_duration,
        "total_seconds": total_seconds,
        "capped_total_seconds": capped_total,
        "wall_clock_seconds": wall_clock_seconds,
        "total_memory_mb_seconds": total_memory_mb_seconds,
        "timeout_seconds": config.timeout_seconds,
        "max_memory_mb": config.max_memory_mb,
    }


def create_sandbox_spec(config: SandboxConfig | None = None) -> SandboxSpec:
    """Factory to create a SandboxSpec with a unique UUID."""
    if config is None:
        config = SandboxConfig()
    return SandboxSpec(
        sandbox_id=str(uuid.uuid4()),
        config=config,
        status="ready",
    )


def format_sandbox_plan(specs: list[SandboxSpec]) -> str:
    """Format a human-readable plan showing sandbox configs."""
    if not specs:
        return "No sandboxes configured."

    lines: list[str] = [f"Sandbox Plan ({len(specs)} sandbox(es)):"]
    lines.append("")
    for i, spec in enumerate(specs, 1):
        c = spec.config
        net = "enabled" if c.network_enabled else "disabled"
        lines.append(f"  [{i}] {spec.sandbox_id}")
        lines.append(f"      Status:  {spec.status}")
        lines.append(f"      Image:   {c.image}")
        lines.append(f"      Timeout: {c.timeout_seconds}s")
        lines.append(f"      Memory:  {c.max_memory_mb}MB")
        lines.append(f"      Network: {net}")
        lines.append(f"      Workdir: {c.working_dir}")
        lines.append("")

    return "\n".join(lines)


def check_docker_available() -> bool:
    """Check if the docker command is available on the system PATH."""
    return shutil.which("docker") is not None
