"""Custom unit builder plugins: user-defined evaluation boundaries via pluggable builders.

Provides a Protocol for unit builders and three built-in implementations
(per-call, per-session, per-tool), plus a registry for managing builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import Span


class UnitBuilder(Protocol):
    """Protocol for pluggable unit builders.

    Each builder takes a list of spans and returns a list of dicts,
    each containing "unit_id", "unit_type", and "spans".
    """

    def build_units(self, spans: list[Span]) -> list[dict[str, Any]]: ...


class PerCallBuilder:
    """Each LLM call span becomes one evaluation unit."""

    def build_units(self, spans: list[Span]) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for s in spans:
            if s.span_type == "llm_call":
                units.append({
                    "unit_id": s.id,
                    "unit_type": "per_call",
                    "spans": [s],
                })
        return units


class PerSessionBuilder:
    """Group spans by session_id attribute into evaluation units."""

    def build_units(self, spans: list[Span]) -> list[dict[str, Any]]:
        sessions: dict[str, list[Span]] = {}
        for s in spans:
            session_id = s.attributes.get("session_id")
            if session_id is not None:
                sessions.setdefault(session_id, []).append(s)
        units: list[dict[str, Any]] = []
        for session_id, session_spans in sessions.items():
            units.append({
                "unit_id": session_id,
                "unit_type": "per_session",
                "spans": session_spans,
            })
        return units


class PerToolBuilder:
    """Each tool_call span becomes one unit, with its parent llm_call span included."""

    def build_units(self, spans: list[Span]) -> list[dict[str, Any]]:
        by_id: dict[str, Span] = {s.id: s for s in spans}
        units: list[dict[str, Any]] = []
        for s in spans:
            if s.span_type == "tool_call":
                unit_spans = [s]
                if s.parent_id and s.parent_id in by_id:
                    parent = by_id[s.parent_id]
                    if parent.span_type == "llm_call":
                        unit_spans.insert(0, parent)
                units.append({
                    "unit_id": s.id,
                    "unit_type": "per_tool",
                    "spans": unit_spans,
                })
        return units


@dataclass
class EvalUnit:
    """A lightweight evaluation unit produced by builders."""

    unit_id: str
    unit_type: str
    span_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "span_ids": list(self.span_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalUnit:
        return cls(
            unit_id=data["unit_id"],
            unit_type=data["unit_type"],
            span_ids=data.get("span_ids", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BuilderRegistry:
    """Registry for named unit builders."""

    builders: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, builder: Any) -> None:
        """Add a builder under the given name."""
        self.builders[name] = builder

    def get(self, name: str) -> Any | None:
        """Get a builder by name, or None if not found."""
        return self.builders.get(name)

    def list_builders(self) -> list[str]:
        """List registered builder names."""
        return list(self.builders.keys())

    def remove(self, name: str) -> bool:
        """Remove a builder by name. Returns True if it existed."""
        if name in self.builders:
            del self.builders[name]
            return True
        return False

    def reset(self) -> None:
        """Clear all builders and re-register defaults."""
        self.builders.clear()
        _register_defaults(self)


def _register_defaults(registry: BuilderRegistry) -> None:
    """Register the three built-in builders."""
    registry.register("per_call", PerCallBuilder())
    registry.register("per_session", PerSessionBuilder())
    registry.register("per_tool", PerToolBuilder())


def get_default_registry() -> BuilderRegistry:
    """Create a registry pre-populated with the 3 built-in builders."""
    registry = BuilderRegistry()
    _register_defaults(registry)
    return registry


def build_units(
    spans: list[Span],
    builder_name: str = "per_call",
    registry: BuilderRegistry | None = None,
) -> list[EvalUnit]:
    """Build evaluation units using a named builder from the registry.

    Returns a list of EvalUnit instances.
    """
    if registry is None:
        registry = get_default_registry()
    builder = registry.get(builder_name)
    if builder is None:
        raise ValueError(f"Unknown builder: {builder_name!r}")
    raw_units = builder.build_units(spans)
    result: list[EvalUnit] = []
    for raw in raw_units:
        span_ids = [s.id for s in raw.get("spans", [])]
        result.append(
            EvalUnit(
                unit_id=raw["unit_id"],
                unit_type=raw["unit_type"],
                span_ids=span_ids,
            )
        )
    return result


def auto_select_builder(spans: list[Span]) -> str:
    """Heuristic: pick best builder name based on span characteristics.

    - If any span has a session_id attribute: "per_session"
    - If any span is a tool_call: "per_tool"
    - Otherwise: "per_call"
    """
    has_session = False
    has_tool = False
    for s in spans:
        if s.attributes.get("session_id") is not None:
            has_session = True
        if s.span_type == "tool_call":
            has_tool = True
    if has_session:
        return "per_session"
    if has_tool:
        return "per_tool"
    return "per_call"
