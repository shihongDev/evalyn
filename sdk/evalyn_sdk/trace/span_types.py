from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpanTypeInfo:
    """Metadata for a registered span type."""

    name: str
    icon: str = ""
    color: str = ""
    description: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "icon": self.icon,
            "color": self.color,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SpanTypeInfo:
        return cls(
            name=data["name"],
            icon=data.get("icon", ""),
            color=data.get("color", ""),
            description=data.get("description", ""),
        )


_BUILTIN_TYPES: dict[str, SpanTypeInfo] = {
    "llm_call": SpanTypeInfo(
        name="llm_call", description="LLM API call",
    ),
    "tool_call": SpanTypeInfo(
        name="tool_call", description="Tool/function call",
    ),
    "retrieval": SpanTypeInfo(
        name="retrieval", description="RAG retrieval",
    ),
    "node": SpanTypeInfo(
        name="node", description="Graph node execution",
    ),
    "agent": SpanTypeInfo(
        name="agent", description="Agent execution",
    ),
    "custom": SpanTypeInfo(
        name="custom", description="User-defined span",
    ),
    "scorer": SpanTypeInfo(
        name="scorer", description="Metric evaluation",
    ),
}

_REGISTRY: dict[str, SpanTypeInfo] = dict(_BUILTIN_TYPES)


def register_span_type(
    name: str,
    icon: str = "",
    color: str = "",
    description: str = "",
) -> SpanTypeInfo:
    """Register a custom span type.

    Raises ValueError if name is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Span type already registered: {name!r}")
    info = SpanTypeInfo(name=name, icon=icon, color=color, description=description)
    _REGISTRY[name] = info
    return info


def get_span_type(name: str) -> SpanTypeInfo | None:
    """Look up a span type by name. Returns None if not found."""
    return _REGISTRY.get(name)


def list_span_types() -> list[SpanTypeInfo]:
    """Return all registered span types sorted by name."""
    return sorted(_REGISTRY.values(), key=lambda t: t.name)


def is_valid_span_type(name: str) -> bool:
    """Check whether a span type name is registered."""
    return name in _REGISTRY


def unregister_span_type(name: str) -> bool:
    """Remove a custom span type. Built-in types cannot be removed.

    Returns True if the type was removed, False otherwise.
    """
    if name in _BUILTIN_TYPES:
        return False
    if name in _REGISTRY:
        del _REGISTRY[name]
        return True
    return False


def reset_registry() -> None:
    """Reset the registry to built-in types only (useful for testing)."""
    _REGISTRY.clear()
    _REGISTRY.update(_BUILTIN_TYPES)
