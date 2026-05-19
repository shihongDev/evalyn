from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..models import Span


@dataclass
class ExtractedAttribute:
    key: str
    value: Any
    source: str = "plugin"

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "source": self.source}


@dataclass
class ExtractionResult:
    attributes: list[ExtractedAttribute] = field(default_factory=list)
    plugin_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "attributes": [a.as_dict() for a in self.attributes],
            "plugin_name": self.plugin_name,
        }


@runtime_checkable
class AttributeExtractor(Protocol):
    def extract(self, span: Span) -> list[ExtractedAttribute]: ...


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------


class ModelExtractor:
    """Extract model name from span attributes."""

    _KEYS = ("model", "model_name", "llm.model")

    def extract(self, span: Span) -> list[ExtractedAttribute]:
        for key in self._KEYS:
            value = span.attributes.get(key)
            if value is not None:
                return [ExtractedAttribute(key="model", value=value, source="ModelExtractor")]
        return []


class TokenExtractor:
    """Extract token counts from span attributes."""

    _INPUT_KEYS = ("input_tokens", "prompt_tokens", "llm.input_tokens")
    _OUTPUT_KEYS = ("output_tokens", "completion_tokens", "llm.output_tokens")
    _TOTAL_KEYS = ("total_tokens", "llm.total_tokens")

    def extract(self, span: Span) -> list[ExtractedAttribute]:
        attrs: list[ExtractedAttribute] = []
        input_tokens = self._find(span, self._INPUT_KEYS)
        output_tokens = self._find(span, self._OUTPUT_KEYS)
        total_tokens = self._find(span, self._TOTAL_KEYS)

        if input_tokens is not None:
            attrs.append(ExtractedAttribute(key="input_tokens", value=input_tokens, source="TokenExtractor"))
        if output_tokens is not None:
            attrs.append(ExtractedAttribute(key="output_tokens", value=output_tokens, source="TokenExtractor"))

        if total_tokens is not None:
            attrs.append(ExtractedAttribute(key="total_tokens", value=total_tokens, source="TokenExtractor"))
        elif input_tokens is not None and output_tokens is not None:
            attrs.append(
                ExtractedAttribute(
                    key="total_tokens",
                    value=input_tokens + output_tokens,
                    source="TokenExtractor",
                )
            )
        return attrs

    @staticmethod
    def _find(span: Span, keys: tuple) -> Any:
        for key in keys:
            value = span.attributes.get(key)
            if value is not None:
                return value
        return None


class CostExtractor:
    """Extract cost from attributes, or estimate from token counts."""

    # Simple per-token pricing estimate (USD) used when no explicit cost is present
    _DEFAULT_INPUT_COST_PER_TOKEN = 0.000003
    _DEFAULT_OUTPUT_COST_PER_TOKEN = 0.000015

    def extract(self, span: Span) -> list[ExtractedAttribute]:
        # Check for explicit cost attribute
        cost = span.attributes.get("cost")
        if cost is not None:
            return [ExtractedAttribute(key="cost", value=cost, source="CostExtractor")]

        # Estimate from tokens
        input_tokens = span.attributes.get("input_tokens") or span.attributes.get("prompt_tokens") or 0
        output_tokens = span.attributes.get("output_tokens") or span.attributes.get("completion_tokens") or 0

        if input_tokens or output_tokens:
            estimated = (
                input_tokens * self._DEFAULT_INPUT_COST_PER_TOKEN
                + output_tokens * self._DEFAULT_OUTPUT_COST_PER_TOKEN
            )
            return [ExtractedAttribute(key="cost", value=estimated, source="CostExtractor")]
        return []


# ---------------------------------------------------------------------------
# Pipeline / application helpers
# ---------------------------------------------------------------------------


def create_extractor_pipeline(
    extractors: list[AttributeExtractor],
) -> Callable[[Span], dict[str, Any]]:
    """Chain multiple extractors. Returns a function that runs all and merges results."""

    def pipeline(span: Span) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for extractor in extractors:
            for attr in extractor.extract(span):
                merged[attr.key] = attr.value
        return merged

    return pipeline


def apply_extractors(span: Span, extractors: list[AttributeExtractor]) -> Span:
    """Apply extractors and add extracted attributes to span. Returns a NEW span."""
    new_span = copy.deepcopy(span)
    for extractor in extractors:
        for attr in extractor.extract(span):
            new_span.attributes[attr.key] = attr.value
    return new_span


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_EXTRACTOR_REGISTRY: dict[str, AttributeExtractor] = {}


def register_extractor(name: str, extractor: AttributeExtractor) -> None:
    """Add an extractor to the global registry."""
    _EXTRACTOR_REGISTRY[name] = extractor


def get_registered_extractors() -> dict[str, AttributeExtractor]:
    """Get all registered extractors."""
    return dict(_EXTRACTOR_REGISTRY)


def reset_extractor_registry() -> None:
    """Clear the global registry (for testing)."""
    _EXTRACTOR_REGISTRY.clear()
