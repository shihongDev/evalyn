from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import Span


@dataclass
class AnonymizationConfig:
    """Configuration for trace anonymization."""

    anonymize_inputs: bool = True
    anonymize_outputs: bool = True
    preserve_structure: bool = True  # keep dict keys, replace values
    salt: str = ""  # for deterministic hashing

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anonymize_inputs": self.anonymize_inputs,
            "anonymize_outputs": self.anonymize_outputs,
            "preserve_structure": self.preserve_structure,
            "salt": self.salt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnonymizationConfig:
        return cls(
            anonymize_inputs=data.get("anonymize_inputs", True),
            anonymize_outputs=data.get("anonymize_outputs", True),
            preserve_structure=data.get("preserve_structure", True),
            salt=data.get("salt", ""),
        )


@dataclass
class AnonymizationResult:
    """Result summary from an anonymization operation."""

    spans_processed: int
    fields_anonymized: int
    original_span_ids: List[str] = field(default_factory=list)
    anonymized_span_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spans_processed": self.spans_processed,
            "fields_anonymized": self.fields_anonymized,
            "original_span_ids": list(self.original_span_ids),
            "anonymized_span_ids": list(self.anonymized_span_ids),
        }


def anonymize_text(text: str, salt: str = "") -> str:
    """Replace text with a deterministic synthetic equivalent.

    Uses SHA-256 hash of (salt + text) to produce a token of similar length.
    Format: [anon-<hash_prefix_8chars>-<length>chars]
    """
    if not text:
        return text
    digest = hashlib.sha256((salt + text).encode()).hexdigest()
    return f"[anon-{digest[:8]}-{len(text)}chars]"


def _anonymize_value(value: Any, salt: str, preserve_structure: bool) -> Tuple[Any, int]:
    """Anonymize a single value, returning (new_value, fields_anonymized)."""
    if isinstance(value, str):
        return anonymize_text(value, salt), 1
    if isinstance(value, dict):
        if preserve_structure:
            new_dict: Dict[str, Any] = {}
            count = 0
            for k, v in value.items():
                anon_v, c = _anonymize_value(v, salt, preserve_structure)
                new_dict[k] = anon_v
                count += c
            return new_dict, count
        # Not preserving structure: replace the whole dict as a string
        return anonymize_text(str(value), salt), 1
    if isinstance(value, list):
        new_list: List[Any] = []
        count = 0
        for item in value:
            anon_item, c = _anonymize_value(item, salt, preserve_structure)
            new_list.append(anon_item)
            count += c
        return new_list, count
    # Non-string primitives (int, float, bool, None) are preserved
    return value, 0


def _hash_id(original_id: str, salt: str) -> str:
    """Generate a new span ID from the original via hashing."""
    digest = hashlib.sha256((salt + original_id).encode()).hexdigest()
    return digest[:32]


_INPUT_KEYS = {"input", "inputs"}
_OUTPUT_KEYS = {"output", "outputs"}


def anonymize_span(span: Span, config: AnonymizationConfig) -> Span:
    """Return a NEW span with anonymized content. Never mutates the original."""
    new_span = copy.deepcopy(span)
    new_span.id = _hash_id(span.id, config.salt)
    if span.parent_id is not None:
        new_span.parent_id = _hash_id(span.parent_id, config.salt)

    fields_anonymized = 0
    new_attrs: Dict[str, Any] = {}
    for key, value in new_span.attributes.items():
        is_input = key in _INPUT_KEYS
        is_output = key in _OUTPUT_KEYS
        should_anonymize = (is_input and config.anonymize_inputs) or (
            is_output and config.anonymize_outputs
        )
        if should_anonymize:
            anon_value, count = _anonymize_value(
                value, config.salt, config.preserve_structure
            )
            new_attrs[key] = anon_value
            fields_anonymized += count
        else:
            new_attrs[key] = value
    new_span.attributes = new_attrs
    # Store count in a transient way (not on the span itself)
    # We return it via a wrapper; callers that need the count use anonymize_spans
    new_span._anon_field_count = fields_anonymized  # type: ignore[attr-defined]
    return new_span


def anonymize_spans(
    spans: List[Span], config: AnonymizationConfig
) -> Tuple[List[Span], AnonymizationResult]:
    """Batch anonymization. Maintains parent-child relationships by remapping IDs."""
    anonymized: List[Span] = []
    original_ids: List[str] = []
    anonymized_ids: List[str] = []
    total_fields = 0

    for s in spans:
        new_span = anonymize_span(s, config)
        count = getattr(new_span, "_anon_field_count", 0)
        total_fields += count
        # Clean up transient attribute
        if hasattr(new_span, "_anon_field_count"):
            del new_span._anon_field_count
        anonymized.append(new_span)
        original_ids.append(s.id)
        anonymized_ids.append(new_span.id)

    result = AnonymizationResult(
        spans_processed=len(spans),
        fields_anonymized=total_fields,
        original_span_ids=original_ids,
        anonymized_span_ids=anonymized_ids,
    )
    return anonymized, result


def export_anonymized(
    spans: List[Span], config: Optional[AnonymizationConfig] = None
) -> List[Dict[str, Any]]:
    """Anonymize and export as list of dicts."""
    if config is None:
        config = AnonymizationConfig()
    anonymized, _ = anonymize_spans(spans, config)
    return [s.as_dict() for s in anonymized]
