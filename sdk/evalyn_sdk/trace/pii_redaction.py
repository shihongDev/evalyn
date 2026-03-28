"""PII redaction for trace spans - scrub sensitive data before storage."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..models import Span

# Built-in PII patterns (compiled at module level)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
IP_ADDRESS_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


@dataclass
class RedactionPattern:
    """A named regex pattern with its replacement template."""

    name: str
    pattern: "re.Pattern[str]"
    replacement: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pattern": self.pattern.pattern,
            "replacement": self.replacement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RedactionPattern:
        return cls(
            name=data["name"],
            pattern=re.compile(data["pattern"]),
            replacement=data["replacement"],
        )


@dataclass
class RedactionConfig:
    """Configuration for PII redaction."""

    strategy: str  # "mask", "hash", or "remove"
    patterns: List[RedactionPattern] = field(default_factory=list)
    custom_patterns: List[RedactionPattern] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "patterns": [p.as_dict() for p in self.patterns],
            "custom_patterns": [p.as_dict() for p in self.custom_patterns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RedactionConfig:
        return cls(
            strategy=data["strategy"],
            patterns=[
                RedactionPattern.from_dict(p)
                for p in data.get("patterns", [])
            ],
            custom_patterns=[
                RedactionPattern.from_dict(p)
                for p in data.get("custom_patterns", [])
            ],
        )


@dataclass
class RedactionResult:
    """Summary of redactions applied."""

    original_length: int
    redacted_length: int
    redactions_count: int
    redacted_fields: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_length": self.original_length,
            "redacted_length": self.redacted_length,
            "redactions_count": self.redactions_count,
            "redacted_fields": list(self.redacted_fields),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RedactionResult:
        return cls(
            original_length=data["original_length"],
            redacted_length=data["redacted_length"],
            redactions_count=data["redactions_count"],
            redacted_fields=data.get("redacted_fields", []),
        )


def _apply_pattern(
    text: str, pat: RedactionPattern, strategy: str
) -> Tuple[str, int]:
    """Apply a single pattern to text. Returns (new_text, match_count)."""
    count = 0

    def _replacer(match: re.Match) -> str:
        nonlocal count
        count += 1
        matched = match.group(0)
        if strategy == "mask":
            return f"[REDACTED:{pat.name}]"
        elif strategy == "hash":
            h = hashlib.sha256(matched.encode()).hexdigest()[:8]
            return h
        else:  # "remove"
            return ""

    result = pat.pattern.sub(_replacer, text)
    return result, count


def redact_text(text: str, config: RedactionConfig) -> str:
    """Apply all patterns from config to text, using the configured strategy."""
    all_patterns = list(config.patterns) + list(config.custom_patterns)
    result = text
    for pat in all_patterns:
        result, _ = _apply_pattern(result, pat, config.strategy)
    return result


def _redact_value(
    value: Any, config: RedactionConfig
) -> Tuple[Any, int, int, int]:
    """Redact a single value. Returns (redacted, orig_len, new_len, count)."""
    if isinstance(value, str):
        orig_len = len(value)
        all_patterns = list(config.patterns) + list(config.custom_patterns)
        result = value
        total_count = 0
        for pat in all_patterns:
            result, c = _apply_pattern(result, pat, config.strategy)
            total_count += c
        return result, orig_len, len(result), total_count
    elif isinstance(value, dict):
        new_dict = {}
        total_orig = 0
        total_new = 0
        total_count = 0
        for k, v in value.items():
            rv, ol, nl, c = _redact_value(v, config)
            new_dict[k] = rv
            total_orig += ol
            total_new += nl
            total_count += c
        return new_dict, total_orig, total_new, total_count
    elif isinstance(value, list):
        new_list = []
        total_orig = 0
        total_new = 0
        total_count = 0
        for item in value:
            rv, ol, nl, c = _redact_value(item, config)
            new_list.append(rv)
            total_orig += ol
            total_new += nl
            total_count += c
        return new_list, total_orig, total_new, total_count
    else:
        return value, 0, 0, 0


_REDACT_KEYS = {"input", "output", "inputs", "outputs"}


def redact_span(
    span: Span, config: RedactionConfig
) -> Tuple[Span, RedactionResult]:
    """Redact PII from span attributes. Returns a NEW span and a result summary."""
    new_span = copy.deepcopy(span)
    total_orig = 0
    total_new = 0
    total_count = 0
    redacted_fields: List[str] = []

    for key in list(new_span.attributes.keys()):
        if key not in _REDACT_KEYS:
            continue
        val = new_span.attributes[key]
        redacted_val, ol, nl, c = _redact_value(val, config)
        new_span.attributes[key] = redacted_val
        total_orig += ol
        total_new += nl
        total_count += c
        if c > 0:
            redacted_fields.append(key)

    result = RedactionResult(
        original_length=total_orig,
        redacted_length=total_new,
        redactions_count=total_count,
        redacted_fields=redacted_fields,
    )
    return new_span, result


def redact_spans(
    spans: List[Span], config: RedactionConfig
) -> Tuple[List[Span], RedactionResult]:
    """Batch redaction across multiple spans."""
    all_spans: List[Span] = []
    total_orig = 0
    total_new = 0
    total_count = 0
    all_fields: List[str] = []

    for s in spans:
        new_s, r = redact_span(s, config)
        all_spans.append(new_s)
        total_orig += r.original_length
        total_new += r.redacted_length
        total_count += r.redactions_count
        all_fields.extend(r.redacted_fields)

    combined = RedactionResult(
        original_length=total_orig,
        redacted_length=total_new,
        redactions_count=total_count,
        redacted_fields=all_fields,
    )
    return all_spans, combined


def default_config(strategy: str = "mask") -> RedactionConfig:
    """Return a config with all built-in PII patterns."""
    return RedactionConfig(
        strategy=strategy,
        patterns=[
            RedactionPattern("EMAIL", EMAIL_PATTERN, "[REDACTED:EMAIL]"),
            RedactionPattern("PHONE", PHONE_PATTERN, "[REDACTED:PHONE]"),
            RedactionPattern("SSN", SSN_PATTERN, "[REDACTED:SSN]"),
            RedactionPattern(
                "CREDIT_CARD", CREDIT_CARD_PATTERN, "[REDACTED:CREDIT_CARD]"
            ),
            RedactionPattern(
                "IP_ADDRESS", IP_ADDRESS_PATTERN, "[REDACTED:IP_ADDRESS]"
            ),
        ],
        custom_patterns=[],
    )
