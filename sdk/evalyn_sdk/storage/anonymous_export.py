"""
Anonymous export: strip identifying information when sharing databases.

Provides glob-style field matching, multiple anonymization methods (replace,
hash, remove), and batch processing with configurable rules.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnonymizationRule:
    """A single rule for anonymizing a field."""

    field_pattern: str
    replacement: str = "[REDACTED]"
    method: str = "replace"  # "replace", "hash", or "remove"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field_pattern": self.field_pattern,
            "replacement": self.replacement,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnonymizationRule:
        return cls(
            field_pattern=data.get("field_pattern", ""),
            replacement=data.get("replacement", "[REDACTED]"),
            method=data.get("method", "replace"),
        )


@dataclass
class AnonymousExportConfig:
    """Configuration for anonymous export with a list of rules."""

    rules: List[AnonymizationRule] = field(default_factory=list)
    preserve_structure: bool = True
    salt: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rules": [r.as_dict() for r in self.rules],
            "preserve_structure": self.preserve_structure,
            "salt": self.salt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnonymousExportConfig:
        rules = [
            AnonymizationRule.from_dict(r)
            for r in data.get("rules", [])
        ]
        return cls(
            rules=rules,
            preserve_structure=data.get("preserve_structure", True),
            salt=data.get("salt", ""),
        )


@dataclass
class ExportResult:
    """Summary of an anonymization batch operation."""

    total_records: int = 0
    fields_anonymized: int = 0
    rules_applied: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "fields_anonymized": self.fields_anonymized,
            "rules_applied": self.rules_applied,
        }

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Export Result")
        lines.append("-" * 40)
        lines.append(f"  Total records: {self.total_records}")
        lines.append(f"  Fields anonymized: {self.fields_anonymized}")
        lines.append(f"  Rules applied: {self.rules_applied}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default Rules
# ---------------------------------------------------------------------------


DEFAULT_RULES = [
    AnonymizationRule("*email*", "[EMAIL]", "replace"),
    AnonymizationRule("*name*", "[NAME]", "replace"),
    AnonymizationRule("*user*", "[USER]", "hash"),
    AnonymizationRule("*api_key*", "[KEY]", "remove"),
    AnonymizationRule("*password*", "[REMOVED]", "remove"),
    AnonymizationRule("*token*", "[TOKEN]", "remove"),
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def matches_pattern(field_name: str, pattern: str) -> bool:
    """Glob-style matching where * matches any sequence of characters.

    Case-insensitive. Supports leading *, trailing *, both, or no wildcards.
    """
    if not pattern or not field_name:
        return False

    name_lower = field_name.lower()
    pat_lower = pattern.lower()

    # Split pattern by * and check that all parts appear in order
    parts = pat_lower.split("*")

    # Check anchoring: no leading * means must start with first part
    starts_with_star = pat_lower.startswith("*")
    ends_with_star = pat_lower.endswith("*")

    # Remove empty strings from parts list
    non_empty = [p for p in parts if p]

    if not non_empty:
        # Pattern is all wildcards
        return True

    pos = 0
    for i, part in enumerate(non_empty):
        if i == 0 and not starts_with_star:
            # Must match at the start
            if not name_lower.startswith(part):
                return False
            pos = len(part)
        else:
            idx = name_lower.find(part, pos)
            if idx < 0:
                return False
            pos = idx + len(part)

    if not ends_with_star:
        # Must match at the end
        if not name_lower.endswith(non_empty[-1]):
            return False

    return True


def apply_rule(value: str, rule: AnonymizationRule, salt: str = "") -> str:
    """Apply an anonymization rule to a string value.

    - "replace": return the rule's replacement text
    - "hash": return a SHA-256 hash prefix (first 16 hex chars)
    - "remove": return empty string
    """
    if rule.method == "replace":
        return rule.replacement
    elif rule.method == "hash":
        to_hash = salt + value
        digest = hashlib.sha256(to_hash.encode("utf-8")).hexdigest()
        return digest[:16]
    elif rule.method == "remove":
        return ""
    return rule.replacement


def anonymize_record(
    record: Dict[str, Any], config: AnonymousExportConfig
) -> Tuple[Dict[str, Any], int]:
    """Anonymize one record by applying matching rules to its fields.

    Returns (new_record, fields_changed). Non-matching fields are preserved
    as-is. Only string values are anonymized; non-string values are kept.
    """
    new_record: Dict[str, Any] = {}
    fields_changed = 0

    for key, value in record.items():
        matched = False
        for rule in config.rules:
            if matches_pattern(key, rule.field_pattern):
                if isinstance(value, str):
                    new_record[key] = apply_rule(value, rule, salt=config.salt)
                else:
                    new_record[key] = rule.replacement if rule.method == "replace" else ""
                fields_changed += 1
                matched = True
                break

        if not matched:
            new_record[key] = value

    return new_record, fields_changed


def anonymize_batch(
    records: List[Dict[str, Any]],
    config: Optional[AnonymousExportConfig] = None,
) -> Tuple[List[Dict[str, Any]], ExportResult]:
    """Anonymize a batch of records.

    Uses the default config (with DEFAULT_RULES) if none is provided.
    Returns (anonymized_records, export_result).
    """
    if config is None:
        config = get_default_config()

    anonymized: List[Dict[str, Any]] = []
    total_fields = 0
    rules_used: set[str] = set()

    for record in records:
        new_record, changed = anonymize_record(record, config)
        anonymized.append(new_record)
        total_fields += changed
        # Track which rules were used
        for key in record:
            for rule in config.rules:
                if matches_pattern(key, rule.field_pattern):
                    rules_used.add(rule.field_pattern)
                    break

    result = ExportResult(
        total_records=len(records),
        fields_anonymized=total_fields,
        rules_applied=len(rules_used),
    )

    return anonymized, result


def get_default_config() -> AnonymousExportConfig:
    """Return a config populated with DEFAULT_RULES."""
    return AnonymousExportConfig(
        rules=list(DEFAULT_RULES),
        preserve_structure=True,
        salt="",
    )
