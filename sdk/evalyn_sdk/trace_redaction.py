"""
Trace content redaction policies.

Configurable rules for what gets stored in trace payloads. Supports
regex-based PII removal, truncation, and audit reporting.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class RedactionRule:
    """A single regex-based redaction rule."""

    rule_id: str
    pattern: str  # regex pattern
    replacement: str = "[REDACTED]"
    description: str = ""
    applies_to: List[str] = field(default_factory=lambda: ["all"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "pattern": self.pattern,
            "replacement": self.replacement,
            "description": self.description,
            "applies_to": list(self.applies_to),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RedactionRule:
        return cls(
            rule_id=data["rule_id"],
            pattern=data["pattern"],
            replacement=data.get("replacement", "[REDACTED]"),
            description=data.get("description", ""),
            applies_to=list(data.get("applies_to", ["all"])),
        )


@dataclass
class RedactionPolicy:
    """A named collection of redaction rules with truncation settings."""

    policy_id: str
    name: str
    rules: List[RedactionRule] = field(default_factory=list)
    truncate_after: int = 0  # 0 means no truncation
    keep_first_n: int = 0
    keep_last_n: int = 0
    mode: str = "strict"  # "strict" or "relaxed"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "rules": [r.as_dict() for r in self.rules],
            "truncate_after": self.truncate_after,
            "keep_first_n": self.keep_first_n,
            "keep_last_n": self.keep_last_n,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RedactionPolicy:
        return cls(
            policy_id=data["policy_id"],
            name=data["name"],
            rules=[RedactionRule.from_dict(r) for r in data.get("rules", [])],
            truncate_after=data.get("truncate_after", 0),
            keep_first_n=data.get("keep_first_n", 0),
            keep_last_n=data.get("keep_last_n", 0),
            mode=data.get("mode", "strict"),
        )


@dataclass
class RedactionResult:
    """Outcome of applying redaction to a single text field."""

    original_length: int
    redacted_length: int
    rules_applied: List[str] = field(default_factory=list)
    truncated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_length": self.original_length,
            "redacted_length": self.redacted_length,
            "rules_applied": list(self.rules_applied),
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RedactionResult:
        return cls(
            original_length=data.get("original_length", 0),
            redacted_length=data.get("redacted_length", 0),
            rules_applied=list(data.get("rules_applied", [])),
            truncated=data.get("truncated", False),
        )


@dataclass
class RedactionAuditEntry:
    """An audit log entry for one redaction operation."""

    trace_id: str
    field_name: str
    result: RedactionResult

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "field_name": self.field_name,
            "result": self.result.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RedactionAuditEntry:
        return cls(
            trace_id=data["trace_id"],
            field_name=data["field_name"],
            result=RedactionResult.from_dict(data["result"]),
        )


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def apply_redaction(
    text: str,
    policy: RedactionPolicy,
    field_name: str = "all",
) -> Tuple[str, RedactionResult]:
    """Apply all matching rules from a policy to the given text.

    A rule matches if ``field_name`` is in the rule's ``applies_to`` list
    or if the rule applies to ``"all"``.

    Returns the redacted text and a result summary.
    """
    original_length = len(text)
    rules_applied: List[str] = []
    result_text = text

    for rule in policy.rules:
        if "all" not in rule.applies_to and field_name not in rule.applies_to:
            continue
        try:
            compiled = re.compile(rule.pattern)
        except re.error:
            continue
        new_text = compiled.sub(rule.replacement, result_text)
        if new_text != result_text:
            rules_applied.append(rule.rule_id)
            result_text = new_text

    truncated = False
    if policy.truncate_after > 0 and len(result_text) > policy.truncate_after:
        result_text = apply_truncation(
            result_text,
            policy.truncate_after,
            keep_first=policy.keep_first_n,
            keep_last=policy.keep_last_n,
        )
        truncated = True

    return result_text, RedactionResult(
        original_length=original_length,
        redacted_length=len(result_text),
        rules_applied=rules_applied,
        truncated=truncated,
    )


def apply_truncation(
    text: str,
    max_len: int,
    keep_first: int = 0,
    keep_last: int = 0,
) -> str:
    """Truncate text to *max_len* characters with a marker.

    When *keep_first* or *keep_last* are set, those character counts are
    preserved from the start/end of the original text around the marker.
    """
    if len(text) <= max_len:
        return text

    marker = "... [truncated] ..."

    if keep_first > 0 or keep_last > 0:
        first_part = text[:keep_first] if keep_first > 0 else ""
        last_part = text[-keep_last:] if keep_last > 0 else ""
        result = first_part + marker + last_part
        # If the assembled result is longer than max_len, trim while
        # preserving both first and last portions as much as possible
        if len(result) > max_len:
            available = max_len - len(marker)
            if available >= keep_first + keep_last and keep_first + keep_last > 0:
                first_part = text[:keep_first]
                last_part = text[-keep_last:]
                return first_part + marker + last_part
            elif available > 0 and keep_last > 0:
                # Prioritize showing some of both ends
                half = available // 2
                first_n = max(1, half)
                last_n = max(1, available - first_n)
                return text[:first_n] + marker + text[-last_n:]
            elif available > 0:
                return text[:available] + marker
            return text[:max_len]
        return result

    # Simple truncation
    if max_len <= len(marker):
        return text[:max_len]
    return text[: max_len - len(marker)] + marker


def build_default_policy(mode: str = "strict") -> RedactionPolicy:
    """Build a default redaction policy.

    strict: redact emails, phone numbers, SSN patterns, API keys.
    relaxed: only API keys.
    """
    api_key_rule = RedactionRule(
        rule_id="api_key",
        pattern=r"(?:sk|pk|api|key|token)[-_][A-Za-z0-9_\-]{20,}",
        replacement="[API_KEY_REDACTED]",
        description="API keys and tokens",
        applies_to=["all"],
    )

    if mode == "relaxed":
        return RedactionPolicy(
            policy_id="default_relaxed",
            name="Default Relaxed",
            rules=[api_key_rule],
            mode="relaxed",
        )

    email_rule = RedactionRule(
        rule_id="email",
        pattern=r"[a-zA-Z0-9_.+-]{1,254}@[a-zA-Z0-9-]{1,63}\.[a-zA-Z0-9-.]{1,254}",
        replacement="[EMAIL_REDACTED]",
        description="Email addresses",
        applies_to=["all"],
    )
    phone_rule = RedactionRule(
        rule_id="phone",
        pattern=r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        replacement="[PHONE_REDACTED]",
        description="US phone numbers",
        applies_to=["all"],
    )
    ssn_rule = RedactionRule(
        rule_id="ssn",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        replacement="[SSN_REDACTED]",
        description="Social Security numbers",
        applies_to=["all"],
    )

    return RedactionPolicy(
        policy_id="default_strict",
        name="Default Strict",
        rules=[email_rule, phone_rule, ssn_rule, api_key_rule],
        mode="strict",
    )


def generate_redaction_audit(entries: List[RedactionAuditEntry]) -> Dict[str, Any]:
    """Produce summary statistics from a list of audit entries.

    Returns a dict with: total_traces, total_redactions, bytes_saved,
    most_triggered_rule.
    """
    trace_ids = set()
    total_redactions = 0
    bytes_saved = 0
    rule_counts: Dict[str, int] = {}

    for entry in entries:
        trace_ids.add(entry.trace_id)
        total_redactions += len(entry.result.rules_applied)
        bytes_saved += entry.result.original_length - entry.result.redacted_length
        for rule_id in entry.result.rules_applied:
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

    most_triggered = ""
    if rule_counts:
        most_triggered = max(rule_counts, key=lambda k: rule_counts[k])

    return {
        "total_traces": len(trace_ids),
        "total_redactions": total_redactions,
        "bytes_saved": bytes_saved,
        "most_triggered_rule": most_triggered,
        "rule_counts": dict(rule_counts),
    }


def format_redaction_audit(audit: Dict[str, Any]) -> str:
    """Format an audit summary dict as a human-readable report."""
    lines: List[str] = []
    lines.append("Redaction Audit Report")
    lines.append("=" * 40)
    lines.append(f"Total traces processed: {audit.get('total_traces', 0)}")
    lines.append(f"Total redactions applied: {audit.get('total_redactions', 0)}")
    lines.append(f"Bytes saved: {audit.get('bytes_saved', 0)}")
    lines.append(f"Most triggered rule: {audit.get('most_triggered_rule', 'n/a')}")

    rule_counts = audit.get("rule_counts", {})
    if rule_counts:
        lines.append("")
        lines.append("Rule Breakdown")
        lines.append("-" * 40)
        for rule_id, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {rule_id}: {count}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built Policies
# ---------------------------------------------------------------------------

STRICT_POLICY: RedactionPolicy = build_default_policy("strict")
RELAXED_POLICY: RedactionPolicy = build_default_policy("relaxed")
