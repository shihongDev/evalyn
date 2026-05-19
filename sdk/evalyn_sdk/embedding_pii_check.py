"""
Embedding PII safety check: detect whether stored embeddings could leak PII
via inversion attacks.

Scans text for personally identifiable information patterns and assesses
the risk when those texts also have associated embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PIIPattern:
    """A named regex pattern for detecting a category of PII."""

    pattern_name: str
    regex: str
    description: str
    severity: str  # "high" / "medium" / "low"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "regex": self.regex,
            "description": self.description,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PIIPattern:
        return cls(
            pattern_name=data["pattern_name"],
            regex=data["regex"],
            description=data["description"],
            severity=data["severity"],
        )


@dataclass
class PIIDetectionResult:
    """Result of scanning a single text item for PII."""

    text_id: str
    pii_found: List[str]
    pii_count: int
    has_embedding: bool
    risk_level: str  # "critical" / "high" / "medium" / "low" / "none"
    recommendations: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text_id": self.text_id,
            "pii_found": list(self.pii_found),
            "pii_count": self.pii_count,
            "has_embedding": self.has_embedding,
            "risk_level": self.risk_level,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PIIDetectionResult:
        return cls(
            text_id=data["text_id"],
            pii_found=list(data.get("pii_found", [])),
            pii_count=data.get("pii_count", 0),
            has_embedding=data.get("has_embedding", False),
            risk_level=data.get("risk_level", "none"),
            recommendations=list(data.get("recommendations", [])),
        )


@dataclass
class PIISafetyReport:
    """Aggregated PII safety report for a dataset."""

    results: List[PIIDetectionResult]
    total_items: int
    items_with_pii: int
    items_at_risk: int
    summary: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_items": self.total_items,
            "items_with_pii": self.items_with_pii,
            "items_at_risk": self.items_at_risk,
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PIISafetyReport:
        return cls(
            results=[
                PIIDetectionResult.from_dict(r) for r in data.get("results", [])
            ],
            total_items=data.get("total_items", 0),
            items_with_pii=data.get("items_with_pii", 0),
            items_at_risk=data.get("items_at_risk", 0),
            summary=data.get("summary", {}),
        )


# ---------------------------------------------------------------------------
# Default PII Patterns
# ---------------------------------------------------------------------------

DEFAULT_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        pattern_name="email",
        regex=r"[a-zA-Z0-9._%+\-]{1,64}@[a-zA-Z0-9.\-]{1,253}\.[a-zA-Z]{2,10}",
        description="Email address",
        severity="high",
    ),
    PIIPattern(
        pattern_name="phone",
        regex=(
            r"(?:\+?1[\s\-.]?)?"
            r"(?:\(?\d{3}\)?[\s\-.]?)"
            r"\d{3}[\s\-.]?\d{4}"
        ),
        description="US phone number",
        severity="high",
    ),
    PIIPattern(
        pattern_name="ssn",
        regex=r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b",
        description="US Social Security Number",
        severity="high",
    ),
    PIIPattern(
        pattern_name="credit_card",
        regex=r"\b(?:\d[\s\-]?){13,19}\b",
        description="Credit card number",
        severity="high",
    ),
    PIIPattern(
        pattern_name="ip_address",
        regex=(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d{1,2})\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d{1,2})\b"
        ),
        description="IPv4 address",
        severity="medium",
    ),
    PIIPattern(
        pattern_name="date_of_birth",
        regex=(
            r"\b(?:"
            r"(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}"
            r"|(?:19|20)\d{2}[/\-](?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])"
            r")\b"
        ),
        description="Date of birth (MM/DD/YYYY or YYYY-MM-DD)",
        severity="medium",
    ),
]


# ---------------------------------------------------------------------------
# Detection Functions
# ---------------------------------------------------------------------------


def detect_pii(
    text: str,
    patterns: Optional[List[PIIPattern]] = None,
) -> List[str]:
    """Return list of pattern names that matched in *text*."""
    if patterns is None:
        patterns = DEFAULT_PII_PATTERNS
    found: List[str] = []
    for pat in patterns:
        if re.search(pat.regex, text):
            found.append(pat.pattern_name)
    return found


def _compute_risk_level(pii_found: List[str], has_embedding: bool,
                        patterns: List[PIIPattern]) -> str:
    """Determine risk level from PII findings and embedding presence."""
    if not pii_found:
        return "none"

    severity_map: Dict[str, str] = {p.pattern_name: p.severity for p in patterns}
    severities = [severity_map.get(name, "low") for name in pii_found]

    has_high = "high" in severities
    has_medium = "medium" in severities

    if has_embedding:
        if has_high:
            return "critical"
        if has_medium:
            return "high"
        return "medium"
    else:
        if has_high:
            return "high"
        if has_medium:
            return "medium"
        return "low"


def _build_recommendations(risk_level: str, pii_found: List[str],
                           has_embedding: bool) -> List[str]:
    """Build contextual recommendations for a single item."""
    recs: List[str] = []

    if risk_level == "none":
        return recs

    if has_embedding and pii_found:
        recs.append(
            "Remove or redact PII before generating embeddings to prevent "
            "potential inversion attacks."
        )

    if "ssn" in pii_found or "credit_card" in pii_found:
        recs.append(
            "Highly sensitive data detected - consider removing this item "
            "from the dataset entirely."
        )

    if "email" in pii_found or "phone" in pii_found:
        recs.append(
            "Replace identifiable contact information with synthetic "
            "placeholders."
        )

    if "ip_address" in pii_found:
        recs.append("Anonymize IP addresses before storing embeddings.")

    if "date_of_birth" in pii_found:
        recs.append(
            "Generalize dates of birth to age ranges to reduce "
            "re-identification risk."
        )

    if has_embedding and risk_level in ("critical", "high"):
        recs.append(
            "Consider using differential-privacy noise injection on "
            "embeddings."
        )

    return recs


def assess_embedding_risk(
    text: str,
    has_embedding: bool,
    text_id: str = "",
    patterns: Optional[List[PIIPattern]] = None,
) -> PIIDetectionResult:
    """Full risk assessment for a single text item."""
    if patterns is None:
        patterns = DEFAULT_PII_PATTERNS

    pii_found = detect_pii(text, patterns)
    risk_level = _compute_risk_level(pii_found, has_embedding, patterns)
    recommendations = _build_recommendations(risk_level, pii_found, has_embedding)

    return PIIDetectionResult(
        text_id=text_id,
        pii_found=pii_found,
        pii_count=len(pii_found),
        has_embedding=has_embedding,
        risk_level=risk_level,
        recommendations=recommendations,
    )


def check_dataset_pii_safety(
    items: List[Dict[str, Any]],
    text_field: str = "input",
    embedding_field: str = "embedding",
) -> PIISafetyReport:
    """Check an entire dataset for PII-plus-embedding risks."""
    results: List[PIIDetectionResult] = []

    for idx, item in enumerate(items):
        text = item.get(text_field, "")
        has_embedding = embedding_field in item and item[embedding_field] is not None
        text_id = item.get("id", str(idx))

        result = assess_embedding_risk(
            text=text,
            has_embedding=has_embedding,
            text_id=text_id,
        )
        results.append(result)

    items_with_pii = sum(1 for r in results if r.pii_count > 0)
    items_at_risk = sum(
        1 for r in results if r.risk_level in ("critical", "high")
    )

    risk_counts: Dict[str, int] = {}
    pattern_counts: Dict[str, int] = {}
    for r in results:
        risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1
        for p in r.pii_found:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

    summary: Dict[str, Any] = {
        "risk_distribution": risk_counts,
        "pattern_distribution": pattern_counts,
    }

    return PIISafetyReport(
        results=results,
        total_items=len(items),
        items_with_pii=items_with_pii,
        items_at_risk=items_at_risk,
        summary=summary,
    )


def format_pii_report(report: PIISafetyReport) -> str:
    """Format a PIISafetyReport as a human-readable string."""
    lines: List[str] = []
    lines.append("PII Safety Report")
    lines.append("=" * 40)
    lines.append(f"Total items scanned: {report.total_items}")
    lines.append(f"Items with PII: {report.items_with_pii}")
    lines.append(f"Items at risk (critical/high): {report.items_at_risk}")
    lines.append("")

    risk_dist = report.summary.get("risk_distribution", {})
    if risk_dist:
        lines.append("Risk distribution:")
        for level in ("critical", "high", "medium", "low", "none"):
            count = risk_dist.get(level, 0)
            if count > 0:
                lines.append(f"  {level}: {count}")
        lines.append("")

    pattern_dist = report.summary.get("pattern_distribution", {})
    if pattern_dist:
        lines.append("PII patterns detected:")
        for name, count in sorted(pattern_dist.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")
        lines.append("")

    at_risk = [r for r in report.results if r.risk_level in ("critical", "high")]
    if at_risk:
        lines.append("Items requiring attention:")
        for r in at_risk:
            lines.append(
                f"  [{r.risk_level.upper()}] {r.text_id} - "
                f"PII: {', '.join(r.pii_found)}"
            )
            for rec in r.recommendations:
                lines.append(f"    - {rec}")
        lines.append("")

    return "\n".join(lines)


def suggest_mitigations(result: PIIDetectionResult) -> List[str]:
    """Return mitigation suggestions based on risk level and PII types."""
    suggestions: List[str] = []

    if result.risk_level == "none":
        suggestions.append("No PII detected - no mitigations needed.")
        return suggestions

    if result.risk_level in ("critical", "high"):
        suggestions.append(
            "Priority: redact or remove PII from source text before "
            "embedding generation."
        )

    if result.has_embedding:
        suggestions.append(
            "Apply differential-privacy mechanisms (e.g., noise injection) "
            "to stored embeddings."
        )
        suggestions.append(
            "Limit embedding dimensionality to reduce information leakage."
        )

    if "ssn" in result.pii_found or "credit_card" in result.pii_found:
        suggestions.append(
            "Remove highly sensitive fields entirely rather than attempting "
            "redaction."
        )

    if "email" in result.pii_found:
        suggestions.append("Replace email addresses with hashed or synthetic values.")

    if "phone" in result.pii_found:
        suggestions.append("Replace phone numbers with synthetic placeholders.")

    if "ip_address" in result.pii_found:
        suggestions.append("Truncate or generalize IP addresses (e.g., zero out last octet).")

    if "date_of_birth" in result.pii_found:
        suggestions.append("Convert dates of birth to age ranges.")

    if result.risk_level in ("medium", "low") and not result.has_embedding:
        suggestions.append(
            "Monitor this item - risk is lower without an associated "
            "embedding but PII should still be handled carefully."
        )

    return suggestions
