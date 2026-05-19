"""
Domain transfer simulation: adapt seed inputs across domains.

Substitutes domain-specific vocabulary while preserving structure and
complexity. Uses built-in vocabularies for medical, legal, finance,
and general domains. Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DomainVocabulary:
    """Domain-specific vocabulary mapping generic terms to domain terms."""

    domain_id: str
    name: str
    terms: dict[str, str]  # generic term -> domain-specific term
    style_hints: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "terms": dict(self.terms),
            "style_hints": list(self.style_hints),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainVocabulary:
        return cls(
            domain_id=data["domain_id"],
            name=data["name"],
            terms=data.get("terms", {}),
            style_hints=data.get("style_hints", []),
        )


@dataclass
class TransferConfig:
    """Configuration for a domain transfer operation."""

    source_domain: str
    target_domain: str
    preserve_structure: bool = True
    preserve_complexity: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "preserve_structure": self.preserve_structure,
            "preserve_complexity": self.preserve_complexity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferConfig:
        return cls(
            source_domain=data["source_domain"],
            target_domain=data["target_domain"],
            preserve_structure=data.get("preserve_structure", True),
            preserve_complexity=data.get("preserve_complexity", True),
        )


@dataclass
class TransferredInput:
    """A single input after domain transfer."""

    original_text: str
    transferred_text: str
    source_domain: str
    target_domain: str
    substitutions_made: int
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "transferred_text": self.transferred_text,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "substitutions_made": self.substitutions_made,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferredInput:
        return cls(
            original_text=data["original_text"],
            transferred_text=data["transferred_text"],
            source_domain=data["source_domain"],
            target_domain=data["target_domain"],
            substitutions_made=data.get("substitutions_made", 0),
            tags=data.get("tags", []),
        )


@dataclass
class TransferReport:
    """Summary of a batch domain transfer operation."""

    inputs: list[TransferredInput]
    total_items: int
    avg_substitutions: float
    domains_used: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "inputs": [i.as_dict() for i in self.inputs],
            "total_items": self.total_items,
            "avg_substitutions": self.avg_substitutions,
            "domains_used": list(self.domains_used),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferReport:
        return cls(
            inputs=[TransferredInput.from_dict(i) for i in data.get("inputs", [])],
            total_items=data.get("total_items", 0),
            avg_substitutions=data.get("avg_substitutions", 0.0),
            domains_used=data.get("domains_used", []),
        )


# ---------------------------------------------------------------------------
# Built-in Vocabularies
# ---------------------------------------------------------------------------

BUILTIN_VOCABULARIES: dict[str, DomainVocabulary] = {
    "medical": DomainVocabulary(
        domain_id="medical",
        name="Medical",
        terms={
            "patient": "client",
            "diagnosis": "assessment",
            "prescription": "recommendation",
            "symptoms": "indicators",
            "treatment": "intervention",
            "hospital": "facility",
        },
        style_hints=["clinical tone", "precise terminology", "passive voice"],
    ),
    "legal": DomainVocabulary(
        domain_id="legal",
        name="Legal",
        terms={
            "client": "party",
            "agreement": "contract",
            "dispute": "litigation",
            "property": "asset",
            "payment": "consideration",
            "breach": "violation",
        },
        style_hints=["formal register", "precise language", "hedging clauses"],
    ),
    "finance": DomainVocabulary(
        domain_id="finance",
        name="Finance",
        terms={
            "customer": "investor",
            "account": "portfolio",
            "transaction": "trade",
            "balance": "position",
            "fee": "commission",
            "deposit": "capital",
        },
        style_hints=["quantitative focus", "risk language", "formal tone"],
    ),
    "general": DomainVocabulary(
        domain_id="general",
        name="General",
        terms={
            "patient": "patient",
            "diagnosis": "diagnosis",
            "prescription": "prescription",
            "symptoms": "symptoms",
            "treatment": "treatment",
            "hospital": "hospital",
            "client": "client",
            "agreement": "agreement",
            "dispute": "dispute",
            "property": "property",
            "payment": "payment",
            "breach": "breach",
            "customer": "customer",
            "account": "account",
            "transaction": "transaction",
            "balance": "balance",
            "fee": "fee",
            "deposit": "deposit",
        },
        style_hints=["neutral tone", "plain language"],
    ),
}


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def _preserve_case(original: str, replacement: str) -> str:
    """Apply the casing pattern of original to replacement."""
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original and original[0].isupper():
        return replacement[0].upper() + replacement[1:] if replacement else replacement
    return replacement


def transfer_text(
    text: str,
    source_vocab: DomainVocabulary,
    target_vocab: DomainVocabulary,
) -> tuple[str, int]:
    """Substitute domain terms using reverse source lookup + target forward lookup.

    Builds a reverse map from the source vocabulary (domain-specific -> generic),
    then applies the target vocabulary (generic -> domain-specific) to replace
    terms. Case is preserved from the original text.

    Returns (transferred_text, substitution_count).
    """
    # Build reverse map: source domain-specific term -> generic term
    reverse_source: dict[str, str] = {}
    for generic, specific in source_vocab.terms.items():
        reverse_source[specific.lower()] = generic
        # Also map the generic term itself (source text may use either)
        reverse_source[generic.lower()] = generic

    # Build forward map: generic -> target domain-specific term
    forward_target: dict[str, str] = {}
    for generic, specific in target_vocab.terms.items():
        forward_target[generic.lower()] = specific

    substitution_count = 0
    result_text = text

    # Collect all terms to substitute, sorted longest first to avoid partial matches
    all_source_terms = sorted(reverse_source.keys(), key=len, reverse=True)

    for source_term in all_source_terms:
        generic = reverse_source[source_term]
        target_term = forward_target.get(generic.lower())
        if target_term is None:
            continue
        # Skip identity replacements
        if source_term.lower() == target_term.lower():
            continue

        # Word-boundary regex for case-insensitive matching
        pattern = re.compile(r"\b" + re.escape(source_term) + r"\b", re.IGNORECASE)
        matches = list(pattern.finditer(result_text))
        if not matches:
            continue

        # Replace from right to left to preserve positions
        for match in reversed(matches):
            original_word = match.group()
            replacement = _preserve_case(original_word, target_term)
            result_text = (
                result_text[: match.start()] + replacement + result_text[match.end() :]
            )
            substitution_count += 1

    return result_text, substitution_count


def transfer_batch(
    texts: list[str],
    config: TransferConfig,
    vocabularies: dict[str, DomainVocabulary] | None = None,
) -> TransferReport:
    """Transfer a batch of texts between domains.

    Uses BUILTIN_VOCABULARIES if vocabularies is not provided.
    """
    vocabs = vocabularies or BUILTIN_VOCABULARIES
    source_domain = config.source_domain
    target_domain = config.target_domain
    general_vocab = vocabs.get("general")

    # For non-general to non-general transfers, route through general
    # as intermediary to ensure overlapping generic keys
    if (
        source_domain != "general"
        and target_domain != "general"
        and general_vocab is not None
    ):
        source_vocab = general_vocab
        target_vocab = vocabs.get(target_domain)
    else:
        source_vocab = vocabs.get(source_domain)
        target_vocab = vocabs.get(target_domain)

    if source_vocab is None or target_vocab is None:
        missing = []
        if source_vocab is None:
            missing.append(config.source_domain)
        if target_vocab is None:
            missing.append(config.target_domain)
        raise ValueError(f"Unknown domain(s): {', '.join(missing)}")

    transferred: list[TransferredInput] = []
    for text in texts:
        new_text, count = transfer_text(text, source_vocab, target_vocab)

        # Validate complexity preservation if configured
        if config.preserve_complexity:
            valid = validate_transfer(text, new_text)
            tags = ["domain_transfer", config.target_domain]
            if not valid:
                tags.append("complexity_drift")
        else:
            tags = ["domain_transfer", config.target_domain]

        transferred.append(
            TransferredInput(
                original_text=text,
                transferred_text=new_text,
                source_domain=config.source_domain,
                target_domain=config.target_domain,
                substitutions_made=count,
                tags=tags,
            )
        )

    total = len(transferred)
    total_subs = sum(t.substitutions_made for t in transferred)
    avg_subs = total_subs / total if total > 0 else 0.0
    domains_used = sorted({config.source_domain, config.target_domain})

    return TransferReport(
        inputs=transferred,
        total_items=total,
        avg_substitutions=avg_subs,
        domains_used=domains_used,
    )


def estimate_complexity(text: str) -> float:
    """Compute a simple complexity score: avg_word_length * unique_word_ratio.

    Returns 0.0 for empty text.
    """
    words = text.split()
    if not words:
        return 0.0
    avg_length = sum(len(w) for w in words) / len(words)
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    return avg_length * unique_ratio


def validate_transfer(
    original: str,
    transferred: str,
    max_drift: float = 0.3,
) -> bool:
    """Check that transferred text preserves complexity within drift threshold.

    Compares the complexity scores of original and transferred text. Returns
    True if the relative drift is within max_drift.
    """
    orig_score = estimate_complexity(original)
    trans_score = estimate_complexity(transferred)

    if orig_score == 0.0:
        # If original has zero complexity, any transfer is valid
        return True

    drift = abs(trans_score - orig_score) / orig_score
    return drift <= max_drift


def format_transfer_report(report: TransferReport) -> str:
    """Format a transfer report as human-readable text."""
    lines: list[str] = []
    lines.append("DOMAIN TRANSFER REPORT")
    lines.append("=" * 50)
    lines.append(f"Total items: {report.total_items}")
    lines.append(f"Avg substitutions: {report.avg_substitutions:.1f}")
    lines.append(f"Domains: {', '.join(report.domains_used)}")
    lines.append("")
    lines.append("TRANSFERS")
    lines.append("-" * 50)

    for item in report.inputs:
        lines.append(f"  [{item.source_domain} -> {item.target_domain}]")
        lines.append(f"    Original:    {item.original_text}")
        lines.append(f"    Transferred: {item.transferred_text}")
        lines.append(f"    Substitutions: {item.substitutions_made}")
        if item.tags:
            lines.append(f"    Tags: {', '.join(item.tags)}")
        lines.append("")

    return "\n".join(lines)
