"""Language detection for dataset items.

Detects the language of text using character set analysis and n-gram
heuristics. No external API or heavy dependencies required.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LanguageDetection:
    """Language detection result."""

    language: str  # ISO 639-1 code (e.g., "en", "ja", "zh")
    confidence: float  # 0.0-1.0
    script: str  # Primary script (e.g., "Latin", "CJK", "Cyrillic")

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "script": self.script,
        }


@dataclass
class DatasetLanguageReport:
    """Language distribution across a dataset."""

    language_counts: dict[str, int] = field(default_factory=dict)
    total_items: int = 0

    @property
    def primary_language(self) -> str | None:
        if not self.language_counts:
            return None
        return max(self.language_counts, key=self.language_counts.get)

    @property
    def is_multilingual(self) -> bool:
        return len(self.language_counts) > 1

    def distribution(self) -> dict[str, float]:
        if self.total_items == 0:
            return {}
        return {
            lang: count / self.total_items
            for lang, count in sorted(self.language_counts.items(), key=lambda x: -x[1])
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_language": self.primary_language,
            "is_multilingual": self.is_multilingual,
            "total_items": self.total_items,
            "distribution": {k: round(v, 4) for k, v in self.distribution().items()},
        }

    def format_text(self) -> str:
        lines = [
            f"Language Distribution ({self.total_items} items):",
            f"  Primary: {self.primary_language or 'unknown'}",
            f"  Multilingual: {'yes' if self.is_multilingual else 'no'}",
        ]
        for lang, frac in self.distribution().items():
            lines.append(f"  {lang}: {frac:.0%} ({self.language_counts[lang]} items)")
        return "\n".join(lines)


def detect_language(text: str) -> LanguageDetection:
    """Detect the language of a text string.

    Uses character script analysis (Unicode categories) for reliable
    detection of CJK, Cyrillic, Arabic, etc. Falls back to n-gram
    heuristics for Latin-script languages.

    Args:
        text: Text to analyze.

    Returns:
        LanguageDetection with language code, confidence, and script.
    """
    if not text or not text.strip():
        return LanguageDetection(language="unknown", confidence=0.0, script="None")

    # Count character scripts
    scripts = _count_scripts(text)
    total_alpha = sum(scripts.values())

    if total_alpha == 0:
        return LanguageDetection(language="unknown", confidence=0.0, script="None")

    primary_script = max(scripts, key=scripts.get)
    script_ratio = scripts[primary_script] / total_alpha

    # Script-based detection (high confidence for non-Latin scripts)
    if primary_script == "CJK":
        # Distinguish Chinese vs Japanese vs Korean
        lang = _detect_cjk_language(text)
        return LanguageDetection(language=lang, confidence=min(0.95, script_ratio), script="CJK")

    if primary_script == "Hangul":
        return LanguageDetection(language="ko", confidence=min(0.95, script_ratio), script="Hangul")

    if primary_script == "Cyrillic":
        return LanguageDetection(
            language="ru", confidence=min(0.9, script_ratio), script="Cyrillic"
        )

    if primary_script == "Arabic":
        return LanguageDetection(language="ar", confidence=min(0.9, script_ratio), script="Arabic")

    if primary_script == "Devanagari":
        return LanguageDetection(
            language="hi", confidence=min(0.9, script_ratio), script="Devanagari"
        )

    if primary_script == "Thai":
        return LanguageDetection(language="th", confidence=min(0.9, script_ratio), script="Thai")

    # Latin script: use common word heuristics
    if primary_script == "Latin":
        lang, conf = _detect_latin_language(text)
        return LanguageDetection(language=lang, confidence=conf, script="Latin")

    return LanguageDetection(language="unknown", confidence=0.3, script=primary_script)


def analyze_dataset_languages(items: list) -> DatasetLanguageReport:
    """Analyze language distribution across dataset items.

    Args:
        items: List of DatasetItem objects.

    Returns:
        DatasetLanguageReport with per-language counts and distribution.
    """
    counts: dict[str, int] = {}
    for item in items:
        text = _item_to_text(item)
        detection = detect_language(text)
        lang = detection.language
        counts[lang] = counts.get(lang, 0) + 1

    return DatasetLanguageReport(
        language_counts=counts,
        total_items=len(items),
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _count_scripts(text: str) -> dict[str, int]:
    """Count characters by Unicode script category."""
    scripts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        script = _char_script(char)
        scripts[script] = scripts.get(script, 0) + 1
    return scripts


def _char_script(char: str) -> str:
    """Determine the script of a Unicode character."""
    cp = ord(char)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "CJK"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
        return "Hangul"
    if 0x0400 <= cp <= 0x04FF:
        return "Cyrillic"
    if 0x0600 <= cp <= 0x06FF or 0xFE70 <= cp <= 0xFEFF:
        return "Arabic"
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x0E00 <= cp <= 0x0E7F:
        return "Thai"
    if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
        return "CJK"  # Hiragana/Katakana -> Japanese (counted with CJK)
    # Default to Latin for alphabet characters
    cat = unicodedata.category(char)
    if cat.startswith("L"):
        return "Latin"
    return "Other"


def _detect_cjk_language(text: str) -> str:
    """Distinguish Chinese, Japanese, Korean from CJK text."""
    has_hiragana = bool(re.search(r"[\u3040-\u309F]", text))
    has_katakana = bool(re.search(r"[\u30A0-\u30FF]", text))
    has_hangul = bool(re.search(r"[\uAC00-\uD7AF]", text))

    if has_hiragana or has_katakana:
        return "ja"
    if has_hangul:
        return "ko"
    return "zh"


# Common words for Latin-script language detection
_LANG_WORDS = {
    "en": {
        "the",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "been",
        "will",
        "would",
        "could",
        "should",
        "with",
        "from",
        "this",
        "that",
        "they",
        "their",
        "what",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "es",
        "son",
        "fue",
        "ser",
        "con",
        "para",
        "por",
        "como",
        "pero",
        "que",
        "del",
        "una",
        "este",
    },
    "fr": {
        "le",
        "la",
        "les",
        "est",
        "sont",
        "avec",
        "pour",
        "dans",
        "pas",
        "que",
        "une",
        "des",
        "sur",
        "qui",
        "nous",
        "mais",
        "cette",
    },
    "de": {
        "der",
        "die",
        "das",
        "ist",
        "sind",
        "war",
        "haben",
        "mit",
        "und",
        "ein",
        "eine",
        "nicht",
        "auch",
        "sich",
        "auf",
        "aus",
        "aber",
    },
    "pt": {
        "o",
        "os",
        "uma",
        "para",
        "com",
        "que",
        "por",
        "como",
        "mais",
        "foi",
        "ser",
        "tem",
        "mas",
        "dos",
        "das",
        "esta",
        "pelo",
    },
}


def _detect_latin_language(text: str) -> tuple:
    """Detect language from Latin-script text using common words."""
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))

    best_lang = "en"
    best_score = 0

    for lang, common_words in _LANG_WORDS.items():
        overlap = len(words & common_words)
        if overlap > best_score:
            best_score = overlap
            best_lang = lang

    confidence = min(0.85, best_score / 5) if best_score > 0 else 0.3
    return best_lang, confidence


def _item_to_text(item) -> str:
    """Extract text from a DatasetItem for language detection."""
    output = getattr(item, "output", None) or ""
    if isinstance(output, dict):
        import json

        return json.dumps(output, ensure_ascii=False)
    return str(output)
