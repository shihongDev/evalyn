"""Locale-aware sampling: sample proportionally by language or region.

Pure Python, no external dependencies. Uses heuristic language detection
based on Unicode character ranges to group items by locale, then samples
proportionally with per-locale minimum guarantees.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class LocaleConfig:
    """Configuration for locale-aware sampling."""

    min_per_locale: int = 1
    proportional: bool = True
    locale_field: str = ""  # if set, read locale from metadata; else auto-detect
    seed: Optional[int] = None
    sample_size: int = 100

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_per_locale": self.min_per_locale,
            "proportional": self.proportional,
            "locale_field": self.locale_field,
            "seed": self.seed,
            "sample_size": self.sample_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LocaleConfig:
        return cls(
            min_per_locale=data.get("min_per_locale", 1),
            proportional=data.get("proportional", True),
            locale_field=data.get("locale_field", ""),
            seed=data.get("seed"),
            sample_size=data.get("sample_size", 100),
        )


@dataclass
class LocaleStats:
    """Statistics for a single locale in sampling results."""

    locale: str
    total_count: int
    sampled_count: int
    target_rate: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "locale": self.locale,
            "total_count": self.total_count,
            "sampled_count": self.sampled_count,
            "target_rate": self.target_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LocaleStats:
        return cls(
            locale=data["locale"],
            total_count=data["total_count"],
            sampled_count=data["sampled_count"],
            target_rate=data.get("target_rate", 0.0),
        )


@dataclass
class LocaleResult:
    """Result of locale-aware sampling."""

    selected_ids: List[str] = field(default_factory=list)
    locale_stats: List[LocaleStats] = field(default_factory=list)
    total_pool: int = 0
    locales_represented: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "locale_stats": [s.as_dict() for s in self.locale_stats],
            "total_pool": self.total_pool,
            "locales_represented": self.locales_represented,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LocaleResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            locale_stats=[LocaleStats.from_dict(s) for s in data.get("locale_stats", [])],
            total_pool=data.get("total_pool", 0),
            locales_represented=data.get("locales_represented", 0),
        )


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------


def detect_locale(text: str) -> str:
    """Heuristic language detection based on Unicode ranges and Latin markers.

    Checks for CJK/Hiragana/Katakana (ja/zh), Hangul (ko), Cyrillic (ru),
    Arabic (ar), Devanagari (hi), Thai (th), Greek (el).
    For Latin-script text, uses accented character and common word markers
    to distinguish French, Spanish, German, Portuguese, Italian, and Dutch.
    Falls back to "en".

    Args:
        text: Input text to analyze.

    Returns:
        Two-letter locale code.
    """
    counts: Dict[str, int] = {
        "ko": 0,
        "ja": 0,
        "zh": 0,
        "ru": 0,
        "ar": 0,
        "hi": 0,
        "th": 0,
        "el": 0,
    }

    for ch in text:
        cp = ord(ch)
        # Korean Hangul: Jamo, Syllables, Compatibility Jamo
        if (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F):
            counts["ko"] += 1
        # Hiragana + Katakana -> ja
        elif (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
            counts["ja"] += 1
        # CJK Unified Ideographs -> zh (unless ja already has kana)
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["zh"] += 1
        # Cyrillic
        elif 0x0400 <= cp <= 0x04FF:
            counts["ru"] += 1
        # Arabic
        elif 0x0600 <= cp <= 0x06FF:
            counts["ar"] += 1
        # Devanagari
        elif 0x0900 <= cp <= 0x097F:
            counts["hi"] += 1
        # Thai
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["th"] += 1
        # Greek
        elif 0x0370 <= cp <= 0x03FF:
            counts["el"] += 1

    # If CJK chars present with kana, it's Japanese
    if counts["ja"] > 0 and counts["zh"] > 0:
        counts["ja"] += counts["zh"]
        counts["zh"] = 0

    best = max(counts, key=lambda k: counts[k])
    if counts[best] > 0:
        return best

    # Latin-script language detection via common word markers
    lower = text.lower()
    _LANG_MARKERS: Dict[str, List[str]] = {
        "fr": ["le ", "la ", "les ", "des ", "est ", "une ", "dans ", "pour ", "avec ", "que "],
        "es": [" el ", " la ", " los ", " las ", " es ", " una ", " del ", " por ", " con ", " que "],
        "de": [" der ", " die ", " das ", " und ", " ist ", " ein ", " mit ", " den ", " von "],
        "pt": [" o ", " os ", " uma ", " com ", " para ", " que ", " mais ", " por "],
        "it": [" il ", " la ", " che ", " per ", " con ", " una ", " del ", " sono "],
        "nl": [" de ", " het ", " een ", " van ", " en ", " is ", " dat ", " niet "],
    }
    marker_hits: Dict[str, int] = {}
    for lang, markers in _LANG_MARKERS.items():
        marker_hits[lang] = sum(1 for m in markers if m in lower)

    best_lang = max(marker_hits, key=lambda k: marker_hits[k])
    if marker_hits[best_lang] >= 3:
        return best_lang
    return "en"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_by_locale(
    items: Dict[str, str],
    locale_field: str = "",
) -> Dict[str, List[str]]:
    """Group item IDs by detected or specified locale.

    Args:
        items: Mapping of item_id to text content (or locale value if
            locale_field is set and the text is the locale directly).
        locale_field: If non-empty, treat the value as a pre-assigned
            locale string instead of auto-detecting from text.

    Returns:
        Mapping of locale code to list of item IDs.
    """
    groups: Dict[str, List[str]] = {}
    for item_id, text in items.items():
        if locale_field:
            locale = text  # value is already the locale
        else:
            locale = detect_locale(text)
        if locale not in groups:
            groups[locale] = []
        groups[locale].append(item_id)
    return groups


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_by_locale(
    groups: Dict[str, List[str]],
    config: LocaleConfig,
) -> List[str]:
    """Sample items proportionally by locale with minimum guarantees.

    Algorithm:
    1. Guarantee min_per_locale items from each locale (capped by available).
    2. Distribute remaining budget proportionally if proportional=True,
       otherwise distribute equally.
    3. Random sample within each locale.

    Args:
        groups: Mapping of locale to list of item IDs.
        config: Sampling configuration.

    Returns:
        List of selected item IDs.
    """
    rng = random.Random(config.seed)

    if not groups:
        return []

    total_pool = sum(len(ids) for ids in groups.values())
    budget = min(config.sample_size, total_pool)

    # Phase 1: guarantee minimums
    allocations: Dict[str, int] = {}
    used = 0
    for locale, ids in groups.items():
        alloc = min(config.min_per_locale, len(ids))
        allocations[locale] = alloc
        used += alloc

    # Phase 2: distribute remaining budget
    remaining = max(0, budget - used)
    if remaining > 0:
        if config.proportional:
            # Proportional to locale size
            for locale, ids in groups.items():
                share = (len(ids) / total_pool) * remaining
                extra = min(int(share), len(ids) - allocations[locale])
                allocations[locale] += max(0, extra)
            # Distribute rounding leftovers
            current_total = sum(allocations.values())
            leftover = budget - current_total
            # Sort locales by fractional share descending for fair rounding
            fractional: List[tuple[str, float]] = []
            for locale, ids in groups.items():
                share = (len(ids) / total_pool) * remaining
                frac = share - int(share)
                fractional.append((locale, frac))
            fractional.sort(key=lambda x: -x[1])
            for locale, _ in fractional:
                if leftover <= 0:
                    break
                headroom = len(groups[locale]) - allocations[locale]
                if headroom > 0:
                    allocations[locale] += 1
                    leftover -= 1
        else:
            # Equal distribution
            locales = sorted(groups.keys())
            per_locale = remaining // len(locales)
            extra_count = remaining % len(locales)
            for i, locale in enumerate(locales):
                extra = per_locale + (1 if i < extra_count else 0)
                headroom = len(groups[locale]) - allocations[locale]
                allocations[locale] += min(extra, headroom)

    # Phase 3: sample within each locale
    selected: List[str] = []
    for locale in sorted(groups.keys()):
        ids = list(groups[locale])
        n = min(allocations.get(locale, 0), len(ids))
        if n > 0:
            sampled = rng.sample(ids, n)
            selected.extend(sampled)

    return selected


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def run_locale_sampling(
    items: Dict[str, str],
    config: LocaleConfig,
) -> LocaleResult:
    """Run locale-aware sampling end to end.

    Args:
        items: Mapping of item_id to text content.
        config: Sampling configuration.

    Returns:
        LocaleResult with selected IDs and per-locale statistics.
    """
    if config.locale_field:
        groups = group_by_locale(items, locale_field=config.locale_field)
    else:
        groups = group_by_locale(items)

    selected = sample_by_locale(groups, config)
    selected_set = set(selected)

    total_pool = len(items)
    stats: List[LocaleStats] = []
    for locale in sorted(groups.keys()):
        ids = groups[locale]
        sampled = sum(1 for i in ids if i in selected_set)
        rate = len(ids) / total_pool if total_pool > 0 else 0.0
        stats.append(
            LocaleStats(
                locale=locale,
                total_count=len(ids),
                sampled_count=sampled,
                target_rate=rate,
            )
        )

    return LocaleResult(
        selected_ids=selected,
        locale_stats=stats,
        total_pool=total_pool,
        locales_represented=sum(1 for s in stats if s.sampled_count > 0),
    )


def format_locale_report(result: LocaleResult) -> str:
    """Format a LocaleResult as a human-readable report.

    Args:
        result: The sampling result to format.

    Returns:
        Multi-line string with summary and per-locale table.
    """
    lines: List[str] = []
    lines.append(f"Total pool: {result.total_pool}")
    lines.append(f"Selected: {len(result.selected_ids)}")
    lines.append(f"Locales represented: {result.locales_represented}")
    lines.append("")

    header = f"{'Locale':>8}  {'Total':>8}  {'Sampled':>8}  {'Rate':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for s in result.locale_stats:
        lines.append(
            f"{s.locale:>8}  {s.total_count:>8}  "
            f"{s.sampled_count:>8}  {s.target_rate:>8.2%}"
        )

    return "\n".join(lines)
