"""Tests for locale-aware sampling module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.locale_sampling import (
    LocaleConfig,
    LocaleStats,
    LocaleResult,
    detect_locale,
    group_by_locale,
    sample_by_locale,
    run_locale_sampling,
    format_locale_report,
)


# ---------------------------------------------------------------------------
# LocaleConfig dataclass
# ---------------------------------------------------------------------------


class TestLocaleConfig:
    def test_defaults(self):
        c = LocaleConfig()
        assert c.min_per_locale == 1
        assert c.proportional is True
        assert c.locale_field == ""
        assert c.seed is None
        assert c.sample_size == 100

    def test_custom(self):
        c = LocaleConfig(min_per_locale=3, proportional=False, locale_field="lang", seed=42, sample_size=50)
        assert c.min_per_locale == 3
        assert c.proportional is False
        assert c.locale_field == "lang"
        assert c.seed == 42
        assert c.sample_size == 50

    def test_as_dict(self):
        c = LocaleConfig(seed=99)
        d = c.as_dict()
        assert d["seed"] == 99
        assert d["min_per_locale"] == 1

    def test_from_dict(self):
        d = {"min_per_locale": 5, "proportional": False, "locale_field": "language", "seed": 7, "sample_size": 200}
        c = LocaleConfig.from_dict(d)
        assert c.min_per_locale == 5
        assert c.seed == 7

    def test_roundtrip(self):
        original = LocaleConfig(min_per_locale=2, seed=123, sample_size=75)
        restored = LocaleConfig.from_dict(original.as_dict())
        assert restored.min_per_locale == original.min_per_locale
        assert restored.seed == original.seed
        assert restored.sample_size == original.sample_size

    def test_from_dict_defaults(self):
        c = LocaleConfig.from_dict({})
        assert c.min_per_locale == 1
        assert c.seed is None


# ---------------------------------------------------------------------------
# LocaleStats dataclass
# ---------------------------------------------------------------------------


class TestLocaleStats:
    def test_creation(self):
        s = LocaleStats(locale="en", total_count=100, sampled_count=20, target_rate=0.5)
        assert s.locale == "en"
        assert s.total_count == 100

    def test_as_dict(self):
        s = LocaleStats(locale="ja", total_count=50, sampled_count=10, target_rate=0.25)
        d = s.as_dict()
        assert d["locale"] == "ja"
        assert d["target_rate"] == 0.25

    def test_from_dict(self):
        d = {"locale": "ko", "total_count": 30, "sampled_count": 5, "target_rate": 0.15}
        s = LocaleStats.from_dict(d)
        assert s.locale == "ko"
        assert s.sampled_count == 5

    def test_roundtrip(self):
        original = LocaleStats(locale="ru", total_count=40, sampled_count=8, target_rate=0.2)
        restored = LocaleStats.from_dict(original.as_dict())
        assert restored.locale == original.locale
        assert restored.total_count == original.total_count


# ---------------------------------------------------------------------------
# LocaleResult dataclass
# ---------------------------------------------------------------------------


class TestLocaleResult:
    def test_empty(self):
        r = LocaleResult()
        assert r.selected_ids == []
        assert r.locale_stats == []
        assert r.total_pool == 0
        assert r.locales_represented == 0

    def test_as_dict(self):
        stats = [LocaleStats(locale="en", total_count=10, sampled_count=5, target_rate=1.0)]
        r = LocaleResult(selected_ids=["a", "b"], locale_stats=stats, total_pool=10, locales_represented=1)
        d = r.as_dict()
        assert d["selected_ids"] == ["a", "b"]
        assert len(d["locale_stats"]) == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["x"],
            "locale_stats": [{"locale": "ar", "total_count": 5, "sampled_count": 2, "target_rate": 0.5}],
            "total_pool": 10,
            "locales_represented": 1,
        }
        r = LocaleResult.from_dict(d)
        assert r.selected_ids == ["x"]
        assert r.locale_stats[0].locale == "ar"

    def test_roundtrip(self):
        stats = [LocaleStats(locale="hi", total_count=20, sampled_count=4, target_rate=0.33)]
        original = LocaleResult(selected_ids=["a", "b", "c", "d"], locale_stats=stats, total_pool=20, locales_represented=1)
        restored = LocaleResult.from_dict(original.as_dict())
        assert restored.total_pool == 20
        assert len(restored.locale_stats) == 1


# ---------------------------------------------------------------------------
# detect_locale
# ---------------------------------------------------------------------------


class TestDetectLocale:
    def test_english(self):
        assert detect_locale("Hello, how are you?") == "en"

    def test_japanese_hiragana(self):
        assert detect_locale("こんにちは世界") == "ja"

    def test_japanese_katakana(self):
        assert detect_locale("カタカナテスト") == "ja"

    def test_korean(self):
        assert detect_locale("안녕하세요") == "ko"

    def test_russian(self):
        assert detect_locale("Привет мир") == "ru"

    def test_arabic(self):
        assert detect_locale("مرحبا بالعالم") == "ar"

    def test_hindi(self):
        assert detect_locale("नमस्ते दुनिया") == "hi"

    def test_empty_string(self):
        assert detect_locale("") == "en"

    def test_numbers_only(self):
        assert detect_locale("12345") == "en"

    def test_mixed_defaults_to_majority(self):
        # Mostly Japanese with a few English words
        text = "これはテストです hello"
        assert detect_locale(text) == "ja"

    def test_cjk_unified(self):
        # CJK Unified Ideographs without kana -> Chinese
        assert detect_locale("\u4e00\u4e01\u4e02") == "zh"


# ---------------------------------------------------------------------------
# group_by_locale
# ---------------------------------------------------------------------------


class TestGroupByLocale:
    def test_auto_detect(self):
        items = {
            "a": "Hello world",
            "b": "こんにちは",
            "c": "Goodbye",
        }
        groups = group_by_locale(items)
        assert "en" in groups
        assert "ja" in groups
        assert "a" in groups["en"]
        assert "b" in groups["ja"]
        assert "c" in groups["en"]

    def test_with_locale_field(self):
        items = {
            "a": "en",
            "b": "ja",
            "c": "en",
        }
        groups = group_by_locale(items, locale_field="language")
        assert groups["en"] == ["a", "c"]
        assert groups["ja"] == ["b"]

    def test_empty_items(self):
        groups = group_by_locale({})
        assert groups == {}

    def test_all_same_locale(self):
        items = {"a": "hello", "b": "world", "c": "test"}
        groups = group_by_locale(items)
        assert len(groups) == 1
        assert "en" in groups
        assert len(groups["en"]) == 3

    def test_multiple_locales(self):
        items = {
            "en1": "hello",
            "ja1": "テスト",
            "ko1": "안녕",
            "ru1": "Привет",
        }
        groups = group_by_locale(items)
        assert len(groups) == 4


# ---------------------------------------------------------------------------
# sample_by_locale
# ---------------------------------------------------------------------------


class TestSampleByLocale:
    def test_basic_proportional(self):
        groups = {"en": [f"en_{i}" for i in range(80)], "ja": [f"ja_{i}" for i in range(20)]}
        config = LocaleConfig(sample_size=10, seed=42)
        selected = sample_by_locale(groups, config)
        assert len(selected) == 10

    def test_minimum_guarantee(self):
        groups = {"en": [f"en_{i}" for i in range(90)], "ja": ["ja_0", "ja_1"]}
        config = LocaleConfig(sample_size=10, min_per_locale=2, seed=42)
        selected = sample_by_locale(groups, config)
        ja_count = sum(1 for s in selected if s.startswith("ja_"))
        assert ja_count >= 2

    def test_empty_groups(self):
        selected = sample_by_locale({}, LocaleConfig(sample_size=10))
        assert selected == []

    def test_sample_size_larger_than_pool(self):
        groups = {"en": ["a", "b", "c"]}
        config = LocaleConfig(sample_size=100, seed=42)
        selected = sample_by_locale(groups, config)
        assert len(selected) == 3  # capped at pool size

    def test_deterministic_with_seed(self):
        groups = {"en": [f"en_{i}" for i in range(50)], "ja": [f"ja_{i}" for i in range(50)]}
        config = LocaleConfig(sample_size=20, seed=123)
        run1 = sample_by_locale(groups, config)
        run2 = sample_by_locale(groups, config)
        assert run1 == run2

    def test_non_proportional(self):
        groups = {"en": [f"en_{i}" for i in range(80)], "ja": [f"ja_{i}" for i in range(20)]}
        config = LocaleConfig(sample_size=10, proportional=False, seed=42)
        selected = sample_by_locale(groups, config)
        assert len(selected) == 10
        # Non-proportional should give more equal distribution
        en_count = sum(1 for s in selected if s.startswith("en_"))
        ja_count = sum(1 for s in selected if s.startswith("ja_"))
        assert ja_count >= 4  # should be roughly equal (5 each)

    def test_min_per_locale_exceeds_available(self):
        groups = {"en": ["en_0"], "ja": ["ja_0", "ja_1"]}
        config = LocaleConfig(sample_size=10, min_per_locale=5, seed=42)
        selected = sample_by_locale(groups, config)
        # Should not sample more than available
        en_count = sum(1 for s in selected if s.startswith("en_"))
        ja_count = sum(1 for s in selected if s.startswith("ja_"))
        assert en_count <= 1
        assert ja_count <= 2

    def test_single_locale(self):
        groups = {"en": [f"en_{i}" for i in range(50)]}
        config = LocaleConfig(sample_size=10, seed=42)
        selected = sample_by_locale(groups, config)
        assert len(selected) == 10
        assert all(s.startswith("en_") for s in selected)

    def test_many_locales_all_represented(self):
        groups = {
            "en": [f"en_{i}" for i in range(40)],
            "ja": [f"ja_{i}" for i in range(20)],
            "ko": [f"ko_{i}" for i in range(15)],
            "ru": [f"ru_{i}" for i in range(15)],
            "ar": [f"ar_{i}" for i in range(10)],
        }
        config = LocaleConfig(sample_size=20, min_per_locale=1, seed=42)
        selected = sample_by_locale(groups, config)
        assert len(selected) == 20
        # Check all locales are represented
        for locale in groups:
            count = sum(1 for s in selected if s.startswith(f"{locale}_"))
            assert count >= 1


# ---------------------------------------------------------------------------
# run_locale_sampling
# ---------------------------------------------------------------------------


class TestRunLocaleSampling:
    def test_basic(self):
        items = {f"en_{i}": "hello world" for i in range(50)}
        items.update({f"ja_{i}": "テストです" for i in range(50)})
        config = LocaleConfig(sample_size=20, seed=42)
        result = run_locale_sampling(items, config)
        assert len(result.selected_ids) == 20
        assert result.total_pool == 100
        assert result.locales_represented == 2

    def test_locale_stats_populated(self):
        items = {"a": "hello", "b": "world", "c": "テスト"}
        config = LocaleConfig(sample_size=3, seed=42)
        result = run_locale_sampling(items, config)
        assert len(result.locale_stats) >= 1
        total = sum(s.total_count for s in result.locale_stats)
        assert total == 3

    def test_with_locale_field(self):
        items = {"a": "en", "b": "ja", "c": "en", "d": "ko"}
        config = LocaleConfig(sample_size=3, locale_field="lang", seed=42)
        result = run_locale_sampling(items, config)
        assert len(result.selected_ids) == 3
        assert result.locales_represented >= 1

    def test_empty_items(self):
        result = run_locale_sampling({}, LocaleConfig(sample_size=10))
        assert result.selected_ids == []
        assert result.total_pool == 0
        assert result.locales_represented == 0

    def test_target_rates_sum_to_one(self):
        items = {f"en_{i}": "hello" for i in range(60)}
        items.update({f"ja_{i}": "テスト" for i in range(40)})
        config = LocaleConfig(sample_size=20, seed=42)
        result = run_locale_sampling(items, config)
        total_rate = sum(s.target_rate for s in result.locale_stats)
        assert abs(total_rate - 1.0) < 0.01

    def test_all_locales_represented(self):
        items = {}
        items.update({f"en_{i}": "hello world" for i in range(30)})
        items.update({f"ja_{i}": "テストです" for i in range(30)})
        items.update({f"ko_{i}": "안녕하세요" for i in range(20)})
        items.update({f"ru_{i}": "Привет мир" for i in range(20)})
        config = LocaleConfig(sample_size=20, min_per_locale=1, seed=42)
        result = run_locale_sampling(items, config)
        assert result.locales_represented == 4


# ---------------------------------------------------------------------------
# format_locale_report
# ---------------------------------------------------------------------------


class TestFormatLocaleReport:
    def test_header_present(self):
        result = LocaleResult(total_pool=100, locales_represented=2)
        text = format_locale_report(result)
        assert "Total pool: 100" in text
        assert "Locales represented: 2" in text
        assert "Locale" in text
        assert "Total" in text

    def test_rows_present(self):
        stats = [LocaleStats(locale="en", total_count=80, sampled_count=16, target_rate=0.8)]
        result = LocaleResult(selected_ids=["a"] * 16, locale_stats=stats, total_pool=100, locales_represented=1)
        text = format_locale_report(result)
        assert "en" in text
        assert "80" in text
        assert "16" in text

    def test_empty_result(self):
        result = LocaleResult()
        text = format_locale_report(result)
        assert "Total pool: 0" in text
        assert "Selected: 0" in text

    def test_multiple_locales(self):
        stats = [
            LocaleStats(locale="en", total_count=50, sampled_count=10, target_rate=0.5),
            LocaleStats(locale="ja", total_count=50, sampled_count=10, target_rate=0.5),
        ]
        result = LocaleResult(selected_ids=["x"] * 20, locale_stats=stats, total_pool=100, locales_represented=2)
        text = format_locale_report(result)
        assert "en" in text
        assert "ja" in text
