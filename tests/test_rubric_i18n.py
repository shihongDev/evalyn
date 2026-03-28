"""Tests for multi-language rubric support (rubric_i18n module)."""

from __future__ import annotations

import pytest

from evalyn_sdk.rubric_i18n import (
    DEFAULT_LOCALE_REGISTRY,
    LocalizedRubric,
    RubricLocaleRegistry,
    detect_output_language,
    format_locale_summary,
    match_rubric_language,
)


# ---------------------------------------------------------------------------
# LocalizedRubric dataclass
# ---------------------------------------------------------------------------


class TestLocalizedRubric:
    def test_creation(self):
        r = LocalizedRubric(
            metric_id="helpfulness",
            locale="en",
            prompt="Rate helpfulness.",
            rubric_levels={"5": "Great", "1": "Poor"},
            description="Helpfulness metric.",
        )
        assert r.metric_id == "helpfulness"
        assert r.locale == "en"
        assert r.rubric_levels["5"] == "Great"

    def test_as_dict(self):
        r = LocalizedRubric(
            metric_id="safety",
            locale="ja",
            prompt="Prompt text",
            rubric_levels={"3": "OK"},
            description="Desc",
        )
        d = r.as_dict()
        assert d["metric_id"] == "safety"
        assert d["locale"] == "ja"
        assert d["rubric_levels"] == {"3": "OK"}

    def test_from_dict(self):
        data = {
            "metric_id": "accuracy",
            "locale": "es",
            "prompt": "Evalua la precision.",
            "rubric_levels": {"5": "Excelente", "1": "Malo"},
            "description": "Precision factual.",
        }
        r = LocalizedRubric.from_dict(data)
        assert r.metric_id == "accuracy"
        assert r.locale == "es"
        assert r.rubric_levels["1"] == "Malo"

    def test_from_dict_defaults(self):
        data = {
            "metric_id": "x",
            "locale": "en",
            "prompt": "p",
        }
        r = LocalizedRubric.from_dict(data)
        assert r.rubric_levels == {}
        assert r.description == ""

    def test_roundtrip(self):
        r = LocalizedRubric(
            metric_id="test",
            locale="fr",
            prompt="Evaluer.",
            rubric_levels={"5": "Excellent", "1": "Mauvais"},
            description="Test metric.",
        )
        r2 = LocalizedRubric.from_dict(r.as_dict())
        assert r2.metric_id == r.metric_id
        assert r2.locale == r.locale
        assert r2.rubric_levels == r.rubric_levels

    def test_as_dict_copies_rubric_levels(self):
        levels = {"5": "Good"}
        r = LocalizedRubric("m", "en", "p", levels, "d")
        d = r.as_dict()
        d["rubric_levels"]["5"] = "Changed"
        assert r.rubric_levels["5"] == "Good"


# ---------------------------------------------------------------------------
# RubricLocaleRegistry
# ---------------------------------------------------------------------------


class TestRubricLocaleRegistry:
    def _make_rubric(self, metric_id: str, locale: str) -> LocalizedRubric:
        return LocalizedRubric(
            metric_id=metric_id,
            locale=locale,
            prompt=f"Prompt for {metric_id} in {locale}",
            rubric_levels={"5": "Best", "1": "Worst"},
            description=f"{metric_id} - {locale}",
        )

    def test_register_and_get(self):
        reg = RubricLocaleRegistry()
        r = self._make_rubric("helpfulness", "en")
        reg.register(r)
        assert reg.get("helpfulness", "en") is r

    def test_get_missing_metric(self):
        reg = RubricLocaleRegistry()
        assert reg.get("nonexistent", "en") is None

    def test_get_missing_locale(self):
        reg = RubricLocaleRegistry()
        reg.register(self._make_rubric("helpfulness", "en"))
        assert reg.get("helpfulness", "ja") is None

    def test_get_with_fallback_primary(self):
        reg = RubricLocaleRegistry()
        r_ja = self._make_rubric("helpfulness", "ja")
        r_en = self._make_rubric("helpfulness", "en")
        reg.register(r_ja)
        reg.register(r_en)
        assert reg.get_with_fallback("helpfulness", "ja") is r_ja

    def test_get_with_fallback_uses_fallback(self):
        reg = RubricLocaleRegistry()
        r_en = self._make_rubric("helpfulness", "en")
        reg.register(r_en)
        assert reg.get_with_fallback("helpfulness", "fr") is r_en

    def test_get_with_fallback_custom_fallback(self):
        reg = RubricLocaleRegistry()
        r_es = self._make_rubric("helpfulness", "es")
        reg.register(r_es)
        assert reg.get_with_fallback("helpfulness", "fr", fallback="es") is r_es

    def test_get_with_fallback_both_missing(self):
        reg = RubricLocaleRegistry()
        assert reg.get_with_fallback("helpfulness", "fr") is None

    def test_list_locales(self):
        reg = RubricLocaleRegistry()
        reg.register(self._make_rubric("safety", "en"))
        reg.register(self._make_rubric("safety", "ja"))
        reg.register(self._make_rubric("safety", "es"))
        assert reg.list_locales("safety") == ["en", "es", "ja"]

    def test_list_locales_missing_metric(self):
        reg = RubricLocaleRegistry()
        assert reg.list_locales("nonexistent") == []

    def test_list_metrics_all(self):
        reg = RubricLocaleRegistry()
        reg.register(self._make_rubric("safety", "en"))
        reg.register(self._make_rubric("helpfulness", "en"))
        assert reg.list_metrics() == ["helpfulness", "safety"]

    def test_list_metrics_by_locale(self):
        reg = RubricLocaleRegistry()
        reg.register(self._make_rubric("safety", "en"))
        reg.register(self._make_rubric("safety", "ja"))
        reg.register(self._make_rubric("helpfulness", "en"))
        assert reg.list_metrics(locale="ja") == ["safety"]

    def test_list_metrics_locale_no_match(self):
        reg = RubricLocaleRegistry()
        reg.register(self._make_rubric("safety", "en"))
        assert reg.list_metrics(locale="ko") == []

    def test_get_all(self):
        reg = RubricLocaleRegistry()
        r1 = self._make_rubric("safety", "en")
        r2 = self._make_rubric("safety", "ja")
        r3 = self._make_rubric("helpfulness", "en")
        reg.register(r1)
        reg.register(r2)
        reg.register(r3)
        all_rubrics = reg.get_all()
        assert len(all_rubrics) == 3
        assert r1 in all_rubrics
        assert r2 in all_rubrics
        assert r3 in all_rubrics

    def test_get_all_empty(self):
        reg = RubricLocaleRegistry()
        assert reg.get_all() == []

    def test_register_overwrites_same_metric_locale(self):
        reg = RubricLocaleRegistry()
        r1 = self._make_rubric("safety", "en")
        r2 = LocalizedRubric("safety", "en", "New prompt", {}, "New desc")
        reg.register(r1)
        reg.register(r2)
        assert reg.get("safety", "en") is r2
        assert len(reg.get_all()) == 1


# ---------------------------------------------------------------------------
# detect_output_language
# ---------------------------------------------------------------------------


class TestDetectOutputLanguage:
    def test_english(self):
        assert detect_output_language("Hello, this is an English sentence.") == "en"

    def test_japanese_hiragana(self):
        assert detect_output_language("これは日本語のテストです。") == "ja"

    def test_japanese_katakana(self):
        assert detect_output_language("カタカナテスト") == "ja"

    def test_japanese_kanji(self):
        assert detect_output_language("漢字テスト") == "ja"

    def test_korean(self):
        assert detect_output_language("이것은 한국어 테스트입니다.") == "ko"

    def test_russian(self):
        assert detect_output_language("Это тест на русском языке.") == "ru"

    def test_arabic(self):
        assert detect_output_language("هذا اختبار باللغة العربية.") == "ar"

    def test_hindi(self):
        assert detect_output_language("यह हिंदी में एक परीक्षण है।") == "hi"

    def test_empty_string(self):
        assert detect_output_language("") == "en"

    def test_numbers_only(self):
        assert detect_output_language("12345") == "en"

    def test_mixed_with_majority_japanese(self):
        text = "Hello これは日本語のテストですね"
        assert detect_output_language(text) == "ja"

    def test_mixed_with_majority_korean(self):
        text = "Hello 이것은 한국어 테스트입니다"
        assert detect_output_language(text) == "ko"

    def test_cyrillic_characters(self):
        assert detect_output_language("АБВГД") == "ru"

    def test_punctuation_only(self):
        assert detect_output_language("...!!!???") == "en"


# ---------------------------------------------------------------------------
# match_rubric_language
# ---------------------------------------------------------------------------


class TestMatchRubricLanguage:
    def _make_registry(self) -> RubricLocaleRegistry:
        reg = RubricLocaleRegistry()
        reg.register(LocalizedRubric("helpfulness", "en", "Rate.", {"5": "Good"}, "English"))
        reg.register(LocalizedRubric("helpfulness", "ja", "評価.", {"5": "良い"}, "Japanese"))
        reg.register(LocalizedRubric("helpfulness", "ko", "평가.", {"5": "좋음"}, "Korean"))
        return reg

    def test_match_english(self):
        reg = self._make_registry()
        result = match_rubric_language("helpfulness", "This is English text.", reg)
        assert result is not None
        assert result.locale == "en"

    def test_match_japanese(self):
        reg = self._make_registry()
        result = match_rubric_language("helpfulness", "これは日本語です。", reg)
        assert result is not None
        assert result.locale == "ja"

    def test_match_korean(self):
        reg = self._make_registry()
        result = match_rubric_language("helpfulness", "이것은 한국어입니다.", reg)
        assert result is not None
        assert result.locale == "ko"

    def test_fallback_to_english(self):
        reg = self._make_registry()
        # Russian text but no Russian rubric, should fall back to English
        result = match_rubric_language("helpfulness", "Это русский текст.", reg)
        assert result is not None
        assert result.locale == "en"

    def test_no_match_at_all(self):
        reg = RubricLocaleRegistry()
        result = match_rubric_language("nonexistent", "Hello world", reg)
        assert result is None

    def test_missing_metric(self):
        reg = self._make_registry()
        result = match_rubric_language("nonexistent", "これは日本語です。", reg)
        assert result is None


# ---------------------------------------------------------------------------
# DEFAULT_LOCALE_REGISTRY
# ---------------------------------------------------------------------------


class TestDefaultLocaleRegistry:
    def test_has_three_metrics(self):
        metrics = DEFAULT_LOCALE_REGISTRY.list_metrics()
        assert set(metrics) == {"helpfulness", "safety", "accuracy"}

    def test_each_metric_has_three_locales(self):
        for metric in ["helpfulness", "safety", "accuracy"]:
            locales = DEFAULT_LOCALE_REGISTRY.list_locales(metric)
            assert set(locales) == {"en", "ja", "es"}

    def test_total_rubric_count(self):
        assert len(DEFAULT_LOCALE_REGISTRY.get_all()) == 9

    def test_english_rubrics_have_content(self):
        for metric in ["helpfulness", "safety", "accuracy"]:
            r = DEFAULT_LOCALE_REGISTRY.get(metric, "en")
            assert r is not None
            assert len(r.prompt) > 10
            assert len(r.rubric_levels) == 5
            assert len(r.description) > 5

    def test_japanese_rubrics_contain_japanese(self):
        for metric in ["helpfulness", "safety", "accuracy"]:
            r = DEFAULT_LOCALE_REGISTRY.get(metric, "ja")
            assert r is not None
            # Check for CJK/Hiragana characters
            has_ja = any(
                (0x3040 <= ord(c) <= 0x309F) or (0x30A0 <= ord(c) <= 0x30FF) or (0x4E00 <= ord(c) <= 0x9FFF)
                for c in r.prompt
            )
            assert has_ja, f"Japanese rubric for {metric} should contain Japanese characters"

    def test_spanish_rubrics_contain_spanish_words(self):
        for metric in ["helpfulness", "safety", "accuracy"]:
            r = DEFAULT_LOCALE_REGISTRY.get(metric, "es")
            assert r is not None
            # Spanish rubrics should have Spanish text (check for common Spanish words)
            prompt_lower = r.prompt.lower()
            assert any(w in prompt_lower for w in ["evalua", "respuesta", "precision"]), (
                f"Spanish rubric for {metric} should contain Spanish text"
            )


# ---------------------------------------------------------------------------
# format_locale_summary
# ---------------------------------------------------------------------------


class TestFormatLocaleSummary:
    def test_empty_registry(self):
        reg = RubricLocaleRegistry()
        assert format_locale_summary(reg) == "No rubrics registered."

    def test_has_header(self):
        summary = format_locale_summary(DEFAULT_LOCALE_REGISTRY)
        assert "Metric" in summary

    def test_has_all_metrics(self):
        summary = format_locale_summary(DEFAULT_LOCALE_REGISTRY)
        for metric in ["helpfulness", "safety", "accuracy"]:
            assert metric in summary

    def test_has_all_locales(self):
        summary = format_locale_summary(DEFAULT_LOCALE_REGISTRY)
        for locale in ["en", "ja", "es"]:
            assert locale in summary

    def test_has_markers(self):
        summary = format_locale_summary(DEFAULT_LOCALE_REGISTRY)
        assert "x" in summary

    def test_single_rubric(self):
        reg = RubricLocaleRegistry()
        reg.register(LocalizedRubric("test", "en", "p", {}, "d"))
        summary = format_locale_summary(reg)
        assert "test" in summary
        assert "en" in summary
