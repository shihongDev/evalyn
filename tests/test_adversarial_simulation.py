"""Tests for adversarial input simulation module."""

from __future__ import annotations

from evalyn_sdk.adversarial_simulation import (
    CATEGORIES,
    AdversarialCategory,
    AdversarialInput,
    AdversarialSuite,
    filter_by_category,
    format_adversarial_report,
    generate_boundary_inputs,
    generate_contradiction_inputs,
    generate_full_suite,
    generate_jailbreak_patterns,
    generate_prompt_injections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_category(**overrides) -> AdversarialCategory:
    defaults = {
        "category_id": "test_cat",
        "name": "Test Category",
        "description": "A test category",
        "severity": "medium",
    }
    defaults.update(overrides)
    return AdversarialCategory(**defaults)


def _make_input(**overrides) -> AdversarialInput:
    defaults = {
        "input_text": "test input",
        "category_id": "test_cat",
        "variant": "basic",
        "description": "a test input",
        "tags": ["test"],
    }
    defaults.update(overrides)
    return AdversarialInput(**defaults)


# ---------------------------------------------------------------------------
# AdversarialCategory
# ---------------------------------------------------------------------------


class TestAdversarialCategory:
    def test_as_dict_roundtrip(self):
        c = _make_category()
        d = c.as_dict()
        c2 = AdversarialCategory.from_dict(d)
        assert c2.category_id == c.category_id
        assert c2.name == c.name
        assert c2.description == c.description
        assert c2.severity == c.severity

    def test_as_dict_fields(self):
        c = _make_category(severity="critical")
        d = c.as_dict()
        assert d["severity"] == "critical"
        assert d["category_id"] == "test_cat"

    def test_from_dict(self):
        c = AdversarialCategory.from_dict(
            {
                "category_id": "x",
                "name": "X",
                "description": "desc",
                "severity": "high",
            }
        )
        assert c.category_id == "x"
        assert c.severity == "high"


# ---------------------------------------------------------------------------
# AdversarialInput
# ---------------------------------------------------------------------------


class TestAdversarialInput:
    def test_as_dict_roundtrip(self):
        inp = _make_input()
        d = inp.as_dict()
        inp2 = AdversarialInput.from_dict(d)
        assert inp2.input_text == inp.input_text
        assert inp2.category_id == inp.category_id
        assert inp2.variant == inp.variant
        assert inp2.tags == inp.tags

    def test_as_dict_fields(self):
        inp = _make_input(input_text="hello", tags=["a", "b"])
        d = inp.as_dict()
        assert d["input_text"] == "hello"
        assert d["tags"] == ["a", "b"]

    def test_from_dict_default_tags(self):
        inp = AdversarialInput.from_dict(
            {
                "input_text": "text",
                "category_id": "cat",
                "variant": "v",
                "description": "d",
            }
        )
        assert inp.tags == []

    def test_tags_default_empty(self):
        inp = AdversarialInput(
            input_text="t", category_id="c", variant="v", description="d"
        )
        assert inp.tags == []


# ---------------------------------------------------------------------------
# AdversarialSuite
# ---------------------------------------------------------------------------


class TestAdversarialSuite:
    def test_as_dict_roundtrip(self):
        inputs = [_make_input(), _make_input(variant="v2")]
        suite = AdversarialSuite(
            inputs=inputs, categories=["test_cat"], total_count=2
        )
        d = suite.as_dict()
        s2 = AdversarialSuite.from_dict(d)
        assert s2.total_count == 2
        assert len(s2.inputs) == 2
        assert s2.categories == ["test_cat"]

    def test_as_dict_serializes_inputs(self):
        suite = AdversarialSuite(
            inputs=[_make_input()], categories=["test_cat"], total_count=1
        )
        d = suite.as_dict()
        assert isinstance(d["inputs"], list)
        assert d["inputs"][0]["input_text"] == "test input"

    def test_from_dict_empty(self):
        suite = AdversarialSuite.from_dict(
            {"inputs": [], "categories": [], "total_count": 0}
        )
        assert suite.inputs == []
        assert suite.total_count == 0

    def test_from_dict_defaults(self):
        suite = AdversarialSuite.from_dict({})
        assert suite.inputs == []
        assert suite.categories == []
        assert suite.total_count == 0


# ---------------------------------------------------------------------------
# CATEGORIES
# ---------------------------------------------------------------------------


class TestCategories:
    def test_four_categories_exist(self):
        assert len(CATEGORIES) == 4

    def test_prompt_injection_exists(self):
        assert "prompt_injection" in CATEGORIES
        assert CATEGORIES["prompt_injection"].severity == "critical"

    def test_boundary_exists(self):
        assert "boundary" in CATEGORIES
        assert CATEGORIES["boundary"].severity == "medium"

    def test_contradiction_exists(self):
        assert "contradiction" in CATEGORIES
        assert CATEGORIES["contradiction"].severity == "high"

    def test_jailbreak_exists(self):
        assert "jailbreak" in CATEGORIES
        assert CATEGORIES["jailbreak"].severity == "critical"

    def test_all_have_descriptions(self):
        for cat in CATEGORIES.values():
            assert cat.description


# ---------------------------------------------------------------------------
# generate_prompt_injections
# ---------------------------------------------------------------------------


class TestGeneratePromptInjections:
    def test_minimum_count(self):
        results = generate_prompt_injections()
        assert len(results) >= 5

    def test_all_have_correct_category(self):
        results = generate_prompt_injections()
        for inp in results:
            assert inp.category_id == "prompt_injection"

    def test_ignore_previous_variant(self):
        results = generate_prompt_injections()
        variants = [r.variant for r in results]
        assert "ignore_previous" in variants

    def test_role_override_variant(self):
        results = generate_prompt_injections()
        variants = [r.variant for r in results]
        assert "role_override" in variants

    def test_delimiter_escape_variant(self):
        results = generate_prompt_injections()
        variants = [r.variant for r in results]
        assert "delimiter_escape" in variants

    def test_encoding_trick_variant(self):
        results = generate_prompt_injections()
        variants = [r.variant for r in results]
        assert "encoding_trick" in variants

    def test_nested_instructions_variant(self):
        results = generate_prompt_injections()
        variants = [r.variant for r in results]
        assert "nested_instructions" in variants

    def test_with_system_prompt_adds_reference(self):
        results_without = generate_prompt_injections()
        results_with = generate_prompt_injections(system_prompt="You are a helpful assistant")
        assert len(results_with) > len(results_without)
        variants = [r.variant for r in results_with]
        assert "prompt_reference" in variants

    def test_all_have_tags(self):
        results = generate_prompt_injections()
        for inp in results:
            assert len(inp.tags) > 0

    def test_all_have_description(self):
        results = generate_prompt_injections()
        for inp in results:
            assert inp.description


# ---------------------------------------------------------------------------
# generate_boundary_inputs
# ---------------------------------------------------------------------------


class TestGenerateBoundaryInputs:
    def test_returns_inputs(self):
        results = generate_boundary_inputs()
        assert len(results) >= 5

    def test_empty_string_included(self):
        results = generate_boundary_inputs()
        variants = [r.variant for r in results]
        assert "empty" in variants
        empty = [r for r in results if r.variant == "empty"][0]
        assert empty.input_text == ""

    def test_single_char_included(self):
        results = generate_boundary_inputs()
        variants = [r.variant for r in results]
        assert "single_char" in variants

    def test_max_length_included(self):
        results = generate_boundary_inputs()
        max_len = [r for r in results if r.variant == "max_length"][0]
        assert len(max_len.input_text) == 10000

    def test_special_chars_included(self):
        results = generate_boundary_inputs()
        variants = [r.variant for r in results]
        assert "special_chars" in variants

    def test_cjk_included(self):
        results = generate_boundary_inputs()
        variants = [r.variant for r in results]
        assert "cjk" in variants

    def test_emoji_placeholder_included(self):
        results = generate_boundary_inputs()
        variants = [r.variant for r in results]
        assert "emoji_placeholder" in variants
        emoji_inp = [r for r in results if r.variant == "emoji_placeholder"][0]
        assert "[emoji]" in emoji_inp.input_text

    def test_rtl_markers_included(self):
        results = generate_boundary_inputs()
        variants = [r.variant for r in results]
        assert "rtl_markers" in variants

    def test_zero_width_included(self):
        results = generate_boundary_inputs()
        variants = [r.variant for r in results]
        assert "zero_width" in variants
        zw = [r for r in results if r.variant == "zero_width"][0]
        assert "\u200b" in zw.input_text

    def test_all_boundary_category(self):
        results = generate_boundary_inputs()
        for inp in results:
            assert inp.category_id == "boundary"


# ---------------------------------------------------------------------------
# generate_contradiction_inputs
# ---------------------------------------------------------------------------


class TestGenerateContradictionInputs:
    def test_generic_contradictions(self):
        results = generate_contradiction_inputs()
        assert len(results) >= 5

    def test_all_contradiction_category(self):
        results = generate_contradiction_inputs()
        for inp in results:
            assert inp.category_id == "contradiction"

    def test_with_system_prompt_adds_specific(self):
        results_generic = generate_contradiction_inputs()
        results_specific = generate_contradiction_inputs(
            system_prompt="Be helpful and concise in English"
        )
        assert len(results_specific) > len(results_generic)

    def test_specific_matches_keywords(self):
        results = generate_contradiction_inputs(
            system_prompt="Always be helpful and polite"
        )
        variants = [r.variant for r in results]
        assert "specific_helpful" in variants
        assert "specific_polite" in variants

    def test_generic_tags(self):
        results = generate_contradiction_inputs()
        for inp in results:
            assert "contradiction" in inp.tags

    def test_no_specific_without_matching_keywords(self):
        results = generate_contradiction_inputs(
            system_prompt="Do the thing xyz"
        )
        specific = [r for r in results if "specific" in r.variant]
        assert len(specific) == 0


# ---------------------------------------------------------------------------
# generate_jailbreak_patterns
# ---------------------------------------------------------------------------


class TestGenerateJailbreakPatterns:
    def test_returns_patterns(self):
        results = generate_jailbreak_patterns()
        assert len(results) >= 4

    def test_dan_pattern(self):
        results = generate_jailbreak_patterns()
        variants = [r.variant for r in results]
        assert "dan" in variants
        dan = [r for r in results if r.variant == "dan"][0]
        assert "DAN" in dan.input_text

    def test_hypothetical_pattern(self):
        results = generate_jailbreak_patterns()
        variants = [r.variant for r in results]
        assert "hypothetical" in variants

    def test_roleplay_pattern(self):
        results = generate_jailbreak_patterns()
        variants = [r.variant for r in results]
        assert "roleplay" in variants

    def test_base64_pattern(self):
        results = generate_jailbreak_patterns()
        variants = [r.variant for r in results]
        assert "base64_instruction" in variants

    def test_all_jailbreak_category(self):
        results = generate_jailbreak_patterns()
        for inp in results:
            assert inp.category_id == "jailbreak"

    def test_all_have_tags(self):
        results = generate_jailbreak_patterns()
        for inp in results:
            assert len(inp.tags) > 0


# ---------------------------------------------------------------------------
# generate_full_suite
# ---------------------------------------------------------------------------


class TestGenerateFullSuite:
    def test_suite_has_all_categories(self):
        suite = generate_full_suite()
        assert "prompt_injection" in suite.categories
        assert "boundary" in suite.categories
        assert "contradiction" in suite.categories
        assert "jailbreak" in suite.categories

    def test_total_count_matches(self):
        suite = generate_full_suite()
        assert suite.total_count == len(suite.inputs)

    def test_total_count_reasonable(self):
        suite = generate_full_suite()
        assert suite.total_count >= 20

    def test_with_system_prompt(self):
        suite_no_prompt = generate_full_suite()
        suite_with_prompt = generate_full_suite(
            system_prompt="Be helpful, concise, and safe"
        )
        assert suite_with_prompt.total_count >= suite_no_prompt.total_count

    def test_categories_sorted(self):
        suite = generate_full_suite()
        assert suite.categories == sorted(suite.categories)


# ---------------------------------------------------------------------------
# filter_by_category
# ---------------------------------------------------------------------------


class TestFilterByCategory:
    def test_filter_prompt_injection(self):
        suite = generate_full_suite()
        filtered = filter_by_category(suite, "prompt_injection")
        for inp in filtered:
            assert inp.category_id == "prompt_injection"
        assert len(filtered) >= 5

    def test_filter_boundary(self):
        suite = generate_full_suite()
        filtered = filter_by_category(suite, "boundary")
        for inp in filtered:
            assert inp.category_id == "boundary"

    def test_filter_nonexistent_empty(self):
        suite = generate_full_suite()
        filtered = filter_by_category(suite, "nonexistent")
        assert filtered == []

    def test_filter_all_categories_cover_total(self):
        suite = generate_full_suite()
        total = 0
        for cat_id in suite.categories:
            total += len(filter_by_category(suite, cat_id))
        assert total == suite.total_count


# ---------------------------------------------------------------------------
# format_adversarial_report
# ---------------------------------------------------------------------------


class TestFormatAdversarialReport:
    def test_report_contains_title(self):
        suite = generate_full_suite()
        report = format_adversarial_report(suite)
        assert "Adversarial Test Suite Report" in report

    def test_report_contains_total(self):
        suite = generate_full_suite()
        report = format_adversarial_report(suite)
        assert f"Total inputs: {suite.total_count}" in report

    def test_report_contains_categories(self):
        suite = generate_full_suite()
        report = format_adversarial_report(suite)
        assert "Prompt Injection" in report
        assert "Boundary" in report
        assert "Contradiction" in report
        assert "Jailbreak" in report

    def test_report_contains_severity(self):
        suite = generate_full_suite()
        report = format_adversarial_report(suite)
        assert "critical" in report
        assert "medium" in report
        assert "high" in report

    def test_report_contains_counts(self):
        suite = generate_full_suite()
        report = format_adversarial_report(suite)
        assert "inputs" in report

    def test_empty_suite_report(self):
        suite = AdversarialSuite(inputs=[], categories=[], total_count=0)
        report = format_adversarial_report(suite)
        assert "Total inputs: 0" in report
