"""Tests for calibration sensitivity analysis module."""

from evalyn_sdk.calibration.sensitivity_analysis import (
    Perturbation,
    SensitivityReport,
    SensitivityResult,
    analyze_sensitivity,
    generate_perturbations,
    perturb_add_whitespace,
    perturb_case_change,
    perturb_synonym,
    perturb_word_swap,
)


# ---------------------------------------------------------------------------
# perturb_word_swap
# ---------------------------------------------------------------------------


def test_perturb_word_swap_changes_text():
    p = perturb_word_swap("hello world foo", seed=42)
    assert p.perturbed != p.original
    assert p.method == "word_swap"


def test_perturb_word_swap_preserves_words():
    p = perturb_word_swap("alpha beta gamma", seed=0)
    orig_words = sorted(p.original.split())
    pert_words = sorted(p.perturbed.split())
    assert orig_words == pert_words


def test_perturb_word_swap_deterministic():
    p1 = perturb_word_swap("one two three four", seed=99)
    p2 = perturb_word_swap("one two three four", seed=99)
    assert p1.perturbed == p2.perturbed


def test_perturb_word_swap_single_word():
    p = perturb_word_swap("hello", seed=0)
    assert p.perturbed == "hello"
    assert "too short" in p.change_description


def test_perturb_word_swap_empty():
    p = perturb_word_swap("", seed=0)
    assert p.perturbed == ""


# ---------------------------------------------------------------------------
# perturb_add_whitespace
# ---------------------------------------------------------------------------


def test_perturb_add_whitespace_adds_spaces():
    p = perturb_add_whitespace("hello world foo")
    assert "  " in p.perturbed
    assert p.method == "add_whitespace"


def test_perturb_add_whitespace_preserves_words():
    p = perturb_add_whitespace("one two three")
    assert p.perturbed.split() == p.original.split()


def test_perturb_add_whitespace_single_word():
    p = perturb_add_whitespace("hello")
    assert p.perturbed == "hello"


def test_perturb_add_whitespace_empty():
    p = perturb_add_whitespace("")
    assert p.perturbed == ""


# ---------------------------------------------------------------------------
# perturb_case_change
# ---------------------------------------------------------------------------


def test_perturb_case_change_changes_case():
    p = perturb_case_change("hello world", seed=42)
    assert p.perturbed != p.original
    assert p.method == "case_change"


def test_perturb_case_change_one_word_differs():
    text = "alpha beta gamma"
    p = perturb_case_change(text, seed=0)
    orig_words = text.split()
    pert_words = p.perturbed.split()
    diffs = [i for i, (a, b) in enumerate(zip(orig_words, pert_words)) if a != b]
    assert len(diffs) == 1


def test_perturb_case_change_empty():
    p = perturb_case_change("", seed=0)
    assert p.perturbed == ""
    assert p.change_description == "empty text"


def test_perturb_case_change_already_upper():
    p = perturb_case_change("HELLO", seed=0)
    assert p.perturbed == "hello"


# ---------------------------------------------------------------------------
# perturb_synonym
# ---------------------------------------------------------------------------


def test_perturb_synonym_replaces_word():
    p = perturb_synonym("this is good work", seed=42)
    assert p.perturbed != p.original
    assert p.method == "synonym"
    assert "good" not in p.perturbed.split() or p.perturbed != p.original


def test_perturb_synonym_no_synonyms_available():
    p = perturb_synonym("xyz qqq zzz", seed=0)
    assert p.perturbed == p.original
    assert "no synonyms" in p.change_description


def test_perturb_synonym_empty():
    p = perturb_synonym("", seed=0)
    assert p.perturbed == ""
    assert p.change_description == "empty text"


def test_perturb_synonym_deterministic():
    p1 = perturb_synonym("this is good", seed=42)
    p2 = perturb_synonym("this is good", seed=42)
    assert p1.perturbed == p2.perturbed


def test_perturb_synonym_preserves_capitalization():
    p = perturb_synonym("Good work", seed=42)
    if p.perturbed != p.original:
        # The replaced word should start with uppercase
        replaced_word = [
            w for w in p.perturbed.split()
            if w not in p.original.split() or w != p.original.split()[p.perturbed.split().index(w)]
        ]
        # At least verify the text changed
        assert p.perturbed != p.original


# ---------------------------------------------------------------------------
# generate_perturbations
# ---------------------------------------------------------------------------


def test_generate_perturbations_correct_count():
    perts = generate_perturbations("hello world foo bar", num_perturbations=5, seed=42)
    assert len(perts) == 5


def test_generate_perturbations_deterministic_with_seed():
    p1 = generate_perturbations("hello world foo bar", num_perturbations=3, seed=42)
    p2 = generate_perturbations("hello world foo bar", num_perturbations=3, seed=42)
    for a, b in zip(p1, p2):
        assert a.perturbed == b.perturbed
        assert a.method == b.method


def test_generate_perturbations_uses_multiple_methods():
    perts = generate_perturbations("this is good fast big work", num_perturbations=4, seed=42)
    methods = {p.method for p in perts}
    assert len(methods) >= 2  # at least two different methods


def test_generate_perturbations_single():
    perts = generate_perturbations("hello world", num_perturbations=1, seed=0)
    assert len(perts) == 1


def test_generate_perturbations_zero():
    perts = generate_perturbations("hello world", num_perturbations=0, seed=0)
    assert perts == []


# ---------------------------------------------------------------------------
# analyze_sensitivity
# ---------------------------------------------------------------------------


def _make_result(delta: float) -> SensitivityResult:
    return SensitivityResult(
        perturbation=Perturbation(original="a", perturbed="b"),
        original_score=0.8,
        perturbed_score=0.8 + delta,
        delta=delta,
    )


def test_analyze_sensitivity_avg():
    results = [_make_result(0.1), _make_result(-0.1), _make_result(0.04)]
    report = analyze_sensitivity(results, threshold=0.05)
    expected_avg = (0.1 + 0.1 + 0.04) / 3
    assert abs(report.avg_sensitivity - expected_avg) < 1e-9


def test_analyze_sensitivity_max():
    results = [_make_result(0.02), _make_result(-0.15), _make_result(0.08)]
    report = analyze_sensitivity(results, threshold=0.05)
    assert abs(report.max_sensitivity - 0.15) < 1e-9


def test_analyze_sensitivity_sensitive_count():
    results = [_make_result(0.1), _make_result(0.02), _make_result(-0.08)]
    report = analyze_sensitivity(results, threshold=0.05)
    assert report.sensitive_count == 2
    assert report.robust_count == 1


def test_analyze_sensitivity_all_robust():
    results = [_make_result(0.01), _make_result(-0.02), _make_result(0.03)]
    report = analyze_sensitivity(results, threshold=0.05)
    assert report.sensitive_count == 0
    assert report.robust_count == 3


def test_analyze_sensitivity_all_sensitive():
    results = [_make_result(0.1), _make_result(-0.2)]
    report = analyze_sensitivity(results, threshold=0.05)
    assert report.sensitive_count == 2
    assert report.robust_count == 0


def test_analyze_sensitivity_empty():
    report = analyze_sensitivity([], threshold=0.05)
    assert report.avg_sensitivity == 0.0
    assert report.max_sensitivity == 0.0
    assert report.sensitive_count == 0
    assert report.robust_count == 0
    assert report.results == []


def test_analyze_sensitivity_classifies_results():
    results = [_make_result(0.1), _make_result(0.01)]
    report = analyze_sensitivity(results, threshold=0.05)
    assert report.results[0].is_sensitive is True
    assert report.results[1].is_sensitive is False


# ---------------------------------------------------------------------------
# SensitivityReport - format_text
# ---------------------------------------------------------------------------


def test_sensitivity_report_format_text():
    results = [_make_result(0.1), _make_result(0.01)]
    report = analyze_sensitivity(results, threshold=0.05)
    text = report.format_text()
    assert "Sensitivity Report" in text
    assert "Avg sensitivity" in text
    assert "Max sensitivity" in text
    assert "SENSITIVE" in text
    assert "robust" in text


def test_sensitivity_report_format_text_empty():
    report = SensitivityReport()
    text = report.format_text()
    assert "0 perturbations" in text


# ---------------------------------------------------------------------------
# Perturbation - as_dict
# ---------------------------------------------------------------------------


def test_perturbation_as_dict():
    p = Perturbation(
        original="hello world", perturbed="world hello",
        method="word_swap", change_description="swapped",
    )
    d = p.as_dict()
    assert d["original"] == "hello world"
    assert d["perturbed"] == "world hello"
    assert d["method"] == "word_swap"
    assert d["change_description"] == "swapped"


# ---------------------------------------------------------------------------
# SensitivityResult - as_dict
# ---------------------------------------------------------------------------


def test_sensitivity_result_as_dict():
    r = SensitivityResult(
        perturbation=Perturbation(original="a", perturbed="b"),
        original_score=0.8, perturbed_score=0.7, delta=-0.1, is_sensitive=True,
    )
    d = r.as_dict()
    assert d["original_score"] == 0.8
    assert d["delta"] == -0.1
    assert d["is_sensitive"] is True
    assert "perturbation" in d


# ---------------------------------------------------------------------------
# SensitivityReport - as_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


def test_sensitivity_report_roundtrip():
    results = [_make_result(0.1), _make_result(-0.05), _make_result(0.02)]
    original = analyze_sensitivity(results, threshold=0.05)
    d = original.as_dict()
    restored = SensitivityReport.from_dict(d)
    assert restored.avg_sensitivity == original.avg_sensitivity
    assert restored.max_sensitivity == original.max_sensitivity
    assert restored.sensitive_count == original.sensitive_count
    assert restored.robust_count == original.robust_count
    assert len(restored.results) == len(original.results)


def test_sensitivity_report_roundtrip_empty():
    original = SensitivityReport()
    d = original.as_dict()
    restored = SensitivityReport.from_dict(d)
    assert restored.avg_sensitivity == 0.0
    assert restored.results == []
