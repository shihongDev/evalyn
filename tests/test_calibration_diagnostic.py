"""Tests for calibration diagnostic report module."""

from evalyn_sdk.calibration.diagnostic_report import (
    CalibrationDiagnostic,
    DiagnosticFinding,
    compute_calibration_delta,
    diagnose_calibration,
    generate_summary,
    identify_regression_items,
)


# ---------------------------------------------------------------------------
# diagnose_calibration - improvement
# ---------------------------------------------------------------------------


def test_diagnose_calibration_all_improved():
    before = {"i1": 0.2, "i2": 0.3}
    after = {"i1": 0.9, "i2": 0.95}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    assert diag.metric_id == "m1"
    assert diag.delta > 0
    assert len(diag.improved_items) == 2
    assert len(diag.degraded_items) == 0


def test_diagnose_calibration_improvement_findings():
    before = {"i1": 0.2, "i2": 0.3}
    after = {"i1": 0.9, "i2": 0.95}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    categories = [f.category for f in diag.findings]
    assert "improvement" in categories


# ---------------------------------------------------------------------------
# diagnose_calibration - degradation
# ---------------------------------------------------------------------------


def test_diagnose_calibration_all_degraded():
    before = {"i1": 0.9, "i2": 0.8}
    after = {"i1": 0.1, "i2": 0.2}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    assert diag.delta < 0
    assert len(diag.degraded_items) == 2
    assert len(diag.improved_items) == 0


def test_diagnose_calibration_degradation_severity():
    before = {"i1": 0.9, "i2": 0.8, "i3": 0.7}
    after = {"i1": 0.1, "i2": 0.1, "i3": 0.1}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    deg_findings = [f for f in diag.findings if f.category == "degradation"]
    assert len(deg_findings) == 1
    # more degraded than improved -> critical
    assert deg_findings[0].severity == "critical"


# ---------------------------------------------------------------------------
# diagnose_calibration - mixed
# ---------------------------------------------------------------------------


def test_diagnose_calibration_mixed():
    before = {"i1": 0.2, "i2": 0.9, "i3": 0.5}
    after = {"i1": 0.9, "i2": 0.1, "i3": 0.5}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    assert len(diag.improved_items) >= 1
    assert len(diag.degraded_items) >= 1


def test_diagnose_calibration_mixed_degradation_warning():
    # 1 improved, 1 degraded -> degradation severity = warning (not critical)
    before = {"i1": 0.2, "i2": 0.9}
    after = {"i1": 0.9, "i2": 0.1}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    deg_findings = [f for f in diag.findings if f.category == "degradation"]
    assert len(deg_findings) == 1
    assert deg_findings[0].severity == "warning"


# ---------------------------------------------------------------------------
# diagnose_calibration - edge cases
# ---------------------------------------------------------------------------


def test_diagnose_calibration_empty_scores():
    diag = diagnose_calibration("m1", {}, {})
    assert diag.before_score == 0.0
    assert diag.after_score == 0.0
    assert diag.delta == 0.0
    assert diag.improved_items == []
    assert diag.degraded_items == []


def test_diagnose_calibration_identical_before_after():
    before = {"i1": 0.5, "i2": 0.5}
    after = {"i1": 0.5, "i2": 0.5}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    assert diag.delta == 0.0
    assert len(diag.improved_items) == 0
    assert len(diag.degraded_items) == 0


def test_diagnose_calibration_no_common_items():
    before = {"i1": 0.5}
    after = {"i2": 0.5}
    diag = diagnose_calibration("m1", before, after)
    assert diag.improved_items == []
    assert diag.degraded_items == []


def test_diagnose_calibration_largest_improvement_finding():
    before = {"i1": 0.1, "i2": 0.1}
    after = {"i1": 0.9, "i2": 0.5}
    diag = diagnose_calibration("m1", before, after, threshold=0.1)
    largest_findings = [
        f for f in diag.findings if "largest improvement" in f.description
    ]
    assert len(largest_findings) == 1
    assert "i1" in largest_findings[0].description


# ---------------------------------------------------------------------------
# identify_regression_items
# ---------------------------------------------------------------------------


def test_identify_regression_items_basic():
    before = {"i1": 0.8, "i2": 0.5, "i3": 0.3}
    after = {"i1": 0.2, "i2": 0.5, "i3": 0.9}
    regressions = identify_regression_items(before, after, threshold=0.01)
    assert regressions == ["i1"]


def test_identify_regression_items_threshold():
    before = {"i1": 0.50}
    after = {"i1": 0.49}
    # delta = -0.01, threshold = 0.01, after (0.49) < before (0.50) - 0.01 (0.49) is false
    regressions = identify_regression_items(before, after, threshold=0.01)
    assert regressions == []


def test_identify_regression_items_empty():
    regressions = identify_regression_items({}, {})
    assert regressions == []


def test_identify_regression_items_all_regressed():
    before = {"i1": 0.9, "i2": 0.8}
    after = {"i1": 0.1, "i2": 0.1}
    regressions = identify_regression_items(before, after, threshold=0.01)
    assert regressions == ["i1", "i2"]


# ---------------------------------------------------------------------------
# compute_calibration_delta
# ---------------------------------------------------------------------------


def test_compute_calibration_delta_basic():
    before = {"i1": 0.3, "i2": 0.5}
    after = {"i1": 0.7, "i2": 0.4}
    deltas = compute_calibration_delta(before, after)
    assert abs(deltas["i1"] - 0.4) < 1e-9
    assert abs(deltas["i2"] - (-0.1)) < 1e-9


def test_compute_calibration_delta_only_common():
    before = {"i1": 0.3, "i2": 0.5}
    after = {"i1": 0.7, "i3": 0.9}
    deltas = compute_calibration_delta(before, after)
    assert "i1" in deltas
    assert "i2" not in deltas
    assert "i3" not in deltas


def test_compute_calibration_delta_empty():
    deltas = compute_calibration_delta({}, {})
    assert deltas == {}


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


def test_generate_summary_improved():
    diag = CalibrationDiagnostic(
        metric_id="m1",
        before_score=0.3,
        after_score=0.8,
        delta=0.5,
        improved_items=["i1", "i2"],
        degraded_items=[],
    )
    summary = generate_summary(diag)
    assert "m1" in summary
    assert "2 items improved" in summary
    assert "none degraded" in summary


def test_generate_summary_degraded():
    diag = CalibrationDiagnostic(
        metric_id="m1",
        before_score=0.8,
        after_score=0.3,
        delta=-0.5,
        improved_items=[],
        degraded_items=["i1", "i2"],
    )
    summary = generate_summary(diag)
    assert "2 items degraded" in summary
    assert "none improved" in summary


def test_generate_summary_mixed():
    diag = CalibrationDiagnostic(
        metric_id="m1",
        before_score=0.5,
        after_score=0.6,
        delta=0.1,
        improved_items=["i1"],
        degraded_items=["i2"],
    )
    summary = generate_summary(diag)
    assert "1 improved" in summary
    assert "1 degraded" in summary


def test_generate_summary_no_changes():
    diag = CalibrationDiagnostic(
        metric_id="m1",
        before_score=0.5,
        after_score=0.5,
        delta=0.0,
        improved_items=[],
        degraded_items=[],
    )
    summary = generate_summary(diag)
    assert "no significant changes" in summary


# ---------------------------------------------------------------------------
# DiagnosticFinding
# ---------------------------------------------------------------------------


def test_diagnostic_finding_defaults():
    f = DiagnosticFinding(category="improvement", description="test")
    assert f.severity == "info"
    assert f.affected_items == []


def test_diagnostic_finding_as_dict():
    f = DiagnosticFinding(
        category="degradation",
        description="3 items degraded",
        severity="warning",
        affected_items=["i1", "i2", "i3"],
    )
    d = f.as_dict()
    assert d["category"] == "degradation"
    assert d["severity"] == "warning"
    assert len(d["affected_items"]) == 3


# ---------------------------------------------------------------------------
# CalibrationDiagnostic format_text
# ---------------------------------------------------------------------------


def test_calibration_diagnostic_format_text():
    before = {"i1": 0.2, "i2": 0.3}
    after = {"i1": 0.9, "i2": 0.95}
    diag = diagnose_calibration("quality", before, after, threshold=0.1)
    text = diag.format_text()
    assert "quality" in text
    assert "Score" in text
    assert "Improved items" in text


def test_calibration_diagnostic_format_text_degraded():
    before = {"i1": 0.9}
    after = {"i1": 0.1}
    diag = diagnose_calibration("quality", before, after, threshold=0.1)
    text = diag.format_text()
    assert "Degraded items" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


def test_calibration_diagnostic_roundtrip():
    before = {"i1": 0.2, "i2": 0.3}
    after = {"i1": 0.9, "i2": 0.95}
    original = diagnose_calibration("m1", before, after, threshold=0.1)
    d = original.as_dict()
    restored = CalibrationDiagnostic.from_dict(d)
    assert restored.metric_id == original.metric_id
    assert restored.before_score == original.before_score
    assert restored.after_score == original.after_score
    assert restored.delta == original.delta
    assert len(restored.findings) == len(original.findings)
    assert restored.improved_items == original.improved_items
    assert restored.degraded_items == original.degraded_items
    assert restored.summary == original.summary


def test_calibration_diagnostic_roundtrip_empty():
    original = diagnose_calibration("m1", {}, {})
    d = original.as_dict()
    restored = CalibrationDiagnostic.from_dict(d)
    assert restored.before_score == 0.0
    assert restored.findings == []
    assert restored.improved_items == []
