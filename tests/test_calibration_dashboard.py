"""Tests for calibration comparison dashboard module."""

from evalyn_sdk.calibration.comparison_dashboard import (
    CalibrationAttemptSummary,
    ComparisonDashboard,
    ComparisonEntry,
    build_comparison_entry,
    build_dashboard,
    rank_optimizers,
    render_dashboard_ascii,
)


# ---------------------------------------------------------------------------
# CalibrationAttemptSummary - as_dict / from_dict
# ---------------------------------------------------------------------------


def test_attempt_summary_as_dict():
    a = CalibrationAttemptSummary(
        attempt_id="a1", metric_id="m1", optimizer="opro", alignment_score=0.85,
        cost_tokens=100, prompt_length=50, timestamp="2026-01-01",
    )
    d = a.as_dict()
    assert d["attempt_id"] == "a1"
    assert d["metric_id"] == "m1"
    assert d["optimizer"] == "opro"
    assert d["alignment_score"] == 0.85
    assert d["cost_tokens"] == 100
    assert d["prompt_length"] == 50
    assert d["timestamp"] == "2026-01-01"


def test_attempt_summary_from_dict():
    d = {
        "attempt_id": "a1", "metric_id": "m1", "optimizer": "opro",
        "alignment_score": 0.9, "cost_tokens": 200,
    }
    a = CalibrationAttemptSummary.from_dict(d)
    assert a.attempt_id == "a1"
    assert a.alignment_score == 0.9
    assert a.cost_tokens == 200
    assert a.prompt_length == 0  # default
    assert a.timestamp == ""  # default


def test_attempt_summary_roundtrip():
    original = CalibrationAttemptSummary(
        attempt_id="a1", metric_id="m1", optimizer="ape",
        alignment_score=0.75, cost_tokens=300, prompt_length=80,
        timestamp="2026-03-01T12:00:00",
    )
    restored = CalibrationAttemptSummary.from_dict(original.as_dict())
    assert restored.attempt_id == original.attempt_id
    assert restored.alignment_score == original.alignment_score
    assert restored.cost_tokens == original.cost_tokens
    assert restored.prompt_length == original.prompt_length
    assert restored.timestamp == original.timestamp


def test_attempt_summary_defaults():
    a = CalibrationAttemptSummary(
        attempt_id="a1", metric_id="m1", optimizer="opro", alignment_score=0.5,
    )
    assert a.cost_tokens == 0
    assert a.prompt_length == 0
    assert a.timestamp == ""


# ---------------------------------------------------------------------------
# build_comparison_entry
# ---------------------------------------------------------------------------


def _make_attempt(
    attempt_id: str, metric_id: str, optimizer: str, score: float, tokens: int = 0,
) -> CalibrationAttemptSummary:
    return CalibrationAttemptSummary(
        attempt_id=attempt_id, metric_id=metric_id,
        optimizer=optimizer, alignment_score=score, cost_tokens=tokens,
    )


def test_build_comparison_entry_finds_best():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6),
        _make_attempt("a2", "m1", "ape", 0.9),
        _make_attempt("a3", "m1", "textgrad", 0.7),
    ]
    entry = build_comparison_entry("m1", attempts)
    assert entry.best_attempt_id == "a2"


def test_build_comparison_entry_improvement():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.5),  # baseline
        _make_attempt("a2", "m1", "ape", 0.8),
    ]
    entry = build_comparison_entry("m1", attempts)
    assert abs(entry.improvement_over_baseline - 0.3) < 1e-9


def test_build_comparison_entry_no_improvement():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.9),  # baseline is best
        _make_attempt("a2", "m1", "ape", 0.5),
    ]
    entry = build_comparison_entry("m1", attempts)
    assert entry.improvement_over_baseline == 0.0
    assert entry.best_attempt_id == "a1"


def test_build_comparison_entry_single_attempt():
    attempts = [_make_attempt("a1", "m1", "opro", 0.7)]
    entry = build_comparison_entry("m1", attempts)
    assert entry.best_attempt_id == "a1"
    assert entry.improvement_over_baseline == 0.0
    assert len(entry.attempts) == 1


def test_build_comparison_entry_empty():
    entry = build_comparison_entry("m1", [])
    assert entry.best_attempt_id == ""
    assert entry.improvement_over_baseline == 0.0
    assert entry.attempts == []


# ---------------------------------------------------------------------------
# build_dashboard
# ---------------------------------------------------------------------------


def test_build_dashboard_groups_by_metric():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6),
        _make_attempt("a2", "m2", "ape", 0.8),
        _make_attempt("a3", "m1", "ape", 0.9),
    ]
    dashboard = build_dashboard(attempts)
    assert dashboard.total_metrics == 2
    assert dashboard.total_attempts == 3
    metric_ids = [e.metric_id for e in dashboard.entries]
    assert "m1" in metric_ids
    assert "m2" in metric_ids


def test_build_dashboard_correct_entry_counts():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6),
        _make_attempt("a2", "m1", "ape", 0.8),
        _make_attempt("a3", "m2", "opro", 0.5),
    ]
    dashboard = build_dashboard(attempts)
    m1_entry = [e for e in dashboard.entries if e.metric_id == "m1"][0]
    m2_entry = [e for e in dashboard.entries if e.metric_id == "m2"][0]
    assert len(m1_entry.attempts) == 2
    assert len(m2_entry.attempts) == 1


def test_build_dashboard_empty():
    dashboard = build_dashboard([])
    assert dashboard.total_metrics == 0
    assert dashboard.total_attempts == 0
    assert dashboard.entries == []


def test_build_dashboard_sorted_metrics():
    attempts = [
        _make_attempt("a1", "z_metric", "opro", 0.5),
        _make_attempt("a2", "a_metric", "opro", 0.5),
    ]
    dashboard = build_dashboard(attempts)
    assert dashboard.entries[0].metric_id == "a_metric"
    assert dashboard.entries[1].metric_id == "z_metric"


# ---------------------------------------------------------------------------
# render_dashboard_ascii
# ---------------------------------------------------------------------------


def test_render_dashboard_ascii_has_table():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6, tokens=100),
        _make_attempt("a2", "m1", "ape", 0.9, tokens=200),
    ]
    dashboard = build_dashboard(attempts)
    table = render_dashboard_ascii(dashboard)
    assert "Metric" in table
    assert "Optimizer" in table
    assert "Score" in table
    assert "Tokens" in table


def test_render_dashboard_ascii_highlights_best():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6),
        _make_attempt("a2", "m1", "ape", 0.9),
    ]
    dashboard = build_dashboard(attempts)
    table = render_dashboard_ascii(dashboard)
    lines = table.split("\n")
    # The best attempt line should have an asterisk
    best_lines = [l for l in lines if "*" in l]
    assert len(best_lines) >= 1
    assert "ape" in best_lines[0]


def test_render_dashboard_ascii_empty():
    dashboard = build_dashboard([])
    table = render_dashboard_ascii(dashboard)
    assert "Metric" in table  # header still present


# ---------------------------------------------------------------------------
# rank_optimizers
# ---------------------------------------------------------------------------


def test_rank_optimizers_sorted():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6),
        _make_attempt("a2", "m1", "ape", 0.9),
        _make_attempt("a3", "m2", "opro", 0.7),
        _make_attempt("a4", "m2", "ape", 0.8),
    ]
    dashboard = build_dashboard(attempts)
    ranking = rank_optimizers(dashboard)
    assert ranking[0][0] == "ape"  # avg 0.85
    assert ranking[1][0] == "opro"  # avg 0.65
    assert ranking[0][1] > ranking[1][1]


def test_rank_optimizers_single_optimizer():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.7),
        _make_attempt("a2", "m2", "opro", 0.8),
    ]
    dashboard = build_dashboard(attempts)
    ranking = rank_optimizers(dashboard)
    assert len(ranking) == 1
    assert ranking[0][0] == "opro"
    assert abs(ranking[0][1] - 0.75) < 1e-9


def test_rank_optimizers_empty():
    dashboard = build_dashboard([])
    ranking = rank_optimizers(dashboard)
    assert ranking == []


# ---------------------------------------------------------------------------
# ComparisonEntry - as_dict
# ---------------------------------------------------------------------------


def test_comparison_entry_as_dict():
    entry = ComparisonEntry(
        metric_id="m1",
        attempts=[_make_attempt("a1", "m1", "opro", 0.8)],
        best_attempt_id="a1",
        improvement_over_baseline=0.3,
    )
    d = entry.as_dict()
    assert d["metric_id"] == "m1"
    assert len(d["attempts"]) == 1
    assert d["best_attempt_id"] == "a1"
    assert d["improvement_over_baseline"] == 0.3


# ---------------------------------------------------------------------------
# ComparisonDashboard - format_text
# ---------------------------------------------------------------------------


def test_dashboard_format_text():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6, tokens=100),
        _make_attempt("a2", "m1", "ape", 0.9, tokens=200),
    ]
    dashboard = build_dashboard(attempts)
    text = dashboard.format_text()
    assert "Comparison Dashboard" in text
    assert "m1" in text
    assert "opro" in text
    assert "ape" in text


def test_dashboard_format_text_marks_best():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6),
        _make_attempt("a2", "m1", "ape", 0.9),
    ]
    dashboard = build_dashboard(attempts)
    text = dashboard.format_text()
    assert "*" in text


# ---------------------------------------------------------------------------
# ComparisonDashboard - as_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


def test_dashboard_roundtrip():
    attempts = [
        _make_attempt("a1", "m1", "opro", 0.6, tokens=100),
        _make_attempt("a2", "m1", "ape", 0.9, tokens=200),
        _make_attempt("a3", "m2", "opro", 0.5, tokens=50),
    ]
    original = build_dashboard(attempts)
    d = original.as_dict()
    restored = ComparisonDashboard.from_dict(d)
    assert restored.total_metrics == original.total_metrics
    assert restored.total_attempts == original.total_attempts
    assert len(restored.entries) == len(original.entries)
    for orig_e, rest_e in zip(original.entries, restored.entries):
        assert rest_e.metric_id == orig_e.metric_id
        assert rest_e.best_attempt_id == orig_e.best_attempt_id
        assert len(rest_e.attempts) == len(orig_e.attempts)


def test_dashboard_roundtrip_empty():
    original = build_dashboard([])
    d = original.as_dict()
    restored = ComparisonDashboard.from_dict(d)
    assert restored.total_metrics == 0
    assert restored.total_attempts == 0
    assert restored.entries == []
