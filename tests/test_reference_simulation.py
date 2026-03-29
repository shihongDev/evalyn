"""Tests for the reference simulation module."""

from __future__ import annotations

import pytest

from evalyn_sdk.reference_simulation import (
    QualityConfig,
    ReferencePair,
    ReferenceSet,
    apply_quality_filter,
    build_reference_set,
    compute_reference_stats,
    detect_low_quality_pairs,
    export_as_dataset,
    format_reference_report,
    score_reference_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair(
    inp: str = "What is X?",
    out: str = "X is a well-known concept in the field",
    conf: float = 0.9,
    meta: dict | None = None,
) -> ReferencePair:
    return ReferencePair(
        input_text=inp,
        reference_output=out,
        confidence=conf,
        metadata=meta or {},
    )


# ---------------------------------------------------------------------------
# ReferencePair
# ---------------------------------------------------------------------------


class TestReferencePair:
    def test_create_defaults(self):
        p = ReferencePair(input_text="hi", reference_output="hello")
        assert p.confidence == 0.0
        assert p.quality_passed is False
        assert p.metadata == {}

    def test_create_with_values(self):
        p = ReferencePair(
            input_text="q", reference_output="a",
            confidence=0.95, quality_passed=True,
            metadata={"source": "test"},
        )
        assert p.confidence == 0.95
        assert p.metadata["source"] == "test"

    def test_as_dict(self):
        p = _pair(meta={"k": "v"})
        d = p.as_dict()
        assert d["input_text"] == "What is X?"
        assert d["confidence"] == 0.9
        assert d["metadata"] == {"k": "v"}

    def test_from_dict(self):
        data = {
            "input_text": "q",
            "reference_output": "a",
            "confidence": 0.85,
            "quality_passed": True,
            "metadata": {"x": 1},
        }
        p = ReferencePair.from_dict(data)
        assert p.confidence == 0.85
        assert p.quality_passed is True

    def test_from_dict_defaults(self):
        data = {"input_text": "q", "reference_output": "a"}
        p = ReferencePair.from_dict(data)
        assert p.confidence == 0.0
        assert p.quality_passed is False
        assert p.metadata == {}

    def test_roundtrip(self):
        p = _pair(meta={"z": 42})
        assert ReferencePair.from_dict(p.as_dict()) == p


# ---------------------------------------------------------------------------
# QualityConfig
# ---------------------------------------------------------------------------


class TestQualityConfig:
    def test_defaults(self):
        c = QualityConfig()
        assert c.min_confidence == 0.8
        assert c.min_output_length == 10
        assert c.max_output_length == 5000
        assert c.require_non_empty is True

    def test_as_dict(self):
        c = QualityConfig(min_confidence=0.5, min_output_length=5)
        d = c.as_dict()
        assert d["min_confidence"] == 0.5
        assert d["min_output_length"] == 5

    def test_from_dict(self):
        data = {"min_confidence": 0.6, "max_output_length": 3000}
        c = QualityConfig.from_dict(data)
        assert c.min_confidence == 0.6
        assert c.max_output_length == 3000
        assert c.min_output_length == 10  # default

    def test_roundtrip(self):
        c = QualityConfig(
            min_confidence=0.7, min_output_length=20,
            max_output_length=4000, require_non_empty=False,
        )
        assert QualityConfig.from_dict(c.as_dict()) == c


# ---------------------------------------------------------------------------
# ReferenceSet
# ---------------------------------------------------------------------------


class TestReferenceSet:
    def test_as_dict(self):
        rs = ReferenceSet(
            pairs=[_pair()], total_generated=3,
            quality_filtered=1, acceptance_rate=0.333,
        )
        d = rs.as_dict()
        assert len(d["pairs"]) == 1
        assert d["total_generated"] == 3

    def test_from_dict(self):
        rs = ReferenceSet(
            pairs=[_pair(), _pair(conf=0.5)],
            total_generated=5, quality_filtered=2, acceptance_rate=0.4,
        )
        restored = ReferenceSet.from_dict(rs.as_dict())
        assert len(restored.pairs) == 2
        assert restored.acceptance_rate == 0.4

    def test_roundtrip(self):
        rs = ReferenceSet(
            pairs=[_pair()], total_generated=1,
            quality_filtered=1, acceptance_rate=1.0,
        )
        assert ReferenceSet.from_dict(rs.as_dict()) == rs


# ---------------------------------------------------------------------------
# apply_quality_filter
# ---------------------------------------------------------------------------


class TestApplyQualityFilter:
    def test_filters_low_confidence(self):
        pairs = [_pair(conf=0.5), _pair(conf=0.9)]
        config = QualityConfig(min_confidence=0.8)
        result = apply_quality_filter(pairs, config)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_filters_short_output(self):
        pairs = [_pair(out="short"), _pair(out="This is a long enough output")]
        config = QualityConfig(min_confidence=0.0, min_output_length=10)
        result = apply_quality_filter(pairs, config)
        assert len(result) == 1

    def test_filters_long_output(self):
        pairs = [_pair(out="x" * 6000), _pair(out="Normal length output")]
        config = QualityConfig(min_confidence=0.0, max_output_length=5000)
        result = apply_quality_filter(pairs, config)
        assert len(result) == 1

    def test_filters_empty_when_required(self):
        pairs = [_pair(out=""), _pair(out="Not empty output text")]
        config = QualityConfig(
            min_confidence=0.0, require_non_empty=True, min_output_length=0,
        )
        result = apply_quality_filter(pairs, config)
        assert len(result) == 1

    def test_allows_empty_when_not_required(self):
        pairs = [_pair(out="")]
        config = QualityConfig(
            min_confidence=0.0, require_non_empty=False,
            min_output_length=0,
        )
        result = apply_quality_filter(pairs, config)
        assert len(result) == 1

    def test_sets_quality_passed(self):
        pairs = [_pair()]
        config = QualityConfig(min_confidence=0.0, min_output_length=0)
        result = apply_quality_filter(pairs, config)
        assert result[0].quality_passed is True

    def test_empty_input(self):
        result = apply_quality_filter([], QualityConfig())
        assert result == []

    def test_all_filtered(self):
        pairs = [_pair(conf=0.1), _pair(conf=0.2)]
        result = apply_quality_filter(pairs, QualityConfig(min_confidence=0.99))
        assert result == []


# ---------------------------------------------------------------------------
# build_reference_set
# ---------------------------------------------------------------------------


class TestBuildReferenceSet:
    def test_basic(self):
        pairs = [_pair(conf=0.9), _pair(conf=0.5), _pair(conf=0.95)]
        rs = build_reference_set(pairs)
        assert rs.total_generated == 3
        assert rs.quality_filtered == 2
        assert abs(rs.acceptance_rate - 2 / 3) < 1e-9

    def test_default_config(self):
        pairs = [_pair(conf=0.9)]
        rs = build_reference_set(pairs)
        assert rs.quality_filtered == 1

    def test_custom_config(self):
        pairs = [_pair(conf=0.5)]
        config = QualityConfig(min_confidence=0.3)
        rs = build_reference_set(pairs, config=config)
        assert rs.quality_filtered == 1

    def test_empty_input(self):
        rs = build_reference_set([])
        assert rs.total_generated == 0
        assert rs.quality_filtered == 0
        assert rs.acceptance_rate == 0.0

    def test_all_rejected(self):
        pairs = [_pair(conf=0.1)]
        rs = build_reference_set(pairs, QualityConfig(min_confidence=0.99))
        assert rs.quality_filtered == 0
        assert rs.acceptance_rate == 0.0


# ---------------------------------------------------------------------------
# score_reference_quality
# ---------------------------------------------------------------------------


class TestScoreReferenceQuality:
    def test_high_quality(self):
        p = _pair(
            inp="What is machine learning?",
            out="Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            conf=0.95,
        )
        score = score_reference_quality(p)
        assert 0.5 < score <= 1.0

    def test_empty_output_zero(self):
        p = _pair(out="", conf=0.9)
        assert score_reference_quality(p) == 0.0

    def test_low_confidence_lowers_score(self):
        high = _pair(conf=1.0)
        low = _pair(conf=0.0)
        assert score_reference_quality(high) > score_reference_quality(low)

    def test_repetitive_output_lowers_score(self):
        varied = _pair(out="The quick brown fox jumps over the lazy dog")
        repetitive = _pair(out="the the the the the the the the the the")
        assert score_reference_quality(varied) > score_reference_quality(repetitive)

    def test_score_in_range(self):
        p = _pair()
        score = score_reference_quality(p)
        assert 0.0 <= score <= 1.0

    def test_zero_confidence_still_has_other_components(self):
        p = _pair(
            out="This is a reasonably diverse and complete answer",
            conf=0.0,
        )
        score = score_reference_quality(p)
        assert score > 0.0  # length and uniqueness still contribute


# ---------------------------------------------------------------------------
# detect_low_quality_pairs
# ---------------------------------------------------------------------------


class TestDetectLowQualityPairs:
    def test_detects_empty_output(self):
        pairs = [_pair(out="", conf=0.0), _pair()]
        low = detect_low_quality_pairs(pairs)
        assert len(low) == 1
        assert low[0].reference_output == ""

    def test_threshold(self):
        pairs = [_pair(conf=0.0, out="ok output"), _pair(conf=0.9)]
        low = detect_low_quality_pairs(pairs, threshold=0.5)
        # The low-confidence pair should score lower
        assert len(low) >= 1

    def test_empty_input(self):
        assert detect_low_quality_pairs([]) == []

    def test_all_high_quality(self):
        pairs = [
            _pair(
                inp="short",
                out="A comprehensive and detailed answer with many unique words here",
                conf=0.95,
            ),
        ]
        low = detect_low_quality_pairs(pairs, threshold=0.3)
        assert len(low) == 0


# ---------------------------------------------------------------------------
# export_as_dataset
# ---------------------------------------------------------------------------


class TestExportAsDataset:
    def test_basic(self):
        rs = ReferenceSet(
            pairs=[_pair(meta={"src": "test"})],
            total_generated=1, quality_filtered=1, acceptance_rate=1.0,
        )
        items = export_as_dataset(rs)
        assert len(items) == 1
        assert items[0]["input"] == "What is X?"
        assert "expected_output" in items[0]
        assert items[0]["metadata"] == {"src": "test"}

    def test_empty_set(self):
        rs = ReferenceSet(
            pairs=[], total_generated=0, quality_filtered=0, acceptance_rate=0.0,
        )
        assert export_as_dataset(rs) == []

    def test_multiple_items(self):
        rs = ReferenceSet(
            pairs=[_pair(inp="q1"), _pair(inp="q2"), _pair(inp="q3")],
            total_generated=3, quality_filtered=3, acceptance_rate=1.0,
        )
        items = export_as_dataset(rs)
        assert len(items) == 3
        assert items[0]["input"] == "q1"
        assert items[2]["input"] == "q3"

    def test_metadata_is_copy(self):
        meta = {"key": "val"}
        rs = ReferenceSet(
            pairs=[_pair(meta=meta)],
            total_generated=1, quality_filtered=1, acceptance_rate=1.0,
        )
        items = export_as_dataset(rs)
        # Modifying export should not affect original
        items[0]["metadata"]["key"] = "changed"
        assert meta["key"] == "val"


# ---------------------------------------------------------------------------
# format_reference_report
# ---------------------------------------------------------------------------


class TestFormatReferenceReport:
    def test_contains_header(self):
        rs = build_reference_set([_pair()])
        text = format_reference_report(rs)
        assert "Reference Set Report" in text

    def test_contains_acceptance_rate(self):
        rs = build_reference_set([_pair()])
        text = format_reference_report(rs)
        assert "Acceptance rate" in text

    def test_contains_quality_stats(self):
        rs = build_reference_set([_pair()])
        text = format_reference_report(rs)
        assert "Avg confidence" in text
        assert "Avg output length" in text

    def test_empty_set(self):
        rs = build_reference_set([])
        text = format_reference_report(rs)
        assert "Total generated: 0" in text


# ---------------------------------------------------------------------------
# compute_reference_stats
# ---------------------------------------------------------------------------


class TestComputeReferenceStats:
    def test_basic(self):
        rs = build_reference_set([_pair(conf=0.9), _pair(conf=0.85)])
        stats = compute_reference_stats(rs)
        assert stats["total"] == 2
        assert stats["accepted"] == 2
        assert 0.85 <= stats["avg_confidence"] <= 0.9
        assert stats["avg_output_length"] > 0
        assert stats["avg_quality_score"] > 0

    def test_empty_set(self):
        rs = build_reference_set([])
        stats = compute_reference_stats(rs)
        assert stats["total"] == 0
        assert stats["accepted"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["avg_output_length"] == 0.0
        assert stats["avg_quality_score"] == 0.0

    def test_all_rejected(self):
        rs = build_reference_set(
            [_pair(conf=0.1)], QualityConfig(min_confidence=0.99)
        )
        stats = compute_reference_stats(rs)
        assert stats["total"] == 1
        assert stats["accepted"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_keys_present(self):
        rs = build_reference_set([_pair()])
        stats = compute_reference_stats(rs)
        expected_keys = {
            "total", "accepted", "avg_confidence",
            "avg_output_length", "avg_quality_score",
        }
        assert set(stats.keys()) == expected_keys
