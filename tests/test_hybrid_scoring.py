"""Tests for hybrid_scoring module."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.hybrid_scoring import (
    HybridConfig,
    HybridItem,
    HybridReport,
    build_hybrid_report,
    compute_final_scores,
    estimate_human_workload,
    identify_for_human_review,
    merge_human_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(
    item_id: str = "i1",
    metric_id: str = "m1",
    ai_score: float = 0.8,
    ai_confidence: float = 0.9,
    **kwargs,
) -> HybridItem:
    return HybridItem(
        item_id=item_id,
        metric_id=metric_id,
        ai_score=ai_score,
        ai_confidence=ai_confidence,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# HybridItem as_dict / from_dict
# ---------------------------------------------------------------------------


class TestHybridItemRoundtrip:
    def test_roundtrip_defaults(self):
        item = _item()
        restored = HybridItem.from_dict(item.as_dict())
        assert restored.item_id == item.item_id
        assert restored.ai_score == item.ai_score
        assert restored.source == "ai"
        assert restored.human_score is None

    def test_roundtrip_with_human_score(self):
        item = _item(human_score=0.6, source="human", final_score=0.6)
        d = item.as_dict()
        restored = HybridItem.from_dict(d)
        assert restored.human_score == 0.6
        assert restored.source == "human"
        assert restored.final_score == 0.6

    def test_as_dict_keys(self):
        d = _item().as_dict()
        expected_keys = {
            "item_id", "metric_id", "ai_score", "ai_confidence",
            "needs_human", "human_score", "final_score", "source",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# HybridConfig
# ---------------------------------------------------------------------------


class TestHybridConfig:
    def test_defaults(self):
        cfg = HybridConfig()
        assert cfg.confidence_threshold == 0.5
        assert cfg.merge_strategy == "human_override"
        assert cfg.human_weight == 0.7

    def test_from_dict(self):
        cfg = HybridConfig.from_dict({
            "confidence_threshold": 0.3,
            "merge_strategy": "weighted_average",
            "human_weight": 0.6,
        })
        assert cfg.confidence_threshold == 0.3
        assert cfg.merge_strategy == "weighted_average"
        assert cfg.human_weight == 0.6

    def test_roundtrip(self):
        cfg = HybridConfig(confidence_threshold=0.8, merge_strategy="ai_only")
        restored = HybridConfig.from_dict(cfg.as_dict())
        assert restored.confidence_threshold == cfg.confidence_threshold
        assert restored.merge_strategy == cfg.merge_strategy


# ---------------------------------------------------------------------------
# HybridReport
# ---------------------------------------------------------------------------


class TestHybridReport:
    def test_format_text_contains_header(self):
        report = HybridReport(total_items=5, ai_only_count=3, human_reviewed_count=2)
        text = report.format_text()
        assert "Hybrid Scoring Report" in text
        assert "Total items: 5" in text
        assert "AI only: 3" in text

    def test_roundtrip(self):
        item = _item(final_score=0.8, source="ai")
        report = HybridReport(
            items=[item], ai_only_count=1, total_items=1,
        )
        restored = HybridReport.from_dict(report.as_dict())
        assert restored.total_items == 1
        assert len(restored.items) == 1
        assert restored.items[0].item_id == "i1"

    def test_empty_report(self):
        report = HybridReport()
        assert report.total_items == 0
        assert report.format_text().startswith("Hybrid Scoring Report")


# ---------------------------------------------------------------------------
# identify_for_human_review
# ---------------------------------------------------------------------------


class TestIdentifyForHumanReview:
    def test_marks_low_confidence(self):
        items = [
            _item("a", ai_confidence=0.3),
            _item("b", ai_confidence=0.7),
        ]
        cfg = HybridConfig(confidence_threshold=0.5)
        result = identify_for_human_review(items, cfg)
        assert result[0].needs_human is True
        assert result[1].needs_human is False

    def test_all_confident(self):
        items = [_item("a", ai_confidence=0.9), _item("b", ai_confidence=0.8)]
        cfg = HybridConfig(confidence_threshold=0.5)
        result = identify_for_human_review(items, cfg)
        assert all(not i.needs_human for i in result)

    def test_all_uncertain(self):
        items = [_item("a", ai_confidence=0.1), _item("b", ai_confidence=0.2)]
        cfg = HybridConfig(confidence_threshold=0.5)
        result = identify_for_human_review(items, cfg)
        assert all(i.needs_human for i in result)

    def test_boundary_value(self):
        items = [_item("a", ai_confidence=0.5)]
        cfg = HybridConfig(confidence_threshold=0.5)
        result = identify_for_human_review(items, cfg)
        # exactly at threshold: not below, so not flagged
        assert result[0].needs_human is False

    def test_empty_list(self):
        result = identify_for_human_review([], HybridConfig())
        assert result == []


# ---------------------------------------------------------------------------
# merge_human_score
# ---------------------------------------------------------------------------


class TestMergeHumanScore:
    def test_human_override(self):
        item = _item(ai_score=0.8)
        cfg = HybridConfig(merge_strategy="human_override")
        merged = merge_human_score(item, 0.5, cfg)
        assert merged.final_score == 0.5
        assert merged.source == "human"
        assert merged.human_score == 0.5

    def test_weighted_average(self):
        item = _item(ai_score=0.8)
        cfg = HybridConfig(merge_strategy="weighted_average", human_weight=0.7)
        merged = merge_human_score(item, 0.4, cfg)
        expected = 0.7 * 0.4 + 0.3 * 0.8
        assert abs(merged.final_score - expected) < 1e-9
        assert merged.source == "merged"

    def test_ai_only_strategy(self):
        item = _item(ai_score=0.8)
        cfg = HybridConfig(merge_strategy="ai_only")
        merged = merge_human_score(item, 0.3, cfg)
        assert merged.final_score == 0.8
        assert merged.source == "ai"

    def test_human_only_strategy(self):
        item = _item(ai_score=0.8)
        cfg = HybridConfig(merge_strategy="human_only")
        merged = merge_human_score(item, 0.3, cfg)
        assert merged.final_score == 0.3
        assert merged.source == "human"

    def test_preserves_item_fields(self):
        item = _item("x", "m2", ai_score=0.8, ai_confidence=0.4, needs_human=True)
        cfg = HybridConfig(merge_strategy="human_override")
        merged = merge_human_score(item, 0.5, cfg)
        assert merged.item_id == "x"
        assert merged.metric_id == "m2"
        assert merged.ai_confidence == 0.4
        assert merged.needs_human is True


# ---------------------------------------------------------------------------
# compute_final_scores
# ---------------------------------------------------------------------------


class TestComputeFinalScores:
    def test_ai_only_items_keep_ai_score(self):
        items = [_item("a", ai_score=0.7)]
        result = compute_final_scores(items, HybridConfig())
        assert result[0].final_score == 0.7
        assert result[0].source == "ai"

    def test_items_with_human_score_merged(self):
        items = [_item("a", ai_score=0.8, human_score=0.5)]
        cfg = HybridConfig(merge_strategy="human_override")
        result = compute_final_scores(items, cfg)
        assert result[0].final_score == 0.5
        assert result[0].source == "human"

    def test_mixed_items(self):
        items = [
            _item("a", ai_score=0.8),
            _item("b", ai_score=0.6, human_score=0.9),
        ]
        cfg = HybridConfig(merge_strategy="human_override")
        result = compute_final_scores(items, cfg)
        assert result[0].source == "ai"
        assert result[1].source == "human"
        assert result[1].final_score == 0.9


# ---------------------------------------------------------------------------
# build_hybrid_report
# ---------------------------------------------------------------------------


class TestBuildHybridReport:
    def test_counts(self):
        items = [
            _item("a", source="ai", final_score=0.8),
            _item("b", source="human", final_score=0.5),
            _item("c", source="merged", final_score=0.6),
            _item("d", source="ai", final_score=0.7),
        ]
        report = build_hybrid_report(items)
        assert report.total_items == 4
        assert report.ai_only_count == 2
        assert report.human_reviewed_count == 1
        assert report.merged_count == 1

    def test_empty(self):
        report = build_hybrid_report([])
        assert report.total_items == 0
        assert report.ai_only_count == 0

    def test_all_ai(self):
        items = [_item("a", source="ai"), _item("b", source="ai")]
        report = build_hybrid_report(items)
        assert report.ai_only_count == 2
        assert report.human_reviewed_count == 0
        assert report.merged_count == 0


# ---------------------------------------------------------------------------
# estimate_human_workload
# ---------------------------------------------------------------------------


class TestEstimateHumanWorkload:
    def test_basic(self):
        items = [
            _item("a", needs_human=True),
            _item("b", needs_human=False),
            _item("c", needs_human=True),
        ]
        est = estimate_human_workload(items, avg_seconds_per_item=60.0)
        assert est["items_needing_review"] == 2
        assert est["total_seconds"] == 120.0
        assert est["total_minutes"] == 2.0
        assert est["total_hours"] == pytest.approx(120.0 / 3600.0)

    def test_none_needing_review(self):
        items = [_item("a", needs_human=False)]
        est = estimate_human_workload(items)
        assert est["items_needing_review"] == 0
        assert est["total_seconds"] == 0.0

    def test_custom_avg_seconds(self):
        items = [_item("a", needs_human=True)]
        est = estimate_human_workload(items, avg_seconds_per_item=45.0)
        assert est["avg_seconds_per_item"] == 45.0
        assert est["total_seconds"] == 45.0

    def test_empty_list(self):
        est = estimate_human_workload([])
        assert est["items_needing_review"] == 0
