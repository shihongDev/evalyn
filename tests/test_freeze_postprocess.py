"""Tests for metric registry freeze and output post-processing."""

import pytest
from pathlib import Path
from evalyn_sdk.models import MetricSpec
from evalyn_sdk.metrics.registry_freeze import (
    freeze_metrics, load_freeze, check_against_freeze, unfreeze_metrics,
    FrozenMetricSet, FreezeCheckResult, FREEZE_FILENAME,
)
from evalyn_sdk.metrics.post_processing import (
    score_clamp, binary_threshold, invert_score, score_round, scale,
    PostProcessorChain, build_chain_from_config,
)


# =============================================================================
# Registry freeze tests
# =============================================================================

def _spec(mid, mtype="objective"):
    return MetricSpec(id=mid, name=mid, type=mtype)


class TestFreezeMetrics:
    def test_freeze_creates_file(self, tmp_path):
        specs = [_spec("a"), _spec("b", "subjective")]
        frozen = freeze_metrics(specs, tmp_path, description="test freeze")
        assert (tmp_path / FREEZE_FILENAME).exists()
        assert frozen.count == 2
        assert frozen.description == "test freeze"

    def test_load_freeze(self, tmp_path):
        freeze_metrics([_spec("a")], tmp_path)
        loaded = load_freeze(tmp_path)
        assert loaded is not None
        assert "a" in loaded.metric_ids

    def test_load_no_freeze(self, tmp_path):
        assert load_freeze(tmp_path) is None

    def test_unfreeze(self, tmp_path):
        freeze_metrics([_spec("a")], tmp_path)
        assert unfreeze_metrics(tmp_path) is True
        assert not (tmp_path / FREEZE_FILENAME).exists()

    def test_unfreeze_nonexistent(self, tmp_path):
        assert unfreeze_metrics(tmp_path) is False


class TestCheckAgainstFreeze:
    def test_no_freeze_passes(self, tmp_path):
        result = check_against_freeze([_spec("a")], tmp_path)
        assert not result.is_frozen
        assert result.matches

    def test_matching(self, tmp_path):
        specs = [_spec("a"), _spec("b")]
        freeze_metrics(specs, tmp_path)
        result = check_against_freeze(specs, tmp_path)
        assert result.is_frozen
        assert result.matches

    def test_added_metric(self, tmp_path):
        freeze_metrics([_spec("a")], tmp_path)
        result = check_against_freeze([_spec("a"), _spec("b")], tmp_path)
        assert not result.matches
        assert "b" in result.added_metrics

    def test_removed_metric(self, tmp_path):
        freeze_metrics([_spec("a"), _spec("b")], tmp_path)
        result = check_against_freeze([_spec("a")], tmp_path)
        assert not result.matches
        assert "b" in result.removed_metrics

    def test_changed_metric(self, tmp_path):
        specs_v1 = [MetricSpec(id="a", name="a", type="subjective", config={"v": 1})]
        freeze_metrics(specs_v1, tmp_path)
        specs_v2 = [MetricSpec(id="a", name="a", type="subjective", config={"v": 2})]
        result = check_against_freeze(specs_v2, tmp_path)
        assert not result.matches
        assert "a" in result.changed_metrics

    def test_format_text_match(self, tmp_path):
        freeze_metrics([_spec("a")], tmp_path)
        text = check_against_freeze([_spec("a")], tmp_path).format_text()
        assert "matching" in text

    def test_format_text_mismatch(self, tmp_path):
        freeze_metrics([_spec("a")], tmp_path)
        text = check_against_freeze([_spec("b")], tmp_path).format_text()
        assert "MISMATCH" in text

    def test_as_dict(self, tmp_path):
        freeze_metrics([_spec("a")], tmp_path)
        d = check_against_freeze([_spec("a")], tmp_path).as_dict()
        assert d["is_frozen"] is True
        assert d["matches"] is True


class TestFrozenMetricSet:
    def test_roundtrip(self):
        f = FrozenMetricSet(
            metrics={"a": {"type": "objective"}},
            frozen_at="2026-03-28",
            description="test",
        )
        d = f.as_dict()
        restored = FrozenMetricSet.from_dict(d)
        assert restored.metric_ids == ["a"]


# =============================================================================
# Post-processing tests
# =============================================================================

class TestBuiltinProcessors:
    def test_clamp(self):
        p = score_clamp(0.0, 1.0)
        assert p.transform(1.5) == 1.0
        assert p.transform(-0.5) == 0.0
        assert p.transform(0.5) == 0.5

    def test_binary_threshold(self):
        p = binary_threshold(0.5)
        assert p.transform(0.6) == 1.0
        assert p.transform(0.4) == 0.0
        assert p.transform(0.5) == 1.0

    def test_invert(self):
        p = invert_score()
        assert p.transform(0.8) == pytest.approx(0.2)
        assert p.transform(0.0) == 1.0

    def test_round(self):
        p = score_round(2)
        assert p.transform(0.8567) == 0.86

    def test_scale(self):
        p = scale(100.0)
        assert p.transform(0.85) == 85.0


class TestPostProcessorChain:
    def test_single_processor(self):
        chain = PostProcessorChain([score_clamp(0, 1)])
        assert chain.apply(1.5) == 1.0

    def test_chain_order(self):
        chain = PostProcessorChain([
            scale(100),
            score_round(0),
        ])
        assert chain.apply(0.856) == 86.0

    def test_empty_chain(self):
        chain = PostProcessorChain()
        assert chain.apply(0.5) == 0.5

    def test_fluent_add(self):
        chain = PostProcessorChain()
        chain.add(score_clamp(0, 1)).add(score_round(2))
        assert chain.apply(1.567) == 1.0

    def test_names(self):
        chain = PostProcessorChain([score_clamp(), invert_score()])
        assert len(chain.names) == 2

    def test_as_dict(self):
        chain = PostProcessorChain([score_clamp()])
        d = chain.as_dict()
        assert "processors" in d
        assert len(d["processors"]) == 1

    def test_processor_error_skipped(self):
        from evalyn_sdk.metrics.post_processing import PostProcessor
        bad = PostProcessor(name="bad", transform=lambda s: 1/0)
        chain = PostProcessorChain([bad, score_round(2)])
        # bad processor errors, but round still applies
        result = chain.apply(0.856)
        assert result == 0.86  # round applied after skip


class TestBuildChainFromConfig:
    def test_basic(self):
        config = [
            {"name": "clamp", "min_val": 0.0, "max_val": 1.0},
            {"name": "round", "decimals": 2},
        ]
        chain = build_chain_from_config(config)
        assert len(chain.processors) == 2
        assert chain.apply(1.567) == 1.0

    def test_unknown_processor_skipped(self):
        config = [
            {"name": "nonexistent"},
            {"name": "round", "decimals": 1},
        ]
        chain = build_chain_from_config(config)
        assert len(chain.processors) == 1

    def test_empty_config(self):
        chain = build_chain_from_config([])
        assert len(chain.processors) == 0
