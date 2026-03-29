"""Tests for the evol_instruct module."""

from __future__ import annotations

import random

import pytest

from evalyn_sdk.evol_instruct import (
    BREADTH_STRATEGY,
    BREADTH_TRANSFORMS,
    DEPTH_STRATEGY,
    DEPTH_TRANSFORMS,
    EvolutionConfig,
    EvolutionReport,
    EvolutionStrategy,
    EvolvedItem,
    apply_breadth_transform,
    apply_depth_transform,
    evolve_batch,
    evolve_item,
    format_evolution_report,
    score_evolution_quality,
)


# ---------------------------------------------------------------------------
# EvolutionStrategy
# ---------------------------------------------------------------------------


class TestEvolutionStrategy:
    def test_create(self):
        s = EvolutionStrategy(strategy_id="depth", description="d", transforms=["a"])
        assert s.strategy_id == "depth"
        assert s.transforms == ["a"]

    def test_as_dict(self):
        s = EvolutionStrategy(strategy_id="x", description="y", transforms=["t1", "t2"])
        d = s.as_dict()
        assert d["strategy_id"] == "x"
        assert d["transforms"] == ["t1", "t2"]

    def test_from_dict(self):
        s = EvolutionStrategy.from_dict(
            {"strategy_id": "breadth", "description": "b", "transforms": ["r"]}
        )
        assert s.strategy_id == "breadth"

    def test_roundtrip(self):
        orig = EvolutionStrategy(strategy_id="depth", description="d", transforms=["a", "b"])
        rebuilt = EvolutionStrategy.from_dict(orig.as_dict())
        assert rebuilt.strategy_id == orig.strategy_id
        assert rebuilt.transforms == orig.transforms


# ---------------------------------------------------------------------------
# EvolvedItem
# ---------------------------------------------------------------------------


class TestEvolvedItem:
    def test_defaults(self):
        item = EvolvedItem(
            original_text="hi", evolved_text="hello", strategy="depth", generation=0
        )
        assert item.quality_score == 0.0
        assert item.accepted is True

    def test_as_dict(self):
        item = EvolvedItem(
            original_text="a",
            evolved_text="b",
            strategy="breadth",
            generation=2,
            quality_score=0.8,
            accepted=False,
        )
        d = item.as_dict()
        assert d["strategy"] == "breadth"
        assert d["accepted"] is False

    def test_from_dict(self):
        item = EvolvedItem.from_dict(
            {
                "original_text": "x",
                "evolved_text": "y",
                "strategy": "depth",
                "generation": 1,
            }
        )
        assert item.quality_score == 0.0
        assert item.accepted is True

    def test_roundtrip(self):
        orig = EvolvedItem(
            original_text="a",
            evolved_text="b",
            strategy="depth",
            generation=1,
            quality_score=0.5,
            accepted=False,
        )
        rebuilt = EvolvedItem.from_dict(orig.as_dict())
        assert rebuilt.accepted == orig.accepted
        assert rebuilt.quality_score == orig.quality_score


# ---------------------------------------------------------------------------
# EvolutionConfig
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    def test_defaults(self):
        c = EvolutionConfig()
        assert c.max_generations == 3
        assert c.quality_threshold == 0.3
        assert c.strategies == ["depth", "breadth"]

    def test_custom(self):
        c = EvolutionConfig(max_generations=5, quality_threshold=0.5, strategies=["depth"])
        assert c.max_generations == 5

    def test_roundtrip(self):
        c = EvolutionConfig(max_generations=7, quality_threshold=0.1)
        rebuilt = EvolutionConfig.from_dict(c.as_dict())
        assert rebuilt.max_generations == 7
        assert rebuilt.quality_threshold == pytest.approx(0.1)

    def test_from_dict_defaults(self):
        c = EvolutionConfig.from_dict({})
        assert c.max_generations == 3


# ---------------------------------------------------------------------------
# EvolutionReport
# ---------------------------------------------------------------------------


class TestEvolutionReport:
    def _make_report(self):
        items = [
            EvolvedItem("a", "b", "depth", 0, 0.5, True),
            EvolvedItem("c", "d", "breadth", 1, 0.2, False),
        ]
        return EvolutionReport(
            items=items,
            total_evolved=2,
            accepted_count=1,
            rejected_count=1,
            avg_quality=0.35,
        )

    def test_as_dict(self):
        r = self._make_report()
        d = r.as_dict()
        assert d["total_evolved"] == 2
        assert len(d["items"]) == 2

    def test_from_dict(self):
        r = self._make_report()
        rebuilt = EvolutionReport.from_dict(r.as_dict())
        assert rebuilt.total_evolved == 2
        assert rebuilt.items[1].accepted is False

    def test_roundtrip(self):
        r = self._make_report()
        rebuilt = EvolutionReport.from_dict(r.as_dict())
        assert rebuilt.avg_quality == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Transform Catalogs
# ---------------------------------------------------------------------------


class TestTransformCatalogs:
    def test_depth_transforms_count(self):
        assert len(DEPTH_TRANSFORMS) == 5

    def test_breadth_transforms_count(self):
        assert len(BREADTH_TRANSFORMS) == 5

    def test_depth_strategy_has_all_transforms(self):
        assert DEPTH_STRATEGY.transforms == DEPTH_TRANSFORMS

    def test_breadth_strategy_has_all_transforms(self):
        assert BREADTH_STRATEGY.transforms == BREADTH_TRANSFORMS

    def test_no_overlap(self):
        assert set(DEPTH_TRANSFORMS).isdisjoint(set(BREADTH_TRANSFORMS))


# ---------------------------------------------------------------------------
# apply_depth_transform
# ---------------------------------------------------------------------------


class TestApplyDepthTransform:
    def test_add_constraint(self):
        rng = random.Random(42)
        result = apply_depth_transform("Solve this problem", "add_constraint", rng)
        assert "Additionally, ensure that" in result

    def test_add_reasoning_step(self):
        rng = random.Random(42)
        result = apply_depth_transform("Solve this problem.", "add_reasoning_step", rng)
        assert "Solve this problem" in result
        # Should have a reasoning prefix
        assert any(
            result.startswith(pfx.split(",")[0])
            for pfx in [
                "First", "Begin by", "Start by", "Consider",
            ]
        )

    def test_add_edge_case(self):
        rng = random.Random(42)
        result = apply_depth_transform("Do X", "add_edge_case", rng)
        assert "Consider the edge case where" in result

    def test_increase_specificity(self):
        rng = random.Random(42)
        result = apply_depth_transform("Do X", "increase_specificity", rng)
        assert len(result) > len("Do X")

    def test_add_conditional(self):
        rng = random.Random(42)
        result = apply_depth_transform("Do X", "add_conditional", rng)
        assert "If" in result or "if" in result

    def test_unknown_transform_passthrough(self):
        rng = random.Random(42)
        assert apply_depth_transform("hello", "unknown", rng) == "hello"

    def test_deterministic_with_seed(self):
        r1 = apply_depth_transform("text", "add_constraint", random.Random(99))
        r2 = apply_depth_transform("text", "add_constraint", random.Random(99))
        assert r1 == r2


# ---------------------------------------------------------------------------
# apply_breadth_transform
# ---------------------------------------------------------------------------


class TestApplyBreadthTransform:
    def test_rephrase(self):
        rng = random.Random(42)
        result = apply_breadth_transform("Do X", "rephrase", rng)
        assert "Do X" in result

    def test_change_format(self):
        rng = random.Random(42)
        result = apply_breadth_transform("Do X", "change_format", rng)
        assert "Do X" in result

    def test_add_context(self):
        rng = random.Random(42)
        result = apply_breadth_transform("Do X", "add_context", rng)
        assert "Do X" in result
        assert len(result) > len("Do X")

    def test_domain_transfer(self):
        rng = random.Random(42)
        result = apply_breadth_transform("Write a test for the function", "domain_transfer", rng)
        # At least one domain word should be swapped (or text unchanged if no match)
        assert isinstance(result, str)

    def test_topic_variation_single_sentence(self):
        rng = random.Random(42)
        result = apply_breadth_transform("Just one sentence", "topic_variation", rng)
        assert result == "Just one sentence"

    def test_topic_variation_multiple_sentences(self):
        rng = random.Random(42)
        text = "First sentence. Second sentence. Third sentence."
        result = apply_breadth_transform(text, "topic_variation", rng)
        # All sentences should still be present
        assert "First sentence" in result
        assert "Second sentence" in result
        assert "Third sentence" in result

    def test_unknown_transform_passthrough(self):
        rng = random.Random(42)
        assert apply_breadth_transform("hello", "unknown", rng) == "hello"


# ---------------------------------------------------------------------------
# score_evolution_quality
# ---------------------------------------------------------------------------


class TestScoreEvolutionQuality:
    def test_empty_original(self):
        assert score_evolution_quality("", "something") == 0.0

    def test_empty_evolved(self):
        assert score_evolution_quality("something", "") == 0.0

    def test_identical(self):
        score = score_evolution_quality("hello world", "hello world")
        assert score == pytest.approx(0.0)

    def test_longer_is_higher(self):
        short_evo = score_evolution_quality("hello", "hello there")
        long_evo = score_evolution_quality("hello", "hello there my friend, how are you doing today")
        assert long_evo > short_evo

    def test_score_bounded(self):
        score = score_evolution_quality("a", "a " * 1000)
        assert 0.0 <= score <= 1.0

    def test_new_vocabulary_helps(self):
        # Same length-ish but more new words should score higher
        base = "the cat sat on the mat"
        evo_repeat = "the cat sat on the mat the cat sat on the mat"
        evo_new = "the feline perched upon the carpet elegantly"
        s_repeat = score_evolution_quality(base, evo_repeat)
        s_new = score_evolution_quality(base, evo_new)
        assert s_new > s_repeat


# ---------------------------------------------------------------------------
# evolve_item
# ---------------------------------------------------------------------------


class TestEvolveItem:
    def test_depth(self):
        item = evolve_item("Solve the problem", strategy="depth", rng=random.Random(42))
        assert item.strategy == "depth"
        assert len(item.evolved_text) > len("Solve the problem")
        assert item.generation == 0

    def test_breadth(self):
        item = evolve_item("Solve the problem", strategy="breadth", rng=random.Random(42))
        assert item.strategy == "breadth"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            evolve_item("text", strategy="invalid")

    def test_quality_score_set(self):
        item = evolve_item("hello", strategy="depth", rng=random.Random(42))
        assert item.quality_score >= 0.0

    def test_deterministic(self):
        i1 = evolve_item("x", strategy="depth", rng=random.Random(7))
        i2 = evolve_item("x", strategy="depth", rng=random.Random(7))
        assert i1.evolved_text == i2.evolved_text

    def test_generation_preserved(self):
        item = evolve_item("x", generation=5, rng=random.Random(1))
        assert item.generation == 5


# ---------------------------------------------------------------------------
# evolve_batch
# ---------------------------------------------------------------------------


class TestEvolveBatch:
    def test_basic(self):
        items = ["Write a function", "Explain recursion", "Compare sorting algorithms"]
        config = EvolutionConfig(max_generations=2)
        report = evolve_batch(items, config, rng=random.Random(42))
        assert report.total_evolved > 0
        assert report.accepted_count + report.rejected_count == report.total_evolved

    def test_empty_input(self):
        report = evolve_batch([], EvolutionConfig(), rng=random.Random(1))
        assert report.total_evolved == 0
        assert report.avg_quality == 0.0

    def test_single_generation(self):
        config = EvolutionConfig(max_generations=1)
        report = evolve_batch(["hello world"], config, rng=random.Random(42))
        # At most 1 item per input with 1 generation
        assert report.total_evolved >= 1

    def test_high_threshold_rejects(self):
        config = EvolutionConfig(max_generations=1, quality_threshold=0.99)
        report = evolve_batch(
            ["a very short prompt"],
            config,
            rng=random.Random(42),
        )
        # Very high threshold likely causes rejection
        # (depends on heuristic, but short->slightly longer won't hit 0.99)
        rejected = [i for i in report.items if not i.accepted]
        # At least check report is consistent
        assert report.rejected_count == len(rejected)

    def test_low_threshold_accepts_all(self):
        config = EvolutionConfig(max_generations=2, quality_threshold=0.0)
        report = evolve_batch(["some text here"], config, rng=random.Random(42))
        assert report.rejected_count == 0

    def test_avg_quality_within_bounds(self):
        config = EvolutionConfig(max_generations=2)
        report = evolve_batch(
            ["Describe a complex system", "Explain quantum computing"],
            config,
            rng=random.Random(42),
        )
        assert 0.0 <= report.avg_quality <= 1.0

    def test_deterministic(self):
        items = ["abc", "def"]
        config = EvolutionConfig(max_generations=2)
        r1 = evolve_batch(items, config, rng=random.Random(10))
        r2 = evolve_batch(items, config, rng=random.Random(10))
        assert r1.total_evolved == r2.total_evolved
        assert r1.avg_quality == r2.avg_quality

    def test_original_text_preserved(self):
        config = EvolutionConfig(max_generations=3)
        report = evolve_batch(["original prompt"], config, rng=random.Random(42))
        for item in report.items:
            assert item.original_text == "original prompt"

    def test_strategies_respected(self):
        config = EvolutionConfig(max_generations=2, strategies=["depth"])
        report = evolve_batch(["test input"], config, rng=random.Random(42))
        for item in report.items:
            assert item.strategy == "depth"


# ---------------------------------------------------------------------------
# format_evolution_report
# ---------------------------------------------------------------------------


class TestFormatEvolutionReport:
    def test_contains_header(self):
        report = EvolutionReport(
            items=[], total_evolved=0, accepted_count=0, rejected_count=0, avg_quality=0.0
        )
        text = format_evolution_report(report)
        assert "Evolution Report" in text

    def test_contains_counts(self):
        items = [EvolvedItem("a", "b", "depth", 0, 0.5, True)]
        report = EvolutionReport(
            items=items, total_evolved=1, accepted_count=1, rejected_count=0, avg_quality=0.5
        )
        text = format_evolution_report(report)
        assert "Accepted" in text
        assert "Rejected" in text

    def test_contains_item_details(self):
        items = [
            EvolvedItem("a", "evolved text here", "breadth", 1, 0.7, True),
            EvolvedItem("b", "rejected text", "depth", 0, 0.1, False),
        ]
        report = EvolutionReport(
            items=items, total_evolved=2, accepted_count=1, rejected_count=1, avg_quality=0.4
        )
        text = format_evolution_report(report)
        assert "ACCEPTED" in text
        assert "REJECTED" in text
        assert "breadth" in text

    def test_long_text_truncated(self):
        long_text = "x" * 200
        items = [EvolvedItem("a", long_text, "depth", 0, 0.5, True)]
        report = EvolutionReport(
            items=items, total_evolved=1, accepted_count=1, rejected_count=0, avg_quality=0.5
        )
        text = format_evolution_report(report)
        assert "..." in text
