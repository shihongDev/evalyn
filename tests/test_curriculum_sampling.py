"""Tests for curriculum sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.curriculum_sampling import (
    CurriculumBatch,
    CurriculumConfig,
    CurriculumResult,
    DifficultyEstimate,
    check_early_stop,
    create_batches,
    estimate_difficulty_by_complexity,
    estimate_difficulty_by_length,
    estimate_difficulty_by_scores,
    format_curriculum_report,
    order_curriculum,
    run_curriculum,
)


# ---------------------------------------------------------------------------
# DifficultyEstimate
# ---------------------------------------------------------------------------


class TestDifficultyEstimate:
    def test_basic_creation(self):
        e = DifficultyEstimate(item_id="a", difficulty=0.5, source="length")
        assert e.item_id == "a"
        assert e.difficulty == 0.5
        assert e.source == "length"

    def test_as_dict(self):
        e = DifficultyEstimate(item_id="x", difficulty=0.8, source="score")
        d = e.as_dict()
        assert d == {"item_id": "x", "difficulty": 0.8, "source": "score"}

    def test_from_dict(self):
        d = {"item_id": "y", "difficulty": 0.3, "source": "complexity"}
        e = DifficultyEstimate.from_dict(d)
        assert e.item_id == "y"
        assert e.difficulty == 0.3
        assert e.source == "complexity"

    def test_roundtrip(self):
        original = DifficultyEstimate(item_id="rt", difficulty=0.77, source="length")
        restored = DifficultyEstimate.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.difficulty == original.difficulty
        assert restored.source == original.source


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


class TestCurriculumConfig:
    def test_defaults(self):
        c = CurriculumConfig()
        assert c.order == "easy_first"
        assert c.early_stop_threshold == 0.5
        assert c.batch_size == 10

    def test_custom_values(self):
        c = CurriculumConfig(order="hard_first", early_stop_threshold=0.3, batch_size=5)
        assert c.order == "hard_first"
        assert c.early_stop_threshold == 0.3
        assert c.batch_size == 5

    def test_as_dict(self):
        c = CurriculumConfig(order="hard_first", batch_size=20)
        d = c.as_dict()
        assert d["order"] == "hard_first"
        assert d["batch_size"] == 20

    def test_from_dict(self):
        d = {"order": "hard_first", "early_stop_threshold": 0.7, "batch_size": 15}
        c = CurriculumConfig.from_dict(d)
        assert c.order == "hard_first"
        assert c.early_stop_threshold == 0.7
        assert c.batch_size == 15

    def test_from_dict_defaults(self):
        c = CurriculumConfig.from_dict({})
        assert c.order == "easy_first"
        assert c.early_stop_threshold == 0.5
        assert c.batch_size == 10

    def test_roundtrip(self):
        original = CurriculumConfig(order="hard_first", batch_size=3)
        restored = CurriculumConfig.from_dict(original.as_dict())
        assert restored.order == original.order
        assert restored.early_stop_threshold == original.early_stop_threshold
        assert restored.batch_size == original.batch_size


# ---------------------------------------------------------------------------
# CurriculumBatch
# ---------------------------------------------------------------------------


class TestCurriculumBatch:
    def test_basic_creation(self):
        b = CurriculumBatch(batch_index=0, item_ids=["a", "b"])
        assert b.batch_index == 0
        assert b.item_ids == ["a", "b"]
        assert b.pass_rate == 0.0
        assert b.should_stop is False

    def test_difficulty_range_default(self):
        b = CurriculumBatch(batch_index=1)
        assert b.difficulty_range == (0.0, 0.0)

    def test_as_dict(self):
        b = CurriculumBatch(
            batch_index=2,
            item_ids=["x"],
            difficulty_range=(0.1, 0.9),
            pass_rate=0.75,
            should_stop=True,
        )
        d = b.as_dict()
        assert d["batch_index"] == 2
        assert d["difficulty_range"] == [0.1, 0.9]
        assert d["pass_rate"] == 0.75
        assert d["should_stop"] is True

    def test_from_dict(self):
        d = {
            "batch_index": 3,
            "item_ids": ["a", "b", "c"],
            "difficulty_range": [0.2, 0.8],
            "pass_rate": 0.6,
            "should_stop": False,
        }
        b = CurriculumBatch.from_dict(d)
        assert b.batch_index == 3
        assert b.difficulty_range == (0.2, 0.8)
        assert b.pass_rate == 0.6

    def test_roundtrip(self):
        original = CurriculumBatch(
            batch_index=1,
            item_ids=["q", "r"],
            difficulty_range=(0.3, 0.7),
            pass_rate=0.55,
        )
        restored = CurriculumBatch.from_dict(original.as_dict())
        assert restored.batch_index == original.batch_index
        assert restored.item_ids == original.item_ids
        assert restored.difficulty_range == original.difficulty_range
        assert restored.pass_rate == original.pass_rate


# ---------------------------------------------------------------------------
# CurriculumResult
# ---------------------------------------------------------------------------


class TestCurriculumResult:
    def test_defaults(self):
        r = CurriculumResult()
        assert r.batches == []
        assert r.total_items == 0
        assert r.items_evaluated == 0
        assert r.stopped_early is False
        assert r.final_batch_index == 0

    def test_as_dict(self):
        batch = CurriculumBatch(batch_index=0, item_ids=["a"])
        r = CurriculumResult(
            batches=[batch], total_items=5, items_evaluated=1, final_batch_index=0
        )
        d = r.as_dict()
        assert len(d["batches"]) == 1
        assert d["total_items"] == 5

    def test_from_dict(self):
        d = {
            "batches": [{"batch_index": 0, "item_ids": ["x"]}],
            "total_items": 10,
            "items_evaluated": 5,
            "stopped_early": True,
            "final_batch_index": 2,
        }
        r = CurriculumResult.from_dict(d)
        assert len(r.batches) == 1
        assert r.stopped_early is True
        assert r.final_batch_index == 2

    def test_roundtrip(self):
        batch = CurriculumBatch(batch_index=0, item_ids=["a", "b"])
        original = CurriculumResult(
            batches=[batch], total_items=2, items_evaluated=2, final_batch_index=0
        )
        restored = CurriculumResult.from_dict(original.as_dict())
        assert restored.total_items == original.total_items
        assert len(restored.batches) == len(original.batches)


# ---------------------------------------------------------------------------
# estimate_difficulty_by_length
# ---------------------------------------------------------------------------


class TestEstimateDifficultyByLength:
    def test_empty_input(self):
        assert estimate_difficulty_by_length({}) == []

    def test_single_item(self):
        result = estimate_difficulty_by_length({"a": "hello"})
        assert len(result) == 1
        assert result[0].difficulty == 0.0  # single item, no span

    def test_two_items(self):
        result = estimate_difficulty_by_length({"short": "hi", "long": "hello world"})
        by_id = {e.item_id: e for e in result}
        assert by_id["short"].difficulty < by_id["long"].difficulty

    def test_normalized_range(self):
        items = {"a": "x", "b": "xx", "c": "xxxxx"}
        result = estimate_difficulty_by_length(items)
        diffs = [e.difficulty for e in result]
        assert min(diffs) == 0.0
        assert max(diffs) == 1.0

    def test_source_is_length(self):
        result = estimate_difficulty_by_length({"a": "text"})
        assert all(e.source == "length" for e in result)

    def test_equal_length_items(self):
        result = estimate_difficulty_by_length({"a": "abc", "b": "def"})
        assert all(e.difficulty == 0.0 for e in result)


# ---------------------------------------------------------------------------
# estimate_difficulty_by_complexity
# ---------------------------------------------------------------------------


class TestEstimateDifficultyByComplexity:
    def test_empty_input(self):
        assert estimate_difficulty_by_complexity({}) == []

    def test_single_item(self):
        result = estimate_difficulty_by_complexity({"a": "Hello world."})
        assert len(result) == 1
        assert result[0].difficulty == 0.0

    def test_simple_vs_complex(self):
        items = {
            "simple": "Hi.",
            "complex": "The extraordinarily sophisticated methodology requires comprehensive understanding. "
            "Furthermore, implementation necessitates careful consideration.",
        }
        result = estimate_difficulty_by_complexity(items)
        by_id = {e.item_id: e for e in result}
        assert by_id["simple"].difficulty < by_id["complex"].difficulty

    def test_source_is_complexity(self):
        result = estimate_difficulty_by_complexity({"a": "test text"})
        assert all(e.source == "complexity" for e in result)

    def test_difficulty_in_range(self):
        items = {"a": "Short.", "b": "Medium length sentence.", "c": "A very long and complex sentence with many words."}
        result = estimate_difficulty_by_complexity(items)
        for e in result:
            assert 0.0 <= e.difficulty <= 1.0


# ---------------------------------------------------------------------------
# estimate_difficulty_by_scores
# ---------------------------------------------------------------------------


class TestEstimateDifficultyByScores:
    def test_empty_input(self):
        assert estimate_difficulty_by_scores({}) == []

    def test_high_score_is_easy(self):
        result = estimate_difficulty_by_scores({"a": 1.0, "b": 0.0})
        by_id = {e.item_id: e for e in result}
        assert by_id["a"].difficulty == 0.0  # high score = easy
        assert by_id["b"].difficulty == 1.0  # low score = hard

    def test_equal_scores(self):
        result = estimate_difficulty_by_scores({"a": 0.5, "b": 0.5})
        assert all(e.difficulty == 0.0 for e in result)

    def test_source_is_score(self):
        result = estimate_difficulty_by_scores({"a": 0.8})
        assert all(e.source == "score" for e in result)

    def test_mid_range_scores(self):
        result = estimate_difficulty_by_scores({"a": 0.0, "b": 0.5, "c": 1.0})
        by_id = {e.item_id: e for e in result}
        assert by_id["b"].difficulty == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# order_curriculum
# ---------------------------------------------------------------------------


class TestOrderCurriculum:
    def test_easy_first(self):
        estimates = [
            DifficultyEstimate("hard", 0.9, "length"),
            DifficultyEstimate("easy", 0.1, "length"),
            DifficultyEstimate("mid", 0.5, "length"),
        ]
        config = CurriculumConfig(order="easy_first")
        result = order_curriculum(estimates, config)
        assert result == ["easy", "mid", "hard"]

    def test_hard_first(self):
        estimates = [
            DifficultyEstimate("hard", 0.9, "length"),
            DifficultyEstimate("easy", 0.1, "length"),
        ]
        config = CurriculumConfig(order="hard_first")
        result = order_curriculum(estimates, config)
        assert result == ["hard", "easy"]

    def test_empty_estimates(self):
        config = CurriculumConfig()
        assert order_curriculum([], config) == []

    def test_single_item(self):
        estimates = [DifficultyEstimate("only", 0.5, "length")]
        config = CurriculumConfig()
        assert order_curriculum(estimates, config) == ["only"]


# ---------------------------------------------------------------------------
# create_batches
# ---------------------------------------------------------------------------


class TestCreateBatches:
    def test_empty(self):
        assert create_batches([], []) == []

    def test_single_batch(self):
        estimates = [
            DifficultyEstimate("a", 0.1, "length"),
            DifficultyEstimate("b", 0.5, "length"),
        ]
        batches = create_batches(["a", "b"], estimates, batch_size=10)
        assert len(batches) == 1
        assert batches[0].item_ids == ["a", "b"]

    def test_multiple_batches(self):
        estimates = [DifficultyEstimate(str(i), i / 10.0, "length") for i in range(5)]
        ids = [str(i) for i in range(5)]
        batches = create_batches(ids, estimates, batch_size=2)
        assert len(batches) == 3  # 2+2+1

    def test_batch_indices(self):
        estimates = [DifficultyEstimate(str(i), 0.5, "length") for i in range(6)]
        ids = [str(i) for i in range(6)]
        batches = create_batches(ids, estimates, batch_size=2)
        assert [b.batch_index for b in batches] == [0, 1, 2]

    def test_difficulty_range_in_batch(self):
        estimates = [
            DifficultyEstimate("a", 0.1, "length"),
            DifficultyEstimate("b", 0.9, "length"),
        ]
        batches = create_batches(["a", "b"], estimates, batch_size=10)
        assert batches[0].difficulty_range == (0.1, 0.9)

    def test_batch_size_one(self):
        estimates = [DifficultyEstimate("a", 0.5, "length")]
        batches = create_batches(["a"], estimates, batch_size=1)
        assert len(batches) == 1
        assert batches[0].item_ids == ["a"]


# ---------------------------------------------------------------------------
# check_early_stop
# ---------------------------------------------------------------------------


class TestCheckEarlyStop:
    def test_below_threshold(self):
        batch = CurriculumBatch(batch_index=0, pass_rate=0.3)
        config = CurriculumConfig(early_stop_threshold=0.5)
        assert check_early_stop(batch, config) is True

    def test_above_threshold(self):
        batch = CurriculumBatch(batch_index=0, pass_rate=0.7)
        config = CurriculumConfig(early_stop_threshold=0.5)
        assert check_early_stop(batch, config) is False

    def test_equal_threshold(self):
        batch = CurriculumBatch(batch_index=0, pass_rate=0.5)
        config = CurriculumConfig(early_stop_threshold=0.5)
        assert check_early_stop(batch, config) is False

    def test_zero_pass_rate(self):
        batch = CurriculumBatch(batch_index=0, pass_rate=0.0)
        config = CurriculumConfig(early_stop_threshold=0.1)
        assert check_early_stop(batch, config) is True


# ---------------------------------------------------------------------------
# run_curriculum
# ---------------------------------------------------------------------------


class TestRunCurriculum:
    def test_basic_run(self):
        items = {"a": "short", "b": "medium text", "c": "a much longer text here"}

        def evaluate_fn(ids):
            return {i: 1.0 for i in ids}

        result = run_curriculum(items, evaluate_fn)
        assert result.total_items == 3
        assert result.items_evaluated == 3
        assert result.stopped_early is False

    def test_early_stop(self):
        items = {str(i): "x" * (i + 1) for i in range(20)}

        def evaluate_fn(ids):
            # All fail
            return {i: 0.0 for i in ids}

        config = CurriculumConfig(batch_size=5, early_stop_threshold=0.5)
        result = run_curriculum(items, evaluate_fn, config)
        assert result.stopped_early is True
        assert result.items_evaluated == 5  # only first batch

    def test_no_early_stop_when_passing(self):
        items = {str(i): "x" * (i + 1) for i in range(10)}

        def evaluate_fn(ids):
            return {i: 1.0 for i in ids}

        config = CurriculumConfig(batch_size=5)
        result = run_curriculum(items, evaluate_fn, config)
        assert result.stopped_early is False
        assert result.items_evaluated == 10

    def test_custom_config(self):
        items = {"a": "short", "b": "longer text here"}

        def evaluate_fn(ids):
            return {i: 0.8 for i in ids}

        config = CurriculumConfig(order="hard_first", batch_size=1)
        result = run_curriculum(items, evaluate_fn, config)
        assert result.total_items == 2

    def test_empty_items(self):
        def evaluate_fn(ids):
            return {}

        result = run_curriculum({}, evaluate_fn)
        assert result.total_items == 0
        assert result.items_evaluated == 0

    def test_default_config(self):
        items = {"a": "hello"}

        def evaluate_fn(ids):
            return {i: 0.9 for i in ids}

        result = run_curriculum(items, evaluate_fn)
        assert result.total_items == 1

    def test_batches_recorded(self):
        items = {str(i): "x" * (i + 1) for i in range(6)}

        def evaluate_fn(ids):
            return {i: 0.7 for i in ids}

        config = CurriculumConfig(batch_size=2)
        result = run_curriculum(items, evaluate_fn, config)
        assert len(result.batches) == 3

    def test_pass_rate_calculated(self):
        items = {"a": "x", "b": "xx"}

        def evaluate_fn(ids):
            # One passes, one fails
            scores = {}
            for i, item_id in enumerate(ids):
                scores[item_id] = 1.0 if i == 0 else 0.0
            return scores

        config = CurriculumConfig(batch_size=2, early_stop_threshold=0.0)
        result = run_curriculum(items, evaluate_fn, config)
        assert result.batches[0].pass_rate == 0.5


# ---------------------------------------------------------------------------
# format_curriculum_report
# ---------------------------------------------------------------------------


class TestFormatCurriculumReport:
    def test_basic_report(self):
        batch = CurriculumBatch(
            batch_index=0,
            item_ids=["a", "b"],
            difficulty_range=(0.1, 0.9),
            pass_rate=0.75,
        )
        result = CurriculumResult(
            batches=[batch],
            total_items=2,
            items_evaluated=2,
            final_batch_index=0,
        )
        report = format_curriculum_report(result)
        assert "Curriculum Evaluation Report" in report
        assert "Total items: 2" in report
        assert "75.00%" in report

    def test_early_stop_in_report(self):
        batch = CurriculumBatch(
            batch_index=0,
            item_ids=["a"],
            difficulty_range=(0.5, 0.5),
            pass_rate=0.3,
            should_stop=True,
        )
        result = CurriculumResult(
            batches=[batch],
            total_items=5,
            items_evaluated=1,
            stopped_early=True,
            final_batch_index=0,
        )
        report = format_curriculum_report(result)
        assert "Early stop triggered" in report
        assert "Stopped early: True" in report

    def test_empty_result(self):
        result = CurriculumResult()
        report = format_curriculum_report(result)
        assert "Total items: 0" in report

    def test_multiple_batches_in_report(self):
        batches = [
            CurriculumBatch(batch_index=i, item_ids=[str(i)], difficulty_range=(0.0, 1.0))
            for i in range(3)
        ]
        result = CurriculumResult(
            batches=batches, total_items=3, items_evaluated=3, final_batch_index=2
        )
        report = format_curriculum_report(result)
        assert "Batch 0:" in report
        assert "Batch 1:" in report
        assert "Batch 2:" in report
