"""Tests for parallel_simulation module."""

from __future__ import annotations

import time

import pytest

from evalyn_sdk.parallel_simulation import (
    GenerationProgress,
    ParallelConfig,
    ParallelResult,
    create_batches,
    format_parallel_report,
    format_progress_bar,
    generate_parallel,
    generate_sequential,
    merge_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simple_gen(n: int) -> list[dict]:
    """Generate n simple dicts."""
    return [{"id": i, "value": f"item_{i}"} for i in range(n)]


def failing_gen(n: int) -> list[dict]:
    """Always raises."""
    raise ValueError("intentional failure")


def partial_gen(n: int) -> list[dict]:
    """Raises on batches of size 5."""
    if n == 5:
        raise RuntimeError("bad batch size 5")
    return [{"ok": True} for _ in range(n)]


# ---------------------------------------------------------------------------
# ParallelConfig
# ---------------------------------------------------------------------------


class TestParallelConfig:
    def test_defaults(self):
        c = ParallelConfig()
        assert c.workers == 4
        assert c.batch_size == 10
        assert c.total_target == 100
        assert c.timeout_seconds == 300.0

    def test_custom_values(self):
        c = ParallelConfig(workers=8, batch_size=20, total_target=200, timeout_seconds=60.0)
        assert c.workers == 8
        assert c.total_target == 200

    def test_as_dict(self):
        c = ParallelConfig(workers=2, batch_size=5)
        d = c.as_dict()
        assert d["workers"] == 2
        assert d["batch_size"] == 5
        assert "total_target" in d
        assert "timeout_seconds" in d

    def test_from_dict(self):
        d = {"workers": 3, "batch_size": 7, "total_target": 50, "timeout_seconds": 120.0}
        c = ParallelConfig.from_dict(d)
        assert c.workers == 3
        assert c.batch_size == 7

    def test_from_dict_defaults(self):
        c = ParallelConfig.from_dict({})
        assert c.workers == 4
        assert c.batch_size == 10

    def test_roundtrip(self):
        c = ParallelConfig(workers=6, batch_size=15, total_target=300, timeout_seconds=60.0)
        c2 = ParallelConfig.from_dict(c.as_dict())
        assert c2.workers == c.workers
        assert c2.batch_size == c.batch_size
        assert c2.total_target == c.total_target
        assert c2.timeout_seconds == c.timeout_seconds


# ---------------------------------------------------------------------------
# GenerationProgress
# ---------------------------------------------------------------------------


class TestGenerationProgress:
    def test_creation(self):
        p = GenerationProgress(generated=50, failed=2, total_target=100, elapsed_seconds=5.0, items_per_second=10.0)
        assert p.generated == 50
        assert p.failed == 2

    def test_as_dict(self):
        p = GenerationProgress(10, 0, 10, 1.0, 10.0)
        d = p.as_dict()
        assert d["generated"] == 10
        assert d["items_per_second"] == 10.0

    def test_from_dict(self):
        d = {"generated": 5, "failed": 1, "total_target": 10, "elapsed_seconds": 2.0, "items_per_second": 2.5}
        p = GenerationProgress.from_dict(d)
        assert p.generated == 5
        assert p.failed == 1

    def test_roundtrip(self):
        p = GenerationProgress(42, 3, 100, 7.5, 5.6)
        p2 = GenerationProgress.from_dict(p.as_dict())
        assert p2.generated == p.generated
        assert p2.items_per_second == p.items_per_second


# ---------------------------------------------------------------------------
# ParallelResult
# ---------------------------------------------------------------------------


class TestParallelResult:
    def test_empty_default(self):
        r = ParallelResult()
        assert r.items == []
        assert r.errors == []
        assert r.progress.generated == 0

    def test_as_dict(self):
        p = GenerationProgress(5, 0, 5, 1.0, 5.0)
        r = ParallelResult(items=[{"a": 1}], progress=p, errors=[])
        d = r.as_dict()
        assert len(d["items"]) == 1
        assert d["progress"]["generated"] == 5

    def test_from_dict(self):
        d = {
            "items": [{"x": 1}],
            "progress": {"generated": 1, "failed": 0, "total_target": 1, "elapsed_seconds": 0.1, "items_per_second": 10.0},
            "errors": ["oops"],
        }
        r = ParallelResult.from_dict(d)
        assert len(r.items) == 1
        assert r.errors == ["oops"]

    def test_roundtrip(self):
        p = GenerationProgress(10, 2, 12, 3.0, 3.33)
        r = ParallelResult(items=[{"k": "v"}], progress=p, errors=["err1"])
        r2 = ParallelResult.from_dict(r.as_dict())
        assert r2.items == r.items
        assert r2.errors == r.errors
        assert r2.progress.generated == r.progress.generated


# ---------------------------------------------------------------------------
# create_batches
# ---------------------------------------------------------------------------


class TestCreateBatches:
    def test_even_split(self):
        assert create_batches(20, 10) == [10, 10]

    def test_uneven_split(self):
        assert create_batches(25, 10) == [10, 10, 5]

    def test_single_batch(self):
        assert create_batches(5, 10) == [5]

    def test_zero_total(self):
        assert create_batches(0, 10) == []

    def test_zero_batch_size(self):
        assert create_batches(10, 0) == []

    def test_negative(self):
        assert create_batches(-5, 10) == []

    def test_one_by_one(self):
        assert create_batches(3, 1) == [1, 1, 1]

    def test_exact_single(self):
        assert create_batches(10, 10) == [10]


# ---------------------------------------------------------------------------
# generate_parallel
# ---------------------------------------------------------------------------


class TestGenerateParallel:
    def test_basic(self):
        cfg = ParallelConfig(workers=2, batch_size=5, total_target=10)
        result = generate_parallel(simple_gen, cfg)
        assert result.progress.generated == 10
        assert result.progress.failed == 0
        assert len(result.items) == 10
        assert result.errors == []

    def test_all_fail(self):
        cfg = ParallelConfig(workers=2, batch_size=5, total_target=10)
        result = generate_parallel(failing_gen, cfg)
        assert result.progress.generated == 0
        assert result.progress.failed == 10
        assert len(result.errors) == 2

    def test_single_worker(self):
        cfg = ParallelConfig(workers=1, batch_size=3, total_target=9)
        result = generate_parallel(simple_gen, cfg)
        assert result.progress.generated == 9

    def test_elapsed_positive(self):
        cfg = ParallelConfig(workers=2, batch_size=5, total_target=10)
        result = generate_parallel(simple_gen, cfg)
        assert result.progress.elapsed_seconds >= 0

    def test_items_per_second(self):
        cfg = ParallelConfig(workers=2, batch_size=5, total_target=10)
        result = generate_parallel(simple_gen, cfg)
        assert result.progress.items_per_second >= 0

    def test_large_batch(self):
        cfg = ParallelConfig(workers=4, batch_size=50, total_target=200)
        result = generate_parallel(simple_gen, cfg)
        assert result.progress.generated == 200


# ---------------------------------------------------------------------------
# generate_sequential
# ---------------------------------------------------------------------------


class TestGenerateSequential:
    def test_basic(self):
        cfg = ParallelConfig(workers=2, batch_size=5, total_target=10)
        result = generate_sequential(simple_gen, cfg)
        assert result.progress.generated == 10
        assert len(result.items) == 10

    def test_all_fail(self):
        cfg = ParallelConfig(workers=1, batch_size=5, total_target=10)
        result = generate_sequential(failing_gen, cfg)
        assert result.progress.generated == 0
        assert result.progress.failed == 10
        assert len(result.errors) == 2

    def test_partial_failure(self):
        cfg = ParallelConfig(workers=1, batch_size=5, total_target=15)
        result = generate_sequential(partial_gen, cfg)
        # 3 batches of 5: all fail since all are size 5
        assert result.progress.failed == 15
        assert result.progress.generated == 0

    def test_preserves_order(self):
        counter = [0]

        def ordered_gen(n):
            items = [{"seq": counter[0] + i} for i in range(n)]
            counter[0] += n
            return items

        cfg = ParallelConfig(workers=1, batch_size=3, total_target=6)
        result = generate_sequential(ordered_gen, cfg)
        seqs = [item["seq"] for item in result.items]
        assert seqs == [0, 1, 2, 3, 4, 5]

    def test_elapsed_tracked(self):
        cfg = ParallelConfig(workers=1, batch_size=5, total_target=5)
        result = generate_sequential(simple_gen, cfg)
        assert result.progress.elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


class TestMergeResults:
    def test_empty_list(self):
        merged = merge_results([])
        assert merged.items == []
        assert merged.progress.generated == 0

    def test_single_result(self):
        p = GenerationProgress(5, 0, 5, 1.0, 5.0)
        r = ParallelResult(items=[{"a": 1}], progress=p, errors=[])
        merged = merge_results([r])
        assert merged.progress.generated == 5
        assert len(merged.items) == 1

    def test_multiple_results(self):
        p1 = GenerationProgress(5, 1, 6, 1.0, 5.0)
        p2 = GenerationProgress(10, 0, 10, 2.0, 5.0)
        r1 = ParallelResult(items=[{"a": 1}] * 5, progress=p1, errors=["e1"])
        r2 = ParallelResult(items=[{"b": 2}] * 10, progress=p2, errors=[])
        merged = merge_results([r1, r2])
        assert merged.progress.generated == 15
        assert merged.progress.failed == 1
        assert merged.progress.total_target == 16
        assert len(merged.items) == 15
        assert len(merged.errors) == 1

    def test_elapsed_summed(self):
        p1 = GenerationProgress(5, 0, 5, 2.0, 2.5)
        p2 = GenerationProgress(5, 0, 5, 3.0, 1.67)
        r1 = ParallelResult(items=[{}] * 5, progress=p1)
        r2 = ParallelResult(items=[{}] * 5, progress=p2)
        merged = merge_results([r1, r2])
        assert merged.progress.elapsed_seconds == pytest.approx(5.0, abs=0.01)

    def test_errors_merged(self):
        p1 = GenerationProgress(0, 5, 5, 1.0, 0.0)
        p2 = GenerationProgress(0, 5, 5, 1.0, 0.0)
        r1 = ParallelResult(items=[], progress=p1, errors=["e1", "e2"])
        r2 = ParallelResult(items=[], progress=p2, errors=["e3"])
        merged = merge_results([r1, r2])
        assert len(merged.errors) == 3


# ---------------------------------------------------------------------------
# format_progress_bar
# ---------------------------------------------------------------------------


class TestFormatProgressBar:
    def test_zero_progress(self):
        p = GenerationProgress(0, 0, 100, 0.0, 0.0)
        bar = format_progress_bar(p, width=20)
        assert "0/100" in bar
        assert "[" in bar and "]" in bar

    def test_half_progress(self):
        p = GenerationProgress(50, 0, 100, 5.0, 10.0)
        bar = format_progress_bar(p, width=40)
        assert "50/100" in bar
        assert "10.0 items/s" in bar

    def test_complete(self):
        p = GenerationProgress(100, 0, 100, 10.0, 10.0)
        bar = format_progress_bar(p, width=20)
        assert "100/100" in bar
        assert "=" * 20 in bar

    def test_zero_target(self):
        p = GenerationProgress(0, 0, 0, 0.0, 0.0)
        bar = format_progress_bar(p, width=10)
        assert "0/0" in bar

    def test_custom_width(self):
        p = GenerationProgress(25, 0, 100, 2.5, 10.0)
        bar = format_progress_bar(p, width=10)
        assert "[" in bar and "]" in bar


# ---------------------------------------------------------------------------
# format_parallel_report
# ---------------------------------------------------------------------------


class TestFormatParallelReport:
    def test_success_report(self):
        p = GenerationProgress(100, 0, 100, 5.0, 20.0)
        r = ParallelResult(items=[{}] * 100, progress=p, errors=[])
        report = format_parallel_report(r)
        assert "100/100" in report
        assert "Failed" in report
        assert "20.0 items/s" in report
        assert "Errors" not in report

    def test_report_with_errors(self):
        p = GenerationProgress(80, 20, 100, 5.0, 16.0)
        r = ParallelResult(items=[{}] * 80, progress=p, errors=["batch failed", "timeout"])
        report = format_parallel_report(r)
        assert "80/100" in report
        assert "Errors (2)" in report
        assert "batch failed" in report
        assert "timeout" in report

    def test_report_header(self):
        p = GenerationProgress(0, 0, 0, 0.0, 0.0)
        r = ParallelResult(progress=p)
        report = format_parallel_report(r)
        assert "Parallel Generation Report" in report


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_parallel_and_sequential_same_count(self):
        cfg = ParallelConfig(workers=2, batch_size=5, total_target=20)
        par = generate_parallel(simple_gen, cfg)
        seq = generate_sequential(simple_gen, cfg)
        assert par.progress.generated == seq.progress.generated == 20

    def test_merge_after_parallel(self):
        cfg = ParallelConfig(workers=2, batch_size=5, total_target=10)
        r1 = generate_parallel(simple_gen, cfg)
        r2 = generate_sequential(simple_gen, cfg)
        merged = merge_results([r1, r2])
        assert merged.progress.generated == 20
