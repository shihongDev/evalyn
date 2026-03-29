"""Tests for the large dataset optimization module."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.large_dataset import (
    ChunkConfig,
    CheckpointState,
    ProcessingStats,
    stream_dataset,
    save_checkpoint,
    load_checkpoint,
    should_checkpoint,
    estimate_memory_usage,
    process_in_chunks,
    format_processing_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, items: list[dict]) -> str:
    """Write a list of dicts as JSONL and return the file path string."""
    fp = str(path)
    with open(fp, "w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    return fp


def _identity(chunk: list[dict]) -> list[dict]:
    """Pass-through processor."""
    return chunk


def _failing(chunk: list[dict]) -> list[dict]:
    """Processor that always raises."""
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# ChunkConfig
# ---------------------------------------------------------------------------


class TestChunkConfig:
    def test_defaults(self):
        c = ChunkConfig()
        assert c.chunk_size == 100
        assert c.checkpoint_interval == 500
        assert c.max_memory_mb == 0.0
        assert c.write_batch_size == 50

    def test_custom_values(self):
        c = ChunkConfig(chunk_size=10, checkpoint_interval=20, max_memory_mb=64.0, write_batch_size=5)
        assert c.chunk_size == 10
        assert c.max_memory_mb == 64.0

    def test_as_dict(self):
        c = ChunkConfig(chunk_size=25)
        d = c.as_dict()
        assert d["chunk_size"] == 25
        assert d["checkpoint_interval"] == 500

    def test_from_dict(self):
        d = {"chunk_size": 50, "max_memory_mb": 128.0}
        c = ChunkConfig.from_dict(d)
        assert c.chunk_size == 50
        assert c.max_memory_mb == 128.0
        assert c.checkpoint_interval == 500  # default

    def test_roundtrip(self):
        c = ChunkConfig(chunk_size=7, checkpoint_interval=14, max_memory_mb=3.5, write_batch_size=2)
        assert ChunkConfig.from_dict(c.as_dict()).as_dict() == c.as_dict()


# ---------------------------------------------------------------------------
# CheckpointState
# ---------------------------------------------------------------------------


class TestCheckpointState:
    def test_as_dict(self):
        s = CheckpointState(
            processed_count=100,
            total_count=500,
            last_item_id="item-99",
            checkpoint_path="/tmp/ckpt.json",
            created_at="2026-01-01T00:00:00Z",
        )
        d = s.as_dict()
        assert d["processed_count"] == 100
        assert d["last_item_id"] == "item-99"

    def test_from_dict(self):
        d = {"processed_count": 42, "total_count": 100, "last_item_id": "x", "checkpoint_path": "p", "created_at": "t"}
        s = CheckpointState.from_dict(d)
        assert s.processed_count == 42

    def test_roundtrip(self):
        s = CheckpointState(10, 20, "abc", "/p", "ts")
        assert CheckpointState.from_dict(s.as_dict()).as_dict() == s.as_dict()


# ---------------------------------------------------------------------------
# ProcessingStats
# ---------------------------------------------------------------------------


class TestProcessingStats:
    def test_as_dict(self):
        s = ProcessingStats(100, 95, 5, 1.5, 63.33, 2.1)
        d = s.as_dict()
        assert d["total_items"] == 100
        assert d["failed_items"] == 5
        assert d["peak_memory_mb"] == 2.1

    def test_from_dict(self):
        d = {"total_items": 10, "processed_items": 8, "failed_items": 2, "elapsed_seconds": 0.5, "items_per_second": 16.0}
        s = ProcessingStats.from_dict(d)
        assert s.total_items == 10
        assert s.peak_memory_mb == 0.0  # default

    def test_roundtrip(self):
        s = ProcessingStats(50, 48, 2, 3.14, 15.28, 1.0)
        assert ProcessingStats.from_dict(s.as_dict()).as_dict() == s.as_dict()


# ---------------------------------------------------------------------------
# stream_dataset
# ---------------------------------------------------------------------------


class TestStreamDataset:
    def test_basic_streaming(self, tmp_path):
        items = [{"id": i, "val": i * 10} for i in range(10)]
        fp = _write_jsonl(tmp_path / "data.jsonl", items)
        chunks = list(stream_dataset(fp, chunk_size=3))
        assert len(chunks) == 4  # 3+3+3+1
        assert len(chunks[0]) == 3
        assert len(chunks[-1]) == 1
        assert chunks[0][0]["id"] == 0

    def test_exact_chunk_boundary(self, tmp_path):
        items = [{"id": i} for i in range(6)]
        fp = _write_jsonl(tmp_path / "data.jsonl", items)
        chunks = list(stream_dataset(fp, chunk_size=3))
        assert len(chunks) == 2
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3

    def test_single_chunk(self, tmp_path):
        items = [{"id": i} for i in range(5)]
        fp = _write_jsonl(tmp_path / "data.jsonl", items)
        chunks = list(stream_dataset(fp, chunk_size=100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_empty_file(self, tmp_path):
        fp = str(tmp_path / "empty.jsonl")
        with open(fp, "w") as fh:
            fh.write("")
        chunks = list(stream_dataset(fp, chunk_size=10))
        assert chunks == []

    def test_blank_lines_skipped(self, tmp_path):
        fp = str(tmp_path / "gaps.jsonl")
        with open(fp, "w") as fh:
            fh.write('{"a":1}\n\n\n{"a":2}\n')
        chunks = list(stream_dataset(fp, chunk_size=10))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_malformed_lines_skipped(self, tmp_path):
        fp = str(tmp_path / "bad.jsonl")
        with open(fp, "w") as fh:
            fh.write('{"ok":1}\nNOT JSON\n{"ok":2}\n')
        chunks = list(stream_dataset(fp, chunk_size=10))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_large_dataset(self, tmp_path):
        items = [{"id": i, "data": "x" * 50} for i in range(1000)]
        fp = _write_jsonl(tmp_path / "large.jsonl", items)
        total = sum(len(c) for c in stream_dataset(fp, chunk_size=100))
        assert total == 1000

    def test_chunk_size_one(self, tmp_path):
        items = [{"id": i} for i in range(3)]
        fp = _write_jsonl(tmp_path / "data.jsonl", items)
        chunks = list(stream_dataset(fp, chunk_size=1))
        assert len(chunks) == 3
        assert all(len(c) == 1 for c in chunks)


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_and_load(self, tmp_path):
        cp = str(tmp_path / "ckpt.json")
        state = CheckpointState(50, 100, "item-49", cp, "2026-01-01T00:00:00Z")
        save_checkpoint(state)
        loaded = load_checkpoint(cp)
        assert loaded is not None
        assert loaded.processed_count == 50
        assert loaded.last_item_id == "item-49"

    def test_load_nonexistent(self, tmp_path):
        result = load_checkpoint(str(tmp_path / "nope.json"))
        assert result is None

    def test_nested_directory_created(self, tmp_path):
        cp = str(tmp_path / "sub" / "dir" / "ckpt.json")
        state = CheckpointState(1, 2, "x", cp, "t")
        save_checkpoint(state)
        assert os.path.exists(cp)

    def test_overwrite_existing(self, tmp_path):
        cp = str(tmp_path / "ckpt.json")
        s1 = CheckpointState(10, 100, "a", cp, "t1")
        save_checkpoint(s1)
        s2 = CheckpointState(20, 100, "b", cp, "t2")
        save_checkpoint(s2)
        loaded = load_checkpoint(cp)
        assert loaded is not None
        assert loaded.processed_count == 20


# ---------------------------------------------------------------------------
# should_checkpoint
# ---------------------------------------------------------------------------


class TestShouldCheckpoint:
    def test_at_interval(self):
        assert should_checkpoint(500, 500) is True

    def test_at_multiple(self):
        assert should_checkpoint(1000, 500) is True

    def test_not_at_interval(self):
        assert should_checkpoint(499, 500) is False

    def test_zero_processed(self):
        assert should_checkpoint(0, 500) is False

    def test_zero_interval(self):
        assert should_checkpoint(100, 0) is False

    def test_negative_interval(self):
        assert should_checkpoint(100, -1) is False


# ---------------------------------------------------------------------------
# estimate_memory_usage
# ---------------------------------------------------------------------------


class TestEstimateMemory:
    def test_empty_list(self):
        assert estimate_memory_usage([]) == 0.0

    def test_small_items(self):
        items = [{"key": "value"} for _ in range(10)]
        mb = estimate_memory_usage(items)
        assert mb > 0
        assert mb < 1  # Should be tiny

    def test_larger_items_more_memory(self):
        small = [{"k": "v"} for _ in range(10)]
        big = [{"k": "v" * 10000} for _ in range(10)]
        assert estimate_memory_usage(big) > estimate_memory_usage(small)

    def test_nested_dicts(self):
        items = [{"a": {"b": {"c": "deep"}}} for _ in range(5)]
        mb = estimate_memory_usage(items)
        assert mb > 0


# ---------------------------------------------------------------------------
# process_in_chunks
# ---------------------------------------------------------------------------


class TestProcessInChunks:
    def test_basic_processing(self, tmp_path):
        items = [[{"id": i} for i in range(5)]]
        config = ChunkConfig(chunk_size=5)
        stats = process_in_chunks(iter(items), _identity, config)
        assert stats.processed_items == 5
        assert stats.failed_items == 0
        assert stats.total_items == 5

    def test_multiple_chunks(self, tmp_path):
        chunks = [[{"id": i} for i in range(3)] for _ in range(4)]
        config = ChunkConfig(chunk_size=3)
        stats = process_in_chunks(iter(chunks), _identity, config)
        assert stats.processed_items == 12
        assert stats.total_items == 12

    def test_failure_handling(self, tmp_path):
        chunks = [[{"id": i} for i in range(5)]]
        config = ChunkConfig(chunk_size=5)
        stats = process_in_chunks(iter(chunks), _failing, config)
        assert stats.processed_items == 0
        assert stats.failed_items == 5

    def test_mixed_success_failure(self, tmp_path):
        call_count = {"n": 0}

        def sometimes_fail(chunk):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("fail")
            return chunk

        chunks = [[{"id": i}] for i in range(4)]
        config = ChunkConfig(chunk_size=1)
        stats = process_in_chunks(iter(chunks), sometimes_fail, config)
        assert stats.processed_items == 2
        assert stats.failed_items == 2

    def test_checkpointing_during_processing(self, tmp_path):
        cp = str(tmp_path / "ckpt.json")
        chunks = [[{"id": i}] for i in range(10)]
        config = ChunkConfig(chunk_size=1, checkpoint_interval=5)
        stats = process_in_chunks(iter(chunks), _identity, config, checkpoint_path=cp)
        assert stats.processed_items == 10
        loaded = load_checkpoint(cp)
        assert loaded is not None
        assert loaded.processed_count in (5, 10)

    def test_stats_timing(self, tmp_path):
        chunks = [[{"id": 0}]]
        config = ChunkConfig()
        stats = process_in_chunks(iter(chunks), _identity, config)
        assert stats.elapsed_seconds >= 0
        assert stats.items_per_second >= 0

    def test_empty_input(self, tmp_path):
        config = ChunkConfig()
        stats = process_in_chunks(iter([]), _identity, config)
        assert stats.total_items == 0
        assert stats.processed_items == 0

    def test_items_per_second_calculated(self, tmp_path):
        chunks = [[{"id": i} for i in range(100)]]
        config = ChunkConfig(chunk_size=100)
        stats = process_in_chunks(iter(chunks), _identity, config)
        assert stats.items_per_second > 0

    def test_peak_memory_tracked(self, tmp_path):
        chunks = [[{"id": i, "data": "x" * 100} for i in range(50)]]
        config = ChunkConfig()
        stats = process_in_chunks(iter(chunks), _identity, config)
        assert stats.peak_memory_mb > 0


# ---------------------------------------------------------------------------
# format_processing_stats
# ---------------------------------------------------------------------------


class TestFormatProcessingStats:
    def test_basic_format(self):
        s = ProcessingStats(100, 95, 5, 2.5, 38.0, 1.2)
        text = format_processing_stats(s)
        assert "Processing Stats" in text
        assert "100" in text
        assert "95" in text
        assert "5" in text
        assert "38.00" in text
        assert "1.2" in text

    def test_zero_memory_hidden(self):
        s = ProcessingStats(10, 10, 0, 0.1, 100.0, 0.0)
        text = format_processing_stats(s)
        assert "Peak memory" not in text

    def test_all_fields_present(self):
        s = ProcessingStats(50, 48, 2, 3.0, 16.0, 0.5)
        text = format_processing_stats(s)
        assert "Total items" in text
        assert "Processed" in text
        assert "Failed" in text
        assert "Elapsed" in text
        assert "Items/sec" in text
        assert "Peak memory" in text


# ---------------------------------------------------------------------------
# Integration: stream -> process
# ---------------------------------------------------------------------------


class TestStreamAndProcess:
    def test_end_to_end(self, tmp_path):
        items = [{"id": i, "val": i} for i in range(25)]
        fp = _write_jsonl(tmp_path / "data.jsonl", items)
        config = ChunkConfig(chunk_size=10)
        stats = process_in_chunks(stream_dataset(fp, chunk_size=10), _identity, config)
        assert stats.total_items == 25
        assert stats.processed_items == 25

    def test_end_to_end_with_checkpoint(self, tmp_path):
        items = [{"id": i} for i in range(50)]
        fp = _write_jsonl(tmp_path / "data.jsonl", items)
        cp = str(tmp_path / "ckpt.json")
        config = ChunkConfig(chunk_size=5, checkpoint_interval=10)
        stats = process_in_chunks(
            stream_dataset(fp, chunk_size=5), _identity, config, checkpoint_path=cp
        )
        assert stats.processed_items == 50
