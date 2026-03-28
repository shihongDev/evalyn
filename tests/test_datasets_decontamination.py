"""Tests for datasets_decontamination module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.datasets_decontamination import (
    ContaminationMatch,
    DecontaminationConfig,
    DecontaminationReport,
    compute_ngrams,
    compute_ngram_similarity,
    check_contamination,
    decontaminate_dataset,
    get_known_benchmark_names,
)


# ---------------------------------------------------------------------------
# ContaminationMatch
# ---------------------------------------------------------------------------


class TestContaminationMatch:
    def test_as_dict(self):
        m = ContaminationMatch(
            item_id="item_0",
            benchmark_name="MMLU",
            similarity=0.95,
            matched_text="some text",
        )
        d = m.as_dict()
        assert d["item_id"] == "item_0"
        assert d["benchmark_name"] == "MMLU"
        assert d["similarity"] == 0.95
        assert d["matched_text"] == "some text"

    def test_as_dict_default_matched_text(self):
        m = ContaminationMatch(item_id="x", benchmark_name="GSM8K", similarity=0.8)
        assert m.as_dict()["matched_text"] == ""


# ---------------------------------------------------------------------------
# DecontaminationConfig
# ---------------------------------------------------------------------------


class TestDecontaminationConfig:
    def test_defaults(self):
        cfg = DecontaminationConfig()
        assert cfg.min_similarity == 0.8
        assert cfg.ngram_size == 5
        assert cfg.check_input is True
        assert cfg.check_output is True

    def test_as_dict_roundtrip(self):
        cfg = DecontaminationConfig(min_similarity=0.7, ngram_size=3)
        restored = DecontaminationConfig.from_dict(cfg.as_dict())
        assert restored.min_similarity == 0.7
        assert restored.ngram_size == 3

    def test_from_dict_defaults(self):
        cfg = DecontaminationConfig.from_dict({})
        assert cfg.min_similarity == 0.8
        assert cfg.ngram_size == 5


# ---------------------------------------------------------------------------
# DecontaminationReport
# ---------------------------------------------------------------------------


class TestDecontaminationReport:
    def test_as_dict_empty(self):
        r = DecontaminationReport()
        d = r.as_dict()
        assert d["matches"] == []
        assert d["clean_count"] == 0
        assert d["contamination_rate"] == 0.0

    def test_from_dict_roundtrip(self):
        original = DecontaminationReport(
            matches=[
                ContaminationMatch(item_id="0", benchmark_name="MMLU", similarity=0.9)
            ],
            clean_count=9,
            contaminated_count=1,
            contamination_rate=0.1,
            benchmarks_checked=["MMLU"],
        )
        restored = DecontaminationReport.from_dict(original.as_dict())
        assert restored.clean_count == 9
        assert restored.contaminated_count == 1
        assert len(restored.matches) == 1
        assert restored.matches[0].benchmark_name == "MMLU"

    def test_format_text_basic(self):
        r = DecontaminationReport(
            clean_count=8,
            contaminated_count=2,
            contamination_rate=0.2,
            benchmarks_checked=["MMLU", "GSM8K"],
        )
        text = r.format_text()
        assert "10 items checked" in text
        assert "Clean: 8" in text
        assert "20.0%" in text
        assert "MMLU" in text

    def test_format_text_with_matches(self):
        r = DecontaminationReport(
            matches=[
                ContaminationMatch(item_id="item_3", benchmark_name="MMLU", similarity=0.92)
            ],
            clean_count=9,
            contaminated_count=1,
            contamination_rate=0.1,
            benchmarks_checked=["MMLU"],
        )
        text = r.format_text()
        assert "item_3" in text
        assert "0.92" in text


# ---------------------------------------------------------------------------
# compute_ngrams
# ---------------------------------------------------------------------------


class TestComputeNgrams:
    def test_basic(self):
        ngrams = compute_ngrams("abcdef", 3)
        assert "abc" in ngrams
        assert "def" in ngrams
        assert len(ngrams) == 4  # abc, bcd, cde, def

    def test_text_shorter_than_n(self):
        ngrams = compute_ngrams("ab", 5)
        assert ngrams == set()

    def test_exact_length(self):
        ngrams = compute_ngrams("abcde", 5)
        assert ngrams == {"abcde"}

    def test_single_char_ngrams(self):
        ngrams = compute_ngrams("hello", 1)
        assert "h" in ngrams
        assert "e" in ngrams
        assert "l" in ngrams
        assert "o" in ngrams

    def test_empty_string(self):
        ngrams = compute_ngrams("", 5)
        assert ngrams == set()


# ---------------------------------------------------------------------------
# compute_ngram_similarity
# ---------------------------------------------------------------------------


class TestComputeNgramSimilarity:
    def test_identical_texts(self):
        text = "The quick brown fox jumps over the lazy dog"
        sim = compute_ngram_similarity(text, text)
        assert sim == 1.0

    def test_no_overlap(self):
        sim = compute_ngram_similarity("aaaaa", "bbbbb", 5)
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = compute_ngram_similarity("abcdefgh", "abcdefxy", 5)
        assert 0.0 < sim < 1.0

    def test_short_text_returns_zero(self):
        sim = compute_ngram_similarity("ab", "abcdef", 5)
        assert sim == 0.0

    def test_both_short(self):
        sim = compute_ngram_similarity("ab", "cd", 5)
        assert sim == 0.0

    def test_symmetry(self):
        a = "hello world foo"
        b = "hello world bar"
        assert compute_ngram_similarity(a, b) == compute_ngram_similarity(b, a)


# ---------------------------------------------------------------------------
# check_contamination
# ---------------------------------------------------------------------------


class TestCheckContamination:
    def test_finds_match(self):
        text = "What is 2 + 2? The answer is 4."
        benchmark_texts = ["What is 2 + 2? The answer is 4."]
        config = DecontaminationConfig(min_similarity=0.8)
        matches = check_contamination(text, benchmark_texts, "GSM8K", config)
        assert len(matches) == 1
        assert matches[0].benchmark_name == "GSM8K"
        assert matches[0].similarity == 1.0

    def test_no_match(self):
        text = "Completely unrelated text about cooking pasta"
        benchmark_texts = ["Solve: integral of x^2 dx from 0 to 1"]
        config = DecontaminationConfig(min_similarity=0.8)
        matches = check_contamination(text, benchmark_texts, "MATH", config)
        assert matches == []

    def test_below_threshold(self):
        text = "abcdefghijklmnop"
        benchmark_texts = ["abcdefXXXXXXXXXX"]
        config = DecontaminationConfig(min_similarity=0.99)
        matches = check_contamination(text, benchmark_texts, "test", config)
        assert matches == []

    def test_multiple_benchmarks(self):
        text = "The capital of France is Paris"
        benchmark_texts = [
            "The capital of France is Paris",
            "Something totally different about quantum physics",
        ]
        config = DecontaminationConfig(min_similarity=0.5)
        matches = check_contamination(text, benchmark_texts, "MMLU", config)
        # At least the identical one should match
        assert any(m.similarity == 1.0 for m in matches)


# ---------------------------------------------------------------------------
# decontaminate_dataset
# ---------------------------------------------------------------------------


class TestDecontaminateDataset:
    def test_removes_contaminated(self):
        items = [
            {"input": "What is 2+2?", "id": "clean1"},
            {"input": "Exact benchmark question about photosynthesis in detail", "id": "dirty1"},
        ]
        benchmarks = {
            "MMLU": ["Exact benchmark question about photosynthesis in detail"],
        }
        config = DecontaminationConfig(min_similarity=0.8)
        clean, report = decontaminate_dataset(items, benchmarks, config)
        assert report.contaminated_count == 1
        assert report.clean_count == 1
        assert len(clean) == 1
        assert clean[0]["id"] == "clean1"

    def test_all_clean(self):
        items = [
            {"input": "Completely original question alpha beta gamma"},
            {"input": "Another original question delta epsilon zeta"},
        ]
        benchmarks = {"GSM8K": ["Train leaves station at 9am, how far"]}
        clean, report = decontaminate_dataset(items, benchmarks)
        assert report.contaminated_count == 0
        assert report.clean_count == 2
        assert len(clean) == 2

    def test_empty_items(self):
        clean, report = decontaminate_dataset([], {"MMLU": ["text"]})
        assert clean == []
        assert report.contamination_rate == 0.0
        assert report.clean_count == 0

    def test_empty_benchmarks(self):
        items = [{"input": "some question"}]
        clean, report = decontaminate_dataset(items, {})
        assert len(clean) == 1
        assert report.contaminated_count == 0

    def test_checks_output_field(self):
        items = [
            {"input": "question", "output": "Exact benchmark answer about photosynthesis"},
        ]
        benchmarks = {"MMLU": ["Exact benchmark answer about photosynthesis"]}
        config = DecontaminationConfig(check_input=False, check_output=True)
        clean, report = decontaminate_dataset(items, benchmarks, config)
        assert report.contaminated_count == 1

    def test_respects_check_input_false(self):
        items = [
            {"input": "Exact benchmark question about photosynthesis in detail"},
        ]
        benchmarks = {"MMLU": ["Exact benchmark question about photosynthesis in detail"]}
        config = DecontaminationConfig(check_input=False, check_output=False)
        clean, report = decontaminate_dataset(items, benchmarks, config)
        # Nothing checked, so nothing contaminated
        assert report.contaminated_count == 0

    def test_benchmarks_checked_populated(self):
        items = [{"input": "q"}]
        benchmarks = {"MMLU": ["x"], "GSM8K": ["y"]}
        _, report = decontaminate_dataset(items, benchmarks)
        assert "MMLU" in report.benchmarks_checked
        assert "GSM8K" in report.benchmarks_checked

    def test_contamination_rate(self):
        items = [
            {"input": "Exact benchmark question about photosynthesis in detail", "id": "0"},
            {"input": "original question alpha beta gamma delta", "id": "1"},
        ]
        benchmarks = {"MMLU": ["Exact benchmark question about photosynthesis in detail"]}
        _, report = decontaminate_dataset(items, benchmarks)
        assert report.contamination_rate == 0.5


# ---------------------------------------------------------------------------
# get_known_benchmark_names
# ---------------------------------------------------------------------------


class TestGetKnownBenchmarkNames:
    def test_non_empty(self):
        names = get_known_benchmark_names()
        assert len(names) > 0

    def test_contains_known_benchmarks(self):
        names = get_known_benchmark_names()
        assert "MMLU" in names
        assert "GSM8K" in names
        assert "HumanEval" in names

    def test_returns_strings(self):
        names = get_known_benchmark_names()
        assert all(isinstance(n, str) for n in names)
