"""
Tests for evalyn_sdk.datasets_cross_contamination.

Run with: uv run pytest tests/test_datasets_cross_contamination.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.datasets_cross_contamination import (
    ContaminationCheckResult,
    LeakageItem,
    check_cross_contamination,
    compute_overlap_rate,
    find_duplicate_items,
    hash_item_content,
    suggest_decontamination,
)


# ---------------------------------------------------------------------------
# hash_item_content
# ---------------------------------------------------------------------------


class TestHashItemContent:
    def test_deterministic(self):
        h1 = hash_item_content("hello world")
        h2 = hash_item_content("hello world")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = hash_item_content("Hello World")
        h2 = hash_item_content("hello world")
        assert h1 == h2

    def test_strips_whitespace(self):
        h1 = hash_item_content("  hello  ")
        h2 = hash_item_content("hello")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = hash_item_content("foo")
        h2 = hash_item_content("bar")
        assert h1 != h2

    def test_empty_string(self):
        h = hash_item_content("")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest length

    def test_returns_hex_string(self):
        h = hash_item_content("test")
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# LeakageItem
# ---------------------------------------------------------------------------


class TestLeakageItem:
    def test_as_dict(self):
        lk = LeakageItem(item_id="i-1", found_in=["train", "test"], input_hash="abc")
        d = lk.as_dict()
        assert d["item_id"] == "i-1"
        assert d["found_in"] == ["train", "test"]
        assert d["input_hash"] == "abc"

    def test_defaults(self):
        lk = LeakageItem(item_id="x")
        assert lk.found_in == []
        assert lk.input_hash == ""


# ---------------------------------------------------------------------------
# ContaminationCheckResult
# ---------------------------------------------------------------------------


class TestContaminationCheckResult:
    def test_as_dict_clean(self):
        r = ContaminationCheckResult(
            total_checked=100, clean=True, splits_checked=["train", "test"]
        )
        d = r.as_dict()
        assert d["clean"] is True
        assert d["leaked_count"] == 0
        assert d["total_checked"] == 100

    def test_from_dict_roundtrip(self):
        r = ContaminationCheckResult(
            leaks=[LeakageItem(item_id="x", found_in=["a", "b"], input_hash="h1")],
            total_checked=50,
            leaked_count=1,
            clean=False,
            splits_checked=["a", "b"],
        )
        restored = ContaminationCheckResult.from_dict(r.as_dict())
        assert restored.leaked_count == 1
        assert restored.clean is False
        assert len(restored.leaks) == 1
        assert restored.leaks[0].item_id == "x"
        assert restored.leaks[0].found_in == ["a", "b"]

    def test_from_dict_defaults(self):
        r = ContaminationCheckResult.from_dict({})
        assert r.clean is True
        assert r.total_checked == 0
        assert r.leaks == []

    def test_format_text_clean(self):
        r = ContaminationCheckResult(
            total_checked=100, clean=True, splits_checked=["train", "test"]
        )
        text = r.format_text()
        assert "CLEAN" in text
        assert "train" in text
        assert "test" in text

    def test_format_text_contaminated(self):
        r = ContaminationCheckResult(
            leaks=[LeakageItem(item_id="leak-1", found_in=["train", "test"])],
            total_checked=100,
            leaked_count=1,
            clean=False,
            splits_checked=["train", "test"],
        )
        text = r.format_text()
        assert "CONTAMINATED" in text
        assert "leak-1" in text
        assert "1 leaked" in text


# ---------------------------------------------------------------------------
# check_cross_contamination
# ---------------------------------------------------------------------------


class TestCheckCrossContamination:
    def test_no_leaks(self):
        splits = {
            "train": [{"input": "a"}, {"input": "b"}],
            "test": [{"input": "c"}, {"input": "d"}],
        }
        result = check_cross_contamination(splits)
        assert result.clean is True
        assert result.leaked_count == 0
        assert result.total_checked == 4

    def test_with_leaks(self):
        splits = {
            "train": [{"input": "shared item"}, {"input": "unique train"}],
            "test": [{"input": "shared item"}, {"input": "unique test"}],
        }
        result = check_cross_contamination(splits)
        assert result.clean is False
        assert result.leaked_count == 1
        assert len(result.leaks) == 1

    def test_case_insensitive_leak(self):
        splits = {
            "train": [{"input": "Hello World"}],
            "test": [{"input": "hello world"}],
        }
        result = check_cross_contamination(splits)
        assert result.clean is False

    def test_three_splits(self):
        splits = {
            "train": [{"input": "x"}],
            "test": [{"input": "y"}],
            "calibration": [{"input": "z"}],
        }
        result = check_cross_contamination(splits)
        assert result.clean is True
        assert sorted(result.splits_checked) == ["calibration", "test", "train"]

    def test_three_splits_with_leak_across_all(self):
        splits = {
            "train": [{"input": "leaked"}],
            "test": [{"input": "leaked"}],
            "calibration": [{"input": "leaked"}],
        }
        result = check_cross_contamination(splits)
        assert result.clean is False
        assert result.leaked_count >= 1
        # The leaked item should appear in all three splits
        assert len(result.leaks[0].found_in) == 3

    def test_empty_splits(self):
        splits = {"train": [], "test": []}
        result = check_cross_contamination(splits)
        assert result.clean is True
        assert result.total_checked == 0

    def test_custom_input_field(self):
        splits = {
            "train": [{"text": "same content"}],
            "test": [{"text": "same content"}],
        }
        result = check_cross_contamination(splits, input_field="text")
        assert result.clean is False

    def test_non_string_input(self):
        splits = {
            "train": [{"input": 42}],
            "test": [{"input": 42}],
        }
        result = check_cross_contamination(splits)
        assert result.clean is False

    def test_missing_input_field(self):
        splits = {
            "train": [{"other": "value"}],
            "test": [{"other": "value2"}],
        }
        # Missing field defaults to empty string - both items hash to same
        result = check_cross_contamination(splits)
        assert result.clean is False  # both hash to hash("")


# ---------------------------------------------------------------------------
# find_duplicate_items
# ---------------------------------------------------------------------------


class TestFindDuplicateItems:
    def test_no_duplicates(self):
        a = [{"input": "x"}, {"input": "y"}]
        b = [{"input": "z"}, {"input": "w"}]
        dupes = find_duplicate_items(a, b)
        assert dupes == []

    def test_with_duplicates(self):
        a = [{"input": "same", "id": "a1"}]
        b = [{"input": "same", "id": "b1"}]
        dupes = find_duplicate_items(a, b)
        assert len(dupes) == 1
        assert dupes[0].item_id == "b1"

    def test_multiple_duplicates(self):
        a = [{"input": "x"}, {"input": "y"}, {"input": "z"}]
        b = [{"input": "x"}, {"input": "y"}, {"input": "w"}]
        dupes = find_duplicate_items(a, b)
        assert len(dupes) == 2

    def test_empty_lists(self):
        assert find_duplicate_items([], []) == []
        assert find_duplicate_items([{"input": "x"}], []) == []
        assert find_duplicate_items([], [{"input": "x"}]) == []

    def test_custom_input_field(self):
        a = [{"text": "hello"}]
        b = [{"text": "hello"}]
        dupes = find_duplicate_items(a, b, input_field="text")
        assert len(dupes) == 1


# ---------------------------------------------------------------------------
# compute_overlap_rate
# ---------------------------------------------------------------------------


class TestComputeOverlapRate:
    def test_no_overlap(self):
        splits = {
            "train": [{"input": "a"}, {"input": "b"}],
            "test": [{"input": "c"}, {"input": "d"}],
        }
        rates = compute_overlap_rate(splits)
        assert rates["test-train"] == 0.0

    def test_full_overlap(self):
        splits = {
            "train": [{"input": "a"}, {"input": "b"}],
            "test": [{"input": "a"}, {"input": "b"}],
        }
        rates = compute_overlap_rate(splits)
        assert rates["test-train"] == 1.0

    def test_partial_overlap(self):
        splits = {
            "train": [{"input": "a"}, {"input": "b"}],
            "test": [{"input": "a"}, {"input": "c"}],
        }
        rates = compute_overlap_rate(splits)
        assert rates["test-train"] == 0.5

    def test_empty_splits(self):
        splits = {"train": [], "test": []}
        rates = compute_overlap_rate(splits)
        assert rates["test-train"] == 0.0

    def test_three_splits(self):
        splits = {
            "train": [{"input": "a"}],
            "test": [{"input": "b"}],
            "cal": [{"input": "c"}],
        }
        rates = compute_overlap_rate(splits)
        assert len(rates) == 3  # cal-test, cal-train, test-train
        assert all(v == 0.0 for v in rates.values())


# ---------------------------------------------------------------------------
# suggest_decontamination
# ---------------------------------------------------------------------------


class TestSuggestDecontamination:
    def test_clean_result(self):
        r = ContaminationCheckResult(clean=True)
        suggestions = suggest_decontamination(r)
        assert len(suggestions) == 1
        assert "clean" in suggestions[0].lower() or "no decontamination" in suggestions[0].lower()

    def test_contaminated_result(self):
        r = ContaminationCheckResult(
            leaks=[LeakageItem(item_id="x", found_in=["train", "test"])],
            leaked_count=1,
            clean=False,
        )
        suggestions = suggest_decontamination(r)
        assert len(suggestions) >= 2
        assert any("remove" in s.lower() or "leaked" in s.lower() for s in suggestions)

    def test_test_train_leak_priority(self):
        r = ContaminationCheckResult(
            leaks=[LeakageItem(item_id="x", found_in=["train", "test"])],
            leaked_count=1,
            clean=False,
        )
        suggestions = suggest_decontamination(r)
        assert any("test split" in s.lower() for s in suggestions)

    def test_calibration_leak(self):
        r = ContaminationCheckResult(
            leaks=[LeakageItem(item_id="x", found_in=["train", "calibration"])],
            leaked_count=1,
            clean=False,
        )
        suggestions = suggest_decontamination(r)
        assert any("calibration" in s.lower() for s in suggestions)
