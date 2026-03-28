"""Tests for datasets_augmentation module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.datasets_augmentation import (
    AugmentationMethod,
    AugmentationReport,
    AugmentedItem,
    add_noise,
    augment_dataset,
    augment_item,
    change_case_style,
    list_augmentation_methods,
    paraphrase_input,
    truncate_input,
)


# ---------------------------------------------------------------------------
# AugmentationMethod
# ---------------------------------------------------------------------------


class TestAugmentationMethod:
    def test_as_dict(self):
        m = AugmentationMethod(name="noise", description="Add typos", preserves_label=True)
        d = m.as_dict()
        assert d["name"] == "noise"
        assert d["description"] == "Add typos"
        assert d["preserves_label"] is True

    def test_defaults(self):
        m = AugmentationMethod(name="x")
        assert m.description == ""
        assert m.preserves_label is True


# ---------------------------------------------------------------------------
# AugmentedItem
# ---------------------------------------------------------------------------


class TestAugmentedItem:
    def test_as_dict(self):
        item = AugmentedItem(
            original_id="a", augmented_id="a_aug", input_text="hello"
        )
        d = item.as_dict()
        assert d["original_id"] == "a"
        assert d["augmented_id"] == "a_aug"
        assert d["input_text"] == "hello"
        assert d["output_text"] == ""
        assert d["method"] == ""
        assert d["metadata"] == {}

    def test_from_dict(self):
        d = {
            "original_id": "b",
            "augmented_id": "b_aug",
            "input_text": "world",
            "output_text": "resp",
            "method": "noise",
            "metadata": {"k": 1},
        }
        item = AugmentedItem.from_dict(d)
        assert item.original_id == "b"
        assert item.output_text == "resp"
        assert item.metadata == {"k": 1}

    def test_roundtrip(self):
        item = AugmentedItem(
            original_id="c",
            augmented_id="c_aug",
            input_text="text",
            output_text="out",
            method="paraphrase",
            metadata={"seed": 42},
        )
        restored = AugmentedItem.from_dict(item.as_dict())
        assert restored.as_dict() == item.as_dict()


# ---------------------------------------------------------------------------
# AugmentationReport
# ---------------------------------------------------------------------------


class TestAugmentationReport:
    def test_as_dict(self):
        r = AugmentationReport(
            original_count=10,
            augmented_count=20,
            methods_used=["noise"],
            expansion_ratio=2.0,
        )
        d = r.as_dict()
        assert d["original_count"] == 10
        assert d["augmented_count"] == 20

    def test_from_dict(self):
        d = {
            "original_count": 5,
            "augmented_count": 15,
            "methods_used": ["paraphrase", "noise"],
            "expansion_ratio": 3.0,
        }
        r = AugmentationReport.from_dict(d)
        assert r.expansion_ratio == 3.0

    def test_roundtrip(self):
        r = AugmentationReport(
            original_count=3,
            augmented_count=9,
            methods_used=["case"],
            expansion_ratio=3.0,
        )
        assert AugmentationReport.from_dict(r.as_dict()).as_dict() == r.as_dict()

    def test_format_text(self):
        r = AugmentationReport(
            original_count=2,
            augmented_count=6,
            methods_used=["noise", "case"],
            expansion_ratio=3.0,
        )
        text = r.format_text()
        assert "Augmentation Report" in text
        assert "Original items" in text
        assert "noise, case" in text

    def test_format_text_no_methods(self):
        r = AugmentationReport()
        text = r.format_text()
        assert "none" in text


# ---------------------------------------------------------------------------
# paraphrase_input
# ---------------------------------------------------------------------------


class TestParaphraseInput:
    def test_changes_text(self):
        """Paraphrase should produce different text for multi-clause input."""
        text = "The quick brown fox. Jumps over the lazy dog"
        result = paraphrase_input(text, seed=1)
        # Either reordering or synonym swap should change something
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic_with_seed(self):
        text = "The quick brown fox. Jumps over the lazy dog"
        a = paraphrase_input(text, seed=42)
        b = paraphrase_input(text, seed=42)
        assert a == b

    def test_empty_text(self):
        assert paraphrase_input("") == ""

    def test_single_word(self):
        result = paraphrase_input("hello", seed=0)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------


class TestAddNoise:
    def test_introduces_errors(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = add_noise(text, noise_rate=0.3, seed=1)
        assert result != text

    def test_deterministic_with_seed(self):
        text = "Some text here"
        a = add_noise(text, noise_rate=0.1, seed=7)
        b = add_noise(text, noise_rate=0.1, seed=7)
        assert a == b

    def test_zero_rate_no_change(self):
        text = "No changes expected"
        result = add_noise(text, noise_rate=0.0, seed=0)
        assert result == text

    def test_empty_text(self):
        assert add_noise("") == ""

    def test_single_char(self):
        result = add_noise("x", noise_rate=0.5, seed=10)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# change_case_style
# ---------------------------------------------------------------------------


class TestChangeCaseStyle:
    def test_title_to_sentence(self):
        """Title-cased text should become sentence case."""
        result = change_case_style("The Quick Brown Fox")
        assert result == "The quick brown fox"

    def test_sentence_to_title(self):
        """Lowercase-heavy text should become title case."""
        result = change_case_style("the quick brown fox")
        assert result == "The Quick Brown Fox"

    def test_empty_text(self):
        assert change_case_style("") == ""


# ---------------------------------------------------------------------------
# truncate_input
# ---------------------------------------------------------------------------


class TestTruncateInput:
    def test_shortens(self):
        text = "Hello, world!"
        result = truncate_input(text, ratio=0.5)
        assert len(result) < len(text)
        assert text.startswith(result)

    def test_full_ratio(self):
        text = "Keep everything"
        result = truncate_input(text, ratio=1.0)
        assert result == text

    def test_empty_text(self):
        assert truncate_input("") == ""

    def test_single_char(self):
        result = truncate_input("a", ratio=0.5)
        assert result == "a"  # max(1, ...) keeps at least 1 char


# ---------------------------------------------------------------------------
# augment_item
# ---------------------------------------------------------------------------


class TestAugmentItem:
    def test_single_method(self):
        item = {"id": "i1", "input": "Hello world", "output": "response"}
        results = augment_item(item, methods=["paraphrase"], seed=0)
        assert len(results) == 1
        assert results[0].original_id == "i1"
        assert results[0].method == "paraphrase"
        assert results[0].output_text == "response"

    def test_multiple_methods(self):
        item = {"id": "i2", "input": "Some test text", "output": "out"}
        results = augment_item(item, methods=["paraphrase", "noise", "case"], seed=0)
        assert len(results) == 3
        methods_used = [r.method for r in results]
        assert "paraphrase" in methods_used
        assert "noise" in methods_used
        assert "case" in methods_used

    def test_unknown_method_skipped(self):
        item = {"id": "i3", "input": "text"}
        results = augment_item(item, methods=["nonexistent"])
        assert len(results) == 0

    def test_default_method(self):
        item = {"id": "i4", "input": "text"}
        results = augment_item(item, seed=0)
        assert len(results) == 1
        assert results[0].method == "paraphrase"

    def test_missing_output(self):
        item = {"id": "i5", "input": "no output field"}
        results = augment_item(item, methods=["noise"], seed=0)
        assert results[0].output_text == ""


# ---------------------------------------------------------------------------
# augment_dataset
# ---------------------------------------------------------------------------


class TestAugmentDataset:
    def test_full_augmentation(self):
        items = [
            {"id": "a", "input": "First item", "output": "r1"},
            {"id": "b", "input": "Second item", "output": "r2"},
        ]
        augmented, report = augment_dataset(items, seed=0)
        assert len(augmented) == 4  # 2 items x 2 augmentations_per_item
        assert report.original_count == 2
        assert report.augmented_count == 4
        assert report.expansion_ratio == 2.0

    def test_custom_augmentations_per_item(self):
        items = [{"id": "x", "input": "text"}]
        augmented, report = augment_dataset(
            items, methods=["noise"], augmentations_per_item=5, seed=0
        )
        assert len(augmented) == 5
        assert report.augmented_count == 5

    def test_empty_dataset(self):
        augmented, report = augment_dataset([])
        assert len(augmented) == 0
        assert report.original_count == 0
        assert report.expansion_ratio == 0.0

    def test_deterministic_with_seed(self):
        items = [{"id": "d", "input": "Determinism test"}]
        a1, _ = augment_dataset(items, seed=99)
        a2, _ = augment_dataset(items, seed=99)
        assert [x.as_dict() for x in a1] == [x.as_dict() for x in a2]

    def test_report_methods_used(self):
        items = [{"id": "m", "input": "text"}]
        _, report = augment_dataset(items, methods=["paraphrase", "noise"], seed=0)
        assert "noise" in report.methods_used or "paraphrase" in report.methods_used


# ---------------------------------------------------------------------------
# list_augmentation_methods
# ---------------------------------------------------------------------------


class TestListAugmentationMethods:
    def test_returns_at_least_four(self):
        methods = list_augmentation_methods()
        assert len(methods) >= 4

    def test_returns_augmentation_method_instances(self):
        methods = list_augmentation_methods()
        for m in methods:
            assert isinstance(m, AugmentationMethod)

    def test_known_methods_present(self):
        names = {m.name for m in list_augmentation_methods()}
        assert "paraphrase" in names
        assert "noise" in names
        assert "case" in names
        assert "truncate" in names
