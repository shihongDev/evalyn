"""Tests for calibration data augmentation."""

from __future__ import annotations

import pytest

from evalyn_sdk.calibration.data_augmentation import (
    AugmentationConfig,
    AugmentationResult,
    AugmentedExample,
    augment_batch,
    augment_example,
    augment_shuffle,
    augment_synonym,
    augment_truncate,
)


# ---------------------------------------------------------------------------
# AugmentedExample
# ---------------------------------------------------------------------------


class TestAugmentedExample:
    def test_as_dict(self):
        ex = AugmentedExample("id1", "text", "synonym", "positive")
        d = ex.as_dict()
        assert d["original_id"] == "id1"
        assert d["augmented_text"] == "text"
        assert d["augmentation_method"] == "synonym"
        assert d["label"] == "positive"

    def test_default_label(self):
        ex = AugmentedExample("id1", "text", "synonym")
        assert ex.label == ""


# ---------------------------------------------------------------------------
# AugmentationConfig
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    def test_defaults(self):
        cfg = AugmentationConfig()
        assert cfg.methods == ["synonym", "shuffle", "truncate"]
        assert cfg.augmentations_per_example == 2
        assert cfg.seed is None

    def test_as_dict(self):
        cfg = AugmentationConfig(methods=["synonym"], augmentations_per_example=3, seed=42)
        d = cfg.as_dict()
        assert d["methods"] == ["synonym"]
        assert d["augmentations_per_example"] == 3
        assert d["seed"] == 42

    def test_from_dict(self):
        cfg = AugmentationConfig.from_dict({"methods": ["truncate"], "seed": 7})
        assert cfg.methods == ["truncate"]
        assert cfg.seed == 7

    def test_from_dict_defaults(self):
        cfg = AugmentationConfig.from_dict({})
        assert cfg.methods == ["synonym", "shuffle", "truncate"]
        assert cfg.augmentations_per_example == 2
        assert cfg.seed is None

    def test_roundtrip(self):
        cfg = AugmentationConfig(methods=["shuffle", "synonym"], augmentations_per_example=5, seed=99)
        restored = AugmentationConfig.from_dict(cfg.as_dict())
        assert restored.methods == cfg.methods
        assert restored.augmentations_per_example == cfg.augmentations_per_example
        assert restored.seed == cfg.seed


# ---------------------------------------------------------------------------
# AugmentationResult
# ---------------------------------------------------------------------------


class TestAugmentationResult:
    def test_as_dict(self):
        r = AugmentationResult(10, 20, ["synonym", "shuffle"])
        d = r.as_dict()
        assert d["original_count"] == 10
        assert d["augmented_count"] == 20
        assert d["methods_used"] == ["synonym", "shuffle"]

    def test_format_text(self):
        r = AugmentationResult(5, 15, ["synonym", "truncate"])
        text = r.format_text()
        assert "Originals:  5" in text
        assert "Augmented:  15" in text
        assert "synonym" in text
        assert "truncate" in text
        assert "Augmentation Result" in text


# ---------------------------------------------------------------------------
# augment_synonym
# ---------------------------------------------------------------------------


class TestAugmentSynonym:
    def test_replaces_known_words(self):
        result = augment_synonym("This is a good idea")
        assert "excellent" in result

    def test_no_matching_words_unchanged(self):
        original = "The cat sat on the mat"
        result = augment_synonym(original)
        assert result == original

    def test_preserves_capitalization(self):
        result = augment_synonym("Good morning")
        assert result.startswith("Excellent")

    def test_preserves_punctuation(self):
        result = augment_synonym("This is good.")
        assert "excellent." in result

    def test_empty_text(self):
        assert augment_synonym("") == ""

    def test_single_known_word(self):
        result = augment_synonym("good")
        assert result == "excellent"

    def test_multiple_replacements(self):
        result = augment_synonym("good and bad")
        assert "excellent" in result
        assert "poor" in result


# ---------------------------------------------------------------------------
# augment_shuffle
# ---------------------------------------------------------------------------


class TestAugmentShuffle:
    def test_reorders_sentences(self):
        text = "First sentence. Second sentence. Third sentence"
        # With seed, shuffle should produce a different order sometimes
        seen_orders: set[str] = set()
        for seed in range(20):
            result = augment_shuffle(text, seed=seed)
            seen_orders.add(result)
        assert len(seen_orders) > 1, "Shuffle should produce multiple orderings"

    def test_deterministic_with_seed(self):
        text = "A. B. C. D"
        r1 = augment_shuffle(text, seed=42)
        r2 = augment_shuffle(text, seed=42)
        assert r1 == r2

    def test_single_sentence_unchanged(self):
        text = "Only one sentence here"
        assert augment_shuffle(text, seed=0) == text

    def test_empty_text(self):
        assert augment_shuffle("") == ""

    def test_no_period_space_delimiter(self):
        text = "No periods at all"
        assert augment_shuffle(text, seed=0) == text


# ---------------------------------------------------------------------------
# augment_truncate
# ---------------------------------------------------------------------------


class TestAugmentTruncate:
    def test_shortens_text(self):
        text = "Hello world, this is a test"
        result = augment_truncate(text, ratio=0.5)
        assert len(result) < len(text)
        assert len(result) == int(len(text) * 0.5)

    def test_default_ratio(self):
        text = "A" * 100
        result = augment_truncate(text)
        assert len(result) == 80

    def test_empty_text(self):
        assert augment_truncate("") == ""

    def test_single_char(self):
        assert augment_truncate("X", ratio=0.5) == "X"

    def test_ratio_one_keeps_all(self):
        text = "Keep everything"
        assert augment_truncate(text, ratio=1.0) == text


# ---------------------------------------------------------------------------
# augment_example
# ---------------------------------------------------------------------------


class TestAugmentExample:
    def test_routes_synonym(self):
        result = augment_example("good idea", "synonym")
        assert "excellent" in result

    def test_routes_shuffle(self):
        text = "A sentence. B sentence. C sentence"
        result = augment_example(text, "shuffle", seed=1)
        # Should be a valid reordering
        assert "sentence" in result

    def test_routes_truncate(self):
        text = "A" * 100
        result = augment_example(text, "truncate")
        assert len(result) == 80

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation method"):
            augment_example("text", "nonexistent")


# ---------------------------------------------------------------------------
# augment_batch
# ---------------------------------------------------------------------------


class TestAugmentBatch:
    def test_correct_count(self):
        examples = [("id1", "good text", "pos"), ("id2", "bad text", "neg")]
        cfg = AugmentationConfig(methods=["synonym", "truncate"], augmentations_per_example=2, seed=42)
        augmented, result = augment_batch(examples, cfg)
        assert len(augmented) == 4  # 2 examples x 2 augmentations
        assert result.original_count == 2
        assert result.augmented_count == 4

    def test_methods_cycle(self):
        examples = [("id1", "good text here", "pos")]
        cfg = AugmentationConfig(methods=["synonym", "truncate"], augmentations_per_example=3, seed=0)
        augmented, result = augment_batch(examples, cfg)
        methods = [a.augmentation_method for a in augmented]
        assert methods == ["synonym", "truncate", "synonym"]

    def test_labels_preserved(self):
        examples = [("id1", "text", "my_label")]
        cfg = AugmentationConfig(methods=["synonym"], augmentations_per_example=1, seed=0)
        augmented, _ = augment_batch(examples, cfg)
        assert augmented[0].label == "my_label"

    def test_original_id_preserved(self):
        examples = [("abc", "text", "")]
        cfg = AugmentationConfig(methods=["synonym"], augmentations_per_example=1, seed=0)
        augmented, _ = augment_batch(examples, cfg)
        assert augmented[0].original_id == "abc"

    def test_empty_examples(self):
        augmented, result = augment_batch([], AugmentationConfig(seed=0))
        assert augmented == []
        assert result.original_count == 0
        assert result.augmented_count == 0

    def test_result_methods_used(self):
        examples = [("id1", "good text", "")]
        cfg = AugmentationConfig(methods=["synonym", "shuffle"], augmentations_per_example=2, seed=0)
        _, result = augment_batch(examples, cfg)
        assert "synonym" in result.methods_used
        assert "shuffle" in result.methods_used

    def test_deterministic_with_seed(self):
        examples = [("id1", "good text here. Another sentence", "pos")]
        cfg = AugmentationConfig(methods=["synonym", "shuffle"], augmentations_per_example=2, seed=42)
        aug1, _ = augment_batch(examples, cfg)
        aug2, _ = augment_batch(examples, cfg)
        for a, b in zip(aug1, aug2):
            assert a.augmented_text == b.augmented_text
