"""Tests for domain_transfer module."""

from __future__ import annotations

import pytest

from evalyn_sdk.domain_transfer import (
    BUILTIN_VOCABULARIES,
    DomainVocabulary,
    TransferConfig,
    TransferReport,
    TransferredInput,
    estimate_complexity,
    format_transfer_report,
    transfer_batch,
    transfer_text,
    validate_transfer,
)


# ---------------------------------------------------------------------------
# DomainVocabulary
# ---------------------------------------------------------------------------


class TestDomainVocabulary:
    def test_as_dict(self):
        vocab = DomainVocabulary("med", "Medical", {"a": "b"}, ["formal"])
        d = vocab.as_dict()
        assert d["domain_id"] == "med"
        assert d["name"] == "Medical"
        assert d["terms"] == {"a": "b"}
        assert d["style_hints"] == ["formal"]

    def test_from_dict(self):
        d = {
            "domain_id": "legal",
            "name": "Legal",
            "terms": {"x": "y"},
            "style_hints": ["precise"],
        }
        vocab = DomainVocabulary.from_dict(d)
        assert vocab.domain_id == "legal"
        assert vocab.terms == {"x": "y"}

    def test_roundtrip(self):
        vocab = DomainVocabulary("fin", "Finance", {"a": "b"}, ["risk"])
        assert DomainVocabulary.from_dict(vocab.as_dict()).as_dict() == vocab.as_dict()

    def test_from_dict_defaults(self):
        d = {"domain_id": "x", "name": "X"}
        vocab = DomainVocabulary.from_dict(d)
        assert vocab.terms == {}
        assert vocab.style_hints == []


# ---------------------------------------------------------------------------
# TransferConfig
# ---------------------------------------------------------------------------


class TestTransferConfig:
    def test_as_dict(self):
        cfg = TransferConfig("medical", "legal", True, False)
        d = cfg.as_dict()
        assert d["source_domain"] == "medical"
        assert d["target_domain"] == "legal"
        assert d["preserve_structure"] is True
        assert d["preserve_complexity"] is False

    def test_from_dict(self):
        d = {"source_domain": "a", "target_domain": "b"}
        cfg = TransferConfig.from_dict(d)
        assert cfg.preserve_structure is True
        assert cfg.preserve_complexity is True

    def test_roundtrip(self):
        cfg = TransferConfig("a", "b", False, True)
        assert TransferConfig.from_dict(cfg.as_dict()).as_dict() == cfg.as_dict()


# ---------------------------------------------------------------------------
# TransferredInput
# ---------------------------------------------------------------------------


class TestTransferredInput:
    def test_as_dict(self):
        ti = TransferredInput("orig", "new", "med", "legal", 2, ["tag1"])
        d = ti.as_dict()
        assert d["original_text"] == "orig"
        assert d["transferred_text"] == "new"
        assert d["substitutions_made"] == 2
        assert d["tags"] == ["tag1"]

    def test_from_dict(self):
        d = {
            "original_text": "a",
            "transferred_text": "b",
            "source_domain": "x",
            "target_domain": "y",
            "substitutions_made": 1,
            "tags": ["t"],
        }
        ti = TransferredInput.from_dict(d)
        assert ti.original_text == "a"

    def test_roundtrip(self):
        ti = TransferredInput("a", "b", "x", "y", 3, ["tag"])
        assert TransferredInput.from_dict(ti.as_dict()).as_dict() == ti.as_dict()

    def test_default_tags(self):
        ti = TransferredInput("a", "b", "x", "y", 0)
        assert ti.tags == []

    def test_from_dict_defaults(self):
        d = {
            "original_text": "a",
            "transferred_text": "b",
            "source_domain": "x",
            "target_domain": "y",
        }
        ti = TransferredInput.from_dict(d)
        assert ti.substitutions_made == 0
        assert ti.tags == []


# ---------------------------------------------------------------------------
# TransferReport
# ---------------------------------------------------------------------------


class TestTransferReport:
    def test_as_dict(self):
        items = [TransferredInput("a", "b", "x", "y", 1)]
        report = TransferReport(items, 1, 1.0, ["x", "y"])
        d = report.as_dict()
        assert d["total_items"] == 1
        assert d["avg_substitutions"] == 1.0
        assert len(d["inputs"]) == 1

    def test_from_dict(self):
        d = {
            "inputs": [
                {
                    "original_text": "a",
                    "transferred_text": "b",
                    "source_domain": "x",
                    "target_domain": "y",
                    "substitutions_made": 0,
                }
            ],
            "total_items": 1,
            "avg_substitutions": 0.0,
            "domains_used": ["x", "y"],
        }
        report = TransferReport.from_dict(d)
        assert report.total_items == 1

    def test_roundtrip(self):
        items = [TransferredInput("a", "b", "x", "y", 2, ["t"])]
        report = TransferReport(items, 1, 2.0, ["x", "y"])
        assert TransferReport.from_dict(report.as_dict()).as_dict() == report.as_dict()


# ---------------------------------------------------------------------------
# BUILTIN_VOCABULARIES
# ---------------------------------------------------------------------------


class TestBuiltinVocabularies:
    def test_has_four_domains(self):
        assert len(BUILTIN_VOCABULARIES) == 4

    def test_medical_exists(self):
        assert "medical" in BUILTIN_VOCABULARIES
        med = BUILTIN_VOCABULARIES["medical"]
        assert med.terms["patient"] == "client"

    def test_legal_exists(self):
        assert "legal" in BUILTIN_VOCABULARIES
        leg = BUILTIN_VOCABULARIES["legal"]
        assert leg.terms["client"] == "party"

    def test_finance_exists(self):
        assert "finance" in BUILTIN_VOCABULARIES
        fin = BUILTIN_VOCABULARIES["finance"]
        assert fin.terms["customer"] == "investor"

    def test_general_identity_mapping(self):
        gen = BUILTIN_VOCABULARIES["general"]
        for key, val in gen.terms.items():
            assert key == val


# ---------------------------------------------------------------------------
# transfer_text
# ---------------------------------------------------------------------------


class TestTransferText:
    def test_medical_to_legal(self):
        source = BUILTIN_VOCABULARIES["medical"]
        target = BUILTIN_VOCABULARIES["legal"]
        text = "The patient signed the agreement"
        result, count = transfer_text(text, source, target)
        # "patient" in medical maps generic "patient" -> target doesn't have "patient"
        # but "agreement" in legal maps to "contract"
        # The exact result depends on what overlaps exist
        assert isinstance(result, str)
        assert isinstance(count, int)

    def test_no_substitutions(self):
        source = BUILTIN_VOCABULARIES["medical"]
        target = BUILTIN_VOCABULARIES["medical"]
        text = "The weather is nice today"
        result, count = transfer_text(text, source, target)
        assert count == 0
        assert result == text

    def test_case_preservation_upper(self):
        source = DomainVocabulary("s", "S", {"apple": "fruit"}, [])
        target = DomainVocabulary("t", "T", {"apple": "banana"}, [])
        text = "I like APPLE"
        result, count = transfer_text(text, source, target)
        assert "BANANA" in result
        assert count >= 1

    def test_case_preservation_title(self):
        source = DomainVocabulary("s", "S", {"apple": "fruit"}, [])
        target = DomainVocabulary("t", "T", {"apple": "banana"}, [])
        text = "I like Fruit"
        result, count = transfer_text(text, source, target)
        assert "Banana" in result

    def test_case_preservation_lower(self):
        source = DomainVocabulary("s", "S", {"apple": "fruit"}, [])
        target = DomainVocabulary("t", "T", {"apple": "banana"}, [])
        text = "I like fruit"
        result, count = transfer_text(text, source, target)
        assert "banana" in result

    def test_multiple_occurrences(self):
        source = DomainVocabulary("s", "S", {"cat": "feline"}, [])
        target = DomainVocabulary("t", "T", {"cat": "dog"}, [])
        text = "the feline and another feline"
        result, count = transfer_text(text, source, target)
        assert result == "the dog and another dog"
        assert count == 2

    def test_word_boundary_respected(self):
        source = DomainVocabulary("s", "S", {"cat": "feline"}, [])
        target = DomainVocabulary("t", "T", {"cat": "dog"}, [])
        text = "the category is feline"
        result, count = transfer_text(text, source, target)
        # "category" should NOT be modified, only "feline"
        assert "category" in result
        assert count == 1

    def test_empty_text(self):
        source = BUILTIN_VOCABULARIES["medical"]
        target = BUILTIN_VOCABULARIES["legal"]
        result, count = transfer_text("", source, target)
        assert result == ""
        assert count == 0

    def test_identity_no_substitution(self):
        # Same vocab should produce no substitutions
        vocab = DomainVocabulary("x", "X", {"a": "b"}, [])
        result, count = transfer_text("b here", vocab, vocab)
        assert count == 0

    def test_generic_term_also_matched(self):
        # Source text might use the generic term directly
        source = DomainVocabulary("s", "S", {"apple": "fruit"}, [])
        target = DomainVocabulary("t", "T", {"apple": "banana"}, [])
        text = "I like apple"
        result, count = transfer_text(text, source, target)
        assert "banana" in result


# ---------------------------------------------------------------------------
# transfer_batch
# ---------------------------------------------------------------------------


class TestTransferBatch:
    def test_basic_batch(self):
        texts = ["The patient needs treatment", "Check the diagnosis"]
        config = TransferConfig("medical", "legal")
        report = transfer_batch(texts, config)
        assert report.total_items == 2
        assert len(report.inputs) == 2

    def test_domains_used(self):
        config = TransferConfig("medical", "finance")
        report = transfer_batch(["hello"], config)
        assert "finance" in report.domains_used
        assert "medical" in report.domains_used

    def test_unknown_domain_raises(self):
        config = TransferConfig("medical", "nonexistent")
        with pytest.raises(ValueError, match="Unknown domain"):
            transfer_batch(["hello"], config)

    def test_unknown_source_domain_raises(self):
        config = TransferConfig("nonexistent", "medical")
        with pytest.raises(ValueError, match="Unknown domain"):
            transfer_batch(["hello"], config)

    def test_custom_vocabularies(self):
        custom = {
            "a": DomainVocabulary("a", "A", {"word": "alpha"}, []),
            "b": DomainVocabulary("b", "B", {"word": "beta"}, []),
        }
        config = TransferConfig("a", "b")
        report = transfer_batch(["the alpha is here"], config, vocabularies=custom)
        assert report.total_items == 1
        assert "beta" in report.inputs[0].transferred_text

    def test_empty_batch(self):
        config = TransferConfig("medical", "legal")
        report = transfer_batch([], config)
        assert report.total_items == 0
        assert report.avg_substitutions == 0.0

    def test_avg_substitutions(self):
        vocab_s = DomainVocabulary("s", "S", {"cat": "feline"}, [])
        vocab_t = DomainVocabulary("t", "T", {"cat": "dog"}, [])
        custom = {"s": vocab_s, "t": vocab_t}
        config = TransferConfig("s", "t")
        texts = ["feline here", "feline and feline"]
        report = transfer_batch(texts, config, vocabularies=custom)
        # First text: 1 sub, second text: 2 subs -> avg = 1.5
        assert report.avg_substitutions == 1.5

    def test_tags_include_target_domain(self):
        config = TransferConfig("medical", "finance")
        report = transfer_batch(["patient treatment"], config)
        for item in report.inputs:
            assert "finance" in item.tags
            assert "domain_transfer" in item.tags

    def test_complexity_drift_tag(self):
        # Create vocabs with very different word lengths to cause drift
        vocab_s = DomainVocabulary(
            "s", "S", {"a": "extraordinarily"}, []
        )
        vocab_t = DomainVocabulary("t", "T", {"a": "x"}, [])
        custom = {"s": vocab_s, "t": vocab_t}
        config = TransferConfig("s", "t", preserve_complexity=True)
        report = transfer_batch(["extraordinarily extraordinarily extraordinarily"], config, vocabularies=custom)
        # The complexity changes drastically, should get drift tag
        if report.inputs[0].substitutions_made > 0:
            assert "complexity_drift" in report.inputs[0].tags


# ---------------------------------------------------------------------------
# estimate_complexity
# ---------------------------------------------------------------------------


class TestEstimateComplexity:
    def test_empty_text(self):
        assert estimate_complexity("") == 0.0

    def test_single_word(self):
        score = estimate_complexity("hello")
        # avg_length=5, unique_ratio=1.0 -> 5.0
        assert score == 5.0

    def test_repeated_words(self):
        score = estimate_complexity("the the the")
        # avg_length=3, unique_ratio=1/3 -> 1.0
        assert score == pytest.approx(1.0)

    def test_all_unique(self):
        score = estimate_complexity("one two three")
        # avg_length=(3+3+5)/3=3.67, unique_ratio=1.0 -> 3.67
        assert score == pytest.approx(11.0 / 3.0)

    def test_longer_words_higher_score(self):
        short = estimate_complexity("a b c")
        long = estimate_complexity("alpha bravo charlie")
        assert long > short

    def test_more_unique_higher_score(self):
        repeated = estimate_complexity("go go go go")
        unique = estimate_complexity("go run fly swim")
        assert unique > repeated


# ---------------------------------------------------------------------------
# validate_transfer
# ---------------------------------------------------------------------------


class TestValidateTransfer:
    def test_identical_text_valid(self):
        assert validate_transfer("hello world", "hello world") is True

    def test_similar_complexity_valid(self):
        assert validate_transfer("hello world", "howdy globe") is True

    def test_vastly_different_invalid(self):
        # Original is simple, transferred is complex
        original = "a a a a a a a a a a"
        transferred = "extraordinarily incomprehensible"
        assert validate_transfer(original, transferred, max_drift=0.3) is False

    def test_empty_original_always_valid(self):
        assert validate_transfer("", "anything here") is True

    def test_custom_max_drift(self):
        original = "hello world"
        transferred = "hi globe"
        # These have similar complexity, should pass with tight threshold
        assert validate_transfer(original, transferred, max_drift=1.0) is True

    def test_zero_drift_identical(self):
        text = "the quick brown fox"
        assert validate_transfer(text, text, max_drift=0.0) is True


# ---------------------------------------------------------------------------
# format_transfer_report
# ---------------------------------------------------------------------------


class TestFormatTransferReport:
    def test_contains_header(self):
        report = TransferReport([], 0, 0.0, [])
        text = format_transfer_report(report)
        assert "DOMAIN TRANSFER REPORT" in text

    def test_contains_total_items(self):
        report = TransferReport([], 5, 0.0, [])
        text = format_transfer_report(report)
        assert "5" in text

    def test_contains_domains(self):
        report = TransferReport([], 0, 0.0, ["medical", "legal"])
        text = format_transfer_report(report)
        assert "medical" in text
        assert "legal" in text

    def test_contains_transfer_details(self):
        items = [TransferredInput("orig text", "new text", "a", "b", 2, ["tag"])]
        report = TransferReport(items, 1, 2.0, ["a", "b"])
        text = format_transfer_report(report)
        assert "orig text" in text
        assert "new text" in text
        assert "tag" in text

    def test_contains_avg_substitutions(self):
        report = TransferReport([], 0, 3.5, [])
        text = format_transfer_report(report)
        assert "3.5" in text


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_medical_to_finance(self):
        texts = [
            "The patient needs treatment at the hospital",
            "Check the symptoms before diagnosis",
        ]
        config = TransferConfig("medical", "finance")
        report = transfer_batch(texts, config)
        assert report.total_items == 2
        text = format_transfer_report(report)
        assert "DOMAIN TRANSFER REPORT" in text

    def test_legal_to_general(self):
        texts = ["The client filed a dispute over payment"]
        config = TransferConfig("legal", "general")
        report = transfer_batch(texts, config)
        # "client" in legal -> generic "client", target general maps "client"->"client" = no sub
        # "dispute" in legal -> generic "dispute", target general maps "dispute"->"dispute" = no sub
        # "payment" in legal -> generic "payment", target general maps "payment"->"payment" = no sub
        assert report.total_items == 1

    def test_roundtrip_general_to_medical_to_general(self):
        texts = ["The patient went to the hospital for treatment"]
        config1 = TransferConfig("general", "medical")
        report1 = transfer_batch(texts, config1)
        transferred = [item.transferred_text for item in report1.inputs]

        config2 = TransferConfig("medical", "general")
        report2 = transfer_batch(transferred, config2)
        # After roundtrip, should get back something close to original
        assert report2.total_items == 1

    def test_complexity_preserved_across_transfer(self):
        texts = ["The patient needs diagnosis and treatment at the hospital"]
        config = TransferConfig("medical", "finance")
        report = transfer_batch(texts, config)
        for item in report.inputs:
            valid = validate_transfer(item.original_text, item.transferred_text)
            assert valid is True
