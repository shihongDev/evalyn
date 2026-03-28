"""Tests for the change attribution module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.change_attribution import (
    ChangeFactor,
    ChangeAttribution,
    AttributionReport,
    detect_dataset_change,
    detect_model_change,
    detect_prompt_change,
    detect_config_change,
    attribute_change,
    build_attribution_report,
)


# ---------------------------------------------------------------------------
# detect_dataset_change
# ---------------------------------------------------------------------------


class TestDetectDatasetChange:
    def test_count_changed(self):
        """Different item counts should attribute to dataset."""
        factor = detect_dataset_change(100, 200)
        assert factor.factor == "dataset"
        assert factor.confidence > 0.0
        assert "100" in factor.evidence and "200" in factor.evidence

    def test_hash_changed(self):
        """Different hashes should attribute to dataset with high confidence."""
        factor = detect_dataset_change(100, 100, hash_a="abc123", hash_b="def456")
        assert factor.factor == "dataset"
        assert factor.confidence >= 0.9

    def test_no_change(self):
        """Same count and no hashes should report zero confidence."""
        factor = detect_dataset_change(100, 100)
        assert factor.confidence == 0.0

    def test_same_hash(self):
        """Same hash should report zero confidence."""
        factor = detect_dataset_change(100, 100, hash_a="abc", hash_b="abc")
        assert factor.confidence == 0.0

    def test_hash_takes_priority(self):
        """Hash difference should be higher confidence than count difference."""
        hash_factor = detect_dataset_change(100, 100, hash_a="a", hash_b="b")
        count_factor = detect_dataset_change(100, 200)
        assert hash_factor.confidence > count_factor.confidence


# ---------------------------------------------------------------------------
# detect_model_change
# ---------------------------------------------------------------------------


class TestDetectModelChange:
    def test_different_models(self):
        """Different model names should attribute to model."""
        factor = detect_model_change("gpt-4", "gpt-4o")
        assert factor.factor == "model"
        assert factor.confidence >= 0.9
        assert "gpt-4" in factor.evidence

    def test_same_model(self):
        """Same model should report zero confidence."""
        factor = detect_model_change("gpt-4", "gpt-4")
        assert factor.confidence == 0.0

    def test_empty_models(self):
        """Two empty strings should report zero confidence."""
        factor = detect_model_change("", "")
        assert factor.confidence == 0.0

    def test_one_empty_model(self):
        """One empty model name should flag change."""
        factor = detect_model_change("gpt-4", "")
        assert factor.confidence > 0.0


# ---------------------------------------------------------------------------
# detect_prompt_change
# ---------------------------------------------------------------------------


class TestDetectPromptChange:
    def test_different_prompts(self):
        """Different prompts should attribute to prompt."""
        factor = detect_prompt_change("Be helpful.", "Be concise.")
        assert factor.factor == "prompt"
        assert factor.confidence > 0.0

    def test_same_prompt(self):
        """Same prompt should report zero confidence."""
        factor = detect_prompt_change("Be helpful.", "Be helpful.")
        assert factor.confidence == 0.0

    def test_small_edit_lower_confidence(self):
        """A small edit should yield lower confidence than a large edit."""
        small = detect_prompt_change("Be helpful and kind.", "Be helpful and nice.")
        large = detect_prompt_change("Be helpful.", "Respond concisely with bullet points.")
        assert small.confidence < large.confidence

    def test_empty_to_nonempty(self):
        """Empty to non-empty should flag change."""
        factor = detect_prompt_change("", "Be helpful.")
        assert factor.confidence > 0.0

    def test_completely_different(self):
        """Completely different prompts should yield high confidence."""
        factor = detect_prompt_change("abc", "xyz")
        assert factor.confidence >= 0.9


# ---------------------------------------------------------------------------
# detect_config_change
# ---------------------------------------------------------------------------


class TestDetectConfigChange:
    def test_different_values(self):
        """Different config values should be detected."""
        factor = detect_config_change(
            {"temperature": 0.7, "top_k": 40},
            {"temperature": 0.9, "top_k": 40},
        )
        assert factor.factor == "config"
        assert factor.confidence > 0.0
        assert "temperature" in factor.evidence

    def test_same_config(self):
        """Identical configs should report zero confidence."""
        cfg = {"temperature": 0.7, "top_k": 40}
        factor = detect_config_change(cfg, cfg)
        assert factor.confidence == 0.0

    def test_empty_configs(self):
        """Two empty configs should report zero confidence."""
        factor = detect_config_change({}, {})
        assert factor.confidence == 0.0

    def test_extra_key(self):
        """New key in config_b should be detected."""
        factor = detect_config_change(
            {"temperature": 0.7},
            {"temperature": 0.7, "top_k": 40},
        )
        assert factor.confidence > 0.0
        assert "top_k" in factor.evidence

    def test_multiple_changes_higher_confidence(self):
        """More changed keys should yield higher confidence."""
        one_change = detect_config_change(
            {"a": 1, "b": 2, "c": 3},
            {"a": 99, "b": 2, "c": 3},
        )
        three_changes = detect_config_change(
            {"a": 1, "b": 2, "c": 3},
            {"a": 99, "b": 99, "c": 99},
        )
        assert three_changes.confidence > one_change.confidence


# ---------------------------------------------------------------------------
# attribute_change
# ---------------------------------------------------------------------------


class TestAttributeChange:
    def test_primary_factor_model(self):
        """Model change should dominate when only model differs."""
        context = {
            "items_a": 100, "items_b": 100,
            "model_a": "gpt-4", "model_b": "gpt-4o",
            "prompt_a": "same", "prompt_b": "same",
            "config_a": {}, "config_b": {},
        }
        result = attribute_change("accuracy", 0.15, context)
        assert result.primary_factor == "model"
        assert result.metric_id == "accuracy"
        assert result.delta == 0.15

    def test_primary_factor_dataset(self):
        """Dataset hash change should attribute to dataset."""
        context = {
            "items_a": 100, "items_b": 100,
            "hash_a": "aaa", "hash_b": "bbb",
            "model_a": "gpt-4", "model_b": "gpt-4",
            "prompt_a": "same", "prompt_b": "same",
            "config_a": {}, "config_b": {},
        }
        result = attribute_change("accuracy", -0.05, context)
        assert result.primary_factor == "dataset"

    def test_no_changes_yields_unknown(self):
        """No detectable changes should yield unknown factor."""
        context = {
            "items_a": 100, "items_b": 100,
            "model_a": "gpt-4", "model_b": "gpt-4",
            "prompt_a": "same", "prompt_b": "same",
            "config_a": {}, "config_b": {},
        }
        result = attribute_change("accuracy", 0.01, context)
        assert result.primary_factor == "unknown"

    def test_multiple_factors(self):
        """Multiple changes should list multiple factors, pick highest confidence."""
        context = {
            "items_a": 100, "items_b": 200,
            "model_a": "gpt-4", "model_b": "gpt-4o",
            "prompt_a": "same", "prompt_b": "same",
            "config_a": {}, "config_b": {},
        }
        result = attribute_change("accuracy", 0.2, context)
        # Model change (0.95) should beat dataset count change (0.8)
        assert result.primary_factor == "model"
        assert len(result.factors) >= 2

    def test_all_factors_change(self):
        """All factors changing should still pick the highest confidence one."""
        context = {
            "items_a": 50, "items_b": 100,
            "model_a": "gpt-3.5", "model_b": "gpt-4",
            "prompt_a": "old prompt", "prompt_b": "completely new different prompt text",
            "config_a": {"temp": 0.5}, "config_b": {"temp": 0.9},
        }
        result = attribute_change("accuracy", 0.3, context)
        assert result.primary_factor in ("model", "dataset", "prompt", "config")
        assert len(result.factors) >= 3


# ---------------------------------------------------------------------------
# build_attribution_report
# ---------------------------------------------------------------------------


class TestBuildAttributionReport:
    def test_dominant_factor(self):
        """Dominant factor should be the most frequent primary factor."""
        a1 = ChangeAttribution(metric_id="m1", delta=0.1, primary_factor="model")
        a2 = ChangeAttribution(metric_id="m2", delta=0.2, primary_factor="model")
        a3 = ChangeAttribution(metric_id="m3", delta=-0.1, primary_factor="dataset")
        report = build_attribution_report([a1, a2, a3])
        assert report.dominant_factor == "model"
        assert report.factor_distribution["model"] == 2
        assert report.factor_distribution["dataset"] == 1

    def test_empty_attributions(self):
        """Empty list should produce empty report."""
        report = build_attribution_report([])
        assert report.attributions == []
        assert report.dominant_factor == ""
        assert report.factor_distribution == {}

    def test_single_attribution(self):
        """Single attribution should be its own dominant."""
        a1 = ChangeAttribution(metric_id="m1", delta=0.5, primary_factor="prompt")
        report = build_attribution_report([a1])
        assert report.dominant_factor == "prompt"


# ---------------------------------------------------------------------------
# AttributionReport.format_text
# ---------------------------------------------------------------------------


class TestAttributionReportFormatText:
    def test_contains_summary(self):
        """Format text should include dominant factor and metric details."""
        a1 = ChangeAttribution(
            metric_id="accuracy", delta=0.15, primary_factor="model",
            factors=[ChangeFactor(factor="model", confidence=0.95, evidence="Model changed")],
        )
        report = build_attribution_report([a1])
        text = report.format_text()
        assert "CHANGE ATTRIBUTION REPORT" in text
        assert "Dominant factor: model" in text
        assert "accuracy" in text

    def test_empty_format(self):
        """Empty report format should note no attributions."""
        text = AttributionReport().format_text()
        assert "No attributions" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestChangeAttributionRoundtrip:
    def test_roundtrip(self):
        """as_dict -> from_dict should preserve all fields."""
        original = ChangeAttribution(
            metric_id="accuracy",
            delta=0.15,
            primary_factor="model",
            factors=[
                ChangeFactor(factor="model", confidence=0.95, evidence="Model changed: gpt-4 -> gpt-4o"),
                ChangeFactor(factor="dataset", confidence=0.8, evidence="Item count changed: 100 -> 200"),
            ],
        )
        restored = ChangeAttribution.from_dict(original.as_dict())
        assert restored.metric_id == original.metric_id
        assert restored.delta == original.delta
        assert restored.primary_factor == original.primary_factor
        assert len(restored.factors) == 2
        assert restored.factors[0].factor == "model"
        assert restored.factors[1].confidence == 0.8


class TestAttributionReportRoundtrip:
    def test_roundtrip(self):
        """as_dict -> from_dict should preserve the full report."""
        a1 = ChangeAttribution(metric_id="m1", delta=0.1, primary_factor="model")
        a2 = ChangeAttribution(metric_id="m2", delta=-0.2, primary_factor="dataset")
        report = build_attribution_report([a1, a2])
        restored = AttributionReport.from_dict(report.as_dict())
        assert restored.dominant_factor == report.dominant_factor
        assert restored.factor_distribution == report.factor_distribution
        assert len(restored.attributions) == 2


class TestChangeFactorAsDict:
    def test_as_dict(self):
        """ChangeFactor.as_dict should return correct dict."""
        f = ChangeFactor(factor="model", confidence=0.95, evidence="Changed")
        d = f.as_dict()
        assert d["factor"] == "model"
        assert d["confidence"] == 0.95
        assert d["evidence"] == "Changed"
