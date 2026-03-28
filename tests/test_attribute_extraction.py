from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.attribute_extraction import (
    AttributeExtractor,
    CostExtractor,
    ExtractedAttribute,
    ExtractionResult,
    ModelExtractor,
    TokenExtractor,
    apply_extractors,
    create_extractor_pipeline,
    get_registered_extractors,
    register_extractor,
    reset_extractor_registry,
)


def _make_span(**attrs) -> Span:
    return Span(
        id="s1",
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        attributes=attrs,
    )


# ---------------------------------------------------------------------------
# ExtractedAttribute
# ---------------------------------------------------------------------------


class TestExtractedAttribute:
    def test_as_dict(self):
        a = ExtractedAttribute(key="model", value="gpt-4", source="plugin")
        d = a.as_dict()
        assert d == {"key": "model", "value": "gpt-4", "source": "plugin"}

    def test_default_source(self):
        a = ExtractedAttribute(key="k", value=1)
        assert a.source == "plugin"


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_as_dict_empty(self):
        r = ExtractionResult(plugin_name="test")
        assert r.as_dict() == {"attributes": [], "plugin_name": "test"}

    def test_as_dict_with_attributes(self):
        r = ExtractionResult(
            attributes=[ExtractedAttribute("k", "v")],
            plugin_name="p",
        )
        d = r.as_dict()
        assert len(d["attributes"]) == 1
        assert d["attributes"][0]["key"] == "k"


# ---------------------------------------------------------------------------
# ModelExtractor
# ---------------------------------------------------------------------------


class TestModelExtractor:
    def test_extract_model(self):
        span = _make_span(model="gpt-4")
        attrs = ModelExtractor().extract(span)
        assert len(attrs) == 1
        assert attrs[0].key == "model"
        assert attrs[0].value == "gpt-4"

    def test_extract_model_name(self):
        span = _make_span(model_name="claude-3")
        attrs = ModelExtractor().extract(span)
        assert attrs[0].value == "claude-3"

    def test_extract_llm_model(self):
        span = _make_span(**{"llm.model": "gemini-pro"})
        attrs = ModelExtractor().extract(span)
        assert attrs[0].value == "gemini-pro"

    def test_no_model(self):
        span = _make_span(foo="bar")
        attrs = ModelExtractor().extract(span)
        assert attrs == []

    def test_source(self):
        span = _make_span(model="gpt-4")
        attrs = ModelExtractor().extract(span)
        assert attrs[0].source == "ModelExtractor"


# ---------------------------------------------------------------------------
# TokenExtractor
# ---------------------------------------------------------------------------


class TestTokenExtractor:
    def test_extract_input_output(self):
        span = _make_span(input_tokens=100, output_tokens=50)
        attrs = TokenExtractor().extract(span)
        keys = {a.key: a.value for a in attrs}
        assert keys["input_tokens"] == 100
        assert keys["output_tokens"] == 50
        assert keys["total_tokens"] == 150

    def test_extract_prompt_completion_keys(self):
        span = _make_span(prompt_tokens=200, completion_tokens=80)
        attrs = TokenExtractor().extract(span)
        keys = {a.key: a.value for a in attrs}
        assert keys["input_tokens"] == 200
        assert keys["output_tokens"] == 80

    def test_extract_explicit_total(self):
        span = _make_span(input_tokens=10, output_tokens=5, total_tokens=999)
        attrs = TokenExtractor().extract(span)
        keys = {a.key: a.value for a in attrs}
        assert keys["total_tokens"] == 999

    def test_no_tokens(self):
        span = _make_span(foo="bar")
        attrs = TokenExtractor().extract(span)
        assert attrs == []

    def test_source(self):
        span = _make_span(input_tokens=1)
        attrs = TokenExtractor().extract(span)
        assert attrs[0].source == "TokenExtractor"


# ---------------------------------------------------------------------------
# CostExtractor
# ---------------------------------------------------------------------------


class TestCostExtractor:
    def test_explicit_cost(self):
        span = _make_span(cost=0.05)
        attrs = CostExtractor().extract(span)
        assert len(attrs) == 1
        assert attrs[0].key == "cost"
        assert attrs[0].value == 0.05

    def test_estimated_cost(self):
        span = _make_span(input_tokens=1000, output_tokens=500)
        attrs = CostExtractor().extract(span)
        assert len(attrs) == 1
        assert attrs[0].key == "cost"
        assert attrs[0].value > 0

    def test_no_cost_no_tokens(self):
        span = _make_span(foo="bar")
        attrs = CostExtractor().extract(span)
        assert attrs == []

    def test_source(self):
        span = _make_span(cost=0.01)
        attrs = CostExtractor().extract(span)
        assert attrs[0].source == "CostExtractor"


# ---------------------------------------------------------------------------
# create_extractor_pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_chain_extractors(self):
        span = _make_span(model="gpt-4", input_tokens=10, output_tokens=5)
        pipeline = create_extractor_pipeline([ModelExtractor(), TokenExtractor()])
        result = pipeline(span)
        assert result["model"] == "gpt-4"
        assert result["input_tokens"] == 10
        assert result["total_tokens"] == 15

    def test_empty_pipeline(self):
        span = _make_span(model="gpt-4")
        pipeline = create_extractor_pipeline([])
        assert pipeline(span) == {}


# ---------------------------------------------------------------------------
# apply_extractors
# ---------------------------------------------------------------------------


class TestApplyExtractors:
    def test_returns_new_span(self):
        span = _make_span(model="gpt-4")
        new_span = apply_extractors(span, [ModelExtractor()])
        assert new_span is not span
        assert new_span.attributes.get("model") == "gpt-4"

    def test_original_unchanged(self):
        span = _make_span(input_tokens=100, output_tokens=50)
        new_span = apply_extractors(span, [TokenExtractor()])
        # Original should not gain total_tokens
        assert "total_tokens" not in span.attributes
        assert new_span.attributes["total_tokens"] == 150


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self):
        reset_extractor_registry()

    def test_register_and_get(self):
        ext = ModelExtractor()
        register_extractor("model", ext)
        reg = get_registered_extractors()
        assert "model" in reg
        assert reg["model"] is ext

    def test_reset(self):
        register_extractor("x", ModelExtractor())
        reset_extractor_registry()
        assert get_registered_extractors() == {}

    def test_get_empty(self):
        assert get_registered_extractors() == {}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_model_extractor_is_attribute_extractor(self):
        assert isinstance(ModelExtractor(), AttributeExtractor)

    def test_token_extractor_is_attribute_extractor(self):
        assert isinstance(TokenExtractor(), AttributeExtractor)

    def test_cost_extractor_is_attribute_extractor(self):
        assert isinstance(CostExtractor(), AttributeExtractor)
