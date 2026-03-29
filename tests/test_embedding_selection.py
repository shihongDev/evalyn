"""Tests for the embedding selection module."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.embedding_selection import (
    BUILTIN_MODELS,
    EmbeddingModelConfig,
    EmbeddingRegistry,
    create_default_registry,
    format_model_catalog,
    get_cache_key,
)


# ---------------------------------------------------------------------------
# EmbeddingModelConfig
# ---------------------------------------------------------------------------


class TestEmbeddingModelConfig:
    def test_basic_creation(self):
        cfg = EmbeddingModelConfig(
            model_id="test", model_name="Test Model", dimensions=128, source="local"
        )
        assert cfg.model_id == "test"
        assert cfg.model_name == "Test Model"
        assert cfg.dimensions == 128
        assert cfg.source == "local"

    def test_defaults(self):
        cfg = EmbeddingModelConfig(
            model_id="x", model_name="X", dimensions=64, source="api"
        )
        assert cfg.max_tokens == 512
        assert cfg.cache_key == ""

    def test_custom_max_tokens(self):
        cfg = EmbeddingModelConfig(
            model_id="x", model_name="X", dimensions=64, source="api", max_tokens=1024
        )
        assert cfg.max_tokens == 1024

    def test_custom_cache_key(self):
        cfg = EmbeddingModelConfig(
            model_id="x", model_name="X", dimensions=64, source="api", cache_key="abc"
        )
        assert cfg.cache_key == "abc"

    def test_as_dict(self):
        cfg = EmbeddingModelConfig(
            model_id="a", model_name="A", dimensions=100, source="huggingface"
        )
        d = cfg.as_dict()
        assert d["model_id"] == "a"
        assert d["model_name"] == "A"
        assert d["dimensions"] == 100
        assert d["source"] == "huggingface"
        assert d["max_tokens"] == 512
        assert d["cache_key"] == ""

    def test_from_dict_minimal(self):
        d = {
            "model_id": "b",
            "model_name": "B",
            "dimensions": 200,
            "source": "local",
        }
        cfg = EmbeddingModelConfig.from_dict(d)
        assert cfg.model_id == "b"
        assert cfg.dimensions == 200
        assert cfg.max_tokens == 512
        assert cfg.cache_key == ""

    def test_from_dict_full(self):
        d = {
            "model_id": "c",
            "model_name": "C",
            "dimensions": 300,
            "source": "api",
            "max_tokens": 256,
            "cache_key": "ck",
        }
        cfg = EmbeddingModelConfig.from_dict(d)
        assert cfg.max_tokens == 256
        assert cfg.cache_key == "ck"

    def test_roundtrip(self):
        cfg = EmbeddingModelConfig(
            model_id="rt",
            model_name="Roundtrip",
            dimensions=512,
            source="huggingface",
            max_tokens=1024,
            cache_key="rtk",
        )
        restored = EmbeddingModelConfig.from_dict(cfg.as_dict())
        assert restored.model_id == cfg.model_id
        assert restored.model_name == cfg.model_name
        assert restored.dimensions == cfg.dimensions
        assert restored.source == cfg.source
        assert restored.max_tokens == cfg.max_tokens
        assert restored.cache_key == cfg.cache_key

    def test_as_dict_returns_new_dict(self):
        cfg = EmbeddingModelConfig(
            model_id="x", model_name="X", dimensions=64, source="api"
        )
        d1 = cfg.as_dict()
        d2 = cfg.as_dict()
        assert d1 is not d2

    def test_equality(self):
        a = EmbeddingModelConfig("x", "X", 64, "api")
        b = EmbeddingModelConfig("x", "X", 64, "api")
        assert a == b


# ---------------------------------------------------------------------------
# BUILTIN_MODELS
# ---------------------------------------------------------------------------


class TestBuiltinModels:
    def test_has_four_models(self):
        assert len(BUILTIN_MODELS) == 4

    def test_minilm_present(self):
        m = BUILTIN_MODELS["all-MiniLM-L6-v2"]
        assert m.dimensions == 384
        assert m.source == "huggingface"

    def test_mpnet_present(self):
        m = BUILTIN_MODELS["all-mpnet-base-v2"]
        assert m.dimensions == 768
        assert m.source == "huggingface"

    def test_bge_present(self):
        m = BUILTIN_MODELS["bge-small-en-v1.5"]
        assert m.dimensions == 384
        assert m.source == "huggingface"

    def test_openai_present(self):
        m = BUILTIN_MODELS["text-embedding-3-small"]
        assert m.dimensions == 1536
        assert m.source == "api"

    def test_all_have_model_id_matching_key(self):
        for key, cfg in BUILTIN_MODELS.items():
            assert cfg.model_id == key


# ---------------------------------------------------------------------------
# EmbeddingRegistry
# ---------------------------------------------------------------------------


class TestEmbeddingRegistry:
    def test_empty_registry(self):
        reg = EmbeddingRegistry()
        assert reg.list_models() == []

    def test_register_and_get(self):
        reg = EmbeddingRegistry()
        cfg = EmbeddingModelConfig("t", "T", 32, "local")
        reg.register(cfg)
        assert reg.get("t") is cfg

    def test_get_missing_returns_none(self):
        reg = EmbeddingRegistry()
        assert reg.get("nope") is None

    def test_list_models(self):
        reg = EmbeddingRegistry()
        c1 = EmbeddingModelConfig("a", "A", 10, "local")
        c2 = EmbeddingModelConfig("b", "B", 20, "api")
        reg.register(c1)
        reg.register(c2)
        models = reg.list_models()
        assert len(models) == 2
        ids = {m.model_id for m in models}
        assert ids == {"a", "b"}

    def test_register_overwrites(self):
        reg = EmbeddingRegistry()
        c1 = EmbeddingModelConfig("x", "X-old", 10, "local")
        c2 = EmbeddingModelConfig("x", "X-new", 20, "api")
        reg.register(c1)
        reg.register(c2)
        assert reg.get("x") is c2
        assert len(reg.list_models()) == 1

    def test_get_default_returns_minilm_fallback(self):
        reg = EmbeddingRegistry()
        default = reg.get_default()
        assert default.model_id == "all-MiniLM-L6-v2"
        assert default.dimensions == 384

    def test_get_default_from_populated_registry(self):
        reg = create_default_registry()
        default = reg.get_default()
        assert default.model_id == "all-MiniLM-L6-v2"

    def test_set_default(self):
        reg = create_default_registry()
        reg.set_default("all-mpnet-base-v2")
        default = reg.get_default()
        assert default.model_id == "all-mpnet-base-v2"

    def test_set_default_missing_falls_back(self):
        reg = EmbeddingRegistry()
        reg.set_default("nonexistent")
        # Falls back to builtin all-MiniLM-L6-v2
        default = reg.get_default()
        assert default.model_id == "all-MiniLM-L6-v2"

    def test_set_default_custom_model(self):
        reg = EmbeddingRegistry()
        custom = EmbeddingModelConfig("my-model", "My Model", 256, "local")
        reg.register(custom)
        reg.set_default("my-model")
        assert reg.get_default().model_id == "my-model"


# ---------------------------------------------------------------------------
# create_default_registry
# ---------------------------------------------------------------------------


class TestCreateDefaultRegistry:
    def test_has_all_builtins(self):
        reg = create_default_registry()
        assert len(reg.list_models()) == 4

    def test_all_builtins_retrievable(self):
        reg = create_default_registry()
        for model_id in BUILTIN_MODELS:
            assert reg.get(model_id) is not None

    def test_returns_new_registry_each_time(self):
        r1 = create_default_registry()
        r2 = create_default_registry()
        assert r1 is not r2


# ---------------------------------------------------------------------------
# get_cache_key
# ---------------------------------------------------------------------------


class TestGetCacheKey:
    def test_returns_hex_string(self):
        key = get_cache_key("model", "hello")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex length

    def test_deterministic(self):
        k1 = get_cache_key("m", "text")
        k2 = get_cache_key("m", "text")
        assert k1 == k2

    def test_different_model_different_key(self):
        k1 = get_cache_key("model-a", "same text")
        k2 = get_cache_key("model-b", "same text")
        assert k1 != k2

    def test_different_text_different_key(self):
        k1 = get_cache_key("model", "text a")
        k2 = get_cache_key("model", "text b")
        assert k1 != k2

    def test_matches_direct_sha256(self):
        model_id = "test-model"
        text = "hello world"
        expected = hashlib.sha256((model_id + text).encode("utf-8")).hexdigest()
        assert get_cache_key(model_id, text) == expected

    def test_empty_inputs(self):
        key = get_cache_key("", "")
        expected = hashlib.sha256(b"").hexdigest()
        assert key == expected


# ---------------------------------------------------------------------------
# format_model_catalog
# ---------------------------------------------------------------------------


class TestFormatModelCatalog:
    def test_empty_registry(self):
        reg = EmbeddingRegistry()
        result = format_model_catalog(reg)
        assert "No models registered" in result

    def test_contains_model_ids(self):
        reg = create_default_registry()
        result = format_model_catalog(reg)
        assert "all-MiniLM-L6-v2" in result
        assert "all-mpnet-base-v2" in result
        assert "bge-small-en-v1.5" in result
        assert "text-embedding-3-small" in result

    def test_contains_dimensions(self):
        reg = create_default_registry()
        result = format_model_catalog(reg)
        assert "384d" in result
        assert "768d" in result
        assert "1536d" in result

    def test_default_marker(self):
        reg = create_default_registry()
        result = format_model_catalog(reg)
        assert "(default)" in result

    def test_header(self):
        reg = create_default_registry()
        result = format_model_catalog(reg)
        assert result.startswith("Embedding Models:")

    def test_source_shown(self):
        reg = create_default_registry()
        result = format_model_catalog(reg)
        assert "source=huggingface" in result
        assert "source=api" in result
