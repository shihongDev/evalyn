"""
Tests for evalyn_sdk.storage.cloud_backend.

Run with: uv run pytest tests/test_cloud_backend.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.cloud_backend import (
    CloudConfig,
    CloudObject,
    CloudOperationResult,
    LocalCloudSimulator,
    create_cloud_backend,
    estimate_cloud_cost,
    validate_cloud_config,
)


# ---------------------------------------------------------------------------
# LocalCloudSimulator - put/get
# ---------------------------------------------------------------------------


class TestLocalCloudSimulatorPutGet:
    def test_put_get_roundtrip(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("key1", b"hello")
        data, result = sim.get("key1")
        assert data == b"hello"
        assert result.success is True

    def test_put_returns_success(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        result = sim.put("key1", b"data")
        assert result.success is True
        assert result.operation == "put"
        assert result.key == "key1"

    def test_get_missing_returns_error(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        data, result = sim.get("nonexistent")
        assert data is None
        assert result.success is False
        assert "not found" in result.error

    def test_put_overwrites(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("key1", b"first")
        sim.put("key1", b"second")
        data, _ = sim.get("key1")
        assert data == b"second"

    def test_put_empty_data(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        result = sim.put("empty", b"")
        assert result.success is True
        data, _ = sim.get("empty")
        assert data == b""

    def test_put_large_data(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        large = b"x" * 1_000_000
        sim.put("large", large)
        data, result = sim.get("large")
        assert data == large
        assert result.success is True


# ---------------------------------------------------------------------------
# LocalCloudSimulator - delete
# ---------------------------------------------------------------------------


class TestLocalCloudSimulatorDelete:
    def test_delete_removes(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("key1", b"data")
        result = sim.delete("key1")
        assert result.success is True
        assert sim.exists("key1") is False

    def test_delete_missing_returns_error(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        result = sim.delete("nonexistent")
        assert result.success is False
        assert "not found" in result.error

    def test_delete_operation_field(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("k", b"v")
        result = sim.delete("k")
        assert result.operation == "delete"


# ---------------------------------------------------------------------------
# LocalCloudSimulator - list_objects
# ---------------------------------------------------------------------------


class TestLocalCloudSimulatorListObjects:
    def test_list_all(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("a/1", b"data1")
        sim.put("b/2", b"data2")
        objects = sim.list_objects()
        assert len(objects) == 2

    def test_list_with_prefix(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("prefix/a", b"data1")
        sim.put("prefix/b", b"data2")
        sim.put("other/c", b"data3")
        objects = sim.list_objects(prefix="prefix/")
        assert len(objects) == 2
        keys = [o.key for o in objects]
        assert "prefix/a" in keys
        assert "prefix/b" in keys

    def test_list_empty_simulator(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        objects = sim.list_objects()
        assert objects == []

    def test_list_objects_have_size(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("key1", b"hello")
        objects = sim.list_objects()
        assert objects[0].size_bytes == 5

    def test_list_sorted_by_key(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("z_key", b"data")
        sim.put("a_key", b"data")
        objects = sim.list_objects()
        assert objects[0].key == "a_key"
        assert objects[1].key == "z_key"


# ---------------------------------------------------------------------------
# LocalCloudSimulator - exists
# ---------------------------------------------------------------------------


class TestLocalCloudSimulatorExists:
    def test_exists_true(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("key1", b"data")
        assert sim.exists("key1") is True

    def test_exists_false(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        assert sim.exists("missing") is False

    def test_exists_after_delete(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("key1", b"data")
        sim.delete("key1")
        assert sim.exists("key1") is False


# ---------------------------------------------------------------------------
# LocalCloudSimulator - get_stats
# ---------------------------------------------------------------------------


class TestLocalCloudSimulatorGetStats:
    def test_empty_stats(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        stats = sim.get_stats()
        assert stats["object_count"] == 0
        assert stats["total_size_bytes"] == 0

    def test_stats_counts(self):
        sim = LocalCloudSimulator(base_dir="/tmp")
        sim.put("a", b"12345")
        sim.put("b", b"67")
        stats = sim.get_stats()
        assert stats["object_count"] == 2
        assert stats["total_size_bytes"] == 7


# ---------------------------------------------------------------------------
# validate_cloud_config
# ---------------------------------------------------------------------------


class TestValidateCloudConfig:
    def test_valid_local(self):
        config = CloudConfig(provider="local")
        valid, errors = validate_cloud_config(config)
        assert valid is True
        assert errors == []

    def test_valid_s3(self):
        config = CloudConfig(
            provider="s3", bucket="my-bucket", region="us-east-1"
        )
        valid, errors = validate_cloud_config(config)
        assert valid is True

    def test_s3_missing_bucket(self):
        config = CloudConfig(provider="s3", region="us-east-1")
        valid, errors = validate_cloud_config(config)
        assert valid is False
        assert any("bucket" in e for e in errors)

    def test_s3_missing_region(self):
        config = CloudConfig(provider="s3", bucket="my-bucket")
        valid, errors = validate_cloud_config(config)
        assert valid is False
        assert any("region" in e for e in errors)

    def test_gcs_missing_bucket(self):
        config = CloudConfig(provider="gcs", region="us-central1")
        valid, errors = validate_cloud_config(config)
        assert valid is False
        assert any("bucket" in e for e in errors)

    def test_unsupported_provider(self):
        config = CloudConfig(provider="azure")
        valid, errors = validate_cloud_config(config)
        assert valid is False
        assert any("unsupported" in e for e in errors)


# ---------------------------------------------------------------------------
# estimate_cloud_cost
# ---------------------------------------------------------------------------


class TestEstimateCloudCost:
    def test_s3_cost_structure(self):
        result = estimate_cloud_cost(1024 ** 3, provider="s3")
        assert "storage_gb" in result
        assert "monthly_storage_usd" in result
        assert "monthly_total_usd" in result

    def test_s3_one_gb(self):
        result = estimate_cloud_cost(1024 ** 3, provider="s3")
        assert result["storage_gb"] == 1.0
        assert result["monthly_storage_usd"] > 0

    def test_gcs_cost(self):
        result = estimate_cloud_cost(1024 ** 3, provider="gcs")
        assert result["monthly_storage_usd"] > 0

    def test_local_cost_zero(self):
        result = estimate_cloud_cost(1024 ** 3, provider="local")
        assert result["monthly_storage_usd"] == 0.0

    def test_zero_bytes(self):
        result = estimate_cloud_cost(0, provider="s3")
        assert result["storage_gb"] == 0.0


# ---------------------------------------------------------------------------
# create_cloud_backend
# ---------------------------------------------------------------------------


class TestCreateCloudBackend:
    def test_returns_simulator(self):
        config = CloudConfig(provider="local")
        backend = create_cloud_backend(config)
        assert isinstance(backend, LocalCloudSimulator)

    def test_simulator_is_usable(self):
        config = CloudConfig(provider="s3", bucket="test")
        backend = create_cloud_backend(config)
        backend.put("key", b"value")
        data, _ = backend.get("key")
        assert data == b"value"


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundTrips:
    def test_cloud_config_roundtrip(self):
        original = CloudConfig(
            provider="s3",
            bucket="my-bucket",
            prefix="data/",
            region="eu-west-1",
            credentials_env="AWS_KEY",
        )
        restored = CloudConfig.from_dict(original.as_dict())
        assert restored.provider == original.provider
        assert restored.bucket == original.bucket
        assert restored.prefix == original.prefix
        assert restored.region == original.region
        assert restored.credentials_env == original.credentials_env

    def test_cloud_object_roundtrip(self):
        original = CloudObject(
            key="evalyn/run-001.json",
            size_bytes=4096,
            last_modified="2026-03-28T12:00:00",
            content_type="application/json",
        )
        restored = CloudObject.from_dict(original.as_dict())
        assert restored.key == original.key
        assert restored.size_bytes == original.size_bytes
        assert restored.content_type == original.content_type

    def test_cloud_operation_result_as_dict(self):
        result = CloudOperationResult(
            success=True,
            operation="put",
            key="test-key",
            duration_ms=1.5,
        )
        d = result.as_dict()
        assert d["success"] is True
        assert d["operation"] == "put"
        assert d["key"] == "test-key"

    def test_cloud_config_defaults(self):
        config = CloudConfig.from_dict({})
        assert config.provider == "local"
        assert config.bucket == ""
        assert config.prefix == "evalyn/"
