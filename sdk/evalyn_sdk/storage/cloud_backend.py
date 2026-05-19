"""
Cloud storage backend: optional S3/GCS storage for large datasets.

Provides a local simulator for development and testing, plus validation
and cost estimation helpers. Real S3/GCS backends would require boto3
or google-cloud-storage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CloudConfig:
    """Configuration for a cloud storage backend."""

    provider: str = "local"  # "local" / "s3" / "gcs"
    bucket: str = ""
    prefix: str = "evalyn/"
    region: str = ""
    credentials_env: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "region": self.region,
            "credentials_env": self.credentials_env,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudConfig:
        return cls(
            provider=data.get("provider", "local"),
            bucket=data.get("bucket", ""),
            prefix=data.get("prefix", "evalyn/"),
            region=data.get("region", ""),
            credentials_env=data.get("credentials_env", ""),
        )


@dataclass
class CloudObject:
    """Metadata for a single cloud object."""

    key: str = ""
    size_bytes: int = 0
    last_modified: str = ""
    content_type: str = "application/json"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudObject:
        return cls(
            key=data.get("key", ""),
            size_bytes=data.get("size_bytes", 0),
            last_modified=data.get("last_modified", ""),
            content_type=data.get("content_type", "application/json"),
        )


@dataclass
class CloudOperationResult:
    """Result of a cloud storage operation."""

    success: bool = False
    operation: str = ""
    key: str = ""
    error: str = ""
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "key": self.key,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Local Cloud Simulator
# ---------------------------------------------------------------------------


class LocalCloudSimulator:
    """Simulates cloud storage using an in-memory dict.

    Useful for development and testing without real cloud credentials.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self._objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> CloudOperationResult:
        """Store an object."""
        start = time.monotonic()
        self._objects[key] = data
        elapsed = (time.monotonic() - start) * 1000.0
        return CloudOperationResult(
            success=True,
            operation="put",
            key=key,
            duration_ms=elapsed,
        )

    def get(self, key: str) -> tuple[bytes | None, CloudOperationResult]:
        """Retrieve an object by key."""
        start = time.monotonic()
        data = self._objects.get(key)
        elapsed = (time.monotonic() - start) * 1000.0
        if data is None:
            return None, CloudOperationResult(
                success=False,
                operation="get",
                key=key,
                error="object not found",
                duration_ms=elapsed,
            )
        return data, CloudOperationResult(
            success=True,
            operation="get",
            key=key,
            duration_ms=elapsed,
        )

    def delete(self, key: str) -> CloudOperationResult:
        """Remove an object by key."""
        start = time.monotonic()
        if key not in self._objects:
            elapsed = (time.monotonic() - start) * 1000.0
            return CloudOperationResult(
                success=False,
                operation="delete",
                key=key,
                error="object not found",
                duration_ms=elapsed,
            )
        del self._objects[key]
        elapsed = (time.monotonic() - start) * 1000.0
        return CloudOperationResult(
            success=True,
            operation="delete",
            key=key,
            duration_ms=elapsed,
        )

    def list_objects(self, prefix: str = "") -> list[CloudObject]:
        """List objects, optionally filtered by key prefix."""
        results: list[CloudObject] = []
        for key, data in sorted(self._objects.items()):
            if prefix and not key.startswith(prefix):
                continue
            results.append(
                CloudObject(
                    key=key,
                    size_bytes=len(data),
                )
            )
        return results

    def exists(self, key: str) -> bool:
        """Check whether an object exists."""
        return key in self._objects

    def get_stats(self) -> dict[str, Any]:
        """Return object count and total size."""
        total_size = sum(len(v) for v in self._objects.values())
        return {
            "object_count": len(self._objects),
            "total_size_bytes": total_size,
        }


# ---------------------------------------------------------------------------
# Factory and Helpers
# ---------------------------------------------------------------------------


def create_cloud_backend(config: CloudConfig) -> LocalCloudSimulator:
    """Create a cloud storage backend from config.

    Currently always returns a LocalCloudSimulator. Real S3/GCS
    implementations would require boto3 or google-cloud-storage.
    """
    return LocalCloudSimulator(base_dir=config.prefix)


def validate_cloud_config(config: CloudConfig) -> tuple[bool, list[str]]:
    """Validate a cloud config and return (valid, errors)."""
    errors: list[str] = []
    if config.provider not in ("local", "s3", "gcs"):
        errors.append(f"unsupported provider: {config.provider}")
    if config.provider in ("s3", "gcs") and not config.bucket:
        errors.append("bucket is required for s3/gcs provider")
    if config.provider in ("s3", "gcs") and not config.region:
        errors.append("region is required for s3/gcs provider")
    return (len(errors) == 0, errors)


def estimate_cloud_cost(
    total_bytes: int, provider: str = "s3"
) -> dict[str, float]:
    """Estimate monthly cloud storage cost.

    Uses approximate public pricing (USD):
    - S3 Standard: $0.023/GB/month storage, $0.0004/1000 PUT, $0.0004/1000 GET
    - GCS Standard: $0.020/GB/month storage, $0.005/1000 PUT, $0.0004/1000 GET
    - Local: free
    """
    gb = total_bytes / (1024 ** 3)
    if provider == "s3":
        storage_cost = gb * 0.023
        request_cost = 0.001  # Rough estimate for typical usage
    elif provider == "gcs":
        storage_cost = gb * 0.020
        request_cost = 0.005
    else:
        storage_cost = 0.0
        request_cost = 0.0
    return {
        "storage_gb": round(gb, 6),
        "monthly_storage_usd": round(storage_cost, 6),
        "monthly_request_usd": round(request_cost, 6),
        "monthly_total_usd": round(storage_cost + request_cost, 6),
    }
