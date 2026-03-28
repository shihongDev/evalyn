"""
Connection pooling: reuse connections for high-throughput multi-threaded evaluation.

Provides a simple pool that manages string-based connection IDs with
acquire/release semantics, stats tracking, and a factory function.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PoolConfig:
    """Configuration for a connection pool."""

    max_connections: int = 5
    timeout_seconds: float = 30.0
    check_on_return: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_connections": self.max_connections,
            "timeout_seconds": self.timeout_seconds,
            "check_on_return": self.check_on_return,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PoolConfig:
        return cls(
            max_connections=data.get("max_connections", 5),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            check_on_return=data.get("check_on_return", True),
        )


@dataclass
class PoolStats:
    """Runtime statistics for a connection pool."""

    total_created: int = 0
    active: int = 0
    idle: int = 0
    total_acquired: int = 0
    total_released: int = 0
    wait_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_created": self.total_created,
            "active": self.active,
            "idle": self.idle,
            "total_acquired": self.total_acquired,
            "total_released": self.total_released,
            "wait_count": self.wait_count,
        }

    def format_text(self) -> str:
        lines = [
            "Pool Statistics",
            f"  total_created: {self.total_created}",
            f"  active: {self.active}",
            f"  idle: {self.idle}",
            f"  total_acquired: {self.total_acquired}",
            f"  total_released: {self.total_released}",
            f"  wait_count: {self.wait_count}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Connection Pool
# ---------------------------------------------------------------------------


class ConnectionPool:
    """Thread-safe connection pool using string-based connection IDs."""

    def __init__(self, config: PoolConfig) -> None:
        self._config = config
        self._pool: List[Any] = []
        self._active: int = 0
        self._total_created: int = 0
        self._stats = PoolStats()
        self._lock = threading.Lock()

    def acquire(self) -> Any:
        """Get a connection from the pool.

        If the pool has idle connections, return one. If the pool is empty
        and under max_connections, create a new connection. If at max,
        increment wait_count and return None.
        """
        with self._lock:
            if self._pool:
                conn = self._pool.pop()
                self._active += 1
                self._stats.active = self._active
                self._stats.idle = len(self._pool)
                self._stats.total_acquired += 1
                return conn

            if self._active < self._config.max_connections:
                self._total_created += 1
                conn = f"conn-{self._total_created}"
                self._active += 1
                self._stats.total_created = self._total_created
                self._stats.active = self._active
                self._stats.idle = len(self._pool)
                self._stats.total_acquired += 1
                return conn

            self._stats.wait_count += 1
            return None

    def release(self, conn: Any) -> None:
        """Return a connection to the pool."""
        with self._lock:
            self._pool.append(conn)
            self._active -= 1
            self._stats.active = self._active
            self._stats.idle = len(self._pool)
            self._stats.total_released += 1

    def get_stats(self) -> PoolStats:
        """Return current pool statistics."""
        with self._lock:
            return PoolStats(
                total_created=self._stats.total_created,
                active=self._stats.active,
                idle=self._stats.idle,
                total_acquired=self._stats.total_acquired,
                total_released=self._stats.total_released,
                wait_count=self._stats.wait_count,
            )

    def close_all(self) -> None:
        """Close all idle connections in the pool."""
        with self._lock:
            self._pool.clear()
            self._stats.idle = 0

    def size(self) -> int:
        """Current pool size (idle + active)."""
        with self._lock:
            return len(self._pool) + self._active

    def is_full(self) -> bool:
        """True if active connections have reached max_connections."""
        with self._lock:
            return self._active >= self._config.max_connections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_pool(max_connections: int = 5) -> ConnectionPool:
    """Create a connection pool with the given max connections."""
    config = PoolConfig(max_connections=max_connections)
    return ConnectionPool(config)
