"""Tests for the connection pool module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.storage.connection_pool import (
    ConnectionPool,
    PoolConfig,
    PoolStats,
    create_pool,
)


# ---------------------------------------------------------------------------
# PoolConfig
# ---------------------------------------------------------------------------


class TestPoolConfig:
    def test_defaults(self):
        cfg = PoolConfig()
        assert cfg.max_connections == 5
        assert cfg.timeout_seconds == 30.0
        assert cfg.check_on_return is True

    def test_as_dict(self):
        cfg = PoolConfig(max_connections=10, timeout_seconds=5.0, check_on_return=False)
        d = cfg.as_dict()
        assert d["max_connections"] == 10
        assert d["timeout_seconds"] == 5.0
        assert d["check_on_return"] is False

    def test_from_dict(self):
        d = {"max_connections": 3, "timeout_seconds": 15.0, "check_on_return": False}
        cfg = PoolConfig.from_dict(d)
        assert cfg.max_connections == 3
        assert cfg.timeout_seconds == 15.0
        assert cfg.check_on_return is False

    def test_from_dict_defaults(self):
        cfg = PoolConfig.from_dict({})
        assert cfg.max_connections == 5
        assert cfg.timeout_seconds == 30.0
        assert cfg.check_on_return is True

    def test_roundtrip(self):
        cfg = PoolConfig(max_connections=8, timeout_seconds=60.0, check_on_return=False)
        restored = PoolConfig.from_dict(cfg.as_dict())
        assert restored.max_connections == cfg.max_connections
        assert restored.timeout_seconds == cfg.timeout_seconds
        assert restored.check_on_return == cfg.check_on_return


# ---------------------------------------------------------------------------
# PoolStats
# ---------------------------------------------------------------------------


class TestPoolStats:
    def test_defaults(self):
        stats = PoolStats()
        assert stats.total_created == 0
        assert stats.active == 0
        assert stats.idle == 0

    def test_as_dict(self):
        stats = PoolStats(total_created=3, active=1, idle=2)
        d = stats.as_dict()
        assert d["total_created"] == 3
        assert d["active"] == 1
        assert d["idle"] == 2

    def test_format_text(self):
        stats = PoolStats(total_created=5, active=2, idle=3, total_acquired=10, total_released=8, wait_count=1)
        text = stats.format_text()
        assert "Pool Statistics" in text
        assert "total_created: 5" in text
        assert "active: 2" in text
        assert "idle: 3" in text
        assert "total_acquired: 10" in text
        assert "total_released: 8" in text
        assert "wait_count: 1" in text


# ---------------------------------------------------------------------------
# ConnectionPool
# ---------------------------------------------------------------------------


class TestConnectionPool:
    def test_acquire_returns_connection(self):
        pool = ConnectionPool(PoolConfig(max_connections=3))
        conn = pool.acquire()
        assert conn is not None
        assert conn == "conn-1"

    def test_acquire_creates_new_when_empty(self):
        pool = ConnectionPool(PoolConfig(max_connections=3))
        c1 = pool.acquire()
        c2 = pool.acquire()
        assert c1 == "conn-1"
        assert c2 == "conn-2"

    def test_acquire_returns_none_when_full(self):
        pool = ConnectionPool(PoolConfig(max_connections=2))
        pool.acquire()
        pool.acquire()
        result = pool.acquire()
        assert result is None

    def test_acquire_increments_wait_count_when_full(self):
        pool = ConnectionPool(PoolConfig(max_connections=1))
        pool.acquire()
        pool.acquire()  # should return None
        stats = pool.get_stats()
        assert stats.wait_count == 1

    def test_release_returns_to_pool(self):
        pool = ConnectionPool(PoolConfig(max_connections=2))
        c1 = pool.acquire()
        pool.release(c1)
        # Re-acquiring should return the same connection
        c2 = pool.acquire()
        assert c2 == c1

    def test_release_updates_stats(self):
        pool = ConnectionPool(PoolConfig(max_connections=2))
        c1 = pool.acquire()
        pool.release(c1)
        stats = pool.get_stats()
        assert stats.active == 0
        assert stats.idle == 1
        assert stats.total_released == 1

    def test_get_stats_correct(self):
        pool = ConnectionPool(PoolConfig(max_connections=5))
        pool.acquire()
        pool.acquire()
        c3 = pool.acquire()
        pool.release(c3)
        stats = pool.get_stats()
        assert stats.total_created == 3
        assert stats.active == 2
        assert stats.idle == 1
        assert stats.total_acquired == 3
        assert stats.total_released == 1

    def test_close_all_empties_pool(self):
        pool = ConnectionPool(PoolConfig(max_connections=3))
        c1 = pool.acquire()
        c2 = pool.acquire()
        pool.release(c1)
        pool.release(c2)
        pool.close_all()
        stats = pool.get_stats()
        assert stats.idle == 0

    def test_size_idle_plus_active(self):
        pool = ConnectionPool(PoolConfig(max_connections=5))
        pool.acquire()
        pool.acquire()
        c3 = pool.acquire()
        pool.release(c3)
        assert pool.size() == 3  # 2 active + 1 idle

    def test_is_full_true(self):
        pool = ConnectionPool(PoolConfig(max_connections=2))
        pool.acquire()
        pool.acquire()
        assert pool.is_full() is True

    def test_is_full_false(self):
        pool = ConnectionPool(PoolConfig(max_connections=3))
        pool.acquire()
        assert pool.is_full() is False

    def test_size_starts_at_zero(self):
        pool = ConnectionPool(PoolConfig(max_connections=5))
        assert pool.size() == 0

    def test_single_connection_pool(self):
        pool = ConnectionPool(PoolConfig(max_connections=1))
        c1 = pool.acquire()
        assert c1 is not None
        assert pool.acquire() is None
        pool.release(c1)
        c2 = pool.acquire()
        assert c2 == c1

    def test_zero_max_connections(self):
        pool = ConnectionPool(PoolConfig(max_connections=0))
        result = pool.acquire()
        assert result is None
        stats = pool.get_stats()
        assert stats.wait_count == 1

    def test_close_all_then_acquire_creates_new(self):
        pool = ConnectionPool(PoolConfig(max_connections=3))
        c1 = pool.acquire()
        pool.release(c1)
        pool.close_all()
        c2 = pool.acquire()
        assert c2 is not None
        assert c2 == "conn-2"  # new connection created

    def test_multiple_acquire_release_cycles(self):
        pool = ConnectionPool(PoolConfig(max_connections=2))
        for _ in range(5):
            c = pool.acquire()
            assert c is not None
            pool.release(c)
        stats = pool.get_stats()
        assert stats.total_acquired == 5
        assert stats.total_released == 5

    def test_stats_snapshot_is_independent(self):
        pool = ConnectionPool(PoolConfig(max_connections=5))
        pool.acquire()
        s1 = pool.get_stats()
        pool.acquire()
        s2 = pool.get_stats()
        assert s1.total_acquired == 1
        assert s2.total_acquired == 2


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreatePool:
    def test_default_max(self):
        pool = create_pool()
        # Should be able to acquire 5
        conns = [pool.acquire() for _ in range(5)]
        assert all(c is not None for c in conns)
        assert pool.acquire() is None

    def test_custom_max(self):
        pool = create_pool(max_connections=3)
        conns = [pool.acquire() for _ in range(3)]
        assert all(c is not None for c in conns)
        assert pool.acquire() is None

    def test_returns_connection_pool(self):
        pool = create_pool()
        assert isinstance(pool, ConnectionPool)
