"""Tests for batch job persistence."""

import json

import pytest
from evalyn_sdk.evaluation.batch_persistence import (
    BatchJobManager,
    BatchJobState,
    PersistenceConfig,
)


# -- BatchJobState -----------------------------------------------------------


class TestBatchJobState:
    def test_defaults(self):
        job = BatchJobState(job_id="j1")
        assert job.status == "pending"
        assert job.total_items == 0
        assert job.completed_items == 0
        assert job.results == []
        assert job.created_at == ""
        assert job.updated_at == ""
        assert job.error == ""

    def test_as_dict(self):
        job = BatchJobState(job_id="j1", status="running", total_items=10)
        d = job.as_dict()
        assert d["job_id"] == "j1"
        assert d["status"] == "running"
        assert d["total_items"] == 10

    def test_from_dict_minimal(self):
        job = BatchJobState.from_dict({"job_id": "j2"})
        assert job.job_id == "j2"
        assert job.status == "pending"

    def test_from_dict_full(self):
        d = {
            "job_id": "j3",
            "status": "completed",
            "total_items": 50,
            "completed_items": 50,
            "results": [{"score": 0.9}],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T01:00:00+00:00",
            "error": "",
        }
        job = BatchJobState.from_dict(d)
        assert job.status == "completed"
        assert job.total_items == 50
        assert len(job.results) == 1

    def test_roundtrip(self):
        original = BatchJobState(
            job_id="rt",
            status="failed",
            total_items=5,
            completed_items=3,
            results=[{"a": 1}],
            error="timeout",
        )
        restored = BatchJobState.from_dict(original.as_dict())
        assert restored.job_id == original.job_id
        assert restored.status == original.status
        assert restored.total_items == original.total_items
        assert restored.completed_items == original.completed_items
        assert restored.results == original.results
        assert restored.error == original.error


# -- PersistenceConfig -------------------------------------------------------


class TestPersistenceConfig:
    def test_defaults(self):
        cfg = PersistenceConfig()
        assert cfg.save_dir == ".evalyn_batch"
        assert cfg.save_interval == 10
        assert cfg.auto_resume is True

    def test_as_dict(self):
        cfg = PersistenceConfig(save_dir="/tmp/batch", save_interval=5)
        d = cfg.as_dict()
        assert d["save_dir"] == "/tmp/batch"
        assert d["save_interval"] == 5

    def test_from_dict(self):
        cfg = PersistenceConfig.from_dict(
            {"save_dir": "/data", "save_interval": 20, "auto_resume": False}
        )
        assert cfg.save_dir == "/data"
        assert cfg.save_interval == 20
        assert cfg.auto_resume is False

    def test_from_dict_defaults(self):
        cfg = PersistenceConfig.from_dict({})
        assert cfg.save_dir == ".evalyn_batch"
        assert cfg.save_interval == 10
        assert cfg.auto_resume is True

    def test_roundtrip(self):
        original = PersistenceConfig(save_dir="/x", save_interval=7, auto_resume=False)
        restored = PersistenceConfig.from_dict(original.as_dict())
        assert restored.save_dir == original.save_dir
        assert restored.save_interval == original.save_interval
        assert restored.auto_resume == original.auto_resume


# -- BatchJobManager ---------------------------------------------------------


class TestBatchJobManager:
    def _make_manager(self) -> BatchJobManager:
        return BatchJobManager(PersistenceConfig())

    def test_create_job(self):
        mgr = self._make_manager()
        job = mgr.create_job("j1", total_items=100)
        assert job.job_id == "j1"
        assert job.status == "pending"
        assert job.total_items == 100
        assert job.completed_items == 0
        assert job.created_at != ""
        assert job.updated_at != ""

    def test_update_job(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=50)
        job = mgr.update_job("j1", completed=10, results=[{"s": 1}])
        assert job.completed_items == 10
        assert job.status == "running"
        assert len(job.results) == 1

    def test_update_job_no_results(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=20)
        job = mgr.update_job("j1", completed=5)
        assert job.completed_items == 5
        assert job.results == []

    def test_complete_job(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        mgr.update_job("j1", completed=10)
        job = mgr.complete_job("j1")
        assert job.status == "completed"

    def test_fail_job(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        job = mgr.fail_job("j1", error="API timeout")
        assert job.status == "failed"
        assert job.error == "API timeout"

    def test_get_job_exists(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=5)
        job = mgr.get_job("j1")
        assert job is not None
        assert job.job_id == "j1"

    def test_get_job_missing(self):
        mgr = self._make_manager()
        assert mgr.get_job("nonexistent") is None

    def test_list_jobs(self):
        mgr = self._make_manager()
        mgr.create_job("a", total_items=1)
        mgr.create_job("b", total_items=2)
        mgr.create_job("c", total_items=3)
        jobs = mgr.list_jobs()
        assert len(jobs) == 3
        ids = {j.job_id for j in jobs}
        assert ids == {"a", "b", "c"}

    def test_list_jobs_empty(self):
        mgr = self._make_manager()
        assert mgr.list_jobs() == []

    def test_can_resume_running(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        mgr.update_job("j1", completed=3)
        assert mgr.can_resume("j1") is True

    def test_can_resume_paused(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        # manually set paused status
        mgr._jobs["j1"].status = "paused"
        assert mgr.can_resume("j1") is True

    def test_can_resume_completed(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        mgr.complete_job("j1")
        assert mgr.can_resume("j1") is False

    def test_can_resume_failed(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        mgr.fail_job("j1", error="err")
        assert mgr.can_resume("j1") is False

    def test_can_resume_pending(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        assert mgr.can_resume("j1") is False

    def test_can_resume_unknown(self):
        mgr = self._make_manager()
        assert mgr.can_resume("nope") is False

    def test_serialize_deserialize_roundtrip(self):
        mgr = self._make_manager()
        original = mgr.create_job("j1", total_items=25)
        mgr.update_job("j1", completed=5, results=[{"x": 42}])
        original = mgr.get_job("j1")
        serialized = mgr.serialize_job(original)
        restored = mgr.deserialize_job(serialized)
        assert restored.job_id == original.job_id
        assert restored.status == original.status
        assert restored.completed_items == original.completed_items
        assert restored.results == original.results

    def test_serialize_is_valid_json(self):
        mgr = self._make_manager()
        job = mgr.create_job("j1", total_items=1)
        serialized = mgr.serialize_job(job)
        parsed = json.loads(serialized)
        assert parsed["job_id"] == "j1"

    # -- Edge cases --

    def test_update_unknown_job_raises(self):
        mgr = self._make_manager()
        with pytest.raises(KeyError):
            mgr.update_job("missing", completed=1)

    def test_complete_unknown_job_raises(self):
        mgr = self._make_manager()
        with pytest.raises(KeyError):
            mgr.complete_job("missing")

    def test_fail_unknown_job_raises(self):
        mgr = self._make_manager()
        with pytest.raises(KeyError):
            mgr.fail_job("missing", error="err")

    def test_create_duplicate_overwrites(self):
        mgr = self._make_manager()
        mgr.create_job("j1", total_items=10)
        mgr.update_job("j1", completed=5)
        job = mgr.create_job("j1", total_items=20)
        assert job.total_items == 20
        assert job.completed_items == 0
        assert job.status == "pending"
