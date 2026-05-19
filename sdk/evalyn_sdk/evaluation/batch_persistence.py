"""Batch job persistence: save batch job state to disk for recovery.

After a crash or restart, previously running batch jobs can be resumed
from the last saved checkpoint instead of starting over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BatchJobState:
    """Persisted state for a single batch evaluation job."""

    job_id: str
    status: str = "pending"  # pending / running / completed / failed / paused
    total_items: int = 0
    completed_items: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "results": list(self.results),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJobState:
        return cls(
            job_id=data["job_id"],
            status=data.get("status", "pending"),
            total_items=data.get("total_items", 0),
            completed_items=data.get("completed_items", 0),
            results=data.get("results", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            error=data.get("error", ""),
        )


@dataclass
class PersistenceConfig:
    """Configuration for batch job persistence."""

    save_dir: str = ".evalyn_batch"
    save_interval: int = 10  # save every N items
    auto_resume: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "save_dir": self.save_dir,
            "save_interval": self.save_interval,
            "auto_resume": self.auto_resume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistenceConfig:
        return cls(
            save_dir=data.get("save_dir", ".evalyn_batch"),
            save_interval=data.get("save_interval", 10),
            auto_resume=data.get("auto_resume", True),
        )


class BatchJobManager:
    """Manages batch job state for persistence and recovery.

    Usage:
        config = PersistenceConfig(save_dir="/tmp/batch_jobs")
        manager = BatchJobManager(config)
        job = manager.create_job("job-1", total_items=100)
        job = manager.update_job("job-1", completed=10, results=[...])
        if manager.can_resume("job-1"):
            job = manager.get_job("job-1")
    """

    def __init__(self, config: PersistenceConfig) -> None:
        self.config = config
        self._jobs: dict[str, BatchJobState] = {}

    def create_job(self, job_id: str, total_items: int) -> BatchJobState:
        """Create a new batch job."""
        now = _now_iso()
        job = BatchJobState(
            job_id=job_id,
            status="pending",
            total_items=total_items,
            created_at=now,
            updated_at=now,
        )
        self._jobs[job_id] = job
        return job

    def update_job(
        self,
        job_id: str,
        completed: int,
        results: list[dict[str, Any]] | None = None,
    ) -> BatchJobState:
        """Update job progress."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")
        job.completed_items = completed
        if results is not None:
            job.results = results
        job.status = "running"
        job.updated_at = _now_iso()
        return job

    def complete_job(self, job_id: str) -> BatchJobState:
        """Mark a job as completed."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")
        job.status = "completed"
        job.updated_at = _now_iso()
        return job

    def fail_job(self, job_id: str, error: str) -> BatchJobState:
        """Mark a job as failed with an error message."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")
        job.status = "failed"
        job.error = error
        job.updated_at = _now_iso()
        return job

    def get_job(self, job_id: str) -> BatchJobState | None:
        """Retrieve a job by ID, or None if not found."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BatchJobState]:
        """Return all tracked jobs."""
        return list(self._jobs.values())

    def can_resume(self, job_id: str) -> bool:
        """True if the job exists, is running or paused, and not completed."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        return job.status in ("running", "paused")

    def serialize_job(self, job: BatchJobState) -> str:
        """Serialize a job state to a JSON string."""
        return json.dumps(job.as_dict())

    def deserialize_job(self, data: str) -> BatchJobState:
        """Deserialize a JSON string into a BatchJobState."""
        return BatchJobState.from_dict(json.loads(data))
