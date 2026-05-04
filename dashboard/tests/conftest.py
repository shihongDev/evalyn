"""Pytest configuration for dashboard tests."""

from __future__ import annotations

import pytest

# pytest-asyncio is configured via per-test ``@pytest.mark.asyncio`` markers
# so no global ``asyncio_mode`` is required here. This file exists to keep
# the dashboard tests directory recognised as a package and to host any
# future shared fixtures.


@pytest.fixture(autouse=True)
def _isolate_jobs_persistence_db(tmp_path, monkeypatch):
    """Per-test sqlite mirror so parallel xdist workers don't collide.

    The dashboard's ``JobPersistence`` defaults to ``.evalyn/data/jobs.sqlite``
    under the cwd, which under xdist would mean every worker writes to the
    same file in the project root. This fixture redirects the default path
    to a unique path per test via ``EVALYN_DASHBOARD_JOBS_DB``.
    """
    db_path = tmp_path / "jobs.sqlite"
    monkeypatch.setenv("EVALYN_DASHBOARD_JOBS_DB", str(db_path))
    yield
