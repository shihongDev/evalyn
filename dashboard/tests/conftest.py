"""Pytest configuration for dashboard tests."""

from __future__ import annotations

# pytest-asyncio is configured via per-test ``@pytest.mark.asyncio`` markers
# so no global ``asyncio_mode`` is required here. This file exists to keep
# the dashboard tests directory recognised as a package and to host any
# future shared fixtures.
