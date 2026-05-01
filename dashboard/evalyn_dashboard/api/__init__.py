"""Phase 1 stub API routers.

Each module exposes ``router: fastapi.APIRouter`` with placeholder routes
that return 501 Not Implemented. Phase 2 lanes (B1, C1) replace these.
"""

from . import agent, cli, files, jobs, runs, settings

__all__ = ["agent", "cli", "files", "jobs", "runs", "settings"]
