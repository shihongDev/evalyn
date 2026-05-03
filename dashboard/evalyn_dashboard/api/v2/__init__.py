"""``/api/v2/*`` routers - the new dashboard surface.

Each module under this package is an independent FastAPI router; they are
wired up in ``server.py::_register_v2_routers``. The JSON shapes returned
by every endpoint are the contract documented in
``dashboard/frontend/src/v2/api/types.ts``. Tests assert against that
contract on both sides.
"""
