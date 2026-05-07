"""``/api/cli`` router: catalog and run-job endpoints (Lane B1.1, B1.2).

GET ``/api/cli`` returns the cached catalog produced at app startup by
:func:`evalyn_dashboard.introspect.build_catalog`.

POST ``/api/cli/run`` validates the submitted ``{cli_id, args}`` payload
against the matching :class:`CliSchema`, converts the dict to an argv
list, and asks the shared :class:`JobManager` to spawn ``["evalyn",
cli_id, ...argv]``. Returns ``{"job_id": <hex>}`` on success.

Validation rules (rejected with 400 on violation):
- unknown ``cli_id``
- unknown flag (``args`` key not in the schema)
- missing required flag
- ``kind=number``: value must be int or float (not bool, not str)
- ``kind=select``: value must appear in ``options``
- ``kind=multiselect``: value must be a list whose elements are all in
  ``options``
- ``kind=bool``: value must be a bool
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..introspect import CliSchema, ParamSchema, catalog_to_payload

router = APIRouter()


# ---------------------------------------------------------------------------
# Argument validation + argv conversion
# ---------------------------------------------------------------------------


def _coerce_number(value: Any) -> float | int | None:
    """Return ``value`` if it's already int/float (rejecting bool which is a
    subtype of int in Python). Returns None when invalid.
    """
    # ``bool`` is a subclass of ``int`` and would otherwise sneak through.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _validate_args(schema: CliSchema, args: dict[str, Any]) -> list[str]:
    """Raise :class:`HTTPException` 400 if ``args`` violate ``schema``.

    Returns the canonical argv tail (everything after ``["evalyn", cli_id]``)
    when validation succeeds.
    """
    by_name: dict[str, ParamSchema] = {p.name: p for p in schema.params}

    # Reject unknown flags first - cheap, deterministic, and keeps the rest
    # of validation simple.
    unknown = sorted(set(args) - set(by_name))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown args: {', '.join(unknown)}",
        )

    # Required-flag check uses both explicit ``required`` and the convention
    # that positional args have no option_strings (already folded into
    # ``required`` by ``introspect_parser``).
    missing = [
        p.name
        for p in schema.params
        if p.required and (p.name not in args or args[p.name] in (None, ""))
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"missing required args: {', '.join(missing)}",
        )

    argv: list[str] = []
    for name, value in args.items():
        param = by_name[name]
        # Skip explicit None / empty string -> behaves like omission. Keeps
        # the frontend honest: don't send a flag if you don't want it set.
        if value is None or value == "":
            continue
        argv.extend(_param_to_argv(param, value))
    return argv


def _param_to_argv(param: ParamSchema, value: Any) -> list[str]:
    """Convert a single (param, value) pair into argv tokens.

    Validation per ``kind`` happens here so callers don't need a second
    pass. Raises :class:`HTTPException` 400 on type/options mismatch.
    """
    flag = "--" + param.name.replace("_", "-")

    if param.kind == "bool":
        if not isinstance(value, bool):
            raise HTTPException(
                status_code=400,
                detail=f"{param.name}: expected bool, got {type(value).__name__}",
            )
        # store_true: emit the flag only when True. False/None means absent.
        return [flag] if value else []

    if param.kind == "number":
        coerced = _coerce_number(value)
        if coerced is None:
            raise HTTPException(
                status_code=400,
                detail=f"{param.name}: expected number, got {type(value).__name__}",
            )
        return [flag, str(coerced)]

    if param.kind == "select":
        if param.options and value not in param.options:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{param.name}: {value!r} not in {param.options}"
                ),
            )
        return [flag, str(value)]

    if param.kind == "multiselect":
        if not isinstance(value, list):
            raise HTTPException(
                status_code=400,
                detail=f"{param.name}: expected list, got {type(value).__name__}",
            )
        if param.options:
            bad = [v for v in value if v not in param.options]
            if bad:
                raise HTTPException(
                    status_code=400,
                    detail=f"{param.name}: invalid options {bad}, allowed {param.options}",
                )
        if not value:
            # Empty list -> omit. argparse would treat ``--tags`` with no
            # values as nargs="*" + empty, which is the same as absent.
            return []
        # argparse ``nargs="*"`` accepts ``--tags x y z`` (space-separated).
        return [flag, *[str(v) for v in value]]

    # string / path / long-text all serialise the same way: --flag <value>.
    return [flag, str(value)]


def args_to_argv(schema: CliSchema, args: dict[str, Any]) -> list[str]:
    """Public wrapper used by tests and the agent runtime.

    Validates ``args`` against ``schema`` and returns the argv tail. Does
    not include ``["evalyn", cli_id]``; the caller prepends those.
    """
    return _validate_args(schema, args)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("")
async def get_catalog(request: Request) -> JSONResponse:
    """Return the full CLI catalog cached on app state at startup."""
    catalog: list[CliSchema] = request.app.state.cli_catalog
    return JSONResponse(catalog_to_payload(catalog))


@router.post("/run")
async def run_cli(request: Request) -> JSONResponse:
    """Validate args and spawn ``evalyn <cli_id> ...``."""
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001 - any json parse failure -> 400
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")

    cli_id = body.get("cli_id")
    args = body.get("args", {})
    if not isinstance(cli_id, str) or not cli_id:
        raise HTTPException(status_code=400, detail="cli_id must be a non-empty string")
    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="args must be a json object")

    catalog: list[CliSchema] = request.app.state.cli_catalog
    schema = next((s for s in catalog if s.id == cli_id), None)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"unknown cli_id: {cli_id}")

    argv_tail = args_to_argv(schema, args)
    cmd = ["evalyn", cli_id, *argv_tail]

    job_manager = request.app.state.job_manager
    job_id = await job_manager.spawn(cmd, cli_id=cli_id, args=args)
    return JSONResponse({"job_id": job_id})


# ---------------------------------------------------------------------------
# Per-command run history (sourced from the JobPersistence sqlite mirror)
# ---------------------------------------------------------------------------


def _persistence_status_to_run_status(status: str, exit_code: int | None) -> str:
    """Map JobPersistence status to the v2 ``CommandHistory`` enum."""
    if status == "running":
        return "running"
    if status == "cancelled":
        return "cancelled"
    if status == "complete" and (exit_code is None or exit_code == 0):
        return "pass"
    return "fail"


def _summarise_args(cli_id: str, args: dict, duration_s: float | None) -> str:
    """One-line summary like ``gold-set v12 - 410 items - 3m41s``.

    The dataset / N items / duration tuple is the most useful triple for
    a re-run UI; missing pieces are silently dropped. Falls back to the
    cli_id when no args are interesting.
    """
    parts: list[str] = []
    ds = args.get("dataset") or args.get("dataset_id") or args.get("dataset_name")
    if isinstance(ds, str) and ds:
        parts.append(ds)
    n = args.get("n_items") or args.get("limit") or args.get("count")
    if isinstance(n, (int, float)):
        parts.append(f"{int(n)} items")
    if isinstance(duration_s, (int, float)) and duration_s > 0:
        secs = int(duration_s)
        if secs >= 60:
            parts.append(f"{secs // 60}m{secs % 60:02d}s")
        else:
            parts.append(f"{secs}s")
    if not parts:
        parts.append(cli_id)
    return " - ".join(parts)


@router.get("/history")
async def get_command_history(
    request: Request,
    command: str | None = None,
    limit: int = 10,
) -> JSONResponse:
    """Per-command run history sourced from the sqlite jobs mirror."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be 1..200")

    job_manager = request.app.state.job_manager
    persistence = getattr(job_manager, "_persistence", None)

    rows: list[dict] = []
    if persistence is not None:
        try:
            persisted = persistence.list_recent(limit=200)
        except Exception:  # noqa: BLE001
            persisted = []
        for row in persisted:
            cli = str(row.get("cli_id") or "")
            if command and cli != command:
                continue
            args = row.get("args") if isinstance(row.get("args"), dict) else {}
            duration_s = row.get("duration_s")
            duration_ms = (
                int(round(float(duration_s) * 1000.0))
                if isinstance(duration_s, (int, float))
                else 0
            )
            status = _persistence_status_to_run_status(
                str(row.get("status") or ""), row.get("exit_code")
            )
            rows.append(
                {
                    "job_id": str(row.get("job_id") or ""),
                    "status": status,
                    "started_at": str(row.get("started_at_iso") or ""),
                    "duration_ms": duration_ms,
                    "exit_code": int(row.get("exit_code") or 0),
                    "summary": _summarise_args(cli, args, duration_s),
                    "args": {k: str(v) for k, v in args.items()},
                }
            )
            if len(rows) >= limit:
                break

    # Compute "used_count_this_week" by walking all persisted rows for the
    # given command and counting those started within the last 7 days.
    from datetime import datetime, timedelta, timezone

    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    used = 0
    if persistence is not None:
        try:
            all_rows = persistence.list_recent(limit=500)
        except Exception:  # noqa: BLE001
            all_rows = []
        for row in all_rows:
            cli = str(row.get("cli_id") or "")
            if command and cli != command:
                continue
            iso = row.get("started_at_iso")
            if not isinstance(iso, str):
                continue
            try:
                dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt >= week_ago:
                used += 1

    return JSONResponse(
        {
            "command": command or "",
            "runs": rows,
            "used_count_this_week": used,
        }
    )


__all__ = ["router", "args_to_argv"]
