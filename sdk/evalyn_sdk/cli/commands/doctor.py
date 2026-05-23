"""Doctor command: environment + config diagnostics.

Like `brew doctor` or `npm doctor`, this command inspects the local
environment and prints a single tidy table summarizing what is healthy,
what is missing, and what may need attention before running evalyn.

Checks (in order):
1. Python version  (must satisfy pyproject.toml's `requires-python`)
2. evalyn version  (import evalyn_sdk and read `__version__`)
3. Required deps   (opentelemetry, sqlite3, yaml)
4. Optional deps   (openai, anthropic, google.generativeai, langchain_core,
                    langgraph, crewai, numpy, playwright)
5. API keys        (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,
                    GOOGLE_API_KEY, XAI_API_KEY) - presence only, NEVER
                    print the value
6. Config file     (DEFAULT_CONFIG_PATHS, parses as YAML if found)
7. Storage health  (.evalyn/ writable, traces DB readable if present)
8. Project shape   (cwd looks like an evalyn project)

Exit code is 1 if any check returns FAIL, else 0. Optional missing deps
and missing API keys produce WARN (not FAIL), so a fresh checkout passes.
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Quickstart"

# Doctor has no useful per-param knobs; expose nothing in the dashboard form.
ESSENTIAL: set[str] = set()
RANGES: dict[str, tuple[int, int, int]] = {}
UNITS: dict[str, str] = {}

import argparse
import importlib
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..constants import DEFAULT_CONFIG_PATHS

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

REQUIRED_DEPS: list[tuple[str, str]] = [
    # (import name, display name)
    ("opentelemetry", "opentelemetry"),
    ("sqlite3", "sqlite3 (stdlib)"),
    ("yaml", "PyYAML"),
]

OPTIONAL_DEPS: list[tuple[str, str]] = [
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("google.generativeai", "google-generativeai"),
    ("langchain_core", "langchain-core"),
    ("langgraph", "langgraph"),
    ("crewai", "crewai"),
    ("numpy", "numpy"),
    ("playwright", "playwright"),
]

API_KEY_VARS: list[str] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
]


@dataclass
class CheckResult:
    """One row in the doctor output table."""

    name: str
    status: str  # PASS / FAIL / WARN / SKIP
    detail: str = ""


def _parse_requires_python(spec: str | None) -> tuple[int, int] | None:
    """Parse a `requires-python` string like ">=3.10" into (3, 10).

    Returns None on parse failure. Only handles the simple `>=X.Y` form
    which is the only thing evalyn currently uses.
    """
    if not spec:
        return None
    spec = spec.strip()
    if not spec.startswith(">="):
        return None
    rest = spec[2:].strip()
    parts = rest.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    return (major, minor)


def _read_min_python_from_pyproject(start: Path | None = None) -> tuple[int, int]:
    """Walk up from `start` looking for a pyproject.toml with requires-python.

    Falls back to (3, 10) if nothing useful is found. `start` defaults to
    this file's directory so the check works no matter the cwd.
    """
    default = (3, 10)
    here = start if start is not None else Path(__file__).resolve().parent

    # tomllib is stdlib on 3.11+; fall back to tomli on 3.10
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ImportError:
            return default

    for candidate_dir in [here, *here.parents]:
        pp = candidate_dir / "pyproject.toml"
        if not pp.exists():
            # also check sdk/pyproject.toml at the repo root
            sdk_pp = candidate_dir / "sdk" / "pyproject.toml"
            if sdk_pp.exists():
                pp = sdk_pp
            else:
                continue
        try:
            with open(pp, "rb") as f:
                data = tomllib.load(f)
        except OSError:
            continue
        spec = data.get("project", {}).get("requires-python")
        parsed = _parse_requires_python(spec)
        if parsed:
            return parsed
    return default


def check_python_version() -> CheckResult:
    """Check that the running Python satisfies the project's minimum version."""
    required = _read_min_python_from_pyproject()
    current = sys.version_info[:2]
    current_str = f"{current[0]}.{current[1]}.{sys.version_info[2]}"
    required_str = f">={required[0]}.{required[1]}"
    if current >= required:
        return CheckResult("Python version", PASS, f"{current_str} ({required_str})")
    return CheckResult(
        "Python version",
        FAIL,
        f"{current_str} does not satisfy {required_str}",
    )


def check_evalyn_version() -> CheckResult:
    """Import evalyn_sdk and read __version__."""
    try:
        import evalyn_sdk  # noqa: PLC0415

        version = getattr(evalyn_sdk, "__version__", None)
        if not version:
            return CheckResult("evalyn version", FAIL, "no __version__ attribute")
        return CheckResult("evalyn version", PASS, str(version))
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("evalyn version", FAIL, f"import failed: {exc}")


def _try_import(modname: str) -> tuple[bool, str | None]:
    """Try importing `modname`. Returns (ok, version_or_None)."""
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return False, None
    # Try common version attrs; fall back to package metadata.
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return True, v
        if isinstance(v, tuple):
            return True, ".".join(str(x) for x in v)
    # Stdlib modules (sqlite3) don't have __version__; that's fine.
    return True, None


def check_required_deps() -> list[CheckResult]:
    """Check each required dependency. Missing = FAIL."""
    results: list[CheckResult] = []
    for modname, display in REQUIRED_DEPS:
        ok, version = _try_import(modname)
        if ok:
            detail = f"installed ({version})" if version else "installed"
            results.append(CheckResult(f"Required: {display}", PASS, detail))
        else:
            results.append(CheckResult(f"Required: {display}", FAIL, "not installed"))
    return results


def check_optional_deps() -> list[CheckResult]:
    """Check each optional dependency. Missing = WARN (not FAIL)."""
    results: list[CheckResult] = []
    for modname, display in OPTIONAL_DEPS:
        ok, version = _try_import(modname)
        if ok:
            detail = f"installed ({version})" if version else "installed"
            results.append(CheckResult(f"Optional: {display}", PASS, detail))
        else:
            results.append(CheckResult(f"Optional: {display}", WARN, "not installed"))
    return results


def check_api_keys(env: dict[str, str] | None = None) -> list[CheckResult]:
    """Check API key env vars. Presence only - never print the value.

    Accepts an `env` mapping for testing; defaults to os.environ.
    """
    env = env if env is not None else dict(os.environ)
    results: list[CheckResult] = []
    for var in API_KEY_VARS:
        value = env.get(var)
        if value:
            # Never include the value itself. Just confirm presence.
            results.append(CheckResult(f"API key: {var}", PASS, "(set)"))
        else:
            results.append(CheckResult(f"API key: {var}", WARN, "(not set)"))
    return results


def check_config_file(search_root: Path | None = None) -> CheckResult:
    """Search DEFAULT_CONFIG_PATHS at `search_root` (default cwd).

    PASS if found and parses as YAML, FAIL if found but invalid, SKIP if not
    found at all.
    """
    root = search_root if search_root is not None else Path.cwd()
    found: Path | None = None
    for name in DEFAULT_CONFIG_PATHS:
        candidate = root / name
        if candidate.is_file():
            found = candidate
            break

    if found is None:
        return CheckResult("Config file", SKIP, "no config file in cwd")

    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        # We can't validate YAML without the parser; still flag the find.
        return CheckResult("Config file", WARN, f"found {found.name} (yaml not installed)")

    try:
        with open(found, encoding="utf-8") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as exc:
        return CheckResult("Config file", FAIL, f"{found.name}: {exc.__class__.__name__}")
    except OSError as exc:
        return CheckResult("Config file", FAIL, f"{found.name}: {exc}")
    return CheckResult("Config file", PASS, f"{found.name} parses OK")


def check_storage(search_root: Path | None = None) -> list[CheckResult]:
    """Check that .evalyn/ is writable and any trace DB is readable."""
    root = search_root if search_root is not None else Path.cwd()
    evalyn_dir = root / ".evalyn"
    results: list[CheckResult] = []

    # Writability: if .evalyn exists, try a temp file there; otherwise try the
    # parent (where we would have to mkdir on first run).
    target = evalyn_dir if evalyn_dir.exists() else root
    if not target.exists():
        results.append(CheckResult("Storage writable", FAIL, f"{target} does not exist"))
    else:
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".evalyn-doctor-", dir=str(target), delete=True
            ):
                pass
            label = ".evalyn/" if evalyn_dir.exists() else "cwd (no .evalyn/ yet)"
            results.append(CheckResult("Storage writable", PASS, label))
        except OSError as exc:
            results.append(CheckResult("Storage writable", FAIL, f"{target}: {exc}"))

    # Trace DB readability. evalyn uses .evalyn/traces/prod.db by convention
    # but users can override with EVALYN_DB; check both.
    db_path: Path | None = None
    env_db = os.environ.get("EVALYN_DB")
    if env_db:
        p = Path(env_db)
        if p.exists():
            db_path = p
    if db_path is None:
        default_db = evalyn_dir / "traces" / "prod.db"
        if default_db.exists():
            db_path = default_db

    if db_path is None:
        results.append(CheckResult("Trace DB", SKIP, "no trace database found"))
        return results

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        finally:
            conn.close()
        results.append(CheckResult("Trace DB", PASS, str(db_path)))
    except sqlite3.Error as exc:
        results.append(CheckResult("Trace DB", FAIL, f"{db_path}: {exc}"))
    return results


def check_project_shape(search_root: Path | None = None) -> CheckResult:
    """Heuristic: does cwd look like an evalyn project?

    PASS if we find evalyn.yaml, .evalyn/, or any data/*/dataset.jsonl.
    WARN otherwise - the user can still run most commands but probably
    has not finished setup.
    """
    root = search_root if search_root is not None else Path.cwd()
    signals: list[str] = []
    for name in ("evalyn.yaml", "evalyn.yml", ".evalyn.yaml", ".evalynrc"):
        if (root / name).exists():
            signals.append(name)
            break
    if (root / ".evalyn").is_dir():
        signals.append(".evalyn/")
    data_dir = root / "data"
    if data_dir.is_dir():
        # Cheap: only walk one level deep.
        for child in data_dir.iterdir():
            if child.is_dir() and (child / "dataset.jsonl").exists():
                signals.append(f"data/{child.name}/dataset.jsonl")
                break
    if signals:
        return CheckResult("Project shape", PASS, ", ".join(signals))
    return CheckResult(
        "Project shape",
        WARN,
        "cwd does not look like an evalyn project (no evalyn.yaml / .evalyn/ / data/)",
    )


def run_all_checks() -> list[CheckResult]:
    """Run every check in order and return the flat result list."""
    results: list[CheckResult] = []
    results.append(check_python_version())
    results.append(check_evalyn_version())
    results.extend(check_required_deps())
    results.extend(check_optional_deps())
    results.extend(check_api_keys())
    results.append(check_config_file())
    results.extend(check_storage())
    results.append(check_project_shape())
    return results


def _status_cell(status: str) -> str:
    """Return a colorized status cell using the shared color utilities."""
    from ..utils.colors import dim, green, red, yellow  # noqa: PLC0415

    if status == PASS:
        return green(PASS)
    if status == FAIL:
        return red(FAIL)
    if status == WARN:
        return yellow(WARN)
    if status == SKIP:
        return dim(SKIP)
    return status


def render_table(results: list[CheckResult]) -> str:
    """Render the results as a 3-column ASCII/box table."""
    from ..utils.rich import table  # noqa: PLC0415

    headers = ["Check", "Status", "Detail"]
    rows = [[r.name, _status_cell(r.status), r.detail] for r in results]
    return table(headers, rows, align=["left", "left", "left"], max_col_width=60)


def render_summary(results: list[CheckResult]) -> str:
    """One-line tallies of each status."""
    counts: dict[str, int] = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    parts = [f"{k}: {counts[k]}" for k in (PASS, FAIL, WARN, SKIP)]
    return "Summary - " + ", ".join(parts)


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run all diagnostic checks and print the result table."""
    from ..utils.rich import banner  # noqa: PLC0415

    print()
    print(banner("EVALYN DOCTOR"))
    print()

    results = run_all_checks()
    print(render_table(results))
    print()
    print(render_summary(results))

    if any(r.status == FAIL for r in results):
        sys.exit(1)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the doctor command."""
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Diagnose environment, dependencies, API keys, config, and storage",
    )
    doctor_parser.set_defaults(func=cmd_doctor)


__all__ = [
    "cmd_doctor",
    "register_commands",
    "run_all_checks",
    "render_table",
    "render_summary",
    "CheckResult",
    "PASS",
    "FAIL",
    "WARN",
    "SKIP",
    "check_python_version",
    "check_evalyn_version",
    "check_required_deps",
    "check_optional_deps",
    "check_api_keys",
    "check_config_file",
    "check_storage",
    "check_project_shape",
]
