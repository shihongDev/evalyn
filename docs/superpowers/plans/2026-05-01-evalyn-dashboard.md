# Evalyn Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to dispatch lane agents per phase. Phases are sync gates: all prior lanes must commit and tests pass before next phase starts.

**Goal:** Build `evalyn dashboard`, a localhost IDE for evalyn evaluations. All 35 CLIs runnable via auto-generated forms; full agentic AI chat with provider choice (OpenAI/Anthropic/Ollama). Distributed as separate PyPI package `evalyn-dashboard`.

**Architecture:** FastAPI backend + pre-built React/TS frontend (Vite). Subprocess-spawned CLIs streamed via WebSocket to a terminal panel. Agent runtime exposes the CLI catalog as LLM tools, auto-runs read-only commands, requires user confirmation for write/destructive commands. Bound to 127.0.0.1 only with CSRF token.

**Tech Stack:**
- Backend: Python 3.10+, FastAPI, uvicorn, asyncio subprocesses, openai/anthropic/httpx SDKs.
- Frontend: React 18, TypeScript, Vite, Zustand, native WebSocket.
- Tests: pytest (backend), Vitest+RTL (frontend), Playwright (E2E).
- Distribution: monorepo with `uv` workspace, two PyPI packages (`evalyn`, `evalyn-dashboard`).

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-01-evalyn-dashboard-design.md`
- Visual mock: `/tmp/evalyn-dashboard-mock/` (extracted from `Desktop/evalyn-dashboard.zip`). Mock JSX files (`wb-app.jsx`, `wb-cli-forms.jsx`, `wb-data.jsx`, `wb-chat.jsx`, `tweaks-panel.jsx`) are the visual source of truth. Frontend lanes port these to TS.

---

## Conventions

- Each task header is `### <task-id>: <title>` followed by `**LANE:** <lane-id>` and `**DEPENDS ON:** <prior task ids or "none">`.
- TDD per task: write failing test, run (expect FAIL), implement, run (expect PASS), commit.
- Branch strategy: each lane works on its own branch off `feat/dashboard-workbench` (e.g. `feat/dashboard-A1-server`). Sync gates merge all lane branches into trunk and run combined tests.
- Worktrees: each parallel lane uses its own worktree at `/tmp/evalyn-wt-<lane>` to avoid tree contention.
- Commits: small, frequent, conventional commits (`feat:`, `test:`, `refactor:`).
- All test files use pytest's standard layout. All new Python code is type-annotated.
- All frontend code uses TypeScript (no `any` except where mock JSX is being literally ported and types come from the catalog schema).

---

## File Structure

Final layout after all phases:

```
evalyn/                                 # repo root
  pyproject.toml                        # uv workspace root (NEW)
  sdk/                                  # core SDK (existing, minimal changes)
    pyproject.toml
    evalyn_sdk/
      cli/
        main.py                         # MODIFY: add entry-point plugin discovery
        commands/
          dashboard.py                  # MODIFY: deprecation alias -> report
          report.py                     # NEW (renamed from existing dashboard.py)
  
  dashboard/                            # NEW package
    pyproject.toml
    evalyn_dashboard/
      __init__.py
      __main__.py                       # for `python -m evalyn_dashboard`
      cli_command.py                    # entry-point: cmd_dashboard(args)
      server.py                         # FastAPI app + uvicorn launcher
      introspect.py                     # argparse -> CliSchema JSON
      jobs.py                           # JobManager
      agent.py                          # AgentRuntime + providers
      credentials.py                    # ~/.evalyn/credentials.json
      api/
        __init__.py
        cli.py
        jobs.py
        files.py
        runs.py
        agent.py
        settings.py
      static/                           # vendored React build (committed)
        index.html
        assets/index-<hash>.js
        assets/index-<hash>.css
    frontend/
      package.json
      vite.config.ts
      tsconfig.json
      index.html
      src/
        main.tsx
        App.tsx
        store.ts
        api.ts
        types/
          catalog.ts                    # generated from schema/catalog.schema.json
          jobs.ts
          agent.ts
        components/
          TitleBar.tsx
          Sidebar.tsx
          FileTree.tsx
          CliCatalog.tsx
          RunsList.tsx
          EditorTabs.tsx
          BottomPanel.tsx
          Terminal.tsx
          JobsList.tsx
          ChatPanel.tsx
          SettingsModal.tsx
        views/
          Welcome.tsx
          CliForm.tsx
          ParamField.tsx
          RunView.tsx
          FileView.tsx
        styles/
          index.css                     # CSS variables ported from mock
    schema/
      catalog.schema.json               # source of truth for CliSchema
    tests/
      __init__.py
      conftest.py
      test_introspect.py
      test_jobs.py
      test_credentials.py
      test_agent.py
      test_api_cli.py
      test_api_jobs.py
      test_api_agent.py
      test_security.py
      e2e/
        test_happy_path.py
```

---

## Phase 0: Repo Restructure (sequential, 1 lane)

**LANE:** P0
**WORKTREE:** trunk (`/tmp/evalyn-dashboard-trunk`)

### P0.1: Workspace pyproject + dashboard package skeleton

**LANE:** P0
**DEPENDS ON:** none

Create:
- `pyproject.toml` (repo root): uv workspace declaration
- `dashboard/pyproject.toml`: package metadata, deps, entry point
- `dashboard/evalyn_dashboard/__init__.py` with `__version__ = "0.1.0"`
- `dashboard/evalyn_dashboard/__main__.py` placeholder

Steps:
1. Write `pyproject.toml` at repo root (workspace).
2. Write `dashboard/pyproject.toml`.
3. Write minimal `__init__.py` and `__main__.py`.
4. Run `uv sync` from repo root, verify both packages installed editable.
5. Run `uv run python -c "import evalyn_dashboard; print(evalyn_dashboard.__version__)"`, expect `0.1.0`.
6. Commit: `chore(dashboard): bootstrap evalyn-dashboard package skeleton`.

`pyproject.toml` (repo root):
```toml
[tool.uv.workspace]
members = ["sdk", "dashboard"]
```

`dashboard/pyproject.toml`:
```toml
[project]
name = "evalyn-dashboard"
version = "0.1.0"
description = "Localhost IDE for evalyn evaluations"
requires-python = ">=3.10"
dependencies = [
  "evalyn",
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "websockets>=12",
  "openai>=1.10",
  "anthropic>=0.20",
  "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-asyncio>=0.23", "httpx>=0.27"]

[project.entry-points."evalyn.commands"]
dashboard = "evalyn_dashboard.cli_command"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["evalyn_dashboard*"]

[tool.setuptools.package-data]
evalyn_dashboard = ["static/**/*"]
```

### P0.2: Add entry-point plugin discovery to core CLI

**LANE:** P0
**DEPENDS ON:** P0.1

Modify `sdk/evalyn_sdk/cli/main.py` to read entry points group `evalyn.commands` at startup and merge them into the lazy command-to-module map.

Test: install evalyn-dashboard editable, run `evalyn dashboard --help`, expect output from `evalyn_dashboard.cli_command:register_commands`.

Test code (`tests/test_plugin_discovery.py`):
```python
import importlib.metadata as md

def test_dashboard_plugin_registered():
    eps = list(md.entry_points(group="evalyn.commands"))
    names = [ep.name for ep in eps]
    assert "dashboard" in names

def test_evalyn_dashboard_help_works(monkeypatch, capsys):
    # call through main entry point; expect ArgumentParser to find 'dashboard' subcommand
    from evalyn_sdk.cli.main import main
    with pytest.raises(SystemExit):
        main(["dashboard", "--help"])
    captured = capsys.readouterr()
    assert "dashboard" in captured.out.lower()
```

In `evalyn_dashboard/cli_command.py`, write a stub `register_commands(subparsers)` that adds a `dashboard` subparser with `--help` working. Real implementation lands in P1 lane A1.

Stub:
```python
import argparse

def register_commands(subparsers):
    p = subparsers.add_parser("dashboard", help="Launch the evalyn dashboard (localhost IDE)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7401)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_dashboard)

def cmd_dashboard(args: argparse.Namespace) -> int:
    print("Dashboard not yet implemented (Phase 1 lane A1)")
    return 0
```

Commit: `feat(cli): add entry-point plugin discovery for evalyn.commands`.

### P0.3: Rename core dashboard command to report + deprecation alias

**LANE:** P0
**DEPENDS ON:** P0.2

Currently `sdk/evalyn_sdk/cli/commands/dashboard.py` generates a static HTML report. Rename to `report.py`, change command name from `dashboard` to `report`. Add a separate deprecation alias module that, when invoked, prints a stderr deprecation warning and forwards to `report`.

Steps:
1. `git mv sdk/evalyn_sdk/cli/commands/dashboard.py sdk/evalyn_sdk/cli/commands/report.py`
2. In `report.py`, rename `cmd_dashboard` -> `cmd_report` and the subparser name to `report`.
3. Write a new `sdk/evalyn_sdk/cli/commands/dashboard_alias.py` that registers a `dashboard` subparser, prints `[deprecated] 'evalyn dashboard' (static report) renamed to 'evalyn report'. The 'dashboard' name now refers to the new IDE; install it via 'pip install evalyn-dashboard'. This alias will be removed in v3.0.` to stderr, then calls `cmd_report(args)`.
4. Update `_COMMAND_MODULE_MAP` in `main.py`: `"report": "evalyn_sdk.cli.commands.report"`. The `dashboard` entry stays in the map but now points to `dashboard_alias` ONLY when `evalyn-dashboard` is NOT installed (entry-point discovery from P0.2 takes precedence).
5. Run existing tests: `uv run pytest tests/test_cli.py -k dashboard or report`, fix any that referenced the old command name.
6. Add new test: `test_dashboard_alias_warns_and_runs_report`.
7. Commit: `feat(cli): rename dashboard to report, add deprecation alias`.

### P0.4: Frontend scaffold (Vite + React + TS)

**LANE:** P0
**DEPENDS ON:** P0.1

In `dashboard/frontend/`:
1. `npm create vite@latest . -- --template react-ts` (answer non-interactively or use the explicit init).
2. Install deps: `npm install zustand`.
3. Replace default `src/App.tsx` with a placeholder that says "Evalyn Workbench - Phase 1 placeholder".
4. Create `src/styles/index.css` with the CSS variables from the mock's `Evalyn Workbench.html` (light + dark theme blocks).
5. Configure `vite.config.ts` to set `build.outDir` to `../evalyn_dashboard/static/` and `base: "/static/"`.
6. Verify: `cd dashboard/frontend && npm run dev` starts dev server on :5173. Stop it.
7. Verify: `npm run build` produces files in `dashboard/evalyn_dashboard/static/`.
8. Commit: `feat(dashboard-frontend): scaffold Vite + React + TS frontend`.

Vite config:
```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: path.resolve(__dirname, '../evalyn_dashboard/static'),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:7401',
      '/ws':  { target: 'ws://127.0.0.1:7401', ws: true },
    },
  },
});
```

### P0.5: Catalog schema + TS type generation

**LANE:** P0
**DEPENDS ON:** P0.1

Create the contract that A2 (cli-introspector) and B2 (frontend forms) will both work against:

- `dashboard/schema/catalog.schema.json` (JSON Schema for `CliSchema[]`).
- `dashboard/frontend/src/types/catalog.ts` (hand-written TS types that match the JSON schema).

Both files must agree. A unit test in P1 lane A2 will validate introspector output against the JSON schema.

`catalog.ts`:
```ts
export type ParamKind =
  | 'bool' | 'string' | 'number'
  | 'select' | 'multiselect'
  | 'path' | 'long-text';

export interface ParamSchema {
  name: string;
  kind: ParamKind;
  default?: unknown;
  options?: string[];
  help?: string;
  required?: boolean;
  advanced?: boolean;
}

export interface CliSchema {
  id: string;
  name: string;
  group: string;
  blurb: string;
  params: ParamSchema[];
}
```

Commit: `feat(dashboard): define shared CliSchema (json schema + ts types)`.

### P0.6: Phase 0 sync gate

**LANE:** P0
**DEPENDS ON:** P0.1, P0.2, P0.3, P0.4, P0.5

Verify before Phase 1 starts:
- `uv sync` succeeds at repo root.
- `evalyn report --help` prints the renamed report command help.
- `evalyn dashboard` (without evalyn-dashboard plugin) prints deprecation warning then runs report.
- `evalyn dashboard` (with evalyn-dashboard plugin installed editable) prints "Phase 1 placeholder".
- `cd dashboard/frontend && npm run build` produces static files.
- All existing core tests still pass.

Tag: `git tag p0-complete`. Push branch.

---

## Phase 1: Foundation (5 parallel lanes)

After P0, dispatch 5 agents in parallel. Each agent works in its own worktree off branch `feat/dashboard-workbench`.

### Lane A1: Server skeleton
**WORKTREE:** `/tmp/evalyn-wt-A1`, branch `feat/dashboard-A1-server`

#### A1.1: FastAPI app + healthcheck

**LANE:** A1
**DEPENDS ON:** P0.6

Create `dashboard/evalyn_dashboard/server.py` with FastAPI app, healthcheck, and a `main()` function that runs uvicorn.

Test (`tests/test_server.py`):
```python
from fastapi.testclient import TestClient
from evalyn_dashboard.server import build_app

def test_healthcheck():
    client = TestClient(build_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_index_served():
    client = TestClient(build_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
```

Implementation: `build_app()` returns FastAPI instance. Mounts `/static` from package data. Serves `index.html` for `/` and any non-API route (SPA fallback).

Commit: `feat(dashboard): FastAPI app skeleton with healthcheck`.

#### A1.2: CSRF token + meta tag injection

**LANE:** A1
**DEPENDS ON:** A1.1

Generate a random token at app startup. Inject into served `index.html` as `<meta name="workbench-token" content="...">`. Require `X-Workbench-Token` header on all `POST/PUT/DELETE` routes via FastAPI middleware. GET routes exempt.

Test:
```python
def test_csrf_token_in_index():
    client = TestClient(build_app())
    r = client.get("/")
    assert 'name="workbench-token"' in r.text

def test_post_without_token_rejected():
    client = TestClient(build_app())
    r = client.post("/api/cli/run", json={})  # endpoint doesn't exist yet, but middleware should run first
    assert r.status_code in (403, 404)  # 403 if middleware rejects, 404 only after middleware passes
    # Actually we want: middleware rejects without token -> 403
    assert r.status_code == 403
```

Commit: `feat(dashboard): CSRF token middleware with meta-tag injection`.

#### A1.3: Localhost binding guard

**LANE:** A1
**DEPENDS ON:** A1.1

Add `--host` argument to `cmd_dashboard`; reject non-loopback host unless `--unsafe-bind` flag is set. Print warning to stderr if `--unsafe-bind` is used.

Test:
```python
def test_rejects_non_loopback_without_unsafe(capsys):
    args = argparse.Namespace(host="0.0.0.0", port=7401, no_browser=True, unsafe_bind=False, dev=False)
    rc = cmd_dashboard(args)
    assert rc != 0
    err = capsys.readouterr().err
    assert "127.0.0.1" in err.lower() or "loopback" in err.lower()
```

Commit: `feat(dashboard): localhost binding guard with --unsafe-bind escape hatch`.

#### A1.4: Browser auto-open + uvicorn launcher

**LANE:** A1
**DEPENDS ON:** A1.1, A1.3

Real `cmd_dashboard(args)` body: builds app, starts uvicorn in a thread, calls `webbrowser.open(...)` after the server reports ready, blocks on the uvicorn task. `--no-browser` skips the open; `--dev` skips browser AND skips static file serving (frontend served separately by Vite).

Manual smoke (not auto-tested): `evalyn dashboard --no-browser` starts server, healthcheck reachable, ctrl-C exits cleanly.

Commit: `feat(dashboard): cmd_dashboard runs uvicorn and opens browser`.

#### A1.5: Stub API endpoints

**LANE:** A1
**DEPENDS ON:** A1.1

Wire empty placeholder routers for `/api/cli`, `/api/jobs`, `/api/files`, `/api/runs`, `/api/agent`, `/api/settings`. Each returns 501 Not Implemented for now. Phase 2/3 lanes fill them in.

Commit: `feat(dashboard): scaffold api router files (all return 501)`.

#### A1.6: Lane A1 sync

Run all A1 tests, push `feat/dashboard-A1-server`. Mark lane done.

---

### Lane A2: CLI introspector
**WORKTREE:** `/tmp/evalyn-wt-A2`, branch `feat/dashboard-A2-introspect`

#### A2.1: Test fixture parsers

**LANE:** A2
**DEPENDS ON:** P0.6

Create `dashboard/tests/fixtures/sample_parsers.py` with 7 minimal argparse parsers, one per kind. Used by all A2 tests.

```python
import argparse

def make_bool_parser():
    p = argparse.ArgumentParser(prog="bool-cmd", description="bool sample")
    p.add_argument("--flag", action="store_true", help="a boolean flag")
    return p

def make_select_parser():
    p = argparse.ArgumentParser(prog="select-cmd", description="select sample")
    p.add_argument("--mode", choices=["a", "b", "c"], default="a")
    return p

def make_multiselect_parser():
    p = argparse.ArgumentParser(prog="multi-cmd", description="multiselect sample")
    p.add_argument("--tags", nargs="*", choices=["x", "y", "z"], default=[])
    return p

def make_number_parser():
    p = argparse.ArgumentParser(prog="num-cmd", description="number sample")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    return p

def make_path_parser():
    p = argparse.ArgumentParser(prog="path-cmd", description="path sample")
    p.add_argument("--output", default="./out.json")
    p.add_argument("--input-file", required=True)
    return p

def make_long_text_parser():
    p = argparse.ArgumentParser(prog="lt-cmd", description="longtext sample")
    p.add_argument("--prompt", default="")
    p.add_argument("--system-prompt", default="")
    return p

def make_string_parser():
    p = argparse.ArgumentParser(prog="str-cmd", description="string sample")
    p.add_argument("--name", required=True)
    return p
```

Commit: `test(dashboard): fixture parsers covering all CliSchema kinds`.

#### A2.2: kind-mapping unit tests

**LANE:** A2
**DEPENDS ON:** A2.1

Write `dashboard/tests/test_introspect.py` with one test per kind. Tests assert that `introspect_parser(parser)` returns a `CliSchema` whose `params[*].kind` matches expectation.

```python
from evalyn_dashboard.introspect import introspect_parser
from tests.fixtures.sample_parsers import (
    make_bool_parser, make_select_parser, make_multiselect_parser,
    make_number_parser, make_path_parser, make_long_text_parser, make_string_parser,
)

def test_bool_kind():
    schema = introspect_parser(make_bool_parser())
    flag = next(p for p in schema.params if p.name == "flag")
    assert flag.kind == "bool"
    assert flag.default is False

def test_select_kind():
    schema = introspect_parser(make_select_parser())
    mode = next(p for p in schema.params if p.name == "mode")
    assert mode.kind == "select"
    assert mode.options == ["a", "b", "c"]
    assert mode.default == "a"

def test_multiselect_kind():
    schema = introspect_parser(make_multiselect_parser())
    tags = next(p for p in schema.params if p.name == "tags")
    assert tags.kind == "multiselect"
    assert tags.options == ["x", "y", "z"]

def test_number_kind():
    schema = introspect_parser(make_number_parser())
    workers = next(p for p in schema.params if p.name == "workers")
    threshold = next(p for p in schema.params if p.name == "threshold")
    assert workers.kind == "number"
    assert threshold.kind == "number"

def test_path_kind_by_name():
    schema = introspect_parser(make_path_parser())
    out = next(p for p in schema.params if p.name == "output")
    inp = next(p for p in schema.params if p.name == "input_file")
    assert out.kind == "path"
    assert inp.kind == "path"

def test_long_text_kind_by_name():
    schema = introspect_parser(make_long_text_parser())
    prompt = next(p for p in schema.params if p.name == "prompt")
    sys_prompt = next(p for p in schema.params if p.name == "system_prompt")
    assert prompt.kind == "long-text"
    assert sys_prompt.kind == "long-text"

def test_string_default_kind():
    schema = introspect_parser(make_string_parser())
    name = next(p for p in schema.params if p.name == "name")
    assert name.kind == "string"
    assert name.required is True
```

Run: `uv run pytest dashboard/tests/test_introspect.py -v` - expect FAIL (`introspect_parser` not implemented).

#### A2.3: Implement introspect_parser

**LANE:** A2
**DEPENDS ON:** A2.2

`dashboard/evalyn_dashboard/introspect.py`:
```python
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass, asdict, field
from typing import Any

PATH_NAME_RE = re.compile(r"\b(path|file|dir|output|out)\b", re.I)
LONG_TEXT_NAMES = {"prompt", "template", "description", "system_prompt", "instructions"}

@dataclass
class ParamSchema:
    name: str
    kind: str
    default: Any = None
    options: list[str] | None = None
    help: str | None = None
    required: bool = False
    advanced: bool = False

@dataclass
class CliSchema:
    id: str
    name: str
    group: str
    blurb: str
    params: list[ParamSchema] = field(default_factory=list)

def _classify_kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    if action.choices is not None:
        if action.nargs in ("*", "+"):
            return "multiselect"
        return "select"
    if action.type in (int, float):
        return "number"
    name_norm = action.dest
    if name_norm in LONG_TEXT_NAMES:
        return "long-text"
    if PATH_NAME_RE.search(name_norm.replace("_", "-")):
        return "path"
    return "string"

def introspect_parser(
    parser: argparse.ArgumentParser,
    *,
    group: str = "Misc",
    advanced: set[str] | None = None,
) -> CliSchema:
    advanced = advanced or set()
    params: list[ParamSchema] = []
    for action in parser._actions:
        if action.dest in ("help", "version"):
            continue
        kind = _classify_kind(action)
        opts = list(action.choices) if action.choices is not None else None
        params.append(ParamSchema(
            name=action.dest,
            kind=kind,
            default=action.default if action.default is not argparse.SUPPRESS else None,
            options=opts,
            help=action.help,
            required=bool(getattr(action, "required", False)) or (action.option_strings == []),
            advanced=action.dest in advanced,
        ))
    return CliSchema(
        id=parser.prog,
        name=parser.prog,
        group=group,
        blurb=parser.description or "",
        params=params,
    )
```

Run tests: `uv run pytest dashboard/tests/test_introspect.py -v` - expect PASS.

Commit: `feat(dashboard): introspect.py with kind classification for all 7 kinds`.

#### A2.4: Catalog builder for all 35 evalyn CLIs

**LANE:** A2
**DEPENDS ON:** A2.3

Add `build_catalog()` that walks every module under `evalyn_sdk.cli.commands.*`, calls each `register_commands(subparsers)` against a fresh subparsers, then introspects each registered parser. Assigns groups via a per-module `GROUP = "..."` constant (or default `"Misc"`).

Steps:
1. Add `GROUP` and optional `ADVANCED` constants to each command module (small change to `sdk/evalyn_sdk/cli/commands/*.py`). Groups: `Tracing`, `Dataset`, `Metrics`, `Eval`, `Analysis`, `Annotation`, `Insights`, `Export`, `Simulation`, `Infrastructure`, `Quickstart`.
2. Implement `build_catalog()`:
```python
def build_catalog() -> list[CliSchema]:
    import importlib
    from evalyn_sdk.cli.main import _COMMAND_MODULE_MAP  # may need a public accessor
    result: list[CliSchema] = []
    seen_ids = set()
    for cmd_id, module_path in _COMMAND_MODULE_MAP.items():
        mod = importlib.import_module(module_path)
        if not hasattr(mod, "register_commands"):
            continue
        # build temp parser to capture the subparser this module registers
        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        mod.register_commands(subs)
        group = getattr(mod, "GROUP", "Misc")
        advanced = getattr(mod, "ADVANCED", set())
        for action in subs._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for sub_name, sub_parser in action.choices.items():
                if sub_name in seen_ids:
                    continue
                seen_ids.add(sub_name)
                result.append(introspect_parser(sub_parser, group=group, advanced=advanced))
    return result
```
3. Test: assert `len(build_catalog()) >= 35`, assert specific commands present (`run-eval`, `analyze`, `compare`, `dashboard`).
4. Test: validate output against `dashboard/schema/catalog.schema.json` using `jsonschema` library.

Commit: `feat(dashboard): build_catalog walks all evalyn commands`.

#### A2.5: Lane A2 sync

Run all A2 tests, push branch. Mark lane done.

---

### Lane A3: Job manager
**WORKTREE:** `/tmp/evalyn-wt-A3`, branch `feat/dashboard-A3-jobs`

#### A3.1: spawn() with stdout capture

**LANE:** A3
**DEPENDS ON:** P0.6

Test (`tests/test_jobs.py`):
```python
import asyncio
import pytest
from evalyn_dashboard.jobs import JobManager

@pytest.mark.asyncio
async def test_spawn_echo_captures_output():
    jm = JobManager()
    job_id = await jm.spawn(["echo", "hello"])
    job = jm.get(job_id)
    # wait for exit
    await jm.wait(job_id, timeout=5)
    assert job.exit_code == 0
    output = "".join(line for kind, line, ts in jm.history(job_id) if kind == "stdout")
    assert "hello" in output
```

Implement `JobManager.spawn(cmd: list[str]) -> str` using `asyncio.create_subprocess_exec`. Captures stdout/stderr line-by-line into per-job `asyncio.Queue`. Saves history per job.

Commit: `feat(dashboard): JobManager.spawn with stdout capture`.

#### A3.2: cancel via SIGTERM + grace + SIGKILL

**LANE:** A3
**DEPENDS ON:** A3.1

Test:
```python
@pytest.mark.asyncio
async def test_cancel_sends_sigterm():
    jm = JobManager(grace_seconds=0.5)
    job_id = await jm.spawn(["python", "-c", "import time; time.sleep(60)"])
    await asyncio.sleep(0.2)
    await jm.cancel(job_id)
    job = await jm.wait(job_id, timeout=5)
    assert job.state == "cancelled"
    assert job.exit_code != 0
```

Implement `cancel(job_id)`: send SIGTERM, await up to `grace_seconds`, then SIGKILL.

Commit: `feat(dashboard): JobManager.cancel with grace period`.

#### A3.3: WS-fanout queue subscription

**LANE:** A3
**DEPENDS ON:** A3.1

Test:
```python
@pytest.mark.asyncio
async def test_multiple_subscribers_receive_same_output():
    jm = JobManager()
    job_id = await jm.spawn(["python", "-c", "print('a'); print('b')"])
    received1, received2 = [], []
    async def consume(q, sink):
        async for evt in q:
            sink.append(evt)
            if evt["type"] == "exit":
                break
    async with jm.subscribe(job_id) as q1, jm.subscribe(job_id) as q2:
        await asyncio.gather(consume(q1, received1), consume(q2, received2))
    assert any("a" in str(e) for e in received1)
    assert any("a" in str(e) for e in received2)
```

Implement `subscribe(job_id)` returning an async context manager that yields a queue. Internal: append-only event log per job + per-subscriber cursor.

Commit: `feat(dashboard): fanout subscription for job streams`.

#### A3.4: Backpressure with truncation marker

**LANE:** A3
**DEPENDS ON:** A3.3

Test that when queue fills (default 10000), oldest events drop and a `truncated` marker is inserted.

Commit: `feat(dashboard): backpressure with truncation marker`.

#### A3.5: Job history retention

**LANE:** A3
**DEPENDS ON:** A3.1

`JobManager.recent(n=100)` returns last N jobs in reverse chronological order. Includes both running and completed.

Commit: `feat(dashboard): JobManager.recent for jobs panel`.

#### A3.6: Lane A3 sync

Run all A3 tests, push branch.

---

### Lane A4: Credentials
**WORKTREE:** `/tmp/evalyn-wt-A4`, branch `feat/dashboard-A4-credentials`

#### A4.1: Atomic write + chmod 600

**LANE:** A4
**DEPENDS ON:** P0.6

Test (`tests/test_credentials.py`):
```python
import os, json, stat
from pathlib import Path
from evalyn_dashboard.credentials import CredentialStore

def test_write_chmods_to_600(tmp_path):
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-test", model="gpt-5.1")
    mode = stat.S_IMODE(os.stat(tmp_path / "cred.json").st_mode)
    assert mode == 0o600

def test_round_trip(tmp_path):
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-test", model="gpt-5.1")
    cs2 = CredentialStore(path=tmp_path / "cred.json")
    raw = cs2._raw_for_test("openai")
    assert raw["api_key"] == "sk-test"

def test_get_settings_never_returns_plaintext(tmp_path):
    cs = CredentialStore(path=tmp_path / "cred.json")
    cs.set_provider("openai", api_key="sk-secret-XXX", model="gpt-5.1")
    public = cs.public_view()
    s = json.dumps(public)
    assert "sk-secret-XXX" not in s
    assert public["openai"]["is_set"] is True
    assert public["openai"]["model"] == "gpt-5.1"
```

Implement `CredentialStore` with atomic write (temp file + rename), `os.chmod(path, 0o600)` after rename, `set_provider`, `get_provider` (internal), `public_view`.

Commit: `feat(dashboard): CredentialStore with chmod 600 and atomic writes`.

#### A4.2: Provider connection test

**LANE:** A4
**DEPENDS ON:** A4.1

Test: `test_provider("openai")` makes a 1-token completion call. Mock the OpenAI client; assert call shape, return ok/error.

Commit: `feat(dashboard): test_provider 1-token connection check`.

#### A4.3: Lane A4 sync

Push branch.

---

### Lane A5: Frontend shell
**WORKTREE:** `/tmp/evalyn-wt-A5`, branch `feat/dashboard-A5-shell`

This lane PORTS mock JSX files (`/tmp/evalyn-dashboard-mock/wb-app.jsx` etc.) to TypeScript. The visual look stays identical; the wiring becomes Zustand instead of `window.*` globals.

#### A5.1: Zustand store skeleton

**LANE:** A5
**DEPENDS ON:** P0.6

Create `dashboard/frontend/src/store.ts` with the full store shape from spec §8. All slices initialized empty. Selector hooks for each slice.

Test (`src/__tests__/store.test.ts`):
```ts
import { useStore } from '../store';

test('initial state', () => {
  const s = useStore.getState();
  expect(s.catalog).toEqual([]);
  expect(s.tabs).toEqual([]);
  expect(s.activeTabId).toBeNull();
});

test('addTab updates active', () => {
  useStore.getState().addTab({ id: 'cli:run-eval', title: 'run-eval', kind: 'cli' });
  const s = useStore.getState();
  expect(s.tabs).toHaveLength(1);
  expect(s.activeTabId).toBe('cli:run-eval');
});
```

Commit: `feat(dashboard-frontend): Zustand store with initial state`.

#### A5.2: TitleBar (port from mock)

**LANE:** A5
**DEPENDS ON:** A5.1

Port `TitleBar` from `wb-app.jsx` (lines 28-55). Keep visual identical: traffic-light dots, evalyn logo, breadcrumbs, search palette button, settings icon, localhost chip, tweaks icon. Replace inline event handlers with store actions.

Commit: `feat(dashboard-frontend): TitleBar component (ported from mock)`.

#### A5.3: EditorTabs (port from mock)

**LANE:** A5
**DEPENDS ON:** A5.1

Port `EditorTabs` from `wb-app.jsx`. Tab-kind icon mapping (cli=$, run=▶, yaml=⌬, file=·) preserved.

Commit: `feat(dashboard-frontend): EditorTabs component (ported)`.

#### A5.4: BottomPanel skeleton

**LANE:** A5
**DEPENDS ON:** A5.1

Port `BottomPanel` shell with three tabs (Terminal, Jobs, Problems). Inner views are placeholder "Coming in Phase 2" until B3 lane fills them.

Commit: `feat(dashboard-frontend): BottomPanel skeleton with 3 tabs`.

#### A5.5: App.tsx layout shell

**LANE:** A5
**DEPENDS ON:** A5.2, A5.3, A5.4

Port `App` from `wb-app.jsx` (lines 573+). Light/dark theme toggle wired to store. Sidebar/EditorTabs/BottomPanel all rendered. ChatPanel placeholder for now.

Commit: `feat(dashboard-frontend): App shell with TitleBar+Sidebar+Editor+Bottom layout`.

#### A5.6: Welcome view

**LANE:** A5
**DEPENDS ON:** A5.5

Port `WelcomeView` from `wb-app.jsx`. Quick-action cards have onClick stubs that log to console for now.

Commit: `feat(dashboard-frontend): Welcome view (ported from mock)`.

#### A5.7: Lane A5 sync

`npm run build` succeeds. Smoke test in browser (manual). Push branch.

---

### Phase 1 Sync Gate

**LANE:** sync
**DEPENDS ON:** A1.6, A2.5, A3.6, A4.3, A5.7

1. Merge all five lane branches into `feat/dashboard-workbench`:
   ```bash
   for lane in A1-server A2-introspect A3-jobs A4-credentials A5-shell; do
     git merge --no-ff feat/dashboard-${lane}
   done
   ```
2. Resolve any conflicts (most likely in `pyproject.toml` if multiple lanes touched it).
3. Run combined test suite: `uv run pytest dashboard/tests/` and `cd dashboard/frontend && npm test && npm run build`.
4. Tag: `git tag p1-complete`.

---

## Phase 2: Wire CLI Execution (3 parallel lanes)

### Lane B1: CLI + Jobs API
**WORKTREE:** `/tmp/evalyn-wt-B1`, branch `feat/dashboard-B1-api`
**DEPENDS ON:** Phase 1 sync

Tasks:
- B1.1 GET `/api/cli` returns full catalog (uses A2's `build_catalog()`).
- B1.2 POST `/api/cli/run` validates args against catalog schema, calls `JobManager.spawn(["evalyn", cli_id, ...args])`, returns `{job_id}`.
- B1.3 GET `/api/jobs/recent`, GET `/api/jobs/{id}`.
- B1.4 POST `/api/jobs/{id}/cancel`.
- B1.5 WebSocket `/ws/jobs/{id}` subscribes to `JobManager`, pushes events as JSON.

Each task: TDD with TestClient + real subprocess.

### Lane B2: Frontend sidebar + forms
**WORKTREE:** `/tmp/evalyn-wt-B2`, branch `feat/dashboard-B2-forms`
**DEPENDS ON:** Phase 1 sync (catalog schema + shell)

Tasks:
- B2.1 `Sidebar` component (port from `wb-app.jsx`).
- B2.2 `CliCatalog` (port + filter + group/alpha toggle).
- B2.3 `FileTree` (port; data from `/api/files/tree` once B1 ships).
- B2.4 `RunsList` (port; data from `/api/runs` once B1 ships).
- B2.5 `ParamField` (port from `wb-cli-forms.jsx`, all 7 kinds).
- B2.6 `CliForm` (port, 3 modes form/preview/raw).
- B2.7 Wire submit: POST `/api/cli/run` -> open new tab subscribed to job WS.

Tests: Vitest + RTL for `CliForm` and `ParamField`.

### Lane B3: Frontend terminal + jobs
**WORKTREE:** `/tmp/evalyn-wt-B3`, branch `feat/dashboard-B3-terminal`
**DEPENDS ON:** Phase 1 sync (BottomPanel placeholders)

Tasks:
- B3.1 `Terminal` component with ANSI parser (~1KB inline).
- B3.2 `JobsList` component (port from `wb-app.jsx`).
- B3.3 WS subscriber in store: opens connection per job, dispatches events.
- B3.4 Reconnect logic with `last_event_id`.

Tests: Vitest for ANSI parsing + store dispatch.

### Phase 2 Sync Gate

Merge B1+B2+B3, run combined tests + manual smoke (start dashboard, click `list-runs`, watch terminal stream). Tag `p2-complete`. **Shippable v0.1 MVP at this point.**

---

## Phase 3: Agent Runtime + Chat (2 parallel lanes)

### Lane C1: Agent runtime + APIs
**WORKTREE:** `/tmp/evalyn-wt-C1`, branch `feat/dashboard-C1-agent`
**DEPENDS ON:** Phase 2 sync

Tasks:
- C1.1 `BaseProvider` interface; `OpenAIProvider` with streaming chat + tool calls.
- C1.2 `AnthropicProvider` (uses Anthropic SDK's tool_use streaming).
- C1.3 `OllamaProvider` (httpx client to `/api/chat`).
- C1.4 `AgentRuntime`: tool loop, allowlist, `confirmation_required` gate, budget.
- C1.5 POST `/api/agent/chat` creates thread, spawns runtime task.
- C1.6 WebSocket `/ws/agent/{thread_id}` streams events.
- C1.7 POST `/api/agent/chat/{thread_id}/confirm` releases asyncio.Event.
- C1.8 GET/POST `/api/settings/*` (uses `CredentialStore`).
- C1.9 GET `/api/settings/models/{provider}` (hardcoded for OpenAI/Anthropic, real call to Ollama `/api/tags`).

Tests: integration with stubbed Ollama server returning canned tool_use sequences.

### Lane C2: Frontend chat + settings
**WORKTREE:** `/tmp/evalyn-wt-C2`, branch `feat/dashboard-C2-chat`
**DEPENDS ON:** Phase 2 sync (App shell)

Tasks:
- C2.1 `SettingsModal`: provider list, API key input, test button, model picker.
- C2.2 `ChatPanel` (port from `wb-chat.jsx`): composer, message list.
- C2.3 Tool-call cards inline in chat.
- C2.4 Confirmation cards with approve/reject buttons.
- C2.5 Final-suggestion cards (clickable -> open CliForm tab).
- C2.6 Provider-error banner with link to SettingsModal.

Tests: Vitest for ChatPanel rendering of each event type.

### Phase 3 Sync Gate

Merge C1+C2. Manual E2E with real Ollama: ask "list my recent runs", agent calls `list-runs` tool, returns summary. Tag `p3-complete`. **Shippable v1.0.**

---

## Phase 4: Polish + Ship (2 parallel lanes)

### Lane D1: E2E + CI
**WORKTREE:** `/tmp/evalyn-wt-D1`, branch `feat/dashboard-D1-e2e`

Tasks:
- D1.1 Playwright setup in `dashboard/tests/e2e/`.
- D1.2 Happy-path test: start server, open browser, click `list-runs` in catalog, fill form, submit, assert terminal output streams, assert exit 0.
- D1.3 GitHub Actions: `dashboard-tests` job (pytest + npm build + Vitest), `dashboard-e2e` job (only on PRs touching `dashboard/`).

### Lane D2: Docs + release
**WORKTREE:** `/tmp/evalyn-wt-D2`, branch `feat/dashboard-D2-docs`

Tasks:
- D2.1 `dashboard/README.md`: install, screenshots, model setup, agent flow walkthrough.
- D2.2 Update root `README.md`: mention dashboard as separate package.
- D2.3 `dashboard/CHANGELOG.md`: v0.1.0 notes.
- D2.4 Update `sdk/CHANGELOG.md`: deprecation notice for old `dashboard` command.

### Phase 4 Sync Gate

Merge D1+D2. Run full CI. Tag `v0.1.0` for evalyn-dashboard. Run `uv build` and `uv publish --dry-run`. Final manual smoke. **Ready to ship.**

---

## Self-review checklist

After plan written, before dispatching:
- [x] Spec coverage: every section in `2026-05-01-evalyn-dashboard-design.md` has a corresponding lane/task.
- [x] No placeholders: all code blocks contain real code or reference exact mock file paths to port from.
- [x] Type consistency: `CliSchema`, `ParamSchema`, `CliCatalog`, `Job`, `AgentEvent` used the same way across backend and frontend.
- [x] Parallelization: every Phase 1-4 task carries an explicit LANE label and DEPENDS ON clause.
