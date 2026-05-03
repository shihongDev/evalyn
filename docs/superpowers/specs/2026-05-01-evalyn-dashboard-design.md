# Evalyn Dashboard (`evalyn dashboard`) - Design

Date: 2026-05-01
Status: Draft, pending implementation
Related mock: `Desktop/evalyn-dashboard.zip` ("Evalyn Workbench")

## 1. Summary

`evalyn dashboard` launches a localhost IDE for evaluations. Users open it in a browser, run any of the 35 evalyn CLIs via auto-generated forms, watch streaming output, and ask an AI agent questions that translate to chained CLI calls. Distributed as a separate PyPI package `evalyn-dashboard` to keep core evalyn lightweight.

## 2. Goals and non-goals

### Goals (v1)
- All 35 evalyn CLIs runnable via the dashboard UI (auto-generated forms from argparse).
- Real-time streaming of subprocess stdout/stderr to a terminal panel.
- Real AI chat agent with provider choice (OpenAI / Anthropic / Ollama), API key entry in UI, and full agentic loop (plan, run, read, reason, propose).
- Visual fidelity to the mock (TitleBar, Sidebar, EditorTabs, BottomPanel, ChatPanel dock-right, light + dark themes).
- Zero user-side build step. Pre-built React bundle ships in the wheel.

### Non-goals (v1)
- Command palette (Cmd+K).
- Rich Run Summary view (metrics tables, sparkcharts, failure clusters in-app). Use existing `evalyn analyze` and `evalyn report` for those.
- Floating/bottom chat placement (dock-right only).
- Theme tweaks panel (theme toggle button only).
- Multi-user / authentication. Strict localhost-only.
- Persistence of dashboard state (tabs, jobs) across server restarts.
- Telemetry, auto-update, daemon mode.

## 3. Architecture

```
$ evalyn dashboard
        |
        v
[ click: opens browser at localhost:7401 ]
        |
        v
+-----------------------------------------+
| FastAPI server (Python, in-process)     |
|                                         |
|  /static/*       served pre-built JS    |
|  /api/cli        catalog (introspected) |
|  /api/cli/run    POST -> spawn job      |
|  /api/jobs/{id}  job status             |
|  /api/files/*    .evalyn/ file tree     |
|  /api/runs/*     run JSON               |
|  /api/agent/chat POST -> agent turn     |
|  /api/settings   GET/POST credentials   |
|  /ws/jobs/{id}   WebSocket: live stdout |
|  /ws/agent/{id}  WebSocket: agent stream|
+-----------------------------------------+
        |
        v
+-----------------------------------------+
| Subprocess: actual `evalyn <cmd>` CLI   |
+-----------------------------------------+
```

Key design choices:
- Backend reuses argparse parsers via introspection. No duplicate command definitions.
- CLIs run as subprocesses (not in-process imports): accurate stdout streaming, crash isolation, cancellation via SIGTERM, parity with terminal usage.
- Pre-built React/Vite bundle vendored at `dashboard/evalyn_dashboard/static/`. Maintainers run `npm run build` locally; bundle committed to git.
- Bound to `127.0.0.1` only. CSRF protection via per-server token in `<meta>` tag.

## 4. Repo structure

```
evalyn/                                 # monorepo root
  sdk/                                  # core SDK (existing)
    pyproject.toml                      # publishes `evalyn`
    evalyn_sdk/
      cli/
        main.py                         # adds entry-point plugin discovery
        commands/
          dashboard.py                  # NEW: deprecated alias -> report
          report.py                     # RENAMED from existing dashboard.py
      ...
  
  dashboard/                            # NEW package
    pyproject.toml                      # publishes `evalyn-dashboard`
    evalyn_dashboard/
      __init__.py
      server.py                         # FastAPI app, uvicorn launcher
      introspect.py                     # argparse -> form schema
      jobs.py                           # JobManager: spawn/stream/cancel
      agent.py                          # AgentRuntime + providers
      credentials.py                    # ~/.evalyn/credentials.json
      cli_command.py                    # `cmd_dashboard(args)` plugin entry
      api/
        cli.py                          # /api/cli (catalog + run)
        jobs.py                         # /api/jobs (status, list, cancel, ws)
        files.py                        # /api/files
        runs.py                         # /api/runs
        agent.py                        # /api/agent (chat, ws, confirm)
        settings.py                     # /api/settings (credentials, models)
      static/                           # vendored React build (committed)
        index.html
        assets/index-<hash>.js
        assets/index-<hash>.css
    frontend/                           # React/Vite source (npm)
      package.json
      vite.config.ts
      tsconfig.json
      index.html
      src/
        main.tsx
        App.tsx
        store.ts                        # Zustand
        api.ts                          # fetch + WS client
        types.ts                        # CliCatalog, Job, AgentEvent
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
          index.css                     # CSS vars + atoms ported from mock
    tests/                              # backend pytest
    schema/
      catalog.schema.json               # shared between A2 and B2 in phasing
  
  pyproject.toml                        # uv workspace root
  tests/                                # existing core tests
  docs/
```

`pyproject.toml` (workspace root):
```toml
[tool.uv.workspace]
members = ["sdk", "dashboard"]
```

`dashboard/pyproject.toml`:
```toml
[project]
name = "evalyn-dashboard"
dependencies = [
  "evalyn>=2.5,<3.0",
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "websockets>=12",
  "openai>=1.10",
  "anthropic>=0.20",
  "httpx>=0.27",
]

[project.entry-points."evalyn.commands"]
dashboard = "evalyn_dashboard.cli_command"

[tool.setuptools.package-data]
evalyn_dashboard = ["static/**/*"]
```

## 5. Naming and backward compatibility

- New IDE takes the `evalyn dashboard` name.
- Existing static HTML report (was `evalyn dashboard`) renamed to `evalyn report`.
- Core's `evalyn dashboard` becomes a deprecation alias: prints stderr warning, runs `evalyn report` for one minor version, then removed.
- Without `evalyn-dashboard` installed, `evalyn dashboard` (post-deprecation) prints `pip install evalyn-dashboard` hint.

## 6. Installation impact

### User flows

```bash
# Existing core user (library use, CI evals): unchanged
pip install evalyn

# New IDE user
pip install evalyn-dashboard       # auto-pulls evalyn
evalyn dashboard                   # opens http://localhost:7401

# Existing static-report user during deprecation period
evalyn dashboard --output r.html   # still works, prints deprecation warning
                                   # actually runs `evalyn report --output r.html`

# Existing static-report user post-deprecation
evalyn report --output r.html
```

### Wheel sizes
- Core evalyn wheel: unchanged. Heavy LLM SDK deps stay in core because `run-eval`, `calibrate`, `insights --deep` need them.
- evalyn-dashboard adds: FastAPI, uvicorn, websockets, React bundle. ~5-10MB on top of core.

## 7. Backend components

### `server.py`
- `def main(host="127.0.0.1", port=7401, no_browser=False, dev=False, agent_budget=None)`.
- Rejects non-loopback host without `--unsafe-bind` flag.
- Mounts `/static` from `evalyn_dashboard/static/`.
- Generates random CSRF token at startup, embeds in served `index.html` as `<meta name="workbench-token">`.
- Calls `webbrowser.open(...)` after uvicorn ready signal (skipped in `--dev`).

### `introspect.py`
- Walks each `register_commands(subparsers)` once at startup.
- Per `argparse.Action`, maps to schema:
  - `store_true`/`store_false` -> `kind: bool`.
  - `choices` non-None and `nargs in (None, "?")` -> `kind: select`.
  - `choices` non-None and `nargs in ("*", "+")` -> `kind: multiselect`.
  - `type=int|float` -> `kind: number`.
  - name matches `(path|file|dir|output|out)` -> `kind: path`.
  - name matches `(prompt|template|description|system_prompt|instructions)` -> `kind: long-text`.
  - Otherwise `kind: string`.
- `advanced` flag: per-module opt-in via `ADVANCED = {"seed", "workers", ...}` constant. Optional. Default false.
- `required`: positional args, or `required=True` on optional args.
- Output cached. Entry point: `def build_catalog() -> list[CliSchema]`.
- Catalog JSON schema (also generated as TS type and JSON schema in `dashboard/schema/`):
  ```ts
  type CliSchema = {
    id: string;            // e.g. "run-eval"
    name: string;
    group: string;         // assigned per module: Data | Eval | Judge | Iterate | Infra
    blurb: string;         // from parser.description
    params: ParamSchema[];
  };
  type ParamSchema = {
    name: string;          // snake_case
    kind: "bool" | "string" | "number" | "select" | "multiselect" | "path" | "long-text";
    default?: unknown;
    options?: string[];    // for select/multiselect
    help?: string;
    required?: boolean;
    advanced?: boolean;
  };
  ```

### `jobs.py`
- `JobManager.spawn(cmd: list[str]) -> job_id`: launches subprocess via `asyncio.create_subprocess_exec` (safe, no shell).
- Output captured line-by-line into per-job `asyncio.Queue(maxsize=10000)`.
- Multiple WS subscribers per job via fanout queue.
- States: `pending | running | complete | failed | cancelled`.
- Cancel: SIGTERM, 3s grace, then SIGKILL.
- Timeout: 60min default, configurable.
- History: last 100 jobs kept in memory (configurable). No persistence v1.
- Backpressure: oldest dropped + `[truncated N lines]` marker if subscriber lags.

### `agent.py`
- `BaseProvider` interface: `async def stream_chat(messages, tools)` returns AsyncIterator of events.
- Implementations: `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`.
- Tool definition: each CLI's introspected schema converted to provider-native tool format.
- Tool runtime flow:
  1. Model emits tool_call.
  2. Runtime checks read-only allowlist.
  3. If allowed, JobManager spawns subprocess, awaits exit, captures stdout (truncate at 50KB), returns as tool_result.
  4. If not allowed, push `confirmation_required` event, block on asyncio.Event with 5min timeout.
  5. On confirm, run as above. On reject/timeout, return `tool_result: "user did not confirm"`.
- Read-only allowlist (auto-run): `list-calls`, `list-runs`, `list-metrics`, `list-calibrations`, `show-call`, `show-trace`, `show-span`, `show-projects`, `analyze`, `compare`, `trend`, `annotation-stats`, `validate`, `status`, `workflow`, `cluster-failures`, `cluster-misalignments`, `insights`, `select-metrics`.
- Write/destructive (needs confirm): `run-eval`, `calibrate`, `delete-traces`, `build-dataset`, `annotate`, `import-annotations`, `simulate`, `one-click`, `export`, `export-for-annotation`, `init`, `quickstart`, `report`, `dashboard`, `suggest-metrics` (when in LLM modes).
- Per-turn tool budget: max 8 tool calls. Triggers `final` event with "tool budget exceeded" if hit.
- Per-turn cost budget: optional via `--agent-budget 5.00`. Estimated post-hoc from token usage.

### `credentials.py`
- File path: `~/.evalyn/credentials.json` (XDG-compliant alternative documented but not implemented v1).
- Schema:
  ```json
  {
    "openai": { "api_key": "sk-...", "model": "gpt-5.1", "added_at": "..." },
    "anthropic": { "api_key": "sk-ant-...", "model": "claude-sonnet-4-6", "added_at": "..." },
    "ollama": { "base_url": "http://localhost:11434", "model": "llama3:70b", "added_at": "..." },
    "active": "anthropic"
  }
  ```
- Atomic write: temp file + rename. `os.chmod(path, 0o600)` immediately after write.
- API never returns plaintext key. Returns `{provider: {is_set: bool, model, added_at}, active}`.
- `test_provider(name)`: makes a 1-token completion call, returns ok/error.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/cli` | Catalog (all 35 CLIs) |
| POST | `/api/cli/run` | Spawn job from `{cli_id, args}` |
| GET | `/api/jobs/recent` | Last N completed jobs |
| GET | `/api/jobs/{id}` | One job's metadata |
| POST | `/api/jobs/{id}/cancel` | Cancel running job |
| WS | `/ws/jobs/{id}` | Stream stdout/stderr/exit events |
| GET | `/api/files/tree` | `.evalyn/` file tree |
| GET | `/api/files/read` | Read one file (param: `path`) |
| GET | `/api/runs` | Run metadata list |
| GET | `/api/runs/{id}` | One run's `results.json` |
| POST | `/api/agent/chat` | New turn, returns `{thread_id}` |
| POST | `/api/agent/chat/{thread_id}/confirm` | Confirmation response `{approve: bool}` |
| WS | `/ws/agent/{thread_id}` | Stream agent events |
| GET | `/api/settings` | Provider state (no plaintext keys) |
| POST | `/api/settings/{provider}` | Set API key + model |
| POST | `/api/settings/test/{provider}` | Test connection |
| GET | `/api/settings/models/{provider}` | List available models |
| POST | `/api/settings/active` | Set active provider |

All mutating routes require `X-Workbench-Token` header.

## 8. Frontend components

### Stack
- React 18 + Vite + TypeScript.
- Zustand for state.
- No CSS framework. CSS variables + inline styles, ported from mock.
- Native WebSocket API.

### Component contracts

**App.tsx**: shell layout. Reads tweaks from store. Renders TitleBar / Sidebar / EditorTabs / content / BottomPanel / ChatPanel.

**Sidebar.tsx**: 3 tabs (Files / CLIs / Runs). Collapsible to icon rail.

**CliCatalog.tsx**: groups + alpha + filter. Reads `store.catalog`. Each row clicks open a tab `cli:<id>`.

**CliForm.tsx**: 3 modes (form / preview / raw, default `preview`).
- Mirrors mock's `buildCli(cli, values)` logic: skip empty, skip default-equal, kebab-case flag names, quote values with spaces.
- Required client-side validation. Server re-validates.
- "Run" POSTs `/api/cli/run`, opens new tab `job:<job_id>`, subscribes to `/ws/jobs/{job_id}`.

**ParamField.tsx**: dispatches on `kind`. bool = two-button toggle. select = native `<select>`. multiselect = chip toggles. number = `<input type=number>`. path = string with placeholder. long-text = `<textarea>`. string = `<input>`.

**Terminal.tsx**: ANSI color parser (~1KB inline). Auto-scroll to bottom unless user scrolled up.

**JobsList.tsx**: live state from store. Click row to open job tab. Cancel button per running row.

**ChatPanel.tsx**: dock-right only. Composer at bottom, scrollable history. Renders text bubbles, inline tool_call cards (with status), confirmation cards (approve/reject buttons), final-suggestion cards (clickable -> open CliForm tab).

**SettingsModal.tsx**: provider list with API key input (password type), test button, model dropdown (populated from `/api/settings/models/{provider}`). Active-provider radio.

**RunView.tsx**: header (pass rate, cost, dataset, items, judge, duration) + collapsible JSON tree viewer for `results.json` + "Open analyze report" button (opens CliForm tab pre-filled with `--run <id>`).

### Store shape (Zustand)

```ts
{
  catalog: CliSchema[];                 // loaded once at boot
  tabs: Tab[];
  activeTabId: string | null;
  jobs: Map<jobId, Job>;
  fileTree: FileNode[];
  runs: RunMeta[];
  agent: {
    threadId: string | null;
    messages: ChatMessage[];
    status: 'idle' | 'streaming' | 'awaiting_confirmation';
    pendingConfirmation: { tool: string; args: object; previewCmd: string } | null;
  };
  settings: { providers: Record<string, ProviderState>; active: string | null };
}
```

## 9. Data flows

### A. Server boot + initial load
1. `evalyn dashboard` -> entry point -> `evalyn_dashboard.cli_command.cmd_dashboard(args)` -> `server.main(...)`.
2. `introspect.build_catalog()` runs (~10ms), caches catalog.
3. uvicorn binds 127.0.0.1:7401, on ready -> `webbrowser.open(...)`.
4. Browser GET `/` -> `index.html` -> JS bundle hydrates.
5. App boot effects in parallel: GET `/api/cli`, `/api/files/tree`, `/api/runs`, `/api/settings`, `/api/jobs/recent`.
6. Welcome view renders within ~200ms.

### B. Run a CLI
1. User clicks `run-eval` -> `openCli("run-eval")` -> tab opens, CliForm renders.
2. User fills params, clicks Run.
3. Frontend POSTs `/api/cli/run` with `{cli_id, args}`.
4. Server validates args via introspector, JobManager spawns `["evalyn", "run-eval", ...]`, returns `{job_id}`.
5. Frontend opens WS `/ws/jobs/{job_id}`.
6. Per stdout/stderr line, server pushes `{type, line, ts}`. On exit, pushes `{type: "exit", code, duration}`.
7. Tab swaps from CliForm to running view (header + live terminal). Bottom Jobs panel shows running row.
8. On exit, tab title gets pass/fail tone.
9. Cancel: POST `/api/jobs/{id}/cancel` -> SIGTERM + 3s grace + SIGKILL. WS pushes exit event.

### C. Agent loop
1. User types question in ChatPanel. POST `/api/agent/chat` -> `{thread_id}`.
2. Frontend opens WS `/ws/agent/{thread_id}`.
3. Server agent loop:
   - `provider.stream_chat(messages, tools=catalog_as_tools)` -> push `text_delta` events.
   - Model emits tool_call -> push `tool_call_proposal`.
   - Check read-only allowlist:
     - Allowed: spawn job, push `tool_call_running`, await exit, push `tool_call_complete` with stdout.
     - Not allowed: push `confirmation_required`, block on asyncio.Event.
   - User confirms -> POST `/api/agent/chat/{thread_id}/confirm` -> set Event -> agent resumes.
   - Tool result fed back into messages, loop iteration N+1.
   - On natural completion or budget exceeded, push `final` event.

### D. View a run
1. User clicks run in Runs sidebar -> `openFile("82dddcc3.run")`.
2. RunView calls GET `/api/runs/82dddcc3` -> reads `results.json`.
3. Renders header + collapsible JSON tree + "Open analyze report" button.

## 10. Error handling and safety

### Network exposure
- Bind 127.0.0.1 only. `--host 0.0.0.0` requires explicit `--unsafe-bind` with stderr warning.
- No auth, no TLS. CSRF: random token in `<meta>` tag, required on mutating routes.

### Subprocess
- Always uses `asyncio.create_subprocess_exec` with arg list. Never shell=True.
- All args validated against introspector schema before spawn.
- 60min hard timeout (configurable), SIGTERM + 3s + SIGKILL.
- No sandboxing v1.

### API keys
- Never sent to frontend in plaintext.
- File mode 600, atomic writes.
- Loaded into memory at server start.

### Agent guards
- Read-only allowlist hardcoded in `agent.py` (not config).
- Confirmation timeout: 5min, defaults to rejected.
- Per-turn tool budget: 8 calls.
- Per-turn cost budget: optional via `--agent-budget`.

### Frontend errors
- Network failure: toast + retry button.
- Schema mismatch: client refetches catalog on 422.
- Provider error (401, rate limit): WS pushes `error` event, chat shows banner with link to SettingsModal.

### Crash recovery
- Server crash: foreground process, ctrl-C kills. State in `.evalyn/` survives. Tab restoration not implemented v1.
- Browser tab close mid-job: subprocess keeps running. Reattach via Jobs panel.

### Documented but unprotected
- Path traversal in user-supplied `--output`: same as terminal usage.
- LLM prompt injection from CLI output: write commands always require user confirmation; documented.
- Sensitive data in CLI output streamed to LLM: v1 streams everything; opt-out flag deferred.

## 11. Testing strategy

### Backend (pytest)
- Unit: `test_introspect.py`, `test_jobs.py`, `test_credentials.py`, `test_agent.py`.
- Integration (FastAPI TestClient + real subprocess): `test_api_cli.py`, `test_api_jobs.py`, `test_api_agent.py`, `test_security.py`.

### Frontend (Vitest + RTL)
- `CliForm.test.tsx`, `store.test.ts`, `api.test.ts`.

### E2E (Playwright)
- One happy path: start server, open browser, run `list-runs`, assert output streams, assert exit code.

### Coverage targets
- Backend: 85%+ on introspect/jobs/credentials, 70%+ on agent, 60%+ on api routers.
- Frontend: smoke only.

### CI
- `core-tests`: existing, unchanged.
- `dashboard-tests`: pytest + npm build + Vitest.
- `dashboard-e2e`: only on PRs touching `dashboard/`.

## 12. Phasing and agent team plan

### Phase 0: Repo restructure (1 agent, sequential, ~0.5 day)
- Create `dashboard/` with pyproject.
- uv workspace at root.
- Scaffold `dashboard/frontend/` (Vite + React + TS + base CSS vars).
- Add entry-point plugin discovery to core's `main.py`.
- Rename core `dashboard.py` -> `report.py` + deprecation alias.
- Write `dashboard/schema/catalog.schema.json` and generate `frontend/src/types/catalog.ts`.

### Phase 1: Parallel foundation (5 agents, ~1-2 days)

| Agent | Scope | Depends on |
|---|---|---|
| A1 server-skeleton | `server.py`, FastAPI app, static mount, CSRF, browser-open, `cmd_dashboard` | Phase 0 |
| A2 cli-introspector | `introspect.py` + tests for all 35 commands | Phase 0 |
| A3 job-manager | `jobs.py` + tests (spawn/stream/cancel/backpressure) | Phase 0 |
| A4 credentials | `credentials.py` + tests (chmod 600, atomic) | Phase 0 |
| A5 frontend-shell | App, TitleBar, EditorTabs, BottomPanel skeleton, store skeleton, themes | Phase 0 |

### Phase 2: Wire CLI execution (3 agents, ~2 days)

| Agent | Scope | Depends on |
|---|---|---|
| B1 cli+jobs-api | `api/cli.py`, `api/jobs.py`, `/ws/jobs/{id}` | A1, A2, A3 |
| B2 frontend-sidebar+forms | Sidebar, CliCatalog, CliForm 3-mode, ParamField | A5, B1 |
| B3 frontend-terminal+jobs | Terminal, JobsList, WS subscriber | A5, B1 |

End of Phase 2: B-scope MVP feature-complete. All 35 CLIs runnable end-to-end.

### Phase 3: Agent + settings (2 agents, ~2-3 days)

| Agent | Scope | Depends on |
|---|---|---|
| C1 agent-runtime | `agent.py`, providers, tool loop, allowlist, confirmation; `api/agent.py` + WS; `api/settings.py` | A3, A4, B1 |
| C2 frontend-chat+settings | ChatPanel, SettingsModal, confirmation UI | A5, B2, C1 |

### Phase 4: Polish + ship (2 agents, ~1 day)

| Agent | Scope | Depends on |
|---|---|---|
| D1 e2e + ci | Playwright happy-path, CI matrix updates | All prior |
| D2 docs + release | `dashboard/README.md`, install docs, CHANGELOG, version compat | All prior |

### Critical path
~7-9 days end-to-end with team. ~3-4 weeks solo.

### Coordination
- Phase 0 writes shared catalog schema. A2 and B2 work against the same spec.
- Frontend agents (A5, B2, B3, C2) all touch `App.tsx` and `store.ts`. Phase 0 writes stubs with named slots; agents fill slots. Minimizes merge conflicts.
- Phase boundaries are sync points: all prior phase tests must pass before next phase starts.

## 13. Open questions

1. Telemetry: emit usage events back to evalyn for product insight? Recommended no for v1.
2. Auto-update: warn at startup if dashboard package is older than core's expected version? Recommended yes (log only, no auto-action).
3. First-run UX with no `.evalyn/` directory: setup wizard or empty welcome? Recommended empty welcome with hint.
4. `evalyn-dashboard` Python version floor: match core (3.10) or raise (3.11 for newer typing)? Recommended match core.
5. Per-job stdout retention beyond memory cap (e.g. tail to disk)? Deferred to v2.
6. Cookie-based session for multiple browser windows hitting one server? v1 has no session, but multi-tab works fine via WS reattach. Defer.
