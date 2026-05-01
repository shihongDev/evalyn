# evalyn-dashboard

Localhost IDE for evalyn evaluations. Auto-generated forms for all 35 evalyn CLIs, real-time subprocess streaming, and an AI chat agent that calls evalyn commands as tools.

Distributed as a separate optional package so the core `evalyn` SDK stays lightweight.

## Install

```bash
pip install evalyn-dashboard
```

This pulls in `evalyn` automatically. Python 3.10+.

## Run

```bash
evalyn dashboard
```

Opens http://localhost:7401 in your default browser. Bound to `127.0.0.1` only. Ctrl-C to exit.

Flags:

```
evalyn dashboard --host 127.0.0.1 --port 7401 --no-browser
```

`--no-browser` skips the auto-open. `--unsafe-bind` is required to bind a non-loopback interface (not recommended).

## Configure

The dashboard ships without any provider keys. To use the AI chat agent:

1. Click the gear icon in the title bar to open Settings.
2. Pick a provider:
   - **OpenAI** - paste an API key (`sk-...`), pick a model (e.g. `gpt-5.1`).
   - **Anthropic** - paste an API key (`sk-ant-...`), pick a model (e.g. `claude-sonnet-4-6`).
   - **Ollama** - set the base URL (default `http://localhost:11434`), pick a locally-pulled model.
3. Click "Test" to verify a 1-token completion call succeeds.
4. Set the active provider via the radio button.

Keys are stored in `~/.evalyn/credentials.json` with mode `0600` and atomic writes. The dashboard never returns plaintext keys to the browser. The credentials file schema:

```json
{
  "openai":    { "api_key": "sk-...",     "model": "gpt-5.1",         "added_at": "..." },
  "anthropic": { "api_key": "sk-ant-...", "model": "claude-sonnet-4-6","added_at": "..." },
  "ollama":    { "base_url": "http://localhost:11434", "model": "llama3:70b", "added_at": "..." },
  "active": "anthropic"
}
```

You can also edit this file by hand. The dashboard re-reads it on each request.

## Use

### Run any evalyn CLI

1. Open the **CLIs** tab in the left sidebar. All 35 commands are listed and grouped (Tracing, Dataset, Metrics, Eval, Analysis, Annotation, Insights, Export, Simulation, Infrastructure, Quickstart).
2. Click a command (e.g. `run-eval`). A new tab opens with an auto-generated form.
3. Three modes: **Form** (typed inputs per param), **Preview** (assembled command line), **Raw** (paste an arg string).
4. Fill required params, click **Run**. The tab swaps to a live terminal showing streamed stdout/stderr. The bottom **Jobs** panel tracks running and recent jobs.
5. Cancel a running job from the Jobs panel: SIGTERM, 3s grace, then SIGKILL.

Forms are introspected from each command's `argparse.ArgumentParser`. Param kinds:

| Kind | Argparse signal | UI control |
|---|---|---|
| `bool` | `action="store_true"` / `"store_false"` | toggle |
| `select` | `choices=[...]` (single) | dropdown |
| `multiselect` | `choices=[...]` + `nargs="*"` | chip toggles |
| `number` | `type=int` or `type=float` | number input |
| `path` | dest matches `path|file|dir|output|out` | text + placeholder |
| `long-text` | dest in `prompt|template|description|...` | textarea |
| `string` | otherwise | text input |

### Talk to the agent

1. Click the chat icon (top-right) to dock the chat panel.
2. Type a question, e.g. "list my recent run-eval results and summarize the lowest pass rate".
3. The agent picks tools from the catalog and:
   - Auto-runs **read-only** commands (`list-calls`, `list-runs`, `analyze`, `compare`, `insights`, ...). Output is captured (50KB cap) and fed back into the conversation.
   - Asks for **confirmation** before running write/destructive commands (`run-eval`, `calibrate`, `delete-traces`, `build-dataset`, `simulate`, `export`, ...). A confirmation card appears with the proposed argv. Click Approve or Reject.
4. Per-turn budget: max 8 tool calls. Optional cost budget via `evalyn dashboard --agent-budget 5.00`.
5. Final-suggestion cards in chat are clickable - they open a pre-filled CLI form tab so you can review and tweak before running.

The full read-only allowlist:

```
list-calls, list-runs, list-metrics, list-calibrations,
show-call, show-trace, show-span, show-projects,
analyze, compare, trend, annotation-stats,
validate, status, workflow,
cluster-failures, cluster-misalignments,
insights, select-metrics
```

Everything else requires confirmation.

## Architecture

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

Key choices:

- **Subprocess, not in-process.** Each CLI runs as `["evalyn", "<cmd>", ...]` via `asyncio.create_subprocess_exec`. Accurate stdout streaming, crash isolation, cancel via SIGTERM, parity with terminal usage.
- **Argparse introspection.** No duplicate command definitions. The catalog is regenerated at server boot in ~10ms.
- **Pre-built React bundle.** `dashboard/evalyn_dashboard/static/` is committed to git. No user-side `npm install` or `npm run build`.
- **Localhost-only.** Bound to `127.0.0.1`. Per-server CSRF token in a `<meta>` tag, required on all mutating routes via `X-Workbench-Token`.
- **Plugin entry point.** The dashboard registers itself with core evalyn through the `evalyn.commands` entry point group. Without `evalyn-dashboard` installed, `evalyn dashboard` falls back to a deprecation alias for the old static-report command.

## Frontend

- React 18 + TypeScript + Vite.
- Zustand for state. No CSS framework - CSS variables and inline styles.
- Native WebSocket. Per-job and per-agent-thread streams. Reconnects with `last_event_id`.
- Components: `TitleBar`, `Sidebar` (Files/CLIs/Runs), `EditorTabs`, `CliCatalog`, `CliForm`, `ParamField`, `Terminal`, `JobsList`, `ChatPanel`, `SettingsModal`, `BottomPanel`.

## Backend

- Python 3.10+, FastAPI, uvicorn, asyncio subprocesses.
- Provider SDKs: `openai`, `anthropic`, `httpx` (for Ollama).
- Modules:
  - `server.py` - FastAPI app, uvicorn launcher, CSRF middleware, browser open.
  - `introspect.py` - argparse -> `CliSchema` JSON.
  - `jobs.py` - `JobManager`: spawn, fanout subscribe, cancel, history, backpressure.
  - `agent.py` - `AgentRuntime` + `OpenAIProvider` / `AnthropicProvider` / `OllamaProvider`. Tool loop, allowlist, confirmation gate.
  - `credentials.py` - atomic-write credential store.
  - `api/` - one router per resource (cli, jobs, files, runs, agent, settings).
  - `cli_command.py` - the `evalyn dashboard` plugin entry.

## Known issues

See [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) for tracked issues with severity and fix sketches.

Two `IMPORTANT` issues are currently deferred:

1. Subscriber race in `JobManager.subscribe()` - new subscribers can miss events that arrive between replay and registration.
2. WS reconnect close handler shares mutable connection ref - retry timer can fail to fire under specific unsubscribe ordering.

Neither blocks v0.1. Both have fix sketches in `KNOWN_ISSUES.md`.

## Contributing

Editable install (uv workspace at repo root):

```bash
git clone https://github.com/<org>/evalyn
cd evalyn
uv sync --extra dev
```

Backend tests:

```bash
uv run pytest dashboard/tests/ -v
```

Frontend dev server (proxies `/api` and `/ws` to the backend on :7401):

```bash
cd dashboard/frontend
npm install
npm run dev          # http://localhost:5173, hot-reload
```

Frontend production build (writes to `dashboard/evalyn_dashboard/static/`):

```bash
cd dashboard/frontend
npm run build
```

The static bundle is committed to git so end users don't need npm. Rebuild and commit when frontend code changes.

E2E tests (Playwright):

```bash
uv run pytest dashboard/tests/e2e/ -v
```

## Versioning

`evalyn-dashboard` follows Semantic Versioning. The dashboard depends on `evalyn-sdk` (workspace) at runtime. Major-version bumps of either package are coordinated.

## License

MIT. Same as core evalyn.
