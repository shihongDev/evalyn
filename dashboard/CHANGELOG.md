# Changelog

All notable changes to `evalyn-dashboard` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Tab title surfaces unacknowledged failures.** The AppShell title prefix already shows `(N)` when jobs are running; it now also appends `!N` when N recent jobs failed since the user last opened the Recent Jobs drawer (e.g. `(1) !2 Eval · Evalyn`). Backed by a new `failed_at_iso` field stamped on first status-transition to `failed` and a localStorage `ACK_KEY` timestamp updated whenever the drawer opens. Cross-tab synced via the `storage` event so opening the drawer in one tab clears the badge in others. Lets a user on another browser tab notice an overnight or background-tab eval failure without having to click into the dashboard.
- **Failed-jobs filter chip in RecentJobsDrawer header.** The drawer keeps up to 30 recent jobs in local history; spotting which ones failed used to mean scanning every status pill. A new clickable chip in the header (`X failed`) appears whenever any row's status is `failed` and toggles a "show failed only" filter, mirroring the pattern from the CliRunner stderr chip. The filter excludes `cancelled` (user-driven) and `unknown` (server-evicted) since neither is an unexpected failure. Auto-clears when the failed count drops to 0 so the chip never disappears mid-filter and strands the user with an empty list. A defensive empty-state covers the brief race window between toggle and auto-clear.
- **Jump-to-first-error button in CliRunner output header.** When stderr exists and the filter is off, a small `↥ first` button next to the stderr-count chip scrolls the first stderr line into view (`scrollIntoView` with `block: 'center'`) so the user can locate the first failure point without losing the surrounding stdout chronology. Hidden when the filter is on (the first error is already on screen) or when stderr is empty. Companion to the click-to-filter chip below: filter for "show only errors", jump-to-first for "show me where it started" while keeping context.
- **Click-to-filter stderr in CliRunner output.** The stderr-count chip in the CliRunner output header is now interactive: click it to filter the visible output to stderr (and system) lines only, click again to show all output. The chip's style (filled background, bold weight, leading checkmark) reflects whether the filter is active. A dedicated empty-state shows "No stderr lines yet" when the filter is on and nothing matches, instead of the default "(no output)" which would be misleading. The Copy button continues to copy ALL lines regardless of the filter — the filter is for scanning, the clipboard is for sharing. The filter auto-clears whenever the stderr count drops to 0 (terminal status with no errors, or all stderr evicted from the ring buffer mid-run) so the user can never get stuck with a stale filter when the chip itself is no longer rendered. Lets users find a buried failure in a chatty eval in one click instead of manual scrolling.

### Performance

- **`subscribeJob` auto-reconnects on unexpected close with `?since=N` cursor.** Network blips, dev-server restarts, and brief disconnects mid-stream used to leave CliRunner stuck on a "Lost connection to job stream" toast; the user had to refresh the page to resume tailing. The WS subscriber now retries with exponential backoff (1s → 2s → 4s → 8s → 8s, max 5 attempts ≈ 23s patience) and passes the highest observed `event_id` as `?since=N` so the backend replays only events the client has not yet seen — no duplicate lines, no missing lines. The "running" onStatus pill fires on first connect only, so a reconnect after the user clicked Cancel does not flip the UI back to running. `onClose` is suppressed for intermediate disconnects we recover from, then fires once on terminal close (user-initiated, server-delivered exit, or attempts exhausted). After 5 failed attempts, `onError` fires with a clear "Lost job stream after 5 reconnect attempts" message. Existing callers benefit transparently — no API change.
- **Co-pilot chat text-delta batching via requestAnimationFrame.** The streaming agent reply previously called `setMessages` per token (~80/sec for Sonnet, faster for Haiku), each rebuilding the messages array, the agent bubble, and concatenating bubble text. `useCoPilotThread` now coalesces deltas into a `Map<messageId, accumulatedText>` ref and flushes once per animation frame, so the chat re-renders at most ~60Hz regardless of token rate. Other event kinds (tool_call_*, final, error) pass through directly. The `final` event force-flushes pending deltas before applying its patch so the bubble's text is current and never "rewinds" mid-stream. WS reconnect, thread reset, and unmount all cancel pending rAF and clear the buffer to prevent stale tokens leaking into a new conversation.
- **CliRunner WS line batching via requestAnimationFrame.** Chatty eval runs (100s of stdout lines/sec) used to call `setLines` once per WebSocket message, triggering one React render per line. CliRunner now buffers lines in a ref and flushes once per animation frame (`requestAnimationFrame`), capping render rate at ~60Hz regardless of line rate. The buffer is bounded at `MAX_OUTPUT_LINES` so a backgrounded tab cannot accumulate unbounded memory while rAF is paused. Cleanup on unmount and Re-run cancels any pending frame and clears stale lines so they do not bleed into a new run. No visible behavior change for slow streams; high-rate streams stop juddering.

### Changed

- **`subscribeJob` reconnect-with-cursor.** `dashboard/frontend/src/v2/api/jobs.ts::subscribeJob` now accepts an optional `{ since }` option and forwards it as `?since=N` on the `/ws/jobs/{id}` URL; the backend's existing `since` support means a reconnecting client can skip events with `event_id <= since`. `JobLine` gained an optional `event_id` field carrying the server-assigned monotonic id, so callers can track the high-water mark and pass it back on reconnect to avoid re-receiving already-displayed lines. Additive: no behavior change for current callers.
- **KNOWN_ISSUES.md #2 marked OBSOLETE.** The "WS reconnect close handler shares mutable conn ref" issue described `store.ts:329, 389` and the `unsubscribeJob` helper, both deleted in the v2 dashboard rewrite. The current shared WS in `v2/api/v2ws.ts` already implements the recommended `_ws === ws` close-handler guard. Cleaned up the entry so future maintainers do not chase a non-existent bug.

### Fixed

- **Backend job-persistence sqlite mirror was growing unboundedly.** `JobPersistence.delete_old()` was defined but never called by any caller, so every spawned job permanently added a row to `.evalyn/data/jobs.sqlite`. Over months of use the file would grow to MBs and `/api/jobs/recent` queries would slow proportionally. `JobManager` now invokes `_persist_gc_maybe()` at the end of every `_persist_job_terminal`; it counts terminal events and prunes the table to the most-recent `persistence_keep` rows (default 200) every `persistence_gc_interval` terminals (default 50). Both are tunable via constructor args. GC runs in its own try/except so a prune failure cannot poison the per-job persist path. Adds three regression tests: GC fires on terminal at threshold, GC is skipped below the interval, and GC is a safe no-op when persistence is disabled.
- **Subscriber race in `AgentRuntime.subscribe()`** (KNOWN_ISSUES.md #3). Same pattern as the JobManager fix below: `_AgentEventStream` gained a replay phase with `_live_buffer`, `_replay_put`, `_begin_replay`, `_end_replay`. `AgentRuntime.subscribe` now registers the stream BEFORE replay, snapshots `_next_event_id`, replays via `_replay_put` (bypasses the buffer), and flushes concurrent `_emit` events at the end. Confirmation_required events emitted during the previous registration gap are no longer dropped; the chat UI cannot get stuck waiting for them.
- **Subscriber race in `JobManager.subscribe()`** (KNOWN_ISSUES.md #1). Live job events emitted between the replay phase and subscriber registration could be silently dropped on slow or freshly-attached WebSocket clients. `_EventStream` now buffers concurrent `_emit()` events into `_live_buffer` during replay; `subscribe()` registers the stream first, snapshots `_next_event_id`, replays from the snapshot, then flushes the buffer in order. Result: no event loss and no out-of-order delivery, even if future changes make queue puts yield mid-replay.

### Added

- **Co-pilot guidance tours.** Each tabbed route (Home, Datasets, Experiments, Review, Metrics) ships a short Driver.js-powered walk-through that auto-fires on first visit, anchored to stable `data-coachmark` attributes on real UI elements. Each tour has an independent `localStorage` completion flag, so finishing one does not suppress the others.
- **Tours menu in the AppShell header.** Always-visible button with a status dot (ember = on, muted = off). Click to: toggle all guidance off/on, manually start the current page's tour, or reset every per-tour completion flag in one action. Closes on outside click and Escape.
- **Per-route Settings reset.** The "Reset first-visit flags" action in Settings now clears every tour's completion flag (previously only the Home first-run tour). Sourced from `KNOWN_TOUR_IDS`, derived automatically from the tour registry.
- **Anchor-not-found policy.** If a tour step targets an element that is not in the DOM (empty workspace, conditional render), the engine waits 500ms then silently elides that step rather than deadlocking. If every step elides, the tour abandons without firing instead of showing an empty popover. The user sees a tighter sequence of real steps; manual re-trigger via the Tours menu still works once the page populates.
- **`prefers-reduced-motion` respected** for the focus ring; static outline replaces the pulse when the OS-level setting is on.

### Fixed

- **`dashboard/frontend/src/v2/tour/scripts/` was being silently gitignored** by the unanchored top-level `scripts/` rule. Added a `!` exception so all tour scripts are first-class source going forward; the Home `firstRun.ts` tour script that previously lived only on disk is now tracked in git.

## [0.1.0] - 2026-05-01

Initial release. Localhost IDE for evalyn evaluations, distributed as a separate optional package.

### Added

- **CLI catalog** - all 35 evalyn CLIs introspected from their argparse parsers and exposed as auto-generated forms. Groups: Tracing, Dataset, Metrics, Eval, Analysis, Annotation, Insights, Export, Simulation, Infrastructure, Quickstart.
- **Auto-form generation** - 7 param kinds (`bool`, `string`, `number`, `select`, `multiselect`, `path`, `long-text`) classified from argparse `Action` shape and dest-name heuristics. Three form modes: Form, Preview, Raw.
- **Subprocess streaming** - `JobManager` spawns each CLI as `["evalyn", <cmd>, ...]` via `asyncio.create_subprocess_exec`. Per-line stdout/stderr capture, fanout subscribe for multiple WebSocket viewers, backpressure with truncation marker, SIGTERM + 3s grace + SIGKILL on cancel, 60min default timeout, last 100 jobs retained in memory.
- **Terminal panel** - inline ~1KB ANSI parser, auto-scroll, one terminal view per job tab.
- **Jobs panel** - live state of running and recent jobs, click-to-open, per-row cancel.
- **AI chat agent** - dock-right `ChatPanel` with full agentic loop: text streaming, tool-call cards, confirmation cards (approve/reject), final-suggestion cards (clickable into pre-filled CLI form). Per-turn budget of 8 tool calls.
- **Multi-provider support** - `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider` behind a shared `BaseProvider` interface. Provider-native tool-call streaming.
- **Credentials store** - `~/.evalyn/credentials.json` with atomic write and `chmod 600`. API never returns plaintext keys to the frontend. Per-provider `test` endpoint makes a 1-token completion call.
- **Settings UI** - `SettingsModal` with API-key input (password type), test button, model dropdown, active-provider radio.
- **Read-only allowlist** - 19 commands auto-run by the agent without confirmation. All other commands require explicit user approval via a confirmation card; 5min timeout defaults to rejected.
- **WebSocket transport** - `/ws/jobs/{id}` for job streams, `/ws/agent/{thread_id}` for agent events. Reconnect with `last_event_id`.
- **Localhost binding** - bound to `127.0.0.1` by default. `--unsafe-bind` required to override, with stderr warning.
- **CSRF protection** - per-server random token injected into served `index.html` as `<meta name="workbench-token">`. Required on all mutating routes via `X-Workbench-Token` header.
- **Plugin discovery** - registers the `dashboard` subcommand on core evalyn via the `evalyn.commands` entry-point group. No core changes needed at install time.
- **Pre-built frontend** - React 18 + TypeScript + Vite bundle vendored at `evalyn_dashboard/static/`. End users never run `npm`.

### Notes

- **Deprecation alias.** The previous `evalyn dashboard` command (a static HTML insights report) is renamed to `evalyn report`. When `evalyn-dashboard` is **not** installed, `evalyn dashboard` prints a stderr deprecation warning and forwards to `evalyn report`. The alias is removed in core evalyn v3.0.
- **Known issues.** See [KNOWN_ISSUES.md](./KNOWN_ISSUES.md). Two `IMPORTANT` items are tracked: (1) subscriber race in `JobManager.subscribe()` between replay and registration; (2) WS reconnect close handler shares mutable connection ref. Both have fix sketches; neither blocks v0.1.
- **State persistence.** Tabs, jobs, and chat threads are in-memory only. Server restart clears them. `.evalyn/` artifacts on disk are unaffected.
- **No telemetry, no auto-update, no daemon mode.**

### Implementation phases

For maintainers: v0.1.0 was built in 5 phases.

- Phase 0 - repo restructure: uv workspace, `dashboard/` package, Vite scaffold, shared `catalog.schema.json`, plugin entry-point on core CLI, rename `dashboard` -> `report` + deprecation alias.
- Phase 1 - foundation (5 parallel lanes): FastAPI skeleton + CSRF + localhost guard, argparse introspector, `JobManager` (spawn / cancel / fanout / backpressure / history), `CredentialStore`, frontend shell (`App` / `TitleBar` / `EditorTabs` / `BottomPanel` / Zustand store).
- Phase 2 - CLI execution (3 parallel lanes): `/api/cli` + `/api/cli/run` + `/api/jobs/*` + `/ws/jobs/{id}`, sidebar + `CliCatalog` + `CliForm` + `ParamField`, `Terminal` + `JobsList` + WS subscriber.
- Phase 3 - agent runtime (2 parallel lanes): `AgentRuntime` + 3 providers + tool loop + allowlist + confirmation gate + `/api/agent/*` + `/ws/agent/{id}` + `/api/settings/*`, `ChatPanel` + `SettingsModal` + tool/confirmation/suggestion cards.
- Phase 4 - polish (2 parallel lanes): Playwright E2E + CI matrix, docs + CHANGELOG + release.
