# Known Issues

Tracked but not yet fixed. Each entry includes severity, file:line, what's wrong, and a fix sketch.

## IMPORTANT

### 1. Subscriber race in JobManager.subscribe() (jobs.py:289-295)

RESOLVED 2026-05-06 (fix/jobs-subscribe-race). `_EventStream` gained a replay phase. `subscribe()` now calls `_begin_replay()` and registers the stream in `_subscribers` BEFORE any await, snapshots the current `_next_event_id`, replays only events with `id <= snapshot_id` from a pre-built list, then calls `_end_replay()` which flushes any concurrent `_emit()` events that arrived during replay. Live events emitted while replaying divert into `_live_buffer` so they deliver in emit order AFTER replay events, never lost or reordered. Regression test: `test_jobs.py::test_subscribe_registers_before_replay_and_buffers_concurrent_emit`.

### 2. WS reconnect close handler shares mutable conn ref (store.ts:329, 389)

OBSOLETE 2026-05-06 (reliability/jobs-since-cursor). The workbench-era `store.ts` and `unsubscribeJob` helper that this entry referenced were deleted during the v2 dashboard rewrite. The current shared WS subscriber lives in `dashboard/frontend/src/v2/api/v2ws.ts` and already implements the recommended pattern: each close handler captures the local `ws` reference (not a shared ref) and only clears `_ws` when `_ws === ws`, so a stale close event from a previous connection cannot null out a fresh one. The per-job WS in `v2/api/jobs.ts::subscribeJob` does not auto-reconnect; callers re-invoke it. As of this entry's resolution, `subscribeJob` accepts `options.since` and surfaces `event_id` on `JobLine`, enabling resume-on-reconnect without re-receiving already-delivered events.

## LOW

### Compare-two-runs side-by-side view (deferred)

The Workspace's run history feed renders one card per run with collapse/expand,
Edit, Pin, and Remove actions. A side-by-side diff view ("compare run A vs run B,
arg-diff + metric-diff + log-diff") is intentionally deferred. Sketch when picked
up: pin two runs and add a "Compare" button on the Pinned row that opens a
two-column panel; reuse the existing `diffArgs` helper in
`views/Workspace.tsx` / `views/RunCard.tsx`. Layout TBD.

### 3. AgentRuntime.subscribe race (agent.py:700-713)

RESOLVED 2026-05-06 (fix/agent-subscribe-race). Mirrored the #1 fix to `_AgentEventStream`: it now has a replay phase with a live-event buffer. `AgentRuntime.subscribe` calls `_begin_replay()`, registers the stream BEFORE replay, snapshots `_next_event_id`, replays via `_replay_put` (which bypasses the buffer), then `_end_replay()` flushes any concurrent `_emit()` events that arrived during replay. Confirmation_required events emitted in the registration window are no longer dropped; UI cannot get stuck waiting for them. Regression tests: `test_agent.py::test_subscribe_registers_before_replay` and `::test_agent_event_stream_replay_buffer_orders_live_after_flush`.

### 4. ToolCallCard confirm doesn't pass call.id (ChatPanel.tsx:208-219)

RESOLVED 2026-05-01 (feat/dashboard-ux-p0). ToolCallCard now passes `call.id` to `confirmAgent(approve, toolCallId)`; the store validates it against `pendingConfirmation.toolCallId` before posting and refuses (with a console warning) on mismatch. Fixed in tandem with #5.

### 5. /api/agent/chat/{thread_id}/confirm ignores tool_call_id (api/agent.py:57-65)

RESOLVED 2026-05-01 (feat/dashboard-ux-p0). The route now reads `tool_call_id` from the body and forwards it to `AgentRuntime.confirm`, which validates against the per-thread `pending_tool_call_id` (set when the gate is armed in `_execute_tool_call`). Mismatches return HTTP 409 instead of silently flipping the wrong gate. The `tool_call_id` parameter remains optional in `AgentRuntime.confirm` to preserve backward compatibility with existing tests.
