# Known Issues

Tracked but not yet fixed. Each entry includes severity, file:line, what's wrong, and a fix sketch.

## IMPORTANT

### 1. Subscriber race in JobManager.subscribe() (jobs.py:289-295)

In `subscribe()`, events are replayed from `job.events` BEFORE the stream is registered in `job._subscribers`. Between the last replay `await` and the registration, `_emit()` can run and deliver live events only to already-registered subscribers, causing the new subscriber to miss events. Truncation horizon does not protect this gap because the events are real, not dropped.

**Fix sketch:** Register the subscriber in `_subscribers` first, capture the current event_id as a snapshot, then replay only events with `id <= snapshot_id`. The live fanout will deliver events with `id > snapshot_id`. No duplication.

### 2. WS reconnect close handler shares mutable conn ref (store.ts:329, 389)

The close handler captures `conn` in its closure but `conn` is the same object mutated by subsequent `open()` calls. If `open()` is called recursively via the retry timer while the old connection's close event fires, both closures share the same `conn` reference. Edge case: retry never fires if `conn.closed` is set by `unsubscribeJob` between timer-set and timer-fire.

**Fix sketch:** Capture `conn` snapshot per close-handler invocation (immutable record), and use a unique generation counter per reconnect attempt to ignore stale close events.

## LOW

### Compare-two-runs side-by-side view (deferred)

The Workspace's run history feed renders one card per run with collapse/expand,
Edit, Pin, and Remove actions. A side-by-side diff view ("compare run A vs run B,
arg-diff + metric-diff + log-diff") is intentionally deferred. Sketch when picked
up: pin two runs and add a "Compare" button on the Pinned row that opens a
two-column panel; reuse the existing `diffArgs` helper in
`views/Workspace.tsx` / `views/RunCard.tsx`. Layout TBD.

### 3. AgentRuntime.subscribe race (agent.py:700-713)

Same pattern as #1: events emitted between replay and subscriber registration are dropped. Risk is higher here because confirmation_required events not reaching client cause stuck UI.

**Fix sketch:** identical to #1.

### 4. ToolCallCard confirm doesn't pass call.id (ChatPanel.tsx:208-219)

RESOLVED 2026-05-01 (feat/dashboard-ux-p0). ToolCallCard now passes `call.id` to `confirmAgent(approve, toolCallId)`; the store validates it against `pendingConfirmation.toolCallId` before posting and refuses (with a console warning) on mismatch. Fixed in tandem with #5.

### 5. /api/agent/chat/{thread_id}/confirm ignores tool_call_id (api/agent.py:57-65)

RESOLVED 2026-05-01 (feat/dashboard-ux-p0). The route now reads `tool_call_id` from the body and forwards it to `AgentRuntime.confirm`, which validates against the per-thread `pending_tool_call_id` (set when the gate is armed in `_execute_tool_call`). Mismatches return HTTP 409 instead of silently flipping the wrong gate. The `tool_call_id` parameter remains optional in `AgentRuntime.confirm` to preserve backward compatibility with existing tests.
