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

(none currently)

### 3. AgentRuntime.subscribe race (agent.py:700-713)

Same pattern as #1: events emitted between replay and subscriber registration are dropped. Risk is higher here because confirmation_required events not reaching client cause stuck UI.

**Fix sketch:** identical to #1.

### 4. ToolCallCard confirm doesn't pass call.id (ChatPanel.tsx:208-219)

`confirmAgent(approve)` reads `agent.pendingConfirmation.toolCallId` from store; the per-card button never passes its own `call.id`. With multiple stale confirmation cards (e.g. after WS replay), clicking any card confirms whatever is currently pending, not the card clicked.

**Fix sketch:** add `tool_call_id` parameter to `confirmAgent`, validate it matches `pendingConfirmation.toolCallId`.

### 5. /api/agent/chat/{thread_id}/confirm ignores tool_call_id (api/agent.py:57-65)

Endpoint accepts `tool_call_id` in body but doesn't pass it to `AgentRuntime.confirm`. Currently safe (single asyncio.Event per thread), but a malicious concurrent request could inject a confirmation. Tied to #4.

**Fix sketch:** plumb `tool_call_id` through `AgentRuntime.confirm` and validate against pending gate before setting Event.
