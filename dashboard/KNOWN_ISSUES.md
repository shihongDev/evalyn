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
