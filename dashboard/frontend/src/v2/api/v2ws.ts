/**
 * v2ws - subscriber for /ws/v2/events.
 *
 * Why: the v2 routers cache responses by dataset-root mtimes. When the
 * CLI writes a fresh run via run-eval the FE has no way to know the
 * cache invalidated - so the backend pushes a tiny ``cache_invalidate``
 * frame and we forward it to every registered listener.
 *
 * Lifecycle:
 *  - One shared WebSocket per tab. ``startV2EventStream`` is idempotent
 *    so the AppShell can call it on every mount without duplicating
 *    sockets; subsequent calls are no-ops while a connection is open
 *    or actively reconnecting.
 *  - Auto-reconnect with exponential backoff plus jitter, capped at
 *    RECONNECT_MAX_MS (see _computeBackoffMs). The ladder is
 *    2s -> 4s -> 8s -> 16s -> 30s with a [0, 500ms) jitter on each
 *    attempt to prevent N tabs from thundering on the server when
 *    it comes back. The attempt counter resets to 0 on every
 *    successful open. Best-effort: if the WS repeatedly fails to
 *    connect we just keep the FE running on the existing
 *    useV2Resource refresh-on-nav semantics.
 *
 * Protocol kept in sync with ``api/v2/v2_ws.py``. Add a new variant
 * here when you add a new event type on the backend.
 */

export type V2Event =
  | { type: 'hello'; v: number }
  | { type: 'cache_invalidate'; keys: string[] }
  | { type: 'pong' }
  // Server-pushed heartbeat ping. The FE doesn't reply or react;
  // the frame's job is just to keep NATs from reaping idle WS
  // connections. The shared `_ws_heartbeat` helper sends one
  // every 25s. Listed in the union so `isV2Event` lets the frame
  // pass through (any unrecognized type would be filtered out
  // before reaching subscribers, who then never see it).
  | { type: 'ping'; ts: number };

/** Coarse connection status surfaced to UI subscribers.
 *
 *   - 'open': WS is live, events flowing.
 *   - 'reconnecting': WS is closed and we are between attempts (or
 *     trying to connect). Includes the brief moment before the first
 *     hello frame on a fresh open.
 *
 * The hook surface intentionally collapses 'connecting' / 'closed' /
 * 'failed' into one bucket because the user response is the same:
 * "the live feed isn't healthy right now." Detail belongs in logs. */
export type ConnectionStatus = 'open' | 'reconnecting';

type Listener = (evt: V2Event) => void;
type StatusListener = (status: ConnectionStatus) => void;

// Reconnect uses exponential backoff with jitter, capped. The
// previous fixed 2s was fine when the server is briefly down
// (dev restart) but wasted bandwidth when the server is gone for
// real (deploy hiccup, network partition). 2s -> 4s -> 8s -> 16s
// -> 30s (capped). Reset to RECONNECT_BASE_MS on every successful
// open. Jitter prevents N tabs from thundering on the server the
// instant it comes back.
const RECONNECT_BASE_MS = 2000;
const RECONNECT_MAX_MS = 30000;
const RECONNECT_JITTER_MS = 500;

let _ws: WebSocket | null = null;
let _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let _reconnectAttempt = 0;
let _status: ConnectionStatus = 'reconnecting';
const _listeners = new Set<Listener>();
const _statusListeners = new Set<StatusListener>();

// Tracks whether we've already console.warn'd about an outage in
// the current reconnect window. Without this, every failed retry
// during a dev-server restart spams the console once per attempt
// (5+ warns for a typical 2s -> 4s -> 8s -> 16s -> 30s ladder),
// which buries real errors. Reset to false on every successful
// open so a future flake produces one new warn.
let _haveLoggedCurrentOutage = false;

function setStatus(next: ConnectionStatus): void {
  if (_status === next) return;
  _status = next;
  for (const l of Array.from(_statusListeners)) {
    try {
      l(next);
    } catch (err) {
      console.error('v2 ws status listener error', err);
    }
  }
}

function wsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/v2/events`;
}

function clearReconnectTimer(): void {
  if (_reconnectTimer != null) {
    clearTimeout(_reconnectTimer);
    _reconnectTimer = null;
  }
}

/** Compute the next backoff delay. Doubled each attempt, capped
 * at RECONNECT_MAX_MS, plus a random jitter of [0, RECONNECT_JITTER_MS)
 * so concurrent dashboards don't reconnect in lockstep after a
 * server restart (thundering herd). Exported for tests. */
export function _computeBackoffMs(attempt: number): number {
  const exp = Math.min(
    RECONNECT_BASE_MS * Math.pow(2, attempt),
    RECONNECT_MAX_MS,
  );
  return exp + Math.random() * RECONNECT_JITTER_MS;
}

function scheduleReconnect(): void {
  clearReconnectTimer();
  const delay = _computeBackoffMs(_reconnectAttempt);
  _reconnectAttempt += 1;
  _reconnectTimer = setTimeout(() => {
    _reconnectTimer = null;
    startV2EventStream();
  }, delay);
}

function isV2Event(value: unknown): value is V2Event {
  if (value == null || typeof value !== 'object') return false;
  const t = (value as { type?: unknown }).type;
  return (
    t === 'hello' ||
    t === 'cache_invalidate' ||
    t === 'pong' ||
    t === 'ping'
  );
}

/**
 * Open the shared WS if it isn't already open or pending. Safe to call
 * multiple times; the AppShell calls it once on mount.
 */
export function startV2EventStream(): void {
  // Skip when SSR or in environments without WebSocket (jsdom tests).
  if (typeof window === 'undefined' || typeof WebSocket === 'undefined') return;
  if (_ws != null) return;

  let ws: WebSocket;
  try {
    ws = new WebSocket(wsUrl());
  } catch (err) {
    // Throttle: same rationale as the 'error' handler below. The
    // WebSocket constructor synchronously throws on some
    // browsers when the URL is invalid or the protocol's
    // disabled; we don't want a permanent block to spam every
    // reconnect attempt.
    if (!_haveLoggedCurrentOutage) {
      _haveLoggedCurrentOutage = true;
      console.warn('v2 ws connect failed', err);
    }
    scheduleReconnect();
    return;
  }
  _ws = ws;

  ws.addEventListener('open', () => {
    setStatus('open');
    // Reset the backoff so a future flake re-starts at
    // RECONNECT_BASE_MS rather than the last attempt's tier.
    _reconnectAttempt = 0;
    // Reset the log-throttle so a future outage produces a
    // fresh log line. Without this, after one warn-then-recovery
    // cycle, all subsequent outages would be silent.
    _haveLoggedCurrentOutage = false;
  });

  ws.addEventListener('message', (e) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(e.data as string);
    } catch (err) {
      console.error('v2 ws parse error', err);
      return;
    }
    if (!isV2Event(parsed)) return;
    // Snapshot so a listener that unsubscribes itself doesn't mutate
    // the iterator mid-loop.
    for (const l of Array.from(_listeners)) {
      try {
        l(parsed);
      } catch (err) {
        console.error('v2 ws listener error', err);
      }
    }
  });

  ws.addEventListener('close', () => {
    if (_ws === ws) _ws = null;
    setStatus('reconnecting');
    scheduleReconnect();
  });

  ws.addEventListener('error', (e) => {
    // ``error`` always precedes ``close`` for failed connects; we
    // schedule the reconnect from ``close`` so we don't double-up.
    // Throttle: log once per outage window (cleared by a successful
    // open). Without throttle, a 5-attempt reconnect ladder during
    // a dev-server restart spams 5 console.warns - useful info
    // becomes noise, real errors get buried.
    if (!_haveLoggedCurrentOutage) {
      _haveLoggedCurrentOutage = true;
      console.warn('v2 ws error', e);
    }
  });
}

/** Subscribe to v2 events. Returns an unsubscribe fn. */
export function subscribeV2Events(fn: Listener): () => void {
  _listeners.add(fn);
  return () => {
    _listeners.delete(fn);
  };
}

/** Subscribe to connection-status transitions. The listener is called
 * with the current status immediately on subscribe so callers don't
 * have to track initial state separately. Returns an unsubscribe fn. */
export function subscribeV2Status(fn: StatusListener): () => void {
  _statusListeners.add(fn);
  fn(_status);
  return () => {
    _statusListeners.delete(fn);
  };
}

/** Test-only: tear the stream down between specs. */
export function _resetV2EventStream(): void {
  clearReconnectTimer();
  if (_ws != null) {
    try {
      _ws.close();
    } catch {
      /* ignore */
    }
    _ws = null;
  }
  _listeners.clear();
  _statusListeners.clear();
  _status = 'reconnecting';
  _reconnectAttempt = 0;
  _haveLoggedCurrentOutage = false;
}
