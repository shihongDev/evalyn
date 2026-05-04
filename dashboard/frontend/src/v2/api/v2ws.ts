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
 *  - Auto-reconnect with a fixed 2s backoff. Best-effort: if the WS
 *    repeatedly fails to connect we just keep the FE running on the
 *    existing useV2Resource refresh-on-nav semantics.
 *
 * Protocol kept in sync with ``api/v2/v2_ws.py``. Add a new variant
 * here when you add a new event type on the backend.
 */

export type V2Event =
  | { type: 'hello'; v: number }
  | { type: 'cache_invalidate'; keys: string[] }
  | { type: 'pong' };

type Listener = (evt: V2Event) => void;

const RECONNECT_DELAY_MS = 2000;

let _ws: WebSocket | null = null;
let _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
const _listeners = new Set<Listener>();

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

function scheduleReconnect(): void {
  clearReconnectTimer();
  _reconnectTimer = setTimeout(() => {
    _reconnectTimer = null;
    startV2EventStream();
  }, RECONNECT_DELAY_MS);
}

function isV2Event(value: unknown): value is V2Event {
  if (value == null || typeof value !== 'object') return false;
  const t = (value as { type?: unknown }).type;
  return t === 'hello' || t === 'cache_invalidate' || t === 'pong';
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
    console.warn('v2 ws connect failed', err);
    scheduleReconnect();
    return;
  }
  _ws = ws;

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
    scheduleReconnect();
  });

  ws.addEventListener('error', (e) => {
    // ``error`` always precedes ``close`` for failed connects; we
    // schedule the reconnect from ``close`` so we don't double-up.
    console.warn('v2 ws error', e);
  });
}

/** Subscribe to v2 events. Returns an unsubscribe fn. */
export function subscribeV2Events(fn: Listener): () => void {
  _listeners.add(fn);
  return () => {
    _listeners.delete(fn);
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
}
