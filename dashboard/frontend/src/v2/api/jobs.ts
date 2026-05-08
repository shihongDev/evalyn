/**
 * Typed wrapper for ``/api/jobs/*`` HTTP routes and the
 * ``/ws/jobs/{job_id}`` WebSocket. Used by the in-UI CLI runner so any
 * route can spawn a CLI subprocess and stream its output live.
 *
 * Backend reality (mirrored here so frontend can be type-safe):
 * - ``POST /api/cli/run`` (CSRF) -> ``{job_id}`` - spawns ``evalyn <id> ...``
 * - ``POST /api/jobs/{id}/cancel`` (CSRF) -> ``{state, ...}``
 * - ``GET  /api/jobs/recent?limit=N`` -> ``Job[]``
 * - ``WS   /ws/jobs/{id}?since=N`` streams events.
 *
 * Backend event shapes (see ``dashboard/evalyn_dashboard/jobs.py``):
 *
 *     {"type": "stdout", "line": str, "ts": float, "event_id": int}
 *     {"type": "stderr", "line": str, "ts": float, "event_id": int}
 *     {"type": "exit",   "code": int, "duration": float, "ts": float, "event_id": int}
 *     {"type": "truncated", "count": int, "ts": float, "event_id": int}
 *
 * We translate those into a friendlier UI-facing shape: ``JobLine`` (one
 * stdout/stderr line with ``kind``) and ``JobStatus`` (status pill +
 * exit_code + duration). Truncation is surfaced as a synthetic
 * ``system`` line so the user knows output was dropped.
 */

import { runCli } from './cli';
import { readCsrfToken, refreshCsrfToken } from './csrf';
import { maybeParseCapacityError } from './errors';
import { fetchWithTimeout } from './_fetchWithTimeout';

// Hard timeout for cancel + restart POSTs. Cancel runs a SIGTERM /
// grace / SIGKILL dance server-side which is bounded at ~5s; 30s
// is generous slack. Restart is a spawn (same envelope as runCli).
// Exported for tests.
export const JOB_MUTATION_TIMEOUT_MS = 30_000;
const JOB_MUTATION_TIMEOUT_MSG =
  `Server didn't respond within ${JOB_MUTATION_TIMEOUT_MS / 1000}s. ` +
  `The dashboard may be wedged - try reloading.`;

export type JobStatusKind =
  | 'queued'
  | 'running'
  | 'complete'
  | 'failed'
  | 'cancelled';

export interface JobLine {
  /** ``stdout``/``stderr`` come from the subprocess; ``system`` is synthesized
   * by this wrapper for transport-level events (truncation, ws errors). */
  kind: 'stdout' | 'stderr' | 'system';
  text: string;
  /** Unix seconds (server clock for stdout/stderr; client clock for system). */
  ts: number;
  /** Server-assigned monotonic id for this event. Present on every event the
   * backend originates (stdout/stderr/truncated/exit); ``undefined`` for
   * client-synthesized ``system`` lines. Callers can pass the highest
   * observed ``event_id`` back as ``subscribeJob({ since })`` on reconnect
   * so the backend skips already-delivered events. */
  event_id?: number;
}

export interface JobStatus {
  id: string;
  status: JobStatusKind;
  /** Joined argv, when known. Populated from ``GET /api/jobs/{id}`` after start. */
  cmd?: string;
  exit_code?: number;
  /** Wallclock seconds, when finished. */
  duration?: number;
}

/** Raw shape of /api/jobs/{id} response (subset we use). */
interface ApiJobRecord {
  id: string;
  cmd?: string[];
  state: string;
  exit_code?: number | null;
  duration?: number | null;
}

function wsUrl(path: string): string {
  const proto =
    typeof window !== 'undefined' && window.location.protocol === 'https:'
      ? 'wss:'
      : 'ws:';
  const host =
    typeof window !== 'undefined' ? window.location.host : '127.0.0.1:7401';
  return `${proto}//${host}${path}`;
}

/**
 * Spawn a CLI command on the server. Thin alias around ``runCli`` so callers
 * don't have to import from two modules.
 */
export async function startJob(
  cliId: string,
  args: Record<string, unknown>,
): Promise<{ job_id: string }> {
  return runCli(cliId, args);
}

/** Cancel a running job (SIGTERM with grace period, then SIGKILL server-side).
 * Self-heals stale CSRF tokens after a server restart by refetching the
 * index, scraping the new token, and retrying once. */
export async function cancelJob(id: string): Promise<{ ok: boolean }> {
  const url = `/api/jobs/${encodeURIComponent(id)}/cancel`;
  const send = async (token: string | null): Promise<Response> => {
    const headers: Record<string, string> = {};
    if (token) headers['X-Workbench-Token'] = token;
    return fetchWithTimeout(
      url,
      { method: 'POST', headers },
      JOB_MUTATION_TIMEOUT_MS,
      JOB_MUTATION_TIMEOUT_MSG,
    );
  };
  let res = await send(readCsrfToken());
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) res = await send(fresh);
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST /api/jobs/${id}/cancel ${res.status}: ${body}`);
  }
  return { ok: true };
}

/** Subset of `/api/jobs/stats` that the FE actually consumes. The
 * server returns more (total, by_status, total_stderr,
 * recent_failures); we type only the fields we render so unrelated
 * additions don't force frontend-side changes. `max_concurrent=0`
 * means the cap is disabled (server tells us so explicitly). */
export interface JobsCapacity {
  running: number;
  max_concurrent: number;
}

/** Fetch `/api/jobs/stats` and return only the capacity slice.
 *
 * Returns null on any error (network, 500, malformed body). The chip
 * caller treats null as "hide the chip" - capacity is a nice-to-have
 * status indicator, never load-bearing, so we never want a stats
 * fetch failure to surface as a user-visible error. */
export async function fetchJobsCapacity(): Promise<JobsCapacity | null> {
  try {
    const res = await fetch('/api/jobs/stats', {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) return null;
    const body = (await res.json()) as Record<string, unknown>;
    const running = body.running;
    const maxConcurrent = body.max_concurrent;
    if (typeof running !== 'number' || typeof maxConcurrent !== 'number') {
      return null;
    }
    return { running, max_concurrent: maxConcurrent };
  } catch {
    return null;
  }
}

/** Re-spawn a finished job using its original `cli_id` + `args`.
 *
 * Calls `POST /api/jobs/{id}/restart`; the server looks up the source's
 * stored metadata, rebuilds argv via `args_to_argv`, and spawns a fresh
 * subprocess. Returns the new `job_id`.
 *
 * Self-heals stale CSRF tokens after a server restart (matches the
 * `cancelJob` retry pattern). 404 if the source is unknown or its
 * `cli_id` is no longer in the catalog; 409 if the source is still
 * running or has no `cli_id`. */
export async function restartJob(id: string): Promise<{ job_id: string }> {
  const url = `/api/jobs/${encodeURIComponent(id)}/restart`;
  const send = async (token: string | null): Promise<Response> => {
    const headers: Record<string, string> = {};
    if (token) headers['X-Workbench-Token'] = token;
    return fetchWithTimeout(
      url,
      { method: 'POST', headers },
      JOB_MUTATION_TIMEOUT_MS,
      JOB_MUTATION_TIMEOUT_MSG,
    );
  };
  let res = await send(readCsrfToken());
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) res = await send(fresh);
  }
  if (!res.ok) {
    const cap = await maybeParseCapacityError(res);
    if (cap) throw cap;
    const body = await res.text();
    throw new Error(`POST /api/jobs/${id}/restart ${res.status}: ${body}`);
  }
  return (await res.json()) as { job_id: string };
}

/** Best-effort fetch of a job's static metadata (cmd, state). */
export async function getJob(id: string): Promise<ApiJobRecord | null> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(id)}`, {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) return null;
    return (await res.json()) as ApiJobRecord;
  } catch {
    return null;
  }
}

/** Outcome of a `fetchJobStatus` call. `notFound` means the job-record was
 * evicted from the backend's in-memory store (the only place jobs live).
 * Callers should treat that as a terminal "we lost track" signal and patch
 * their local cache to reflect that. */
export type JobStatusFetch =
  | { kind: 'found'; status: JobStatusKind; exit_code?: number | null; duration?: number | null; cmd?: string }
  | { kind: 'notFound' }
  | { kind: 'error' };

/** Map the backend's `state` string to our normalized `JobStatusKind`. */
function normalizeJobState(state: string | undefined): JobStatusKind {
  switch (state) {
    case 'queued':
      return 'queued';
    case 'running':
      return 'running';
    case 'complete':
    case 'completed':
    case 'success':
      return 'complete';
    case 'failed':
    case 'error':
      return 'failed';
    case 'cancelled':
    case 'canceled':
      return 'cancelled';
    default:
      // Unknown / future states get bucketed into `failed` so the UI surfaces
      // them rather than silently treating them as success.
      return 'failed';
  }
}

/**
 * Fetch a single job's snapshot and return a normalized status. Distinguishes
 * `notFound` (404 - backend evicted the record) from `error` (network /
 * parse) so the UI can react differently (the former is terminal, the latter
 * is transient and worth retrying).
 */
export async function fetchJobStatus(id: string): Promise<JobStatusFetch> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(id)}`, {
      headers: { Accept: 'application/json' },
    });
    if (res.status === 404) return { kind: 'notFound' };
    if (!res.ok) return { kind: 'error' };
    const body = (await res.json()) as ApiJobRecord;
    return {
      kind: 'found',
      status: normalizeJobState(body.state),
      exit_code: body.exit_code ?? undefined,
      duration: body.duration ?? undefined,
      cmd: Array.isArray(body.cmd) ? body.cmd.join(' ') : undefined,
    };
  } catch {
    return { kind: 'error' };
  }
}

interface SubscribeHandlers {
  onLine: (line: JobLine) => void;
  onStatus: (status: JobStatus) => void;
  onError?: (err: Event | Error) => void;
  onClose?: (ev: CloseEvent) => void;
  /** Fired when the wrapper has scheduled a reconnect attempt after an
   * unexpected close. ``attempt`` is 1-indexed: 1 on the first retry,
   * 2 on the second, etc. Consumers typically debounce this signal
   * (~1.5s) before showing a "Reconnecting" indicator so brief blips
   * we recover from quickly stay invisible. */
  onReconnecting?: (attempt: number) => void;
  /** Fired when a reconnect attempt successfully connected and the
   * stream is live again. Consumers should clear any "Reconnecting"
   * UI surfaced from a prior onReconnecting. Not fired on the very
   * first connect (use onStatus 'running' for that). */
  onReconnected?: () => void;
}

interface SubscribeOptions {
  /** When set, ask the backend to replay only events with ``event_id > since``.
   * Use the highest ``event_id`` observed on the prior connection to avoid
   * re-delivering lines the caller already has. Omit on first subscribe. */
  since?: number;
}

/** Maximum reconnect attempts before giving up and surfacing onError.
 * 1s + 2s + 4s + 8s + 8s ≈ 23s of "trying" before declaring the
 * stream lost. Tuned to outlast a typical dev-server restart (~3s)
 * but not exceed reasonable user patience. */
const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Open a ``/ws/jobs/{id}`` subscription. Returns a ``{close()}`` handle that
 * tears the socket down. The wrapper translates raw backend events into
 * ``JobLine`` and ``JobStatus`` callbacks so callers don't have to know the
 * wire shape. Handlers are best-effort: any throws are swallowed to keep
 * the socket alive.
 *
 * Auto-reconnect: if the WS closes WITHOUT having delivered an ``exit``
 * event (network blip, dev-server restart, etc.) and the caller has not
 * invoked ``close()``, we transparently reconnect with exponential
 * backoff (1s, 2s, 4s, 8s, 8s; up to MAX_RECONNECT_ATTEMPTS) plus
 * [0, 500ms) jitter so multiple tabs/jobs don't reconnect in
 * lockstep on server recovery (thundering herd). Each reconnect
 * passes ``?since=lastSeenEventId`` so the backend replays ONLY
 * events the caller has not yet seen. The "running" onStatus pill fires on first
 * connect only - reconnects do not flip status back to running after a
 * cancel/exit was painted from a prior event.
 *
 * onClose is suppressed for intermediate disconnects we recover from,
 * so consumers don't paint a "Lost connection" toast for every blip.
 * It fires once if the user invokes ``close()``, once if the server
 * delivers ``exit``, or once if reconnect attempts are exhausted.
 *
 * Pass ``options.since`` to start the FIRST connect at a cursor; the
 * backend honors ``?since=N`` and replays only events with ``event_id > N``.
 */
export function subscribeJob(
  id: string,
  handlers: SubscribeHandlers,
  options?: SubscribeOptions,
): { close: () => void } {
  let lastSeen: number | undefined =
    options?.since != null && Number.isFinite(options.since)
      ? options.since
      : undefined;
  let userClosed = false; // caller invoked the returned close handle
  let gotExit = false; // server delivered an exit event - job is over
  let attempts = 0; // failed-reconnect counter; resets on a successful open
  let firstOpen = true; // emit the "running" pill on first connect only
  let reconnectTimer: number | null = null;
  let ws: WebSocket | null = null;

  function buildUrl(): string {
    const qs =
      lastSeen != null && Number.isFinite(lastSeen)
        ? `?since=${encodeURIComponent(String(lastSeen))}`
        : '';
    return wsUrl(`/ws/jobs/${encodeURIComponent(id)}${qs}`);
  }

  function scheduleReconnect(): void {
    if (userClosed || gotExit) return;
    if (attempts >= MAX_RECONNECT_ATTEMPTS) {
      safeCall(
        handlers.onError,
        new Error(
          `Lost job stream after ${MAX_RECONNECT_ATTEMPTS} reconnect attempts`,
        ),
      );
      return;
    }
    // Exponential backoff with jitter. Cap at 8s (per-job streams
    // benefit from faster reconnect than the v2 events stream;
    // user is likely watching live output). Jitter [0, 500ms)
    // spreads reconnect across concurrent tabs/jobs so the
    // server doesn't see a thundering herd after a blip.
    const exp = Math.min(8000, 1000 * 2 ** attempts);
    const backoffMs = exp + Math.random() * 500;
    attempts += 1;
    // Notify consumers that we are about to retry. attempt is 1-indexed
    // so the first retry fires onReconnecting(1). Inline the try/catch
    // rather than route through safeCall<T> because the helper's
    // generic signature does not handle a callback with a non-trivial
    // argument cleanly here.
    try {
      handlers.onReconnecting?.(attempts);
    } catch {
      // ignore - keep the reconnect path alive
    }
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      open();
    }, backoffMs);
  }

  function open(): void {
    if (userClosed || gotExit) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(buildUrl());
    } catch (err) {
      safeCall(
        handlers.onError,
        err instanceof Error ? err : new Error(String(err)),
      );
      scheduleReconnect();
      return;
    }
    ws = socket;

    socket.addEventListener('open', () => {
      const wasReconnecting = attempts > 0;
      attempts = 0;
      if (firstOpen) {
        firstOpen = false;
        safeCall(handlers.onStatus, { id, status: 'running' });
      } else if (wasReconnecting) {
        // We had at least one failed attempt before this success, so
        // the consumer may have surfaced a "Reconnecting" pill via
        // onReconnecting. Tell them the stream is live again.
        try {
          handlers.onReconnected?.();
        } catch {
          // ignore
        }
      }
    });

    socket.addEventListener('message', (ev) => {
      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(ev.data as string) as Record<string, unknown>;
      } catch {
        return;
      }
      const t = evt.type as string | undefined;
      const eid =
        typeof evt.event_id === 'number' ? (evt.event_id as number) : undefined;
      // Track high-water mark for resume-on-reconnect. event_id is
      // server-monotonic per the backend wire spec.
      if (eid != null && (lastSeen == null || eid > lastSeen)) {
        lastSeen = eid;
      }
      if (t === 'stdout' || t === 'stderr') {
        const text = (evt.line as string | undefined) ?? '';
        const ts = (evt.ts as number | undefined) ?? Date.now() / 1000;
        safeCall(handlers.onLine, { kind: t, text, ts, event_id: eid });
      } else if (t === 'truncated') {
        const count = (evt.count as number | undefined) ?? 0;
        safeCall(handlers.onLine, {
          kind: 'system',
          text: `[server dropped ${count} earlier lines]`,
          ts: (evt.ts as number | undefined) ?? Date.now() / 1000,
          event_id: eid,
        });
      } else if (t === 'exit') {
        gotExit = true;
        const code = (evt.code as number | undefined) ?? -1;
        const duration = evt.duration as number | undefined;
        safeCall(handlers.onStatus, {
          id,
          status: code === 0 ? 'complete' : 'failed',
          exit_code: code,
          duration,
        });
      }
    });

    socket.addEventListener('error', (e) => safeCall(handlers.onError, e));
    socket.addEventListener('close', (e) => {
      // Only surface onClose to the caller if this is a TERMINAL close
      // (user-initiated, server delivered exit). Intermediate blips we
      // are about to recover from stay invisible to the consumer.
      if (userClosed || gotExit) {
        safeCall(handlers.onClose, e);
        return;
      }
      scheduleReconnect();
    });
  }

  open();

  return {
    close: () => {
      userClosed = true;
      if (reconnectTimer != null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      try {
        ws?.close();
      } catch {
        // ignore
      }
    },
  };
}

function safeCall<T>(fn: ((arg: T) => void) | undefined, arg: T): void {
  if (!fn) return;
  try {
    fn(arg);
  } catch {
    // swallow - keep the socket alive
  }
}
