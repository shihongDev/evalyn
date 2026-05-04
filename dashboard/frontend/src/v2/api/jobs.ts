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

function csrfToken(): string | null {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>(
    'meta[name="workbench-token"]',
  );
  return meta?.content ?? null;
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

/** Cancel a running job (SIGTERM with grace period, then SIGKILL server-side). */
export async function cancelJob(id: string): Promise<{ ok: boolean }> {
  const headers: Record<string, string> = {};
  const tok = csrfToken();
  if (tok) headers['X-Workbench-Token'] = tok;
  const res = await fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, {
    method: 'POST',
    headers,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST /api/jobs/${id}/cancel ${res.status}: ${body}`);
  }
  return { ok: true };
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
}

/**
 * Open a ``/ws/jobs/{id}`` subscription. Returns a ``{close()}`` handle that
 * tears the socket down. The wrapper translates raw backend events into
 * ``JobLine`` and ``JobStatus`` callbacks so callers don't have to know the
 * wire shape. Handlers are best-effort: any throws are swallowed to keep
 * the socket alive.
 */
export function subscribeJob(
  id: string,
  handlers: SubscribeHandlers,
): { close: () => void } {
  const ws = new WebSocket(wsUrl(`/ws/jobs/${encodeURIComponent(id)}`));
  // Emit an initial "running" status so the UI flips out of "queued" the
  // moment the socket connects (the backend already started the process
  // before we even subscribed - the WS is just for live tail).
  ws.addEventListener('open', () => {
    safeCall(handlers.onStatus, { id, status: 'running' });
  });
  ws.addEventListener('message', (ev) => {
    let evt: Record<string, unknown>;
    try {
      evt = JSON.parse(ev.data as string) as Record<string, unknown>;
    } catch {
      return;
    }
    const t = evt.type as string | undefined;
    if (t === 'stdout' || t === 'stderr') {
      const text = (evt.line as string | undefined) ?? '';
      const ts = (evt.ts as number | undefined) ?? Date.now() / 1000;
      safeCall(handlers.onLine, { kind: t, text, ts });
    } else if (t === 'truncated') {
      const count = (evt.count as number | undefined) ?? 0;
      safeCall(handlers.onLine, {
        kind: 'system',
        text: `[server dropped ${count} earlier lines]`,
        ts: (evt.ts as number | undefined) ?? Date.now() / 1000,
      });
    } else if (t === 'exit') {
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
  ws.addEventListener('error', (e) => safeCall(handlers.onError, e));
  ws.addEventListener('close', (e) => safeCall(handlers.onClose, e));
  return {
    close: () => {
      try {
        ws.close();
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
