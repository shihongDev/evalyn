/**
 * Thin API + WebSocket wrapper for the Evalyn dashboard.
 *
 * Phase 1 stub: typed signatures and basic fetch/WS helpers. Real endpoints
 * land in Phase 2 (Lane B1). The shape here is what the frontend will call
 * once those endpoints exist.
 */

import type { CliSchema } from './types/catalog';
import type { FileNode, Job, RunMeta } from './types/jobs';
import type { SettingsState } from './types/agent';

/** Base URL for API requests. The dev server proxies `/api` to the FastAPI
 *  backend (see `vite.config.ts`); in production the bundle is served from
 *  the same origin so a relative path is correct. */
const API_BASE = '/api';

/** Build a WebSocket URL for the same origin. Honors `wss://` for HTTPS. */
const wsUrl = (path: string): string => {
  const proto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = typeof window !== 'undefined' ? window.location.host : '127.0.0.1:7401';
  return `${proto}//${host}${path}`;
};

/** Read CSRF token from the `<meta name="workbench-token">` tag injected by the
 *  server (see spec §10). Returns null in test/SSR contexts. */
const csrfToken = (): string | null => {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="workbench-token"]');
  return meta?.content ?? null;
};

/** Generic JSON fetch helper. Throws on non-2xx with the body as the message. */
async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  const token = csrfToken();
  if (token && init?.method && init.method !== 'GET') {
    headers['X-Workbench-Token'] = token;
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return (await res.json()) as T;
}

/* === Endpoints (typed signatures; backed by Lane B1 in Phase 2) ======== */

export const api = {
  /** GET /api/cli — return the introspected catalog. */
  catalog: (): Promise<CliSchema[]> => jsonFetch('/cli'),

  /** GET /api/files/tree — return the workspace file tree. */
  fileTree: (): Promise<FileNode[]> => jsonFetch('/files/tree'),

  /** GET /api/runs — return the runs list. */
  runs: (): Promise<RunMeta[]> => jsonFetch('/runs'),

  /** GET /api/settings — provider keys + active provider. */
  settings: (): Promise<SettingsState> => jsonFetch('/settings'),

  /** GET /api/jobs/recent — recent jobs (running + finished). */
  recentJobs: (): Promise<Job[]> => jsonFetch('/jobs/recent'),

  /** POST /api/cli/run — spawn a CLI subprocess. Returns `{ job_id }`. */
  runCli: (cliId: string, args: Record<string, unknown>): Promise<{ job_id: string }> =>
    jsonFetch('/cli/run', {
      method: 'POST',
      body: JSON.stringify({ cli_id: cliId, args }),
    }),

  /** POST /api/jobs/{id}/cancel — SIGTERM + 3s grace + SIGKILL. */
  cancelJob: (jobId: string): Promise<{ ok: boolean }> =>
    jsonFetch(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
};

/* === WebSocket wrapper ================================================ */

export interface WsHandlers<T> {
  onMessage: (msg: T) => void;
  onError?: (err: Event) => void;
  onClose?: (ev: CloseEvent) => void;
  onOpen?: () => void;
}

/** Open a WebSocket and parse incoming text frames as JSON of type T. The
 *  returned object exposes `close()`; reconnection is the caller's
 *  responsibility (Phase 2 wires this for jobs and the agent). */
export function openWs<T>(path: string, handlers: WsHandlers<T>): { close: () => void } {
  const ws = new WebSocket(wsUrl(path));
  ws.addEventListener('open', () => handlers.onOpen?.());
  ws.addEventListener('message', (ev) => {
    try {
      const data = JSON.parse(ev.data as string) as T;
      handlers.onMessage(data);
    } catch (err) {
      console.error('ws json parse error', err, ev.data);
    }
  });
  if (handlers.onError) ws.addEventListener('error', handlers.onError);
  if (handlers.onClose) ws.addEventListener('close', handlers.onClose);
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

/** Subscribe to a job's stdout/stderr stream. */
export const subscribeJob = <T>(jobId: string, handlers: WsHandlers<T>): { close: () => void } =>
  openWs<T>(`/ws/jobs/${encodeURIComponent(jobId)}`, handlers);

/**
 * Open a raw WebSocket for a job, optionally resuming from a server-assigned
 * `event_id`. Returns the underlying `WebSocket` so callers can manage
 * reconnection (the store's `subscribeJob` slice does exactly that).
 *
 * `factory` is injected so tests can swap in a mock socket — when omitted
 * the global `WebSocket` constructor is used.
 */
export function openJobWs(
  jobId: string,
  options: { sinceEventId?: number | null; factory?: (url: string) => WebSocket } = {},
): WebSocket {
  const { sinceEventId, factory } = options;
  const qs = sinceEventId != null && sinceEventId >= 0 ? `?since=${encodeURIComponent(sinceEventId)}` : '';
  const url = wsUrl(`/ws/jobs/${encodeURIComponent(jobId)}${qs}`);
  const ctor = factory ?? ((u) => new WebSocket(u));
  return ctor(url);
}

/** Subscribe to the agent's chat thread. */
export const subscribeAgent = <T>(threadId: string, handlers: WsHandlers<T>): { close: () => void } =>
  openWs<T>(`/ws/agent/${encodeURIComponent(threadId)}`, handlers);
