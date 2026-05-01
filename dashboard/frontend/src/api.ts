/**
 * Thin API + WebSocket wrapper for the Evalyn dashboard.
 *
 * Phase 1 stub: typed signatures and basic fetch/WS helpers. Real endpoints
 * land in Phase 2 (Lane B1). The shape here is what the frontend will call
 * once those endpoints exist.
 */

import type { CliSchema } from './types/catalog';
import type { FileNode, Job, RunMeta } from './types/jobs';
import type { ProviderState, SettingsState } from './types/agent';

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

/** Read CSRF token from the `<meta name="csrf-token">` tag injected by the
 *  server (see spec §10). Returns null in test/SSR contexts. */
const csrfToken = (): string | null => {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="csrf-token"]');
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
    headers['X-CSRF-Token'] = token;
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

  /* === agent endpoints (Lane C1) ===================================== */

  /** POST /api/agent/chat — start a new agent thread; returns the assigned id. */
  startAgentThread: (
    message: string,
    opts: { provider?: string; model?: string } = {},
  ): Promise<{ thread_id: string }> =>
    jsonFetch('/agent/chat', {
      method: 'POST',
      body: JSON.stringify({ message, ...opts }),
    }),

  /** POST /api/agent/chat/{thread_id} — append a user message to an existing thread. */
  sendAgentMessage: (threadId: string, message: string): Promise<{ ok: boolean }> =>
    jsonFetch(`/agent/chat/${encodeURIComponent(threadId)}`, {
      method: 'POST',
      body: JSON.stringify({ message }),
    }),

  /** POST /api/agent/chat/{thread_id}/confirm — approve or reject a pending tool call. */
  confirmAgentTool: (
    threadId: string,
    approve: boolean,
    toolCallId?: string,
  ): Promise<{ ok: boolean }> =>
    jsonFetch(`/agent/chat/${encodeURIComponent(threadId)}/confirm`, {
      method: 'POST',
      body: JSON.stringify({ approve, tool_call_id: toolCallId }),
    }),

  /* === settings endpoints (Lane C1) ================================== */

  /** GET /api/settings/models/{provider} — list available models for a provider. */
  listProviderModels: (provider: string): Promise<{ models: string[] }> =>
    jsonFetch(`/settings/models/${encodeURIComponent(provider)}`),

  /** POST /api/settings/providers/{provider} — save api key + model selection. */
  saveProvider: (
    provider: string,
    payload: { api_key?: string; model?: string },
  ): Promise<ProviderState> =>
    jsonFetch(`/settings/providers/${encodeURIComponent(provider)}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  /** POST /api/settings/providers/{provider}/test — 1-token connection test. */
  testProvider: (provider: string): Promise<{ ok: boolean; error?: string }> =>
    jsonFetch(`/settings/providers/${encodeURIComponent(provider)}/test`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  /** POST /api/settings/active — set the active provider. */
  setActiveProvider: (provider: string): Promise<{ ok: boolean }> =>
    jsonFetch('/settings/active', {
      method: 'POST',
      body: JSON.stringify({ provider }),
    }),
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

/**
 * Open a raw WebSocket for an agent thread. Mirrors `openJobWs` so the
 * store's reconnect logic can manage the underlying socket directly.
 *
 * `factory` is injected so tests can swap in a mock socket.
 */
export function openAgentWs(
  threadId: string,
  options: { factory?: (url: string) => WebSocket } = {},
): WebSocket {
  const { factory } = options;
  const url = wsUrl(`/ws/agent/${encodeURIComponent(threadId)}`);
  const ctor = factory ?? ((u) => new WebSocket(u));
  return ctor(url);
}
