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

/* === Compare types ==================================================== */

/** Per-run summary block returned by `/api/compare`. */
export interface CompareRunSummary {
  id: string;
  dataset: string;
  /** Pass rate 0..1 (NaN/missing -> null). */
  pass: number | null;
  /** Mean score per metric_id (null if unavailable). */
  metrics: Record<string, number | null>;
  /** Total cost across this run in USD, when available. */
  cost?: number | null;
}

/** Per-row side data: pass flag, per-metric scores, output preview. */
export interface CompareRowSide {
  /** True when every metric on this side passed. Null when no metrics. */
  pass: boolean | null;
  /** Per-metric score (0..1). Missing metrics are absent. */
  metrics: Record<string, number>;
  /** First ~200 chars of output text, when available. */
  output?: string | null;
}

/** Aligned per-input row from `/api/compare`. */
export interface CompareRow {
  /** Stable hash of the canonical input — alignment key. */
  input_hash: string;
  /** First ~200 chars of input. */
  input_preview: string;
  /** Side A (run_a) view. Null when this input only exists in B. */
  a: CompareRowSide | null;
  /** Side B (run_b) view. Null when this input only exists in A. */
  b: CompareRowSide | null;
  /** Sum of (b.metric - a.metric) over shared metrics. Negative = regression. */
  delta: number;
}

/** Full payload returned by `/api/compare`. */
export interface CompareResponse {
  run_a: CompareRunSummary;
  run_b: CompareRunSummary;
  items: CompareRow[];
}

/* === Promote types ===================================================== */

/** Per-row metric verdict surfaced inside the run-detail view. */
export interface RunMetricResult {
  metric_id?: string;
  item_id: string;
  call_id?: string;
  score?: number;
  passed?: boolean;
  details?: { judge?: string; reason?: string } & Record<string, unknown>;
  [k: string]: unknown;
}

/** Shape of `GET /api/runs/{id}` (subset that the dashboard cares about). */
export interface RunDetail {
  id?: string;
  dataset_name?: string;
  metric_results?: RunMetricResult[];
  summary?: { pass_rate?: number } & Record<string, unknown>;
  [k: string]: unknown;
}

/** 200 response from `POST /api/promote/run-failures`. */
export interface PromoteResponse {
  dataset_path: string;
  dataset_name: string;
  item_count: number;
  skipped: string[];
}

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

  /** GET /api/runs/{id} — return parsed results.json for a single run. */
  runDetail: (runId: string): Promise<RunDetail> =>
    jsonFetch(`/runs/${encodeURIComponent(runId)}`),

  /** POST /api/promote/run-failures — promote selected rows to a new dataset. */
  promoteRunFailures: (
    runId: string,
    rowHashes: string[],
    datasetName?: string,
  ): Promise<PromoteResponse> =>
    jsonFetch('/promote/run-failures', {
      method: 'POST',
      body: JSON.stringify({
        run_id: runId,
        row_hashes: rowHashes,
        ...(datasetName ? { dataset_name: datasetName } : {}),
      }),
    }),

  /** GET /api/compare?run_a=&run_b= — return aligned per-row diff data. */
  compare: (runA: string, runB: string): Promise<CompareResponse> =>
    jsonFetch(
      `/compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`,
    ),

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

  /** POST /api/agent/chat with {message, thread_id} — append to existing thread. */
  sendAgentMessage: (threadId: string, message: string): Promise<{ thread_id: string }> =>
    jsonFetch('/agent/chat', {
      method: 'POST',
      body: JSON.stringify({ message, thread_id: threadId }),
    }),

  /** POST /api/agent/chat/{thread_id}/confirm — approve or reject a pending tool call.
   *
   * `options.argsOverride` lets the user inline-edit the agent's argv before
   * approving (P1 spec §5.5). `options.autoApproveSession` adds the tool's
   * name to the per-thread whitelist so subsequent calls bypass the gate. */
  confirmAgentTool: (
    threadId: string,
    approve: boolean,
    toolCallId?: string,
    options?: {
      argsOverride?: Record<string, unknown>;
      autoApproveSession?: boolean;
    },
  ): Promise<{ ok: boolean }> =>
    jsonFetch(`/agent/chat/${encodeURIComponent(threadId)}/confirm`, {
      method: 'POST',
      body: JSON.stringify({
        approve,
        tool_call_id: toolCallId,
        ...(options?.argsOverride !== undefined
          ? { args_override: options.argsOverride }
          : {}),
        ...(options?.autoApproveSession
          ? { auto_approve_session: true }
          : {}),
      }),
    }),

  /* === settings endpoints (Lane C1) ================================== */

  /** GET /api/settings/models/{provider} — list available models for a provider. */
  listProviderModels: (provider: string): Promise<{ models: string[] }> =>
    jsonFetch(`/settings/models/${encodeURIComponent(provider)}`),

  /** POST /api/settings/{provider} — save api key + model selection. */
  saveProvider: (
    provider: string,
    payload: { api_key?: string; model?: string },
  ): Promise<ProviderState> =>
    jsonFetch(`/settings/${encodeURIComponent(provider)}`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  /** POST /api/settings/test/{provider} — 1-token connection test. */
  testProvider: (provider: string): Promise<{ ok: boolean; error?: string }> =>
    jsonFetch(`/settings/test/${encodeURIComponent(provider)}`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  /** POST /api/settings/active — set the active provider. */
  setActiveProvider: (provider: string): Promise<{ ok: boolean }> =>
    jsonFetch('/settings/active', {
      method: 'POST',
      body: JSON.stringify({ provider }),
    }),

  /* === demo endpoints (P0 onboarding) ================================== */

  /** POST /api/demo/load — copy the bundled demo fixture into `.evalyn/`.
   *
   * Returns 200 ``{loaded: true, project: 'research-v1'}`` on success or
   * if the demo was already loaded. Throws on 409 when `.evalyn/`
   * contains non-demo content. */
  demoLoad: (): Promise<{ loaded: boolean; project: string }> =>
    jsonFetch('/demo/load', {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  /* === thread persistence (P2 polish) ================================ */

  /** POST /api/threads/save — write a chat thread to disk as markdown. */
  saveThreadToDisk: (payload: {
    id: string;
    title: string;
    messages: unknown[];
  }): Promise<{ path: string }> =>
    jsonFetch('/threads/save', {
      method: 'POST',
      body: JSON.stringify(payload),
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
