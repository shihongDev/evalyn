/**
 * Minimal /api/agent/* + /ws/agent/{tid} client used by the v2 co-pilot.
 *
 * The full workbench api.ts was deleted with the workbench rewrite; this
 * file keeps just the agent-thread helpers that the co-pilot needs.
 * v2 endpoint fetchers live in src/v2/api/client.ts.
 */

const API_BASE = '/api';

function csrfToken(): string | null {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="workbench-token"]');
  return meta?.content ?? null;
}

function wsUrl(path: string): string {
  const proto =
    typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = typeof window !== 'undefined' ? window.location.host : '127.0.0.1:7401';
  return `${proto}//${host}${path}`;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((init?.headers as Record<string, string> | undefined) ?? {}),
  };
  const tok = csrfToken();
  if (tok && init?.method && init.method !== 'GET') {
    headers['X-Workbench-Token'] = tok;
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return (await res.json()) as T;
}

export const api = {
  startAgentThread: (
    message: string,
    opts: { provider?: string; model?: string } = {},
  ): Promise<{ thread_id: string }> =>
    jsonFetch('/agent/chat', {
      method: 'POST',
      body: JSON.stringify({ message, ...opts }),
    }),

  sendAgentMessage: (
    threadId: string,
    message: string,
  ): Promise<{ thread_id: string }> =>
    jsonFetch('/agent/chat', {
      method: 'POST',
      body: JSON.stringify({ message, thread_id: threadId }),
    }),

  confirmAgentTool: (
    threadId: string,
    approve: boolean,
    toolCallId?: string,
    options?: { argsOverride?: Record<string, unknown>; autoApproveSession?: boolean },
  ): Promise<{ ok: boolean }> =>
    jsonFetch(`/agent/chat/${encodeURIComponent(threadId)}/confirm`, {
      method: 'POST',
      body: JSON.stringify({
        approve,
        tool_call_id: toolCallId,
        ...(options?.argsOverride !== undefined ? { args_override: options.argsOverride } : {}),
        ...(options?.autoApproveSession ? { auto_approve_session: true } : {}),
      }),
    }),
};

export interface WsHandlers<T> {
  onMessage: (msg: T) => void;
  onError?: (err: Event) => void;
  onClose?: (ev: CloseEvent) => void;
  onOpen?: () => void;
}

export function subscribeAgent<T>(
  threadId: string,
  handlers: WsHandlers<T>,
): { close: () => void } {
  const ws = new WebSocket(wsUrl(`/ws/agent/${encodeURIComponent(threadId)}`));
  ws.addEventListener('open', () => handlers.onOpen?.());
  ws.addEventListener('message', (ev) => {
    try {
      handlers.onMessage(JSON.parse(ev.data as string) as T);
    } catch (err) {
      console.error('agent ws json parse error', err, ev.data);
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
