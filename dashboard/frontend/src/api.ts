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
  onError?: (err: Event | Error) => void;
  onClose?: (ev: CloseEvent) => void;
  onOpen?: () => void;
}

interface SubscribeAgentOptions {
  /** First-connect cursor. Backend honors ``?since=N`` and replays only
   * events with ``event_id > N``. Pass when restarting from a known
   * state; auto-reconnect tracks lastSeenEventId itself afterwards. */
  since?: number;
}

/** Same ceiling as subscribeJob's reconnect ladder.
 *  1s + 2s + 4s + 8s + 8s ≈ 23s before declaring the chat stream lost. */
const MAX_AGENT_RECONNECT_ATTEMPTS = 5;

/**
 * Open a ``/ws/agent/{thread_id}`` subscription. Returns a ``{close()}`` handle
 * that tears the socket down. Auto-reconnects on unexpected close: if the WS
 * drops WITHOUT having delivered a ``final`` event and the caller has not
 * invoked ``close()``, retries with exponential backoff (1s → 2s → 4s → 8s
 * → 8s, max 5 attempts) and forwards ``?since=lastSeenEventId`` so the
 * backend skips events the consumer already saw.
 *
 * onOpen fires on first connect only - reconnects do not double-fire it,
 * mirroring subscribeJob's "running" pill suppression.
 *
 * onClose is suppressed for intermediate disconnects we recover from. It
 * fires once on terminal close (user-initiated, server-delivered ``final``,
 * or attempts exhausted). After 5 failed attempts, onError fires with a
 * clear "Lost agent stream after 5 reconnect attempts" message.
 *
 * Existing callers benefit transparently - no API change.
 */
export function subscribeAgent<T>(
  threadId: string,
  handlers: WsHandlers<T>,
  options?: SubscribeAgentOptions,
): { close: () => void } {
  let lastSeen: number | undefined =
    options?.since != null && Number.isFinite(options.since)
      ? options.since
      : undefined;
  let userClosed = false;
  let gotFinal = false; // server delivered a 'final' event - thread done
  let attempts = 0;
  let firstOpen = true;
  let reconnectTimer: number | null = null;
  let ws: WebSocket | null = null;

  function buildUrl(): string {
    const qs =
      lastSeen != null && Number.isFinite(lastSeen)
        ? `?since=${encodeURIComponent(String(lastSeen))}`
        : '';
    return wsUrl(`/ws/agent/${encodeURIComponent(threadId)}${qs}`);
  }

  function scheduleReconnect(): void {
    if (userClosed || gotFinal) return;
    if (attempts >= MAX_AGENT_RECONNECT_ATTEMPTS) {
      handlers.onError?.(
        new Error(
          `Lost agent stream after ${MAX_AGENT_RECONNECT_ATTEMPTS} reconnect attempts`,
        ),
      );
      return;
    }
    const backoffMs = Math.min(8000, 1000 * 2 ** attempts);
    attempts += 1;
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      open();
    }, backoffMs);
  }

  function open(): void {
    if (userClosed || gotFinal) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(buildUrl());
    } catch (err) {
      handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
      scheduleReconnect();
      return;
    }
    ws = socket;

    socket.addEventListener('open', () => {
      attempts = 0;
      if (firstOpen) {
        firstOpen = false;
        handlers.onOpen?.();
      }
    });

    socket.addEventListener('message', (ev) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data as string);
      } catch (err) {
        console.error('agent ws json parse error', err, ev.data);
        return;
      }
      // Track high-water event_id for reconnect resume. The agent wire
      // protocol stamps event_id on every event emitted by AgentRuntime.
      if (parsed && typeof parsed === 'object') {
        const obj = parsed as { event_id?: unknown; type?: unknown };
        if (typeof obj.event_id === 'number') {
          if (lastSeen == null || obj.event_id > lastSeen) {
            lastSeen = obj.event_id;
          }
        }
        if (obj.type === 'final') {
          gotFinal = true;
        }
      }
      try {
        handlers.onMessage(parsed as T);
      } catch (err) {
        // Mirror subscribeJob's safeCall: don't let a consumer throw kill the socket.
        console.error('agent ws onMessage handler threw', err);
      }
    });

    socket.addEventListener('error', (e) => handlers.onError?.(e));
    socket.addEventListener('close', (e) => {
      // Only surface onClose if this is a TERMINAL close. Intermediate
      // blips we recover from stay invisible to the consumer so the
      // chat panel does not flash "disconnected" toasts on every blip.
      if (userClosed || gotFinal) {
        handlers.onClose?.(e);
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
