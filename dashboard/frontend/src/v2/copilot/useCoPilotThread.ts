/**
 * useCoPilotThread - lifecycle hook for one co-pilot conversation.
 *
 * Wraps the existing /api/agent/chat + /ws/agent/{tid} endpoints (no
 * backend changes). Returns:
 *   - messages: synthesized timeline of you/agent bubbles + tool blocks
 *   - pending: when set, an awaiting-confirmation card needs the user
 *   - send: append a user turn and stream the agent reply
 *   - confirm: approve / reject a pending tool call
 *   - resetTo: switch the hook to a different (existing) thread id
 *   - threadId: current thread id (stable across resets)
 *   - status: idle | streaming | awaiting_confirmation | error
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, subscribeAgent } from '../../api';
import type { AgentWsEvent, ConvMessage, PendingConfirmation, ToolBlockEntry } from './types';

type Status = 'idle' | 'streaming' | 'awaiting_confirmation' | 'error';

interface UseCoPilotOptions {
  /** When provided, the hook attaches to this existing thread instead of creating one on first send. */
  initialThreadId?: string | null;
}

/** Char limits for captured tool output. Backend caps stdout at MAX_TOOL_OUTPUT
 * in api/threads.py (4000 chars); we match it here. */
const MAX_OUTPUT_PREVIEW = 300;
const MAX_OUTPUT_FULL = 4000;

let _msgCounter = 0;
const newMsgId = (prefix: string) => `${prefix}-${Date.now()}-${++_msgCounter}`;

function findOrCreateAgentBubble(messages: ConvMessage[], messageId: string): {
  next: ConvMessage[];
  bubble: ConvMessage;
} {
  const idx = messages.findIndex((m) => m.id === messageId);
  if (idx >= 0) {
    return { next: messages, bubble: messages[idx] };
  }
  const bubble: ConvMessage = {
    id: messageId,
    role: 'agent',
    text: '',
    streaming: true,
    tools: [],
    pending_confirm: null,
    ts: Date.now() / 1000,
  };
  return { next: [...messages, bubble], bubble };
}

function lastAgentBubble(messages: ConvMessage[]): ConvMessage | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'agent') return messages[i];
  }
  return null;
}

function patchAgentBubble(
  messages: ConvMessage[],
  bubbleId: string,
  patch: (b: ConvMessage) => ConvMessage,
): ConvMessage[] {
  return messages.map((m) => (m.id === bubbleId ? patch(m) : m));
}

function ensureAgentBubble(messages: ConvMessage[]): {
  next: ConvMessage[];
  bubble: ConvMessage;
} {
  const last = lastAgentBubble(messages);
  if (last && last.streaming) return { next: messages, bubble: last };
  const bubble: ConvMessage = {
    id: newMsgId('a'),
    role: 'agent',
    text: '',
    streaming: true,
    tools: [],
    pending_confirm: null,
    ts: Date.now() / 1000,
  };
  return { next: [...messages, bubble], bubble };
}

export function useCoPilotThread(opts: UseCoPilotOptions = {}) {
  const [threadId, setThreadId] = useState<string | null>(opts.initialThreadId ?? null);
  const [messages, setMessages] = useState<ConvMessage[]>([]);
  const [pending, setPending] = useState<PendingConfirmation | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<{ close: () => void } | null>(null);
  const threadIdRef = useRef<string | null>(threadId);
  threadIdRef.current = threadId;

  // text_delta arrives once per token. On a fast-streaming model
  // (Sonnet ~80 tok/sec, Haiku faster) that becomes 80+ setMessages calls
  // per second, each rebuilding the messages array, the agent bubble,
  // and concatenating the bubble text. We coalesce deltas per message_id
  // into a Map ref and flush once per animation frame, so the chat view
  // re-renders at most ~60Hz regardless of token rate. Other event kinds
  // (tool_call_*, final, error) are infrequent and pass through directly.
  const textBufferRef = useRef<Map<string, string>>(new Map());
  const flushScheduledRef = useRef<number | null>(null);

  const flushTextBuffer = useCallback(() => {
    if (flushScheduledRef.current != null) {
      if (typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(flushScheduledRef.current);
      } else {
        window.clearTimeout(flushScheduledRef.current);
      }
      flushScheduledRef.current = null;
    }
    const buf = textBufferRef.current;
    if (buf.size === 0) return;
    const pending = new Map(buf);
    buf.clear();
    setMessages((prev) => {
      let arr = prev;
      for (const [messageId, deltaText] of pending) {
        const { next } = findOrCreateAgentBubble(arr, messageId);
        arr = patchAgentBubble(next, messageId, (b) => ({
          ...b,
          text: (b.text ?? '') + deltaText,
          streaming: true,
        }));
      }
      return arr;
    });
  }, []);

  const enqueueTextDelta = useCallback(
    (messageId: string, text: string) => {
      const buf = textBufferRef.current;
      buf.set(messageId, (buf.get(messageId) ?? '') + text);
      if (flushScheduledRef.current != null) return;
      if (typeof requestAnimationFrame === 'function') {
        flushScheduledRef.current = requestAnimationFrame(flushTextBuffer);
      } else {
        // jsdom / SSR fallback.
        flushScheduledRef.current = window.setTimeout(flushTextBuffer, 16);
      }
    },
    [flushTextBuffer],
  );

  const handleEvent = useCallback((evt: AgentWsEvent) => {
    if (evt.type === 'text_delta') {
      enqueueTextDelta(evt.message_id, evt.text);
      // setStatus with the same value is a no-op in React 18+, so this is
      // effectively a single render-trigger per status transition rather
      // than per token. Cheap to leave inline.
      setStatus('streaming');
      return;
    }

    if (evt.type === 'tool_call_proposal') {
      setMessages((prev) => {
        const { next, bubble } = ensureAgentBubble(prev);
        const existing = bubble.tools.find((t) => t.tool_call_id === evt.tool_call_id);
        const cmd = evt.preview_cmd ?? `${evt.tool} ${formatArgsShort(evt.args)}`.trim();
        const newTool: ToolBlockEntry = existing
          ? {
              ...existing,
              tool: existing.tool || evt.tool,
              cmd: existing.cmd || cmd,
              args: Object.keys(existing.args).length > 0 ? existing.args : evt.args,
              ts_started: existing.ts_started ?? evt.ts,
            }
          : {
              tool_call_id: evt.tool_call_id,
              tool: evt.tool,
              cmd,
              args: evt.args,
              status: 'proposed',
              duration_s: null,
              output_preview: '',
              output_full: '',
              exit_code: null,
              ts_started: evt.ts,
              ts_completed: null,
            };
        const tools = existing
          ? bubble.tools.map((t) => (t.tool_call_id === evt.tool_call_id ? newTool : t))
          : [...bubble.tools, newTool];
        return patchAgentBubble(next, bubble.id, (b) => ({ ...b, tools }));
      });
      return;
    }

    if (evt.type === 'tool_call_running') {
      setMessages((prev) => {
        const { next, bubble } = ensureAgentBubble(prev);
        const existing = bubble.tools.find((t) => t.tool_call_id === evt.tool_call_id);
        const newTool: ToolBlockEntry = existing
          ? {
              ...existing,
              status: 'running',
              ts_started: evt.ts,
            }
          : {
              tool_call_id: evt.tool_call_id,
              tool: evt.tool,
              cmd: evt.tool,
              args: {},
              status: 'running',
              duration_s: null,
              output_preview: '',
              output_full: '',
              exit_code: null,
              ts_started: evt.ts,
              ts_completed: null,
            };
        const tools = existing
          ? bubble.tools.map((t) => (t.tool_call_id === evt.tool_call_id ? newTool : t))
          : [...bubble.tools, newTool];
        return patchAgentBubble(next, bubble.id, (b) => ({ ...b, tools }));
      });
      return;
    }

    if (evt.type === 'tool_call_complete') {
      const out = evt.output ?? evt.stdout ?? '';
      setMessages((prev) => {
        const { next, bubble } = ensureAgentBubble(prev);
        const tools = bubble.tools.map((t) => {
          if (t.tool_call_id !== evt.tool_call_id) return t;
          const startTs = t.ts_started;
          const duration = startTs != null ? Math.max(0, evt.ts - startTs) : null;
          return {
            ...t,
            status: evt.ok ? 'complete' : 'error',
            duration_s: duration,
            output_preview: out.slice(0, MAX_OUTPUT_PREVIEW),
            output_full: out.slice(0, MAX_OUTPUT_FULL),
            exit_code: evt.exit_code ?? null,
            ts_completed: evt.ts,
          };
        }) as ToolBlockEntry[];
        return patchAgentBubble(next, bubble.id, (b) => ({ ...b, tools }));
      });
      return;
    }

    if (evt.type === 'confirmation_required') {
      const conf: PendingConfirmation = {
        tool_call_id: evt.tool_call_id,
        tool: evt.tool,
        preview_cmd: evt.preview_cmd,
        args: evt.args,
        side_effects: evt.side_effects,
      };
      setPending(conf);
      setStatus('awaiting_confirmation');
      setMessages((prev) => {
        const { next, bubble } = ensureAgentBubble(prev);
        const existing = bubble.tools.find((t) => t.tool_call_id === evt.tool_call_id);
        const entry: ToolBlockEntry = existing
          ? { ...existing, status: 'awaiting_confirmation', args: existing.args ?? evt.args }
          : {
              tool_call_id: evt.tool_call_id,
              tool: evt.tool,
              cmd: evt.preview_cmd,
              args: evt.args,
              status: 'awaiting_confirmation',
              duration_s: null,
              output_preview: '',
              output_full: '',
              exit_code: null,
              ts_started: evt.ts,
              ts_completed: null,
            };
        const tools = existing
          ? bubble.tools.map((t) => (t.tool_call_id === evt.tool_call_id ? entry : t))
          : [...bubble.tools, entry];
        return patchAgentBubble(next, bubble.id, (b) => ({ ...b, tools, pending_confirm: conf }));
      });
      return;
    }

    if (evt.type === 'final') {
      // Drain any pending text-delta batch synchronously so the bubble's
      // accumulated text is committed BEFORE the final patch runs.
      // Without this, a final event arriving between two rAF flushes
      // would patch the bubble while it still missed the last few tokens,
      // causing visible text to "rewind" when final.text == undefined.
      flushTextBuffer();
      // agent.py emits 'final' WITHOUT message_id (see emit sites at lines
      // 911, 923, 949, 975, 989, 1018). Fall back to the last streaming
      // bubble so the typing indicator clears.
      setMessages((prev) => {
        if (evt.message_id) {
          const { next } = findOrCreateAgentBubble(prev, evt.message_id);
          return patchAgentBubble(next, evt.message_id, (b) => ({
            ...b,
            text: evt.text ?? b.text,
            streaming: false,
          }));
        }
        const last = lastAgentBubble(prev);
        if (!last) return prev;
        return patchAgentBubble(prev, last.id, (b) => ({
          ...b,
          text: evt.text ?? b.text,
          streaming: false,
        }));
      });
      setStatus((s) => (s === 'awaiting_confirmation' ? 'awaiting_confirmation' : 'idle'));
      return;
    }

    if (evt.type === 'error') {
      setError(evt.message);
      setStatus('error');
    }
  }, [enqueueTextDelta, flushTextBuffer]);

  // Open / re-open the WS whenever threadId changes.
  useEffect(() => {
    if (!threadId) return;
    const sub = subscribeAgent<AgentWsEvent>(threadId, {
      onMessage: handleEvent,
      onError: (evt) => {
        console.error('agent ws error', evt);
        setError(`WebSocket error on /ws/agent/${threadId}`);
      },
      onClose: (evt) => {
        if (!evt.wasClean) {
          console.error('agent ws closed unexpectedly', { code: evt.code, reason: evt.reason });
          setError(`WebSocket closed (code ${evt.code}${evt.reason ? `: ${evt.reason}` : ''})`);
        }
      },
    });
    wsRef.current = sub;
    return () => {
      sub.close();
      wsRef.current = null;
      // Cancel any pending text-delta flush so we do not setMessages
      // after the hook has detached from this thread (or unmounted).
      if (flushScheduledRef.current != null) {
        if (typeof cancelAnimationFrame === 'function') {
          cancelAnimationFrame(flushScheduledRef.current);
        } else {
          window.clearTimeout(flushScheduledRef.current);
        }
        flushScheduledRef.current = null;
      }
      textBufferRef.current.clear();
    };
  }, [threadId, handleEvent]);

  const send = useCallback(
    async (message: string) => {
      const userMsg: ConvMessage = {
        id: newMsgId('u'),
        role: 'you',
        text: message,
        streaming: false,
        tools: [],
        pending_confirm: null,
        ts: Date.now() / 1000,
      };
      setMessages((prev) => [...prev, userMsg]);
      // Clear any stale error from a previous failed turn so the user
      // doesn't see the old fail banner alongside their fresh message.
      // If THIS turn fails, the catch below sets a new error.
      setError(null);
      setStatus('streaming');
      try {
        const tid = threadIdRef.current;
        const res = tid
          ? await api.sendAgentMessage(tid, message)
          : await api.startAgentThread(message);
        if (res.thread_id !== threadIdRef.current) {
          setThreadId(res.thread_id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setStatus('error');
      }
    },
    [],
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const confirm = useCallback(
    async (approve: boolean) => {
      const tid = threadIdRef.current;
      if (!tid || !pending) return;
      try {
        await api.confirmAgentTool(tid, approve, pending.tool_call_id);
        setPending(null);
        setStatus('streaming');
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setStatus('error');
      }
    },
    [pending],
  );

  const resetTo = useCallback((newThreadId: string | null) => {
    // Drop any pending text deltas from the previous thread so they do not
    // bleed into the new conversation when the next rAF fires.
    if (flushScheduledRef.current != null) {
      if (typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(flushScheduledRef.current);
      } else {
        window.clearTimeout(flushScheduledRef.current);
      }
      flushScheduledRef.current = null;
    }
    textBufferRef.current.clear();
    setMessages([]);
    setPending(null);
    setStatus('idle');
    setError(null);
    setThreadId(newThreadId);
  }, []);

  return { threadId, messages, pending, status, error, send, confirm, resetTo, clearError };
}

function formatArgsShort(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => {
      const s = typeof v === 'string' ? v : JSON.stringify(v);
      return `--${k} ${s.length > 30 ? s.slice(0, 27) + '...' : s}`;
    })
    .join(' ');
}
