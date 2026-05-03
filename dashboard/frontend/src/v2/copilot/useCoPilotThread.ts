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

  const handleEvent = useCallback((evt: AgentWsEvent) => {
    if (evt.type === 'text_delta') {
      setMessages((prev) => {
        const { next } = findOrCreateAgentBubble(prev, evt.message_id);
        return patchAgentBubble(next, evt.message_id, (b) => ({
          ...b,
          text: (b.text ?? '') + evt.text,
          streaming: true,
        }));
      });
      setStatus('streaming');
      return;
    }

    if (evt.type === 'tool_call_proposal' || evt.type === 'tool_call_running') {
      setMessages((prev) => {
        const { next, bubble } = ensureAgentBubble(prev);
        const existing = bubble.tools.find((t) => t.tool_call_id === evt.tool_call_id);
        const cmd =
          evt.type === 'tool_call_proposal'
            ? `${evt.tool} ${formatArgsShort(evt.args)}`.trim()
            : existing?.cmd ?? evt.tool;
        const newTool: ToolBlockEntry = existing
          ? { ...existing, status: evt.type === 'tool_call_running' ? 'running' : existing.status }
          : {
              tool_call_id: evt.tool_call_id,
              cmd,
              status: evt.type === 'tool_call_running' ? 'running' : 'proposed',
              duration_s: null,
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
        const tools = bubble.tools.map((t) =>
          t.tool_call_id === evt.tool_call_id
            ? {
                ...t,
                status: evt.ok ? 'complete' : 'error',
                duration_s: t.duration_s,
                output_preview: out.slice(0, 200),
              }
            : t,
        ) as ToolBlockEntry[];
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
        return patchAgentBubble(next, bubble.id, (b) => ({ ...b, pending_confirm: conf }));
      });
      return;
    }

    if (evt.type === 'final') {
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
  }, []);

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
    setMessages([]);
    setPending(null);
    setStatus('idle');
    setError(null);
    setThreadId(newThreadId);
  }, []);

  return { threadId, messages, pending, status, error, send, confirm, resetTo };
}

function formatArgsShort(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => {
      const s = typeof v === 'string' ? v : JSON.stringify(v);
      return `--${k} ${s.length > 30 ? s.slice(0, 27) + '...' : s}`;
    })
    .join(' ');
}
