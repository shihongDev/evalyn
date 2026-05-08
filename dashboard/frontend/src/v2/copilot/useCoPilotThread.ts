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
 *   - status: idle | streaming | awaiting_confirmation | reconnecting | error
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, subscribeAgent } from '../../api';
import { errorMessage } from '../api/errors';
import { useV2Store } from '../store/store';
import type { AgentWsEvent, ConvMessage, PendingConfirmation, ToolBlockEntry } from './types';

/**
 * Tools the backend declares but never executes server-side; the frontend
 * acts on the proposal event and the backend emits a synthetic completion
 * so the agent loop can continue. Currently just `start_tour` (drives the
 * Driver.js tour engine via Zustand). Extend this set when adding more
 * UI-driving agent tools (e.g. navigate, focus, openPanel).
 */
const FRONTEND_ONLY_TOOL_NAMES = new Set<string>(['start_tour']);

function dispatchFrontendTool(
  toolName: string,
  args: Record<string, unknown>,
): void {
  if (toolName === 'start_tour') {
    const tourId = typeof args.tour_id === 'string' ? args.tour_id : null;
    if (tourId) {
      useV2Store.getState().setTour(tourId);
    }
  }
}

type Status = 'idle' | 'streaming' | 'awaiting_confirmation' | 'reconnecting' | 'error';

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

/**
 * Locate the bubble that owns a given tool_call_id by scanning every
 * message's tools array. Used by tool_call_running / tool_call_complete:
 * when these events arrive AFTER the bubble has been closed (e.g. agent
 * already emitted `final` and a new bubble was opened), `ensureAgentBubble`
 * would otherwise return the wrong (empty) bubble and the patch would
 * silently no-op, leaving the tool stuck in `running`.
 */
function findBubbleByToolCallId(
  messages: ConvMessage[],
  toolCallId: string,
): ConvMessage | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== 'agent') continue;
    if (m.tools.some((t) => t.tool_call_id === toolCallId)) return m;
  }
  return null;
}

export function useCoPilotThread(opts: UseCoPilotOptions = {}) {
  const [threadId, setThreadId] = useState<string | null>(opts.initialThreadId ?? null);
  const [messages, setMessages] = useState<ConvMessage[]>([]);
  const [pending, setPending] = useState<PendingConfirmation | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  // Debounced "Reconnecting" indicator. The agent WS retries
  // transparently with exp backoff; we surface it to the user only
  // when a disconnect lasts >1.5s, hiding the cosmetic flash for
  // blips we recover from quickly.
  const [reconnecting, setReconnecting] = useState(false);
  const reconnectArmTimerRef = useRef<number | null>(null);

  const wsRef = useRef<{ close: () => void } | null>(null);
  const threadIdRef = useRef<string | null>(threadId);
  threadIdRef.current = threadId;
  // Synchronous in-flight flag for same-frame double-fire
  // protection. React state (`status`) flips to 'streaming' via
  // setStatus, but that's queued - the second click in the same
  // paint frame still sees status='idle' (closure or ref) and
  // sails past the React-state guard. Mutating this ref
  // synchronously inside `send` blocks that race. The
  // CoPilotDock + CoPilotThread submit wrappers gate on the
  // React-state status for the visible-disabled UX; this ref
  // defends against the sub-frame race regardless of caller.
  const sendInFlightRef = useRef(false);
  /**
   * Generation counter for `resetTo`. Captured at the start of `send` and
   * compared after the await: if it advanced, the response belongs to a
   * thread the user has since switched away from and must be discarded
   * so we don't stomp the new thread's id (F4).
   */
  const generationRef = useRef(0);

  const handleReconnecting = useCallback(() => {
    if (reconnectArmTimerRef.current != null) return;
    reconnectArmTimerRef.current = window.setTimeout(() => {
      reconnectArmTimerRef.current = null;
      setReconnecting(true);
    }, 1500);
  }, []);

  const handleReconnected = useCallback(() => {
    if (reconnectArmTimerRef.current != null) {
      window.clearTimeout(reconnectArmTimerRef.current);
      reconnectArmTimerRef.current = null;
    }
    setReconnecting(false);
  }, []);

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
      // Drain pending text deltas first so the bubble that owns the
      // streaming text exists in `messages` before we attach the tool.
      // Without this, ensureAgentBubble spawns a fresh bubble for the
      // tool while a later text-buffer flush spawns ANOTHER bubble for
      // the text - leaving a dangling empty bubble (caught by F1 test).
      flushTextBuffer();
      // Frontend-only tools (start_tour etc.) are dispatched here. The
      // tool block still renders in the conversation timeline so the user
      // sees what the agent decided to do, but the *effect* (e.g. starting
      // a Driver.js tour) happens in the store, not via a backend run.
      if (FRONTEND_ONLY_TOOL_NAMES.has(evt.tool)) {
        dispatchFrontendTool(evt.tool, evt.args);
      }
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
        // Patch the bubble that already owns this tool_call_id (the one
        // emitted by tool_call_proposal). Falling back to ensureAgentBubble
        // would create a fresh bubble if `final` already closed the last
        // one, leaving the proposal entry stuck in `proposed` forever.
        const owning = findBubbleByToolCallId(prev, evt.tool_call_id);
        if (owning) {
          return patchAgentBubble(prev, owning.id, (b) => ({
            ...b,
            tools: b.tools.map((t) =>
              t.tool_call_id === evt.tool_call_id
                ? { ...t, status: 'running', ts_started: evt.ts }
                : t,
            ),
          }));
        }
        const { next, bubble } = ensureAgentBubble(prev);
        const newTool: ToolBlockEntry = {
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
        return patchAgentBubble(next, bubble.id, (b) => ({
          ...b,
          tools: [...b.tools, newTool],
        }));
      });
      return;
    }

    if (evt.type === 'tool_call_complete') {
      const out = evt.output ?? evt.stdout ?? '';
      setMessages((prev) => {
        // Locate the bubble whose tools array already contains this
        // tool_call_id. Without this, a complete event arriving after
        // `final` (which sets the bubble to streaming=false) would land in
        // a freshly-spawned empty bubble and the tool entry would stay
        // `running` forever.
        const owning = findBubbleByToolCallId(prev, evt.tool_call_id);
        const targetId = owning?.id;
        if (!targetId) {
          // No bubble owns this id - nothing to patch. Drop silently
          // rather than spawn a stray bubble.
          return prev;
        }
        return patchAgentBubble(prev, targetId, (b) => ({
          ...b,
          tools: b.tools.map((t) => {
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
          }) as ToolBlockEntry[],
        }));
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

  // Open / re-open the WS whenever threadId changes. Includes exponential
  // backoff reconnect (1s, 2s, 4s, 8s, 16s, capped at 30s) for up to 5
  // consecutive unclean closes. Cancelled on unmount or threadId change.
  useEffect(() => {
    if (!threadId) return;

    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;
    const MAX_ATTEMPTS = 5;
    const MAX_DELAY_MS = 30_000;

    const open = () => {
      if (cancelled) return;
      const sub = subscribeAgent<AgentWsEvent>(threadId, {
        onMessage: (evt) => {
          // Successful traffic resets backoff so a later transient blip
          // gets a fresh 5-attempt budget.
          attempt = 0;
          handleEvent(evt);
        },
        onOpen: () => {
          attempt = 0;
          setStatus((s) => (s === 'reconnecting' ? 'idle' : s));
        },
        onError: (evt) => {
          // Throttle: log only on the FIRST error of this outage
          // window. Subsequent retries inside the same outage flip
          // attempt > 0; successful onOpen / onMessage resets it
          // to 0, so the next outage produces a fresh log line.
          // Without this throttle, a 5-attempt reconnect ladder
          // produced 5+ identical "agent ws error" lines, burying
          // any unique error in operator-side log review.
          if (attempt === 0) {
            console.error('agent ws error', evt);
          }
          setError(`WebSocket error on /ws/agent/${threadId}`);
        },
        onClose: (evt) => {
          if (cancelled) return;
          if (evt.wasClean) {
            wsRef.current = null;
            return;
          }
          // Same throttle rationale as onError above.
          if (attempt === 0) {
            console.error('agent ws closed unexpectedly', {
              code: evt.code,
              reason: evt.reason,
            });
          }
          if (attempt >= MAX_ATTEMPTS) {
            setError(
              `WebSocket closed (code ${evt.code}${evt.reason ? `: ${evt.reason}` : ''}) - giving up after ${MAX_ATTEMPTS} reconnect attempts`,
            );
            setStatus('error');
            return;
          }
          // Exp backoff with [0, 500ms) jitter so the outer
          // (this hook) and inner (subscribeAgent) reconnect
          // ladders don't reconnect in lockstep across tabs on
          // server recovery. Same jitter magnitude as the other
          // three FE-side WS reconnect paths.
          const exp = Math.min(1000 * 2 ** attempt, MAX_DELAY_MS);
          const delay = exp + Math.random() * 500;
          attempt += 1;
          setStatus('reconnecting');
          reconnectTimer = setTimeout(open, delay);
        },
      });
      wsRef.current = sub;
    };

    open();

    return () => {
      cancelled = true;
      if (reconnectTimer != null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      wsRef.current?.close();
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
      // Cancel the reconnect-arm debounce so we do not setReconnecting
      // after the hook has detached from this thread.
      if (reconnectArmTimerRef.current != null) {
        window.clearTimeout(reconnectArmTimerRef.current);
        reconnectArmTimerRef.current = null;
      }
      setReconnecting(false);
    };
  }, [threadId, handleEvent, handleReconnecting, handleReconnected]);

  const send = useCallback(
    async (message: string) => {
      // Same-frame double-fire guard. Synchronous ref mutation
      // makes a second call entered before this one's first await
      // see in-flight=true and bail. Without this, two parallel
      // api.startAgentThread() calls at the start of a fresh
      // thread would create two backend threads, the FE losing
      // one to the setState ordering.
      if (sendInFlightRef.current) return;
      sendInFlightRef.current = true;
      try {
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
        // Capture the generation BEFORE the await so we can detect a
        // resetTo() that landed mid-flight (F4).
        const generationAtCall = generationRef.current;
        try {
          const tid = threadIdRef.current;
          const res = tid
            ? await api.sendAgentMessage(tid, message)
            : await api.startAgentThread(message);
          if (generationRef.current !== generationAtCall) {
            // The user switched threads while this request was in flight.
            // Drop the response - applying setThreadId here would swap the
            // hook back to a stale thread.
            return;
          }
          if (res.thread_id !== threadIdRef.current) {
            setThreadId(res.thread_id);
          }
        } catch (err) {
          if (generationRef.current !== generationAtCall) return;
          setError(errorMessage(err));
          setStatus('error');
        }
      } finally {
        sendInFlightRef.current = false;
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
        setError(errorMessage(err));
        setStatus('error');
        // Without this, the awaiting-confirmation card stays mounted and
        // the composer (which guards on `pending != null`) is locked
        // forever (F2). Clearing pending lets the user retry or send.
        setPending(null);
      }
    },
    [pending],
  );

  const resetTo = useCallback((newThreadId: string | null) => {
    // Bump the generation so any in-flight `send` resolves into a no-op
    // instead of stomping the new thread id (F4).
    generationRef.current += 1;
    setMessages([]);
    setPending(null);
    setStatus('idle');
    setError(null);
    setThreadId(newThreadId);
  }, []);

  return { threadId, messages, pending, status, error, reconnecting, send, confirm, resetTo, clearError };
}

function formatArgsShort(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => {
      const s = typeof v === 'string' ? v : JSON.stringify(v);
      return `--${k} ${s.length > 30 ? s.slice(0, 27) + '...' : s}`;
    })
    .join(' ');
}
