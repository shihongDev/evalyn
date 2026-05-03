/**
 * Lean v2 store - holds only what the new screens need.
 *
 * Replaces the workbench-era store entirely. Co-pilot dock + thread list
 * lives here (the renderer wires to /ws/agent on its own; this just holds
 * UI state like "which thread is open" and "is the dock visible").
 */

import { create } from 'zustand';

export interface CoPilotMessage {
  id: string;
  role: 'you' | 'agent';
  text: string;
  /** Tool blocks recorded for this turn (synthesized from WS events). */
  tools?: { ok: boolean; cmd: string; duration_s: number }[];
  /** Pending confirmation card payload (only when waiting on user). */
  pending_confirm?: {
    tool_call_id: string;
    tool: string;
    preview_cmd: string;
  } | null;
}

export interface CoPilotThread {
  id: string;
  title: string;
  messages: CoPilotMessage[];
  created_at_iso: string;
}

interface V2Store {
  /** Co-pilot side dock visibility. */
  dockOpen: boolean;
  setDockOpen: (open: boolean) => void;

  /** Open thread (id) and the message list. Side dock and CoPilotThread route share this. */
  threads: Record<string, CoPilotThread>;
  activeThreadId: string | null;
  setActiveThreadId: (id: string | null) => void;
  upsertThread: (thread: CoPilotThread) => void;
  appendMessage: (threadId: string, msg: CoPilotMessage) => void;
  patchMessage: (
    threadId: string,
    messageId: string,
    patch: Partial<CoPilotMessage>,
  ) => void;
  removeThread: (id: string) => void;
}

export const useV2Store = create<V2Store>((set) => ({
  dockOpen: true,
  setDockOpen: (open) => set({ dockOpen: open }),

  threads: {},
  activeThreadId: null,
  setActiveThreadId: (id) => set({ activeThreadId: id }),
  upsertThread: (thread) =>
    set((s) => ({ threads: { ...s.threads, [thread.id]: thread } })),
  appendMessage: (threadId, msg) =>
    set((s) => {
      const t = s.threads[threadId];
      if (!t) return s;
      return {
        threads: {
          ...s.threads,
          [threadId]: { ...t, messages: [...t.messages, msg] },
        },
      };
    }),
  patchMessage: (threadId, messageId, patch) =>
    set((s) => {
      const t = s.threads[threadId];
      if (!t) return s;
      return {
        threads: {
          ...s.threads,
          [threadId]: {
            ...t,
            messages: t.messages.map((m) =>
              m.id === messageId ? { ...m, ...patch } : m,
            ),
          },
        },
      };
    }),
  removeThread: (id) =>
    set((s) => {
      const next = { ...s.threads };
      delete next[id];
      return {
        threads: next,
        activeThreadId: s.activeThreadId === id ? null : s.activeThreadId,
      };
    }),
}));
