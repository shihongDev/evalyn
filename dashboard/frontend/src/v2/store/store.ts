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

/**
 * Co-pilot UI guidance tour state.
 *
 * Tour state is in-memory only (Zustand). A page refresh mid-tour abandons
 * the run; that is acceptable for v1 (single-route Home tour). Completion
 * flags persist to localStorage so a finished tour does not re-fire.
 *
 *   tour storage keys:
 *     evalyn.tour.enabled         boolean - global feature toggle (Settings)
 *     evalyn.tour.completed.<id>  '1' if user completed tour <id>
 */
export const TOUR_ENABLED_KEY = 'evalyn.tour.enabled';
export const tourCompletedKey = (tourId: string) => `evalyn.tour.completed.${tourId}`;

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

  /**
   * Active tour state.
   *
   *   idle           tourActiveId === null
   *   running        tourActiveId === <id>, tourStep >= 0
   *   completed      tourActiveId === null + localStorage flag set
   *   abandoned      tourActiveId === null + no localStorage flag (re-fires next visit)
   */
  tourActiveId: string | null;
  tourStep: number;
  setTour: (id: string, step?: number) => void;
  advanceTour: () => void;
  markTourComplete: (id: string) => void;
  abandonTour: () => void;
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

  tourActiveId: null,
  tourStep: 0,
  setTour: (id, step = 0) => set({ tourActiveId: id, tourStep: step }),
  advanceTour: () => set((s) => ({ tourStep: s.tourStep + 1 })),
  markTourComplete: (id) => {
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(tourCompletedKey(id), '1');
      } catch {
        // Quota exceeded, private mode, etc. - completion is best-effort.
      }
    }
    set({ tourActiveId: null, tourStep: 0 });
  },
  abandonTour: () => set({ tourActiveId: null, tourStep: 0 }),
}));
