/**
 * Zustand store for Evalyn Workbench.
 *
 * Shape mirrors spec §8 "Store shape (Zustand)". Slices are populated by
 * later lanes:
 *   - catalog        Lane B2 (CliCatalog)
 *   - tabs           This lane (A5)
 *   - jobs           Lane B3 (Terminal/Jobs)
 *   - fileTree       Lane B2 (Sidebar Files)
 *   - runs           Lane B2 (Sidebar Runs)
 *   - agent          Lane C2 (ChatPanel)
 *   - settings       Lane C2 (SettingsModal)
 *   - tweaks         This lane (A5)
 *
 * Mock parity: theme + layout tweaks are sourced from the JSX mock's
 * `useTweaks(TWEAK_DEFAULTS)` hook (see /tmp/evalyn-dashboard-mock/wb-app.jsx).
 */

import { create } from 'zustand';
import { api } from './api';
import type { CliSchema } from './types/catalog';
import type { Tab, Job, JobLine, JobWsEvent, FileNode, RunMeta } from './types/jobs';
import type {
  AgentState,
  AgentWsEvent,
  ChatMessage,
  ProviderState,
  SettingsState,
  ToolCall,
} from './types/agent';
import { openAgentWs, openJobWs } from './api';

export type Theme = 'dark' | 'light';
export type ChatPlacement = 'dock' | 'bottom' | 'bubble';
export type CliFormMode = 'form' | 'preview' | 'raw';

export interface Tweaks {
  theme: Theme;
  chatPlacement: ChatPlacement;
  sidebarCollapsed: boolean;
  cliFormMode: CliFormMode;
  monoOnly: boolean;
  showJobsPanel: boolean;
}

export const TWEAK_DEFAULTS: Tweaks = {
  theme: 'light',
  chatPlacement: 'dock',
  sidebarCollapsed: false,
  cliFormMode: 'preview',
  monoOnly: false,
  showJobsPanel: true,
};

export type SidebarView = 'files' | 'clis' | 'runs';
export type BottomTab = 'terminal' | 'jobs' | 'problems';

/**
 * Cap on how many lines we keep in memory per job. The backend already
 * enforces a hard cap (10000) but we keep a smaller window on the
 * client so the Terminal stays cheap to re-render.
 */
export const MAX_JOB_LINES = 5000;

export interface StoreState {
  /* === core data slices ================================================ */
  catalog: CliSchema[];
  tabs: Tab[];
  activeTabId: string | null;
  jobs: Map<string, Job>;
  /** Per-job buffer of streamed lines (Terminal renders from here). */
  jobEvents: Map<string, JobLine[]>;
  /** Per-job last received `event_id` — used for `?since=` on reconnect. */
  jobLastEventId: Map<string, number>;
  fileTree: FileNode[];
  runs: RunMeta[];
  agent: AgentState;
  settings: SettingsState;

  /* === ui state ======================================================== */
  tweaks: Tweaks;
  sidebarView: SidebarView;
  bottomTab: BottomTab;
  paletteOpen: boolean;
  tweaksOpen: boolean;
  chatVisible: boolean;
  /** Settings modal open flag. */
  settingsOpen: boolean;

  /* === actions ========================================================= */
  /** Replace the catalog (called once after `/api/cli` boot fetch). */
  setCatalog: (catalog: CliSchema[]) => void;
  /** Fetch the catalog from `/api/cli` and store it. Safe to call repeatedly. */
  loadCatalog: () => Promise<void>;
  /** Fetch the file tree from `/api/files/tree` and store it. */
  loadFileTree: () => Promise<void>;
  /** Fetch the runs list from `/api/runs` and store it. */
  loadRuns: () => Promise<void>;
  /** Add a new tab and make it active. No-op if the id already exists; in
   *  that case, just activate the existing tab. */
  addTab: (tab: Tab) => void;
  /** Switch the active tab. */
  setActiveTab: (id: string | null) => void;
  /** Close a tab; activate the previous one (or null if last). */
  closeTab: (id: string) => void;
  /** Convenience helper: open a CLI form tab. */
  openCli: (id: string) => void;
  /** Convenience helper: open a file/run tab by name. */
  openFile: (name: string) => void;
  /** Open a job-output tab and insert a placeholder Job entry. */
  openJobTab: (jobId: string, cmd?: string, cliId?: string) => void;
  /** POST `/api/cli/run` with the given args; on success open a job tab and
   *  return the assigned `jobId`. Throws on network/validation error. */
  runCli: (cliId: string, args: Record<string, unknown>) => Promise<string>;
  /** Insert or replace a job by id. */
  upsertJob: (job: Job) => void;
  /** Append a streamed line to a job's buffer (capped to MAX_JOB_LINES). */
  appendJobLine: (jobId: string, line: JobLine, eventId?: number) => void;
  /** Open a `/ws/jobs/{id}` connection; returns an unsubscribe fn. Idempotent. */
  subscribeJob: (jobId: string, options?: { factory?: (url: string) => WebSocket }) => () => void;
  /** Close the WS for one job and free buffers. */
  unsubscribeJob: (jobId: string) => void;
  /** Replace the file tree (called by Lane B2). */
  setFileTree: (tree: FileNode[]) => void;
  /** Replace the runs list (called by Lane B2). */
  setRuns: (runs: RunMeta[]) => void;
  /** Patch one tweak field. */
  setTweak: <K extends keyof Tweaks>(key: K, value: Tweaks[K]) => void;
  /** Set sidebar view (Files / CLIs / Runs). */
  setSidebarView: (view: SidebarView) => void;
  /** Set bottom tab (Terminal / Jobs / Problems). */
  setBottomTab: (tab: BottomTab) => void;
  /** Set palette visibility. */
  setPaletteOpen: (open: boolean) => void;
  /** Set tweaks panel visibility. */
  setTweaksOpen: (open: boolean) => void;
  /** Set chat panel visibility. */
  setChatVisible: (visible: boolean) => void;
  /** Open / close the SettingsModal. */
  openSettings: () => void;
  closeSettings: () => void;
  /** Initial fetch of /api/settings — populates the providers map. */
  loadSettings: () => Promise<void>;
  /** Save provider api key + model selection. */
  saveProvider: (
    provider: string,
    payload: { api_key?: string; model?: string },
  ) => Promise<void>;
  /** 1-token connection test against a provider. */
  testProvider: (provider: string) => Promise<void>;
  /** Mark a provider active (server enforces single-active). */
  setActiveProvider: (provider: string) => Promise<void>;
  /** List models for a provider, caching the result on the provider state. */
  listProviderModels: (provider: string) => Promise<string[]>;

  /* === agent actions ================================================== */
  /** Send a chat message. Starts a new thread on first send, reuses an
   *  existing one otherwise. Opens the WS subscription on thread create. */
  sendChatMessage: (message: string) => Promise<void>;
  /** Approve or reject a pending tool call. */
  confirmAgent: (approve: boolean) => Promise<void>;
  /** Reset the chat: close socket, clear messages, drop thread id. */
  resetChat: () => void;
  /** Internal: dispatch a single agent WS event into the store. Exported
   *  for tests; not used directly by components. */
  dispatchAgentEvent: (evt: AgentWsEvent) => void;
}

const initialAgent: AgentState = {
  threadId: null,
  messages: [],
  status: 'idle',
  pendingConfirmation: null,
  error: null,
};

const DEFAULT_PROVIDERS: ProviderState[] = [
  { id: 'openai', name: 'OpenAI', hasKey: false, requiresKey: true, model: null, testStatus: 'untested' },
  { id: 'anthropic', name: 'Anthropic', hasKey: false, requiresKey: true, model: null, testStatus: 'untested' },
  { id: 'ollama', name: 'Ollama (local)', hasKey: false, requiresKey: false, model: null, testStatus: 'untested' },
];

const initialSettings: SettingsState = {
  providers: Object.fromEntries(DEFAULT_PROVIDERS.map((p) => [p.id, p])),
  active: null,
};

/** Stable ids for client-assigned messages (user turns, fallback assistant ids). */
let _msgCounter = 0;
const newMsgId = (prefix: string): string => `${prefix}-${++_msgCounter}-${Date.now().toString(36)}`;

const _seedProvider = (id: string): ProviderState => ({
  id,
  name: id,
  hasKey: false,
  requiresKey: true,
  model: null,
  testStatus: 'untested',
});

/**
 * Returns a settings slice with a single provider patched. Centralises the
 * `{ ...settings, providers: { ...providers, [id]: { ...prior, ...patch } } }`
 * boilerplate that every settings action would otherwise repeat.
 */
const _withProviderPatch = (
  s: StoreState,
  id: string,
  patch: Partial<ProviderState>,
): { settings: SettingsState } => ({
  settings: {
    ...s.settings,
    providers: {
      ...s.settings.providers,
      [id]: { ...(s.settings.providers[id] ?? _seedProvider(id)), ...patch, id },
    },
  },
});

const toneForName = (name: string): string | undefined => {
  if (name.endsWith('.run')) return 'var(--fail)';
  if (name.endsWith('.yaml')) return 'var(--text-2)';
  return 'var(--text-2)';
};

const kindForName = (name: string): Tab['kind'] => {
  if (name.endsWith('.run')) return 'run';
  if (name.endsWith('.yaml')) return 'yaml';
  return 'file';
};

/**
 * Per-job WS connection bookkeeping. Lives outside the store so React
 * doesn't re-render when sockets attach/detach. The keys are job ids.
 */
interface JobConn {
  socket: WebSocket;
  /** Reconnect attempt counter, used to back off. */
  attempts: number;
  /** Closed intentionally — skip reconnect. */
  closed: boolean;
  /** Backoff timer handle, when one is pending. */
  retryTimer: ReturnType<typeof setTimeout> | null;
  /** Override for tests. */
  factory?: (url: string) => WebSocket;
}

const jobConns = new Map<string, JobConn>();

const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000];

/**
 * Singleton agent socket. Lives outside the store for the same reason as
 * jobConns — we don't want React to re-render when the WS attaches/detaches.
 */
interface AgentConn {
  socket: WebSocket;
  threadId: string;
  closed: boolean;
  factory?: (url: string) => WebSocket;
}

let agentConn: AgentConn | null = null;

const jobLineKindFromEvent = (type: JobWsEvent['type']): JobLine['kind'] => {
  switch (type) {
    case 'stdout':
    case 'stderr':
    case 'info':
    case 'prompt':
    case 'ok':
    case 'warn':
    case 'fail':
    case 'exit':
      return type;
    default:
      return 'info';
  }
};

export const useStore = create<StoreState>((set, get) => ({
  /* === initial state =================================================== */
  catalog: [],
  tabs: [],
  activeTabId: null,
  jobs: new Map(),
  jobEvents: new Map(),
  jobLastEventId: new Map(),
  fileTree: [],
  runs: [],
  agent: initialAgent,
  settings: initialSettings,

  tweaks: { ...TWEAK_DEFAULTS },
  sidebarView: 'files',
  bottomTab: 'terminal',
  paletteOpen: false,
  tweaksOpen: false,
  chatVisible: true,
  settingsOpen: false,

  /* === actions ========================================================= */
  setCatalog: (catalog) => set({ catalog }),

  loadCatalog: async () => {
    const catalog = await api.catalog();
    set({ catalog });
  },

  loadFileTree: async () => {
    const tree = await api.fileTree();
    set({ fileTree: tree });
  },

  loadRuns: async () => {
    const runs = await api.runs();
    set({ runs });
  },

  addTab: (tab) => {
    const existing = get().tabs.find((t) => t.id === tab.id);
    if (existing) {
      set({ activeTabId: tab.id });
      return;
    }
    set((s) => ({ tabs: [...s.tabs, tab], activeTabId: tab.id }));
  },

  setActiveTab: (id) => set({ activeTabId: id }),

  closeTab: (id) => {
    const { tabs, activeTabId } = get();
    const idx = tabs.findIndex((t) => t.id === id);
    if (idx === -1) return;
    const next = tabs.filter((t) => t.id !== id);
    let nextActive = activeTabId;
    if (activeTabId === id) {
      const fallback = next[Math.max(0, idx - 1)];
      nextActive = fallback ? fallback.id : null;
    }
    set({ tabs: next, activeTabId: nextActive });
  },

  openCli: (id) => {
    const tabId = `cli:${id}`;
    get().addTab({ id: tabId, title: id, kind: 'cli', tone: 'var(--accent)' });
  },

  openFile: (name) => {
    get().addTab({
      id: name,
      title: name,
      kind: kindForName(name),
      tone: toneForName(name),
    });
  },

  openJobTab: (jobId, cmd, cliId) => {
    const tabId = `job:${jobId}`;
    get().addTab({ id: tabId, title: jobId, kind: 'job', tone: 'var(--accent)' });
    const existing = get().jobs.get(jobId);
    if (!existing) {
      set((s) => {
        const next = new Map(s.jobs);
        next.set(jobId, {
          id: jobId,
          cmd: cmd ?? '',
          cliId,
          status: 'pending',
          lines: [],
        });
        return { jobs: next };
      });
    }
  },

  runCli: async (cliId, args) => {
    const { job_id } = await api.runCli(cliId, args);
    get().openJobTab(job_id, `evalyn ${cliId}`, cliId);
    return job_id;
  },

  upsertJob: (job) =>
    set((s) => {
      const next = new Map(s.jobs);
      next.set(job.id, job);
      return { jobs: next };
    }),

  appendJobLine: (jobId, line, eventId) =>
    set((s) => {
      const next = new Map(s.jobEvents);
      const prior = next.get(jobId) ?? [];
      const buf = prior.length >= MAX_JOB_LINES
        ? [...prior.slice(prior.length - MAX_JOB_LINES + 1), line]
        : [...prior, line];
      next.set(jobId, buf);
      const updates: Partial<StoreState> = { jobEvents: next };
      if (eventId != null) {
        const ids = new Map(s.jobLastEventId);
        ids.set(jobId, eventId);
        updates.jobLastEventId = ids;
      }
      return updates;
    }),

  subscribeJob: (jobId, options = {}) => {
    const existing = jobConns.get(jobId);
    if (existing && !existing.closed) {
      // Already subscribed — return an unsubscribe that closes it.
      return () => get().unsubscribeJob(jobId);
    }

    const factory = options.factory;
    const open = (attempts: number): void => {
      const since = get().jobLastEventId.get(jobId) ?? null;
      const socket = openJobWs(jobId, { sinceEventId: since, factory });
      const conn: JobConn = { socket, attempts, closed: false, retryTimer: null, factory };
      jobConns.set(jobId, conn);

      socket.addEventListener('open', () => {
        conn.attempts = 0;
      });

      socket.addEventListener('message', (ev: MessageEvent) => {
        let evt: JobWsEvent;
        try {
          evt = JSON.parse(ev.data as string) as JobWsEvent;
        } catch {
          return;
        }
        // Update lastEventId and dispatch to job state.
        const eventId = evt.event_id;
        if (evt.type === 'exit') {
          const job = get().jobs.get(jobId);
          if (job) {
            const code = evt.code;
            const status: Job['status'] =
              code === 0 ? 'complete' : code === 130 || code === -15 || code === 143 ? 'cancelled' : 'failed';
            get().upsertJob({ ...job, status, exitCode: code, duration: evt.duration ?? job.duration });
          }
          get().appendJobLine(
            jobId,
            { kind: 'exit', text: `exit ${evt.code}${evt.duration ? ` · ${evt.duration}` : ''}`, ts: evt.ts },
            eventId,
          );
          return;
        }
        if (evt.type === 'progress') {
          const job = get().jobs.get(jobId);
          if (job) {
            get().upsertJob({ ...job, progress: evt.progress, eta: evt.eta ?? job.eta });
          }
          if (eventId != null) {
            set((s) => {
              const ids = new Map(s.jobLastEventId);
              ids.set(jobId, eventId);
              return { jobLastEventId: ids };
            });
          }
          return;
        }
        if (evt.type === 'truncated') {
          get().appendJobLine(
            jobId,
            { kind: 'warn', text: `[truncated ${evt.count} lines]`, ts: evt.ts },
            eventId,
          );
          return;
        }
        // stdout/stderr/info/prompt/ok/warn/fail line event.
        get().appendJobLine(
          jobId,
          { kind: jobLineKindFromEvent(evt.type), text: evt.line, ts: evt.ts },
          eventId,
        );
      });

      socket.addEventListener('close', () => {
        if (conn.closed) return;
        const job = get().jobs.get(jobId);
        // If the job has finalized, don't reconnect.
        if (job && (job.status === 'complete' || job.status === 'failed' || job.status === 'cancelled')) {
          conn.closed = true;
          jobConns.delete(jobId);
          return;
        }
        const attempt = Math.min(conn.attempts, RECONNECT_DELAYS_MS.length - 1);
        const delay = RECONNECT_DELAYS_MS[attempt];
        conn.retryTimer = setTimeout(() => {
          conn.retryTimer = null;
          if (!conn.closed) open(attempt + 1);
        }, delay);
      });

      socket.addEventListener('error', () => {
        // Errors are followed by close events — let the close handler retry.
      });
    };

    open(0);
    return () => get().unsubscribeJob(jobId);
  },

  unsubscribeJob: (jobId) => {
    const conn = jobConns.get(jobId);
    if (!conn) return;
    conn.closed = true;
    if (conn.retryTimer) {
      clearTimeout(conn.retryTimer);
      conn.retryTimer = null;
    }
    try {
      conn.socket.close();
    } catch {
      // ignore
    }
    jobConns.delete(jobId);
  },

  setFileTree: (tree) => set({ fileTree: tree }),
  setRuns: (runs) => set({ runs }),

  setTweak: (key, value) =>
    set((s) => ({ tweaks: { ...s.tweaks, [key]: value } })),

  setSidebarView: (view) => set({ sidebarView: view }),
  setBottomTab: (tab) => set({ bottomTab: tab }),
  setPaletteOpen: (open) => set({ paletteOpen: open }),
  setTweaksOpen: (open) => set({ tweaksOpen: open }),
  setChatVisible: (visible) => set({ chatVisible: visible }),

  openSettings: () => set({ settingsOpen: true }),
  closeSettings: () => set({ settingsOpen: false }),

  loadSettings: async () => {
    try {
      const fetched = await api.settings();
      set((s) => {
        // Merge server-side state into the seeded defaults so the UI keeps
        // showing the three providers even if the server's response only
        // covers a subset.
        const merged: Record<string, ProviderState> = { ...s.settings.providers };
        for (const [id, partial] of Object.entries(fetched.providers ?? {})) {
          merged[id] = {
            ...(merged[id] ?? _seedProvider(id)),
            ...(partial as Partial<ProviderState>),
            id,
          };
        }
        return { settings: { providers: merged, active: fetched.active ?? s.settings.active } };
      });
    } catch (err) {
      console.warn('loadSettings failed', err);
    }
  },

  saveProvider: async (provider, payload) => {
    const updated = await api.saveProvider(provider, payload);
    set((s) => _withProviderPatch(s, provider, updated));
  },

  testProvider: async (provider) => {
    set((s) => _withProviderPatch(s, provider, { testStatus: 'testing', testError: undefined }));
    try {
      const res = await api.testProvider(provider);
      set((s) =>
        _withProviderPatch(s, provider, {
          testStatus: res.ok ? 'ok' : 'error',
          testError: res.ok ? undefined : (res.error ?? 'Test failed'),
        }),
      );
    } catch (err) {
      set((s) =>
        _withProviderPatch(s, provider, {
          testStatus: 'error',
          testError: err instanceof Error ? err.message : String(err),
        }),
      );
    }
  },

  setActiveProvider: async (provider) => {
    await api.setActiveProvider(provider);
    set((s) => ({ settings: { ...s.settings, active: provider } }));
  },

  listProviderModels: async (provider) => {
    set((s) => _withProviderPatch(s, provider, { loadingModels: true }));
    try {
      const { models } = await api.listProviderModels(provider);
      set((s) => _withProviderPatch(s, provider, { models, loadingModels: false }));
      return models;
    } catch (err) {
      // Set models to an empty array so the load-on-expand effect doesn't
      // retry on every re-render. A manual refresh is the recovery path.
      set((s) => _withProviderPatch(s, provider, { models: [], loadingModels: false }));
      throw err;
    }
  },

  sendChatMessage: async (message) => {
    const trimmed = message.trim();
    if (!trimmed) return;
    // Append user turn locally so the bubble shows up immediately.
    const userMsg: ChatMessage = {
      id: newMsgId('u'),
      role: 'user',
      text: trimmed,
      ts: new Date().toISOString(),
    };
    set((s) => ({
      agent: {
        ...s.agent,
        messages: [...s.agent.messages, userMsg],
        status: 'streaming',
        error: null,
      },
    }));

    const existing = get().agent.threadId;
    try {
      if (!existing) {
        const { thread_id } = await api.startAgentThread(trimmed);
        set((s) => ({ agent: { ...s.agent, threadId: thread_id } }));
        _attachAgentSocket(thread_id, get, set);
      } else {
        await api.sendAgentMessage(existing, trimmed);
      }
    } catch (err) {
      set((s) => ({
        agent: {
          ...s.agent,
          status: 'error',
          error: {
            kind: 'network',
            message: err instanceof Error ? err.message : String(err),
          },
        },
      }));
    }
  },

  confirmAgent: async (approve) => {
    const { agent } = get();
    if (!agent.threadId || !agent.pendingConfirmation) return;
    const toolCallId = agent.pendingConfirmation.toolCallId;
    // Optimistically clear the pending confirmation; the server will push
    // tool_call_running / tool_call_complete (or error) next.
    set((s) => ({
      agent: {
        ...s.agent,
        pendingConfirmation: null,
        status: approve ? 'streaming' : 'idle',
        messages: s.agent.messages.map((m) =>
          m.toolCall && m.toolCall.id === toolCallId
            ? {
                ...m,
                toolCall: {
                  ...m.toolCall,
                  status: approve ? 'running' : 'rejected',
                },
              }
            : m,
        ),
      },
    }));
    try {
      await api.confirmAgentTool(agent.threadId, approve, toolCallId);
    } catch (err) {
      set((s) => ({
        agent: {
          ...s.agent,
          status: 'error',
          error: {
            kind: 'network',
            message: err instanceof Error ? err.message : String(err),
          },
        },
      }));
    }
  },

  resetChat: () => {
    if (agentConn) {
      agentConn.closed = true;
      try {
        agentConn.socket.close();
      } catch {
        // ignore
      }
      agentConn = null;
    }
    set({ agent: { ...initialAgent } });
  },

  dispatchAgentEvent: (evt) => {
    _dispatchAgentEvent(evt, get, set);
  },
}));

/* ---------------------------------------------------------------------------
 * Agent socket helpers. Module-private so the store action surface stays
 * narrow and the connection bookkeeping doesn't leak into React state.
 * ------------------------------------------------------------------------- */

type AgentSetFn = (
  partial: Partial<StoreState> | ((s: StoreState) => Partial<StoreState>),
) => void;

type AgentGetFn = () => StoreState;

const _attachAgentSocket = (
  threadId: string,
  get: AgentGetFn,
  set: AgentSetFn,
  factory?: (url: string) => WebSocket,
): void => {
  // Tear down any prior conn first.
  if (agentConn) {
    agentConn.closed = true;
    try {
      agentConn.socket.close();
    } catch {
      // ignore
    }
    agentConn = null;
  }
  const socket = openAgentWs(threadId, { factory });
  const conn: AgentConn = { socket, threadId, closed: false, factory };
  agentConn = conn;
  socket.addEventListener('message', (ev: MessageEvent) => {
    let payload: AgentWsEvent;
    try {
      payload = JSON.parse(ev.data as string) as AgentWsEvent;
    } catch {
      return;
    }
    _dispatchAgentEvent(payload, get, set);
  });
  socket.addEventListener('close', () => {
    if (agentConn === conn) {
      agentConn = null;
    }
  });
};

/**
 * Apply a single agent WS event to the store. Tagged-union dispatch keeps
 * the message list and `pendingConfirmation` in sync with the runtime.
 */
const _dispatchAgentEvent = (
  evt: AgentWsEvent,
  _get: AgentGetFn,
  set: AgentSetFn,
): void => {
  switch (evt.type) {
    case 'text_delta': {
      set((s) => {
        const messages = [...s.agent.messages];
        const idx = messages.findIndex((m) => m.id === evt.message_id);
        if (idx >= 0) {
          const prior = messages[idx];
          messages[idx] = {
            ...prior,
            text: (prior.text ?? '') + evt.delta,
            streaming: true,
          };
        } else {
          messages.push({
            id: evt.message_id,
            role: 'assistant',
            text: evt.delta,
            streaming: true,
            ts: evt.ts,
          });
        }
        return { agent: { ...s.agent, messages, status: 'streaming' } };
      });
      return;
    }
    case 'tool_call_proposal': {
      const card: ToolCall = {
        id: evt.tool_call_id,
        tool: evt.tool,
        args: evt.args,
        previewCmd: evt.preview_cmd,
        status: 'proposed',
      };
      set((s) => ({
        agent: {
          ...s.agent,
          messages: [
            ...s.agent.messages,
            {
              id: `tc-${evt.tool_call_id}`,
              role: 'assistant',
              toolCall: card,
              ts: evt.ts,
            },
          ],
        },
      }));
      return;
    }
    case 'tool_call_running': {
      set((s) => ({
        agent: {
          ...s.agent,
          messages: s.agent.messages.map((m) =>
            m.toolCall && m.toolCall.id === evt.tool_call_id
              ? {
                  ...m,
                  toolCall: { ...m.toolCall, status: 'running', jobId: evt.job_id },
                }
              : m,
          ),
        },
      }));
      return;
    }
    case 'tool_call_complete': {
      const errored = evt.exit_code != null && evt.exit_code !== 0;
      set((s) => ({
        agent: {
          ...s.agent,
          messages: s.agent.messages.map((m) =>
            m.toolCall && m.toolCall.id === evt.tool_call_id
              ? {
                  ...m,
                  toolCall: {
                    ...m.toolCall,
                    status: errored ? 'error' : 'complete',
                    output: evt.output,
                    error: errored ? `exit ${evt.exit_code}` : undefined,
                  },
                }
              : m,
          ),
        },
      }));
      return;
    }
    case 'confirmation_required': {
      // Find / create the matching ToolCall card; flip status to awaiting.
      set((s) => {
        const messages = [...s.agent.messages];
        const idx = messages.findIndex(
          (m) => m.toolCall && m.toolCall.id === evt.tool_call_id,
        );
        if (idx >= 0) {
          const prior = messages[idx];
          if (prior.toolCall) {
            messages[idx] = {
              ...prior,
              toolCall: { ...prior.toolCall, status: 'awaiting_confirmation' },
            };
          }
        } else {
          messages.push({
            id: `tc-${evt.tool_call_id}`,
            role: 'assistant',
            toolCall: {
              id: evt.tool_call_id,
              tool: evt.tool,
              args: evt.args,
              previewCmd: evt.preview_cmd,
              status: 'awaiting_confirmation',
            },
            ts: evt.ts,
          });
        }
        return {
          agent: {
            ...s.agent,
            messages,
            status: 'awaiting_confirmation',
            pendingConfirmation: {
              toolCallId: evt.tool_call_id,
              tool: evt.tool,
              args: evt.args,
              previewCmd: evt.preview_cmd,
            },
          },
        };
      });
      return;
    }
    case 'error': {
      set((s) => ({
        agent: {
          ...s.agent,
          status: 'error',
          error: { kind: evt.kind, message: evt.message, provider: evt.provider },
        },
      }));
      return;
    }
    case 'final': {
      const suggestions = evt.suggestions?.map((s) => ({
        label: s.label,
        cliId: s.cli_id,
        args: s.args,
      }));
      set((s) => {
        const messages = [...s.agent.messages];
        // If a streaming assistant message exists with the same id, finalize it.
        const idx = evt.message_id
          ? messages.findIndex((m) => m.id === evt.message_id)
          : -1;
        if (idx >= 0) {
          messages[idx] = {
            ...messages[idx],
            text: evt.text || messages[idx].text,
            streaming: false,
            finalSuggestions: suggestions,
          };
        } else {
          messages.push({
            id: evt.message_id ?? newMsgId('a'),
            role: 'assistant',
            text: evt.text,
            finalSuggestions: suggestions,
            ts: evt.ts,
          });
        }
        return {
          agent: {
            ...s.agent,
            messages,
            status: 'idle',
          },
        };
      });
      return;
    }
  }
};

/** Reset helper for tests; not used in app code. */
export const __resetStore = (): void => {
  // Tear down any live WS connections so tests don't leak handles.
  for (const [id, conn] of jobConns.entries()) {
    conn.closed = true;
    if (conn.retryTimer) {
      clearTimeout(conn.retryTimer);
    }
    try {
      conn.socket.close();
    } catch {
      // ignore
    }
    jobConns.delete(id);
  }
  if (agentConn) {
    agentConn.closed = true;
    try {
      agentConn.socket.close();
    } catch {
      // ignore
    }
    agentConn = null;
  }
  useStore.setState({
    catalog: [],
    tabs: [],
    activeTabId: null,
    jobs: new Map(),
    jobEvents: new Map(),
    jobLastEventId: new Map(),
    fileTree: [],
    runs: [],
    agent: { ...initialAgent },
    settings: {
      providers: Object.fromEntries(DEFAULT_PROVIDERS.map((p) => [p.id, { ...p }])),
      active: null,
    },
    tweaks: { ...TWEAK_DEFAULTS },
    sidebarView: 'files',
    bottomTab: 'terminal',
    paletteOpen: false,
    tweaksOpen: false,
    chatVisible: true,
    settingsOpen: false,
  });
};

/** Test-only accessor for the WS connection registry. */
export const __jobConnsForTest = jobConns;

/**
 * Test-only injection: open the agent socket using a mock factory, without
 * going through the POST /api/agent/chat round-trip.
 */
export const __attachAgentSocketForTest = (
  threadId: string,
  factory: (url: string) => WebSocket,
): void => {
  useStore.setState((s) => ({ agent: { ...s.agent, threadId } }));
  _attachAgentSocket(
    threadId,
    () => useStore.getState(),
    (partial) => {
      if (typeof partial === 'function') {
        useStore.setState(partial as (s: StoreState) => Partial<StoreState>);
      } else {
        useStore.setState(partial);
      }
    },
    factory,
  );
};
