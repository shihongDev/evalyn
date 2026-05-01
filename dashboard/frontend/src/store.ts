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
import type { AgentState, SettingsState } from './types/agent';
import { openJobWs } from './api';

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
}

const initialAgent: AgentState = {
  threadId: null,
  messages: [],
  status: 'idle',
  pendingConfirmation: null,
};

const initialSettings: SettingsState = {
  providers: {},
  active: null,
};

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
            { kind: 'warn', text: `[truncated ${evt.dropped} lines]`, ts: evt.ts },
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
}));

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
  useStore.setState({
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
  });
};

/** Test-only accessor for the WS connection registry. */
export const __jobConnsForTest = jobConns;
