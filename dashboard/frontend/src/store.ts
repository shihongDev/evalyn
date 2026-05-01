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
import type { Tab, Job, FileNode, RunMeta } from './types/jobs';
import type { AgentState, SettingsState } from './types/agent';

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

export interface StoreState {
  /* === core data slices ================================================ */
  catalog: CliSchema[];
  tabs: Tab[];
  activeTabId: string | null;
  jobs: Map<string, Job>;
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

export const useStore = create<StoreState>((set, get) => ({
  /* === initial state =================================================== */
  catalog: [],
  tabs: [],
  activeTabId: null,
  jobs: new Map(),
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
  useStore.setState({
    catalog: [],
    tabs: [],
    activeTabId: null,
    jobs: new Map(),
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
