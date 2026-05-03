/**
 * Zustand store for Evalyn Workbench.
 *
 * Shape mirrors spec §8 "Store shape (Zustand)". The "Action + Run History"
 * UX (Workspace.tsx) replaced the earlier Notebook prototype: a single
 * active form lives at the top, and every CLI invocation appends a flat
 * RunRecord to `runHistory` (newest first when rendered).
 *
 * Mock parity: theme + layout tweaks are sourced from the JSX mock's
 * `useTweaks(TWEAK_DEFAULTS)` hook (see /tmp/evalyn-dashboard-mock/wb-app.jsx).
 */

import { create } from 'zustand';
import { api, type PromoteResponse } from './api';
import type { CliSchema } from './types/catalog';
import type { Tab, Job, JobLine, JobWsEvent, FileNode, RunMeta } from './types/jobs';
import { defaultValues as cliDefaultValues } from './views/buildCli';
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

export type SidebarView = 'files' | 'clis' | 'runs' | 'jobs';
export type BottomTab = 'terminal' | 'jobs' | 'problems';

/**
 * Click-to-attach context chip. The user clicks an "attach" button on any
 * run row (RunCard, RecentRunsStrip, RunsList) to mint one of these and
 * append it to `chatAttachments`. The ChatComposer renders pending chips
 * above the textarea and serializes them into the next user message
 * (e.g. `[run-id: abc123] <user text>`) before clearing the list.
 */
export interface ChatAttachment {
  /** Stable id, e.g. `run:abc123`. Used for dedupe + remove. */
  id: string;
  kind: 'run' | 'metric' | 'file';
  /** Display label rendered inside the chip. */
  label: string;
  /** The actual id / path the agent should resolve. */
  ref: string;
}

/**
 * One persisted chat thread (P1 chat persistence). The full message log is
 * stored separately under `evalyn:threads:msgs:${id}` to keep the index small.
 */
export interface ChatThread {
  /** Stable id, prefixed `t-`. */
  id: string;
  /** Auto-derived from first user message; truncated to ~60 chars. */
  title: string;
  /** Epoch ms when first created. */
  createdAt: number;
  /** Epoch ms when last modified. */
  modifiedAt: number;
  /** Current message count for the thread. */
  messageCount: number;
}

/* -------------------- thread persistence storage helpers ------------------ */

const THREAD_INDEX_KEY = 'evalyn:threads:index';
const THREAD_MSGS_PREFIX = 'evalyn:threads:msgs:';
const THREAD_ACTIVE_KEY = 'evalyn:threads:active';
const MAX_THREADS = 20;
const MAX_MESSAGES_PER_THREAD = 200;

const _safeStorage = (): Storage | null => {
  try {
    if (typeof localStorage !== 'undefined') return localStorage;
  } catch {
    // ignore
  }
  return null;
};

const _readThreadIndex = (): ChatThread[] => {
  const ls = _safeStorage();
  if (!ls) return [];
  try {
    const raw = ls.getItem(THREAD_INDEX_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatThread[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

const _writeThreadIndex = (threads: ChatThread[]): void => {
  const ls = _safeStorage();
  if (!ls) return;
  try {
    ls.setItem(THREAD_INDEX_KEY, JSON.stringify(threads));
  } catch {
    // quota exceeded — silent fail
  }
};

const _readThreadMessages = (id: string): ChatMessage[] => {
  const ls = _safeStorage();
  if (!ls) return [];
  try {
    const raw = ls.getItem(`${THREAD_MSGS_PREFIX}${id}`);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatMessage[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

/**
 * Per-message tool-output cap (in chars) when persisting to localStorage.
 * The backend caps subprocess stdout per tool call at 50KB; 20 threads ×
 * 200 messages × that ceiling easily exceeds the ~5MB per-origin cap.
 * Snapshots stay searchable but truncate big stdout dumps to a tail. The
 * full output is replayable from `/api/jobs/{id}` while the job exists.
 */
const PERSIST_MAX_TOOL_OUTPUT = 4000;

const _stripForPersist = (m: ChatMessage): ChatMessage => {
  if (!m.toolCall || !m.toolCall.output) return m;
  if (m.toolCall.output.length <= PERSIST_MAX_TOOL_OUTPUT) return m;
  const tail = m.toolCall.output.slice(-PERSIST_MAX_TOOL_OUTPUT);
  return {
    ...m,
    toolCall: {
      ...m.toolCall,
      output: `[truncated for storage; original ${m.toolCall.output.length} chars]\n...${tail}`,
    },
  };
};

let _quotaWarned = false;

const _writeThreadMessages = (id: string, messages: ChatMessage[]): void => {
  const ls = _safeStorage();
  if (!ls) return;
  try {
    const capped =
      messages.length > MAX_MESSAGES_PER_THREAD
        ? messages.slice(messages.length - MAX_MESSAGES_PER_THREAD)
        : messages;
    const stripped = capped.map(_stripForPersist);
    ls.setItem(`${THREAD_MSGS_PREFIX}${id}`, JSON.stringify(stripped));
  } catch (err) {
    // QuotaExceededError is the common case; warn once so the user sees
    // why persistence stopped instead of silently losing threads.
    if (!_quotaWarned) {
      _quotaWarned = true;
      console.warn(
        'evalyn: localStorage quota exceeded; thread persistence is paused for this session',
        err,
      );
    }
  }
};

const _deleteThreadMessages = (id: string): void => {
  const ls = _safeStorage();
  if (!ls) return;
  try {
    ls.removeItem(`${THREAD_MSGS_PREFIX}${id}`);
  } catch {
    // ignore
  }
};

const _readActiveThreadId = (): string | null => {
  const ls = _safeStorage();
  if (!ls) return null;
  try {
    return ls.getItem(THREAD_ACTIVE_KEY);
  } catch {
    return null;
  }
};

const _writeActiveThreadId = (id: string | null): void => {
  const ls = _safeStorage();
  if (!ls) return;
  try {
    if (id == null) ls.removeItem(THREAD_ACTIVE_KEY);
    else ls.setItem(THREAD_ACTIVE_KEY, id);
  } catch {
    // ignore
  }
};

const _newThreadId = (): string =>
  `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const _deriveThreadTitle = (messages: ChatMessage[]): string => {
  const firstUser = messages.find((m) => m.role === 'user' && m.text);
  const raw = firstUser?.text?.trim() ?? 'New conversation';
  if (raw.length <= 60) return raw;
  return `${raw.slice(0, 57)}...`;
};

/** Drop the oldest entries by `modifiedAt` so the index stays <= MAX_THREADS. */
const _capThreadIndex = (threads: ChatThread[]): ChatThread[] => {
  if (threads.length <= MAX_THREADS) return threads;
  const sorted = [...threads].sort((a, b) => b.modifiedAt - a.modifiedAt);
  const kept = sorted.slice(0, MAX_THREADS);
  const dropped = sorted.slice(MAX_THREADS);
  for (const t of dropped) _deleteThreadMessages(t.id);
  return kept;
};

/**
 * A starred preset: a saved (cliId, args) tuple the user can reapply to
 * the active form with one click. Persisted to localStorage under
 * `evalyn:presets`. The `label` is auto-derived at create time but may be
 * renamed by the user in a future iteration.
 */
export interface Preset {
  /** Stable id, prefixed `preset-`. */
  id: string;
  /** CLI this preset targets. */
  cliId: string;
  /** Display label (auto-derived from args by default). */
  label: string;
  /** Snapshot of the form values at star time. */
  args: Record<string, unknown>;
  /** Epoch ms when first created. */
  createdAt: number;
}

/* -------------------- preset persistence storage helpers ----------------- */

const PRESETS_KEY = 'evalyn:presets';
const MAX_PRESETS = 50;

const _readPresets = (): Preset[] => {
  const ls = _safeStorage();
  if (!ls) return [];
  try {
    const raw = ls.getItem(PRESETS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    // Defensive shape check: drop any entry missing required fields so a
    // corrupted store can't crash the app.
    return parsed.filter((p): p is Preset => {
      if (!p || typeof p !== 'object') return false;
      const r = p as Record<string, unknown>;
      return (
        typeof r.id === 'string' &&
        typeof r.cliId === 'string' &&
        typeof r.label === 'string' &&
        typeof r.createdAt === 'number' &&
        r.args != null &&
        typeof r.args === 'object'
      );
    });
  } catch {
    return [];
  }
};

const _writePresets = (presets: Preset[]): void => {
  const ls = _safeStorage();
  if (!ls) return;
  try {
    ls.setItem(PRESETS_KEY, JSON.stringify(presets));
  } catch {
    // quota exceeded - silent fail (presets fit the same budget guidance as
    // the thread index; the in-memory copy still works for this session).
  }
};

const _newPresetId = (): string =>
  `preset-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/**
 * Best-effort label derivation for a starred preset. Picks a recognisable
 * signal from the args dict in priority order:
 *   1. `dataset` basename + `workers` count (e.g. "spam.jsonl . 8w")
 *   2. `dataset` basename only
 *   3. `metrics` (joined, truncated)
 *   4. fallback "<cliId> preset"
 */
const _deriveLabel = (
  cliId: string,
  args: Record<string, unknown>,
): string => {
  const dataset = args['dataset'];
  const workers = args['workers'];
  const metrics = args['metrics'];
  let datasetBase: string | null = null;
  if (typeof dataset === 'string' && dataset.length > 0) {
    const slash = Math.max(dataset.lastIndexOf('/'), dataset.lastIndexOf('\\'));
    datasetBase = slash >= 0 ? dataset.slice(slash + 1) : dataset;
  }
  if (datasetBase && (typeof workers === 'number' || typeof workers === 'string')) {
    const w = String(workers).trim();
    if (w && w !== '0') return `${datasetBase} . ${w}w`;
  }
  if (datasetBase) return datasetBase;
  if (Array.isArray(metrics) && metrics.length > 0) {
    const joined = metrics.map(String).join(',');
    return joined.length > 32 ? `${joined.slice(0, 29)}...` : joined;
  }
  if (typeof metrics === 'string' && metrics.length > 0) {
    return metrics.length > 32 ? `${metrics.slice(0, 29)}...` : metrics;
  }
  return `${cliId} preset`;
};

/**
 * Stable serialization for arg-equality checks. Sorts keys so two dicts
 * with identical values but different insertion order compare equal.
 */
const _argsKey = (args: Record<string, unknown>): string => {
  try {
    const keys = Object.keys(args).sort();
    const sorted: Record<string, unknown> = {};
    for (const k of keys) sorted[k] = args[k];
    return JSON.stringify(sorted);
  } catch {
    return JSON.stringify(args);
  }
};

/**
 * One CLI invocation, captured at submit time. Args are a snapshot of the
 * active form's values when Run was clicked. Pinning is a UI hint - pinned
 * runs cluster at the top of the Workspace feed.
 */
export interface RunRecord {
  id: string;
  cliId: string;
  args: Record<string, unknown>;
  jobId: string;
  startedAt: number;
  pinned: boolean;
}

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
  /** Currently selected CLI for the Workspace's active form. */
  activeCliId: string | null;
  /** Flat list of every CLI invocation (any cliId). Newest entries are
   *  appended at the end; the view renders newest-first. */
  runHistory: RunRecord[];
  /**
   * Transient seed for the active form. Set by `editRunArgs(runId)` so
   * the Workspace's ActiveForm picks up the args once and clears the seed.
   * Lives in the store (not refs) because the trigger originates from a
   * sibling RunCard component; component-local state would not see it.
   */
  activeFormSeed: { cliId: string; args: Record<string, unknown> } | null;

  /* === ui state ======================================================== */
  tweaks: Tweaks;
  sidebarView: SidebarView;
  bottomTab: BottomTab;
  paletteOpen: boolean;
  tweaksOpen: boolean;
  chatVisible: boolean;
  /**
   * Inline chat (Cmd+I) popover state. When `inlineChatOpen` is true the
   * popover is mounted; the anchor describes the screen-space point the
   * popover should orient itself around. A null anchor means "viewport
   * center" — used when no element was focused at trigger time.
   */
  inlineChatOpen: boolean;
  inlineChatAnchor: { x: number; y: number } | null;
  /**
   * Session-scoped flag for the non-blocking API-key banner inside the
   * ChatPanel. Cleared on store reset; not persisted to localStorage so the
   * banner reappears on a fresh dashboard session.
   */
  chatBannerDismissed: boolean;
  /** Settings modal open flag. */
  settingsOpen: boolean;

  /* === chat thread persistence (P1) =================================== */
  /** Index of persisted threads, capped at MAX_THREADS. */
  threads: ChatThread[];
  /** Currently active thread id, or null when no thread is loaded. */
  activeThreadId: string | null;

  /* === chat attachments (click-to-attach context) ===================== */
  /** Pending attachments to be sent with the next user message. */
  chatAttachments: ChatAttachment[];

  /* === saved presets (P2) ============================================= */
  /** Index of starred presets, persisted to localStorage. */
  presets: Preset[];

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
  /** Open a side-by-side compare tab between two run ids. The tab id format
   *  is `compare:<runA>:<runB>`. No-op if either run id is empty. */
  openCompareTab: (runIdA: string, runIdB: string) => void;
  /** POST `/api/cli/run` with the given args; on success open a job tab and
   *  return the assigned `jobId`. Throws on network/validation error. Used
   *  by the legacy CliForm tab path. */
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
  /** Open the inline chat popover. Pass null anchor for viewport-center. */
  openInlineChat: (anchor: { x: number; y: number } | null) => void;
  /** Close the inline chat popover and clear its anchor. */
  closeInlineChat: () => void;
  /** Dismiss the in-chat API-key banner for the current session. */
  dismissChatBanner: () => void;
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

  /* === workspace actions ============================================== */
  /** Set the active CLI for the Workspace form. Pass null to clear. */
  selectActiveCli: (cliId: string | null) => void;
  /**
   * POST /api/cli/run for the active CLI with the supplied args, append a
   * RunRecord to `runHistory`, subscribe the WS, and return the assigned
   * job id. Does NOT open a job tab. Throws if no active CLI is set or
   * required fields are empty.
   */
  runActive: (args: Record<string, unknown>) => Promise<string>;
  /** Cancel + unsubscribe + GC a single run from `runHistory`. */
  removeRun: (runId: string) => void;
  /** Mark a run as pinned (sticks to the top of the Workspace feed). */
  pinRun: (runId: string) => void;
  /** Remove the pin from a run. */
  unpinRun: (runId: string) => void;
  /** Look up a RunRecord by id (helper; small enough that components can use it). */
  getRun: (runId: string) => RunRecord | undefined;
  /**
   * Copy a past run's args into the active form. If the run targets a
   * different CLI, also switches the active CLI. The form picks up the
   * seed on next render and calls `clearActiveFormSeed`.
   */
  editRunArgs: (runId: string) => void;
  /** Clear the transient `activeFormSeed`. */
  clearActiveFormSeed: () => void;

  /* === agent actions ================================================== */
  /** Send a chat message. Starts a new thread on first send, reuses an
   *  existing one otherwise. Opens the WS subscription on thread create. */
  sendChatMessage: (message: string) => Promise<void>;
  /**
   * Approve or reject a pending tool call.
   *
   * `toolCallId` is the id of the card the user clicked. When provided, we
   * validate it against the currently-pending confirmation gate before
   * sending the API call; mismatches are refused (and logged) so a stale
   * card can't approve whatever happens to be pending now. When omitted,
   * the call falls back to `pendingConfirmation.toolCallId`.
   *
   * `options.argsOverride` (P1 spec §5.5) replaces the agent's argv with
   * the user's edits when supplied alongside `approve=true`. The runtime
   * mutates the in-flight ProviderToolCall before resuming so the executed
   * subprocess sees the edited args.
   *
   * `options.autoApproveSession=true` adds the pending tool's name to the
   * per-thread session whitelist; subsequent calls to that tool in this
   * thread bypass the confirmation gate entirely.
   */
  confirmAgent: (
    approve: boolean,
    toolCallId?: string,
    options?: {
      argsOverride?: Record<string, unknown>;
      autoApproveSession?: boolean;
    },
  ) => Promise<void>;
  /** Reset the chat: close socket, clear messages, drop thread id. */
  resetChat: () => void;
  /** Internal: dispatch a single agent WS event into the store. Exported
   *  for tests; not used directly by components. */
  dispatchAgentEvent: (evt: AgentWsEvent) => void;

  /* === thread persistence actions (P1) ================================ */
  /** Read the thread index + active thread from localStorage on boot. */
  loadThreads: () => void;
  /** Persist the current `agent.messages` + index entry for a given thread id. */
  saveThread: (id: string) => void;
  /** Make this the active thread; load its messages from localStorage. */
  loadThread: (id: string) => void;
  /** Remove a thread (index + messages); clear active if it was active. */
  deleteThread: (id: string) => void;
  /** Start a fresh thread; persist the current first if it has messages. */
  newThread: () => void;
  /** Fork a new thread copying messages up to and including `messageId`. */
  branchFromMessage: (messageId: string) => void;

  /* === chat attachment actions ======================================== */
  /** Append an attachment; dedupe by id (no-op if already present). */
  attachToChatInput: (attachment: ChatAttachment) => void;
  /** Remove a single pending attachment by id. */
  removeChatAttachment: (id: string) => void;
  /** Clear the pending attachment list. */
  clearChatAttachments: () => void;

  /* === preset actions (P2) ============================================ */
  /** Re-read the persisted preset list from localStorage. Idempotent. */
  loadPresets: () => void;
  /**
   * Star a (cliId, args) tuple. If a preset with matching cliId+args
   * already exists, returns its id (no duplicate is created). The optional
   * `label` overrides the auto-derived label.
   */
  addPreset: (
    cliId: string,
    args: Record<string, unknown>,
    label?: string,
  ) => string;
  /** Remove a preset by id. No-op if no match. */
  removePreset: (id: string) => void;
  /**
   * Apply a preset: switch the active CLI to the preset's target, and seed
   * the active form with a copy of the preset's args. No-op if the preset
   * was already removed.
   */
  applyPreset: (id: string) => void;
  /**
   * Helper: returns the matching preset id when one already exists for
   * (cliId, args), or null. Used by the star button to render filled vs
   * outlined state without re-deriving the key on every render.
   */
  findPresetByArgs: (
    cliId: string,
    args: Record<string, unknown>,
  ) => string | null;

  /* === promote actions (P2 trace-to-dataset) ========================== */
  /**
   * Promote a subset of a run's rows into a brand-new dataset on disk.
   * Thin wrapper around `api.promoteRunFailures`; lives in the store
   * (rather than the calling component) so future call sites - e.g.
   * a Compare view "promote regressions" button - can share the action
   * and any optimistic state we add later.
   *
   * Throws on non-2xx so the caller can render an inline error.
   */
  promoteRowsToDataset: (
    runId: string,
    rowHashes: string[],
    datasetName?: string,
  ) => Promise<PromoteResponse>;
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

/** Tear down the singleton agent socket if one is attached. No-op otherwise. */
const _closeAgentConn = (): void => {
  if (!agentConn) return;
  agentConn.closed = true;
  try {
    agentConn.socket.close();
  } catch {
    // ignore
  }
  agentConn = null;
};

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

const newRunId = (): string =>
  `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/**
 * Core "POST /api/cli/run + subscribe WS + insert Job placeholder" flow.
 * Used by both the legacy `runCli` (which then opens a tab) and the new
 * `runActive` (Workspace path). Splitting it out means the Workspace path
 * never opens a job tab as a side effect.
 */
const _runCliCore = async (
  cliId: string,
  args: Record<string, unknown>,
  get: () => StoreState,
  set: (
    partial: Partial<StoreState> | ((s: StoreState) => Partial<StoreState>),
  ) => void,
): Promise<string> => {
  const { job_id } = await api.runCli(cliId, args);
  // Insert the placeholder Job entry without opening a tab.
  const existing = get().jobs.get(job_id);
  if (!existing) {
    set((s) => {
      const next = new Map(s.jobs);
      next.set(job_id, {
        id: job_id,
        cmd: `evalyn ${cliId}`,
        cliId,
        status: 'pending',
        lines: [],
      });
      return { jobs: next };
    });
  }
  // Subscribe so streamed events feed the inline Terminal.
  get().subscribeJob(job_id);
  return job_id;
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
  activeCliId: null,
  runHistory: [],
  activeFormSeed: null,

  tweaks: { ...TWEAK_DEFAULTS },
  sidebarView: 'clis',
  bottomTab: 'terminal',
  paletteOpen: false,
  tweaksOpen: false,
  chatVisible: true,
  inlineChatOpen: false,
  inlineChatAnchor: null,
  chatBannerDismissed: false,
  settingsOpen: false,

  threads: [],
  activeThreadId: null,
  chatAttachments: [],

  // Initialize presets from localStorage at store creation so the strip can
  // render on first paint (App.tsx boot effect would arrive a tick later).
  // Failures are absorbed by _readPresets - we never throw out of the
  // factory.
  presets: _readPresets(),

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
    // Drive the Workspace's active form to this CLI so the rendered content
    // matches the opened tab. Chat suggestions land here via SuggestionCard.
    set({ activeCliId: id });
    // Pull prefill args (set by ChatPanel SuggestionCard before openCli)
    // and hand them to the form via the same seed channel editRunArgs uses.
    let prefill: Record<string, unknown> | null = null;
    try {
      if (typeof sessionStorage !== 'undefined') {
        const raw = sessionStorage.getItem(`cli:prefill:${id}`);
        if (raw) {
          prefill = JSON.parse(raw) as Record<string, unknown>;
          sessionStorage.removeItem(`cli:prefill:${id}`);
        }
      }
    } catch {
      // ignore — sessionStorage / JSON failures are non-fatal
    }
    // Always reset the seed so a stale entry from a different CLI does
    // not linger. When no prefill is found, mirror selectActiveCli's
    // auto-seed-from-runHistory behavior so chat suggestions and sidebar
    // clicks behave consistently.
    let seed: { cliId: string; args: Record<string, unknown> } | null = null;
    if (prefill) {
      seed = { cliId: id, args: prefill };
    } else {
      const history = get().runHistory;
      for (let i = history.length - 1; i >= 0; i--) {
        if (history[i].cliId === id) {
          seed = { cliId: id, args: { ...history[i].args } };
          break;
        }
      }
    }
    set({ activeFormSeed: seed });
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

  openCompareTab: (runIdA, runIdB) => {
    const a = (runIdA ?? '').trim();
    const b = (runIdB ?? '').trim();
    if (!a || !b) return;
    const tabId = `compare:${a}:${b}`;
    const shortA = a.length > 8 ? a.slice(0, 8) : a;
    const shortB = b.length > 8 ? b.slice(0, 8) : b;
    get().addTab({
      id: tabId,
      title: `${shortA} vs ${shortB}`,
      kind: 'compare',
      tone: 'var(--accent)',
    });
  },

  runCli: async (cliId, args) => {
    const jobId = await _runCliCore(cliId, args, get, set);
    // Legacy CliForm path: open a job tab as the side effect that the
    // Workspace path explicitly avoids. addTab will activate an existing
    // tab if one is already present.
    get().addTab({
      id: `job:${jobId}`,
      title: jobId,
      kind: 'job',
      tone: 'var(--accent)',
    });
    return jobId;
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
  openInlineChat: (anchor) =>
    set({ inlineChatOpen: true, inlineChatAnchor: anchor }),
  closeInlineChat: () =>
    set({ inlineChatOpen: false, inlineChatAnchor: null }),
  dismissChatBanner: () => set({ chatBannerDismissed: true }),

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

  selectActiveCli: (cliId) => {
    if (cliId == null) {
      set({ activeCliId: null });
      return;
    }
    // Approach D auto-seed: when a CLI is selected and a prior run exists for
    // it, cherry-pick the most recent run's args into the active form. The
    // Workspace's ActiveForm consumes `activeFormSeed` once via
    // `clearActiveFormSeed`. If no prior run exists, leave the form at
    // argparse defaults (null seed).
    const history = get().runHistory;
    let seed: { cliId: string; args: Record<string, unknown> } | null = null;
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].cliId === cliId) {
        seed = { cliId, args: { ...history[i].args } };
        break;
      }
    }
    set({ activeCliId: cliId, activeFormSeed: seed });
  },

  runActive: async (args) => {
    const cliId = get().activeCliId;
    if (!cliId) throw new Error('no active cli selected');
    const cli = get().catalog.find((c) => c.id === cliId);
    if (cli) {
      const missing = cli.params
        .filter((p) => p.required)
        .filter((p) => {
          const v = args[p.name];
          return v === undefined || v === null || v === '' || (Array.isArray(v) && v.length === 0);
        });
      if (missing.length > 0) {
        throw new Error(`missing required: ${missing.map((p) => p.name).join(', ')}`);
      }
    }
    const jobId = await _runCliCore(cliId, args, get, set);
    const record: RunRecord = {
      id: newRunId(),
      cliId,
      args: { ...args },
      jobId,
      startedAt: Date.now(),
      pinned: false,
    };
    set((s) => ({ runHistory: [...s.runHistory, record] }));
    return jobId;
  },

  removeRun: (runId) => {
    const record = get().runHistory.find((r) => r.id === runId);
    if (!record) return;
    // Cancel the underlying job if still in flight; ignore network errors.
    const job = get().jobs.get(record.jobId);
    if (job && (job.status === 'pending' || job.status === 'running')) {
      api.cancelJob(record.jobId).catch(() => {
        /* swallow — best-effort cancellation */
      });
    }
    // Tear down the WS connection.
    get().unsubscribeJob(record.jobId);
    // GC the per-job buffers + run history entry.
    set((s) => {
      const nextJobs = new Map(s.jobs);
      nextJobs.delete(record.jobId);
      const nextEvents = new Map(s.jobEvents);
      nextEvents.delete(record.jobId);
      const nextLastEvent = new Map(s.jobLastEventId);
      nextLastEvent.delete(record.jobId);
      return {
        jobs: nextJobs,
        jobEvents: nextEvents,
        jobLastEventId: nextLastEvent,
        runHistory: s.runHistory.filter((r) => r.id !== runId),
      };
    });
  },

  pinRun: (runId) =>
    set((s) => ({
      runHistory: s.runHistory.map((r) => (r.id === runId ? { ...r, pinned: true } : r)),
    })),

  unpinRun: (runId) =>
    set((s) => ({
      runHistory: s.runHistory.map((r) => (r.id === runId ? { ...r, pinned: false } : r)),
    })),

  getRun: (runId) => get().runHistory.find((r) => r.id === runId),

  editRunArgs: (runId) => {
    const record = get().runHistory.find((r) => r.id === runId);
    if (!record) return;
    set({
      activeCliId: record.cliId,
      activeFormSeed: { cliId: record.cliId, args: { ...record.args } },
    });
  },

  clearActiveFormSeed: () => set({ activeFormSeed: null }),

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

  confirmAgent: async (approve, toolCallId, options) => {
    const { agent } = get();
    if (!agent.threadId || !agent.pendingConfirmation) return;
    const pendingId = agent.pendingConfirmation.toolCallId;
    // Validate the caller's tool_call_id against the pending gate. Stale
    // ToolCallCards (e.g. after a WS replay or rapid back-to-back proposals)
    // would otherwise approve whatever is currently pending — fix for
    // KNOWN_ISSUES #4. Callers that omit the id (legacy tests) fall back
    // to pendingId.
    const targetId = toolCallId ?? pendingId;
    if (toolCallId != null && toolCallId !== pendingId) {
      console.warn(
        `confirmAgent: stale tool_call_id ${toolCallId} (pending ${pendingId}); refusing to send`,
      );
      return;
    }
    // Snapshot pre-optimistic state so we can roll back if the server
    // rejects the confirm (e.g. 409 stale tool_call_id) — otherwise the
    // toolCall card stays "running" forever with no recovery path.
    // Snapshot the targeted tool call's prior args + status (NOT the
    // entire messages array) so if a WS event arrives between the
    // optimistic set and the catch, we don't clobber it during rollback.
    const previousPending = agent.pendingConfirmation;
    const previousStatus = agent.status;
    const previousCard = agent.messages.find(
      (m) => m.toolCall && m.toolCall.id === targetId,
    )?.toolCall;
    const previousCardStatus = previousCard?.status ?? 'awaiting_confirmation';
    const previousCardArgs = previousCard?.args;
    // Optimistically clear the pending confirmation; the server will push
    // tool_call_running / tool_call_complete (or error) next. When
    // args_override is supplied, also patch the optimistic args on the
    // tool-call card so the user sees their edited argv immediately.
    const editedArgs = options?.argsOverride;
    set((s) => ({
      agent: {
        ...s.agent,
        pendingConfirmation: null,
        status: approve ? 'streaming' : 'idle',
        messages: s.agent.messages.map((m) =>
          m.toolCall && m.toolCall.id === targetId
            ? {
                ...m,
                toolCall: {
                  ...m.toolCall,
                  status: approve ? 'running' : 'rejected',
                  ...(approve && editedArgs ? { args: editedArgs } : {}),
                },
              }
            : m,
        ),
      },
    }));
    try {
      await api.confirmAgentTool(agent.threadId, approve, targetId, options);
    } catch (err) {
      // Roll back ONLY the targeted tool-call card's status (and args if
      // we had patched them). Use a fresh setter that maps over the
      // CURRENT messages so any WS events that arrived during the in-
      // flight POST stay applied — restoring `previousMessages` whole-
      // sale would silently discard them.
      set((s) => ({
        agent: {
          ...s.agent,
          pendingConfirmation: previousPending,
          status: previousStatus,
          messages: s.agent.messages.map((m) =>
            m.toolCall && m.toolCall.id === targetId
              ? {
                  ...m,
                  toolCall: {
                    ...m.toolCall,
                    status: previousCardStatus,
                    ...(previousCardArgs !== undefined
                      ? { args: previousCardArgs }
                      : {}),
                  },
                }
              : m,
          ),
          error: {
            kind: 'network',
            message: err instanceof Error ? err.message : String(err),
          },
        },
      }));
    }
  },

  resetChat: () => {
    _closeAgentConn();
    set({ agent: { ...initialAgent } });
  },

  dispatchAgentEvent: (evt) => {
    _dispatchAgentEvent(evt, get, set);
  },

  /* === thread persistence actions ==================================== */

  loadThreads: () => {
    const threads = _readThreadIndex();
    const activeId = _readActiveThreadId();
    // If an active thread id was persisted, hydrate its messages too.
    let agentPatch: Partial<AgentState> = {};
    if (activeId && threads.some((t) => t.id === activeId)) {
      const messages = _readThreadMessages(activeId);
      agentPatch = { messages, threadId: activeId };
    }
    set((s) => ({
      threads,
      activeThreadId:
        activeId && threads.some((t) => t.id === activeId) ? activeId : null,
      agent: { ...s.agent, ...agentPatch },
    }));
  },

  saveThread: (id) => {
    const { agent, threads } = get();
    const messages = agent.messages;
    if (messages.length === 0) return;
    const now = Date.now();
    const title = _deriveThreadTitle(messages);
    const existingIdx = threads.findIndex((t) => t.id === id);
    let nextThreads: ChatThread[];
    if (existingIdx >= 0) {
      const prior = threads[existingIdx];
      const updated: ChatThread = {
        ...prior,
        title,
        modifiedAt: now,
        messageCount: messages.length,
      };
      nextThreads = [...threads];
      nextThreads[existingIdx] = updated;
    } else {
      const created: ChatThread = {
        id,
        title,
        createdAt: now,
        modifiedAt: now,
        messageCount: messages.length,
      };
      nextThreads = _capThreadIndex([...threads, created]);
    }
    _writeThreadMessages(id, messages);
    _writeThreadIndex(nextThreads);
    set({ threads: nextThreads });
  },

  loadThread: (id) => {
    const { threads, activeThreadId, agent } = get();
    if (!threads.some((t) => t.id === id)) return;
    // Persist the currently active thread before swapping (best-effort).
    if (activeThreadId && activeThreadId !== id && agent.messages.length > 0) {
      get().saveThread(activeThreadId);
    }
    const messages = _readThreadMessages(id);
    // Switching thread invalidates the live WS.
    _closeAgentConn();
    _writeActiveThreadId(id);
    set((s) => ({
      activeThreadId: id,
      agent: {
        ...s.agent,
        messages,
        threadId: id,
        status: 'idle',
        pendingConfirmation: null,
        error: null,
      },
    }));
  },

  deleteThread: (id) => {
    const { threads, activeThreadId } = get();
    const nextThreads = threads.filter((t) => t.id !== id);
    _deleteThreadMessages(id);
    _writeThreadIndex(nextThreads);
    if (activeThreadId === id) {
      _writeActiveThreadId(null);
      _closeAgentConn();
      set({
        threads: nextThreads,
        activeThreadId: null,
        agent: { ...initialAgent },
      });
    } else {
      set({ threads: nextThreads });
    }
  },

  newThread: () => {
    const { activeThreadId, agent } = get();
    // Persist whatever is on screen before clearing.
    if (activeThreadId && agent.messages.length > 0) {
      get().saveThread(activeThreadId);
    }
    _closeAgentConn();
    _writeActiveThreadId(null);
    set({
      activeThreadId: null,
      agent: { ...initialAgent },
    });
  },

  attachToChatInput: (attachment) =>
    set((s) => {
      // Dedupe by id — clicking attach a second time on the same row is
      // a silent no-op rather than a double-chip.
      if (s.chatAttachments.some((a) => a.id === attachment.id)) {
        return {};
      }
      return { chatAttachments: [...s.chatAttachments, attachment] };
    }),

  removeChatAttachment: (id) =>
    set((s) => ({
      chatAttachments: s.chatAttachments.filter((a) => a.id !== id),
    })),

  clearChatAttachments: () => set({ chatAttachments: [] }),

  /* === preset actions (P2) ============================================ */

  loadPresets: () => {
    set({ presets: _readPresets() });
  },

  addPreset: (cliId, args, label) => {
    const snapshot: Record<string, unknown> = { ...args };
    const snapshotKey = _argsKey(snapshot);
    const existing = get().presets.find(
      (p) => p.cliId === cliId && _argsKey(p.args) === snapshotKey,
    );
    if (existing) return existing.id;
    const preset: Preset = {
      id: _newPresetId(),
      cliId,
      label: label?.trim() || _deriveLabel(cliId, snapshot),
      args: snapshot,
      createdAt: Date.now(),
    };
    // Cap at MAX_PRESETS - drop the oldest by createdAt.
    let next = [...get().presets, preset];
    if (next.length > MAX_PRESETS) {
      next = [...next].sort((a, b) => b.createdAt - a.createdAt).slice(0, MAX_PRESETS);
    }
    _writePresets(next);
    set({ presets: next });
    return preset.id;
  },

  removePreset: (id) => {
    const next = get().presets.filter((p) => p.id !== id);
    if (next.length === get().presets.length) return;
    _writePresets(next);
    set({ presets: next });
  },

  applyPreset: (id) => {
    const preset = get().presets.find((p) => p.id === id);
    if (!preset) return;
    set({
      activeCliId: preset.cliId,
      activeFormSeed: { cliId: preset.cliId, args: { ...preset.args } },
    });
  },

  findPresetByArgs: (cliId, args) => {
    const key = _argsKey(args);
    const match = get().presets.find(
      (p) => p.cliId === cliId && _argsKey(p.args) === key,
    );
    return match?.id ?? null;
  },

  /* === promote actions (P2) =========================================== */

  promoteRowsToDataset: async (runId, rowHashes, datasetName) => {
    // The store doesn't cache the result on success - components decide
    // what to do (toast, refresh runs, etc). Errors propagate so the
    // caller can render an inline error in its modal.
    return api.promoteRunFailures(runId, rowHashes, datasetName);
  },

  branchFromMessage: (messageId) => {
    const { agent, activeThreadId } = get();
    const idx = agent.messages.findIndex((m) => m.id === messageId);
    if (idx < 0) return;
    const slice = agent.messages.slice(0, idx + 1);
    if (slice.length === 0) return;
    // Persist the source thread first (so its current state survives).
    if (activeThreadId && agent.messages.length > 0) {
      get().saveThread(activeThreadId);
    }
    // Mint a new thread, write its messages, and activate it.
    const newId = _newThreadId();
    const now = Date.now();
    const title = _deriveThreadTitle(slice);
    const created: ChatThread = {
      id: newId,
      title,
      createdAt: now,
      modifiedAt: now,
      messageCount: slice.length,
    };
    const nextThreads = _capThreadIndex([...get().threads, created]);
    _writeThreadMessages(newId, slice);
    _writeThreadIndex(nextThreads);
    _writeActiveThreadId(newId);
    // The new thread has no server-side counterpart yet.
    _closeAgentConn();
    set((s) => ({
      threads: nextThreads,
      activeThreadId: newId,
      agent: {
        ...s.agent,
        messages: slice,
        // The new thread is client-only until the user sends something; null
        // threadId triggers the POST /api/agent/chat path on next send.
        threadId: null,
        status: 'idle',
        pendingConfirmation: null,
        error: null,
      },
    }));
  },
}));

/**
 * Auto-save: persist the active thread on every change to `agent.messages`.
 * Throttled to once per 1s via a trailing-edge timer. Lives outside the
 * store so React doesn't re-render when the timer fires. The first user
 * message of a brand-new chat triggers thread creation here too — when no
 * `activeThreadId` is set but `agent.messages` is non-empty, we mint a new
 * thread id and adopt it.
 */
let _autoSaveTimer: ReturnType<typeof setTimeout> | null = null;
const _AUTO_SAVE_DELAY_MS = 1000;

useStore.subscribe((state, prev) => {
  if (state.agent.messages === prev.agent.messages) return;
  if (state.agent.messages.length === 0) return;
  // Capture the thread id (and the messages reference) at SCHEDULE time,
  // not FIRE time. If the user calls loadThread(B) within the debounce
  // window, the timer would otherwise persist the new thread's messages
  // under the new thread id, silently overwriting whatever was there —
  // or worse, mint a fresh id and orphan the prior thread mid-load.
  const scheduledId = state.activeThreadId;
  const scheduledMessages = state.agent.messages;
  if (_autoSaveTimer) clearTimeout(_autoSaveTimer);
  _autoSaveTimer = setTimeout(() => {
    _autoSaveTimer = null;
    // Re-read current state — if the user switched threads, the captured
    // id no longer matches and we must skip; the new thread has its own
    // pending save (or will, on the next message change).
    const current = useStore.getState();
    if (scheduledId !== null && scheduledId !== current.activeThreadId) return;
    if (current.agent.messages !== scheduledMessages) {
      // Messages changed since schedule — let the next subscribe tick
      // schedule a fresh timer rather than persisting stale state.
      return;
    }
    if (current.agent.messages.length === 0) return;
    let id = current.activeThreadId;
    if (!id) {
      // Brand-new chat: mint an id and adopt it. Safe here because we
      // already verified current.activeThreadId still matches the
      // scheduledId (both null in this branch).
      id = _newThreadId();
      _writeActiveThreadId(id);
      useStore.setState({ activeThreadId: id });
    }
    useStore.getState().saveThread(id);
  }, _AUTO_SAVE_DELAY_MS);
});

/**
 * Merge fresh CLI defaults with a possibly-stale user-values map, dropping
 * any keys the CLI no longer declares. Exported so the Workspace form can
 * reuse the same merge rule when the catalog refreshes mid-session.
 */
export const mergeFormValues = (
  cli: CliSchema,
  userValues: Record<string, unknown> | undefined | null,
): Record<string, unknown> => {
  const fresh = cliDefaultValues(cli);
  const merged: Record<string, unknown> = { ...fresh };
  if (userValues) {
    for (const p of cli.params) {
      if (Object.prototype.hasOwnProperty.call(userValues, p.name)) {
        merged[p.name] = userValues[p.name];
      }
    }
  }
  return merged;
};

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
  _closeAgentConn();
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
              sideEffects: evt.side_effects,
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
  if (_autoSaveTimer) {
    clearTimeout(_autoSaveTimer);
    _autoSaveTimer = null;
  }
  // Best-effort: drop the persisted thread index + active id so each test
  // gets a clean storage namespace.
  const ls = _safeStorage();
  if (ls) {
    try {
      // Wipe only keys we own.
      const toRemove: string[] = [];
      for (let i = 0; i < ls.length; i++) {
        const k = ls.key(i);
        if (
          k &&
          (k === THREAD_INDEX_KEY ||
            k === THREAD_ACTIVE_KEY ||
            k === PRESETS_KEY ||
            k.startsWith(THREAD_MSGS_PREFIX))
        ) {
          toRemove.push(k);
        }
      }
      for (const k of toRemove) ls.removeItem(k);
    } catch {
      // ignore
    }
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
    activeCliId: null,
    runHistory: [],
    activeFormSeed: null,
    tweaks: { ...TWEAK_DEFAULTS },
    sidebarView: 'clis',
    bottomTab: 'terminal',
    paletteOpen: false,
    tweaksOpen: false,
    chatVisible: true,
    inlineChatOpen: false,
    inlineChatAnchor: null,
    chatBannerDismissed: false,
    settingsOpen: false,
    threads: [],
    activeThreadId: null,
    chatAttachments: [],
    presets: [],
  });
};

/**
 * Serialize a list of pending attachments into a prefix to prepend to the
 * user's chat text. Each attachment becomes a bracketed reference token the
 * agent's system prompt understands (e.g. `[run-id: abc123]`). Returns an
 * empty string when there are no attachments so callers can safely
 * `prefix + text` without conditional plumbing.
 */
export const serializeChatAttachments = (
  attachments: ChatAttachment[],
  existingText: string = '',
): string => {
  if (!attachments || attachments.length === 0) return '';
  // Skip any token the user already typed into the textarea — otherwise
  // copy-pasting from a prior turn (or composing the bracket form by hand)
  // would land twice in the outgoing message.
  const tokens = attachments
    .map((a) => {
      if (a.kind === 'metric') return `[metric-id: ${a.ref}]`;
      if (a.kind === 'file') return `[file-path: ${a.ref}]`;
      return `[run-id: ${a.ref}]`;
    })
    .filter((tok) => !existingText.includes(tok));
  if (tokens.length === 0) return '';
  return `${tokens.join(' ')}\n`;
};

/**
 * Compute the empty-state chip suggestions shown in the ChatPanel before
 * the first message has been sent. Cold-start (no runs) returns three
 * onboarding prompts; warm-start returns up-to-three data-grounded prompts
 * derived from `runs`. Pure function for easy unit testing.
 */
export const chatEmptyStateSuggestions = (runs: RunMeta[]): string[] => {
  // Assumes the backend `/api/runs` returns runs newest-first. RunMeta.at
  // is human-friendly text (not ISO), so we cannot sort defensively here.
  // If the backend contract changes, the warm-start chips will reference
  // the wrong run id — guard at the source instead.
  const latest = runs?.[0];
  if (!latest) {
    return [
      'Walk me through evalyn',
      'Show me what evalyn can do',
      'Help me instrument my agent',
    ];
  }
  const out: string[] = [];
  if (typeof latest.delta === 'number' && latest.delta < 0) {
    out.push(`Why did pass rate drop on ${latest.id}?`);
  }
  if (runs.length >= 2) out.push('Compare my last 2 run-eval runs');
  out.push(`Cluster failures in ${latest.id}`);
  return out;
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
