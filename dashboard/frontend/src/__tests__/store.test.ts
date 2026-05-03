/**
 * Vitest unit tests for the Zustand store.
 *
 * Covers initial state, tab actions, tweak patching, and job upserts.
 */

import { beforeEach, describe, expect, test, vi, afterEach } from 'vitest';
import { useStore, __resetStore, TWEAK_DEFAULTS } from '../store';
import type { Job, Tab } from '../types/jobs';
import type { CliSchema } from '../types/catalog';

beforeEach(() => {
  __resetStore();
});

describe('initial state', () => {
  test('all slices empty', () => {
    const s = useStore.getState();
    expect(s.catalog).toEqual([]);
    expect(s.tabs).toEqual([]);
    expect(s.activeTabId).toBeNull();
    expect(s.jobs.size).toBe(0);
    expect(s.fileTree).toEqual([]);
    expect(s.runs).toEqual([]);
  });

  test('agent slice initialized', () => {
    const s = useStore.getState();
    expect(s.agent.threadId).toBeNull();
    expect(s.agent.messages).toEqual([]);
    expect(s.agent.status).toBe('idle');
    expect(s.agent.pendingConfirmation).toBeNull();
  });

  test('settings slice initialized', () => {
    const s = useStore.getState();
    // Lane C2 seeds the three known providers so the SettingsModal can
    // render rows even before /api/settings completes its first fetch.
    expect(Object.keys(s.settings.providers).sort()).toEqual([
      'anthropic',
      'ollama',
      'openai',
    ]);
    expect(s.settings.providers.openai.testStatus).toBe('untested');
    expect(s.settings.providers.openai.hasKey).toBe(false);
    expect(s.settings.active).toBeNull();
  });

  test('tweaks default to mock parity', () => {
    const s = useStore.getState();
    expect(s.tweaks).toEqual(TWEAK_DEFAULTS);
  });

  test('ui flags default sane', () => {
    const s = useStore.getState();
    expect(s.sidebarView).toBe('clis');
    expect(s.bottomTab).toBe('terminal');
    expect(s.paletteOpen).toBe(false);
    expect(s.tweaksOpen).toBe(false);
    expect(s.chatVisible).toBe(true);
  });
});

describe('tab actions', () => {
  test('addTab updates tabs and activeTabId', () => {
    const tab: Tab = { id: 'cli:run-eval', title: 'run-eval', kind: 'cli' };
    useStore.getState().addTab(tab);
    const s = useStore.getState();
    expect(s.tabs).toHaveLength(1);
    expect(s.tabs[0]).toEqual(tab);
    expect(s.activeTabId).toBe('cli:run-eval');
  });

  test('addTab is idempotent on existing id (just activates)', () => {
    const tab: Tab = { id: 'cli:run-eval', title: 'run-eval', kind: 'cli' };
    useStore.getState().addTab(tab);
    useStore.getState().setActiveTab(null);
    useStore.getState().addTab(tab);
    const s = useStore.getState();
    expect(s.tabs).toHaveLength(1);
    expect(s.activeTabId).toBe('cli:run-eval');
  });

  test('openCli builds the cli: prefixed id', () => {
    useStore.getState().openCli('run-eval');
    const s = useStore.getState();
    expect(s.tabs).toHaveLength(1);
    expect(s.tabs[0].id).toBe('cli:run-eval');
    expect(s.tabs[0].kind).toBe('cli');
    expect(s.activeTabId).toBe('cli:run-eval');
  });

  test('openFile derives kind from extension', () => {
    useStore.getState().openFile('82dddcc3.run');
    expect(useStore.getState().tabs[0].kind).toBe('run');
    useStore.getState().openFile('config.yaml');
    expect(useStore.getState().tabs[1].kind).toBe('yaml');
    useStore.getState().openFile('readme.md');
    expect(useStore.getState().tabs[2].kind).toBe('file');
  });

  test('closeTab activates previous tab', () => {
    useStore.getState().openCli('run-eval');
    useStore.getState().openCli('calibrate');
    useStore.getState().openCli('annotate');
    expect(useStore.getState().activeTabId).toBe('cli:annotate');

    useStore.getState().closeTab('cli:annotate');
    const s = useStore.getState();
    expect(s.tabs).toHaveLength(2);
    expect(s.activeTabId).toBe('cli:calibrate');
  });

  test('closeTab returns null when last tab closes', () => {
    useStore.getState().openCli('run-eval');
    useStore.getState().closeTab('cli:run-eval');
    const s = useStore.getState();
    expect(s.tabs).toEqual([]);
    expect(s.activeTabId).toBeNull();
  });

  test('closeTab on inactive tab keeps active id', () => {
    useStore.getState().openCli('run-eval');
    useStore.getState().openCli('calibrate');
    useStore.getState().closeTab('cli:run-eval');
    const s = useStore.getState();
    expect(s.tabs).toHaveLength(1);
    expect(s.activeTabId).toBe('cli:calibrate');
  });
});

describe('tweaks', () => {
  test('setTweak patches one field', () => {
    useStore.getState().setTweak('theme', 'dark');
    expect(useStore.getState().tweaks.theme).toBe('dark');
    expect(useStore.getState().tweaks.chatPlacement).toBe(TWEAK_DEFAULTS.chatPlacement);
  });

  test('setTweak handles boolean fields', () => {
    useStore.getState().setTweak('sidebarCollapsed', true);
    expect(useStore.getState().tweaks.sidebarCollapsed).toBe(true);
  });
});

describe('jobs', () => {
  test('upsertJob inserts new job', () => {
    const job: Job = { id: 'j-1', cmd: 'evalyn list-runs', status: 'running' };
    useStore.getState().upsertJob(job);
    const s = useStore.getState();
    expect(s.jobs.size).toBe(1);
    expect(s.jobs.get('j-1')).toEqual(job);
  });

  test('upsertJob replaces existing job by id', () => {
    useStore.getState().upsertJob({ id: 'j-1', cmd: 'evalyn list-runs', status: 'running' });
    useStore.getState().upsertJob({
      id: 'j-1',
      cmd: 'evalyn list-runs',
      status: 'complete',
      exitCode: 0,
    });
    const s = useStore.getState();
    expect(s.jobs.size).toBe(1);
    expect(s.jobs.get('j-1')?.status).toBe('complete');
    expect(s.jobs.get('j-1')?.exitCode).toBe(0);
  });
});

describe('workspace actions', () => {
  const cliFixture: CliSchema = {
    id: 'run-eval',
    name: 'run-eval',
    group: 'Eval',
    blurb: 'Run an evaluation.',
    params: [
      { name: 'dataset', kind: 'path', required: true },
      { name: 'workers', kind: 'number', default: 4 },
      { name: 'verbose', kind: 'bool', default: false },
    ],
  };

  beforeEach(() => {
    useStore.getState().setCatalog([cliFixture]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('selectActiveCli sets the active id', () => {
    expect(useStore.getState().activeCliId).toBeNull();
    useStore.getState().selectActiveCli('run-eval');
    expect(useStore.getState().activeCliId).toBe('run-eval');
    useStore.getState().selectActiveCli(null);
    expect(useStore.getState().activeCliId).toBeNull();
  });

  test('selectActiveCli with no prior run leaves activeFormSeed null', () => {
    expect(useStore.getState().activeFormSeed).toBeNull();
    useStore.getState().selectActiveCli('run-eval');
    expect(useStore.getState().activeFormSeed).toBeNull();
    expect(useStore.getState().activeCliId).toBe('run-eval');
  });

  test('selectActiveCli with prior run seeds activeFormSeed from latest matching run', () => {
    useStore.setState({
      runHistory: [
        // Older run for the same CLI - should be ignored.
        {
          id: 'r-0',
          cliId: 'run-eval',
          args: { dataset: './old.jsonl', workers: 2 },
          jobId: 'j-0',
          startedAt: 1,
          pinned: false,
        },
        // Run for a different CLI - should be ignored.
        {
          id: 'r-x',
          cliId: 'calibrate',
          args: { judge: 'gpt-3.5' },
          jobId: 'j-x',
          startedAt: 2,
          pinned: false,
        },
        // Newest run for run-eval - this one should be picked.
        {
          id: 'r-1',
          cliId: 'run-eval',
          args: { dataset: './newest.jsonl', workers: 16 },
          jobId: 'j-1',
          startedAt: 3,
          pinned: false,
        },
      ],
    });
    useStore.getState().selectActiveCli('run-eval');
    expect(useStore.getState().activeCliId).toBe('run-eval');
    expect(useStore.getState().activeFormSeed).toEqual({
      cliId: 'run-eval',
      args: { dataset: './newest.jsonl', workers: 16 },
    });
    // Seed args are cloned, not aliased to the run record.
    const seed = useStore.getState().activeFormSeed!;
    seed.args.dataset = 'mutated';
    expect(useStore.getState().runHistory[2].args.dataset).toBe('./newest.jsonl');
  });

  test('selectActiveCli with no matching prior run clears any stale seed', () => {
    // Pre-existing seed targeting a different CLI - should be replaced with null
    // when no run-eval history exists.
    useStore.setState({
      activeFormSeed: {
        cliId: 'calibrate',
        args: { judge: 'gpt-4' },
      },
      runHistory: [
        {
          id: 'r-x',
          cliId: 'calibrate',
          args: { judge: 'gpt-4' },
          jobId: 'j-x',
          startedAt: 1,
          pinned: false,
        },
      ],
    });
    useStore.getState().selectActiveCli('run-eval');
    expect(useStore.getState().activeFormSeed).toBeNull();
  });

  test('runActive POSTs /api/cli/run, captures args, opens NO job tab', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ job_id: 'j-77' }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    useStore.getState().selectActiveCli('run-eval');
    const jobId = await useStore.getState().runActive({
      dataset: './x.jsonl',
      workers: 4,
      verbose: false,
    });

    expect(jobId).toBe('j-77');
    const history = useStore.getState().runHistory;
    expect(history).toHaveLength(1);
    expect(history[0].cliId).toBe('run-eval');
    expect(history[0].jobId).toBe('j-77');
    expect(history[0].args.dataset).toBe('./x.jsonl');
    expect(history[0].pinned).toBe(false);
    expect(fetchMock).toHaveBeenCalledOnce();
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.cli_id).toBe('run-eval');
    // No job tab — the Workspace path stays out of the tab strip.
    const tabs = useStore.getState().tabs;
    expect(tabs.find((t) => t.id === 'job:j-77')).toBeFalsy();
    // Job placeholder inserted so the Terminal can render.
    expect(useStore.getState().jobs.get('j-77')?.status).toBe('pending');
  });

  test('runActive throws when required field empty', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    useStore.getState().selectActiveCli('run-eval');
    await expect(useStore.getState().runActive({ workers: 4, verbose: false }))
      .rejects.toThrow(/dataset/);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(useStore.getState().runHistory).toEqual([]);
  });

  test('runActive throws when no active CLI is set', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    await expect(useStore.getState().runActive({})).rejects.toThrow(/no active cli/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('runActive appends successive RunRecords flat (newest at end)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true, status: 200, statusText: 'OK',
        json: async () => ({ job_id: 'j-1' }), text: async () => '',
      })
      .mockResolvedValueOnce({
        ok: true, status: 200, statusText: 'OK',
        json: async () => ({ job_id: 'j-2' }), text: async () => '',
      });
    vi.stubGlobal('fetch', fetchMock);

    useStore.getState().selectActiveCli('run-eval');
    await useStore.getState().runActive({ dataset: './a.jsonl' });
    await useStore.getState().runActive({ dataset: './b.jsonl' });
    const history = useStore.getState().runHistory;
    expect(history.map((r) => r.jobId)).toEqual(['j-1', 'j-2']);
  });

  test('removeRun cancels in-flight, unsubscribes, and GCs the entry', async () => {
    // First push a run record manually so we don't have to round-trip a fetch.
    const cancelMock = vi.fn().mockResolvedValue({
      ok: true, status: 200, statusText: 'OK',
      json: async () => ({ ok: true }), text: async () => '',
    });
    vi.stubGlobal('fetch', cancelMock);

    useStore.setState({
      runHistory: [
        {
          id: 'r-1',
          cliId: 'run-eval',
          args: { dataset: './x.jsonl' },
          jobId: 'j-99',
          startedAt: 1,
          pinned: false,
        },
      ],
      jobs: new Map([
        ['j-99', { id: 'j-99', cmd: 'evalyn run-eval', status: 'running' as const }],
      ]),
      jobEvents: new Map([['j-99', [{ kind: 'stdout', text: 'hi' } as const]]]),
      jobLastEventId: new Map([['j-99', 5]]),
    });

    useStore.getState().removeRun('r-1');
    // History entry gone.
    expect(useStore.getState().runHistory).toEqual([]);
    // Per-job buffers GC'd.
    expect(useStore.getState().jobs.get('j-99')).toBeUndefined();
    expect(useStore.getState().jobEvents.get('j-99')).toBeUndefined();
    expect(useStore.getState().jobLastEventId.get('j-99')).toBeUndefined();
    // POST /api/jobs/j-99/cancel was issued.
    expect(cancelMock).toHaveBeenCalled();
    expect(cancelMock.mock.calls[0][0]).toBe('/api/jobs/j-99/cancel');
  });

  test('removeRun does not cancel when job already complete', () => {
    const cancelMock = vi.fn();
    vi.stubGlobal('fetch', cancelMock);

    useStore.setState({
      runHistory: [
        {
          id: 'r-1',
          cliId: 'run-eval',
          args: {},
          jobId: 'j-done',
          startedAt: 1,
          pinned: false,
        },
      ],
      jobs: new Map([
        ['j-done', { id: 'j-done', cmd: '', status: 'complete' as const, exitCode: 0 }],
      ]),
    });
    useStore.getState().removeRun('r-1');
    expect(cancelMock).not.toHaveBeenCalled();
    expect(useStore.getState().runHistory).toEqual([]);
  });

  test('pinRun and unpinRun toggle the pinned flag', () => {
    useStore.setState({
      runHistory: [
        {
          id: 'r-1',
          cliId: 'run-eval',
          args: {},
          jobId: 'j-1',
          startedAt: 1,
          pinned: false,
        },
      ],
    });
    useStore.getState().pinRun('r-1');
    expect(useStore.getState().runHistory[0].pinned).toBe(true);
    useStore.getState().unpinRun('r-1');
    expect(useStore.getState().runHistory[0].pinned).toBe(false);
  });

  test('editRunArgs seeds activeFormSeed and switches active CLI', () => {
    useStore.setState({
      runHistory: [
        {
          id: 'r-1',
          cliId: 'calibrate',
          args: { dataset: './a.jsonl', workers: 8 },
          jobId: 'j-1',
          startedAt: 1,
          pinned: false,
        },
      ],
      activeCliId: 'run-eval',
    });
    useStore.getState().editRunArgs('r-1');
    expect(useStore.getState().activeCliId).toBe('calibrate');
    expect(useStore.getState().activeFormSeed).toEqual({
      cliId: 'calibrate',
      args: { dataset: './a.jsonl', workers: 8 },
    });
    useStore.getState().clearActiveFormSeed();
    expect(useStore.getState().activeFormSeed).toBeNull();
  });
});

describe('schema-drift safety', () => {
  test('mergeFormValues seeds defaults for new params and keeps user values', async () => {
    const { mergeFormValues } = await import('../store');
    const cli: CliSchema = {
      id: 'run-eval',
      name: 'run-eval',
      group: 'Eval',
      blurb: '',
      params: [
        { name: 'dataset', kind: 'path', required: true },
        { name: 'workers', kind: 'number', default: 4 },
        // New field that wasn't present in the cached values:
        { name: 'judge', kind: 'select', options: ['a', 'b'], default: 'a' },
      ],
    };
    const cached = { dataset: './x.jsonl', workers: 8 };
    const merged = mergeFormValues(cli, cached);
    expect(merged).toEqual({
      dataset: './x.jsonl',
      workers: 8,
      judge: 'a',
    });
  });

  test('mergeFormValues drops keys the schema no longer declares', async () => {
    const { mergeFormValues } = await import('../store');
    const cli: CliSchema = {
      id: 'run-eval',
      name: 'run-eval',
      group: 'Eval',
      blurb: '',
      params: [{ name: 'dataset', kind: 'path', required: true }],
    };
    const cached = { dataset: './x.jsonl', removed_param: 'oops' };
    const merged = mergeFormValues(cli, cached);
    expect(merged).toEqual({ dataset: './x.jsonl' });
    expect(merged.removed_param).toBeUndefined();
  });

  test('mergeFormValues with null cached values returns plain defaults', async () => {
    const { mergeFormValues } = await import('../store');
    const cli: CliSchema = {
      id: 'run-eval',
      name: 'run-eval',
      group: 'Eval',
      blurb: '',
      params: [
        { name: 'dataset', kind: 'path', required: true },
        { name: 'workers', kind: 'number', default: 4 },
      ],
    };
    expect(mergeFormValues(cli, null)).toEqual({ dataset: '', workers: 4 });
  });
});

describe('ui state setters', () => {
  test('setSidebarView updates view', () => {
    useStore.getState().setSidebarView('clis');
    expect(useStore.getState().sidebarView).toBe('clis');
  });

  test('setBottomTab updates tab', () => {
    useStore.getState().setBottomTab('jobs');
    expect(useStore.getState().bottomTab).toBe('jobs');
  });

  test('setPaletteOpen toggles flag', () => {
    useStore.getState().setPaletteOpen(true);
    expect(useStore.getState().paletteOpen).toBe(true);
  });
});

describe('chat thread persistence', () => {
  test('saveThread + loadThread roundtrips messages via localStorage', () => {
    const messages = [
      { id: 'u1', role: 'user' as const, text: 'why did this regress?' },
      { id: 'a1', role: 'assistant' as const, text: 'Looking now.' },
    ];
    // Seed the agent slice with messages, then save under a fresh thread id.
    useStore.setState((s) => ({ agent: { ...s.agent, messages } }));
    useStore.getState().saveThread('t-test-1');
    // Index entry exists with the correct title (derived from first user msg).
    const threads = useStore.getState().threads;
    expect(threads).toHaveLength(1);
    expect(threads[0].id).toBe('t-test-1');
    expect(threads[0].title).toBe('why did this regress?');
    expect(threads[0].messageCount).toBe(2);
    // Per-thread messages persisted to localStorage.
    const stored = JSON.parse(
      localStorage.getItem('evalyn:threads:msgs:t-test-1') ?? 'null',
    );
    expect(stored).toEqual(messages);
    // Wipe in-memory agent state and re-load — messages should round-trip.
    useStore.setState((s) => ({
      agent: { ...s.agent, messages: [], threadId: null },
    }));
    useStore.getState().loadThread('t-test-1');
    const after = useStore.getState();
    expect(after.activeThreadId).toBe('t-test-1');
    expect(after.agent.threadId).toBe('t-test-1');
    expect(after.agent.messages).toEqual(messages);
  });

  test('branchFromMessage forks a new thread up to and including the named message', () => {
    const messages = [
      { id: 'u1', role: 'user' as const, text: 'first question' },
      { id: 'a1', role: 'assistant' as const, text: 'first answer' },
      { id: 'u2', role: 'user' as const, text: 'follow up' },
      { id: 'a2', role: 'assistant' as const, text: 'follow up answer' },
    ];
    // Seed the source thread.
    useStore.setState((s) => ({
      agent: { ...s.agent, messages, threadId: 't-source' },
      activeThreadId: 't-source',
    }));
    // Branch from the first assistant reply — fork should contain u1 + a1 only.
    useStore.getState().branchFromMessage('a1');
    const after = useStore.getState();
    // A new thread is now active (id differs from source).
    expect(after.activeThreadId).not.toBe('t-source');
    expect(after.activeThreadId).not.toBeNull();
    // Active messages are exactly the slice up to and including a1.
    expect(after.agent.messages.map((m) => m.id)).toEqual(['u1', 'a1']);
    // Both source and forked thread are in the index.
    const ids = after.threads.map((t) => t.id).sort();
    expect(ids.length).toBe(2);
    expect(ids).toContain('t-source');
    expect(ids).toContain(after.activeThreadId!);
    // Forked thread's persisted messages match the slice.
    const stored = JSON.parse(
      localStorage.getItem(`evalyn:threads:msgs:${after.activeThreadId}`) ?? 'null',
    );
    expect(stored).toEqual([
      { id: 'u1', role: 'user', text: 'first question' },
      { id: 'a1', role: 'assistant', text: 'first answer' },
    ]);
    // The new thread is client-only until the user sends; threadId is reset
    // so the next sendChatMessage starts a fresh server-side thread.
    expect(after.agent.threadId).toBeNull();
  });
});

describe('saved presets', () => {
  test('initial presets slice is empty', () => {
    expect(useStore.getState().presets).toEqual([]);
  });

  test('addPreset / removePreset roundtrip with localStorage persistence', () => {
    const id = useStore
      .getState()
      .addPreset('run-eval', { dataset: 'evals/spam.jsonl', workers: 8 });
    const after = useStore.getState();
    expect(after.presets).toHaveLength(1);
    expect(after.presets[0].id).toBe(id);
    expect(after.presets[0].cliId).toBe('run-eval');
    expect(after.presets[0].label).toBe('spam.jsonl . 8w');
    // Persisted to localStorage.
    const stored = JSON.parse(localStorage.getItem('evalyn:presets') ?? '[]');
    expect(stored).toHaveLength(1);
    expect(stored[0].id).toBe(id);

    useStore.getState().removePreset(id);
    expect(useStore.getState().presets).toEqual([]);
    const cleared = JSON.parse(localStorage.getItem('evalyn:presets') ?? '[]');
    expect(cleared).toEqual([]);
  });

  test('addPreset dedupes on identical (cliId, args)', () => {
    const args = { dataset: 'evals/x.jsonl', workers: 4 };
    const id1 = useStore.getState().addPreset('run-eval', args);
    const id2 = useStore.getState().addPreset('run-eval', { ...args });
    expect(id1).toBe(id2);
    expect(useStore.getState().presets).toHaveLength(1);
  });

  test('addPreset key-order independence (sorted-key equality)', () => {
    const id1 = useStore
      .getState()
      .addPreset('run-eval', { dataset: 'a.jsonl', workers: 2 });
    const id2 = useStore
      .getState()
      .addPreset('run-eval', { workers: 2, dataset: 'a.jsonl' });
    expect(id1).toBe(id2);
    expect(useStore.getState().presets).toHaveLength(1);
  });

  test('addPreset labels fall back when no dataset/metrics', () => {
    const id = useStore.getState().addPreset('analyze', { mode: 'fast' });
    const p = useStore.getState().presets.find((x) => x.id === id);
    expect(p?.label).toBe('analyze preset');
  });

  test('addPreset honours user-supplied label override', () => {
    const id = useStore
      .getState()
      .addPreset('run-eval', { dataset: 'a.jsonl' }, 'My favourite');
    const p = useStore.getState().presets.find((x) => x.id === id);
    expect(p?.label).toBe('My favourite');
  });

  test('applyPreset seeds activeFormSeed and switches active CLI', () => {
    const id = useStore
      .getState()
      .addPreset('compare', { runA: 'r1', runB: 'r2' });
    useStore.getState().applyPreset(id);
    const after = useStore.getState();
    expect(after.activeCliId).toBe('compare');
    expect(after.activeFormSeed).toEqual({
      cliId: 'compare',
      args: { runA: 'r1', runB: 'r2' },
    });
  });

  test('applyPreset on missing id is a no-op', () => {
    useStore.getState().applyPreset('preset-does-not-exist');
    expect(useStore.getState().activeFormSeed).toBeNull();
  });

  test('findPresetByArgs returns id when matching, null otherwise', () => {
    const id = useStore
      .getState()
      .addPreset('run-eval', { dataset: 'b.jsonl', workers: 1 });
    expect(
      useStore.getState().findPresetByArgs('run-eval', {
        workers: 1,
        dataset: 'b.jsonl',
      }),
    ).toBe(id);
    expect(
      useStore.getState().findPresetByArgs('run-eval', { dataset: 'other' }),
    ).toBeNull();
    expect(useStore.getState().findPresetByArgs('other-cli', {})).toBeNull();
  });

  test('loadPresets re-reads from localStorage', () => {
    const seed = [
      {
        id: 'preset-seed',
        cliId: 'run-eval',
        label: 'seeded',
        args: { dataset: 'x' },
        createdAt: Date.now(),
      },
    ];
    localStorage.setItem('evalyn:presets', JSON.stringify(seed));
    useStore.getState().loadPresets();
    const after = useStore.getState();
    expect(after.presets).toHaveLength(1);
    expect(after.presets[0].id).toBe('preset-seed');
  });

  test('loadPresets ignores corrupted JSON without throwing', () => {
    localStorage.setItem('evalyn:presets', '{not json');
    expect(() => useStore.getState().loadPresets()).not.toThrow();
    expect(useStore.getState().presets).toEqual([]);
  });

  test('loadPresets drops malformed entries defensively', () => {
    localStorage.setItem(
      'evalyn:presets',
      JSON.stringify([
        { id: 'good', cliId: 'x', label: 'l', args: {}, createdAt: 1 },
        { id: 'no-args' }, // missing fields
        null,
        'string-not-object',
      ]),
    );
    useStore.getState().loadPresets();
    expect(useStore.getState().presets.map((p) => p.id)).toEqual(['good']);
  });

  test('preset args are deep-copied so mutations do not bleed in', () => {
    const args: Record<string, unknown> = { dataset: 'z.jsonl' };
    const id = useStore.getState().addPreset('run-eval', args);
    args.dataset = 'mutated';
    const stored = useStore.getState().presets.find((p) => p.id === id);
    expect(stored?.args.dataset).toBe('z.jsonl');
  });
});

