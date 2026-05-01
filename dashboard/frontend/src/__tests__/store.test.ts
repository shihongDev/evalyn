/**
 * Vitest unit tests for the Zustand store.
 *
 * Covers initial state, tab actions, tweak patching, and job upserts.
 */

import { beforeEach, describe, expect, test } from 'vitest';
import { useStore, __resetStore, TWEAK_DEFAULTS } from '../store';
import type { Job, Tab } from '../types/jobs';

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
    expect(s.sidebarView).toBe('files');
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
