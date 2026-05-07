/**
 * Unit tests for cliRunnerDrafts.ts: per-cli form draft persistence
 * with LRU cap.
 *
 * Sibling of copilotDrafts but with deliberately different
 * semantics:
 *
 *  - draft is Record<string, unknown> (form values), not a string
 *  - save("empty form") IS persisted - "I cleared all fields" is a
 *    valid intent that should survive a refresh
 *  - empty cliId is a no-op (no synthetic shared bucket like the
 *    copilot's "__new__")
 *  - loadDraft returns null on miss (not "")
 *
 * These differences are why the test files don't share a fixture.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { clearDraft, loadDraft, saveDraft } from './cliRunnerDrafts';

const STORAGE_KEY = 'evalyn.dashboard.cliRunnerDrafts.v1';

beforeEach(() => {
  window.localStorage.removeItem(STORAGE_KEY);
});

describe('cliRunnerDrafts round-trip', () => {
  it('loadDraft returns null when no draft exists', () => {
    expect(loadDraft('run-eval')).toBeNull();
  });

  it('save then load returns the same values', () => {
    saveDraft('run-eval', { dataset: 'frosty', limit: 50 });
    expect(loadDraft('run-eval')).toEqual({
      dataset: 'frosty',
      limit: 50,
    });
  });

  it('keeps drafts isolated per cliId', () => {
    saveDraft('run-eval', { dataset: 'a' });
    saveDraft('compare', { runA: 'r1', runB: 'r2' });
    expect(loadDraft('run-eval')).toEqual({ dataset: 'a' });
    expect(loadDraft('compare')).toEqual({ runA: 'r1', runB: 'r2' });
  });
});

describe('empty cliId is a no-op', () => {
  it('saveDraft("", ...) does nothing', () => {
    saveDraft('', { x: 1 });
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('loadDraft("") returns null even if other drafts exist', () => {
    saveDraft('run-eval', { x: 1 });
    expect(loadDraft('')).toBeNull();
  });

  it('clearDraft("") does nothing', () => {
    saveDraft('run-eval', { x: 1 });
    clearDraft('');
    expect(loadDraft('run-eval')).toEqual({ x: 1 });
  });
});

describe('"empty form" is a valid draft', () => {
  it('saving an empty values object preserves the entry', () => {
    // Customer scenario: user opens run-eval, types "frosty",
    // realises wrong dataset, deletes everything. The empty
    // form IS the current state and SHOULD survive a refresh.
    saveDraft('run-eval', {});
    expect(loadDraft('run-eval')).toEqual({});
    // Distinct from "no draft": loadDraft returns the object,
    // not null.
    expect(loadDraft('run-eval')).not.toBeNull();
  });
});

describe('clearDraft', () => {
  it('removes the entry entirely', () => {
    saveDraft('run-eval', { x: 1 });
    clearDraft('run-eval');
    expect(loadDraft('run-eval')).toBeNull();
  });

  it('is a no-op for unknown cliId', () => {
    saveDraft('run-eval', { x: 1 });
    clearDraft('does-not-exist');
    // The unrelated draft is untouched.
    expect(loadDraft('run-eval')).toEqual({ x: 1 });
  });
});

describe('LRU cap (MAX_DRAFTS=30)', () => {
  it('evicts the oldest draft when over the cap', () => {
    for (let i = 0; i < 32; i++) {
      saveDraft(`cli-${i}`, { i });
    }
    // Oldest two evicted, newest 30 survive.
    expect(loadDraft('cli-0')).toBeNull();
    expect(loadDraft('cli-1')).toBeNull();
    expect(loadDraft('cli-2')).toEqual({ i: 2 });
    expect(loadDraft('cli-31')).toEqual({ i: 31 });
  });

  it('save promotes a re-saved cliId to most-recently-used', () => {
    for (let i = 0; i < 30; i++) {
      saveDraft(`cli-${i}`, { i });
    }
    // Touch cli-0 (currently OLDEST) so it becomes NEWEST.
    saveDraft('cli-0', { updated: true });
    // Add a fresh one - cli-1 (next-oldest) gets evicted, not cli-0.
    saveDraft('cli-NEW', { fresh: true });
    expect(loadDraft('cli-0')).toEqual({ updated: true });
    expect(loadDraft('cli-1')).toBeNull();
    expect(loadDraft('cli-NEW')).toEqual({ fresh: true });
  });
});

describe('corruption tolerance', () => {
  it('treats poisoned JSON as empty store', () => {
    window.localStorage.setItem(STORAGE_KEY, '{not valid json');
    expect(loadDraft('run-eval')).toBeNull();
    // Save still works (overwrites the bad blob).
    saveDraft('run-eval', { x: 1 });
    expect(loadDraft('run-eval')).toEqual({ x: 1 });
  });

  it('treats wrong-shape store (array) as empty', () => {
    window.localStorage.setItem(STORAGE_KEY, '[]');
    expect(loadDraft('run-eval')).toBeNull();
  });
});
