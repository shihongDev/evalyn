/**
 * Unit tests for jobsHistory.ts pin / cap / sort semantics.
 *
 * Pure-function tests against a real localStorage (jsdom) - no React
 * involved. Each test resets the storage and history state so order
 * does not matter.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import {
  activeJobCliId,
  clearJobsHistory,
  loadJobsHistory,
  saveJobsHistory,
  setJobPinned,
  upsertJob,
  type JobHistoryEntry,
} from './jobsHistory';

const STORAGE_KEY = 'evalyn:v2:jobsHistory';

function makeEntry(
  job_id: string,
  startedAtIso: string,
  overrides: Partial<JobHistoryEntry> = {},
): JobHistoryEntry {
  return {
    job_id,
    cli_id: 'list-runs',
    cli_args: {},
    started_at_iso: startedAtIso,
    status: 'complete',
    ...overrides,
  };
}

beforeEach(() => {
  window.localStorage.removeItem(STORAGE_KEY);
});

describe('saveJobsHistory cap', () => {
  it('drops the OLDEST unpinned entry when over the 30-cap', () => {
    // Fill the cap with 35 entries, all unpinned, ages 0..34.
    const entries: JobHistoryEntry[] = [];
    for (let i = 0; i < 35; i++) {
      entries.push(makeEntry(`j${i}`, `2026-05-01T00:00:${i.toString().padStart(2, '0')}.000Z`));
    }
    saveJobsHistory(entries);
    const loaded = loadJobsHistory();
    // 30 newest survive.
    expect(loaded).toHaveLength(30);
    // Oldest (j0) was evicted; newest (j34) survived.
    expect(loaded.map((e) => e.job_id)).toContain('j34');
    expect(loaded.map((e) => e.job_id)).not.toContain('j0');
  });

  it('preserves pinned entries beyond the 30-cap', () => {
    const entries: JobHistoryEntry[] = [];
    // 5 pinned ancient entries
    for (let i = 0; i < 5; i++) {
      entries.push(
        makeEntry(`pin${i}`, `2024-01-01T00:00:0${i}.000Z`, { pinned: true }),
      );
    }
    // 30 unpinned recent entries
    for (let i = 0; i < 30; i++) {
      entries.push(makeEntry(`j${i}`, `2026-05-01T00:00:${i.toString().padStart(2, '0')}.000Z`));
    }
    saveJobsHistory(entries);
    const loaded = loadJobsHistory();
    // Total: pinned (5) + at most (30 - 5 = 25) unpinned = 30.
    expect(loaded).toHaveLength(30);
    // All pins survive.
    for (let i = 0; i < 5; i++) {
      expect(loaded.find((e) => e.job_id === `pin${i}`)).toBeTruthy();
    }
    // Newest unpinned (j29) survives; oldest unpinned (j0..j4) evicted
    // because the cap budget was 25 unpinned.
    expect(loaded.find((e) => e.job_id === 'j29')).toBeTruthy();
    expect(loaded.find((e) => e.job_id === 'j0')).toBeFalsy();
  });

  it('keeps all pinned even when pins exceed the cap', () => {
    // Degenerate case: 35 pinned entries. We respect every pin even
    // though that grows the total beyond MAX_ENTRIES.
    const entries: JobHistoryEntry[] = [];
    for (let i = 0; i < 35; i++) {
      entries.push(
        makeEntry(`pin${i}`, `2024-01-01T00:${i.toString().padStart(2, '0')}:00.000Z`, {
          pinned: true,
        }),
      );
    }
    saveJobsHistory(entries);
    const loaded = loadJobsHistory();
    expect(loaded).toHaveLength(35);
    expect(loaded.every((e) => e.pinned)).toBe(true);
  });
});

describe('sortNewestFirst (via loadJobsHistory)', () => {
  it('puts pinned entries above unpinned regardless of age', () => {
    const ancientPin = makeEntry('ancient', '2024-01-01T00:00:00Z', {
      pinned: true,
    });
    const recentUnpinned = makeEntry('recent', '2026-05-07T12:00:00Z');
    saveJobsHistory([recentUnpinned, ancientPin]);
    const loaded = loadJobsHistory();
    expect(loaded[0].job_id).toBe('ancient');
    expect(loaded[1].job_id).toBe('recent');
  });

  it('within each group, sorts newest-first by started_at_iso', () => {
    const entries = [
      makeEntry('a', '2026-05-01T10:00:00Z'),
      makeEntry('b', '2026-05-01T11:00:00Z'),
      makeEntry('c', '2026-05-01T09:00:00Z'),
    ];
    saveJobsHistory(entries);
    const loaded = loadJobsHistory();
    expect(loaded.map((e) => e.job_id)).toEqual(['b', 'a', 'c']);
  });
});

describe('setJobPinned', () => {
  it('toggles the pinned flag on an existing entry', () => {
    upsertJob(makeEntry('j1', '2026-05-07T12:00:00Z'));
    setJobPinned('j1', true);
    let loaded = loadJobsHistory();
    expect(loaded[0].pinned).toBe(true);
    setJobPinned('j1', false);
    loaded = loadJobsHistory();
    expect(loaded[0].pinned).toBeFalsy();
  });

  it('is a no-op for unknown job_ids (does not synthesize a stub)', () => {
    setJobPinned('does-not-exist', true);
    expect(loadJobsHistory()).toEqual([]);
  });
});

describe('clearJobsHistory', () => {
  it('drops unpinned entries but preserves pinned ones', () => {
    upsertJob(makeEntry('keep', '2026-05-07T12:00:00Z', { pinned: true }));
    upsertJob(makeEntry('drop', '2026-05-07T12:00:00Z'));
    clearJobsHistory();
    const loaded = loadJobsHistory();
    expect(loaded).toHaveLength(1);
    expect(loaded[0].job_id).toBe('keep');
  });

  it('fully clears storage when no pinned entries exist', () => {
    upsertJob(makeEntry('a', '2026-05-07T12:00:00Z'));
    clearJobsHistory();
    expect(loadJobsHistory()).toEqual([]);
  });
});

describe('activeJobCliId', () => {
  it('returns the cli_id when EXACTLY one job is queued/running', () => {
    upsertJob(makeEntry('a', '2026-05-07T12:00:00Z', {
      cli_id: 'run-eval',
      status: 'running',
    }));
    upsertJob(makeEntry('b', '2026-05-07T11:00:00Z', { status: 'complete' }));
    expect(activeJobCliId()).toBe('run-eval');
  });

  it('returns null when zero or 2+ are active', () => {
    expect(activeJobCliId([])).toBeNull();

    upsertJob(makeEntry('a', '2026-05-07T12:00:00Z', { status: 'running' }));
    upsertJob(makeEntry('b', '2026-05-07T11:00:00Z', { status: 'running' }));
    expect(activeJobCliId()).toBeNull();
  });
});
