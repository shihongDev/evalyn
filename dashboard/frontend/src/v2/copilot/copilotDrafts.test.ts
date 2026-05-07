/**
 * Unit tests for copilotDrafts.ts: per-thread draft persistence
 * with LRU cap.
 *
 * Same shape as cliRunnerDrafts (which we don't test yet); both
 * came from the same per-context drafts pattern. Covers:
 *
 *  - load / save round-trip per threadId
 *  - the __new__ pre-thread state used before the first message
 *    creates a server-side thread id
 *  - clearCoPilotDraft removes the entry (not just sets it to "")
 *  - empty-string save also removes (cleaner storage footprint)
 *  - LRU eviction when over MAX_DRAFTS=20
 *  - LRU promotes on save (move-to-end)
 *  - storage corruption tolerance (poisoned JSON returns empty)
 */

import { beforeEach, describe, expect, it } from 'vitest';
import {
  clearCoPilotDraft,
  loadCoPilotDraft,
  saveCoPilotDraft,
} from './copilotDrafts';

const STORAGE_KEY = 'evalyn.dashboard.copilotDrafts.v1';

beforeEach(() => {
  window.localStorage.removeItem(STORAGE_KEY);
});

describe('copilotDrafts round-trip', () => {
  it('load returns empty string when no draft exists', () => {
    expect(loadCoPilotDraft('thread-a')).toBe('');
  });

  it('save then load returns the same text', () => {
    saveCoPilotDraft('thread-a', 'half-typed question');
    expect(loadCoPilotDraft('thread-a')).toBe('half-typed question');
  });

  it('keeps drafts isolated per threadId', () => {
    saveCoPilotDraft('thread-a', 'a-text');
    saveCoPilotDraft('thread-b', 'b-text');
    expect(loadCoPilotDraft('thread-a')).toBe('a-text');
    expect(loadCoPilotDraft('thread-b')).toBe('b-text');
  });
});

describe('the __new__ pre-thread key', () => {
  it('null threadId routes to a single shared key', () => {
    saveCoPilotDraft(null, 'pre-thread typing');
    expect(loadCoPilotDraft(null)).toBe('pre-thread typing');
    // Empty-string and undefined both fall into the same bucket.
    expect(loadCoPilotDraft(undefined)).toBe('pre-thread typing');
    expect(loadCoPilotDraft('')).toBe('pre-thread typing');
  });

  it('does NOT bleed pre-thread drafts into named threads', () => {
    saveCoPilotDraft(null, 'in pre-thread');
    expect(loadCoPilotDraft('thread-a')).toBe('');
  });
});

describe('clearCoPilotDraft / empty save', () => {
  it('clearCoPilotDraft removes the entry entirely', () => {
    saveCoPilotDraft('thread-a', 'x');
    clearCoPilotDraft('thread-a');
    expect(loadCoPilotDraft('thread-a')).toBe('');
    // Storage shouldn't grow with empty cells; a save("") of a
    // never-saved id should be a no-op (no entry created).
    saveCoPilotDraft('thread-b', '');
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as { drafts: Record<string, string> };
      expect('thread-b' in parsed.drafts).toBe(false);
    }
  });

  it('saving empty string removes a previously-saved draft', () => {
    saveCoPilotDraft('thread-a', 'something');
    saveCoPilotDraft('thread-a', '');
    expect(loadCoPilotDraft('thread-a')).toBe('');
  });
});

describe('LRU cap (MAX_DRAFTS=20)', () => {
  it('evicts the oldest draft when over the cap', () => {
    for (let i = 0; i < 22; i++) {
      saveCoPilotDraft(`thread-${i}`, `text-${i}`);
    }
    // Oldest two (thread-0, thread-1) evicted; newest 20 survive.
    expect(loadCoPilotDraft('thread-0')).toBe('');
    expect(loadCoPilotDraft('thread-1')).toBe('');
    expect(loadCoPilotDraft('thread-2')).toBe('text-2');
    expect(loadCoPilotDraft('thread-21')).toBe('text-21');
  });

  it('save promotes a re-saved threadId to most-recently-used', () => {
    // Fill the cap.
    for (let i = 0; i < 20; i++) {
      saveCoPilotDraft(`thread-${i}`, `text-${i}`);
    }
    // Re-save thread-0 (currently OLDEST). It should become NEWEST.
    saveCoPilotDraft('thread-0', 'updated');
    // Now save thread-NEW which would normally evict thread-0; but
    // since thread-0 was promoted, thread-1 (next-oldest) goes
    // instead.
    saveCoPilotDraft('thread-NEW', 'fresh');
    expect(loadCoPilotDraft('thread-0')).toBe('updated');
    expect(loadCoPilotDraft('thread-1')).toBe('');
    expect(loadCoPilotDraft('thread-NEW')).toBe('fresh');
  });
});

describe('corruption tolerance', () => {
  it('treats poisoned JSON as empty store', () => {
    window.localStorage.setItem(STORAGE_KEY, '{not valid json');
    expect(loadCoPilotDraft('thread-a')).toBe('');
    // Save should still succeed (overwrites the bad blob).
    saveCoPilotDraft('thread-a', 'recovered');
    expect(loadCoPilotDraft('thread-a')).toBe('recovered');
  });

  it('treats an array (wrong shape) as empty store', () => {
    window.localStorage.setItem(STORAGE_KEY, '[]');
    expect(loadCoPilotDraft('any')).toBe('');
  });
});
