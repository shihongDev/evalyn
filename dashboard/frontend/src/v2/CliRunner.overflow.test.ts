import { describe, expect, it } from 'vitest';
import { mergeLinesWithOverflow } from './CliRunner';
import type { JobLine } from './api/jobs';

/** Build a synthetic stdout line. Tests only care about kind + text;
 * the timestamp is fixed to keep snapshots deterministic. */
function L(text: string, kind: JobLine['kind'] = 'stdout'): JobLine {
  return { kind, text, ts: 1000 };
}

describe('mergeLinesWithOverflow', () => {
  it('passes lines through untouched when under the cap', () => {
    const prev = [L('a'), L('b')];
    const buf = [L('c')];
    const r = mergeLinesWithOverflow(prev, buf, 0, 10);
    expect(r.lines.map((l) => l.text)).toEqual(['a', 'b', 'c']);
    expect(r.dropped).toBe(0);
  });

  it('slices oldest lines when over the cap and accumulates dropped count', () => {
    const prev = Array.from({ length: 5 }, (_, i) => L(`p${i}`));
    const buf = Array.from({ length: 4 }, (_, i) => L(`b${i}`));
    // 5 + 4 = 9 > cap=6 -> drop 3 oldest, keep last 6.
    const r = mergeLinesWithOverflow(prev, buf, 0, 6);
    expect(r.dropped).toBe(3);
    // First entry is the marker, then the kept 5 lines (one slot
    // taken by the marker).
    expect(r.lines[0].kind).toBe('system');
    expect(r.lines[0].text).toContain('3 earlier lines hidden');
    expect(r.lines.slice(1).map((l) => l.text)).toEqual([
      'p3',
      'p4',
      'b0',
      'b1',
      'b2',
      'b3',
    ]);
  });

  it('strips its own previous marker so it does not double-count it', () => {
    // Simulate the state after a prior overflow flush: prev has a
    // marker at index 0 + 5 real lines.
    const prev: JobLine[] = [
      { kind: 'system', text: '[client buffer rolled over: 3 earlier lines hidden]', ts: 1 },
      L('a'),
      L('b'),
      L('c'),
      L('d'),
      L('e'),
    ];
    const buf = [L('f'), L('g')];
    // Real lines (5) + buf (2) = 7 > cap=6 -> 1 more line dropped.
    const r = mergeLinesWithOverflow(prev, buf, 3, 6);
    // Cumulative count: 3 (prior) + 1 (now) = 4.
    expect(r.dropped).toBe(4);
    expect(r.lines[0].text).toContain('4 earlier lines hidden');
    // The marker itself was not counted as one of the "real lines"
    // dropped - if it had been, dropped would be 5 not 4.
    expect(r.lines.slice(1).map((l) => l.text)).toEqual([
      'b',
      'c',
      'd',
      'e',
      'f',
      'g',
    ]);
  });

  it('does NOT strip server-dropped markers (they are real diagnostic lines)', () => {
    const prev: JobLine[] = [
      { kind: 'system', text: '[server dropped 42 earlier lines]', ts: 1 },
      L('a'),
    ];
    const buf = [L('b')];
    const r = mergeLinesWithOverflow(prev, buf, 0, 10);
    expect(r.dropped).toBe(0);
    // Server marker stayed at index 0 - it is treated like any real line.
    expect(r.lines.map((l) => l.text)).toEqual([
      '[server dropped 42 earlier lines]',
      'a',
      'b',
    ]);
  });

  it('formats the marker count with locale separators', () => {
    const prev: JobLine[] = [];
    const buf = Array.from({ length: 1500 }, (_, i) => L(`x${i}`));
    const r = mergeLinesWithOverflow(prev, buf, 4523, 1000);
    // 4523 prior + 500 new overflow = 5023.
    expect(r.dropped).toBe(5023);
    expect(r.lines[0].text).toMatch(/5,023 earlier lines hidden/);
  });

  it('returns prevDropped unchanged when no overflow occurs but marker is preserved', () => {
    // User had a prior overflow and now small-buffer flushes arrive
    // - the marker should remain and the count is unchanged.
    const prev: JobLine[] = [
      { kind: 'system', text: '[client buffer rolled over: 7 earlier lines hidden]', ts: 1 },
      L('a'),
    ];
    const buf = [L('b')];
    const r = mergeLinesWithOverflow(prev, buf, 7, 10);
    expect(r.dropped).toBe(7);
    expect(r.lines[0].kind).toBe('system');
    expect(r.lines[0].text).toContain('7 earlier lines hidden');
    expect(r.lines.slice(1).map((l) => l.text)).toEqual(['a', 'b']);
  });
});
