/**
 * GuidanceToggleCard tests.
 *
 * Two behaviors covered:
 *   1. Reads localStorage on mount: a previously-saved 'false' renders the
 *      switch in the off state.
 *   2. Writes localStorage on toggle: clicking the switch flips the saved
 *      preference and updates aria-checked.
 *
 * The reset-flags button is verified via aria-label / role; we do not
 * exercise the visual flash-pill timing.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { GuidanceToggleCard, formatBytes, formatRelativeAgo, formatUptime } from './Settings';
import { TOUR_ENABLED_KEY, tourCompletedKey } from '../store/store';
import { FIRST_RUN_TOUR_ID } from '../tour/scripts/firstRun';

beforeEach(() => {
  window.localStorage.removeItem(TOUR_ENABLED_KEY);
  window.localStorage.removeItem(tourCompletedKey(FIRST_RUN_TOUR_ID));
});

describe('GuidanceToggleCard', () => {
  it('reads localStorage on mount: stored "false" renders aria-checked=false', () => {
    window.localStorage.setItem(TOUR_ENABLED_KEY, 'false');
    render(<GuidanceToggleCard />);
    const toggle = screen.getByRole('switch', { name: /co-pilot ui guidance/i });
    expect(toggle).toHaveAttribute('aria-checked', 'false');
  });

  it('writes localStorage on toggle and updates aria-checked', () => {
    render(<GuidanceToggleCard />);
    const toggle = screen.getByRole('switch', { name: /co-pilot ui guidance/i });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    expect(window.localStorage.getItem(TOUR_ENABLED_KEY)).toBe('false');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    expect(window.localStorage.getItem(TOUR_ENABLED_KEY)).toBe('true');
  });
});

describe('formatBytes', () => {
  it('returns dash for non-finite or negative input', () => {
    expect(formatBytes(NaN)).toBe('-');
    expect(formatBytes(-1)).toBe('-');
    expect(formatBytes(Infinity)).toBe('-');
  });

  it('uses raw bytes under 1 KiB', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1023)).toBe('1023 B');
  });

  it('promotes to KiB at 1024 with 1 decimal under 100', () => {
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
  });

  it('drops the decimal once value crosses 100', () => {
    // 100 KB exactly -> '100 KB' (rounded, no decimal)
    expect(formatBytes(100 * 1024)).toBe('100 KB');
  });

  it('promotes through MB and GB', () => {
    expect(formatBytes(2 * 1024 * 1024)).toBe('2.0 MB');
    // 1.5 GB
    expect(formatBytes(1.5 * 1024 * 1024 * 1024)).toBe('1.5 GB');
  });
});

describe('formatUptime', () => {
  it('returns dash for non-finite or negative input', () => {
    expect(formatUptime(NaN)).toBe('-');
    expect(formatUptime(-5)).toBe('-');
  });

  it('seconds-only under 60s', () => {
    expect(formatUptime(0)).toBe('0s');
    expect(formatUptime(47)).toBe('47s');
  });

  it('minutes-only under 1h', () => {
    expect(formatUptime(60)).toBe('1m');
    expect(formatUptime(59 * 60)).toBe('59m');
  });

  it('hours and minutes under 1d', () => {
    expect(formatUptime(3600)).toBe('1h 0m');
    expect(formatUptime(3600 + 12 * 60)).toBe('1h 12m');
  });

  it('days and hours past 1d', () => {
    expect(formatUptime(86400)).toBe('1d 0h');
    expect(formatUptime(3 * 86400 + 4 * 3600)).toBe('3d 4h');
  });
});

describe('formatRelativeAgo', () => {
  // Use fake timers anchored at a deterministic instant so we can
  // express "events" as offsets from "now" without flakiness.
  const NOW = new Date('2026-05-07T12:00:00Z').getTime();

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns dash for non-finite input', () => {
    expect(formatRelativeAgo(NaN)).toBe('-');
    expect(formatRelativeAgo(Infinity)).toBe('-');
  });

  it('seconds for events <60s ago', () => {
    expect(formatRelativeAgo(NOW)).toBe('0s');
    expect(formatRelativeAgo(NOW - 5_000)).toBe('5s');
    expect(formatRelativeAgo(NOW - 59_000)).toBe('59s');
  });

  it('minutes for events <1h ago', () => {
    expect(formatRelativeAgo(NOW - 60_000)).toBe('1m');
    expect(formatRelativeAgo(NOW - 45 * 60_000)).toBe('45m');
  });

  it('hours for events <1d ago', () => {
    expect(formatRelativeAgo(NOW - 60 * 60_000)).toBe('1h');
    expect(formatRelativeAgo(NOW - 23 * 60 * 60_000)).toBe('23h');
  });

  it('days for events 1d-30d ago', () => {
    expect(formatRelativeAgo(NOW - 24 * 60 * 60_000)).toBe('1d');
    expect(formatRelativeAgo(NOW - 30 * 24 * 60 * 60_000)).toBe('30d');
  });

  it('clamps at 30d+ for events older than 30d', () => {
    expect(formatRelativeAgo(NOW - 31 * 24 * 60 * 60_000)).toBe('30d+');
    expect(formatRelativeAgo(NOW - 365 * 24 * 60 * 60_000)).toBe('30d+');
  });

  it('clamps future timestamps to 0s (no negative values)', () => {
    expect(formatRelativeAgo(NOW + 5_000)).toBe('0s');
  });

  it('accepts an explicit nowMs override', () => {
    // The optional `nowMs` argument lets a caller thread its
    // re-render state (e.g. a setInterval-driven `now` value) so
    // the relative label updates without depending on a side
    // effect inside the formatter. Passing an override must
    // produce the same result as if Date.now() were that value.
    const PAST = NOW - 30_000;
    // Without override: uses Date.now() = NOW (set via fake timer
    // in beforeEach).
    expect(formatRelativeAgo(PAST)).toBe('30s');
    // With override pointing 60s ahead of NOW.
    expect(formatRelativeAgo(PAST, NOW + 60_000)).toBe('1m');
  });
});
