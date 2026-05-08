import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act, cleanup } from '@testing-library/react';
import { useLiveDuration } from './useLiveDuration';

describe('useLiveDuration', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-05-07T12:00:00Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
    cleanup();
  });

  it('returns null when not live', () => {
    const startedAt = new Date('2026-05-07T11:59:00Z').toISOString();
    const { result } = renderHook(() => useLiveDuration(startedAt, false));
    expect(result.current).toBeNull();
  });

  it('formats elapsed seconds for jobs running under a minute', () => {
    const startedAt = new Date('2026-05-07T11:59:48Z').toISOString();
    const { result } = renderHook(() => useLiveDuration(startedAt, true));
    expect(result.current).toBe('12s');
  });

  it('updates the elapsed value as time advances', () => {
    const startedAt = new Date('2026-05-07T11:59:50Z').toISOString();
    const { result } = renderHook(() => useLiveDuration(startedAt, true));
    expect(result.current).toBe('10s');

    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(result.current).toBe('15s');
  });

  it('snaps the counter to the current time on visibilitychange (tab return)', () => {
    const startedAt = new Date('2026-05-07T11:59:00Z').toISOString();
    const { result } = renderHook(() => useLiveDuration(startedAt, true));
    expect(result.current).toBe('1m00s');

    // Simulate the tab being backgrounded for 30s. The setInterval
    // would normally update during this window but a backgrounded
    // tab may throttle it. Advance the system clock without
    // triggering the interval (hidden tab condition).
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value: 'hidden',
      writable: true,
    });
    vi.setSystemTime(new Date('2026-05-07T12:00:30Z'));

    // Now simulate the user returning to the tab.
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value: 'visible',
      writable: true,
    });
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    // Should have snapped to the current elapsed (1m30s) without
    // waiting for the next 1s interval tick.
    expect(result.current).toBe('1m30s');
  });

  it('returns null for an unparseable timestamp', () => {
    const { result } = renderHook(() => useLiveDuration('not-a-date', true));
    expect(result.current).toBeNull();
  });
});
