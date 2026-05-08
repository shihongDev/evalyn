/**
 * useVisibilityPoll unit tests.
 *
 * Pin the contract that two surfaces (SystemStatusCard now,
 * RecentJobsDrawer capacity chip in a follow-up) rely on:
 *   - immediate poll on mount when visible
 *   - interval-driven polls thereafter
 *   - pause when document.visibilityState flips to 'hidden'
 *   - immediate refetch + interval restart on tab-return
 *   - manual refetch() bypasses the timer
 *   - cleanup stops timer + listener; in-flight fetches don't
 *     setState on unmounted components
 *   - enabled=false disables the hook entirely
 *
 * Visibility events are simulated by mutating
 * document.visibilityState (writable in jsdom) and dispatching
 * a 'visibilitychange' event on the document.
 */

import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useVisibilityPoll } from './useVisibilityPoll';

function setVisibility(state: 'visible' | 'hidden') {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => state,
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

describe('useVisibilityPoll', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setVisibility('visible');
  });

  afterEach(() => {
    vi.useRealTimers();
    setVisibility('visible');
  });

  it('fetches immediately on mount', async () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    const { result } = renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 1000 }),
    );

    expect(fetcher).toHaveBeenCalledTimes(1);
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.value).toEqual({ n: 1 });
    expect(result.current.loaded).toBe(true);
  });

  it('polls on the configured interval', async () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    renderHook(() => useVisibilityPoll({ fetcher, intervalMs: 1000 }));

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    // Advance one tick at a time with a microtask flush between
    // each. The hook's in-flight dedup correctly coalesces back-
    // to-back interval ticks fired by a single advanceTimersByTime
    // when the prior poll's promise hasn't yet settled - so the
    // test must yield to the microtask queue between ticks for
    // each fetcher call to land. (Pre-dedup the test could lump
    // ticks together in one advance call; that was relying on
    // undefined fake-timer + microtask interleaving.)
    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(2);

    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(3);

    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(4);
  });

  it('stops polling when tab becomes hidden', async () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    renderHook(() => useVisibilityPoll({ fetcher, intervalMs: 1000 }));

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    act(() => setVisibility('hidden'));
    // Advance past several intervals - no more polls.
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('refetches immediately when tab becomes visible again', async () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    renderHook(() => useVisibilityPoll({ fetcher, intervalMs: 1000 }));

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    act(() => setVisibility('hidden'));
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    act(() => setVisibility('visible'));
    await act(async () => {
      await Promise.resolve();
    });
    // Immediate fetch on tab-return + interval restarts.
    expect(fetcher).toHaveBeenCalledTimes(2);

    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it('refetch() triggers an immediate fetch outside the interval', async () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    const { result } = renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 10000 }),
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    act(() => result.current.refetch());
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('does not fetch or subscribe when enabled=false', () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 1000, enabled: false }),
    );

    expect(fetcher).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('starts polling when enabled flips from false to true', async () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    let enabled = false;
    const { rerender } = renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 1000, enabled }),
    );
    expect(fetcher).not.toHaveBeenCalled();

    enabled = true;
    rerender();
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('cleanup stops the interval after unmount', async () => {
    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    const { unmount } = renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 1000 }),
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    unmount();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('lastFetchAt advances on every poll resolution', async () => {
    // Anchor wall-clock at a known instant. advanceTimersByTime
    // moves the fake clock too, so the test computes expected
    // values relative to NOW + accumulated advance.
    const NOW = new Date('2026-05-07T12:00:00Z').getTime();
    vi.setSystemTime(new Date(NOW));

    const fetcher = vi.fn().mockResolvedValue({ n: 1 });
    const { result } = renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 1000 }),
    );

    expect(result.current.lastFetchAt).toBeNull();

    await act(async () => {
      await Promise.resolve();
    });
    // First fetch resolved at NOW.
    expect(result.current.lastFetchAt).toBe(NOW);

    // advanceTimersByTime moves the fake clock by 1000ms AND fires
    // the next interval tick, so the second poll resolves at NOW+1000.
    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(result.current.lastFetchAt).toBe(NOW + 1000);
  });

  it('lastFetchAt updates even when value is identical (freshness, not change)', async () => {
    // Critical: a freshness indicator must reset on every fetch,
    // not just identity-changing ones. Without this, a polling
    // surface returning the same JSON twice in a row would freeze
    // its "Refreshed Xs ago" label and look stalled.
    const NOW = new Date('2026-05-07T12:00:00Z').getTime();
    vi.setSystemTime(new Date(NOW));

    // Same object reference returned every call.
    const FROZEN = { n: 1 };
    const fetcher = vi.fn().mockResolvedValue(FROZEN);
    const { result } = renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 1000 }),
    );
    await act(async () => {
      await Promise.resolve();
    });
    const firstStamp = result.current.lastFetchAt;
    expect(firstStamp).toBe(NOW);

    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    // Same value reference but lastFetchAt MUST have advanced.
    expect(result.current.value).toBe(FROZEN);
    expect(result.current.lastFetchAt).toBe(NOW + 1000);
  });

  it('uses the latest fetcher reference across re-renders', async () => {
    const a = vi.fn().mockResolvedValue({ src: 'a' });
    const b = vi.fn().mockResolvedValue({ src: 'b' });
    let active = a;
    const { rerender, result } = renderHook(() =>
      useVisibilityPoll({ fetcher: active, intervalMs: 1000 }),
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(a).toHaveBeenCalledTimes(1);
    expect(result.current.value).toEqual({ src: 'a' });

    active = b;
    rerender();

    // The next interval tick must call the NEW fetcher because
    // we hold the latest reference in a ref. Without this, the
    // hook would freeze on the original fetcher closure -
    // surprising behavior for callers passing an inline arrow.
    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('refetch dedupes concurrent calls (rapid Refresh-button mash)', async () => {
    // Simulate a slow fetcher: 1s before resolve. While in flight,
    // the user hammers refetch() three times. Only ONE fetch should
    // fire. Without the dedup the server would see 3 concurrent
    // GETs for no benefit.
    let resolveFetcher: ((v: { n: number }) => void) | null = null;
    const fetcher = vi.fn().mockImplementation(() => {
      return new Promise<{ n: number }>((resolve) => {
        resolveFetcher = resolve;
      });
    });
    const { result } = renderHook(() =>
      useVisibilityPoll({ fetcher, intervalMs: 60_000 }),
    );
    // Initial mount fires the first poll. fetcher() called once.
    expect(fetcher).toHaveBeenCalledTimes(1);
    // While in flight, mash refetch three times.
    act(() => {
      result.current.refetch();
      result.current.refetch();
      result.current.refetch();
    });
    // No new fetches - all three reuse the in-flight promise.
    expect(fetcher).toHaveBeenCalledTimes(1);
    // Once the in-flight resolves, a future refetch fires fresh.
    await act(async () => {
      resolveFetcher!({ n: 1 });
      await Promise.resolve();
    });
    act(() => {
      result.current.refetch();
    });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
