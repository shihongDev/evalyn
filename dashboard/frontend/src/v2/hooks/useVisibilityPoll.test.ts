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

    await act(async () => {
      vi.advanceTimersByTime(1000);
      await Promise.resolve();
    });
    expect(fetcher).toHaveBeenCalledTimes(2);

    await act(async () => {
      vi.advanceTimersByTime(2500);
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
});
