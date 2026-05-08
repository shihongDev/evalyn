/**
 * Focused tests for useV2Resource's error-formatting path.
 * The hook has many concerns (caching, prefetch, WS invalidation);
 * we only cover the bug we just fixed: errors must flow through
 * the shared errorMessage() helper, not String(e).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act, cleanup, waitFor } from '@testing-library/react';
import { useV2Resource, _resetV2Cache } from './useV2Resource';

describe('useV2Resource - error formatting', () => {
  beforeEach(() => {
    _resetV2Cache();
  });

  afterEach(() => {
    cleanup();
    _resetV2Cache();
  });

  it('rewrites a fetch-level TypeError to the friendly network message', async () => {
    const fetcher = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'));
    const { result } = renderHook(() => useV2Resource('k1', fetcher));
    await waitFor(() => {
      expect(result.current.err).toBe(
        'Network unreachable - check your connection and try again.',
      );
    });
  });

  it('extracts the message from a plain Error (no Error: prefix)', async () => {
    const fetcher = vi.fn().mockRejectedValueOnce(new Error('500 Internal Server Error'));
    const { result } = renderHook(() => useV2Resource('k2', fetcher));
    await waitFor(() => {
      // Pre-fix this would have been "Error: 500 Internal Server Error"
      // (String(e) on an Error). Now it's just the message.
      expect(result.current.err).toBe('500 Internal Server Error');
    });
  });

  it('falls back to String() for non-Error throws', async () => {
    const fetcher = vi.fn().mockRejectedValueOnce('plain string thrown');
    const { result } = renderHook(() => useV2Resource('k3', fetcher));
    await waitFor(() => {
      expect(result.current.err).toBe('plain string thrown');
    });
  });

  it('clears the error on a subsequent successful refetch', async () => {
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new Error('first try fails'))
      .mockResolvedValueOnce({ ok: true });
    const { result } = renderHook(() => useV2Resource('k4', fetcher));
    await waitFor(() => {
      expect(result.current.err).toBe('first try fails');
    });
    await act(async () => {
      await result.current.refetch();
    });
    expect(result.current.err).toBeNull();
    expect(result.current.data).toEqual({ ok: true });
  });

  it('cache_invalidate keeps isInitialLoad=false during the refresh', async () => {
    // Pin: a refetch (the path cache_invalidate uses) must NOT
    // regress isInitialLoad to true mid-session. Otherwise the
    // UpdatingChip (visible={reloading && !isInitialLoad}) hides
    // during exactly the background refresh it's meant to indicate.
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce({ v: 1 })
      .mockResolvedValueOnce({ v: 2 });
    const { result } = renderHook(() => useV2Resource('k5', fetcher));
    await waitFor(() => {
      expect(result.current.data).toEqual({ v: 1 });
    });
    expect(result.current.isInitialLoad).toBe(false);

    await act(async () => {
      await result.current.refetch();
    });
    // Pre-fix: deleting the cache entry up-front made isInitialLoad
    // flip to true mid-refresh. After the fix the entry stays in
    // place so isInitialLoad stays false throughout.
    expect(result.current.isInitialLoad).toBe(false);
    expect(result.current.data).toEqual({ v: 2 });
  });
});
