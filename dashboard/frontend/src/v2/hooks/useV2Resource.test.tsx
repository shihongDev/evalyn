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
});
