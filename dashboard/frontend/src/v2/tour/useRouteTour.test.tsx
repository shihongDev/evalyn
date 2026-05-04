/**
 * useRouteTour hook tests.
 *
 * Each tabbed route uses this hook to fire its first-visit guidance:
 *   1. Does NOT fire while `ready` is false (data still loading or
 *      errored). Important so we don't anchor on a skeleton render.
 *   2. Fires exactly once after `ready` flips true and the 500ms
 *      post-mount delay elapses. Sets store.tourActiveId to the
 *      requested id; collapses the co-pilot dock.
 *   3. Does NOT fire if the per-tour completion flag is already set.
 *      Each tab's flag is independent of the others.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useRouteTour } from './useRouteTour';
import { useV2Store, tourCompletedKey } from '../store/store';

const TID = 'datasetUpload.v1';

beforeEach(() => {
  vi.useFakeTimers();
  useV2Store.setState({ tourActiveId: null, tourStep: 0, dockOpen: true });
  window.localStorage.removeItem(tourCompletedKey(TID));
});

afterEach(() => {
  vi.useRealTimers();
  window.localStorage.removeItem(tourCompletedKey(TID));
});

describe('useRouteTour', () => {
  it('does not fire while ready is false', () => {
    renderHook(({ ready }) => useRouteTour(TID, ready), {
      initialProps: { ready: false },
    });
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(useV2Store.getState().tourActiveId).toBeNull();
  });

  it('fires once 500ms after ready flips true and collapses the dock', () => {
    const { rerender } = renderHook(({ ready }) => useRouteTour(TID, ready), {
      initialProps: { ready: false },
    });
    expect(useV2Store.getState().tourActiveId).toBeNull();

    rerender({ ready: true });
    expect(useV2Store.getState().dockOpen).toBe(false);
    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(useV2Store.getState().tourActiveId).toBe(TID);
    expect(useV2Store.getState().tourStep).toBe(0);
  });

  it('does not re-fire once the user has completed this tour', () => {
    window.localStorage.setItem(tourCompletedKey(TID), '1');
    renderHook(({ ready }) => useRouteTour(TID, ready), {
      initialProps: { ready: true },
    });
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(useV2Store.getState().tourActiveId).toBeNull();
  });
});
