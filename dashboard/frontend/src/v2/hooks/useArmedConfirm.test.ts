/**
 * useArmedConfirm unit tests.
 *
 * The hook gates destructive actions across multiple surfaces:
 *   - RecentJobsDrawer Clear history
 *   - Datasets bulk-evaluate
 *   - AnnotateSession reset-bookmarks / clear-local-draft
 *   - Settings "Prune old" (added this loop)
 *
 * Pin the contract:
 *   - arm() flips armed=true
 *   - auto-resets to false after delayMs without a second click
 *   - reset() drops the timer + flips back to false
 *   - re-arming clears any pending auto-reset (no flicker between
 *     the second arm() and a delayed first auto-reset firing)
 *   - unmount cancels the timer (no setState on unmounted)
 */

import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useArmedConfirm } from './useArmedConfirm';

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useArmedConfirm', () => {
  it('starts un-armed', () => {
    const { result } = renderHook(() => useArmedConfirm());
    expect(result.current.armed).toBe(false);
  });

  it('arm() flips armed to true', () => {
    const { result } = renderHook(() => useArmedConfirm(1000));
    act(() => result.current.arm());
    expect(result.current.armed).toBe(true);
  });

  it('auto-resets to false after delayMs', () => {
    const { result } = renderHook(() => useArmedConfirm(1000));
    act(() => result.current.arm());
    expect(result.current.armed).toBe(true);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current.armed).toBe(false);
  });

  it('reset() flips armed false and cancels the timer', () => {
    const { result } = renderHook(() => useArmedConfirm(1000));
    act(() => result.current.arm());
    expect(result.current.armed).toBe(true);

    act(() => result.current.reset());
    expect(result.current.armed).toBe(false);

    // Advance well past the original delay - no re-arm should
    // fire because reset() cancelled the timer.
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(result.current.armed).toBe(false);
  });

  it('re-arming cancels any pending auto-reset', () => {
    // Regression-prone: if arm() didn't clear the previous timer,
    // the FIRST auto-reset would fire mid-second-arm window and
    // cause a visual flicker between armed/un-armed states.
    const { result } = renderHook(() => useArmedConfirm(1000));
    act(() => result.current.arm());
    // Advance partially - the original auto-reset is now scheduled
    // but not fired.
    act(() => {
      vi.advanceTimersByTime(800);
    });
    expect(result.current.armed).toBe(true);

    act(() => result.current.arm());
    expect(result.current.armed).toBe(true);

    // 800ms more would have fired the FIRST timer if it weren't
    // cancelled. armed should still be true.
    act(() => {
      vi.advanceTimersByTime(800);
    });
    expect(result.current.armed).toBe(true);

    // The SECOND arm's 1000ms window starts at the second arm
    // call. After total of 1000ms past second arm, expect reset.
    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(result.current.armed).toBe(false);
  });

  it('unmount cancels the timer (no console errors after delay)', () => {
    const { result, unmount } = renderHook(() => useArmedConfirm(1000));
    act(() => result.current.arm());
    expect(result.current.armed).toBe(true);

    unmount();

    // Advancing past the delay must not throw, log, or attempt
    // a setState on the unmounted component. We can't directly
    // inspect setState, but we CAN confirm no error throws and
    // the test runner doesn't complain about unhandled work.
    expect(() => {
      act(() => {
        vi.advanceTimersByTime(2000);
      });
    }).not.toThrow();
  });

  it('respects a custom delayMs', () => {
    const { result } = renderHook(() => useArmedConfirm(250));
    act(() => result.current.arm());
    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(result.current.armed).toBe(true);
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(result.current.armed).toBe(false);
  });
});
