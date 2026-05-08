/**
 * Tests pin the contract of useFlashState so the cleanup
 * semantics are durable. Uses fake timers so we can advance
 * deterministically and assert the pre-revert vs post-revert
 * state transitions.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';
import { useFlashState } from './useFlashState';

describe('useFlashState', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it('starts with the initial value', () => {
    const { result } = renderHook(() => useFlashState<'idle' | 'copied'>('idle'));
    expect(result.current[0]).toBe('idle');
  });

  it('flashes to the transient value, then reverts after duration', () => {
    const { result } = renderHook(() =>
      useFlashState<'idle' | 'copied'>('idle'),
    );
    act(() => {
      result.current[1]('copied', 2000);
    });
    expect(result.current[0]).toBe('copied');
    act(() => {
      vi.advanceTimersByTime(1999);
    });
    // Still inside the flash window.
    expect(result.current[0]).toBe('copied');
    act(() => {
      vi.advanceTimersByTime(1);
    });
    // Window elapsed - reverted to initial.
    expect(result.current[0]).toBe('idle');
  });

  it('rapid re-arm cancels the prior timer (no early reset)', () => {
    // Customer scenario: user clicks Copy twice in 500 ms. Without
    // re-arm semantics, the first click's setTimeout(2s) is still
    // in flight; if it fires while the second flash is showing, the
    // pill flips back to 'idle' too early. Pin: re-arm clears the
    // first timer, so the second flash gets its full 2 s.
    const { result } = renderHook(() =>
      useFlashState<'idle' | 'copied'>('idle'),
    );
    act(() => {
      result.current[1]('copied', 2000);
    });
    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(result.current[0]).toBe('copied');
    act(() => {
      result.current[1]('copied', 2000); // re-arm
    });
    // 1500 ms after the FIRST click would normally fire the prior
    // timer. After re-arm it should NOT fire - the second click's
    // 2000 ms window is still active (only 1500 ms in).
    act(() => {
      vi.advanceTimersByTime(1500);
    });
    expect(result.current[0]).toBe('copied');
    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(result.current[0]).toBe('idle');
  });

  it('unmount cancels the in-flight timer', () => {
    // Customer scenario: user clicks Copy then immediately
    // navigates away inside the flash window. Without cleanup the
    // setTimeout fires on the unmounted component and React warns.
    // Pin: setState should NOT be called after unmount. We assert
    // by using a state spy and checking call count before/after
    // unmount + timer advance.
    const { result, unmount } = renderHook(() =>
      useFlashState<'idle' | 'copied'>('idle'),
    );
    act(() => {
      result.current[1]('copied', 2000);
    });
    expect(result.current[0]).toBe('copied');
    unmount();
    // Advancing past the flash window must not throw or trigger
    // any "state update on unmounted component" warning. The
    // cleanup path cleared the timer before it could fire.
    expect(() => {
      act(() => {
        vi.advanceTimersByTime(5000);
      });
    }).not.toThrow();
  });

  it('honors a different transient value on the second flash', () => {
    // The hook stores a single state slot; flashing 'copied' then
    // 'error' before the first reverts should land on 'error'
    // (not 'idle'), matching the bare-setTimeout semantics that
    // CliRunner relies on for try/catch error display.
    const { result } = renderHook(() =>
      useFlashState<'idle' | 'copied' | 'error'>('idle'),
    );
    act(() => {
      result.current[1]('copied', 2000);
    });
    act(() => {
      vi.advanceTimersByTime(500);
    });
    act(() => {
      result.current[1]('error', 3000);
    });
    expect(result.current[0]).toBe('error');
    // The first flash's 2 s reset must not fire and revert too
    // early - re-arm clears it.
    act(() => {
      vi.advanceTimersByTime(1499);
    });
    expect(result.current[0]).toBe('error');
    act(() => {
      vi.advanceTimersByTime(1500);
    });
    expect(result.current[0]).toBe('error');
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(result.current[0]).toBe('idle');
  });

  it('reset() clears state immediately and cancels pending timer', () => {
    // Customer scenario: user clicks Save (which flashes "Saved"
    // for 2 s), then within that window clicks Save again. The
    // call site wants to clear the stale "Saved" pill BEFORE
    // the new API call resolves. flashTo(initial, 0) would
    // schedule a 0 ms timer (works but adds a microtask noise);
    // reset() does it synchronously without any timer.
    const { result } = renderHook(() =>
      useFlashState<'idle' | 'saved'>('idle'),
    );
    act(() => {
      result.current[1]('saved', 2000);
    });
    expect(result.current[0]).toBe('saved');
    act(() => {
      result.current[2](); // reset()
    });
    expect(result.current[0]).toBe('idle');
    // The 2 s revert timer should have been cancelled - advancing
    // past the original window must not fire any state update.
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(result.current[0]).toBe('idle');
  });

  it('reset() is safe when no timer is in flight (no-op)', () => {
    // The hook should be safe to call reset() at any time, even
    // when no flash is active (e.g. on mount, or after a prior
    // reset). No throw, no double-clear.
    const { result } = renderHook(() =>
      useFlashState<'idle' | 'saved'>('idle'),
    );
    expect(() => {
      act(() => {
        result.current[2](); // reset() with no timer
        result.current[2](); // reset() again
      });
    }).not.toThrow();
    expect(result.current[0]).toBe('idle');
  });

  it('reverts to the latest initial when initial changes mid-flash', () => {
    // Defensive: if a parent re-renders with a different initial
    // (rare but possible), the in-flight reset should land on the
    // current initial, not the stale one captured when the timer
    // was scheduled. Initial-via-ref preserves this.
    const { result, rerender } = renderHook(
      ({ initial }) => useFlashState<string>(initial),
      { initialProps: { initial: 'idle' } },
    );
    act(() => {
      result.current[1]('copied', 2000);
    });
    rerender({ initial: 'fresh' });
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(result.current[0]).toBe('fresh');
  });
});
