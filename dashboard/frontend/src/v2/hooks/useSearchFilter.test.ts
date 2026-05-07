/**
 * useSearchFilter unit tests.
 *
 * Covers:
 *   - debounced query update (input changes immediately, query lags)
 *   - sessionStorage round-trip when sessionKey is provided
 *   - missing sessionKey leaves storage untouched
 *   - empty input clears the storage entry instead of writing ""
 *   - private-mode / quota errors don't crash on read or write
 *   - Esc clears non-empty input (and stops propagation)
 *   - Esc on empty input blurs (no clear)
 *   - non-Esc keys pass through
 */

import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useSearchFilter } from './useSearchFilter';

describe('useSearchFilter', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    window.sessionStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('updates input immediately and query after debounce', () => {
    const { result } = renderHook(() => useSearchFilter({ debounceMs: 100 }));
    expect(result.current.input).toBe('');
    expect(result.current.query).toBe('');

    act(() => result.current.setInput('Foo Bar'));
    expect(result.current.input).toBe('Foo Bar');
    // Pre-debounce, the lowercased query has not propagated yet.
    expect(result.current.query).toBe('');

    act(() => {
      vi.advanceTimersByTime(100);
    });
    // Post-debounce, query is lowercased + trimmed.
    expect(result.current.query).toBe('foo bar');
  });

  it('persists to sessionStorage when sessionKey is provided', () => {
    const KEY = 'evalyn:test:search';
    const { result } = renderHook(() =>
      useSearchFilter({ sessionKey: KEY, debounceMs: 50 }),
    );

    act(() => result.current.setInput('hello'));
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(window.sessionStorage.getItem(KEY)).toBe('hello');
  });

  it('removes the storage entry when input is cleared', () => {
    const KEY = 'evalyn:test:clear';
    window.sessionStorage.setItem(KEY, 'previous');

    const { result } = renderHook(() =>
      useSearchFilter({ sessionKey: KEY, debounceMs: 50 }),
    );
    // Initial mount loads from storage.
    expect(result.current.input).toBe('previous');

    act(() => result.current.setInput(''));
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(window.sessionStorage.getItem(KEY)).toBeNull();
  });

  it('does not touch sessionStorage when sessionKey is omitted', () => {
    const setSpy = vi.spyOn(Storage.prototype, 'setItem');
    const { result } = renderHook(() => useSearchFilter({ debounceMs: 50 }));

    act(() => result.current.setInput('value'));
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(setSpy).not.toHaveBeenCalled();
    setSpy.mockRestore();
  });

  it('survives sessionStorage failures on read', () => {
    const getSpy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('quota');
    });
    const { result } = renderHook(() =>
      useSearchFilter({ sessionKey: 'ignored' }),
    );
    expect(result.current.input).toBe('');
    getSpy.mockRestore();
  });

  it('survives sessionStorage failures on write', () => {
    const setSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    const { result } = renderHook(() =>
      useSearchFilter({ sessionKey: 'k', debounceMs: 25 }),
    );
    act(() => result.current.setInput('x'));
    expect(() => {
      act(() => {
        vi.advanceTimersByTime(25);
      });
    }).not.toThrow();
    setSpy.mockRestore();
  });

  it('clear() empties the input', () => {
    const { result } = renderHook(() => useSearchFilter({ debounceMs: 50 }));
    act(() => result.current.setInput('typed'));
    act(() => result.current.clear());
    expect(result.current.input).toBe('');
  });

  it('Esc clears non-empty input and stops propagation', () => {
    const { result } = renderHook(() => useSearchFilter({ debounceMs: 50 }));
    act(() => result.current.setInput('something'));

    const stopPropagation = vi.fn();
    const preventDefault = vi.fn();
    const event = {
      key: 'Escape',
      stopPropagation,
      preventDefault,
    } as unknown as React.KeyboardEvent<HTMLInputElement>;

    act(() => result.current.onKeyDown(event));
    expect(result.current.input).toBe('');
    expect(stopPropagation).toHaveBeenCalled();
    expect(preventDefault).toHaveBeenCalled();
  });

  it('Esc on empty input blurs without clearing', () => {
    const { result } = renderHook(() => useSearchFilter({ debounceMs: 50 }));
    const blur = vi.fn();
    // Mock the inputRef.current.blur path.
    Object.defineProperty(result.current.inputRef, 'current', {
      value: { blur },
      writable: true,
    });

    const stopPropagation = vi.fn();
    const event = {
      key: 'Escape',
      stopPropagation,
      preventDefault: vi.fn(),
    } as unknown as React.KeyboardEvent<HTMLInputElement>;

    act(() => result.current.onKeyDown(event));
    expect(blur).toHaveBeenCalled();
    expect(stopPropagation).not.toHaveBeenCalled();
  });

  it('non-Esc keys are ignored', () => {
    const { result } = renderHook(() => useSearchFilter({ debounceMs: 50 }));
    act(() => result.current.setInput('keep'));
    const stopPropagation = vi.fn();
    const event = {
      key: 'Enter',
      stopPropagation,
      preventDefault: vi.fn(),
    } as unknown as React.KeyboardEvent<HTMLInputElement>;
    act(() => result.current.onKeyDown(event));
    expect(result.current.input).toBe('keep');
    expect(stopPropagation).not.toHaveBeenCalled();
  });
});
