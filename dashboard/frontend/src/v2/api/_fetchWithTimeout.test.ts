/**
 * Tests for fetchWithTimeout - the shared mutation-fetch helper.
 *
 * Three behaviors to pin:
 *   1. successful fetch passes through unchanged
 *   2. timeout aborts and throws the configured message
 *   3. external signal abort short-circuits the timer
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchWithTimeout } from './_fetchWithTimeout';

describe('fetchWithTimeout', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    vi.useRealTimers();
  });

  it('passes through a successful response', async () => {
    fetchSpy.mockResolvedValueOnce(new Response('ok', { status: 200 }));
    const res = await fetchWithTimeout('/x', undefined, 5000, 'timed out');
    expect(res.status).toBe(200);
    expect(await res.text()).toBe('ok');
  });

  it('throws the configured timeout message after timeoutMs', async () => {
    fetchSpy.mockImplementation((_url, init) => {
      return new Promise((_resolve, reject) => {
        const signal = (init as RequestInit).signal as AbortSignal | undefined;
        signal?.addEventListener('abort', () => {
          reject(new DOMException('aborted', 'AbortError'));
        });
      });
    });

    vi.useFakeTimers();
    const promise = fetchWithTimeout('/x', undefined, 5000, 'custom timeout msg');
    vi.advanceTimersByTime(5001);
    await expect(promise).rejects.toThrow('custom timeout msg');
  });

  it('passes through non-AbortError rejections (e.g. network failure)', async () => {
    fetchSpy.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(
      fetchWithTimeout('/x', undefined, 5000, 'timed out'),
    ).rejects.toThrow('Failed to fetch');
  });

  it('honors an external AbortSignal as well as the timer', async () => {
    fetchSpy.mockImplementation((_url, init) => {
      return new Promise((_resolve, reject) => {
        const signal = (init as RequestInit).signal as AbortSignal | undefined;
        signal?.addEventListener('abort', () => {
          reject(new DOMException('aborted', 'AbortError'));
        });
      });
    });

    const ext = new AbortController();
    const promise = fetchWithTimeout(
      '/x',
      { signal: ext.signal },
      60_000,
      'timed out',
    );
    // Caller's external abort fires before the timer would.
    ext.abort();
    await expect(promise).rejects.toThrow('timed out');
  });
});
