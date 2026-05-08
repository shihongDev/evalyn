/**
 * Tests for runCli's hard-timeout behavior. The other branches
 * (CSRF retry, capacity 503, success) are exercised end-to-end
 * via TestClient on the backend; this file covers the FE-only
 * timeout-on-AbortError path.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { runCli, RUN_CLI_TIMEOUT_MS } from './cli';

describe('runCli timeout', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Install a workbench-token meta tag so readCsrfToken() returns
    // a non-null value (avoids confusing the CSRF retry path).
    const existing = document.querySelector('meta[name="workbench-token"]');
    if (existing) existing.remove();
    const meta = document.createElement('meta');
    meta.setAttribute('name', 'workbench-token');
    meta.setAttribute('content', 'TOK');
    document.head.appendChild(meta);
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    const meta = document.querySelector('meta[name="workbench-token"]');
    if (meta) meta.remove();
    vi.useRealTimers();
  });

  it('AbortError from the internal timer surfaces a clear message', async () => {
    // Simulate a wedged server: fetch listens for the abort signal
    // and rejects with the standard DOMException('AbortError').
    fetchSpy.mockImplementation((_url, init) => {
      return new Promise((_resolve, reject) => {
        const signal = (init as RequestInit).signal as AbortSignal | undefined;
        signal?.addEventListener('abort', () => {
          reject(new DOMException('aborted', 'AbortError'));
        });
      });
    });

    vi.useFakeTimers();
    const promise = runCli('run-eval', { dataset: 'x' });
    // Advance past the timeout to fire the AbortController.
    vi.advanceTimersByTime(RUN_CLI_TIMEOUT_MS + 1);
    await expect(promise).rejects.toThrow(
      `Server didn't respond within ${RUN_CLI_TIMEOUT_MS / 1000}s`,
    );
  });

  it('passes through non-AbortError rejections (e.g. network failure)', async () => {
    fetchSpy.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(runCli('run-eval', { dataset: 'x' })).rejects.toThrow(
      'Failed to fetch',
    );
  });

  it('attaches AbortSignal to the fetch init', async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ job_id: 'j1' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await runCli('run-eval', { dataset: 'x' });
    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeDefined();
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });
});
