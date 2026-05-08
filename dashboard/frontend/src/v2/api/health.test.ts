/**
 * Tests pin that vacuumPersistence + pruneOldJobs send the correct
 * CSRF header name. The server's CSRF middleware checks
 * `X-Workbench-Token` (server.py:CSRF_HEADER); a previous bug had
 * health.ts sending `X-CSRF-Token` instead, which made every
 * Compact / Prune click 403 in production. This test catches that
 * regression class by spying on `globalThis.fetch` and asserting
 * the header dictionary the helper passed.
 *
 * Existing test coverage in test_api_jobs.py only verified that a
 * MISSING header gets 403 - it never exercised a request with the
 * wrong-name header. So this is the missing pin.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { vacuumPersistence, pruneOldJobs } from './health';

const TOKEN = 'WORKBENCH_TOKEN_FOR_TEST';

describe('admin endpoints CSRF header', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Install a workbench-token meta tag so readCsrfToken() returns
    // a non-null value.
    const existing = document.querySelector('meta[name="workbench-token"]');
    if (existing) existing.remove();
    const meta = document.createElement('meta');
    meta.setAttribute('name', 'workbench-token');
    meta.setAttribute('content', TOKEN);
    document.head.appendChild(meta);
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    const meta = document.querySelector('meta[name="workbench-token"]');
    if (meta) meta.remove();
  });

  it('vacuumPersistence sends X-Workbench-Token header', async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ ok: true, before: 100, after: 50, bytes_saved: 50 }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    await vacuumPersistence();
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    // The previous bug shipped 'X-CSRF-Token' which the server
    // ignored, leading to a 403 on every Compact click. Pin the
    // correct header name so the regression can't return.
    expect(headers['X-Workbench-Token']).toBe(TOKEN);
    // Defensive: the wrong header name must NOT also be sent.
    expect(headers['X-CSRF-Token']).toBeUndefined();
  });

  it('pruneOldJobs sends X-Workbench-Token header', async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, deleted: 5, kept: 100 }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await pruneOldJobs(100);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers['X-Workbench-Token']).toBe(TOKEN);
    expect(headers['X-CSRF-Token']).toBeUndefined();
  });
});
