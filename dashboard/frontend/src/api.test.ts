/**
 * Tests for the agent api.ts CSRF retry path. The v2/api/* endpoints
 * already had this pattern; this file covers the legacy api.ts that
 * was missing it.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';

describe('api.ts agent helpers - CSRF retry', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Install a workbench-token meta tag so readCsrfToken() reads
    // a known initial value.
    const existing = document.querySelector('meta[name="workbench-token"]');
    if (existing) existing.remove();
    const meta = document.createElement('meta');
    meta.setAttribute('name', 'workbench-token');
    meta.setAttribute('content', 'OLD_TOKEN');
    document.head.appendChild(meta);
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    const meta = document.querySelector('meta[name="workbench-token"]');
    if (meta) meta.remove();
  });

  it('attaches X-Workbench-Token on POST and succeeds on first try', async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ thread_id: 't-1' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const r = await api.startAgentThread('hi');
    expect(r.thread_id).toBe('t-1');
    // The mutation included the OLD_TOKEN header.
    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)['X-Workbench-Token']).toBe('OLD_TOKEN');
  });

  it('refreshes the token and retries once on 403', async () => {
    // First call: 403 (stale token). Second call (refresh GET /):
    // returns HTML with a new token. Third call: retry succeeds.
    fetchSpy
      .mockResolvedValueOnce(new Response('forbidden', { status: 403 }))
      .mockResolvedValueOnce(
        new Response(
          '<html><meta name="workbench-token" content="NEW_TOKEN"></html>',
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ thread_id: 't-2' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      );

    const r = await api.startAgentThread('hi');
    expect(r.thread_id).toBe('t-2');
    expect(fetchSpy).toHaveBeenCalledTimes(3);
    // The retry used the NEW_TOKEN.
    const retryInit = fetchSpy.mock.calls[2][1] as RequestInit;
    expect((retryInit.headers as Record<string, string>)['X-Workbench-Token']).toBe(
      'NEW_TOKEN',
    );
  });

  it('does not retry-loop: a second 403 surfaces as an error', async () => {
    fetchSpy
      .mockResolvedValueOnce(new Response('forbidden', { status: 403 }))
      .mockResolvedValueOnce(
        new Response(
          '<html><meta name="workbench-token" content="NEW_TOKEN"></html>',
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(new Response('still forbidden', { status: 403 }));

    await expect(api.startAgentThread('hi')).rejects.toThrow(/403/);
    expect(fetchSpy).toHaveBeenCalledTimes(3);
  });
});
