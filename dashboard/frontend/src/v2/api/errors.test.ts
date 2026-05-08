import { describe, expect, it } from 'vitest';
import {
  formatCapacityRetryHint,
  errorMessage,
  CapacityError,
  maybeParseCapacityError,
} from './errors';

describe('formatCapacityRetryHint', () => {
  it('says "now" when the server hint is 0 or 1 second', () => {
    expect(formatCapacityRetryHint(0)).toBe('Try again now');
    expect(formatCapacityRetryHint(1)).toBe('Try again now');
  });

  it('renders precise seconds for values in [2, 59]', () => {
    expect(formatCapacityRetryHint(2)).toBe('Try again in 2s');
    expect(formatCapacityRetryHint(5)).toBe('Try again in 5s');
    expect(formatCapacityRetryHint(30)).toBe('Try again in 30s');
    expect(formatCapacityRetryHint(59)).toBe('Try again in 59s');
  });

  it('rounds down to whole minutes for >= 60 seconds', () => {
    expect(formatCapacityRetryHint(60)).toBe('Try again in 1m');
    expect(formatCapacityRetryHint(89)).toBe('Try again in 1m');
    expect(formatCapacityRetryHint(120)).toBe('Try again in 2m');
    expect(formatCapacityRetryHint(305)).toBe('Try again in 5m');
  });
});

describe('errorMessage', () => {
  it('extracts message from Error instances', () => {
    expect(errorMessage(new Error('boom'))).toBe('boom');
  });

  it('handles CapacityError (subclass of Error)', () => {
    const e = new CapacityError({
      message: 'queue full',
      running: 8,
      maxConcurrent: 8,
      retryAfterSeconds: 10,
    });
    expect(errorMessage(e)).toBe('queue full');
  });

  it('falls back to String() for non-Error values', () => {
    expect(errorMessage('plain string')).toBe('plain string');
    expect(errorMessage(42)).toBe('42');
  });

  it('rewrites Chrome fetch network failures to a friendly message', () => {
    const e = new TypeError('Failed to fetch');
    expect(errorMessage(e)).toBe(
      'Network unreachable - check your connection and try again.',
    );
  });

  it('rewrites Safari Load-failed network failures to a friendly message', () => {
    const e = new TypeError('Load failed');
    expect(errorMessage(e)).toBe(
      'Network unreachable - check your connection and try again.',
    );
  });

  it('rewrites Firefox NetworkError fetch failures to a friendly message', () => {
    const e = new TypeError(
      'NetworkError when attempting to fetch resource.',
    );
    expect(errorMessage(e)).toBe(
      'Network unreachable - check your connection and try again.',
    );
  });

  it('does NOT rewrite TypeErrors with unrelated messages', () => {
    // A TypeError from e.g. "x.toLowerCase is not a function" should
    // NOT be hijacked - that would mask programmer bugs as network
    // problems. Only known-network-shaped messages get the rewrite.
    const e = new TypeError("Cannot read properties of undefined (reading 'foo')");
    expect(errorMessage(e)).toBe(
      "Cannot read properties of undefined (reading 'foo')",
    );
  });

  it('does NOT rewrite plain Errors that happen to mention "network"', () => {
    // A server returning {error: "Network error from upstream API"}
    // would be wrapped in a regular Error by jsonFetch. We don't
    // want the heuristic to kick in - the server's message is more
    // precise than ours.
    const e = new Error('Network error from upstream API');
    expect(errorMessage(e)).toBe('Network error from upstream API');
  });
});

describe('maybeParseCapacityError', () => {
  function build503(
    body: Record<string, unknown>,
    headers: Record<string, string> = {},
  ): Response {
    return new Response(JSON.stringify(body), {
      status: 503,
      headers: { 'Content-Type': 'application/json', ...headers },
    });
  }

  it('returns null for non-503 responses', async () => {
    const res = new Response('{}', { status: 500 });
    expect(await maybeParseCapacityError(res)).toBeNull();
  });

  it('parses a well-formed capacity error', async () => {
    const res = build503(
      { running: 8, max_concurrent: 8, error: 'queue full' },
      { 'Retry-After': '7' },
    );
    const e = await maybeParseCapacityError(res);
    expect(e).toBeInstanceOf(CapacityError);
    expect(e?.running).toBe(8);
    expect(e?.maxConcurrent).toBe(8);
    expect(e?.retryAfterSeconds).toBe(7);
    expect(e?.message).toBe('queue full');
  });

  it('preserves Retry-After: 0 (was collapsed to 5 by `|| 5`)', async () => {
    // Customer scenario: a load balancer or future server returning
    // "retry immediately" via Retry-After: 0. Pre-fix the FE used
    // parseInt('0') || 5 which evaluated to 5, forcing a synthetic
    // 5s wait the user shouldn't have. Pin the zero-second hint.
    const res = build503(
      { running: 8, max_concurrent: 8 },
      { 'Retry-After': '0' },
    );
    const e = await maybeParseCapacityError(res);
    expect(e?.retryAfterSeconds).toBe(0);
  });

  it('falls back to 5 on missing Retry-After header', async () => {
    const res = build503({ running: 8, max_concurrent: 8 });
    const e = await maybeParseCapacityError(res);
    expect(e?.retryAfterSeconds).toBe(5);
  });

  it('falls back to 5 on unparseable Retry-After header', async () => {
    const res = build503(
      { running: 8, max_concurrent: 8 },
      { 'Retry-After': 'tomorrow' },
    );
    const e = await maybeParseCapacityError(res);
    expect(e?.retryAfterSeconds).toBe(5);
  });

  it('falls back to 5 on negative Retry-After (defensive)', async () => {
    const res = build503(
      { running: 8, max_concurrent: 8 },
      { 'Retry-After': '-1' },
    );
    const e = await maybeParseCapacityError(res);
    expect(e?.retryAfterSeconds).toBe(5);
  });
});
