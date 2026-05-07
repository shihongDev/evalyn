/**
 * Home route helper tests.
 *
 * `experimentIdFromPath` parses the QueueRow's `cta_target` to
 * decide whether the hover/focus warmup should fire. A parser bug
 * here means hover-prefetch silently no-ops on valid paths
 * (perf regression, hard to spot) OR fires on non-experiment
 * paths (wasted fetch, also hard to spot). Pinning the contract.
 */

import { describe, expect, it } from 'vitest';
import { experimentIdFromPath } from './Home';

describe('experimentIdFromPath', () => {
  it('extracts the id from a clean experiment route', () => {
    expect(experimentIdFromPath('/experiments/abc123')).toBe('abc123');
  });

  it('extracts the id even with a query string', () => {
    expect(experimentIdFromPath('/experiments/abc123?tab=items')).toBe(
      'abc123',
    );
  });

  it('extracts the id even with a hash', () => {
    expect(experimentIdFromPath('/experiments/abc123#section')).toBe('abc123');
  });

  it('decodes percent-encoded characters in the id', () => {
    // Real-world: an id might contain spaces or special chars
    // that get URL-encoded during nav. Decoding ensures the
    // prefetch key matches the route's actual `:runId` param.
    expect(experimentIdFromPath('/experiments/run%20one')).toBe('run one');
  });

  it('returns the id segment only, not deeper path segments', () => {
    // Defensive: future routes might have /experiments/<id>/foo,
    // we still want just the id for the cache key.
    expect(experimentIdFromPath('/experiments/abc/items')).toBe('abc');
  });

  it('returns null for non-experiment paths', () => {
    expect(experimentIdFromPath('/datasets/foo')).toBeNull();
    expect(experimentIdFromPath('/copilot')).toBeNull();
    expect(experimentIdFromPath('/')).toBeNull();
    expect(experimentIdFromPath('')).toBeNull();
  });

  it('returns null for an empty experiment id', () => {
    // /experiments/ alone has no id segment.
    expect(experimentIdFromPath('/experiments/')).toBeNull();
    expect(experimentIdFromPath('/experiments')).toBeNull();
  });

  it('does not match similarly-named prefixes', () => {
    // `/experimentsfoo` is not /experiments/.
    expect(experimentIdFromPath('/experimentsfoo')).toBeNull();
  });
});
