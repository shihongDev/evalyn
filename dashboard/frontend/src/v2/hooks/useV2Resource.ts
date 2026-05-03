/**
 * useV2Resource - module-level memoization for v2 API fetches.
 *
 * Why: nav clicks (Home -> Experiments -> Home) used to refetch every page.
 * The cache makes the second visit instant; data older than STALE_AFTER_MS
 * is refetched in the background while the cached copy renders immediately.
 *
 * Inflight de-duplication ensures hover-prefetch + click-on-the-same-route
 * fire only one network request even if both happen within milliseconds.
 *
 * Live updates: each mounted resource subscribes once to ``/ws/v2/events``
 * via subscribeV2Events. When the backend pushes a ``cache_invalidate``
 * frame whose key list matches this resource's key (by prefix), the
 * cache entry is dropped and a background refetch runs. Stale-while-
 * revalidate semantics keep the UI smooth - the old data renders until
 * the new payload lands.
 *
 * Limits:
 * - Cache lives in module scope -> cleared on full page reload (acceptable
 *   for the v2 first cut; the route still loads fresh on refresh).
 * - No cache eviction strategy yet; the working set is small (~7 endpoints).
 */

import { useCallback, useEffect, useState } from 'react';
import { subscribeV2Events, type V2Event } from '../api/v2ws';

const _cache = new Map<string, { data: unknown; ts: number }>();
const _inflight = new Map<string, Promise<unknown>>();
const STALE_AFTER_MS = 30_000;

/**
 * Match a resource ``key`` against the invalidation ``keys`` set from
 * the backend. The backend lists base names (``"experiments"``,
 * ``"dataset"``, etc.); a resource key matches when it equals a base
 * name OR when it starts with ``"<base>:"`` (the convention used by
 * ``experiment:<id>``, ``dataset:<name>``, ``rubric:<id>``, and the
 * paginated experiment-items keys).
 */
function keyMatchesInvalidation(
  resourceKey: string,
  invalidatedKeys: string[],
): boolean {
  for (const base of invalidatedKeys) {
    if (resourceKey === base) return true;
    if (resourceKey.startsWith(`${base}:`)) return true;
  }
  return false;
}

interface ResourceState<T> {
  /** Cached or freshly-loaded data; null until the first response lands. */
  data: T | null;
  /** Latest error message, or null. Cleared on a successful refetch. */
  err: string | null;
  /** Imperative refetch (used by refresh buttons, regenerate, etc). */
  refetch: () => Promise<void>;
  /** True while a network request is in flight (for spinners). */
  reloading: boolean;
  /**
   * True iff this is the first visit to the resource (no cached copy).
   * Routes use this to choose between skeleton (first load) and the
   * cached layout + corner "Updating..." chip (background refresh).
   */
  isInitialLoad: boolean;
}

export function useV2Resource<T>(
  key: string,
  fetcher: () => Promise<T>,
  options: { enabled?: boolean } = {},
): ResourceState<T> {
  const enabled = options.enabled ?? true;
  const cached = _cache.get(key) as { data: T; ts: number } | undefined;
  const isStale = cached ? Date.now() - cached.ts > STALE_AFTER_MS : true;
  const [data, setData] = useState<T | null>(cached?.data ?? null);
  const [err, setErr] = useState<string | null>(null);
  const [reloading, setReloading] = useState(false);

  const refetch = useCallback(async (): Promise<void> => {
    if (!enabled) return;
    setReloading(true);
    let p = _inflight.get(key) as Promise<T> | undefined;
    if (!p) {
      p = fetcher();
      _inflight.set(key, p);
      void p.finally(() => _inflight.delete(key));
    }
    try {
      const d = await p;
      _cache.set(key, { data: d, ts: Date.now() });
      setData(d);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setReloading(false);
    }
  }, [key, fetcher, enabled]);

  useEffect(() => {
    if (enabled && (!cached || isStale)) {
      void refetch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);

  // Subscribe once per mounted resource. The backend emits coarse
  // ``cache_invalidate`` frames for any dataset-root mtime change; we
  // drop our cache entry and refetch only when the keys overlap.
  useEffect(() => {
    if (!enabled) return;
    const off = subscribeV2Events((evt: V2Event) => {
      if (evt.type !== 'cache_invalidate') return;
      if (!keyMatchesInvalidation(key, evt.keys)) return;
      _cache.delete(key);
      void refetch();
    });
    return off;
  }, [key, enabled, refetch]);

  return { data, err, refetch, reloading, isInitialLoad: !cached };
}

/**
 * Imperative prefetch for hover/focus warmup. Safe to spam - returns early
 * if the cache is fresh OR a request is already in flight for the key.
 */
export function prefetchV2<T>(key: string, fetcher: () => Promise<T>): void {
  const cached = _cache.get(key);
  const isStale = cached ? Date.now() - cached.ts > STALE_AFTER_MS : true;
  if (!isStale || _inflight.has(key)) return;
  const p = fetcher();
  _inflight.set(key, p);
  void p
    .then((d) => {
      _cache.set(key, { data: d, ts: Date.now() });
    })
    .catch(() => {
      /* swallow - the visiting route will surface the error on its real fetch */
    })
    .finally(() => _inflight.delete(key));
}

/** Test-only: clear the in-memory cache between specs. */
export function _resetV2Cache(): void {
  _cache.clear();
  _inflight.clear();
}
