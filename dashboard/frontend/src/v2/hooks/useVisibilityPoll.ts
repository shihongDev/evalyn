/**
 * useVisibilityPoll - the canonical "poll a fetcher on an interval,
 * pause on tab-hidden, refetch on tab-return" pattern.
 *
 * Used by:
 *   - SystemStatusCard (15s) - /api/health snapshot
 *   - RecentJobsDrawer capacity chip (5s, gated on `open`) - the
 *     drawer-only retrofit comes in a follow-up tick once this
 *     hook is proven via SystemStatusCard
 *
 * Why a hook: the start/stop visibility plumbing is ~25 lines of
 * subtle code (intervalId in scope, idempotent start, visible-on-
 * mount gate, cleanup ordering). Two near-identical copies were
 * paying interest; a shared hook lets future call sites pick this
 * up without re-deriving the contract.
 *
 * Why not just a generic interval hook: the visibility-pause is
 * the LOAD-bearing behavior here. A naive `useInterval` would
 * waste backend cycles when the user is on another tab AND show
 * stale data on tab-return. Bundling both into one hook makes
 * the path of least resistance the right one.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface UseVisibilityPollOptions<T> {
  /** Async fetcher that returns the current snapshot. Resolved
   * value is delivered via the hook's `value` state. Errors are
   * NOT caught here - the fetcher should already return null /
   * default on failure (matches the existing
   * `fetchSystemHealth` / `fetchJobsCapacity` pattern). */
  fetcher: () => Promise<T>;
  /** Interval between polls when the tab is visible, in ms. The
   * interval timer is paused (cleared) when the tab is hidden
   * and restarted on tab-return. */
  intervalMs: number;
  /** Optional gate. When false, the hook does not poll AND does
   * not subscribe to visibility events. Used by call sites where
   * polling should only run while a parent surface is open
   * (e.g. drawer capacity-chip while the drawer itself is open).
   * Defaults to true. Toggling false -> true triggers a fresh
   * fetch like a tab-return; true -> false stops polling and
   * removes the listener. */
  enabled?: boolean;
}

export interface UseVisibilityPollResult<T> {
  /** Latest fetched value, or `null` until the first fetch
   * resolves. Subscribers should treat null as "loading or
   * unavailable" and render a skeleton or hide the surface. */
  value: T | null;
  /** True once the first fetch has resolved (regardless of
   * outcome). Useful for distinguishing initial-load skeleton
   * from "loaded but null/empty". */
  loaded: boolean;
  /** Wall-clock millisecond timestamp of the last successful
   * fetch resolution. `null` until the first fetch resolves.
   * Useful for rendering a "Refreshed Ns ago" indicator so
   * the user can spot a stalled poll loop (visibility-paused
   * tabs return stale-looking cards on tab-return until the
   * next tick lands). Updated on EACH fetch, not just identity-
   * changing ones, so a fresh poll that returned identical data
   * still counts as "the data is current". */
  lastFetchAt: number | null;
  /** Manual refetch. Useful after a user action that's expected
   * to change the polled state (e.g. clicking "Compact now" on
   * the SystemStatusCard - immediate refetch lands the post-
   * vacuum size on the card without waiting for the next poll
   * tick). */
  refetch: () => void;
}

export function useVisibilityPoll<T>(
  options: UseVisibilityPollOptions<T>,
): UseVisibilityPollResult<T> {
  const { fetcher, intervalMs, enabled = true } = options;

  const [value, setValue] = useState<T | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [lastFetchAt, setLastFetchAt] = useState<number | null>(null);

  // Stash the latest fetcher in a ref so the polling effect
  // captures it without restarting when the caller passes a
  // new fetcher reference (e.g. closes over fresh state). This
  // matches the React community pattern for "callback held by
  // a long-lived effect" without forcing the caller to memoize.
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  }, [fetcher]);

  // Stash the manual refetch trigger in a ref so the public
  // `refetch` callback can call it without depending on the
  // effect's closure. The effect installs a fresh trigger each
  // mount.
  const triggerRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let intervalId: number | null = null;
    // In-flight dedup. A user mashing the Refresh button (or rapid
    // visibilitychange flips) would otherwise fire N concurrent
    // fetcher() calls for no benefit - the server gets pummeled and
    // the state effects race over the same logical slot. Hold the
    // current promise here; concurrent callers reuse it.
    let pending: Promise<void> | null = null;

    async function poll() {
      if (pending) return pending;
      pending = (async () => {
        try {
          const next = await fetcherRef.current();
          if (cancelled) return;
          setValue(next);
          setLoaded(true);
          // Stamp the timestamp regardless of value identity. Two
          // consecutive identical responses still mean "the data is
          // current"; the freshness indicator should not freeze on
          // an unchanged value.
          setLastFetchAt(Date.now());
        } finally {
          pending = null;
        }
      })();
      return pending;
    }

    function start() {
      if (intervalId !== null) return;
      void poll();
      intervalId = window.setInterval(() => void poll(), intervalMs);
    }

    function stop() {
      if (intervalId !== null) {
        window.clearInterval(intervalId);
        intervalId = null;
      }
    }

    triggerRef.current = () => void poll();

    if (typeof document === 'undefined' || document.visibilityState === 'visible') {
      start();
    }
    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        start();
      } else {
        stop();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      cancelled = true;
      stop();
      document.removeEventListener('visibilitychange', onVisibility);
      triggerRef.current = null;
    };
  }, [enabled, intervalMs]);

  const refetch = useCallback(() => {
    triggerRef.current?.();
  }, []);

  return { value, loaded, lastFetchAt, refetch };
}
