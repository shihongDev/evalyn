/**
 * Imperative bridge for opening the global Recent Jobs drawer
 * from any route. Mirrors the cliRunnerBridge pattern: a tiny
 * module-level emitter so non-AppShell surfaces can request the
 * drawer open without threading `setJobsDrawerOpen` through props.
 *
 * Why a bridge instead of zustand: the drawer's open state is
 * pure transient UI - storing it in the global store would couple
 * every store consumer to drawer mechanics. The emitter has a
 * surface area of two functions and stays out of the way.
 *
 * Use cases (so far):
 *   - SystemStatusCard's "Failures (24h)" row: clicking opens the
 *     drawer with the failure filter pre-applied so the operator
 *     can see WHICH runs failed without manually flipping a chip.
 *   - Future: deep-linking from a notification, command-palette
 *     entry, etc.
 */

/** Optional payload supplied when opening the drawer programmatically. */
export interface JobsDrawerOpenOptions {
  /**
   * When true, the drawer mounts with the failure-filter chip
   * already on so the user lands directly in the "show me failed
   * jobs" view. The chip is still toggleable inside the drawer -
   * this is a starting point, not a lock.
   */
  failureFilter?: boolean;
}

export interface JobsDrawerOpenEvent {
  open: boolean;
  failureFilter?: boolean;
  /**
   * Monotonic counter so a subscriber re-applies options even
   * when the drawer is "opened" while already open (e.g. user
   * clicks the same trigger twice with different filters).
   */
  nonce: number;
}

type Listener = (e: JobsDrawerOpenEvent) => void;

const listeners = new Set<Listener>();
let currentEvent: JobsDrawerOpenEvent = { open: false, nonce: 0 };

export function openJobsDrawer(opts?: JobsDrawerOpenOptions): void {
  currentEvent = {
    open: true,
    failureFilter: opts?.failureFilter,
    nonce: currentEvent.nonce + 1,
  };
  for (const fn of listeners) fn(currentEvent);
}

export function closeJobsDrawer(): void {
  currentEvent = { open: false, nonce: currentEvent.nonce + 1 };
  for (const fn of listeners) fn(currentEvent);
}

/** Subscribe to drawer-open events. Replays the current value so
 * late-mounting subscribers (e.g. AppShell that mounts after a
 * route transition) don't miss an open call that fired during
 * the same tick. */
export function subscribeJobsDrawer(fn: Listener): () => void {
  listeners.add(fn);
  fn(currentEvent);
  return () => {
    listeners.delete(fn);
  };
}
