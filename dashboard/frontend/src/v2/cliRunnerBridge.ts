/**
 * Imperative bridge for the global :class:`CliRunner` panel.
 *
 * Lives in its own module (rather than alongside the component) so React
 * Fast Refresh stays happy: the lint rule
 * ``react-refresh/only-export-components`` requires component files to
 * export ONLY components. The runner needs to be invokable from any route
 * with a function call, so the open/close API ships from here and the
 * component subscribes to the same listener set.
 *
 * Why a tiny event emitter instead of zustand? The runner is purely
 * transient UI state (open + selected command + in-flight job). Pulling
 * it through the v2 store would couple every consumer to the store; a
 * module-level emitter keeps the surface area to two functions.
 */

import type { CliSchema } from './api/cli';

/** Optional payload supplied when opening the runner from a deep link. */
export interface RunnerOpenOptions {
  /**
   * Initial form values keyed by `CliParam.name`. Merged on top of
   * per-param defaults so any field the caller cares about appears
   * pre-filled while the rest fall back to schema defaults. The form
   * stays fully editable - this is just a starting point.
   */
  initialValues?: Record<string, unknown>;
  /**
   * Resume mode: skip the form and re-attach the runner directly to an
   * existing `job_id`. Used by the Recent Jobs drawer so the user can
   * navigate back to a job they kicked off earlier (or one still running
   * after a page reload). When set, the runner ignores `initialValues`
   * for input rendering and instead reads the args off the local
   * `jobsHistory` entry to display the original argv.
   */
  resumeJobId?: string;
}

/** Snapshot delivered to runner subscribers. `null` cli means closed. */
export interface RunnerState {
  cli: CliSchema | null;
  initialValues?: Record<string, unknown>;
  /** Job id to re-attach to instead of POST /api/cli/run. See RunnerOpenOptions. */
  resumeJobId?: string;
  /**
   * Monotonic counter incremented on every `openCliRunner` call so the
   * subscriber can re-key its body even when the same `cli` is opened
   * twice in a row with different `initialValues`.
   */
  nonce: number;
}

type RunnerListener = (state: RunnerState) => void;

const listeners = new Set<RunnerListener>();
let currentState: RunnerState = { cli: null, nonce: 0 };

/** Open the runner panel for ``cli``. Idempotent; safe to call from any route. */
export function openCliRunner(cli: CliSchema, opts?: RunnerOpenOptions): void {
  currentState = {
    cli,
    initialValues: opts?.initialValues,
    resumeJobId: opts?.resumeJobId,
    nonce: currentState.nonce + 1,
  };
  for (const fn of listeners) fn(currentState);
}

/** Close the runner panel (also called by the component's X button). */
export function closeCliRunner(): void {
  currentState = { cli: null, nonce: currentState.nonce + 1 };
  for (const fn of listeners) fn(currentState);
}

/** Subscribe to runner state changes. Replays the current value to ``fn`` so
 * a late-mounting subscriber doesn't miss an open call that fired during
 * the same tick. Returns an unsubscribe function. */
export function subscribeRunner(fn: RunnerListener): () => void {
  listeners.add(fn);
  fn(currentState);
  return () => {
    listeners.delete(fn);
  };
}
