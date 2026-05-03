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

type RunnerListener = (cli: CliSchema | null) => void;

const listeners = new Set<RunnerListener>();
let currentCli: CliSchema | null = null;

/** Open the runner panel for ``cli``. Idempotent; safe to call from any route. */
export function openCliRunner(cli: CliSchema): void {
  currentCli = cli;
  for (const fn of listeners) fn(cli);
}

/** Close the runner panel (also called by the component's X button). */
export function closeCliRunner(): void {
  currentCli = null;
  for (const fn of listeners) fn(null);
}

/** Subscribe to runner state changes. Replays the current value to ``fn`` so
 * a late-mounting subscriber doesn't miss an open call that fired during
 * the same tick. Returns an unsubscribe function. */
export function subscribeRunner(fn: RunnerListener): () => void {
  listeners.add(fn);
  fn(currentCli);
  return () => {
    listeners.delete(fn);
  };
}
