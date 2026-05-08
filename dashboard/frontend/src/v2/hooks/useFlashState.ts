/**
 * useFlashState - "set this state for N ms, then revert to the
 * initial value" with safe cleanup.
 *
 * Customer-cared scenario: a Copy / Save / Send button flips a
 * label or pill to "copied" / "saved" / "sent" for a couple of
 * seconds before reverting. The naive pattern is:
 *
 *   const [s, setS] = useState('idle');
 *   ...
 *   setS('copied');
 *   window.setTimeout(() => setS('idle'), 2000);
 *
 * which leaks the timer on unmount (React warns with "Can't
 * perform a state update on an unmounted component" when the
 * user navigates away inside the 2 s window) and stacks timers
 * on rapid re-clicks (the second click's reset can fire
 * earlier than expected because the first click's reset is
 * still in flight).
 *
 * This hook returns ``[state, flashTo, reset]``:
 *   - ``flashTo(transient, durationMs)`` flips state to ``transient``,
 *     cancels any in-flight reset, and schedules a fresh reset to
 *     the initial value after ``durationMs``.
 *   - ``reset()`` immediately drops state back to the initial value
 *     and cancels any in-flight reset timer. Used by the
 *     "pre-API-call clear" pattern (e.g. clear a stale "Saved"
 *     pill before kicking off a new save) where the auto-revert
 *     timer would be wrong because the call site wants an
 *     immediate, no-auto-undo clear.
 *
 * Unmount cleanup clears any pending reset so the timer callback
 * never fires on an unmounted component.
 *
 * Same defensive pattern as ``useArmedConfirm``; the duplication
 * across copy/save/sent buttons is what motivated factoring this
 * out.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export function useFlashState<T>(
  initial: T,
): [T, (transient: T, durationMs: number) => void, () => void] {
  const [state, setState] = useState<T>(initial);
  // Keep the latest ``initial`` in a ref so the reset callback
  // honors a changed initial without recreating ``flashTo`` (which
  // would force callers to memoize their handlers).
  const initialRef = useRef(initial);
  initialRef.current = initial;
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, []);

  const flashTo = useCallback(
    (transient: T, durationMs: number) => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      setState(transient);
      timerRef.current = window.setTimeout(() => {
        setState(initialRef.current);
        timerRef.current = null;
      }, durationMs);
    },
    [],
  );

  // Imperative reset: clear any in-flight timer + drop state to
  // initial. Useful for the "pre-API-call clear" pattern (e.g.
  // clear stale "Saved" pill before a fresh save). flashTo's
  // implicit revert wouldn't fit that case because the call site
  // wants the clear to happen NOW, not after a duration.
  const reset = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setState(initialRef.current);
  }, []);

  return [state, flashTo, reset];
}
