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
 * This hook returns ``[state, flashTo]`` where ``flashTo`` cancels
 * any in-flight reset before scheduling a fresh one, and the
 * unmount cleanup clears any pending reset so the callback never
 * fires on an unmounted component.
 *
 * Callers that need an immediate reset (e.g. a parent prop
 * changes and we want to drop the flash early) can call
 * ``flashTo(initial, 0)`` - the new timer fires on the next
 * macrotask which is functionally an immediate reset, and the
 * prior in-flight timer is cancelled.
 *
 * Same defensive pattern as ``useArmedConfirm``; the duplication
 * across copy/save/sent buttons is what motivated factoring this
 * out.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export function useFlashState<T>(
  initial: T,
): [T, (transient: T, durationMs: number) => void] {
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

  return [state, flashTo];
}
