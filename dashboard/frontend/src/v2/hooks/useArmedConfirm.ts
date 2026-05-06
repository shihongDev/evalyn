/**
 * useArmedConfirm - inline two-click confirmation for destructive
 * actions where a modal would feel too heavy and a native window.confirm
 * jars against the design system.
 *
 * Pattern (caller side):
 *
 *   const { armed, arm, reset } = useArmedConfirm();
 *   function handleClick() {
 *     if (!armed) { arm(); return; }
 *     reset();
 *     reallyDestructiveThing();
 *   }
 *
 * UI side: when `armed`, flip the button (or menuitem) label to a
 * "Click again to confirm?"-style prompt. The arm flips back to false
 * automatically after `delayMs` if the user doesn't click again - same
 * forgiveness as a real-world "are you sure?" prompt that disappears
 * if you walk away from the keyboard.
 *
 * Cleanup: the timer is cancelled on unmount so leaving the surface
 * mid-arm doesn't leak.
 *
 * Used by Jobs drawer Clear history, Datasets bulk-evaluate, and
 * AnnotateSession's reset-bookmarks / clear-local-draft menu items.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

const DEFAULT_DELAY_MS = 4000;

export function useArmedConfirm(delayMs: number = DEFAULT_DELAY_MS): {
  armed: boolean;
  arm: () => void;
  reset: () => void;
} {
  const [armed, setArmed] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current != null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  const arm = useCallback(() => {
    setArmed(true);
    if (timerRef.current != null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(() => {
      setArmed(false);
      timerRef.current = null;
    }, delayMs);
  }, [delayMs]);

  const reset = useCallback(() => {
    if (timerRef.current != null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setArmed(false);
  }, []);

  return { armed, arm, reset };
}
