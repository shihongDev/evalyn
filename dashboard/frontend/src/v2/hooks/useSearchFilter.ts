/**
 * useSearchFilter - the canonical "search input + debounced query"
 * pattern shared by Recent Jobs drawer, CliRunner output filter, and
 * ExperimentsList run search.
 *
 * Shape:
 *   - immediate `input` for responsive typing
 *   - `query` is the debounced lowercased trimmed value used for
 *     filter predicates
 *   - `inputRef` for the page-level "/" hotkey to focus
 *   - `onKeyDown` handler for two-step Escape (clear then blur)
 *   - optional `sessionKey` persists `input` to sessionStorage so a
 *     re-mount or tab-switch restores the query (right TTL for
 *     transient filter state). When omitted, no persistence.
 *
 * The predicate (what fields to match against) stays at the call
 * site - different surfaces match different shapes (cli_id vs run
 * name vs line text). The hook only owns the state machine.
 *
 * Why not a Context: the call sites are sibling routes/components,
 * not nested. A hook with explicit state is simpler.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type RefObject,
} from 'react';

export interface UseSearchFilterOptions {
  /** sessionStorage key for the raw input. When set, the hook
   * lazy-loads on mount and persists on every change. Omit for
   * non-persistent contexts (e.g. CliRunner output, where each
   * run's content differs). */
  sessionKey?: string;
  /** Debounce interval for query updates. Defaults to 120ms which
   * matches the existing call sites. */
  debounceMs?: number;
}

export interface UseSearchFilterResult {
  /** Raw input value, updated immediately on each keystroke. */
  input: string;
  /** Setter for the raw input. */
  setInput: (next: string) => void;
  /** Debounced lowercased trimmed value used for filter predicates.
   * Empty string means "no filter active". */
  query: string;
  /** Ref to attach to the search input element. */
  inputRef: RefObject<HTMLInputElement | null>;
  /** onKeyDown handler that implements the two-step Escape:
   *   - first press with non-empty input: clears the input and
   *     stops propagation so a global Esc handler doesn't fire
   *   - empty input: blurs the input so the next Esc reaches the
   *     parent's close-the-overlay handler
   */
  onKeyDown: (e: KeyboardEvent<HTMLInputElement>) => void;
  /** Imperative clear (e.g. for a "Clear filter" button). */
  clear: () => void;
}

export function useSearchFilter(
  options: UseSearchFilterOptions = {},
): UseSearchFilterResult {
  const { sessionKey, debounceMs = 120 } = options;

  // Lazy-load from sessionStorage on first mount when a key is
  // provided. The try/catch defends against private mode / quota
  // failures - we treat all failures as "start empty".
  const [input, setInputState] = useState<string>(() => {
    if (!sessionKey) return '';
    try {
      return window.sessionStorage.getItem(sessionKey) ?? '';
    } catch {
      return '';
    }
  });
  const [query, setQuery] = useState<string>(() => input.trim().toLowerCase());

  const inputRef = useRef<HTMLInputElement | null>(null);

  const setInput = useCallback((next: string) => {
    setInputState(next);
  }, []);

  // Debounced query update + sessionStorage mirror. Empty input
  // removes the storage entry rather than persisting "" (cleaner
  // storage; subsequent loads return '' from the missing key
  // anyway).
  useEffect(() => {
    const handle = window.setTimeout(() => {
      setQuery(input.trim().toLowerCase());
      if (sessionKey) {
        try {
          if (input) {
            window.sessionStorage.setItem(sessionKey, input);
          } else {
            window.sessionStorage.removeItem(sessionKey);
          }
        } catch {
          // best-effort persistence
        }
      }
    }, debounceMs);
    return () => window.clearTimeout(handle);
  }, [input, sessionKey, debounceMs]);

  const clear = useCallback(() => {
    setInputState('');
  }, []);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key !== 'Escape') return;
      if (input) {
        e.stopPropagation();
        e.preventDefault();
        setInputState('');
        return;
      }
      // Empty input + Esc: blur so the parent's Esc handler can
      // close the surrounding overlay.
      inputRef.current?.blur();
    },
    [input],
  );

  return { input, setInput, query, inputRef, onKeyDown, clear };
}
