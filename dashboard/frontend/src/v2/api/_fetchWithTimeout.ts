/**
 * Shared fetch helper with a hard timeout via AbortController.
 *
 * Why: a wedged server (TCP accepted but never returns a response)
 * leaves bare ``fetch()`` hanging forever. The mutation buttons
 * (Run, Cancel, Re-run) flip `submitting=true` while awaiting and
 * have nothing else to do but wait, so the user gets a forever-
 * disabled control with no recovery short of reloading the tab.
 *
 * Surface: ``fetchWithTimeout(url, init, timeoutMs)`` is fetch with
 * an AbortController wired up. On timeout the helper throws a clear
 * ``Error`` whose message tells the user what to do; on any other
 * fetch error (network failure, etc.) the original error passes
 * through so the existing ``errorMessage()`` rewriter still applies.
 *
 * The AbortSignal is composed: callers that pass their own signal
 * via ``init.signal`` keep that wired up too (their abort still
 * works AND our timeout is layered on top).
 *
 * Used by: runCli (Run-button POST), cancelJob, restartJob.
 * Not used for GETs (those don't lock up a UI button on hang) or
 * for the CSRF refresh path (which itself uses no signal so the
 * outer fetch's timeout is sufficient).
 */

/** Wrap a fetch call with a timeout; throws a clear Error on
 * timeout. Other rejections pass through unchanged so callers'
 * existing error handling still works.
 *
 * If ``init.signal`` is already provided, both signals are
 * respected: the fetch aborts on whichever fires first.
 *
 * Returns the same Response shape as a normal fetch on success,
 * and the same RequestInit-supplied signal is observable via
 * ``init.signal`` if the caller needs it (we don't replace it).
 */
export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  timeoutMs: number,
  timeoutMessage: string,
): Promise<Response> {
  // Compose signals: the timeout is its own controller; the caller
  // may have passed an external signal. AbortSignal.any() exists on
  // modern engines but not Node 18 / older Safari, so we fall back
  // to manual composition.
  const timer = new AbortController();
  const timerId = setTimeout(() => timer.abort(), timeoutMs);
  const externalSignal = init?.signal ?? null;
  let composedSignal: AbortSignal = timer.signal;
  let externalListener: (() => void) | null = null;
  if (externalSignal != null) {
    if (externalSignal.aborted) {
      // External signal already aborted - mirror that immediately.
      timer.abort();
    } else {
      externalListener = () => timer.abort();
      externalSignal.addEventListener('abort', externalListener);
    }
    composedSignal = timer.signal;
  }
  try {
    return await fetch(input, { ...init, signal: composedSignal });
  } catch (e) {
    // AbortError from our timer -> rewrite to a clearer message.
    // External-signal aborts get the same message; the caller's
    // wrapper around fetchWithTimeout can re-classify if needed.
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error(timeoutMessage);
    }
    throw e;
  } finally {
    clearTimeout(timerId);
    if (externalListener && externalSignal) {
      externalSignal.removeEventListener('abort', externalListener);
    }
  }
}
