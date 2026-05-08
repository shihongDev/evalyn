/**
 * Typed errors thrown by the API client.
 *
 * The API layer mostly speaks generic Error so callers can do `try`
 * + `console.warn`. But a small set of failure modes are recoverable
 * by the user (retry after a few seconds, switch providers, etc.) and
 * deserve a clearer surface than a flattened error string. This file
 * collects those typed errors so callers can do `instanceof` checks.
 */

/**
 * Thrown by `runCli` and `restartJob` when the server returns 503
 * because `JobManager.running_count >= max_concurrent`.
 *
 * The 503 body shape is `{ok: false, error, running, max_concurrent}`
 * (set by the backend tick that added the cap). We parse those fields
 * so a UI can render "queue full: 16 / 16 running, retry in a few
 * seconds" without re-parsing the message string.
 *
 * `Retry-After` is captured from the response header so a caller
 * that wants to schedule an automatic retry has the server's hint.
 */
export class CapacityError extends Error {
  readonly running: number;
  readonly maxConcurrent: number;
  readonly retryAfterSeconds: number;

  constructor(opts: {
    message: string;
    running: number;
    maxConcurrent: number;
    retryAfterSeconds: number;
  }) {
    super(opts.message);
    this.name = 'CapacityError';
    this.running = opts.running;
    this.maxConcurrent = opts.maxConcurrent;
    this.retryAfterSeconds = opts.retryAfterSeconds;
  }
}

/** Parse a 503 response body + Retry-After header into a CapacityError.
 * Returns null if the response is not the expected shape (caller should
 * fall back to a generic Error). Defensive parsing because a proxy
 * could return a 503 with arbitrary body. */
export async function maybeParseCapacityError(
  res: Response,
): Promise<CapacityError | null> {
  if (res.status !== 503) return null;
  let body: unknown;
  try {
    body = await res.clone().json();
  } catch {
    return null;
  }
  if (!body || typeof body !== 'object') return null;
  const b = body as Record<string, unknown>;
  if (typeof b.running !== 'number' || typeof b.max_concurrent !== 'number') {
    return null;
  }
  const retryHeader = res.headers.get('Retry-After');
  const retryAfterSeconds = retryHeader ? parseInt(retryHeader, 10) || 5 : 5;
  return new CapacityError({
    message: typeof b.error === 'string' ? b.error : 'job manager at capacity',
    running: b.running,
    maxConcurrent: b.max_concurrent,
    retryAfterSeconds,
  });
}

/**
 * Normalise an `unknown` thrown value to a user-friendly message
 * string. Repeated 17+ times across v2 routes as
 * `e instanceof Error ? e.message : String(e)` - extracted here
 * so the pattern lives in one place and any future enrichment
 * (e.g. CapacityError-specific phrasing, network error
 * formatting) only needs editing at one site.
 *
 * Bare `String(e)` on an Error renders as "Error: <message>",
 * which double-prefixes when the caller already says "Failed to
 * X:". Going through this helper keeps the visible string clean.
 *
 * Network-level fetch failures (offline, DNS fail, CSP block,
 * CORS pre-flight reject) reject with TypeError - the message
 * varies by browser ("Failed to fetch" in Chrome, "Load failed"
 * in Safari, "NetworkError when attempting to fetch resource"
 * in Firefox). All of these are jargon-y to a non-technical user.
 * When we see one, swap the raw text for a friendlier phrase so
 * every callsite (UpdatingChip, retry buttons, alert banners)
 * inherits the better wording for free.
 */
export function errorMessage(e: unknown): string {
  if (e instanceof TypeError && isLikelyNetworkFailure(e.message)) {
    return 'Network unreachable - check your connection and try again.';
  }
  if (e instanceof Error) return e.message;
  return String(e);
}

/** Heuristic for "this TypeError is a fetch network failure"
 * rather than e.g. a programmer-error type mismatch.
 *
 * Only TypeErrors thrown by fetch() during the network leg use
 * these specific messages; a TypeError from somewhere else (a
 * .toLowerCase on undefined, etc.) is much less likely to match
 * any of these strings. Match liberally on substrings so future
 * browser engine variants ("Network connection lost", etc.) tend
 * to be caught too. */
function isLikelyNetworkFailure(message: string): boolean {
  const m = message.toLowerCase();
  return (
    m.includes('failed to fetch') ||
    m.includes('load failed') ||
    m.includes('networkerror') ||
    m.includes('network error') ||
    m.includes('network request failed') ||
    m.includes('connection refused')
  );
}

/**
 * Format a CapacityError retry-after hint into a human-friendly
 * suffix. The server's Retry-After header is already parsed into
 * `CapacityError.retryAfterSeconds`; previously every callsite
 * just said "Try again in a few seconds" and dropped the precise
 * value. With this helper:
 *
 *   - <= 1s: "Try again now" (server hint says we can retry
 *     immediately; happens when the cap drops the moment the
 *     503 fires).
 *   - < 60s: "Try again in 30s" (precise seconds).
 *   - >= 60s: "Try again in 2m" (round down to minutes).
 *
 * Pure helper so callers can compose it into longer messages
 * (e.g. "Re-ran 3/5; queue full. Try again in 30s for the rest.").
 */
export function formatCapacityRetryHint(seconds: number): string {
  if (seconds <= 1) return 'Try again now';
  if (seconds < 60) return `Try again in ${seconds}s`;
  const m = Math.floor(seconds / 60);
  return `Try again in ${m}m`;
}
