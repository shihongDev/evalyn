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
