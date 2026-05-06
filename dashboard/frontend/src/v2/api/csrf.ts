/**
 * CSRF / workbench-token helpers shared across all mutating API calls.
 *
 * The dashboard server generates a fresh CSRF token at startup. Browser
 * tabs loaded before a server restart hold the old token in their
 * `<meta name="workbench-token">` tag and get 403s on mutations until
 * the page reloads. This module provides:
 *
 *   - `readCsrfToken()`: pull the current token from the meta tag.
 *   - `refreshCsrfToken()`: refetch `/`, scrape the new token, patch
 *     the in-DOM meta so future requests pick it up.
 *
 * API modules wrap their mutating fetches in a "send → if 403, refresh,
 * retry once" pattern. Don't loop - if the second attempt also 403s,
 * the user has a bigger problem than a stale token.
 *
 * Originally inline in `annotation.ts`; extracted here so settings,
 * cli, jobs, and any future mutating endpoint inherit the self-heal
 * behavior without copy-pasting.
 */

export function readCsrfToken(): string | null {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="workbench-token"]');
  return meta?.content ?? null;
}

export async function refreshCsrfToken(): Promise<string | null> {
  try {
    const res = await fetch('/', { headers: { Accept: 'text/html' } });
    const html = await res.text();
    const match = html.match(/name="workbench-token"\s+content="([^"]+)"/);
    if (!match) return null;
    const fresh = match[1];
    if (typeof document !== 'undefined') {
      const meta = document.querySelector<HTMLMetaElement>('meta[name="workbench-token"]');
      if (meta) meta.content = fresh;
    }
    return fresh;
  } catch {
    return null;
  }
}
