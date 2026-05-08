/**
 * Demo loader - typed wrapper for POST /api/demo/load.
 *
 * The demo endpoint seeds the workspace with a populated example project so
 * a brand-new user can see the dashboard in its real, populated form before
 * they have any of their own runs. Self-heals stale CSRF tokens after a
 * server restart via the shared csrf.ts helpers.
 */

import { readCsrfToken, refreshCsrfToken } from './csrf';
import { fetchWithTimeout } from './_fetchWithTimeout';

// Hard timeout for the demo-load POST. The endpoint copies the
// bundled fixture into the workspace - a few file writes, normally
// <1s. 30s bounds a wedged-server hang. The Welcome card has the
// only Run button for this; on hang the user sees a spinner with
// no recovery without the timeout.
const DEMO_TIMEOUT_MS = 30_000;
const DEMO_TIMEOUT_MSG =
  `Server didn't respond within ${DEMO_TIMEOUT_MS / 1000}s. ` +
  `The dashboard may be wedged - try reloading.`;

export async function loadDemo(): Promise<{ loaded: boolean; project: string }> {
  const send = async (token: string | null): Promise<Response> => {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['X-Workbench-Token'] = token;
    return fetchWithTimeout(
      '/api/demo/load',
      { method: 'POST', headers, body: '{}' },
      DEMO_TIMEOUT_MS,
      DEMO_TIMEOUT_MSG,
    );
  };
  let res = await send(readCsrfToken());
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) res = await send(fresh);
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST /api/demo/load ${res.status}: ${body}`);
  }
  return (await res.json()) as { loaded: boolean; project: string };
}
