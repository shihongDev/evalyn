/**
 * Demo loader - typed wrapper for POST /api/demo/load.
 *
 * The demo endpoint seeds the workspace with a populated example project so
 * a brand-new user can see the dashboard in its real, populated form before
 * they have any of their own runs. Self-heals stale CSRF tokens after a
 * server restart via the shared csrf.ts helpers.
 */

import { readCsrfToken, refreshCsrfToken } from './csrf';

export async function loadDemo(): Promise<{ loaded: boolean; project: string }> {
  const send = async (token: string | null): Promise<Response> => {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['X-Workbench-Token'] = token;
    return fetch('/api/demo/load', { method: 'POST', headers, body: '{}' });
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
