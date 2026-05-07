/**
 * Settings API wrapper - typed fetchers for /api/settings/*.
 *
 * Mirrors the CSRF + JSON conventions used by src/v2/api/client.ts: read the
 * <meta name="workbench-token"> tag on mutations and forward it as the
 * X-Workbench-Token header. On 403 (stale token after a server restart),
 * refetch the index page to scrape the new token, patch the in-DOM meta,
 * and retry once. The endpoints predate the v2 namespace so they live
 * under /api/settings (no /api/v2 prefix).
 *
 * Backend reference: dashboard/evalyn_dashboard/api/settings.py.
 */

import { readCsrfToken, refreshCsrfToken } from './csrf';

const BASE = '/api/settings';

export interface ProviderState {
  is_set: boolean;
  model?: string | null;
  added_at?: string | null;
  updated_at?: string | null;
  base_url?: string | null;
}

export interface SettingsState {
  providers: Record<string, ProviderState>;
  active: string | null;
}

export interface ModelsResponse {
  models: string[];
}

export interface TestResult {
  ok: boolean;
  error?: string;
}

export interface SaveBody {
  api_key?: string;
  model?: string;
  base_url?: string;
}

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GET ${path} ${res.status}: ${body}`);
  }
  return (await res.json()) as T;
}

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const url = `${BASE}${path}`;
  const send = async (token: string | null): Promise<Response> => {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['X-Workbench-Token'] = token;
    return fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  };
  let res = await send(readCsrfToken());
  // Stale-token self-heal: server rotated its CSRF token (typically
  // because of a restart). Refetch the index, patch the meta, retry
  // once. Don't loop - if the second 403 fires, something else is wrong.
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) {
      res = await send(fresh);
    }
  }
  // The /test/{provider} endpoint deliberately returns 400 with a JSON body
  // when a connection probe fails. Surface that body to the caller rather
  // than throwing, so the UI can render the error message inline.
  if (!res.ok) {
    let detail: string;
    try {
      const j = (await res.json()) as { error?: string; detail?: string; ok?: boolean };
      detail = j.error ?? j.detail ?? `${res.status}`;
      // Special-case the test endpoint shape so callers can keep their typing.
      if (path.startsWith('/test/')) {
        return { ok: false, error: detail } as unknown as T;
      }
    } catch {
      detail = await res.text();
    }
    throw new Error(`POST ${path} ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const settingsApi = {
  list: (): Promise<SettingsState> => jget(''),
  models: (provider: string): Promise<ModelsResponse> =>
    jget(`/models/${encodeURIComponent(provider)}`),
  save: (provider: string, body: SaveBody): Promise<ProviderState> =>
    jpost(`/${encodeURIComponent(provider)}`, body),
  test: (provider: string): Promise<TestResult> =>
    jpost(`/test/${encodeURIComponent(provider)}`, {}),
  setActive: (provider: string): Promise<{ ok: boolean }> =>
    jpost('/active', { provider }),
};
