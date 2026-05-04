/**
 * Typed client for /api/v2/annotation/*.
 *
 * Mirrors the CSRF + JSON conventions in `client.ts`. Reads the
 * workbench-token meta tag and forwards it as `X-Workbench-Token`
 * on every mutating call.
 */

import type {
  AnnotationCreatePayload,
  AnnotationItemsResponse,
  AnnotationSessionList,
  AnnotationSessionMeta,
  AnnotationVerdictPayload,
  AnnotationVerdictResponse,
} from './types';

const BASE = '/api/v2/annotation';

function csrfToken(): string | null {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="workbench-token"]');
  return meta?.content ?? null;
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
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const tok = csrfToken();
  if (tok) headers['X-Workbench-Token'] = tok;
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} ${res.status}: ${text}`);
  }
  return (await res.json()) as T;
}

async function jdelete<T>(path: string): Promise<T> {
  const headers: Record<string, string> = {};
  const tok = csrfToken();
  if (tok) headers['X-Workbench-Token'] = tok;
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE', headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`DELETE ${path} ${res.status}: ${text}`);
  }
  return (await res.json()) as T;
}

export const annotationApi = {
  listSessions(): Promise<AnnotationSessionList> {
    return jget('/sessions');
  },
  getSession(id: string): Promise<AnnotationSessionMeta> {
    return jget(`/sessions/${encodeURIComponent(id)}`);
  },
  getItems(
    id: string,
    opts: { offset?: number; limit?: number } = {},
  ): Promise<AnnotationItemsResponse> {
    const params = new URLSearchParams();
    if (opts.offset != null) params.set('offset', String(opts.offset));
    if (opts.limit != null) params.set('limit', String(opts.limit));
    const qs = params.toString();
    return jget(`/sessions/${encodeURIComponent(id)}/items${qs ? `?${qs}` : ''}`);
  },
  createSession(payload: AnnotationCreatePayload): Promise<AnnotationSessionMeta> {
    return jpost('/sessions', payload);
  },
  postVerdict(
    id: string,
    payload: AnnotationVerdictPayload,
  ): Promise<AnnotationVerdictResponse> {
    return jpost(`/sessions/${encodeURIComponent(id)}/verdict`, payload);
  },
  finalize(id: string): Promise<{ ok: true; merged: number }> {
    return jpost(`/sessions/${encodeURIComponent(id)}/finalize`, {});
  },
  abandon(id: string): Promise<{ ok: true; status: string }> {
    return jdelete(`/sessions/${encodeURIComponent(id)}`);
  },
};
