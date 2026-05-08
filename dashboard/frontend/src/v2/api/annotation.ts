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
import { fetchWithTimeout } from './_fetchWithTimeout';

// Hard timeout for annotation mutations (create/verdict/finalize/
// abandon). Verdict is the high-frequency path - one POST per
// item per click - so a hung server here would block the user
// mid-session. 30s gives slack for finalize (jsonl merge across
// many items) while bounding hangs.
const ANNOTATION_TIMEOUT_MS = 30_000;
const ANNOTATION_TIMEOUT_MSG =
  `Server didn't respond within ${ANNOTATION_TIMEOUT_MS / 1000}s. ` +
  `The dashboard may be wedged - try reloading.`;

const BASE = '/api/v2/annotation';

function readCsrfToken(): string | null {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="workbench-token"]');
  return meta?.content ?? null;
}

/** Refresh the workbench-token meta tag from the server.
 *
 * The dashboard server generates a fresh CSRF token at startup. Pages
 * loaded before a server restart hold the old token and get 403s on
 * mutations. We detect that, refetch ``/`` to scrape the new token,
 * and patch the in-DOM meta so subsequent retries pick it up.
 */
async function refreshCsrfToken(): Promise<string | null> {
  try {
    const html = await (await fetch('/', { headers: { Accept: 'text/html' } })).text();
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

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GET ${path} ${res.status}: ${body}`);
  }
  return (await res.json()) as T;
}

async function mutating<T>(method: 'POST' | 'DELETE', path: string, body?: unknown): Promise<T> {
  const url = `${BASE}${path}`;
  const send = async (token: string | null): Promise<Response> => {
    const headers: Record<string, string> = {};
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    if (token) headers['X-Workbench-Token'] = token;
    return fetchWithTimeout(
      url,
      {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      },
      ANNOTATION_TIMEOUT_MS,
      ANNOTATION_TIMEOUT_MSG,
    );
  };
  let res = await send(readCsrfToken());
  // CSRF rejection: server token rotated since the page loaded. Refetch
  // the index, patch the meta, retry once. Don't loop.
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) {
      res = await send(fresh);
    }
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${method} ${path} ${res.status}: ${text}`);
  }
  return (await res.json()) as T;
}

const jpost = <T>(path: string, body: unknown): Promise<T> => mutating<T>('POST', path, body);
const jdelete = <T>(path: string): Promise<T> => mutating<T>('DELETE', path);

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
