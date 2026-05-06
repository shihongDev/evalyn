/**
 * v2 API client - typed fetchers for /api/v2/*.
 *
 * Mutations attach the X-Workbench-Token header from the
 * <meta name="workbench-token"> tag. On 403 (stale token after a
 * server restart), refetch the index, scrape the new token, retry
 * once via the shared csrf.ts helpers.
 */

import { readCsrfToken, refreshCsrfToken } from './csrf';
import type {
  HomeSnapshot,
  ExperimentList,
  ExperimentDetail,
  ExperimentItemsResponse,
  ExperimentItemsFilter,
  ExperimentItemsSort,
  ClusterDetail,
  DatasetList,
  DatasetDetail,
  RubricList,
  RubricDetail,
  ReviewQueue,
  ReviewVerdictPayload,
  WeeklyReport,
} from './types';
import { loadDemo as demoLoadHelper } from './demo';

const BASE = '/api/v2';

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
    if (fresh) res = await send(fresh);
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} ${res.status}: ${text}`);
  }
  return (await res.json()) as T;
}

export const v2 = {
  home: (): Promise<HomeSnapshot> => jget('/home'),
  experiments: (): Promise<ExperimentList> => jget('/experiments'),
  experiment: (id: string): Promise<ExperimentDetail> =>
    jget(`/experiments/${encodeURIComponent(id)}`),
  experimentItems: (
    runId: string,
    opts: {
      offset?: number;
      limit?: number;
      filter?: ExperimentItemsFilter;
      sort?: ExperimentItemsSort;
    } = {},
  ): Promise<ExperimentItemsResponse> => {
    const params = new URLSearchParams();
    if (opts.offset != null) params.set('offset', String(opts.offset));
    if (opts.limit != null) params.set('limit', String(opts.limit));
    if (opts.filter) params.set('filter', opts.filter);
    if (opts.sort) params.set('sort', opts.sort);
    const qs = params.toString();
    return jget(
      `/experiments/${encodeURIComponent(runId)}/items${qs ? `?${qs}` : ''}`,
    );
  },
  cluster: (runId: string, clusterId: string): Promise<ClusterDetail> =>
    jget(
      `/experiments/${encodeURIComponent(runId)}/cluster/${encodeURIComponent(clusterId)}`,
    ),
  datasets: (): Promise<DatasetList> => jget('/datasets'),
  dataset: (name: string): Promise<DatasetDetail> =>
    jget(`/datasets/${encodeURIComponent(name)}`),
  rubrics: (): Promise<RubricList> => jget('/rubrics'),
  rubric: (id: string): Promise<RubricDetail> =>
    jget(`/rubrics/${encodeURIComponent(id)}/calibration`),
  reviewQueue: (): Promise<ReviewQueue> => jget('/review/queue'),
  submitVerdict: (p: ReviewVerdictPayload): Promise<{ ok: true }> =>
    jpost('/review/verdict', p),
  weeklyReport: (): Promise<WeeklyReport> => jget('/reports/weekly'),
  demoLoad: (): Promise<{ loaded: boolean; project: string }> => demoLoadHelper(),
};
