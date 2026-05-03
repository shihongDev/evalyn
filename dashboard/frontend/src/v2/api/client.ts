/**
 * v2 API client - typed fetchers for /api/v2/*.
 *
 * Reuses the CSRF + JSON conventions from the existing dashboard ./api.ts
 * (workbench-token meta + X-Workbench-Token header on mutations).
 */

import type {
  HomeSnapshot,
  ExperimentList,
  ExperimentDetail,
  ClusterDetail,
  DatasetList,
  RubricList,
  RubricDetail,
  ReviewQueue,
  ReviewVerdictPayload,
  WeeklyReport,
} from './types';
import { loadDemo as demoLoadHelper } from './demo';

const BASE = '/api/v2';

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

export const v2 = {
  home: (): Promise<HomeSnapshot> => jget('/home'),
  experiments: (): Promise<ExperimentList> => jget('/experiments'),
  experiment: (id: string): Promise<ExperimentDetail> =>
    jget(`/experiments/${encodeURIComponent(id)}`),
  cluster: (runId: string, clusterId: string): Promise<ClusterDetail> =>
    jget(
      `/experiments/${encodeURIComponent(runId)}/cluster/${encodeURIComponent(clusterId)}`,
    ),
  datasets: (): Promise<DatasetList> => jget('/datasets'),
  rubrics: (): Promise<RubricList> => jget('/rubrics'),
  rubric: (id: string): Promise<RubricDetail> =>
    jget(`/rubrics/${encodeURIComponent(id)}/calibration`),
  reviewQueue: (): Promise<ReviewQueue> => jget('/review/queue'),
  submitVerdict: (p: ReviewVerdictPayload): Promise<{ ok: true }> =>
    jpost('/review/verdict', p),
  weeklyReport: (): Promise<WeeklyReport> => jget('/reports/weekly'),
  demoLoad: (): Promise<{ loaded: boolean; project: string }> => demoLoadHelper(),
};
