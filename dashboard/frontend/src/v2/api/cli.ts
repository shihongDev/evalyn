/**
 * Typed wrapper for the CLI catalog endpoint (`GET /api/cli`).
 *
 * The shape mirrors the introspector at
 * `dashboard/evalyn_dashboard/introspect.py::catalog_to_payload` -
 * keep this file in lockstep with that module.
 *
 * Notes on the contract:
 * - The server returns `blurb` (and the introspector strips empty/null
 *   `description` fields). We surface `blurb` directly and accept a
 *   `description` alias for forward-compat.
 * - `kind` includes `path` and `long-text` in addition to the basic
 *   string/number/bool/select/multiselect set.
 * - Optional fields (`options`, `default`, `range`, `unit`) are dropped
 *   server-side when null, so callers should treat them as `undefined`.
 */

import { readCsrfToken, refreshCsrfToken } from './csrf';

export type CliParamKind =
  | 'string'
  | 'long-text'
  | 'number'
  | 'bool'
  | 'select'
  | 'multiselect'
  | 'path';

export interface CliNumberRange {
  min: number;
  max: number;
  step?: number;
}

export interface CliParam {
  name: string;
  kind: CliParamKind;
  default?: unknown;
  options?: string[];
  help?: string | null;
  required?: boolean;
  advanced?: boolean;
  essential?: boolean;
  range?: CliNumberRange;
  unit?: string;
}

export interface CliSchema {
  id: string;
  /** Display name, usually identical to `id`. */
  name?: string;
  group?: string;
  blurb?: string;
  /** Forward-compat alias; the current backend only sets `blurb`. */
  description?: string;
  params: CliParam[];
}

export async function listCli(): Promise<CliSchema[]> {
  const res = await fetch('/api/cli', { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    throw new Error(`GET /api/cli ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as CliSchema[];
}

/** Return the human-friendly group label for a command, falling back to "Other". */
export function commandGroup(cmd: CliSchema): string {
  return (cmd.group && cmd.group.trim()) || 'Other';
}

/** Return the best summary string for a command (blurb wins, then description). */
export function commandSummary(cmd: CliSchema): string {
  return (cmd.blurb && cmd.blurb.trim()) || (cmd.description && cmd.description.trim()) || '';
}

/**
 * Render an argv preview the user can sanity-check ("evalyn cli-id --flag value
 * --other ..."). Skips empty / null / undefined values, drops false booleans
 * (the backend ignores them anyway), unfolds array values into multiple
 * positional args after the flag. Used by CliRunner's preview row and the
 * RecentJobsDrawer row tooltip so similar runs are easier to disambiguate.
 */
export function previewCommand(
  cliId: string,
  values: Record<string, unknown>,
): string {
  const parts: string[] = ['evalyn', cliId];
  for (const [name, value] of Object.entries(values)) {
    if (value === undefined || value === null || value === '') continue;
    const flag = `--${name.replace(/_/g, '-')}`;
    if (typeof value === 'boolean') {
      if (value) parts.push(flag);
      continue;
    }
    if (Array.isArray(value)) {
      if (value.length === 0) continue;
      parts.push(flag);
      for (const v of value) parts.push(String(v));
      continue;
    }
    parts.push(flag, String(value));
  }
  return parts.join(' ');
}

/**
 * POST /api/cli/run - spawns ``evalyn <cli_id> <args...>`` on the server and
 * returns a job_id the caller can stream from. Throws on non-2xx with the
 * server's error body included so callers can surface it to the user.
 *
 * Self-heals stale CSRF tokens after a server restart (refetch index,
 * scrape new token, retry once) via the shared csrf.ts helpers.
 */
export async function runCli(
  cliId: string,
  args: Record<string, unknown>,
): Promise<{ job_id: string }> {
  const send = async (token: string | null): Promise<Response> => {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['X-Workbench-Token'] = token;
    return fetch('/api/cli/run', {
      method: 'POST',
      headers,
      body: JSON.stringify({ cli_id: cliId, args }),
    });
  };
  let res = await send(readCsrfToken());
  if (res.status === 403) {
    const fresh = await refreshCsrfToken();
    if (fresh) res = await send(fresh);
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST /api/cli/run ${res.status}: ${body}`);
  }
  return (await res.json()) as { job_id: string };
}
