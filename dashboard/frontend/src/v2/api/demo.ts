/**
 * Demo loader - typed wrapper for POST /api/demo/load.
 *
 * The demo endpoint seeds the workspace with a populated example project so
 * a brand-new user can see the dashboard in its real, populated form before
 * they have any of their own runs. Reuses the workbench-token CSRF pattern
 * from the v2 client since /api/demo/load is a mutating endpoint.
 */

function csrfToken(): string | null {
  if (typeof document === 'undefined') return null;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="workbench-token"]');
  return meta?.content ?? null;
}

export async function loadDemo(): Promise<{ loaded: boolean; project: string }> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const tok = csrfToken();
  if (tok) headers['X-Workbench-Token'] = tok;
  const res = await fetch('/api/demo/load', { method: 'POST', headers, body: '{}' });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST /api/demo/load ${res.status}: ${body}`);
  }
  return (await res.json()) as { loaded: boolean; project: string };
}
