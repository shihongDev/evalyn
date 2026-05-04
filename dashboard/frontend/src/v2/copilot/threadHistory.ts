/**
 * Client-side thread history for the v2 co-pilot.
 *
 * The backend exposes `POST /api/threads/save` (markdown export) but no
 * thread-list endpoint. We persist a small directory in localStorage so the
 * full-thread route can show the user's recent conversations across reloads.
 *
 * Storage shape: a JSON array of ThreadIndexEntry under `evalyn:v2:threads`.
 * No backwards-compat shims - on parse failure we return [] silently.
 */

const STORAGE_KEY = 'evalyn:v2:threads';

export interface ThreadIndexEntry {
  id: string;
  title: string;
  turn_count: number;
  created_at_iso: string;
}

function safeStorage(): Storage | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function loadThreadIndex(): ThreadIndexEntry[] {
  const store = safeStorage();
  if (!store) return [];
  const raw = store.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidEntry);
  } catch {
    return [];
  }
}

export function saveThreadIndex(entries: ThreadIndexEntry[]): void {
  const store = safeStorage();
  if (!store) return;
  try {
    store.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    /* quota exceeded or denied - history is best-effort */
  }
}

export function upsertThread(entry: ThreadIndexEntry): void {
  const current = loadThreadIndex();
  const idx = current.findIndex((e) => e.id === entry.id);
  if (idx >= 0) {
    current[idx] = { ...current[idx], ...entry };
  } else {
    current.push(entry);
  }
  saveThreadIndex(current);
}

export function removeThread(id: string): void {
  const current = loadThreadIndex();
  const next = current.filter((e) => e.id !== id);
  saveThreadIndex(next);
}

function isValidEntry(value: unknown): value is ThreadIndexEntry {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === 'string' &&
    typeof v.title === 'string' &&
    typeof v.turn_count === 'number' &&
    typeof v.created_at_iso === 'string'
  );
}
