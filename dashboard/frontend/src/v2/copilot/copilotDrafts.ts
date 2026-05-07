/**
 * Per-thread draft persistence for the CoPilot textarea.
 *
 * Customer-cared scenario: user types a long question into the
 * CoPilot dock, accidentally refreshes the tab (or closes/reopens
 * it), and the half-formed thought is gone. With this module, the
 * draft is restored automatically on mount.
 *
 * Drafts are keyed by ``threadId`` so a draft on thread A doesn't
 * leak into thread B if the user has multiple tabs open. The
 * special string ``__new__`` is used for the pre-thread state
 * (before the first message is sent and a server-side thread is
 * created) so a refresh during composition doesn't lose work.
 *
 * Same defensive shape as cliRunnerDrafts: an LRU cap so localStorage
 * doesn't grow unbounded over months of use, and try/catch around
 * every storage call so quota / private-mode failures stay silent.
 */

const KEY = 'evalyn.dashboard.copilotDrafts.v1';
const MAX_DRAFTS = 20;
const NEW_THREAD_KEY = '__new__';

interface DraftStore {
  /** order[0] is OLDEST, order[last] is NEWEST. */
  order: string[];
  drafts: Record<string, string>;
}

function safeStorage(): Storage | null {
  try {
    if (typeof window === 'undefined') return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

function loadStore(): DraftStore {
  const s = safeStorage();
  if (!s) return { order: [], drafts: {} };
  try {
    const raw = s.getItem(KEY);
    if (!raw) return { order: [], drafts: {} };
    const parsed = JSON.parse(raw) as unknown;
    if (
      parsed &&
      typeof parsed === 'object' &&
      Array.isArray((parsed as DraftStore).order) &&
      typeof (parsed as DraftStore).drafts === 'object'
    ) {
      return parsed as DraftStore;
    }
  } catch {
    // Poisoned storage; we'll overwrite on the next save.
  }
  return { order: [], drafts: {} };
}

function saveStore(store: DraftStore): void {
  const s = safeStorage();
  if (!s) return;
  try {
    s.setItem(KEY, JSON.stringify(store));
  } catch {
    // Quota exceeded / private mode - drafts are best-effort.
  }
}

function effectiveKey(threadId: string | null | undefined): string {
  return threadId && threadId.length > 0 ? threadId : NEW_THREAD_KEY;
}

/** Return the persisted draft for ``threadId`` or empty string if none. */
export function loadCoPilotDraft(threadId: string | null | undefined): string {
  const store = loadStore();
  return store.drafts[effectiveKey(threadId)] ?? '';
}

/** Save ``text`` as the draft for ``threadId``. Empty string deletes the
 * entry rather than persisting an empty value (cleaner localStorage
 * footprint for users who type and then delete). */
export function saveCoPilotDraft(
  threadId: string | null | undefined,
  text: string,
): void {
  const k = effectiveKey(threadId);
  const store = loadStore();
  if (text === '') {
    if (k in store.drafts) {
      delete store.drafts[k];
      const idx = store.order.indexOf(k);
      if (idx >= 0) store.order.splice(idx, 1);
      saveStore(store);
    }
    return;
  }
  store.drafts[k] = text;
  // LRU: move-to-end so old drafts get evicted first.
  const idx = store.order.indexOf(k);
  if (idx >= 0) store.order.splice(idx, 1);
  store.order.push(k);
  while (store.order.length > MAX_DRAFTS) {
    const oldest = store.order.shift();
    if (oldest) delete store.drafts[oldest];
  }
  saveStore(store);
}

/** Drop the draft for ``threadId`` (called after a successful submit). */
export function clearCoPilotDraft(
  threadId: string | null | undefined,
): void {
  saveCoPilotDraft(threadId, '');
}
