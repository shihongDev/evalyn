/**
 * Per-CLI form drafts persisted to localStorage.
 *
 * Customer-cared scenario: user opens CliRunner for ``run-eval``, fills
 * in 5 fields, gets pulled into Slack for 10 minutes, comes back and
 * finds an empty form. With this module, returning restores the
 * partial draft so the user picks up exactly where they left off.
 *
 * Drafts are keyed by ``cli_id`` and saved on every input change
 * (debounced by the caller). They are cleared on successful submit
 * (the user finished using these values) and on explicit "Reset"
 * actions if/when those land. A fixed cap on the number of stored
 * drafts (LRU) prevents the localStorage entry from growing forever
 * for users who experiment with many commands once.
 *
 * Drafts are deliberately user-local (localStorage, not server-side):
 * they're transient editing state, not run history, and pinning them
 * to one browser tab matches the user's mental model.
 */

const KEY = 'evalyn.dashboard.cliRunnerDrafts.v1';
const MAX_DRAFTS = 30;

/** Persisted shape: a tiny LRU keyed by cli_id. ``order`` keeps the
 * least-recently-used at index 0 so we can pop from the front when
 * we hit the cap. */
interface DraftStore {
  /** order[0] is the OLDEST cli_id, order[last] is the NEWEST. */
  order: string[];
  drafts: Record<string, Record<string, unknown>>;
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
    // localStorage poisoned. Fall through to fresh store; we'll
    // overwrite on the next save.
  }
  return { order: [], drafts: {} };
}

function saveStore(store: DraftStore): void {
  const s = safeStorage();
  if (!s) return;
  try {
    s.setItem(KEY, JSON.stringify(store));
  } catch {
    // Quota exceeded or private mode; drafts are best-effort.
  }
}

/** Return the persisted draft for ``cliId`` or ``null`` if none exists. */
export function loadDraft(cliId: string): Record<string, unknown> | null {
  if (!cliId) return null;
  const store = loadStore();
  const draft = store.drafts[cliId];
  return draft ?? null;
}

/** Save ``values`` as the current draft for ``cliId``.
 *
 * Bumps ``cliId`` to the most-recently-used end of the LRU. Evicts the
 * oldest entry if storing this would exceed ``MAX_DRAFTS``. An empty
 * draft (no truthy values) is stored as-is; we don't second-guess the
 * user's clearing-the-form intent. */
export function saveDraft(cliId: string, values: Record<string, unknown>): void {
  if (!cliId) return;
  const store = loadStore();
  store.drafts[cliId] = values;
  // Move-to-end LRU: drop existing, then push.
  const idx = store.order.indexOf(cliId);
  if (idx >= 0) store.order.splice(idx, 1);
  store.order.push(cliId);
  while (store.order.length > MAX_DRAFTS) {
    const oldest = store.order.shift();
    if (oldest) delete store.drafts[oldest];
  }
  saveStore(store);
}

/** Drop the draft for ``cliId``. Called after a successful submit so
 * the next open of this command starts from a clean form. */
export function clearDraft(cliId: string): void {
  if (!cliId) return;
  const store = loadStore();
  if (!(cliId in store.drafts)) return;
  delete store.drafts[cliId];
  const idx = store.order.indexOf(cliId);
  if (idx >= 0) store.order.splice(idx, 1);
  saveStore(store);
}
