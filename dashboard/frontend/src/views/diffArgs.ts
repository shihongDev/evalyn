/**
 * Param-diff helper shared by `Workspace.tsx` (active-form chip) and
 * `RunCard.tsx` (one-line run summary). Returns the list of arg keys that
 * differ between two value maps; arrays are compared element-wise.
 */

export interface DiffEntry {
  name: string;
  before: unknown;
  after: unknown;
}

export const diffArgs = (
  prev: Record<string, unknown> | null | undefined,
  curr: Record<string, unknown>,
): DiffEntry[] => {
  if (!prev) return [];
  const out: DiffEntry[] = [];
  const keys = new Set([...Object.keys(prev), ...Object.keys(curr)]);
  for (const k of keys) {
    const a = prev[k];
    const b = curr[k];
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length || a.some((v, i) => v !== b[i])) {
        out.push({ name: k, before: a, after: b });
      }
      continue;
    }
    if (a !== b) {
      out.push({ name: k, before: a, after: b });
    }
  }
  return out;
};

export const fmtArgVal = (v: unknown): string => {
  if (v === undefined || v === null || v === '') return '∅';
  if (Array.isArray(v)) return v.length === 0 ? '∅' : `[${v.join(',')}]`;
  return String(v);
};
