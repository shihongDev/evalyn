/**
 * contextChips - shared logic for the CoPilot composer's context chips.
 *
 * The chips ("+ this run", "@ dataset", "+ rubric") prepend a tag like
 * `[context: run abc-123]` to the composer draft. The agent runtime sees
 * these tags in the user message and can follow the convention.
 *
 * This is a pure UI affordance - the backend has no formal context-tag
 * protocol yet, but the LLM will read the tag verbatim and respond
 * accordingly. Used by both CoPilotDock and CoPilotThread.
 */

export type ContextKind = 'run' | 'cluster' | 'dataset' | 'rubric';

export interface ContextChipState {
  kind: ContextKind;
  label: string;
  /** When set, the chip is enabled; clicking prepends `[context: <kind> <id>]`. */
  id: string | null;
  /** Tooltip shown on the enabled chip (e.g. preview of what gets attached). */
  enabledTitle?: string;
}

/**
 * Derive which chips should be enabled for the current pathname.
 *
 * Route mapping:
 *   /experiments/:runId                       -> run
 *   /experiments/:runId/cluster/:clusterId    -> run + cluster
 *   /datasets/:name                           -> dataset
 *   /metrics                                  -> rubric (no id available from
 *                                                URL; left disabled until the
 *                                                page can plumb a selection in)
 *   anything else                             -> all disabled
 *
 * Returns chips in display order. Always returns the same 4 entries so
 * the composer layout stays stable across routes.
 */
export function deriveContextChips(pathname: string): ContextChipState[] {
  const run = matchRun(pathname);
  const cluster = matchCluster(pathname);
  const dataset = matchDataset(pathname);

  return [
    {
      kind: 'run',
      label: '+ this run',
      id: run,
      enabledTitle: run ? `Attach context: run ${run}` : undefined,
    },
    {
      kind: 'cluster',
      label: '+ this cluster',
      id: cluster,
      enabledTitle: cluster ? `Attach context: cluster ${cluster}` : undefined,
    },
    {
      kind: 'dataset',
      label: '＠ dataset',
      id: dataset,
      enabledTitle: dataset ? `Attach context: dataset ${dataset}` : undefined,
    },
    {
      kind: 'rubric',
      label: '+ rubric',
      // The rubric id isn't carried in the route today. Leave disabled until
      // the metrics page wires a selection through (e.g. ?rubric=...).
      id: null,
    },
  ];
}

/**
 * Prepend a `[context: <kind> <id>]` tag to a draft, idempotently.
 * Returns the original string if the same tag is already at the head.
 */
export function prependContextTag(
  draft: string,
  kind: ContextKind,
  id: string,
): string {
  const tag = `[context: ${kind} ${id}]`;
  if (draft.trimStart().startsWith(tag)) return draft;
  return `${tag}\n\n${draft}`;
}

function matchRun(pathname: string): string | null {
  // Matches /experiments/:runId and /experiments/:runId/cluster/...
  const m = pathname.match(/^\/experiments\/([^/]+)(?:\/|$)/);
  if (!m) return null;
  return m[1];
}

function matchCluster(pathname: string): string | null {
  const m = pathname.match(/^\/experiments\/[^/]+\/cluster\/([^/]+)/);
  if (!m) return null;
  return m[1];
}

function matchDataset(pathname: string): string | null {
  const m = pathname.match(/^\/datasets\/([^/]+)/);
  if (!m) return null;
  return decodeURIComponent(m[1]);
}
