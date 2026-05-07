/**
 * Route chunk preloaders - co-located with V2App's lazy() calls so the
 * import paths stay in sync, but factored into a leaf module so AppShell
 * (a dep of V2App) can import them without a cycle.
 *
 * Each export is a () -> Promise that triggers the dynamic import for one
 * route. The browser dedupes concurrent imports of the same module, so
 * calling preloadX() on hover and then again on click is free - the
 * second call returns the same in-flight Promise. By the time the user
 * clicks a nav link they've already hovered for a few hundred ms, the
 * chunk is in the HTTP cache and the Suspense fallback never fires.
 */

export const preloadCoPilotThread = () => import('./routes/CoPilotThread');
export const preloadExperimentsList = () => import('./routes/ExperimentsList');
export const preloadRunDetail = () => import('./routes/RunDetail');
export const preloadFailureCluster = () => import('./routes/FailureCluster');
export const preloadDatasets = () => import('./routes/Datasets');
export const preloadDatasetDetail = () => import('./routes/DatasetDetail');
export const preloadMetrics = () => import('./routes/Metrics');
export const preloadReview = () => import('./routes/Review');
export const preloadReports = () => import('./routes/Reports');
export const preloadCommands = () => import('./routes/Commands');
export const preloadSettings = () => import('./routes/Settings');
export const preloadAnnotate = () => import('./routes/Annotate');
export const preloadAnnotateSession = () => import('./routes/AnnotateSession');
export const preloadNotFound = () => import('./routes/NotFound');

/**
 * Map a route path (top-level only) to its preload function. Used by
 * surfaces that hold a string path and want to warm the chunk on
 * hover - NotFound's "Did you mean?" suggestions, the AppShell nav
 * rail, the Cmd+K palette. Returns `undefined` for paths that don't
 * have a separate chunk (Home is eager) or don't match a known route.
 *
 * Top-level paths only - sub-routes like `/experiments/:runId` aren't
 * keyed here; their parents preload the same chunk anyway since the
 * sub-route ships in the parent's bundle.
 */
const PRELOAD_BY_PATH: Record<string, () => Promise<unknown>> = {
  '/copilot': preloadCoPilotThread,
  '/experiments': preloadExperimentsList,
  '/datasets': preloadDatasets,
  '/metrics': preloadMetrics,
  '/review': preloadReview,
  '/reports': preloadReports,
  '/commands': preloadCommands,
  '/settings': preloadSettings,
  '/annotate': preloadAnnotate,
};

export function preloadByPath(path: string): (() => Promise<unknown>) | undefined {
  return PRELOAD_BY_PATH[path];
}
