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
