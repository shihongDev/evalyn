/**
 * V2App - root router for the v2 dashboard.
 *
 * Mounts every screen route. Replaces the old workbench App entirely.
 *
 * Routes besides Home are loaded via React.lazy(): the initial JS chunk
 * ships only Home + AppShell + the router itself, and other routes load
 * on first navigation. This trims the entry-point payload meaningfully
 * since most sessions hit 2-3 routes, not all 14. Vite's default code
 * splitting puts each lazy import into its own chunk; the static asset
 * dir already serves whatever filenames Vite emits.
 */

import { lazy, Suspense, useEffect } from 'react';
import {
  BrowserRouter,
  Route,
  Routes,
  useLocation,
  useNavigationType,
} from 'react-router-dom';
import Home from './routes/Home';
import { AppShell } from './AppShell';
import { ErrorBoundary } from './ErrorBoundary';
import { Spinner } from './ui';
import { E } from './tokens';

// Lazy-load every non-home route. Home stays eager because most sessions
// land there first and a Suspense flash on the entry page would feel
// worse than the small JS save. Preload functions live in
// `./routePreloads` so AppShell's nav-link onMouseEnter can warm a
// chunk without creating an import cycle (AppShell is a dep of V2App).
import * as preload from './routePreloads';

const CoPilotThread = lazy(preload.preloadCoPilotThread);
const ExperimentsList = lazy(preload.preloadExperimentsList);
const RunDetail = lazy(preload.preloadRunDetail);
const FailureCluster = lazy(preload.preloadFailureCluster);
const Datasets = lazy(preload.preloadDatasets);
const DatasetDetail = lazy(preload.preloadDatasetDetail);
const Metrics = lazy(preload.preloadMetrics);
const Review = lazy(preload.preloadReview);
const Reports = lazy(preload.preloadReports);
const Commands = lazy(preload.preloadCommands);
const Settings = lazy(preload.preloadSettings);
const Annotate = lazy(preload.preloadAnnotate);
const AnnotateSession = lazy(preload.preloadAnnotateSession);
const NotFound = lazy(preload.preloadNotFound);

/** Suspense fallback shown while a route chunk is downloading. We render
 * the AppShell chrome around a centered spinner so the navbar stays in
 * place across the transition - feels like a route change, not a full
 * page reload. The .eRouteFallback class delays the spinner fade-in by
 * 100 ms so cached / hover-preloaded chunks (which resolve in <50 ms)
 * never visibly flash a loading state. */
function RouteFallback() {
  return (
    <AppShell>
      <div
        className="eRouteFallback"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '120px 36px',
          color: E.text3,
          gap: 10,
          fontSize: 12,
          fontFamily: E.fMono,
        }}
        role="status"
        aria-live="polite"
      >
        <Spinner size={14} />
        Loading...
      </div>
    </AppShell>
  );
}

/** Reset the document scroll to the top whenever a forward navigation
 * lands on a new pathname. Without this, clicking a nav link from the
 * middle of a long page (e.g. /experiments scrolled to the bottom)
 * dropped the user mid-page on the next route, since BrowserRouter
 * only swaps DOM and does not reset scroll.
 *
 * Only PUSH navigations scroll. POP (back/forward) defers to the
 * browser's restoration. REPLACE is excluded because routes use it
 * to sync URL state without a real navigation - e.g. AnnotateSession
 * mirrors its cursor into ?item=, and yanking the page back to the
 * top on every keystroke would fight that screen's own scroll logic. */
function ScrollToTopOnNav() {
  const { pathname } = useLocation();
  const navType = useNavigationType();
  useEffect(() => {
    if (navType !== 'PUSH') return;
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, [pathname, navType]);
  return null;
}

/** Inner shell that lives below BrowserRouter so it can read useLocation
 * for the ErrorBoundary's resetKey. A render crash on route A doesn't
 * strand the user permanently - navigating to route B clears the
 * boundary so they can keep working. */
function RoutedShell() {
  const location = useLocation();
  return (
    <ErrorBoundary resetKey={location.pathname}>
      <ScrollToTopOnNav />
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/copilot" element={<CoPilotThread />} />
          <Route path="/copilot/:threadId" element={<CoPilotThread />} />
          <Route path="/experiments" element={<ExperimentsList />} />
          <Route path="/experiments/:runId" element={<RunDetail />} />
          <Route
            path="/experiments/:runId/cluster/:clusterId"
            element={<FailureCluster />}
          />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/datasets/:name" element={<DatasetDetail />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/review" element={<Review />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/commands" element={<Commands />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/annotate" element={<Annotate />} />
          <Route path="/annotate/:sessionId" element={<AnnotateSession />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}

export function V2App() {
  return (
    <BrowserRouter>
      <RoutedShell />
    </BrowserRouter>
  );
}
