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

import { lazy, Suspense } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Home from './routes/Home';
import { AppShell } from './AppShell';
import { Spinner } from './ui';
import { E } from './tokens';

// Lazy-load every non-home route. Home stays eager because most sessions
// land there first and a Suspense flash on the entry page would feel
// worse than the small JS save.
const CoPilotThread = lazy(() => import('./routes/CoPilotThread'));
const ExperimentsList = lazy(() => import('./routes/ExperimentsList'));
const RunDetail = lazy(() => import('./routes/RunDetail'));
const FailureCluster = lazy(() => import('./routes/FailureCluster'));
const Datasets = lazy(() => import('./routes/Datasets'));
const DatasetDetail = lazy(() => import('./routes/DatasetDetail'));
const Metrics = lazy(() => import('./routes/Metrics'));
const Review = lazy(() => import('./routes/Review'));
const Reports = lazy(() => import('./routes/Reports'));
const Commands = lazy(() => import('./routes/Commands'));
const Settings = lazy(() => import('./routes/Settings'));
const Annotate = lazy(() => import('./routes/Annotate'));
const AnnotateSession = lazy(() => import('./routes/AnnotateSession'));
const NotFound = lazy(() => import('./routes/NotFound'));

/** Suspense fallback shown while a route chunk is downloading. We render
 * the AppShell chrome around a centered spinner so the navbar stays in
 * place across the transition - feels like a route change, not a full
 * page reload. */
function RouteFallback() {
  return (
    <AppShell>
      <div
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

export function V2App() {
  return (
    <BrowserRouter>
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
    </BrowserRouter>
  );
}
