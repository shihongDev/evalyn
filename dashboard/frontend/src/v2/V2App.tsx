/**
 * V2App - root router for the v2 dashboard.
 * Mounts every screen route. Replaces the old workbench App entirely.
 */

import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Home from './routes/Home';
import CoPilotThread from './routes/CoPilotThread';
import ExperimentsList from './routes/ExperimentsList';
import RunDetail from './routes/RunDetail';
import FailureCluster from './routes/FailureCluster';
import Datasets from './routes/Datasets';
import DatasetDetail from './routes/DatasetDetail';
import Metrics from './routes/Metrics';
import Review from './routes/Review';
import Reports from './routes/Reports';
import Commands from './routes/Commands';
import Settings from './routes/Settings';
import Annotate from './routes/Annotate';
import AnnotateSession from './routes/AnnotateSession';
import NotFound from './routes/NotFound';

export function V2App() {
  return (
    <BrowserRouter>
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
    </BrowserRouter>
  );
}
