/**
 * Placeholder routes - replaced by the screen agents.
 * Kept here so the router compiles before all screens land.
 */

import { AppShell } from '../AppShell';
import { Eyebrow } from '../ui';
import { E } from '../tokens';

export function makePlaceholder(title: string, sub: string) {
  return function Placeholder() {
    return (
      <AppShell>
        <div style={{ padding: '32px 36px' }}>
          <Eyebrow>{sub}</Eyebrow>
          <h1
            style={{
              fontFamily: E.fSerif,
              fontSize: 32,
              fontWeight: 400,
              margin: 0,
              color: E.text0,
              letterSpacing: '-0.015em',
            }}
          >
            {title}
          </h1>
          <div style={{ marginTop: 18, color: E.text3, fontSize: 13 }}>
            Pending screen-agent implementation.
          </div>
        </div>
      </AppShell>
    );
  };
}

export const ExperimentsListPlaceholder = makePlaceholder('Experiments', 'All evaluations');
export const RunDetailPlaceholder = makePlaceholder('Run detail', 'Reading a completed run');
export const FailureClusterPlaceholder = makePlaceholder('Failure cluster', 'Deep-dive on a regression');
export const DatasetsPlaceholder = makePlaceholder('Datasets', 'Evaluation inputs');
export const MetricsPlaceholder = makePlaceholder('Metrics & rubrics', 'How quality is graded');
export const ReviewPlaceholder = makePlaceholder('Human review', 'Where the judge wants a second opinion');
export const ReportsPlaceholder = makePlaceholder('Reports', 'Weekly summary');
export const CoPilotThreadPlaceholder = makePlaceholder('Co-pilot', 'Full thread');
