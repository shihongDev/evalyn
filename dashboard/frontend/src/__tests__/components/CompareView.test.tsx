/**
 * Smoke tests for CompareView (P2 run-comparison view).
 *
 * Covers:
 *   - Renders the header with both run ids when data is loaded.
 *   - Renders one row per item from the compare payload.
 *   - Surfaces a 404 / network error state.
 *   - Same-run-id case still renders rows (deltas are 0).
 *   - Sort dropdown changes row order.
 *   - Malformed tab id (missing B) renders the inline error guard.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import CompareView from '../../views/CompareView';
import * as apiModule from '../../api';
import type { CompareResponse } from '../../api';

const samplePayload: CompareResponse = {
  run_a: {
    id: 'run-a',
    dataset: 'ds',
    pass: 1.0,
    metrics: { helpfulness: 1.0 },
    cost: null,
  },
  run_b: {
    id: 'run-b',
    dataset: 'ds',
    pass: 0.5,
    metrics: { helpfulness: 0.5 },
    cost: null,
  },
  items: [
    {
      input_hash: 'h1',
      input_preview: 'first question',
      a: { pass: true, metrics: { helpfulness: 1.0 }, output: 'A1' },
      b: { pass: true, metrics: { helpfulness: 1.0 }, output: 'B1' },
      delta: 0,
    },
    {
      input_hash: 'h2',
      input_preview: 'second question',
      a: { pass: true, metrics: { helpfulness: 1.0 }, output: 'A2' },
      b: { pass: false, metrics: { helpfulness: 0.0 }, output: 'B2' },
      delta: -1.0,
    },
    {
      input_hash: 'h3',
      input_preview: 'third question',
      a: { pass: false, metrics: { helpfulness: 0.0 }, output: 'A3' },
      b: { pass: true, metrics: { helpfulness: 1.0 }, output: 'B3' },
      delta: 1.0,
    },
  ],
};

let compareSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  compareSpy = vi
    .spyOn(apiModule.api, 'compare')
    .mockResolvedValue(samplePayload);
});

afterEach(() => {
  compareSpy.mockRestore();
});

describe('CompareView - smoke', () => {
  test('renders the header with both run ids and one row per item', async () => {
    render(<CompareView runA="run-a" runB="run-b" />);

    // Header eyebrow + title appear once data lands.
    await waitFor(() => {
      expect(screen.getByText(/Compare runs/i)).toBeInTheDocument();
    });
    // Run id prefixes shown in the header (8-char slice).
    expect(screen.getByText(/run-a/i)).toBeInTheDocument();
    expect(screen.getByText(/run-b/i)).toBeInTheDocument();

    // One <tr data-testid="compare-row"> per item in the payload.
    const rows = screen.getAllByTestId('compare-row');
    expect(rows).toHaveLength(samplePayload.items.length);

    // Pass-rate header line shows both percentages.
    expect(screen.getByText(/100\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/50\.0%/)).toBeInTheDocument();

    expect(compareSpy).toHaveBeenCalledWith('run-a', 'run-b');
  });

  test('renders an error state when the API rejects', async () => {
    compareSpy.mockRejectedValueOnce(new Error('404 Not Found'));
    render(<CompareView runA="run-a" runB="missing" />);
    await waitFor(() => {
      expect(screen.getByTestId('compare-error')).toBeInTheDocument();
    });
    expect(screen.getByTestId('compare-error')).toHaveTextContent(/404/);
  });

  test('shows a guard when both run ids are missing', () => {
    render(<CompareView runA="" runB="" />);
    expect(screen.getByText(/missing a run id/i)).toBeInTheDocument();
    // The fetch should NOT have fired in the malformed case.
    expect(compareSpy).not.toHaveBeenCalled();
  });

  test('sort dropdown reorders rows by regression first', async () => {
    render(<CompareView runA="run-a" runB="run-b" />);
    await waitFor(() => {
      expect(screen.getAllByTestId('compare-row')).toHaveLength(3);
    });
    const select = screen.getByLabelText(/Sort rows/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'regression' } });
    // After regression sort the most negative delta should lead.
    const rows = screen.getAllByTestId('compare-row');
    expect(rows[0]).toHaveTextContent('second question');
  });

  test('same-run-id case still renders rows', async () => {
    const samePayload: CompareResponse = {
      ...samplePayload,
      run_a: { ...samplePayload.run_a, id: 'run-a' },
      run_b: { ...samplePayload.run_b, id: 'run-a', pass: 1.0 },
      items: samplePayload.items.map((it) => ({ ...it, delta: 0 })),
    };
    compareSpy.mockResolvedValueOnce(samePayload);
    render(<CompareView runA="run-a" runB="run-a" />);
    await waitFor(() => {
      expect(screen.getAllByTestId('compare-row')).toHaveLength(3);
    });
    expect(screen.getByText(/Comparing a run against itself/i)).toBeInTheDocument();
  });

  test('empty results renders the empty placeholder', async () => {
    compareSpy.mockResolvedValueOnce({
      ...samplePayload,
      items: [],
    });
    render(<CompareView runA="run-a" runB="run-b" />);
    await waitFor(() => {
      expect(screen.getByTestId('compare-empty')).toBeInTheDocument();
    });
  });
});
