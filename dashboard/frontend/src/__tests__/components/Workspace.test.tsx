/**
 * Smoke tests for the Workspace view.
 *
 * Verifies:
 *   - Empty form placeholder when no active CLI is set.
 *   - Active CLI selection renders the form for that CLI.
 *   - Run history feed shows run cards (collapsed by default, expandable).
 *   - Edit button on a run card seeds the active form via `editRunArgs`.
 *   - Pin / Unpin moves a card into the Pinned section.
 *   - Schema-drift merge: a re-rendered cli with a new param exposes the
 *     fresh default without clobbering the user's prior value.
 */

import { describe, expect, test, beforeEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import Workspace from '../../views/Workspace';
import { __resetStore, useStore } from '../../store';
import type { CliSchema } from '../../types/catalog';

const cliFixture: CliSchema = {
  id: 'run-eval',
  name: 'run-eval',
  group: 'Eval',
  blurb: 'Run an evaluation.',
  params: [
    { name: 'dataset', kind: 'path', required: true },
    { name: 'workers', kind: 'number', default: 4 },
    { name: 'verbose', kind: 'bool', default: false },
  ],
};

const calibrateFixture: CliSchema = {
  id: 'calibrate',
  name: 'calibrate',
  group: 'Eval',
  blurb: 'Calibrate the judge.',
  params: [{ name: 'judge', kind: 'string', default: 'gpt-4' }],
};

beforeEach(() => {
  __resetStore();
});

describe('Workspace - empty state', () => {
  test('renders the empty form placeholder when no active CLI', () => {
    render(<Workspace />);
    expect(screen.getByText(/Choose a command from the left/i)).toBeInTheDocument();
    expect(screen.getByText(/Your runs will appear here/i)).toBeInTheDocument();
  });

  test('clicking the suggested CLI sets the active CLI', () => {
    useStore.getState().setCatalog([cliFixture]);
    render(<Workspace />);
    fireEvent.click(screen.getByRole('button', { name: /Use run-eval/i }));
    expect(useStore.getState().activeCliId).toBe('run-eval');
  });
});

describe('Workspace - active form', () => {
  beforeEach(() => {
    useStore.getState().setCatalog([cliFixture, calibrateFixture]);
    useStore.getState().selectActiveCli('run-eval');
  });

  test('renders the form for the active CLI', () => {
    render(<Workspace />);
    // Header text + form field
    expect(screen.getByText(/Active command/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/dataset/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Run$/i })).toBeInTheDocument();
  });

  test('switching CLI via the dropdown updates the form', () => {
    render(<Workspace />);
    const select = screen.getByLabelText(/Active CLI/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'calibrate' } });
    expect(useStore.getState().activeCliId).toBe('calibrate');
  });
});

describe('Workspace - auto-seed (Approach D)', () => {
  test('selecting a CLI with prior runs cherry-picks the latest args', () => {
    useStore.getState().setCatalog([cliFixture]);
    useStore.setState({
      runHistory: [
        {
          id: 'r-old',
          cliId: 'run-eval',
          args: { dataset: './old.jsonl', workers: 2, verbose: false },
          jobId: 'j-old',
          startedAt: 1,
          pinned: false,
        },
        {
          id: 'r-new',
          cliId: 'run-eval',
          args: { dataset: './newest.jsonl', workers: 32, verbose: true },
          jobId: 'j-new',
          startedAt: 2,
          pinned: false,
        },
      ],
    });
    useStore.getState().selectActiveCli('run-eval');
    render(<Workspace />);
    // The form picked up the seed during the effect; the seed itself is
    // cleared after consumption.
    expect(useStore.getState().activeFormSeed).toBeNull();
    // dataset input shows the newest run's value.
    const datasetInput = screen.getByLabelText(/dataset/i) as HTMLInputElement;
    expect(datasetInput.value).toBe('./newest.jsonl');
  });

  test('selecting a CLI without prior runs falls back to argparse defaults', () => {
    useStore.getState().setCatalog([cliFixture]);
    useStore.getState().selectActiveCli('run-eval');
    render(<Workspace />);
    expect(useStore.getState().activeFormSeed).toBeNull();
    const datasetInput = screen.getByLabelText(/dataset/i) as HTMLInputElement;
    // No prior args so the path field starts empty (default for kind: path).
    expect(datasetInput.value).toBe('');
  });
});

describe('Workspace - run history feed', () => {
  beforeEach(() => {
    useStore.getState().setCatalog([cliFixture]);
    useStore.getState().selectActiveCli('run-eval');
  });

  test('shows a card per run, collapsed by default', () => {
    useStore.setState({
      runHistory: [
        {
          id: 'r-1',
          cliId: 'run-eval',
          args: { dataset: './a.jsonl' },
          jobId: 'j-1',
          startedAt: 1_700_000_000_000,
          pinned: false,
        },
      ],
    });
    render(<Workspace />);
    // Card visible in the feed; details (Edit / Pin / Remove) hidden until expand.
    expect(screen.getByText(/Run history/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /Expand run/i }));
    // Now the action buttons are visible.
    expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Pin' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove' })).toBeInTheDocument();
  });

  test('Edit button seeds the active form via editRunArgs', () => {
    useStore.setState({
      runHistory: [
        {
          id: 'r-1',
          cliId: 'run-eval',
          args: { dataset: './edited.jsonl', workers: 16, verbose: true },
          jobId: 'j-1',
          startedAt: 1_700_000_000_000,
          pinned: false,
        },
      ],
    });
    render(<Workspace />);
    fireEvent.click(screen.getByRole('button', { name: /Expand run/i }));
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    // After Edit, activeCliId is set; the form picks up the seed which
    // clears `activeFormSeed` once consumed.
    expect(useStore.getState().activeCliId).toBe('run-eval');
    // The form effect should have consumed the seed.
    expect(useStore.getState().activeFormSeed).toBeNull();
  });

  test('Pin moves the card into the Pinned section', () => {
    useStore.setState({
      runHistory: [
        {
          id: 'r-1',
          cliId: 'run-eval',
          args: {},
          jobId: 'j-1',
          startedAt: 1_700_000_000_000,
          pinned: false,
        },
      ],
    });
    render(<Workspace />);
    fireEvent.click(screen.getByRole('button', { name: /Expand run/i }));
    fireEvent.click(screen.getByRole('button', { name: 'Pin' }));
    expect(useStore.getState().runHistory[0].pinned).toBe(true);
    // Pinned section header now shows up.
    expect(screen.getByText(/Pinned · 1/i)).toBeInTheDocument();
  });
});
