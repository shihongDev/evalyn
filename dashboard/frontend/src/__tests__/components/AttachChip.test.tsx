/**
 * Tests for AttachChip + click-to-attach behaviour on RunCard.
 *
 * Covers:
 *   - AttachChip renders kind + label and fires onRemove on click.
 *   - RunCard "+ chat" button calls attachToChatInput with the right payload.
 *   - serializeChatAttachments shape sanity (run / metric / file kinds).
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import AttachChip from '../../components/AttachChip';
import RunCard from '../../views/RunCard';
import {
  __resetStore,
  serializeChatAttachments,
  useStore,
  type ChatAttachment,
  type RunRecord,
} from '../../store';

beforeEach(() => {
  __resetStore();
});

describe('AttachChip', () => {
  test('renders kind + label and fires onRemove', () => {
    const attachment: ChatAttachment = {
      id: 'run:abc',
      kind: 'run',
      label: 'run-2026-04-30-ghi',
      ref: 'abc',
    };
    const onRemove = vi.fn();
    render(<AttachChip attachment={attachment} onRemove={onRemove} />);

    // Label is rendered.
    expect(screen.getByText('run-2026-04-30-ghi')).toBeInTheDocument();
    // Container exists with the right testid.
    expect(screen.getByTestId('attach-chip-run:abc')).toBeInTheDocument();

    // Click the × button → onRemove fires.
    fireEvent.click(screen.getByTestId('attach-chip-remove-run:abc'));
    expect(onRemove).toHaveBeenCalledTimes(1);
  });
});

describe('RunCard - attach button', () => {
  test('clicking "+ chat" calls attachToChatInput with the run payload', () => {
    const run: RunRecord = {
      id: 'run-xyz',
      cliId: 'run-eval',
      args: { dataset: './data.jsonl' },
      jobId: 'j-1',
      startedAt: Date.now(),
      pinned: false,
    };
    render(<RunCard run={run} prev={null} />);
    expect(useStore.getState().chatAttachments).toEqual([]);

    fireEvent.click(screen.getByTestId('attach-run-run-xyz'));
    expect(useStore.getState().chatAttachments).toEqual([
      {
        id: 'run:run-xyz',
        kind: 'run',
        label: 'run-xyz',
        ref: 'run-xyz',
      },
    ]);

    // Clicking again is a silent no-op (dedupe by id).
    fireEvent.click(screen.getByTestId('attach-run-run-xyz'));
    expect(useStore.getState().chatAttachments).toHaveLength(1);
  });
});

describe('serializeChatAttachments', () => {
  test('returns empty string for an empty list', () => {
    expect(serializeChatAttachments([])).toBe('');
  });

  test('emits [run-id: ...] [metric-id: ...] [file-path: ...] tokens', () => {
    const out = serializeChatAttachments([
      { id: 'run:a', kind: 'run', label: 'a', ref: 'abc' },
      { id: 'metric:m', kind: 'metric', label: 'm', ref: 'helpfulness' },
      { id: 'file:f', kind: 'file', label: 'f', ref: '/tmp/x.json' },
    ]);
    expect(out).toBe(
      '[run-id: abc] [metric-id: helpfulness] [file-path: /tmp/x.json]\n',
    );
  });
});
