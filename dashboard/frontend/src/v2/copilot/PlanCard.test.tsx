/**
 * Tests for PlanCard's in-flight busy guard.
 *
 * Customer scenario: user clicks "Approve" on an agent tool-call
 * confirmation card. Network is slow. They click again. Without
 * the guard, two confirmAgentTool POSTs fire; the second one 4xxs
 * because the tool_call_id is already consumed.
 *
 * The guard has three layers (Convention 8):
 *   - external: `disabled={busy}` blocks mouse + keyboard
 *   - internal: `if (busy) return` inside onClick, in case state
 *     is still flushing
 *   - aria: `aria-busy={clicked === 'approve'}` for SR users
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { PlanCard } from './atoms';
import type { PendingConfirmation } from './types';

const PENDING: PendingConfirmation = {
  tool_call_id: 'tc-1',
  tool: 'run-eval',
  args: {},
  preview_cmd: 'run-eval --dataset x',
  side_effects: [],
};

describe('PlanCard - busy guard', () => {
  afterEach(() => cleanup());

  it('Approve fires onApprove exactly once even on rapid double-click', () => {
    const onApprove = vi.fn();
    const onReject = vi.fn();
    render(
      <PlanCard pending={PENDING} onApprove={onApprove} onReject={onReject} />,
    );
    const approveBtn = screen.getByRole('button', { name: /approve & run/i });

    fireEvent.click(approveBtn);
    fireEvent.click(approveBtn);
    fireEvent.click(approveBtn);

    expect(onApprove).toHaveBeenCalledTimes(1);
    expect(onReject).not.toHaveBeenCalled();
  });

  it('Approve disables both buttons + aria-busy on the clicked one', () => {
    const onApprove = vi.fn();
    const onReject = vi.fn();
    render(
      <PlanCard pending={PENDING} onApprove={onApprove} onReject={onReject} />,
    );
    const approveBtn = screen.getByRole('button', { name: /approve & run/i });
    const rejectBtn = screen.getByRole('button', { name: /reject/i });

    fireEvent.click(approveBtn);

    expect(approveBtn).toBeDisabled();
    expect(rejectBtn).toBeDisabled();
    expect(approveBtn.getAttribute('aria-busy')).toBe('true');
    // The non-clicked button is disabled but NOT busy - it isn't
    // the one doing work.
    expect(rejectBtn.getAttribute('aria-busy')).toBe('false');
  });

  it('Reject after Approve is blocked (race protection)', () => {
    const onApprove = vi.fn();
    const onReject = vi.fn();
    render(
      <PlanCard pending={PENDING} onApprove={onApprove} onReject={onReject} />,
    );
    const approveBtn = screen.getByRole('button', { name: /approve & run/i });
    const rejectBtn = screen.getByRole('button', { name: /reject/i });

    fireEvent.click(approveBtn);
    fireEvent.click(rejectBtn);

    expect(onApprove).toHaveBeenCalledTimes(1);
    expect(onReject).not.toHaveBeenCalled();
  });

  it('button label flips to in-progress copy after click', () => {
    const onApprove = vi.fn();
    const onReject = vi.fn();
    render(
      <PlanCard pending={PENDING} onApprove={onApprove} onReject={onReject} />,
    );
    const approveBtn = screen.getByRole('button', { name: /approve & run/i });
    fireEvent.click(approveBtn);
    // After click the accessible name changes to the in-progress label.
    expect(screen.getByRole('button', { name: /approving/i })).toBeTruthy();
  });
});
