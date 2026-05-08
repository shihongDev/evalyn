import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import { ErrorBoundary } from './ErrorBoundary';

/** Minimal child that throws on render. We can't catch render errors
 * cleanly in jsdom without ErrorBoundary, which is what we're testing.
 * React logs the caught error to console.error - silence that one
 * channel so the test output stays scannable. */
function Boom({ msg }: { msg: string }) {
  throw new Error(msg);
}

describe('ErrorBoundary', () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // React logs caught errors via console.error; the ErrorBoundary
    // itself also logs. Silence both so the suite output is clean.
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
    cleanup();
  });

  it('renders the fallback with the error name + message in the report', () => {
    render(
      <ErrorBoundary>
        <Boom msg="catastrophic-failure-xyzzy" />
      </ErrorBoundary>,
    );
    // The report pre includes the error message verbatim.
    expect(screen.getByText(/catastrophic-failure-xyzzy/)).toBeTruthy();
  });

  it('Copy details button writes a multi-section paste-ready report', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      writable: true,
      value: { writeText },
    });

    render(
      <ErrorBoundary>
        <Boom msg="boom-from-test" />
      </ErrorBoundary>,
    );
    const copyBtn = screen.getByRole('button', { name: /copy error details/i });
    fireEvent.click(copyBtn);

    // Wait for the async clipboard write to settle.
    await Promise.resolve();
    await Promise.resolve();

    expect(writeText).toHaveBeenCalledTimes(1);
    const report = writeText.mock.calls[0][0] as string;
    expect(report).toContain('boom-from-test');
    // Includes URL + UA so a teammate reading the paste has env context.
    expect(report).toContain('URL:');
    expect(report).toContain('User-Agent:');
  });

  it('button label flips to "Copied" after a successful copy', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      writable: true,
      value: { writeText },
    });

    render(
      <ErrorBoundary>
        <Boom msg="boom-2" />
      </ErrorBoundary>,
    );
    const copyBtn = screen.getByRole('button', { name: /copy error details/i });
    fireEvent.click(copyBtn);

    // Wait for the async clipboard write + setState that flips copied=true.
    // The async clipboard.writeText is awaited inside handleCopy before the
    // setState fires, so a single microtask flush isn't enough; waitFor
    // polls until the DOM reflects the state.
    await waitFor(() => {
      expect(copyBtn.textContent).toContain('Copied');
    });
  });
});
