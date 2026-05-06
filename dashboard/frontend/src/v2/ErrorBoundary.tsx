/**
 * ErrorBoundary - top-level catch for render errors and lazy-chunk
 * load failures.
 *
 * The route subtree is wrapped so that:
 *
 *   - Render exceptions (thrown by any component below this) render
 *     a friendly fallback instead of crashing to a blank page.
 *   - Lazy chunk-load failures (deploy mid-session, network blip,
 *     CSP block) surface a "reload to get the latest version" hint
 *     and a primary Reload button.
 *
 * React doesn't expose a hook for this. Class component is the only
 * way to implement getDerivedStateFromError + componentDidCatch.
 *
 * Reset semantics: on every navigation (route change), the boundary
 * resets so a one-route failure doesn't keep the user stuck. The
 * resetKey prop is wired by the caller to location.key (or pathname)
 * so changing it triggers reset.
 */

import { Component, type ErrorInfo, type ReactNode } from 'react';
import { E } from './tokens';

interface ErrorBoundaryProps {
  /** When this changes, the boundary resets - typically wired to
   * location.key so navigating to a different route gives the user a
   * fresh chance even after a previous crash. */
  resetKey?: string;
  children: ReactNode;
}

interface ErrorBoundaryState {
  err: Error | null;
}

/** Heuristic: is the error a lazy-chunk load failure?
 *
 * Vite's dynamic-import error typically reads "Failed to fetch
 * dynamically imported module: <url>" but Safari/Firefox use slightly
 * different phrasing. Match liberally on substrings that show up
 * across browsers. */
function isChunkLoadError(err: Error): boolean {
  const msg = `${err.name} ${err.message}`.toLowerCase();
  return (
    msg.includes('chunkloaderror') ||
    msg.includes('dynamically imported module') ||
    msg.includes('failed to fetch') ||
    msg.includes('loading chunk') ||
    msg.includes('importing a module script failed')
  );
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { err: null };

  static getDerivedStateFromError(err: Error): ErrorBoundaryState {
    return { err };
  }

  componentDidCatch(err: Error, info: ErrorInfo) {
    // Log to the dev console so debug builds stay debuggable. Prod
    // logging would go through a real telemetry channel, which this
    // app doesn't have wired today.
    console.error('[ErrorBoundary]', err, info.componentStack);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    // Reset on resetKey change so a per-route crash doesn't strand
    // the user across navigations.
    if (this.state.err && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ err: null });
    }
  }

  render() {
    const { err } = this.state;
    if (!err) return this.props.children;
    const chunkFailure = isChunkLoadError(err);
    return (
      <div
        role="alert"
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px',
          background: E.ink,
          color: E.text1,
          fontFamily: E.fSans,
        }}
      >
        <div
          style={{
            maxWidth: 480,
            width: '100%',
            background: E.panel,
            border: `1px solid ${E.hair2}`,
            borderRadius: 12,
            padding: 28,
            boxShadow: '0 12px 32px rgba(0,0,0,0.18)',
          }}
        >
          <div
            style={{
              fontFamily: E.fMono,
              fontSize: 11,
              color: E.fail,
              letterSpacing: 0.6,
              textTransform: 'uppercase',
              marginBottom: 8,
            }}
          >
            {chunkFailure ? 'A new version was deployed' : 'Something went wrong'}
          </div>
          <h1
            style={{
              fontFamily: E.fSerif,
              fontSize: 24,
              fontWeight: 400,
              margin: 0,
              color: E.text0,
              lineHeight: 1.2,
            }}
          >
            {chunkFailure
              ? 'Reload to get the latest evalyn'
              : 'The dashboard hit an unexpected error'}
          </h1>
          <p
            style={{
              fontSize: 13.5,
              color: E.text2,
              lineHeight: 1.55,
              marginTop: 12,
            }}
          >
            {chunkFailure
              ? 'A piece of the page tried to load but the server returned a stale chunk. This usually means the dashboard was redeployed while you had it open. Reload to fetch the new version.'
              : 'The error has been logged to the console. You can try reloading the page; if the problem keeps happening, copy the message below and share it with your team.'}
          </p>
          <pre
            style={{
              marginTop: 14,
              padding: '10px 12px',
              background: E.panel2,
              border: `1px solid ${E.hair}`,
              borderRadius: 8,
              fontFamily: E.fMono,
              fontSize: 11,
              color: E.text2,
              maxHeight: 160,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              margin: '14px 0 0',
            }}
          >
            {err.name}: {err.message}
          </pre>
          <div style={{ marginTop: 20, display: 'flex', gap: 8 }}>
            <button
              type="button"
              onClick={() => window.location.reload()}
              style={{
                padding: '8px 16px',
                background: E.ember,
                color: E.emberInk,
                border: 'none',
                borderRadius: 8,
                cursor: 'pointer',
                fontSize: 13,
                fontFamily: E.fSans,
                fontWeight: 500,
              }}
            >
              Reload page
            </button>
            <button
              type="button"
              onClick={() => {
                window.location.href = '/';
              }}
              style={{
                padding: '8px 16px',
                background: 'transparent',
                color: E.text2,
                border: `1px solid ${E.hair2}`,
                borderRadius: 8,
                cursor: 'pointer',
                fontSize: 13,
                fontFamily: E.fSans,
              }}
            >
              Go home
            </button>
          </div>
        </div>
      </div>
    );
  }
}
