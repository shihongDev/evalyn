/**
 * Skeleton - shimmering placeholder for content that hasn't loaded yet.
 *
 * Used by the v2 routes to render the page chrome (cards, headers, layout)
 * while the data fetch is in flight, avoiding the empty "Loading..." string
 * that makes the app feel frozen on first paint.
 *
 * The shimmer keyframes (`eShimmer`) are defined in `v2/styles.css`.
 *
 * Companion helpers:
 *  - <Spinner /> - a small spinning glyph for refresh/regenerate buttons
 *  - <UpdatingChip /> - corner chip shown when cached data is being refreshed
 */

import type { CSSProperties } from 'react';
import { E } from '../tokens';

interface SkeletonProps {
  /** Width - number = px, string = any CSS length. Defaults to 100%. */
  w?: number | string;
  /** Height in px. Defaults to 12. */
  h?: number;
  /** Optional style overrides (e.g. borderRadius, marginTop). */
  style?: CSSProperties;
}

export function Skeleton({ w = '100%', h = 12, style }: SkeletonProps) {
  return (
    <span
      // Skeletons are purely decorative shimmer placeholders. The
      // actual "loading" announcement for AT comes from the route's
      // Suspense fallback (initial load) or its UpdatingChip
      // (background refresh) - both already aria-live. Without this
      // hide, SR users would otherwise hear a series of empty spans
      // as the skeleton grid iterates, with no indication of what
      // they represent.
      aria-hidden="true"
      style={{
        display: 'inline-block',
        width: w,
        height: h,
        borderRadius: 4,
        background:
          'linear-gradient(90deg, #f0e9da 0%, #f6f0e1 50%, #f0e9da 100%)',
        backgroundSize: '200% 100%',
        animation: 'eShimmer 1.4s ease-in-out infinite',
        ...style,
      }}
    />
  );
}

/** Inline spinning glyph used by refresh/regenerate buttons.
 *
 * `role="img"` makes the aria-label authoritative (without a role,
 * many SR engines silently drop aria-label on a bare span). The ◐
 * glyph becomes the image's visual content - SR users hear
 * "loading" once instead of either "circle with vertical fill" or
 * the literal Unicode codepoint.
 */
export function Spinner({ size = 12 }: { size?: number }) {
  return (
    <span
      role="img"
      aria-label="loading"
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        lineHeight: `${size}px`,
        fontSize: size,
        animation: 'eSpin 0.9s linear infinite',
      }}
    >
      ◐
    </span>
  );
}

interface UpdatingChipProps {
  /** True while a background refresh is in flight - shows "Updating". */
  visible: boolean;
  /** Background-refresh error. When set (and we still have cached
   * data), surfaces a "Refresh failed" chip with the error in a tooltip
   * so the user knows the on-screen data may be stale. Routes pass
   * `data ? err : null` - if data is missing entirely, they own the
   * full-page error fallback instead. */
  error?: string | null;
  /** Optional retry callback. When provided alongside error, the
   * "Refresh failed" chip becomes clickable and re-runs the fetch. */
  onRetry?: () => void;
}

/**
 * Small chip rendered in the corner of a page that surfaces background-
 * refresh state. Two display states:
 *
 *   - error set + cached data on screen -> "Refresh failed" pill with
 *     the error in a tooltip and an optional inline Retry button.
 *   - visible=true (and no error) -> "Updating..." pill with spinner.
 *   - otherwise nothing.
 *
 * The error state takes precedence so a stale-data warning isn't hidden
 * by a re-attempted refresh that's also in flight.
 */
export function UpdatingChip({ visible, error, onRetry }: UpdatingChipProps) {
  if (error) {
    return (
      <span
        role="status"
        aria-label={`Refresh failed: ${error}`}
        title={error}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '3px 8px',
          borderRadius: 999,
          background: 'rgba(255, 149, 128, 0.12)',
          border: `1px solid rgba(255, 149, 128, 0.3)`,
          color: '#ff9580',
          fontFamily: E.fMono,
          fontSize: 10,
          letterSpacing: '0.04em',
        }}
      >
        <span aria-hidden style={{ lineHeight: 1, fontSize: 11 }}>⚠</span>
        Refresh failed
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            aria-label="Retry refresh"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'inherit',
              fontFamily: 'inherit',
              fontSize: 10,
              padding: '0 2px',
              cursor: 'pointer',
              textDecoration: 'underline',
              textUnderlineOffset: 2,
            }}
          >
            retry
          </button>
        )}
      </span>
    );
  }
  if (!visible) return null;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '3px 8px',
        borderRadius: 999,
        background: E.panel2,
        border: `1px solid ${E.hair2}`,
        color: E.text3,
        fontFamily: E.fMono,
        fontSize: 10,
        letterSpacing: '0.04em',
      }}
    >
      <Spinner size={9} />
      Updating
    </span>
  );
}
