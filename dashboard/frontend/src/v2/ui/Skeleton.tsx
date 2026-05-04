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

/** Inline spinning glyph used by refresh/regenerate buttons. */
export function Spinner({ size = 12 }: { size?: number }) {
  return (
    <span
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

/**
 * Small "Updating..." chip rendered in the corner of a page when the cached
 * data is being refreshed in the background. Only visible while reloading.
 */
export function UpdatingChip({ visible }: { visible: boolean }) {
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
