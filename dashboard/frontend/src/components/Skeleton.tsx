/**
 * Skeleton — pulsing placeholder rows used while a list is loading.
 *
 * Renders `rows` rows of `height`px boxes with a CSS gradient that animates
 * between `--bg-2` and `--bg-3` so the user gets a hint that something is on
 * the way. Pure presentational; the parent owns the show/hide decision.
 *
 * The animation is injected as a `<style>` tag inside the component so we
 * don't need to touch the global stylesheet (which several other agents are
 * concurrently editing).
 */

const ANIMATION_NAME = 'evalyn-skeleton-pulse';

const KEYFRAMES = `
@keyframes ${ANIMATION_NAME} {
  0%   { background-position: 0% 0%; opacity: 0.7; }
  50%  { background-position: 100% 0%; opacity: 1; }
  100% { background-position: 0% 0%; opacity: 0.7; }
}`;

export interface SkeletonProps {
  /** How many rows to render. Defaults to 3. */
  rows?: number;
  /** Per-row height in pixels. Defaults to 40. */
  height?: number;
  /** Optional `data-testid` for assertions. */
  testId?: string;
}

const Skeleton = ({ rows = 3, height = 40, testId }: SkeletonProps) => {
  return (
    <div
      data-testid={testId ?? 'skeleton'}
      role="status"
      aria-busy="true"
      aria-live="polite"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        padding: '12px 14px',
      }}
    >
      <style>{KEYFRAMES}</style>
      <span className="visually-hidden" style={{ position: 'absolute', left: -9999 }}>
        Loading...
      </span>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          aria-hidden="true"
          style={{
            height,
            borderRadius: 4,
            background:
              'linear-gradient(90deg, var(--bg-2) 0%, var(--bg-3) 50%, var(--bg-2) 100%)',
            backgroundSize: '200% 100%',
            animation: `${ANIMATION_NAME} 1.4s ease-in-out infinite`,
            // Stagger so adjacent rows aren't perfectly synchronized — feels
            // more "loading" and less like a strobe.
            animationDelay: `${i * 0.12}s`,
          }}
        />
      ))}
    </div>
  );
};

export default Skeleton;
