/**
 * NotFound - rendered for any route that doesn't match a known path.
 *
 * Was previously silent: the wildcard route in V2App rendered Home
 * for unknown URLs, so a typo or stale link landed the user on the
 * dashboard root with no indication of what happened. This page
 * surfaces the failed path explicitly and offers escape hatches,
 * including a "did you mean?" suggestion based on substring match
 * against known top-level routes.
 */

import { useLocation, useNavigate } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow } from '../ui';
import { preloadByPath } from '../routePreloads';
import { E } from '../tokens';

const KNOWN_ROUTES: Array<{ path: string; label: string }> = [
  { path: '/', label: 'Home' },
  { path: '/experiments', label: 'Experiments' },
  { path: '/datasets', label: 'Datasets' },
  { path: '/metrics', label: 'Metrics' },
  { path: '/review', label: 'Review' },
  { path: '/reports', label: 'Reports' },
  { path: '/commands', label: 'Commands' },
  { path: '/settings', label: 'Settings' },
  { path: '/annotate', label: 'Annotate' },
  { path: '/copilot', label: 'Co-pilot' },
];

/** Compute the suggestion list for a failed path. Cheap substring +
 * Levenshtein-ish distance: if the user's path's basename (after the
 * last /) shares 3+ chars with a known route's basename, suggest it.
 * Returns up to 2 best matches, never the literal failed path. */
function suggestRoutes(failedPath: string): typeof KNOWN_ROUTES {
  const base = failedPath.replace(/^\/+|\/+$/g, '').split('/')[0].toLowerCase();
  if (!base) return [];
  const scored = KNOWN_ROUTES.map((r) => {
    const target = r.path.replace(/^\//, '').toLowerCase();
    if (target === base) return { r, score: 0 }; // exact match (can't happen — would have routed)
    // Score = how many leading chars match
    let leadingShared = 0;
    while (
      leadingShared < Math.min(base.length, target.length) &&
      base[leadingShared] === target[leadingShared]
    ) {
      leadingShared += 1;
    }
    // Bonus if one is a prefix of the other (typos like 'anotate' → 'annotate')
    const includes = target.includes(base) || base.includes(target);
    return { r, score: leadingShared + (includes ? 3 : 0) };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.filter((s) => s.score >= 3).slice(0, 2).map((s) => s.r);
}

export default function NotFound() {
  const location = useLocation();
  const navigate = useNavigate();
  const suggestions = suggestRoutes(location.pathname);
  return (
    <AppShell>
      <div style={{ padding: '64px 36px', maxWidth: 720, margin: '0 auto' }}>
        <Card style={{ padding: 32, textAlign: 'center' }}>
          <span
            style={{
              display: 'block',
              fontSize: 36,
              color: E.text4,
              fontFamily: E.fSerif,
              lineHeight: 1,
              marginBottom: 14,
            }}
            aria-hidden
          >
            ◌
          </span>
          <Eyebrow>Page not found</Eyebrow>
          <div
            style={{
              marginTop: 10,
              fontSize: 14,
              color: E.text2,
              lineHeight: 1.55,
            }}
          >
            We couldn't find a page at this URL.
          </div>
          <div
            style={{
              marginTop: 8,
              fontFamily: E.fMono,
              fontSize: 12,
              color: E.text3,
              wordBreak: 'break-all',
            }}
          >
            {location.pathname}
            {location.search}
          </div>
          {suggestions.length > 0 && (
            <div style={{ marginTop: 18 }}>
              <div
                style={{
                  fontSize: 11,
                  color: E.text3,
                  fontFamily: E.fMono,
                  textTransform: 'uppercase',
                  letterSpacing: 0.5,
                  marginBottom: 8,
                }}
              >
                Did you mean
              </div>
              <div
                style={{
                  display: 'flex',
                  gap: 8,
                  justifyContent: 'center',
                  flexWrap: 'wrap',
                }}
              >
                {suggestions.map((s) => (
                  <button
                    key={s.path}
                    type="button"
                    onClick={() => navigate(s.path)}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = E.ember;
                      e.currentTarget.style.color = E.ember;
                      // Warm the route chunk while the user is still
                      // hovering so the click->paint path is in cache.
                      // Same pattern as the AppShell nav rail and
                      // every in-page list link.
                      const fn = preloadByPath(s.path);
                      if (fn) void fn();
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = E.hair;
                      e.currentTarget.style.color = E.text2;
                    }}
                    style={{
                      fontFamily: E.fMono,
                      fontSize: 12,
                      color: E.text2,
                      background: 'transparent',
                      border: `1px solid ${E.hair}`,
                      borderRadius: 999,
                      padding: '5px 12px',
                      cursor: 'pointer',
                      transition: 'all 140ms',
                    }}
                  >
                    {s.label} <span style={{ color: E.text3 }}>({s.path})</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          <div
            style={{
              marginTop: 22,
              display: 'flex',
              gap: 8,
              justifyContent: 'center',
            }}
          >
            <Btn kind="primary" size="md" onClick={() => navigate('/')}>
              Back to Home
            </Btn>
            <Btn kind="secondary" size="md" onClick={() => navigate(-1)}>
              ← Previous page
            </Btn>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
