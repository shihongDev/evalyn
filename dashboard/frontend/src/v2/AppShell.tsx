/**
 * AppShell - top bar + left nav + co-pilot dock used by every v2 route.
 * Ported from /tmp/evalyn-v2/app-shell.jsx and adapted for react-router-dom.
 *
 * The shell is route-aware: `useLocation()` highlights the active nav item,
 * `useNavigate()` powers nav clicks. The Co-pilot dock visibility is held
 * in the v2 store so other surfaces (e.g. the full-thread route) can hide
 * it when they take over the screen.
 *
 * Responsive behavior (three breakpoints, watched via `useViewport`):
 *   - desktop  (>= 1100px): 230px nav + 1fr main + 420px dock when open.
 *   - tablet   (700-1099px): 56px icon-only nav + 1fr main; dock hidden by
 *     default and surfaced as a 420px right-side overlay when toggled.
 *   - mobile   (< 700px): nav collapses; top bar gains a hamburger that
 *     opens a slide-in drawer. Dock surfaces as a bottom sheet (~75vh).
 *
 * Route content padding is intentionally left to the routes themselves;
 * this shell is responsible for chrome only.
 */

import { useEffect, useState, type CSSProperties, type ReactNode } from 'react';
import { useLocation, useNavigate, NavLink } from 'react-router-dom';
import { E } from './tokens';
import { Btn, Eyebrow, Pill, StatusDot } from './ui';
import { MOD_KEY } from './platform';
import { useV2Store } from './store/store';
import { CoPilotDock } from './copilot/CoPilotDock';
import { CommandPalette } from './CommandPalette';
import { CliRunner } from './CliRunner';
import { RecentJobsDrawer } from './RecentJobsDrawer';
import {
  activeJobCount,
  loadJobsHistory,
  subscribeJobsHistory,
} from './jobsHistory';
import { v2 } from './api/client';
import { listCli } from './api/cli';
import { prefetchV2 } from './hooks/useV2Resource';
import { startV2EventStream } from './api/v2ws';
import { useTour } from './tour/useTour';
import { TourMenu } from './tour/TourMenu';

interface AppShellProps {
  children: ReactNode;
  contextChip?: { name: string; version: string | null };
  breadcrumb?: string[];
  headerExtra?: ReactNode;
  /** When true, the right co-pilot dock is hidden (used by the full-thread route). */
  hideCoPilot?: boolean;
}

interface NavItem {
  id: string;
  path: string;
  icon: string;
  label: string;
  /** Imperative warmup for hover/focus - kicks off the route's primary fetch. */
  prefetch?: () => void;
}

/** Maps common breadcrumb labels back to the route path they came from
 * so the breadcrumb's parent segments can be clickable. Routes pass
 * `breadcrumb={['Annotate', 'run/abc']}` and we want "Annotate" to link
 * to /annotate. Built off-line since route names are stable. */
const BREADCRUMB_ROUTE_FOR_LABEL: Record<string, string> = {
  Home: '/',
  Annotate: '/annotate',
  Experiments: '/experiments',
  Datasets: '/datasets',
  Metrics: '/metrics',
  Review: '/review',
  Reports: '/reports',
  Commands: '/commands',
  Settings: '/settings',
};

/** Maps a pathname prefix to a friendly tab title. Prefix-based so
 * /experiments/abc inherits "Experiments". Used by useDocumentTitle. */
const TITLE_FOR_PATH: Array<[string, string]> = [
  ['/experiments', 'Experiments'],
  ['/datasets', 'Datasets'],
  ['/metrics', 'Metrics'],
  ['/review', 'Review'],
  ['/reports', 'Reports'],
  ['/commands', 'Commands'],
  ['/settings', 'Settings'],
  ['/annotate', 'Annotate'],
  ['/copilot', 'Co-pilot'],
  ['/', 'Home'],
];

const NAV_ITEMS: NavItem[] = [
  { id: 'home', path: '/', icon: '◐', label: 'Home', prefetch: () => prefetchV2('home', v2.home) },
  {
    id: 'experiments',
    path: '/experiments',
    icon: '◆',
    label: 'Experiments',
    prefetch: () => prefetchV2('experiments', v2.experiments),
  },
  {
    id: 'commands',
    path: '/commands',
    icon: '⌥',
    label: 'Commands',
    prefetch: () => prefetchV2('commands', listCli),
  },
  {
    id: 'datasets',
    path: '/datasets',
    icon: '◫',
    label: 'Datasets',
    prefetch: () => prefetchV2('datasets', v2.datasets),
  },
  {
    id: 'metrics',
    path: '/metrics',
    icon: '◈',
    label: 'Metrics & rubrics',
    prefetch: () => prefetchV2('rubrics', v2.rubrics),
  },
  {
    id: 'review',
    path: '/review',
    icon: '◉',
    label: 'Human review',
    prefetch: () => prefetchV2('reviewQueue', v2.reviewQueue),
  },
  {
    id: 'annotate',
    path: '/annotate',
    icon: '✎',
    label: 'Annotate',
    // The landing reads the session list; warm it on hover.
    prefetch: () =>
      prefetchV2('annotation/sessions', () =>
        import('./api/annotation').then((m) => m.annotationApi.listSessions()),
      ),
  },
  {
    id: 'reports',
    path: '/reports',
    icon: '▤',
    label: 'Reports',
    prefetch: () => prefetchV2('weeklyReport', v2.weeklyReport),
  },
];

const PINNED: { name: string; q: number; status: 'pass' | 'warn' | 'fail' }[] = [];

export type Viewport = 'mobile' | 'tablet' | 'desktop';

const TABLET_MAX = 1099;
const MOBILE_MAX = 699;

function detectViewport(): Viewport {
  if (typeof window === 'undefined') return 'desktop';
  if (window.innerWidth <= MOBILE_MAX) return 'mobile';
  if (window.innerWidth <= TABLET_MAX) return 'tablet';
  return 'desktop';
}

/** Tracks viewport bucket via window.matchMedia. Re-renders on bucket change only. */
export function useViewport(): Viewport {
  const [vp, setVp] = useState<Viewport>(detectViewport);
  useEffect(() => {
    function handle() {
      const next = detectViewport();
      setVp((prev) => (prev === next ? prev : next));
    }
    window.addEventListener('resize', handle);
    return () => window.removeEventListener('resize', handle);
  }, []);
  return vp;
}

function activeIdFromPath(pathname: string): string {
  if (pathname === '/' || pathname.startsWith('/copilot')) return 'home';
  if (pathname === '/settings' || pathname.startsWith('/settings/')) return 'settings';
  for (const item of NAV_ITEMS) {
    if (item.id === 'home') continue;
    if (pathname === item.path || pathname.startsWith(`${item.path}/`)) return item.id;
  }
  return 'home';
}

export function AppShell({
  children,
  contextChip,
  breadcrumb,
  headerExtra,
  hideCoPilot,
}: AppShellProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const dockOpen = useV2Store((s) => s.dockOpen);
  const setDockOpen = useV2Store((s) => s.setDockOpen);
  const active = activeIdFromPath(location.pathname);
  const vp = useViewport();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [jobsDrawerOpen, setJobsDrawerOpen] = useState(false);
  // `runningCount` drives the badge on the floating + topbar Recent Jobs
  // buttons. We re-read once on mount and on every history mutation -
  // jobsHistory's notifier covers same-tab + cross-tab changes.
  const [runningCount, setRunningCount] = useState<number>(() =>
    activeJobCount(loadJobsHistory()),
  );
  useEffect(() => {
    return subscribeJobsHistory(() => {
      setRunningCount(activeJobCount(loadJobsHistory()));
    });
  }, []);

  // Open the shared /ws/v2/events socket once, app-wide. Idempotent
  // and best-effort: if the WS fails to connect the FE falls back to
  // useV2Resource's existing refresh-on-nav semantics.
  useEffect(() => {
    startV2EventStream();
  }, []);

  // Drive the active tour (if any) globally. Mounting at the shell level
  // lets any route trigger a tour via setTour() in the store; the hook
  // resolves the registered TourDef and runs Driver.js. Per the tour state
  // machine in useTour.ts, navigating away mid-tour abandons it (cleanup
  // calls driver.destroy() which fires onDestroyStarted -> abandonTour).
  useTour();

  // On viewport change, default the dock state to "closed" for narrow widths
  // so users do not land in an overlay-on-load state. Desktop keeps prior
  // value (which defaults to true in the store).
  useEffect(() => {
    if (vp !== 'desktop' && dockOpen) setDockOpen(false);
    // Intentionally only run on viewport change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vp]);

  // Close the mobile drawer whenever we leave mobile or the route changes.
  useEffect(() => {
    if (vp !== 'mobile' && drawerOpen) setDrawerOpen(false);
  }, [vp, drawerOpen]);
  useEffect(() => {
    setDrawerOpen(false);
  }, [location.pathname]);

  // Browser tab title reflects the current route. Without this every
  // tab reads "Evalyn · Workbench", making multi-tab workflows
  // (compare two runs, annotate while watching a job) hard to scan.
  // First-match-wins prefix lookup against TITLE_FOR_PATH.
  useEffect(() => {
    const path = location.pathname;
    const match = TITLE_FOR_PATH.find(([prefix]) =>
      prefix === '/' ? path === '/' : path.startsWith(prefix),
    );
    document.title = match ? `${match[1]} · Evalyn` : 'Evalyn · Workbench';
  }, [location.pathname]);

  // Global hotkeys:
  //   Cmd/Ctrl+K toggles the command palette.
  //   Cmd/Ctrl+, navigates to /settings (canonical "preferences"
  //     shortcut). Skipped on /annotate/:sessionId routes which
  //     have their own local Cmd+, for the per-session settings menu.
  //   Esc closes palette/drawer/dock as appropriate.
  // Bound at window level so they work from any focus.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const isToggle = (e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K');
      if (isToggle) {
        e.preventDefault();
        setPaletteOpen((v) => !v);
        return;
      }
      // Settings shortcut. Skip when AnnotateSession handles it locally.
      if (
        (e.metaKey || e.ctrlKey) &&
        e.key === ',' &&
        !location.pathname.startsWith('/annotate/')
      ) {
        e.preventDefault();
        if (location.pathname !== '/settings') navigate('/settings');
        return;
      }
      if (e.key === 'Escape') {
        setPaletteOpen((v) => (v ? false : v));
        setDrawerOpen((v) => (v ? false : v));
        // Also close the dock when it is overlaying content (tablet/mobile).
        if (vp !== 'desktop' && dockOpen) setDockOpen(false);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [vp, dockOpen, setDockOpen, location.pathname, navigate]);

  const showDock = !hideCoPilot && dockOpen;
  // Dock layout mode: 'docked' takes a grid column on desktop; 'overlay' and
  // 'sheet' position fixed and float above content.
  const dockMode: 'docked' | 'overlay' | 'sheet' =
    vp === 'desktop' ? 'docked' : vp === 'tablet' ? 'overlay' : 'sheet';

  // Grid template per viewport. On tablet/mobile the dock is taken out of the
  // grid (it overlays), so the column is removed entirely.
  const gridTemplateColumns =
    vp === 'desktop'
      ? `230px 1fr ${showDock ? '420px' : '0px'}`
      : vp === 'tablet'
        ? '56px 1fr'
        : '1fr';

  const sidebarMode: 'full' | 'icon' | 'drawer' =
    vp === 'desktop' ? 'full' : vp === 'tablet' ? 'icon' : 'drawer';

  return (
    <div
      data-vp={vp}
      style={{
        width: '100%',
        height: '100%',
        background: E.ink,
        color: E.text1,
        fontFamily: E.fSans,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <style>{`
        @keyframes eDotPulse { 0%,100% { opacity: 1 } 50% { opacity: 0.4 } }
        @keyframes eFadeIn { from { opacity: 0 } to { opacity: 1 } }
        @keyframes eSlideInLeft { from { transform: translateX(-100%) } to { transform: translateX(0) } }
        @keyframes eSlideInRight { from { transform: translateX(100%) } to { transform: translateX(0) } }
        @keyframes eSlideUp { from { transform: translateY(100%) } to { transform: translateY(0) } }
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-thumb { background: ${E.panel3}; border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: ${E.panel4}; }
        body { background: ${E.ink}; }
      `}</style>

      {/* TOP BAR */}
      <div
        style={{
          height: 56,
          background: E.panel,
          borderBottom: `1px solid ${E.hair}`,
          display: 'flex',
          alignItems: 'center',
          gap: vp === 'mobile' ? 10 : 16,
          padding: vp === 'mobile' ? '0 12px' : '0 22px',
          flexShrink: 0,
        }}
      >
        {vp === 'mobile' && (
          <button
            type="button"
            onClick={() => setDrawerOpen((v) => !v)}
            aria-label="Open navigation"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 34,
              height: 34,
              borderRadius: 6,
              background: 'transparent',
              border: `1px solid ${E.hair2}`,
              color: E.text1,
              cursor: 'pointer',
              padding: 0,
              fontSize: 16,
              lineHeight: 1,
            }}
          >
            ☰
          </button>
        )}
        <button
          type="button"
          onClick={() => navigate('/')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            padding: 0,
          }}
        >
          <div style={{ width: 14, height: 14, borderRadius: 4, background: E.ember }} />
          <span
            style={{
              fontFamily: E.fSerif,
              fontSize: 19,
              color: E.text0,
              letterSpacing: '-0.01em',
            }}
          >
            evalyn
          </span>
        </button>
        {/* Context chip + breadcrumb hide on narrow widths to save space.
            Tablet keeps the chip; mobile drops both. */}
        {contextChip && vp !== 'mobile' && (
          <>
            <span style={{ color: E.text4 }}>·</span>
            {/* Project chip - currently a passive identity badge. Switching
                projects from the UI is on the roadmap; until then, render as
                a non-clickable <span> so the caret doesn't imply a dropdown. */}
            <span
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 10px',
                borderRadius: 6,
                background: E.panel2,
                border: `1px solid ${E.hair2}`,
                color: E.text1,
                fontSize: 12.5,
              }}
            >
              <span style={{ fontWeight: 500 }}>{contextChip.name}</span>
              <span style={{ color: E.text3, fontSize: 11 }}>{contextChip.version ?? '-'}</span>
            </span>
          </>
        )}
        {breadcrumb && breadcrumb.length > 0 && vp === 'desktop' && (
          <>
            <span style={{ color: E.text4 }}>·</span>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 12.5,
                color: E.text2,
              }}
            >
              {breadcrumb.map((b, i) => {
                const isLast = i === breadcrumb.length - 1;
                // Map common labels to their canonical route paths so
                // parent segments become clickable. Last segment stays
                // text ("you are here"). Backwards-compatible - existing
                // string[] callers get this for free without API changes.
                const path = isLast ? null : BREADCRUMB_ROUTE_FOR_LABEL[b] ?? null;
                return (
                  <span key={`${b}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {path ? (
                      <button
                        type="button"
                        onClick={() => navigate(path)}
                        title={`Go to ${b}`}
                        style={{
                          color: E.text2,
                          background: 'transparent',
                          border: 'none',
                          padding: 0,
                          font: 'inherit',
                          cursor: 'pointer',
                          textDecoration: 'none',
                          transition: 'color 160ms',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.color = E.ember;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.color = E.text2;
                        }}
                      >
                        {b}
                      </button>
                    ) : (
                      <span style={{ color: isLast ? E.text1 : E.text2 }}>{b}</span>
                    )}
                    {!isLast && <span style={{ color: E.text4 }}>›</span>}
                  </span>
                );
              })}
            </div>
          </>
        )}
        <span style={{ flex: 1 }} />
        {headerExtra}
        <TourMenu />
        {vp === 'desktop' && (
          <Btn
            kind="ghost"
            size="sm"
            onClick={() => setJobsDrawerOpen(true)}
            title="Recent jobs"
            style={{ gap: 6, position: 'relative' }}
          >
            <span style={{ fontSize: 13, lineHeight: 1 }}>⟳</span>
            Jobs
            {runningCount > 0 && <BadgeCount n={runningCount} inline />}
          </Btn>
        )}
        {vp !== 'mobile' && (
          <Btn
            kind="ghost"
            size="sm"
            onClick={() => setPaletteOpen(true)}
            style={{ fontFamily: E.fMono, gap: 4 }}
          >
            {MOD_KEY}K Search
          </Btn>
        )}
        {/* User initials - decorative until a profile menu exists. */}
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 30,
            height: 30,
            borderRadius: '50%',
            background: E.panel3,
            color: E.text0,
            fontSize: 11,
            fontWeight: 600,
            border: `1px solid ${E.hair2}`,
          }}
          title="Signed in (profile menu coming soon)"
        >
          SK
        </span>
      </div>

      {/* BODY */}
      <div
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns,
          minHeight: 0,
          position: 'relative',
        }}
      >
        {/* SIDEBAR - only rendered as a grid column on desktop/tablet.
            On mobile it becomes a fixed slide-in drawer below. */}
        {sidebarMode !== 'drawer' && (
          <Sidebar
            mode={sidebarMode}
            active={active}
            navigate={navigate}
            dockOpen={dockOpen}
            setDockOpen={setDockOpen}
          />
        )}

        {/* MAIN CONTENT */}
        <div style={{ overflow: 'auto', background: E.ink }}>{children}</div>

        {/* CO-PILOT DOCK - rendered only when open + not hidden by route.
            Mode varies with viewport (docked column vs overlay vs sheet). */}
        {showDock && (
          <CoPilotDock onClose={() => setDockOpen(false)} mode={dockMode} />
        )}

        {/* Floating "Ask co-pilot" button on tablet/mobile when dock closed.
            Sits inside the body so it visually belongs to the main area. */}
        {!hideCoPilot && !showDock && vp !== 'desktop' && (
          <button
            type="button"
            onClick={() => setDockOpen(true)}
            aria-label="Open co-pilot"
            style={{
              position: 'fixed',
              right: 18,
              bottom: 18,
              zIndex: 700,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '10px 14px',
              borderRadius: 999,
              background: E.ember,
              color: E.emberInk,
              border: 'none',
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 500,
              boxShadow: `0 8px 24px rgba(217,106,44,0.32)`,
            }}
          >
            <span style={{ fontFamily: E.fSerif, fontSize: 15 }}>e</span>
            Ask co-pilot
          </button>
        )}

        {/* Floating Recent Jobs button - bottom-left mirror of the co-pilot
            FAB. Visible on every viewport. On desktop we offset past the
            230px sidebar so the FAB doesn't land on top of it; on tablet
            (56px icon-rail) we clear that, and mobile gets the natural
            18px corner. Badge surfaces queued/running count from the
            local jobsHistory ledger (hidden when 0). */}
        <button
          type="button"
          onClick={() => setJobsDrawerOpen(true)}
          aria-label="Open recent jobs"
          style={{
            position: 'fixed',
            left: vp === 'desktop' ? 248 : vp === 'tablet' ? 74 : 18,
            bottom: 18,
            zIndex: 700,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            padding: '10px 14px',
            borderRadius: 999,
            background: E.panel,
            color: E.text1,
            border: `1px solid ${E.hair2}`,
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 500,
            boxShadow: '0 8px 24px rgba(26,24,18,0.18)',
          }}
        >
          <span style={{ fontSize: 14, lineHeight: 1 }}>⟳</span>
          Recent jobs
          {runningCount > 0 && <BadgeCount n={runningCount} />}
        </button>
      </div>

      {/* MOBILE NAV DRAWER - mounted always so the slide animation can play
          on close. We drive it via transform + visibility so a closed drawer
          is fully unreachable to keyboard nav. */}
      {sidebarMode === 'drawer' && (
        <>
          {drawerOpen && (
            <div
              onClick={() => setDrawerOpen(false)}
              style={{
                position: 'fixed',
                inset: 0,
                top: 56,
                background: 'rgba(20, 18, 14, 0.45)',
                zIndex: 850,
                animation: 'eFadeIn 140ms ease',
              }}
            />
          )}
          <div
            aria-hidden={!drawerOpen}
            style={{
              position: 'fixed',
              top: 56,
              left: 0,
              bottom: 0,
              width: 260,
              maxWidth: '82vw',
              background: E.panel,
              borderRight: `1px solid ${E.hair}`,
              zIndex: 860,
              transform: drawerOpen ? 'translateX(0)' : 'translateX(-100%)',
              transition: 'transform 220ms ease',
              boxShadow: drawerOpen ? '4px 0 24px rgba(0,0,0,0.16)' : 'none',
              visibility: drawerOpen ? 'visible' : 'hidden',
            }}
          >
            <Sidebar
              mode="full"
              active={active}
              navigate={navigate}
              dockOpen={dockOpen}
              setDockOpen={setDockOpen}
              onAfterNavigate={() => setDrawerOpen(false)}
            />
          </div>
        </>
      )}

      {/* COMMAND PALETTE - rendered last so it overlays the rest of the shell. */}
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      {/* CLI RUNNER - global slide-over driven by openCliRunner() from any route.
          z-index 900 sits above the dock/nav (default) but below the palette
          modal (1000) so Cmd+K stays usable while a job is streaming. */}
      <CliRunner />
      {/* RECENT JOBS DRAWER - z-index 880 sits below CliRunner (900) so a
          drawer-launched resume reads naturally as "drawer hands off to runner". */}
      <RecentJobsDrawer
        open={jobsDrawerOpen}
        onClose={() => setJobsDrawerOpen(false)}
      />
    </div>
  );
}

/** Compact numeric badge used by the Recent Jobs buttons. The `inline` variant
 * sits on a Btn (less prominent) while the default ships on the floating
 * pill. Number is clamped to "9+" past 9 to keep the pill round. */
function BadgeCount({ n, inline }: { n: number; inline?: boolean }) {
  const text = n > 9 ? '9+' : String(n);
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: inline ? 16 : 18,
        height: inline ? 16 : 18,
        padding: '0 5px',
        borderRadius: 999,
        background: E.ember,
        color: E.emberInk,
        fontSize: inline ? 10 : 11,
        fontFamily: E.fMono,
        fontWeight: 600,
        marginLeft: 2,
        lineHeight: 1,
      }}
    >
      {text}
    </span>
  );
}

interface SidebarProps {
  mode: 'full' | 'icon';
  active: string;
  navigate: (to: string) => void;
  dockOpen: boolean;
  setDockOpen: (v: boolean) => void;
  /** Called after a nav item is clicked - used by the mobile drawer to close. */
  onAfterNavigate?: () => void;
}

function Sidebar({ mode, active, navigate, dockOpen, setDockOpen, onAfterNavigate }: SidebarProps) {
  const isIcon = mode === 'icon';
  const containerStyle: CSSProperties = {
    background: E.panel,
    borderRight: `1px solid ${E.hair}`,
    display: 'flex',
    flexDirection: 'column',
    padding: isIcon ? '12px 6px' : '16px 12px',
    overflow: 'hidden',
    height: '100%',
  };

  return (
    <div style={containerStyle}>
      <Btn
        kind="primary"
        size="md"
        onClick={() => {
          navigate('/experiments?new=1');
          onAfterNavigate?.();
        }}
        title="New evaluation"
        style={{
          justifyContent: 'center',
          padding: isIcon ? '10px 0' : '10px 14px',
          fontSize: 13,
          marginBottom: 16,
          boxShadow: `0 6px 20px rgba(255,140,77,0.22)`,
        }}
      >
        <span style={{ fontSize: 14 }}>＋</span>
        {!isIcon && ' New evaluation'}
      </Btn>

      {!isIcon && (
        <Eyebrow style={{ padding: '4px 10px', marginBottom: 4 }}>Workspace</Eyebrow>
      )}
      {NAV_ITEMS.map((item) => {
        const isActive = active === item.id;
        const warm = item.prefetch ?? (() => {});
        return (
          <NavLink
            key={item.id}
            to={item.path}
            end={item.path === '/'}
            onMouseEnter={(e) => {
              warm();
              // Subtle hover bg on inactive items so the nav feels
              // responsive. Skip on the active item (it already has
              // its own bg) to avoid a visual blip.
              if (!isActive) {
                e.currentTarget.style.background = E.panel2;
              }
            }}
            onMouseLeave={(e) => {
              if (!isActive) {
                e.currentTarget.style.background = 'transparent';
              }
            }}
            onFocus={warm}
            onTouchStart={warm}
            onClick={() => onAfterNavigate?.()}
            title={isIcon ? item.label : undefined}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: isIcon ? 0 : 10,
              justifyContent: isIcon ? 'center' : 'flex-start',
              padding: isIcon ? '9px 0' : '7px 10px',
              borderRadius: 6,
              fontSize: 13,
              marginBottom: 1,
              cursor: 'pointer',
              background: isActive ? E.panel3 : 'transparent',
              color: isActive ? E.text0 : E.text2,
              fontWeight: isActive ? 500 : 400,
              border: 'none',
              textAlign: 'left',
              textDecoration: 'none',
              transition: 'background 140ms',
            }}
          >
            <span
              style={{
                fontSize: isIcon ? 14 : 12,
                color: isActive ? E.ember : E.text3,
                width: isIcon ? 'auto' : 14,
              }}
            >
              {item.icon}
            </span>
            {!isIcon && <span style={{ flex: 1 }}>{item.label}</span>}
          </NavLink>
        );
      })}

      {!isIcon && PINNED.length > 0 && (
        <>
          <Eyebrow style={{ padding: '4px 10px', marginTop: 18, marginBottom: 4 }}>
            Pinned runs
          </Eyebrow>
          {PINNED.map((p) => (
            <button
              key={p.name}
              type="button"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '6px 10px',
                borderRadius: 6,
                fontSize: 12,
                color: E.text2,
                cursor: 'pointer',
                background: 'transparent',
                border: 'none',
                textAlign: 'left',
              }}
            >
              <StatusDot status={p.status} />
              <span
                style={{
                  flex: 1,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {p.name}
              </span>
              <span style={{ fontSize: 10, fontFamily: E.fMono, color: E.text3 }}>{p.q}</span>
            </button>
          ))}
        </>
      )}

      <span style={{ flex: 1 }} />
      {!isIcon && (
        <button
          type="button"
          onClick={() => {
            navigate('/settings');
            onAfterNavigate?.();
          }}
          title="Configure LLM providers and API keys"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '7px 10px',
            borderRadius: 6,
            fontSize: 12,
            color: active === 'settings' ? E.text0 : E.text2,
            cursor: 'pointer',
            background: active === 'settings' ? E.panel3 : 'transparent',
            border: 'none',
            textAlign: 'left',
          }}
        >
          <span>⚙</span> Settings & keys
        </button>
      )}
      <button
        type="button"
        onClick={() => setDockOpen(!dockOpen)}
        title={dockOpen ? 'Hide co-pilot' : 'Show co-pilot'}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: isIcon ? 'center' : 'flex-start',
          gap: isIcon ? 0 : 8,
          padding: isIcon ? '7px 0' : '7px 10px',
          borderRadius: 6,
          fontSize: 12,
          color: E.text3,
          cursor: 'pointer',
          background: 'transparent',
          border: 'none',
          textAlign: 'left',
        }}
      >
        <span>◑</span>
        {!isIcon && ` ${dockOpen ? 'Hide' : 'Show'} co-pilot`}
      </button>
      {!isIcon && (
        <div
          style={{
            marginTop: 8,
            padding: '6px 10px',
            fontSize: 10,
            color: E.text4,
            fontFamily: E.fMono,
          }}
        >
          Local · {(typeof window !== 'undefined' && window.location.port) || 'default'} · v2
        </div>
      )}
    </div>
  );
}

/** Top-bar quality pill helper - some routes pass it via headerExtra. */
export function QualityPill({ value, delta }: { value: number | null; delta: number | null }) {
  if (value == null) {
    return (
      <Pill mono color={E.text3} bg={E.panel3}>
        <StatusDot status="idle" /> Quality –
      </Pill>
    );
  }
  const up = delta != null && delta >= 0;
  return (
    <Pill mono color={up ? E.pass : E.fail} bg={up ? E.passDim : E.failDim}>
      <StatusDot status={up ? 'pass' : 'fail'} /> Quality {value.toFixed(1)}%{' '}
      {delta != null ? (up ? '↑' : '↓') : ''}
    </Pill>
  );
}
