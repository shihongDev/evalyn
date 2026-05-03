/**
 * AppShell - top bar + left nav + co-pilot dock used by every v2 route.
 * Ported from /tmp/evalyn-v2/app-shell.jsx and adapted for react-router-dom.
 *
 * The shell is route-aware: `useLocation()` highlights the active nav item,
 * `useNavigate()` powers nav clicks. The Co-pilot dock visibility is held
 * in the v2 store so other surfaces (e.g. the full-thread route) can hide
 * it when they take over the screen.
 */

import type { ReactNode } from 'react';
import { useLocation, useNavigate, NavLink } from 'react-router-dom';
import { E } from './tokens';
import { Btn, Eyebrow, Pill, StatusDot } from './ui';
import { useV2Store } from './store/store';
import { CoPilotDock } from './copilot/CoPilotDock';

interface AppShellProps {
  children: ReactNode;
  contextChip?: { name: string; version: string | null };
  breadcrumb?: string[];
  headerExtra?: ReactNode;
  /** When true, the right co-pilot dock is hidden (used by the full-thread route). */
  hideCoPilot?: boolean;
}

const NAV_ITEMS: { id: string; path: string; icon: string; label: string }[] = [
  { id: 'home', path: '/', icon: '◐', label: 'Home' },
  { id: 'experiments', path: '/experiments', icon: '◆', label: 'Experiments' },
  { id: 'datasets', path: '/datasets', icon: '◫', label: 'Datasets' },
  { id: 'metrics', path: '/metrics', icon: '◈', label: 'Metrics & rubrics' },
  { id: 'review', path: '/review', icon: '◉', label: 'Human review' },
  { id: 'reports', path: '/reports', icon: '▤', label: 'Reports' },
];

const PINNED: { name: string; q: number; status: 'pass' | 'warn' | 'fail' }[] = [];

function activeIdFromPath(pathname: string): string {
  if (pathname === '/' || pathname.startsWith('/copilot')) return 'home';
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
  const showDock = !hideCoPilot && dockOpen;

  return (
    <div
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
          gap: 16,
          padding: '0 22px',
          flexShrink: 0,
        }}
      >
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
        {contextChip && (
          <>
            <span style={{ color: E.text4 }}>·</span>
            <button
              type="button"
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
                cursor: 'pointer',
              }}
            >
              <span style={{ fontWeight: 500 }}>{contextChip.name}</span>
              <span style={{ color: E.text3, fontSize: 11 }}>{contextChip.version ?? '-'}</span>
              <span style={{ color: E.text3, fontSize: 11, marginLeft: 2 }}>▾</span>
            </button>
          </>
        )}
        {breadcrumb && breadcrumb.length > 0 && (
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
              {breadcrumb.map((b, i) => (
                <span key={`${b}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ color: i === breadcrumb.length - 1 ? E.text1 : E.text2 }}>{b}</span>
                  {i < breadcrumb.length - 1 && <span style={{ color: E.text4 }}>›</span>}
                </span>
              ))}
            </div>
          </>
        )}
        <span style={{ flex: 1 }} />
        {headerExtra}
        <Btn kind="ghost" size="sm" style={{ fontFamily: E.fMono, gap: 4 }}>
          ⌘K Search
        </Btn>
        <button
          type="button"
          style={{
            width: 30,
            height: 30,
            borderRadius: '50%',
            background: E.panel3,
            color: E.text0,
            fontSize: 11,
            fontWeight: 600,
            cursor: 'pointer',
            border: `1px solid ${E.hair2}`,
          }}
        >
          SK
        </button>
      </div>

      {/* BODY */}
      <div
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: `230px 1fr ${showDock ? '420px' : '0px'}`,
          minHeight: 0,
        }}
      >
        {/* SIDEBAR */}
        <div
          style={{
            background: E.panel,
            borderRight: `1px solid ${E.hair}`,
            display: 'flex',
            flexDirection: 'column',
            padding: '16px 12px',
            overflow: 'hidden',
          }}
        >
          <Btn
            kind="primary"
            size="md"
            onClick={() => navigate('/experiments?new=1')}
            style={{
              justifyContent: 'center',
              padding: '10px 14px',
              fontSize: 13,
              marginBottom: 16,
              boxShadow: `0 6px 20px rgba(255,140,77,0.22)`,
            }}
          >
            <span style={{ fontSize: 14 }}>＋</span> New evaluation
          </Btn>

          <Eyebrow style={{ padding: '4px 10px', marginBottom: 4 }}>Workspace</Eyebrow>
          {NAV_ITEMS.map((item) => {
            const isActive = active === item.id;
            return (
              <NavLink
                key={item.id}
                to={item.path}
                end={item.path === '/'}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '7px 10px',
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
                }}
              >
                <span
                  style={{
                    fontSize: 12,
                    color: isActive ? E.ember : E.text3,
                    width: 14,
                  }}
                >
                  {item.icon}
                </span>
                <span style={{ flex: 1 }}>{item.label}</span>
              </NavLink>
            );
          })}

          {PINNED.length > 0 && (
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
          <button
            type="button"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '7px 10px',
              borderRadius: 6,
              fontSize: 12,
              color: E.text3,
              cursor: 'pointer',
              background: 'transparent',
              border: 'none',
              textAlign: 'left',
            }}
          >
            <span>⚙</span> Settings & keys
          </button>
          <button
            type="button"
            onClick={() => setDockOpen(!dockOpen)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '7px 10px',
              borderRadius: 6,
              fontSize: 12,
              color: E.text3,
              cursor: 'pointer',
              background: 'transparent',
              border: 'none',
              textAlign: 'left',
            }}
          >
            <span>◑</span> {dockOpen ? 'Hide' : 'Show'} co-pilot
          </button>
          <div
            style={{
              marginTop: 8,
              padding: '6px 10px',
              fontSize: 10,
              color: E.text4,
              fontFamily: E.fMono,
            }}
          >
            Local · 7401 · v2
          </div>
        </div>

        {/* MAIN CONTENT */}
        <div style={{ overflow: 'auto', background: E.ink }}>{children}</div>

        {/* CO-PILOT DOCK */}
        {showDock && <CoPilotDock onClose={() => setDockOpen(false)} />}
      </div>
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
