/**
 * TourMenu - always-visible header widget that surfaces the co-pilot
 * guidance state and gives the user direct control without diving into
 * Settings.
 *
 * Layout:
 *   [? Tours] button in the AppShell header -> click opens a small
 *   panel with:
 *     - Current state: "Guidance ON" / "Guidance OFF" with a toggle.
 *     - "Take a tour of this page" button (manually fires the tour
 *       associated with the current pathname; disabled if the current
 *       page has no tour).
 *     - "Reset all tour flags" button (clears every tour's completion
 *       flag so auto-tours can re-fire on the next visit).
 *
 * Closes on outside click and on Escape.
 */

import { useEffect, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { useV2Store, TOUR_ENABLED_KEY, tourCompletedKey } from '../store/store';
import { KNOWN_TOUR_IDS } from './useTour';
import { tourIdForPathname } from './routeTourMap';
import { E } from '../tokens';

function readEnabled(): boolean {
  try {
    return window.localStorage.getItem(TOUR_ENABLED_KEY) !== 'false';
  } catch {
    return true;
  }
}

export function TourMenu() {
  const [open, setOpen] = useState(false);
  const [enabled, setEnabled] = useState<boolean>(readEnabled);
  const [flash, setFlash] = useState<string | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const location = useLocation();
  const setTour = useV2Store((s) => s.setTour);

  const currentTourId = tourIdForPathname(location.pathname);

  useEffect(() => {
    if (!open) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (!wrapperRef.current) return;
      if (!wrapperRef.current.contains(e.target as Node)) setOpen(false);
    };
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEsc);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEsc);
    };
  }, [open]);

  const flashMessage = (msg: string) => {
    setFlash(msg);
    window.setTimeout(() => setFlash(null), 2000);
  };

  const toggleEnabled = () => {
    const next = !enabled;
    setEnabled(next);
    try {
      window.localStorage.setItem(TOUR_ENABLED_KEY, String(next));
    } catch {
      // localStorage unavailable; in-memory state still flips for this session.
    }
  };

  const startCurrentTour = () => {
    if (!currentTourId) return;
    setTour(currentTourId);
    setOpen(false);
  };

  const resetAll = () => {
    try {
      for (const id of KNOWN_TOUR_IDS) {
        window.localStorage.removeItem(tourCompletedKey(id));
      }
    } catch {
      // best-effort
    }
    flashMessage('Reset - tours will fire again on next visit');
  };

  return (
    <div ref={wrapperRef} style={{ position: 'relative' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={`Co-pilot guidance: ${enabled ? 'on' : 'off'}`}
        aria-label="Co-pilot tours"
        aria-expanded={open}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '4px 10px',
          borderRadius: 6,
          background: open ? E.panel2 : 'transparent',
          border: `1px solid ${E.hair2}`,
          color: E.text1,
          fontSize: 12,
          cursor: 'pointer',
          fontFamily: E.fSans,
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: enabled ? E.ember : E.text4,
            display: 'inline-block',
          }}
        />
        Tours
      </button>

      {open && (
        <div
          role="menu"
          style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            right: 0,
            zIndex: 900,
            minWidth: 280,
            background: E.panel,
            border: `1px solid ${E.hair}`,
            borderRadius: 10,
            boxShadow: '0 8px 24px rgba(20, 18, 14, 0.14)',
            padding: 12,
            fontFamily: E.fSans,
            color: E.text1,
          }}
        >
          {/* Toggle row */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
              padding: '4px 4px 10px',
              borderBottom: `1px solid ${E.hair}`,
            }}
          >
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>
                Co-pilot guidance
              </div>
              <div style={{ fontSize: 11, color: E.text3, marginTop: 2 }}>
                {enabled
                  ? 'First-visit tours will auto-fire on each tab.'
                  : 'No tours will fire automatically.'}
              </div>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={enabled}
              aria-label="Co-pilot guidance toggle"
              onClick={toggleEnabled}
              style={{
                width: 36,
                height: 20,
                borderRadius: 10,
                border: 'none',
                background: enabled ? E.ember : E.panel3,
                position: 'relative',
                cursor: 'pointer',
                transition: 'background 120ms',
                padding: 0,
                flexShrink: 0,
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  top: 2,
                  left: enabled ? 18 : 2,
                  width: 16,
                  height: 16,
                  borderRadius: '50%',
                  background: '#fff',
                  transition: 'left 120ms',
                }}
              />
            </button>
          </div>

          {/* Take a tour of this page */}
          <button
            type="button"
            onClick={startCurrentTour}
            disabled={!currentTourId}
            title={
              currentTourId
                ? 'Start the guided walk-through for this page'
                : 'No tour is available for this page'
            }
            style={{
              width: '100%',
              textAlign: 'left',
              padding: '10px 8px',
              borderRadius: 6,
              border: 'none',
              background: 'transparent',
              color: currentTourId ? E.text0 : E.text3,
              fontSize: 13,
              cursor: currentTourId ? 'pointer' : 'not-allowed',
              fontFamily: E.fSans,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginTop: 6,
            }}
            onMouseEnter={(e) => {
              if (currentTourId) e.currentTarget.style.background = E.panel2;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <span style={{ fontSize: 14 }}>▶</span>
            Take a tour of this page
          </button>

          {/* Reset */}
          <button
            type="button"
            onClick={resetAll}
            title="Clear every tour's completion flag so they fire again on next visit"
            style={{
              width: '100%',
              textAlign: 'left',
              padding: '10px 8px',
              borderRadius: 6,
              border: 'none',
              background: 'transparent',
              color: E.text2,
              fontSize: 13,
              cursor: 'pointer',
              fontFamily: E.fSans,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = E.panel2;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <span style={{ fontSize: 13 }}>↺</span>
            Reset all tour flags
          </button>

          {flash && (
            <div
              style={{
                marginTop: 8,
                padding: '6px 8px',
                fontSize: 11,
                color: E.pass,
                background: E.passDim,
                borderRadius: 6,
                fontFamily: E.fMono,
              }}
            >
              {flash}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
