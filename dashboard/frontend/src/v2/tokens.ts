/**
 * Evalyn v2 design tokens.
 *
 * Ported from /tmp/evalyn-v2/design-system.jsx (the designer's canvas
 * source of truth). Cream-on-paper palette, single ember accent, steel
 * counterpoint. Light-only - no dark variant in v2.
 *
 * Round-trip rule: if the designer ships a new design-system.jsx, this
 * file is the only place to update. All UI primitives import from here.
 */

export const E = {
  // Surfaces - warm paper-cream canvas, layered panels
  ink: '#f3eee4',
  panel: '#fbf7ee',
  panel2: '#f0e9da',
  panel3: '#e6dec9',
  panel4: '#d6cdb4',
  hair: '#e6dec9',
  hair2: '#d6cdb4',

  // Type - dark charcoal hierarchy
  text0: '#1a1812',
  text1: '#3a352b',
  text2: '#6b6557',
  text3: '#94907f',
  text4: '#bcb6a3',

  // Single ember accent + steel counterpoint
  ember: '#d96a2c',
  emberDim: 'rgba(217,106,44,0.10)',
  emberRim: 'rgba(217,106,44,0.32)',
  emberInk: '#fff8f1',
  steel: '#3d6b9c',
  steelDim: 'rgba(61,107,156,0.12)',

  // Semantics - desaturated for light surfaces
  pass: '#3d8e4f',
  passDim: 'rgba(61,142,79,0.10)',
  fail: '#c14a3a',
  failDim: 'rgba(193,74,58,0.10)',
  warn: '#a87a1f',
  warnDim: 'rgba(168,122,31,0.12)',
  info: '#2c8090',
  infoDim: 'rgba(44,128,144,0.10)',

  // Type stacks
  fSans: "'Geist', system-ui, sans-serif",
  fMono: "'Geist Mono', ui-monospace, monospace",
  fSerif: "'Instrument Serif', Georgia, serif",
} as const;

export type EToken = typeof E;
