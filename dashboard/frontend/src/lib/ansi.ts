/**
 * Tiny ANSI SGR parser for the Terminal component.
 *
 * Scope: just enough to colorize streamed CLI output. Recognizes the SGR
 * escape sequence `\x1b[<codes>m` for:
 *   - Reset:                0  (or empty `\x1b[m`)
 *   - Foreground basic:     30-37
 *   - Foreground bright:    90-97
 *   - Foreground default:   39
 *   - Bold:                 1  -> mapped to fontWeight: 600
 *   - Faint:                2  -> mapped to opacity: 0.7
 *
 * Out of scope (intentional): backgrounds, 256/24-bit color, italic,
 * underline, cursor moves, screen clears, mouse, OSC, etc. Anything
 * unrecognized is dropped silently so the output stays clean.
 *
 * The parser returns a flat array of `AnsiSpan`s ready for React render
 * (one `<span>` per element). All other escapes pass through as the raw
 * text stripped of the escape sequence so users still see content.
 */

/**
 * Mapping of basic foreground codes to CSS color values. We use the
 * project's CSS variables so the palette stays theme-aware (defined in
 * `src/styles/index.css`). Bright variants reuse the same vars; the
 * theme already has enough contrast.
 */
const FG_COLORS: Record<number, string> = {
  30: 'var(--text-3)',     // black -> muted (terminal black is invisible on dark bg)
  31: 'var(--fail)',       // red
  32: 'var(--pass)',       // green
  33: 'var(--warn)',       // yellow
  34: '#7aa2f7',           // blue
  35: '#bb9af7',           // magenta
  36: '#7dcfff',           // cyan
  37: 'var(--text-1)',     // white -> default text
  // Bright (90-97): same palette, slightly brighter where it matters.
  90: 'var(--text-2)',
  91: 'var(--fail)',
  92: 'var(--pass)',
  93: 'var(--warn)',
  94: '#9eb8ff',
  95: '#d4b6ff',
  96: '#a4dfff',
  97: 'var(--text-0)',
};

/** A styled chunk emitted by the parser. */
export interface AnsiSpan {
  /** Visible text. Never contains escape sequences. */
  text: string;
  /** CSS color (resolved from FG_COLORS) when set. */
  color?: string;
  /** True when the bold attribute was active. */
  bold?: boolean;
  /** True when the faint attribute was active. */
  faint?: boolean;
}

interface SgrState {
  color?: string;
  bold?: boolean;
  faint?: boolean;
}

const RESET: SgrState = {};

/**
 * Apply a single SGR code to the running state. Unknown codes are
 * ignored. `0` and empty params return a fresh reset state.
 */
function applyCode(state: SgrState, code: number): SgrState {
  if (code === 0) return { ...RESET };
  if (code === 1) return { ...state, bold: true };
  if (code === 2) return { ...state, faint: true };
  if (code === 22) return { ...state, bold: false, faint: false };
  if (code === 39) return { ...state, color: undefined };
  if (FG_COLORS[code] !== undefined) return { ...state, color: FG_COLORS[code] };
  return state;
}

/**
 * Parse a string containing zero or more SGR escapes into a list of
 * styled spans. Empty input yields an empty array.
 */
export function parseAnsi(input: string): AnsiSpan[] {
  if (!input) return [];
  const out: AnsiSpan[] = [];
  let state: SgrState = { ...RESET };
  let buf = '';
  let i = 0;
  const n = input.length;

  const flush = () => {
    if (!buf) return;
    out.push({ text: buf, ...state });
    buf = '';
  };

  while (i < n) {
    // Look for ESC [ ... m sequences. ESC = 0x1b.
    if (input.charCodeAt(i) === 0x1b && input[i + 1] === '[') {
      // Scan until terminator. We accept any final byte but only
      // interpret 'm' (SGR); others are silently dropped.
      let j = i + 2;
      while (j < n) {
        const c = input.charCodeAt(j);
        // Final byte: 0x40..0x7e per ECMA-48.
        if (c >= 0x40 && c <= 0x7e) break;
        j++;
      }
      const final = input[j];
      const params = input.slice(i + 2, j);
      if (final === 'm') {
        flush();
        if (params === '') {
          state = { ...RESET };
        } else {
          for (const part of params.split(';')) {
            const code = parseInt(part, 10);
            if (!Number.isNaN(code)) state = applyCode(state, code);
          }
        }
      }
      // Skip the whole sequence (incl. final byte) regardless.
      i = j + 1;
      continue;
    }
    buf += input[i];
    i++;
  }
  flush();
  return out;
}

/** Strip ANSI escapes without colorizing. Useful for plain-text logs. */
export function stripAnsi(input: string): string {
  return parseAnsi(input)
    .map((s) => s.text)
    .join('');
}
