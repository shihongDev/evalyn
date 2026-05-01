/**
 * Unit tests for the ANSI SGR parser.
 *
 * Covers: plain text passthrough, basic + bright FG colors, reset,
 * empty-reset (`\x1b[m`), bold + faint attributes, multi-code combos,
 * unknown codes ignored, malformed escapes stripped, stripAnsi helper.
 */

import { describe, expect, test } from 'vitest';
import { parseAnsi, stripAnsi } from '../../lib/ansi';

describe('parseAnsi', () => {
  test('empty string yields empty array', () => {
    expect(parseAnsi('')).toEqual([]);
  });

  test('plain text yields a single uncolored span', () => {
    const spans = parseAnsi('hello world');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('hello world');
    expect(spans[0].color).toBeUndefined();
    expect(spans[0].bold).toBeUndefined();
  });

  test('basic foreground color 31 (red)', () => {
    const spans = parseAnsi('\x1b[31merror\x1b[0m');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('error');
    expect(spans[0].color).toBe('var(--fail)');
  });

  test('basic foreground color 32 (green)', () => {
    const spans = parseAnsi('\x1b[32mok\x1b[0m');
    expect(spans[0].color).toBe('var(--pass)');
  });

  test('basic foreground color 33 (yellow)', () => {
    const spans = parseAnsi('\x1b[33mwarn\x1b[0m');
    expect(spans[0].color).toBe('var(--warn)');
  });

  test('all basic foreground codes 30-37 mapped', () => {
    for (let code = 30; code <= 37; code++) {
      const spans = parseAnsi(`\x1b[${code}mx\x1b[0m`);
      expect(spans).toHaveLength(1);
      expect(spans[0].color).toBeDefined();
    }
  });

  test('all bright foreground codes 90-97 mapped', () => {
    for (let code = 90; code <= 97; code++) {
      const spans = parseAnsi(`\x1b[${code}mx\x1b[0m`);
      expect(spans).toHaveLength(1);
      expect(spans[0].color).toBeDefined();
    }
  });

  test('reset 0 clears color', () => {
    const spans = parseAnsi('\x1b[31mred\x1b[0mplain');
    expect(spans).toHaveLength(2);
    expect(spans[0].color).toBe('var(--fail)');
    expect(spans[1].text).toBe('plain');
    expect(spans[1].color).toBeUndefined();
  });

  test('empty reset \\x1b[m clears color', () => {
    const spans = parseAnsi('\x1b[31mred\x1b[mplain');
    expect(spans).toHaveLength(2);
    expect(spans[0].color).toBe('var(--fail)');
    expect(spans[1].color).toBeUndefined();
  });

  test('default-color code 39 clears color but preserves bold', () => {
    const spans = parseAnsi('\x1b[1;31mred\x1b[39mbold\x1b[0m');
    expect(spans).toHaveLength(2);
    expect(spans[0].color).toBe('var(--fail)');
    expect(spans[0].bold).toBe(true);
    expect(spans[1].color).toBeUndefined();
    expect(spans[1].bold).toBe(true);
  });

  test('bold attribute (code 1)', () => {
    const spans = parseAnsi('\x1b[1mhello\x1b[0m');
    expect(spans[0].bold).toBe(true);
  });

  test('faint attribute (code 2)', () => {
    const spans = parseAnsi('\x1b[2mdim\x1b[0m');
    expect(spans[0].faint).toBe(true);
  });

  test('combined bold + color (multi-param)', () => {
    const spans = parseAnsi('\x1b[1;31mboom\x1b[0m');
    expect(spans[0].bold).toBe(true);
    expect(spans[0].color).toBe('var(--fail)');
  });

  test('code 22 disables bold and faint', () => {
    const spans = parseAnsi('\x1b[1mbold\x1b[22mnormal\x1b[0m');
    expect(spans).toHaveLength(2);
    expect(spans[0].bold).toBe(true);
    expect(spans[1].bold).toBeFalsy();
  });

  test('multiple color changes split into multiple spans', () => {
    const spans = parseAnsi('\x1b[31mred\x1b[32mgreen\x1b[34mblue\x1b[0m');
    expect(spans).toHaveLength(3);
    expect(spans[0].text).toBe('red');
    expect(spans[1].text).toBe('green');
    expect(spans[2].text).toBe('blue');
  });

  test('text before first escape is preserved uncolored', () => {
    const spans = parseAnsi('plain \x1b[31mred\x1b[0m');
    expect(spans).toHaveLength(2);
    expect(spans[0].text).toBe('plain ');
    expect(spans[0].color).toBeUndefined();
    expect(spans[1].text).toBe('red');
  });

  test('text after final reset is preserved uncolored', () => {
    const spans = parseAnsi('\x1b[31mred\x1b[0m and plain');
    expect(spans).toHaveLength(2);
    expect(spans[1].text).toBe(' and plain');
    expect(spans[1].color).toBeUndefined();
  });

  test('unknown SGR code is ignored without throwing', () => {
    const spans = parseAnsi('\x1b[999mtext\x1b[0m');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('text');
    expect(spans[0].color).toBeUndefined();
  });

  test('non-SGR CSI sequence is stripped silently', () => {
    // \x1b[2J is "clear screen" — drop the escape, keep surrounding text.
    const spans = parseAnsi('before\x1b[2Jafter');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('beforeafter');
  });

  test('truncated escape (no terminator) consumes rest of string', () => {
    // No final byte: the parser swallows the partial sequence rather
    // than emitting garbage. This matches xterm behavior.
    const spans = parseAnsi('hello\x1b[31');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('hello');
  });

  test('color persists across multiple chunks until reset', () => {
    const spans = parseAnsi('\x1b[31mone two three\x1b[0m');
    expect(spans).toHaveLength(1);
    expect(spans[0].text).toBe('one two three');
  });
});

describe('stripAnsi', () => {
  test('returns plain text unchanged', () => {
    expect(stripAnsi('hello')).toBe('hello');
  });

  test('removes color escapes', () => {
    expect(stripAnsi('\x1b[31mred\x1b[0m')).toBe('red');
  });

  test('removes multiple escape types', () => {
    expect(stripAnsi('\x1b[1;31mboom\x1b[0m\x1b[32m ok\x1b[0m')).toBe('boom ok');
  });

  test('handles empty input', () => {
    expect(stripAnsi('')).toBe('');
  });
});
