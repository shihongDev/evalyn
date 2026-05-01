/**
 * Unit tests for `buildCli` and `defaultValues`.
 *
 * Mirrors the mock's behavior 1:1 (see /tmp/evalyn-dashboard-mock/wb-cli-forms.jsx).
 */

import { describe, expect, test } from 'vitest';
import { buildCli, defaultValues } from '../views/buildCli';
import type { CliSchema } from '../types/catalog';

const cli = (params: CliSchema['params']): CliSchema => ({
  id: 'demo',
  name: 'demo',
  group: 'Eval',
  blurb: 'demo',
  params,
});

describe('defaultValues', () => {
  test('uses param.default when present', () => {
    const c = cli([{ name: 'workers', kind: 'number', default: 4 }]);
    expect(defaultValues(c)).toEqual({ workers: 4 });
  });

  test('bool with no default -> false', () => {
    const c = cli([{ name: 'verbose', kind: 'bool' }]);
    expect(defaultValues(c)).toEqual({ verbose: false });
  });

  test('multiselect with no default -> []', () => {
    const c = cli([{ name: 'tags', kind: 'multiselect', options: ['a', 'b'] }]);
    expect(defaultValues(c)).toEqual({ tags: [] });
  });

  test('string with no default -> empty string', () => {
    const c = cli([{ name: 'name', kind: 'string' }]);
    expect(defaultValues(c)).toEqual({ name: '' });
  });
});

describe('buildCli - empty / default values omitted', () => {
  test('no params -> bare command', () => {
    expect(buildCli(cli([]), {})).toBe('evalyn demo');
  });

  test('empty string omitted', () => {
    const c = cli([{ name: 'name', kind: 'string' }]);
    expect(buildCli(c, { name: '' })).toBe('evalyn demo');
  });

  test('null and undefined omitted', () => {
    const c = cli([
      { name: 'a', kind: 'string' },
      { name: 'b', kind: 'string' },
    ]);
    expect(buildCli(c, { a: null, b: undefined })).toBe('evalyn demo');
  });

  test('empty multiselect array omitted', () => {
    const c = cli([{ name: 'tags', kind: 'multiselect', options: ['a'] }]);
    expect(buildCli(c, { tags: [] })).toBe('evalyn demo');
  });

  test('value equal to default omitted', () => {
    const c = cli([{ name: 'workers', kind: 'number', default: 4 }]);
    expect(buildCli(c, { workers: 4 })).toBe('evalyn demo');
  });

  test('string equal to default omitted', () => {
    const c = cli([{ name: 'name', kind: 'string', default: 'foo' }]);
    expect(buildCli(c, { name: 'foo' })).toBe('evalyn demo');
  });
});

describe('buildCli - bool toggle', () => {
  test('true with default false -> --flag', () => {
    const c = cli([{ name: 'verbose', kind: 'bool', default: false }]);
    expect(buildCli(c, { verbose: true })).toBe('evalyn demo --verbose');
  });

  test('false with default true -> --no-flag', () => {
    const c = cli([{ name: 'cache', kind: 'bool', default: true }]);
    expect(buildCli(c, { cache: false })).toBe('evalyn demo --no-cache');
  });

  test('bool equal to default -> omitted', () => {
    const c = cli([
      { name: 'a', kind: 'bool', default: false },
      { name: 'b', kind: 'bool', default: true },
    ]);
    expect(buildCli(c, { a: false, b: true })).toBe('evalyn demo');
  });

  test('underscored bool name kebab-cased', () => {
    const c = cli([{ name: 'dry_run', kind: 'bool', default: false }]);
    expect(buildCli(c, { dry_run: true })).toBe('evalyn demo --dry-run');
    const c2 = cli([{ name: 'use_cache', kind: 'bool', default: true }]);
    expect(buildCli(c2, { use_cache: false })).toBe('evalyn demo --no-use-cache');
  });
});

describe('buildCli - multiselect', () => {
  test('space-joined values (matches argparse nargs="*")', () => {
    const c = cli([{ name: 'tags', kind: 'multiselect', options: ['a', 'b', 'c'] }]);
    expect(buildCli(c, { tags: ['a', 'c'] })).toBe('evalyn demo --tags a c');
  });

  test('underscores kebab-cased', () => {
    const c = cli([{ name: 'metric_ids', kind: 'multiselect', options: ['x', 'y'] }]);
    expect(buildCli(c, { metric_ids: ['x'] })).toBe('evalyn demo --metric-ids x');
  });
});

describe('buildCli - string quoting', () => {
  test('value with space gets quoted', () => {
    const c = cli([{ name: 'name', kind: 'string' }]);
    expect(buildCli(c, { name: 'hello world' })).toBe('evalyn demo --name "hello world"');
  });

  test('value with tab gets quoted', () => {
    const c = cli([{ name: 'name', kind: 'string' }]);
    expect(buildCli(c, { name: 'a\tb' })).toBe('evalyn demo --name "a\tb"');
  });

  test('plain value not quoted', () => {
    const c = cli([{ name: 'name', kind: 'string' }]);
    expect(buildCli(c, { name: 'hello' })).toBe('evalyn demo --name hello');
  });

  test('underscores kebab-cased', () => {
    const c = cli([{ name: 'judge_model', kind: 'string' }]);
    expect(buildCli(c, { judge_model: 'gpt' })).toBe('evalyn demo --judge-model gpt');
  });
});

describe('buildCli - number / select / path / long-text', () => {
  test('number non-default emitted', () => {
    const c = cli([{ name: 'workers', kind: 'number', default: 4 }]);
    expect(buildCli(c, { workers: 8 })).toBe('evalyn demo --workers 8');
  });

  test('zero is preserved when default differs', () => {
    const c = cli([{ name: 'workers', kind: 'number', default: 4 }]);
    expect(buildCli(c, { workers: 0 })).toBe('evalyn demo --workers 0');
  });

  test('select value emitted', () => {
    const c = cli([
      { name: 'judge', kind: 'select', options: ['gpt', 'claude'] },
    ]);
    expect(buildCli(c, { judge: 'claude' })).toBe('evalyn demo --judge claude');
  });

  test('path with space quoted', () => {
    const c = cli([{ name: 'dataset', kind: 'path' }]);
    expect(buildCli(c, { dataset: '/a path/x' })).toBe(
      'evalyn demo --dataset "/a path/x"',
    );
  });

  test('long-text value emitted', () => {
    const c = cli([{ name: 'note', kind: 'long-text' }]);
    expect(buildCli(c, { note: 'short' })).toBe('evalyn demo --note short');
  });
});

describe('buildCli - composition', () => {
  test('multiple flags joined with single space', () => {
    const c = cli([
      { name: 'workers', kind: 'number', default: 4 },
      { name: 'verbose', kind: 'bool', default: false },
      { name: 'name', kind: 'string' },
      { name: 'tags', kind: 'multiselect', options: ['a', 'b'] },
    ]);
    const out = buildCli(c, {
      workers: 8,
      verbose: true,
      name: 'foo',
      tags: ['a', 'b'],
    });
    expect(out).toBe('evalyn demo --workers 8 --verbose --name foo --tags a b');
  });

  test('mixed defaults preserved', () => {
    const c = cli([
      { name: 'workers', kind: 'number', default: 4 },
      { name: 'verbose', kind: 'bool', default: false },
    ]);
    // workers stays at default, verbose flipped
    expect(buildCli(c, { workers: 4, verbose: true })).toBe(
      'evalyn demo --verbose',
    );
  });
});
