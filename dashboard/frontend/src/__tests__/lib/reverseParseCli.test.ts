/**
 * Unit tests for the reverse-CLI parser used by the "paste raw" modal in
 * CliFormBody. Covers tokenisation, recognised flag shapes, and the
 * graceful-degradation path for unknown flags + cli-id mismatch.
 */

import { describe, expect, test } from 'vitest';
import { reverseParseCli, tokenizeCli } from '../../lib/reverseParseCli';
import type { CliSchema } from '../../types/catalog';

const schema: CliSchema = {
  id: 'run-eval',
  name: 'run-eval',
  group: 'Eval',
  blurb: 'Run an eval',
  params: [
    { name: 'dataset', kind: 'path', required: true },
    { name: 'workers', kind: 'number', default: 4 },
    { name: 'verbose', kind: 'bool', default: false },
    { name: 'label', kind: 'string' },
    { name: 'metric_ids', kind: 'multiselect', options: ['a', 'b', 'c'] },
  ],
};

describe('tokenizeCli', () => {
  test('respects double-quoted whitespace', () => {
    expect(tokenizeCli('evalyn run-eval --label "two words"')).toEqual([
      'evalyn',
      'run-eval',
      '--label',
      'two words',
    ]);
  });

  test('respects single-quoted whitespace', () => {
    expect(tokenizeCli("evalyn run-eval --label 'two words'")).toEqual([
      'evalyn',
      'run-eval',
      '--label',
      'two words',
    ]);
  });

  test('keeps quoted = inside --flag="value with spaces"', () => {
    expect(tokenizeCli('evalyn run-eval --label="two words"')).toEqual([
      'evalyn',
      'run-eval',
      '--label=two words',
    ]);
  });
});

describe('reverseParseCli', () => {
  test('happy path: dataset + numeric + boolean', () => {
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --dataset evals/spam.jsonl --workers 8 --verbose',
      schema,
    );
    expect(errors).toEqual([]);
    expect(values).toEqual({
      dataset: 'evals/spam.jsonl',
      workers: 8,
      verbose: true,
    });
  });

  test('quoted string with spaces', () => {
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --label "v2 staging run" --dataset x.jsonl',
      schema,
    );
    expect(errors).toEqual([]);
    expect(values.label).toBe('v2 staging run');
    expect(values.dataset).toBe('x.jsonl');
  });

  test('--flag=value form', () => {
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --dataset=evals/foo.jsonl --workers=2',
      schema,
    );
    expect(errors).toEqual([]);
    expect(values).toEqual({ dataset: 'evals/foo.jsonl', workers: 2 });
  });

  test('--no-flag sets boolean false', () => {
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --dataset x.jsonl --no-verbose',
      schema,
    );
    expect(errors).toEqual([]);
    expect(values.verbose).toBe(false);
  });

  test('unknown flag is skipped with a warning, known args still parsed', () => {
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --dataset x.jsonl --bogus 42 --workers 4',
      schema,
    );
    expect(values).toEqual({ dataset: 'x.jsonl', workers: 4 });
    expect(errors.some((e) => e.includes('--bogus'))).toBe(true);
  });

  test('cli-id mismatch returns error and does not parse args', () => {
    const { values, errors } = reverseParseCli(
      'evalyn list-runs --limit 10',
      schema,
    );
    expect(values).toEqual({});
    expect(errors[0]).toMatch(/list-runs/);
    expect(errors[0]).toMatch(/run-eval/);
  });

  test('multiselect accumulates repeated occurrences', () => {
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --dataset x.jsonl --metric-ids a --metric-ids b',
      schema,
    );
    expect(errors).toEqual([]);
    expect(values.metric_ids).toEqual(['a', 'b']);
  });

  test('--flag with empty quoted value keeps the empty string', () => {
    // Edge case: user explicitly clears a string flag via `--label ""`.
    // The tokenizer must emit an empty token rather than swallowing it,
    // and the parser must keep it (string kind allows empty).
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --dataset x.jsonl --label ""',
      schema,
    );
    expect(errors).toEqual([]);
    expect(values.dataset).toBe('x.jsonl');
    expect(values.label).toBe('');
  });

  test('--flag="" inline empty value keeps the empty string', () => {
    const { values, errors } = reverseParseCli(
      'evalyn run-eval --dataset x.jsonl --label=""',
      schema,
    );
    expect(errors).toEqual([]);
    expect(values.label).toBe('');
  });
});
