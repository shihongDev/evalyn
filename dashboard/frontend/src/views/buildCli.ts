/**
 * buildCli + defaultValues helpers.
 *
 * Ported 1:1 from /tmp/evalyn-dashboard-mock/wb-cli-forms.jsx.
 *
 * Rules (preserved from mock):
 *  - Skip empty / null / undefined / empty-array values.
 *  - Skip values equal to the default (so default flags do not appear in the
 *    rendered command). Bools have their own rule (see below).
 *  - bool true   -> `--flag`         when the default is false.
 *  - bool false  -> `--no-flag`      when the default is true.
 *  - bool == default -> omitted.
 *  - multiselect -> `--flag v1,v2`.
 *  - string with whitespace -> quoted.
 *  - underscores in param.name converted to hyphens for the flag.
 */

import type { CliSchema, ParamSchema } from '../types/catalog';

/** Form values keyed by `param.name`. */
export type CliFormValues = Record<string, unknown>;

const flagName = (name: string): string => name.replace(/_/g, '-');

/** Initialize form state from a CLI's defaults. */
export function defaultValues(cli: CliSchema): CliFormValues {
  const v: CliFormValues = {};
  for (const p of cli.params) {
    if (p.default !== undefined) {
      v[p.name] = p.default;
    } else if (p.kind === 'bool') {
      v[p.name] = false;
    } else if (p.kind === 'multiselect') {
      v[p.name] = [];
    } else {
      v[p.name] = '';
    }
  }
  return v;
}

/** Assemble the equivalent shell command string. */
export function buildCli(cli: CliSchema, values: CliFormValues): string {
  const flags: string[] = [];
  for (const p of cli.params) {
    const v = values[p.name];
    if (v === undefined || v === null || v === '' || (Array.isArray(v) && v.length === 0)) {
      continue;
    }
    if (p.kind === 'bool') {
      const def = !!p.default;
      const cur = !!v;
      if (cur !== def) {
        flags.push(cur ? `--${flagName(p.name)}` : `--no-${flagName(p.name)}`);
      }
      continue;
    }
    if (p.kind === 'multiselect') {
      const arr = v as unknown[];
      flags.push(`--${flagName(p.name)} ${arr.join(',')}`);
      continue;
    }
    if (typeof v === 'number' && v === p.default) {
      continue;
    }
    if (v === p.default) {
      continue;
    }
    const s = String(v);
    const val = /\s/.test(s) ? `"${s}"` : s;
    flags.push(`--${flagName(p.name)} ${val}`);
  }
  return `evalyn ${cli.id}${flags.length ? ' ' + flags.join(' ') : ''}`;
}

/** True if a param is "filled" (used by the cost/duration heuristic only). */
export const isFilled = (_p: ParamSchema, v: unknown): boolean =>
  v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && v.length === 0);
