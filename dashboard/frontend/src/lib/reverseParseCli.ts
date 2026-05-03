/**
 * reverseParseCli — parse a literal `evalyn ...` command into form values.
 *
 * Used by the CliFormBody "paste raw" modal so a user can paste a CLI
 * invocation and have the form fields pre-filled. The inverse of
 * `buildCli` (in views/buildCli.ts).
 *
 * Recognised token shapes:
 *   --flag value       (next token is the value)
 *   --flag=value       (single token, `=`-separated)
 *   --flag             (boolean → true)
 *   --no-flag          (boolean → false; only when schema has matching bool param)
 *
 * Quoted strings (`"..."` and `'...'`) are tokenised as a single value so
 * `--label "with spaces"` round-trips. Backslash escapes inside quotes are
 * passed through (keeps Windows paths intact).
 *
 * Unrecognised flags do NOT abort parsing — they are recorded in the
 * returned `errors` array as warnings so the caller can surface them in the
 * modal but still pre-fill what we CAN match.
 */

import type { CliSchema, ParamSchema } from '../types/catalog';

export interface ReverseParseResult {
  /** Form values keyed by `param.name`. Only contains successfully parsed args. */
  values: Record<string, unknown>;
  /** Soft errors / warnings. Empty on a clean parse. */
  errors: string[];
}

/* ---------- Tokenizer ----------------------------------------------- */

/**
 * Split `input` into tokens, respecting `"..."` and `'...'` quoted
 * substrings. Quoted whitespace stays inside the token; the surrounding
 * quotes are stripped. Backslash + quote inside a quoted region escapes the
 * quote without ending the region.
 */
export const tokenizeCli = (input: string): string[] => {
  const out: string[] = [];
  let buf = '';
  let quote: '"' | "'" | null = null;
  let inToken = false;

  for (let i = 0; i < input.length; i++) {
    const c = input[i];
    if (quote) {
      if (c === '\\' && i + 1 < input.length && input[i + 1] === quote) {
        buf += quote;
        i++;
        continue;
      }
      if (c === quote) {
        quote = null;
        // Stay inside the same token — `--flag="value with spaces"` should
        // produce a single token `--flag=value with spaces`.
        continue;
      }
      buf += c;
      continue;
    }
    if (c === '"' || c === "'") {
      quote = c;
      inToken = true;
      continue;
    }
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') {
      if (inToken) {
        out.push(buf);
        buf = '';
        inToken = false;
      }
      continue;
    }
    buf += c;
    inToken = true;
  }
  if (inToken) out.push(buf);
  return out;
};

/* ---------- Coercion ------------------------------------------------ */

/** Convert a raw string value into the type implied by the param's `kind`. */
const coerceValue = (
  param: ParamSchema,
  raw: string,
): { value: unknown; ok: boolean } => {
  switch (param.kind) {
    case 'number': {
      const n = Number(raw);
      if (!Number.isFinite(n)) return { value: raw, ok: false };
      return { value: n, ok: true };
    }
    case 'bool': {
      // Only used for --flag=true / --flag=false; bare --flag handled at the
      // call site.
      const lc = raw.toLowerCase();
      if (lc === 'true' || lc === '1') return { value: true, ok: true };
      if (lc === 'false' || lc === '0') return { value: false, ok: true };
      return { value: raw, ok: false };
    }
    case 'multiselect': {
      // Comma-separated for repeated tokens. Caller pushes one at a time.
      return { value: raw.split(',').map((s) => s.trim()).filter(Boolean), ok: true };
    }
    default:
      return { value: raw, ok: true };
  }
};

/* ---------- Param lookup ------------------------------------------- */

/**
 * Convert a CLI-style flag (`--metric-id`) to the canonical argparse dest
 * (`metric_id`) used as `param.name`.
 */
const flagToName = (flag: string): string => flag.replace(/^--/, '').replace(/-/g, '_');

const findParam = (schema: CliSchema, flag: string): ParamSchema | undefined =>
  schema.params.find((p) => p.name === flagToName(flag));

/* ---------- Main entry --------------------------------------------- */

export const reverseParseCli = (
  input: string,
  schema: CliSchema,
): ReverseParseResult => {
  const errors: string[] = [];
  const values: Record<string, unknown> = {};

  const tokens = tokenizeCli(input);
  if (tokens.length === 0) {
    errors.push('empty input');
    return { values, errors };
  }
  if (tokens[0] !== 'evalyn') {
    errors.push(`expected first token to be \`evalyn\`, got \`${tokens[0]}\``);
    return { values, errors };
  }
  if (tokens.length < 2) {
    errors.push('missing CLI subcommand after `evalyn`');
    return { values, errors };
  }
  if (tokens[1] !== schema.id) {
    errors.push(
      `command \`${tokens[1]}\` does not match the current form (\`${schema.id}\`)`,
    );
    return { values, errors };
  }

  // Walk remaining tokens.
  let i = 2;
  while (i < tokens.length) {
    const tok = tokens[i];

    if (!tok.startsWith('--')) {
      errors.push(`skipping positional token \`${tok}\``);
      i++;
      continue;
    }

    // Split `--flag=value` so the value is paired with the flag in one
    // iteration and we don't accidentally consume the next token.
    let flag = tok;
    let inlineValue: string | null = null;
    const eq = tok.indexOf('=');
    if (eq > 0) {
      flag = tok.slice(0, eq);
      inlineValue = tok.slice(eq + 1);
    }

    // --no-flag → boolean false (only meaningful when the schema actually
    // has a bool param of that name without the `no_` prefix).
    if (flag.startsWith('--no-')) {
      const trimmed = `--${flag.slice(5)}`;
      const param = findParam(schema, trimmed);
      if (param && param.kind === 'bool') {
        values[param.name] = false;
        i++;
        continue;
      }
      // Fall through — caller might have a literal `--no-foo` flag.
    }

    const param = findParam(schema, flag);
    if (!param) {
      errors.push(`unknown flag \`${flag}\` — skipped`);
      // If the user wrote `--foo bar` and we don't know `--foo`, also skip
      // `bar` so we don't mis-consume it as a positional.
      if (inlineValue === null && i + 1 < tokens.length && !tokens[i + 1].startsWith('--')) {
        i += 2;
      } else {
        i++;
      }
      continue;
    }

    if (param.kind === 'bool') {
      if (inlineValue !== null) {
        const { value, ok } = coerceValue(param, inlineValue);
        if (!ok) errors.push(`invalid bool value for \`${flag}\`: ${inlineValue}`);
        values[param.name] = ok ? value : true;
      } else {
        values[param.name] = true;
      }
      i++;
      continue;
    }

    // Non-bool: needs a value. Either inline (`--flag=value`) or the next
    // token (`--flag value`). If the next token is another flag, we treat
    // the value as missing.
    let raw: string | null = inlineValue;
    if (raw === null) {
      const next = tokens[i + 1];
      if (next === undefined || next.startsWith('--')) {
        errors.push(`missing value for \`${flag}\``);
        i++;
        continue;
      }
      raw = next;
      i += 2;
    } else {
      i++;
    }

    if (param.kind === 'multiselect') {
      const { value } = coerceValue(param, raw);
      const arr = Array.isArray(value) ? value : [raw];
      const existing = values[param.name];
      if (Array.isArray(existing)) {
        values[param.name] = [...existing, ...arr];
      } else {
        values[param.name] = arr;
      }
      continue;
    }

    const { value, ok } = coerceValue(param, raw);
    if (!ok) {
      errors.push(`failed to coerce value for \`${flag}\`: \`${raw}\``);
    }
    values[param.name] = value;
  }

  return { values, errors };
};
