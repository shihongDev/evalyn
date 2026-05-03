/**
 * CliCatalog types.
 *
 * Hand-written to match the JSON Schema at
 * `dashboard/schema/catalog.schema.json`. Both files are sources of truth and
 * must agree: a Phase 1 unit test (lane A2) validates the introspector's
 * output against the schema and its TypeScript counterpart.
 *
 * If you change one, change the other.
 */

export type ParamKind =
  | 'bool'
  | 'string'
  | 'number'
  | 'select'
  | 'multiselect'
  | 'path'
  | 'long-text';

export interface ParamSchema {
  /** argparse dest for the argument. */
  name: string;
  /** Discriminator the form renderer uses to pick a control. */
  kind: ParamKind;
  /** Default value (any JSON type). */
  default?: unknown;
  /** Allowed values for select / multiselect. */
  options?: string[];
  /** Argparse help string. */
  help?: string | null;
  /** Whether the argument must be supplied. */
  required?: boolean;
  /** Hide behind an "Advanced" disclosure in the form view. */
  advanced?: boolean;
  /**
   * Render in the default form alongside required params even though
   * argparse marks it optional. Curated per CLI module via the
   * `ESSENTIAL = {...}` constant.
   */
  essential?: boolean;
  /**
   * Optional numeric range hint for `kind === 'number'`. When present the
   * dashboard renders a slider alongside the numeric input. Curated per CLI
   * module via the `RANGES = {...}` constant or auto-extracted from the
   * argparse help string.
   */
  range?: {
    min: number;
    max: number;
    step?: number;
  };
  /**
   * Optional unit label for `kind === 'number'`. Rendered as a suffix next to
   * the input (e.g. "workers", "seconds"). Best-effort extraction from the
   * argparse help string with per-CLI overrides.
   */
  unit?: string;
}

export interface CliSchema {
  /** Unique CLI subcommand id (e.g. "run-eval"). */
  id: string;
  /** Human-readable name for display. */
  name: string;
  /** Category bucket (Tracing, Dataset, Metrics, Eval, Analysis, ...). */
  group: string;
  /** One-line description from the argparse parser. */
  blurb: string;
  /** Argument list. */
  params: ParamSchema[];
}

export type CliCatalog = CliSchema[];
