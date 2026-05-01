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
