# CLI Rich Output Design

Unified visual system for all 34 evalyn CLI commands using box-drawing
characters, semantic icons, consistent section headers, and color-coded
status indicators.

## Rendering Primitives

New module: `sdk/evalyn_sdk/cli/utils/rich.py`

### banner(title: str, width: int = 60)

Double-line box for top-level command title:

    +===========================================================+
    |  EVALUATION ANALYSIS                                       |
    +===========================================================+

Uses Unicode box-drawing: top/bottom with double lines, sides with
double pipes.

### section(title: str, width: int = 55)

Thin-line separator for sub-sections:

    -- METRIC SUMMARY ------------------------------------------

Uses em-dash style line with title inset.

### table(headers, rows, align=None)

Box-drawing table with configurable column alignment:

    +----------+-------------+----------+--------+----------+
    | ID       | Function    | Project  | Status | Duration |
    +----------+-------------+----------+--------+----------+
    | 53bf0e9b | test_grok   | xai-test |   OK   |    1.5s  |
    | ad40738c | test_grok   | xai-test |   OK   |    2.0s  |
    +----------+-------------+----------+--------+----------+

- Auto-calculates column widths from content
- Alignment per column: left (default), right, center
- Strips ANSI codes for width calculation
- Max column width clamped at terminal width / num_columns

### kv(pairs: list[tuple[str, str]], indent: int = 2)

Key-value block with aligned colons:

      Run:      cd347c59
      Dataset:  my-dataset (20 items)
      Created:  2026-03-30 01:25

### footer(commands: list[str])

Hint footer with triangle bullets:

    ---------------------------------------------------------
     > evalyn show-call --id 53bf0e9b

### progress_bar(value, total, width=20) -> str

Inline bar returning a string:

    Overall: ============-------- 69%  NEEDS ATTENTION

### Icons (via icon function or constants)

    PASS  = green checkmark
    FAIL  = red X
    WARN  = yellow tilde
    INFO  = blue bullet
    NEXT  = triangle bullet (for commands)

All icons respect NO_COLOR and non-TTY: fall back to text labels
[PASS], [FAIL], [WARN], etc.

## Color System

Reuses existing `colors.py`. Semantic mapping:

- green: pass, success, positive delta
- red: fail, error, negative delta, regression
- yellow: warning, partial, near-zero delta
- blue: headers, info bullets
- cyan: secondary info, metadata
- dim: less important text, IDs
- bold: emphasis, titles

## Per-Command Design

### List Commands (tabular)

All use: banner + table + footer

- list-calls: columns ID, Function, Project, Status, Duration, Started
  - Drop: version, sim, file, ended_at, duration_ms (use human duration)
  - Status gets icon: checkmark for OK, X for ERROR
  - Timestamps trimmed to YYYY-MM-DD HH:MM
  - Pagination line: "3 of 103 shown"
- show-projects: columns Project, Calls, Errors, Last Active
  - Sort by calls descending (most active first)
- list-metrics: banner "OBJECTIVE METRICS (73)" + table
  - Drop config column (noisy). Add --verbose to show it.
  - Separate section header for "SUBJECTIVE METRICS (60)"
- list-runs: columns ID, Name, Items, Results, Created
- list-calibrations: columns Metric, Optimizer, Score, Created

### Detail Commands (single entity)

All use: banner + kv + sections

- show-call: banner "CALL DETAILS" + kv block + sections for
  INPUT, OUTPUT, SPAN TREE, EVENTS (table)
- show-trace: banner "TRACE: <name> (<dur>) icon" + tree + summary
- show-span: banner "SPAN DETAILS" + kv + content sections
- show-run: banner "EVAL RUN" + kv + metric summary (like analyze)

### Analysis Commands (reports)

All use: banner + kv + multiple sections + footer

- analyze: banner "EVALUATION ANALYSIS" + kv (run info) +
  section METRIC SUMMARY (aligned list with icons + progress bar) +
  section KEY FINDINGS (bullet list) +
  section NEXT STEPS (footer-style commands)
- compare: banner "RUN COMPARISON" + kv (baseline/current) +
  section METRIC CHANGES (with delta colors) +
  section REGRESSION ALERTS
- insights: banner "EVALYN INSIGHTS" + sections DIAGNOSTICS,
  RECOMMENDATIONS (numbered with category tags + commands)
- trend: banner "EVALUATION TREND" + section per metric with
  run-over-run values
- cluster-failures: banner "FAILURE CLUSTERS" + section per metric
  with pattern descriptions
- cluster-misalignments: same pattern as cluster-failures

### Status/Info Commands

- status: banner "DATASET STATUS" + kv (dataset info) +
  section PIPELINE (checklist with icons) +
  section NEXT STEP (single command)
- validate: banner "VALIDATION" + kv + field table +
  section WARNINGS + section RESULT (icon VALID/INVALID)
- workflow: banner "EVALYN WORKFLOW" + 3 phase sections with
  numbered steps and command bullets
- quickstart: banner "EVALYN QUICKSTART" + numbered steps
- annotation-stats: banner "ANNOTATION COVERAGE" + kv + progress bar

### Action Commands (do work + report)

- run-eval: spinner during execution, then analyze-style summary
- build-dataset: progress bar, then status-style kv summary
- suggest-metrics: spinner, then bullet list of suggestions
- calibrate: progress bar, then result kv
- export: single confirmation line with icon
- export-for-annotation: single confirmation line
- simulate: progress bar, then summary kv
- import-annotations: confirmation line with count
- annotate: interactive (no change to interaction, but wrap
  summary in banner)
- select-metrics: interactive (same approach)
- delete-traces: confirmation with icon
- dashboard: confirmation with path
- init: confirmation with path
- one-click: combines sub-command outputs (each uses its own
  rendering)

## Non-TTY / JSON Mode

- All box-drawing and icons disabled when stdout is not a TTY
- Falls back to plain text with [PASS]/[FAIL] labels
- JSON mode (--format json) unchanged: raw JSON, no chrome
- Compact mode (--compact) unchanged: single-line CI format

## Existing Utils

- colors.py: keep as-is, used by rich.py
- formatters.py: keep print_table for backward compat, commands
  migrate to rich.table
- compact.py: keep as-is for CI mode
- hints.py: print_hint stays, but commands that adopt rich.py use
  footer() instead
- ui.py: Spinner and ProgressIndicator stay as-is
- errors.py: fatal_error and warning stay as-is

## Implementation Order

1. Build rich.py with all primitives + tests
2. Update list commands (list-calls, show-projects, list-metrics,
   list-runs, list-calibrations)
3. Update detail commands (show-call, show-trace, show-span, show-run)
4. Update analysis commands (analyze, compare, insights, trend,
   cluster-failures, cluster-misalignments)
5. Update status/info commands (status, validate, workflow,
   quickstart, annotation-stats)
6. Update action commands (remaining)
7. Run full test suite, fix breakages
