<claude-instructions>

<python>
  Use uv for everything if necessary: uv run, uv pip, uv venv.
</python>


<style>No emojis. No em dashes - use hyphens or colons instead.</style>

<branching>When you start to build a new feature, please create a new branch. Once you finish, test and make sure it is ok, then merge the branch and clean</branching>

<code-simplifier>After you add new codes, at the end of the implementation, you should run the code simplifier</code-simplifier>

<test>Everytime you change the functionalities of CLI, please review its corresponding tests and see if you need to make any change. After making all the changes, please review if any tests are failing and correct your code until the test pass.
</test>

<plan> 
Make the plan extreme concise. Sacrifice grammar for the sake of concise. At the end of each plan, give me a list of unsolved questions to answer, if any. 
</plan>

<epistemology>
  Assumptions are the enemy. Never guess numerical values - benchmark instead of estimating.
  When uncertain, measure. Say "this needs to be measured" rather than inventing statistics.
</epistemology>

<scaling>
  Validate at small scale before scaling up. Run a sub-minute version first to verify the
  full pipeline works. When scaling, only the scale parameter should change.
</scaling>

<interaction>
  Clarify unclear requests, then proceed autonomously. Only ask for help when scripts timeout
  (>2min), sudo is needed, or genuine blockers arise.
</interaction>

<ground-truth-clarification>
  For non-trivial tasks, reach ground truth understanding before coding. Simple tasks execute
  immediately. Complex tasks (refactors, new features, ambiguous requirements) require
  clarification first: research codebase, ask targeted questions, confirm understanding,
  persist the plan, then execute autonomously.
</ground-truth-clarification>

<spec-driven-development>
  When starting a new project, after compaction, or when SPEC.md is missing/stale and
  substantial work is requested: invoke /spec skill to interview the user. The spec persists
  across compactions and prevents context loss. Update SPEC.md as the project evolves.
  If stuck or losing track of goals, re-read SPEC.md or re-interview.
</spec-driven-development>

<first-principles-reimplementation>
  Building from scratch can beat adapting legacy code when implementations are in wrong
  languages, carry historical baggage, or need architectural rewrites. Understand domain
  at spec level, choose optimal stack, implement incrementally with human verification.
</first-principles-reimplementation>

<constraint-persistence>
  When user defines constraints ("never X", "always Y", "from now on"), immediately persist
  to project's local CLAUDE.md. Acknowledge, write, confirm.
</constraint-persistence>

<edge-cases>
  Before claiming a feature done, enumerate edge cases and write tests for them: empty input,
  single element, None, duplicates, concurrent access, very small (n=1,2,3) AND very large
  (n>=10^4) sizes. "Tests pass" only counts when tests cover boundaries, not just typical
  values. A p95 test with n=100 is not a p95 test - it must also pass at n=3.
</edge-cases>

<reuse-audit>
  When reusing existing infrastructure (parsers, exporters, ContextVars, helpers), audit
  every load-bearing assumption: does the existing code actually support your new use case?
  "Looks right" routinely hides implicit limitations - parsers that do not recurse,
  ContextVars that default to None, exporters that silently truncate at fixed sizes.
  Verify, do not infer.
</reuse-audit>

<parallel-paths>
  Cross-cutting concerns (cost tracking, sampling, redaction, error handling, telemetry)
  must touch ALL execution paths, not just the visible one. Before declaring such a feature
  done, grep for parallel paths: regular vs unit-based runners, sync vs async, sequential vs
  parallel strategies, every provider instrumentor, every CLI subcommand. Missing one path
  produces a silent inert feature.
</parallel-paths>

<trust-boundaries>
  Treat user-supplied data (file contents, env vars, network responses, dataset fields,
  webhook URLs) as untrusted. Mark the boundary explicitly in code. Sanitize at boundary.
  Never embed untrusted text directly into LLM prompts without delimiters plus explicit
  instructions to treat it as data. Never log secrets - webhook URLs, API keys, tokens -
  even inside error messages or stack traces.
</trust-boundaries>

<atomic-persistence>
  Persistent state writes use the temp-file + os.replace() pattern. Plain open("w") is
  acceptable only for caches that can be regenerated. JSON catalogs, baselines, configs,
  lock files: always atomic. Reads of these files should distinguish "corrupted JSON" from
  "I/O error" and not silently return empty defaults on both.
</atomic-persistence>

<loud-failure>
  Loud failure beats silent silence. Do not swallow exceptions to "continue gracefully" -
  catch the specific type you expect and let the rest propagate. Do not silently drop
  malformed records - warn, count, surface in summary. Do not overwrite files without
  --force. Do not return empty results to mask errors. Bare except Exception: pass is a bug.
</loud-failure>

<magic-numbers>
  Timeouts, size limits, retry counts, truncation lengths must cite their source in a
  comment (vendor docs, benchmark, RFC) or be configurable via a flag. "5 second timeout",
  "500 character truncation", "3000 character Slack block": every unsourced number is a
  future bug.
</magic-numbers>

<unit-validation>
  Flags accepting numeric input (--threshold, --rate, --percentage, --cost, --duration)
  must validate magnitude against the documented unit and reject obvious mis-units with a
  clear error. If --threshold is a fraction (0.05 = 5%), reject --threshold 5 with a hint
  about the expected scale, do not silently accept 500%.
</unit-validation>

</claude-instructions>