# Eval-Tool Competitive Scan

Date: 2026-05-01
Scope: 5 LLM evaluation/observability tools + 5 polish references, scoped to inform Evalyn Dashboard UX (a localhost IDE for running 35 evalyn CLIs and chatting with an agent that calls them).

Method note: I worked from public marketing pages, official docs, changelog posts, third-party reviews, and demo write-ups. I did not log into authenticated dashboards (no cookies imported), so screenshots are not attached. Where a finding comes from a marketing page or third-party review rather than live use, I label it inline.

---

## Executive summary

### 5 patterns Evalyn dashboard should steal

1. **Three-pane "playground" layout (Anthropic Console, Braintrust, Langfuse).** Left = config (prompt/system/params), center = output, right = settings/scoring. This is the canonical shape for "tweak inputs, see outputs" and Evalyn's "run a CLI command" surface is structurally identical: left = argparse form, center = stdout stream, right = run metadata + agent context. The mock already implies this; the spec should commit to it as a primary "command runner" mode.
2. **Side-by-side run comparison with green/red diff highlighting (Braintrust, Langfuse).** Score deltas in red/green at the column header, per-row diff with matching trial alignment, "order columns by regression" sort. Evalyn's `compare` CLI already produces this data; the dashboard should expose it as a first-class view, not a CLI-output dump.
3. **Tree/Timeline toggle on traces (Langfuse March 2025 redesign).** Same data, two presentations: hierarchical for "what called what," chronological for "where did time go." Evalyn doesn't yet have rich trace UX, but the underlying call graph from `get_calls_batch` is hierarchical and would benefit.
4. **Convert production trace -> eval dataset in one click (Braintrust).** Evalyn currently loads datasets from disk only. A "promote this run's failures into a new dataset" button closes the most painful loop in eval engineering.
5. **Command palette with grouped, searchable actions (Linear, Vercel, Stripe, Anthropic Console).** Evalyn has 35 commands. A Cmd+K palette is the single highest-leverage discoverability feature; it doubles as the entry point for the agent ("type a question, hit Tab to convert to an agent prompt"). The spec lists this as a v1 non-goal -- I'd push back on that.

### 3 patterns to deliberately avoid

1. **Hiding everything behind a "playground" workflow (Phoenix, Helicone).** These tools nudge first-time users into a sandbox before they have data, which feels like a demo rather than a tool. Evalyn users arrive with real data on disk -- skip the playground gate, show them their data immediately.
2. **Command palette + sidebar + dashboard cards + in-chat suggestions, all at once (Braintrust, LangSmith).** When everything is everywhere, nothing is anywhere. Pick two surfaces (sidebar nav + Cmd+K) and stop.
3. **"Custom dashboards" as the headline feature (Langfuse, Helicone).** Letting users build pivot tables in the UI is impressive but burns enormous design budget for a feature 5% of users will configure once. Evalyn's audience is engineers iterating on prompts; ship great defaults, not a chart builder.

---

## Per-tool deep dives (5 eval tools)

### 1. Braintrust (braintrust.dev)

**Source:** docs (`/docs/start/eval-ui`, `/docs/guides/experiments/interpret`), blog posts (`/blog/faster-experiments`, `/blog/stakeholder-trust-evals-observability`). Did not log in.

**Information architecture.** Icon-rail sidebar with Datasets, Playgrounds, Settings, plus Logs/Experiments/Prompts. The mental model is project -> {dataset, prompt, scorer, experiment, log}. Each is a top-level resource, not a sub-tab.

**Signature interactions.**
- *Playground*: 3-pane workspace. Left = model + prompt template (with `{{input}}` syntax), center = dataset rows + outputs, right = scorer panel with `+ Scorer` button. "+ Task" duplicates the prompt column to compare side-by-side. Run button executes against all rows.
- *Experiment table*: 4 layout modes -- List (default table), Grid (side-by-side outputs), Summary (big-number scores), Summary table (scores as rows). User chooses how dense they want it.
- *Diff mode*: Toggle in experiment view. Score regressions in red, improvements in green at column-header level. "Order columns by regression" sort surfaces the worst-degraded rows. Auto-detects matching inputs across experiments and aligns them as "Trials."
- *Custom trace view*: Replaces raw JSON spans with domain-specific cards (e.g., a customer-support ticket card with badges, score gauges, cost in footer). Configurable per project.
- *Loop AI*: An agent that analyzes results, identifies patterns, suggests prompt/scorer improvements. Lives inside the experiment view, not as a separate chat.

**What they do well.**
- Diff-mode is the gold standard for run comparison. Aligned trials + per-row diff + colored score deltas is the right answer.
- Trace -> dataset (one click promote a production failure to a regression test) is the killer workflow.
- "Order by regression" is one knob that solves "where did I get worse?" instantly.

**What they do poorly.**
- Too many surface areas. Logs, experiments, datasets, playgrounds, prompts, scorers all top-level -- onboarding requires explaining six concepts before users can run anything.
- Custom trace view is powerful but requires writing config, which most teams won't.

**Borrow for Evalyn:** the diff-mode pattern, the "order by regression" sort, the trace-to-dataset promotion button. Skip the multi-resource sidebar -- Evalyn's structure is simpler (project -> runs).

### 2. LangSmith (smith.langchain.com)

**Source:** docs (`docs.langchain.com/langsmith`), marketing page (`langchain.com/langsmith`), third-party Medium write-ups. Did not log in.

**Information architecture.** Tabbed top nav: Tracing Projects, Datasets, Prompts, Evaluators, Annotation Queues, Monitoring. Inside a project: Traces tab is the default landing.

**Signature interactions.**
- *Trace viewer*: Click trace -> hierarchical tree on the left (root span -> nested children, e.g. `ResearchAgent -> LangGraph -> [router, retriever, grader, answer, formatter]`). Click any node -> right panel shows inputs, outputs, metrics, latency, tokens, cost. Standard split-pane pattern. (From third-party review, not live.)
- *Insights Agent*: An auto-clustering view that runs over your traces, groups them by topic and failure mode, generates an executive summary. Sits as a tab inside a project.
- *Dashboards*: Customizable widgets for token usage, latency P50/P99, error rates, cost. Webhook + PagerDuty alerts when thresholds breach.
- *Annotation Queues*: Reviewer workflow -- a queue of traces routed to humans for scoring.

**What they do well.**
- Insights Agent is conceptually similar to Evalyn's `evalyn insights` command. Auto-clustering of failure modes is high-value.
- Trace tree on left + detail on right is the canonical pattern for any nested call graph.

**What they do poorly.**
- Dashboard customization is a maze; defaults are weak so users have to build their own.
- "Deeply integrated with LangChain" feels like a pro until you're not on LangChain.

**Borrow for Evalyn:** split-pane trace viewer (tree left, detail right) for any future call-graph view. Surface insights inline (already partly done -- KEY FINDINGS in `cmd_analyze`), don't hide them in a separate tab.

### 3. Arize Phoenix (phoenix.arize.com / arize.com/docs/phoenix)

**Source:** marketing page, docs root. No screenshots accessed.

**Information architecture.** Open-source, self-hosted. Top-level: Traces, Datasets & Experiments, Prompts (Playground), Evaluations.

**Signature interactions.**
- *Trace timeline*: Step-by-step view of a single run. Spans show model calls, retrieval, tool use. Click a span -> attribute panel (`llm.invocation_parameters.temperature`, token counts, etc.).
- *Datasets & Experiments*: An experiment is a task that runs over each dataset example, scored by attached evaluators. Compare experiments to track changes to prompts/models/retrieval.
- *Prompt Playground*: Optimize prompts, compare models, replay traced LLM calls.
- *Embedding clustering*: Visualizes datasets in 2D via UMAP/t-SNE to find semantically similar failures. (Phoenix's distinctive feature.)

**What they do well.**
- Embedding visualization is genuinely unique -- nobody else makes "find clusters of similar failures" visual.
- OpenTelemetry-native (no vendor lock-in).
- "Replay a traced LLM call in the playground" closes the debug loop.

**What they do poorly.**
- The product feels like a research tool first, product second. Defaults aren't opinionated.
- Heavy reliance on the user understanding tracing concepts upfront.

**Borrow for Evalyn:** the "replay this run with a tweak" pattern is a great fit for Evalyn's agent surface -- "rerun this command with `--limit 50` instead of `--limit 10`". Embedding clustering is interesting but probably out of scope for v1.

### 4. Helicone (helicone.ai)

**Source:** marketing page, docs (`/features/sessions`), changelog (`/changelog/20250506-smarter-sessions-insights`). Did not log in.

**Information architecture.** Sidebar with Dashboard, Requests, Segments, Sessions, Users, Prompts, Datasets, Playground, Rate Limits, Alerts. HQL (Helicone Query Language) for ad-hoc queries.

**Signature interactions.**
- *Sessions*: Multi-step workflows grouped by path syntax (e.g. `/abstract/outline/lesson-1`). Three views available -- Chat (conversation flow), Tree (hierarchy), Span (timeline). 2026 redesign added session-level metrics (avg latency, total cost) and smarter time filters.
- *Requests*: Flat log of every LLM call. Filter, segment, drill in.
- *HQL*: SQL-like query language for the analytics layer. Power users only.

**What they do well.**
- Three views (Chat / Tree / Span) on the same session data -- different cognitive frames for different debug questions.
- Path syntax for grouping requests is lightweight and doesn't require structural changes upstream.

**What they do poorly.**
- 10+ sidebar items with no obvious hierarchy. Feels like a tool that grew features faster than it grew IA.
- HQL is powerful but a tax on casual users.

**Borrow for Evalyn:** the multi-view-on-same-data pattern (tree + timeline + chat). Evalyn's agent transcripts could absolutely use a "show as conversation" vs "show as call tree" toggle.

### 5. Langfuse (langfuse.com)

**Source:** docs, changelog (`/changelog/2025-03-19-new-trace-view`, `/changelog/2025-07-28-playground-side-by-side`, `/changelog/2026-03-23-v4-dashboard-changes`).

**Information architecture.** Sidebar: Tracing, Sessions, Users, Prompts, Playground, Datasets, Evaluation, Dashboards, Settings. Open-source, self-hostable.

**Signature interactions.**
- *Trace view (March 2025 redesign)*: Tree/Timeline toggle, both views fully featured for metrics and scores. Toggleable visibility for scores, comments, metrics. Color coding for span types. Search by observation type, ID, or name.
- *Playground (July 2025 redesign)*: Side-by-side prompt comparison with parallel LLM execution. Run all variants at once or focus on one.
- *Prompt Experiments*: Dataset + prompt version + model -> run -> aggregated score in experiments table -> compare side-by-side. Optional LLM-as-judge for auto-scoring.
- *Dashboards (2026 v4)*: Observation-centric data model with custom widgets (line, bar, time series, pie). Pre-built Latency / Cost / Usage dashboards.

**What they do well.**
- Tree/Timeline toggle is the cleanest implementation of "two cognitive frames, one dataset" I've seen.
- Side-by-side playground with "run all at once" is the right interaction for prompt iteration.
- Pre-built dashboards mean users get value before configuring anything.

**What they do poorly.**
- v4 dashboard changes broke some existing widgets (per their own changelog) -- a sign of accumulating complexity in the analytics layer.
- Prompt management vs Experiments vs Playground vs Datasets is four overlapping mental models for "I want to iterate on this prompt."

**Borrow for Evalyn:** the Tree/Timeline toggle, side-by-side playground execution. Don't replicate the four-way split between prompt resources -- Evalyn doesn't need that complexity.

---

## Polish references (5 general tools)

### Linear (linear.app)
**Patterns to import:**
1. **Cmd+K command palette as the primary navigation surface.** Linear's palette is the keyboard-first power user's interface to literally every action. For Evalyn with 35 commands, this is non-negotiable -- a sidebar listing 35 things is a wall, but Cmd+K + fuzzy search + grouped results ("Datasets," "Runs," "Calibrate," "Annotate") makes 35 feel like 5.
2. **Single-letter shortcuts after the palette opens.** Linear uses `E` to assign, `M` to move, etc. Evalyn could use `R` for "run last command," `A` for "ask agent," `D` for "open dataset."

### Vercel dashboard (vercel.com)
**Patterns to import:**
1. **Streaming logs that start the moment the export/build kicks off** (Vercel changelog, Feb 2026: "your download starts immediately and you can continue to use the dashboard while the export runs in the background"). Evalyn's CLI runs are subprocesses streaming over WebSocket -- this needs to feel as responsive. No spinners on the whole page. Stream into a panel, let users keep working.
2. **Calm density.** Vercel's deployment list is small text, generous padding, monochrome until something needs attention (red for failed). Evalyn's run lists should follow this -- avoid colored badges for status that isn't actionable.

### Stripe dashboard (stripe.com)
**Patterns to import:**
1. **Three states for every dashboard component: Loading (skeleton matching the layout), Empty (illustration + one-sentence + CTA), Loaded.** From Stripe's design system docs. Evalyn's panels currently jump from "nothing" to "data" -- skeletons reduce perceived latency and the empty-state CTAs become the onboarding flow.
2. **Big-number + sparkline cards** as the headline metric pattern: number, trend arrow with %, tiny chart. Evalyn's run summary should look like this for scores.

### Anthropic Console (console.anthropic.com)
**Patterns to import:**
1. **Three-pane Workbench: prompt left, output center, settings right.** This is the platonic shape for "configure -> run -> inspect." Evalyn's command runner mode should adopt the same proportions. The "Get Code" button -- which exports the current playground state as runnable Python/TypeScript/cURL -- maps perfectly to "Get CLI" -- "I configured this run in the form, now show me the `evalyn ...` command I can paste into a script."
2. **Prompt Improver inline in the workbench.** Anthropic added an AI-powered "improve this prompt" button that lives in the workbench, not in a separate tool. Evalyn could mirror this: an "ask agent for help with this command" button on every form.

### GitHub Copilot Chat / Cursor
**Patterns to import:**
1. **Inline chat (Cmd+I) in addition to sidebar chat.** Copilot Chat has both a Chat sidebar and an inline chat that opens at the cursor. Cursor 3 (April 2026) added an "Agents Window" with parallel agents in tabs. Evalyn should think about this: the chat panel is good for long conversations, but most actions ("rerun this row with X tweaked," "explain this metric") want an inline ephemeral prompt.
2. **Cursor 3's Design Mode** (click an element, instruct the agent on it). Evalyn equivalent: click a metric / a failed row / a span, and the agent gets that context auto-attached. This is far better than asking users to copy-paste IDs into chat.

---

## Cross-cutting patterns table

| Pattern | Who does it | How it works | Applicability to Evalyn |
| --- | --- | --- | --- |
| Cmd+K command palette | Linear (gold std), Vercel, Stripe, Anthropic Console | Fuzzy-search action list, grouped, keyboard-first | High -- 35 CLIs without a palette is unmanageable. Spec lists as v1 non-goal; reconsider |
| 3-pane playground (config/output/settings) | Anthropic Console, Braintrust, Langfuse, Phoenix | Left input, center result, right config | High -- matches Evalyn's "form / stdout / metadata" structure |
| Side-by-side run comparison + diff highlighting | Braintrust (best), Langfuse | Aligned rows by input hash, green/red score deltas, sort by regression | High -- Evalyn's `compare` CLI produces this data; dashboard should render it visually |
| Tree/Timeline toggle on traces | Langfuse, Helicone (Tree/Span/Chat) | Same data, two cognitive frames | Medium -- relevant once Evalyn has hierarchical call data in UI |
| Trace -> dataset one-click promotion | Braintrust | "Add to dataset" button on any trace row | High -- closes the regression-test loop; Evalyn doesn't have this |
| Insights / auto-clustering | LangSmith (Insights Agent), Phoenix (embedding clusters) | LLM groups failures by topic/mode, generates summary | High -- Evalyn already has `evalyn insights`; surface inline in dashboard |
| Inline chat at cursor | Cursor, Copilot Chat | Cmd+I opens ephemeral prompt | Medium -- complements dock-right chat; better for context-attached actions |
| Click-to-attach context | Cursor 3 Design Mode | Click element, agent gets it as context | High -- "click this row, ask agent" is the killer chat pattern |
| Big-number + sparkline cards | Stripe, Vercel, Braintrust dashboards | Headline metric with trend + tiny chart | High -- run summary cards |
| Skeleton + empty + loaded states | Stripe (codified) | Skeleton matches layout, empty has illustration + CTA | High -- improves perceived perf, doubles as onboarding |
| Streaming logs that start instantly | Vercel | No buffering, no full-page spinner | Critical -- Evalyn's WebSocket job streaming should feel exactly like this |
| "Get Code" export from playground | Anthropic Console | Button copies config as runnable code | High -- "Get CLI command" from a form is the perfect bridge between dashboard and terminal |
| Custom trace cards | Braintrust | Domain-specific visual replacing raw JSON | Low for v1 -- power feature, requires config |
| Custom dashboards / chart builder | Langfuse, Helicone | Drag widgets, query metrics | Low -- ship great defaults instead |
| HQL / SQL-like query | Helicone (HQL), Braintrust (custom columns) | Power-user analytics surface | Low -- adds tax for casual users |

---

## Recommendations for Evalyn

Ranked by leverage / effort. Each item is a borrowable pattern with an implementation sketch sized to Evalyn's current architecture (FastAPI + React, subprocess-driven CLIs).

### 1. Cmd+K command palette (HIGH leverage, MEDIUM effort)

**Sketch:** A modal opened by Cmd+K. Three sections: "Run command" (all 35 CLIs, fuzzy-searchable, grouped by domain -- Calibrate, Annotate, Analyze, Insights, etc.), "Open" (recent runs, datasets, projects), "Ask agent" (free-text input that sends to chat). Selecting a CLI opens its argparse-generated form in the center pane. Selecting "Ask agent" focuses the chat input pre-filled.

**Why now:** spec lists this as v1 non-goal, but with 35 commands the sidebar alone won't scale. This is the single highest-leverage discoverability lever.

### 2. Three-pane Workbench layout for command runner (HIGH leverage, LOW effort)

**Sketch:** Mirror Anthropic Console exactly. Left = argparse form (collapsible sections for required vs. optional args). Center = streaming stdout/stderr terminal panel + tabs for "Output / Result JSON / Files written." Right = run metadata (duration, exit code, command line, environment) + "Get CLI" button that copies the literal `evalyn ...` invocation.

**Why now:** the spec already implies this with TitleBar/Sidebar/EditorTabs/BottomPanel/ChatPanel, but explicitly committing to the 3-pane Workbench shape (vs. tabs across the top) makes the form-to-output relationship spatial and obvious.

### 3. Run comparison view (HIGH leverage, MEDIUM effort)

**Sketch:** Take two run IDs. Show Braintrust-style aligned table: rows aligned by input hash, columns are scores per metric, cells show value + delta (green up / red down). Header has "Sort by regression" toggle. Click a row -> right panel shows full input + both outputs in a unified diff. Reuse the existing `evalyn compare` data layer.

**Why now:** comparison is the question users actually ask ("did my prompt change make things better?") and Evalyn currently answers it via CLI text output. Visual diff is the entire value-add of a dashboard for this workflow.

### 4. Click-to-attach context for the chat agent (HIGH leverage, LOW effort)

**Sketch:** Every metric, every failed row, every run row has an attach icon on hover. Click it -> the chat input gets a chip representing that context (`@run-3a8f`, `@row-12`, `@metric-faithfulness`). Agent receives the chip as structured context. Inspired by Cursor 3 Design Mode.

**Why now:** the chat-as-primary-surface tools all suffer when users have to copy-paste IDs. Click-to-attach makes the agent feel native to the dashboard rather than a separate window.

### 5. Skeleton + opinionated empty states (MEDIUM leverage, LOW effort)

**Sketch:** Borrow Stripe's pattern. Every panel has three states. Empty states are the onboarding -- the "no runs yet" panel CTA is "Run your first eval -> opens Cmd+K with `evalyn quickstart` pre-selected." This converts an empty state into a guided first step.

**Why now:** localhost-only tool means most users will hit empty states first. Make those the front door.

### 6. "Promote run failures to dataset" button (MEDIUM leverage, MEDIUM effort)

**Sketch:** On any run's failed-rows table, multi-select rows -> "Add to dataset" -> pick existing dataset or create new. Stores the input + expected output as a new dataset item. This is the Braintrust trace-to-dataset workflow adapted to Evalyn's file-based dataset format.

**Why now:** Evalyn currently has no UI flow to grow datasets from observed failures. This closes a real loop.

### 7. Inline chat (Cmd+I) in addition to dock-right (LOW leverage, MEDIUM effort)

**Sketch:** Cmd+I opens an ephemeral chat at the cursor / focused element. For long conversations, dock-right. For "explain this row" or "rerun with X tweaked," inline.

**Why now:** less critical than #1-6, but matches the Cursor/Copilot model that users will have muscle memory for.

### 8. Streaming-first UX (CRITICAL, already planned)

**Sketch:** No full-page spinners. Subprocess output streams into the terminal panel character-by-character as it arrives. Cancel button is always available. After completion, the output panel becomes scrollable + searchable. Match Vercel's "the dashboard remains usable while jobs stream."

**Why now:** the spec already commits to WebSocket streaming. Make the UI affordances match -- cancel button visible, stdout never blocks the rest of the UI.

---

## Anti-patterns observed

Things competitors do badly that Evalyn should consciously avoid.

1. **Sidebar with 10+ top-level items (Helicone, Langfuse).** When every concept is a top-level resource, users can't form a mental model. Evalyn should keep the sidebar to 4-6 items and use Cmd+K for everything else.

2. **Multiple overlapping mental models for "iterate on a prompt" (Langfuse: Prompts, Playground, Experiments, Datasets).** Pick one primary verb. For Evalyn, that verb is "run a CLI command" -- everything else (datasets, results, comparisons) is downstream of that act.

3. **Custom dashboard builder as a headline feature (Langfuse v4, Helicone HQL).** This is "we shipped a chart builder so users can solve their own problems." It's a tax on casual users and a maintenance burden. Ship great defaults; if power users need more, give them a JSON export, not a pivot table builder.

4. **Forcing users into a playground/sandbox before they have data (Phoenix, Helicone marketing flows).** Evalyn's audience arrives with real data on disk via the SDK. The first screen should reflect "you have N runs, here's the latest," not "let me show you a demo dataset."

5. **Trace viewers that show raw JSON spans by default (Phoenix, early LangSmith).** Hierarchical spans with formatted inputs/outputs/timings are far more useful than raw JSON. JSON should be a "show raw" toggle, not the default.

6. **Burying insights in a separate tab (LangSmith Insights Agent).** Auto-detected patterns should appear inline on the run summary, not behind a click. Evalyn's `cmd_analyze` already surfaces KEY FINDINGS at the top of CLI output -- the dashboard should preserve that hierarchy.

7. **Custom trace views that require configuration (Braintrust).** Powerful but most teams won't write the config. Evalyn should ship one well-designed default view per domain (eval run, calibration run, annotation session) instead of a config-driven framework.

8. **Treating chat as a separate product from the rest of the dashboard (most tools have no chat).** Where chat exists, it's often disconnected from the data. Evalyn's spec is right to dock chat alongside the workspace -- the recommendation is to go further and make every data element click-attachable as chat context.

---

## Sources

- [Braintrust homepage](https://www.braintrust.dev/)
- [Braintrust Eval via UI docs](https://www.braintrust.dev/docs/start/eval-ui)
- [Braintrust Interpret evals docs](https://www.braintrust.dev/docs/guides/experiments/interpret)
- [Braintrust Experiments UI: Now 10x faster](https://www.braintrust.dev/blog/faster-experiments)
- [Braintrust Stakeholder trust evals/observability blog](https://www.braintrust.dev/blog/stakeholder-trust-evals-observability)
- [LangSmith homepage](https://www.langchain.com/langsmith)
- [LangSmith docs](https://docs.langchain.com/langsmith)
- [LangSmith Observability docs](https://docs.langchain.com/langsmith/observability)
- [Phoenix marketing](https://phoenix.arize.com/)
- [Phoenix docs root](https://arize.com/docs/phoenix)
- [Phoenix Datasets & Experiments overview](https://arize.com/docs/phoenix/datasets-and-experiments/overview-datasets)
- [Helicone homepage](https://www.helicone.ai/)
- [Helicone Sessions docs](https://docs.helicone.ai/features/sessions)
- [Helicone Smarter Sessions changelog](https://www.helicone.ai/changelog/20250506-smarter-sessions-insights)
- [Langfuse homepage](https://langfuse.com/)
- [Langfuse New Trace View changelog](https://langfuse.com/changelog/2025-03-19-new-trace-view)
- [Langfuse Side-by-Side Playground changelog](https://langfuse.com/changelog/2025-07-28-playground-side-by-side)
- [Langfuse v4 Dashboard changes](https://langfuse.com/changelog/2026-03-23-v4-dashboard-changes)
- [Langfuse Prompt Experiments changelog](https://langfuse.com/changelog/2024-11-22-prompt-experimentation)
- [Linear keyboard shortcuts collection](https://keycombiner.com/collections/linear/)
- [Command Palette UX Patterns (Medium)](https://medium.com/design-bootcamp/command-palette-ux-patterns-1-d6b6e68f30c1)
- [How to build a remarkable command palette (Superhuman blog)](https://blog.superhuman.com/how-to-build-a-remarkable-command-palette/)
- [Vercel improved streaming runtime logs exports changelog](https://vercel.com/changelog/improved-streaming-runtime-logs-exports)
- [Vercel runtime logs docs](https://vercel.com/docs/logs/runtime)
- [Stripe Apps empty state pattern](https://docs.stripe.com/stripe-apps/patterns/empty-state)
- [Stripe Apps design patterns](https://docs.stripe.com/stripe-apps/patterns)
- [Anthropic Console upgrade announcement](https://claude.com/blog/upgraded-anthropic-console)
- [Anthropic Console review (Nick Garnett)](https://nickgarnett.substack.com/p/the-anthropic-console-a-practical)
- [VS Code Copilot Chat docs](https://code.visualstudio.com/docs/copilot/chat/copilot-chat)
- [Cursor 3 changelog](https://cursor.com/changelog/3-0)
- [Cursor 3 InfoQ writeup](https://www.infoq.com/news/2026/04/cursor-3-agent-first-interface/)
