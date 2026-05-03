# Onboarding and Empty States Design

Domain: first-run experience, empty panels, progressive disclosure of 35 CLIs, and re-onboarding. Research only - no implementation.

## Executive summary

The pitch the dashboard must land in 30 seconds: "Evalyn measures whether your LLM agent is doing what you want." Today, the user opens `evalyn dashboard` and lands on `Welcome.tsx` (`/tmp/evalyn-dashboard-trunk/dashboard/frontend/src/views/Welcome.tsx`), which shows a serif headline "Run and analyze evaluations." plus six quick-action cards. Three of the six cards are dead ends for a first-run user (Latest run / Quick eval / Calibrate judge - they all assume traces or runs already exist). The chat panel renders `// Ask anything about your evals.` as a code-comment when there is nothing to ask about yet. There is no API key, no `.evalyn/`, no traces - so most of the UI is a museum of empty rooms.

The recommended flow is a four-rung ladder, not a wizard:

1. **Hero state** picks one of three onboarding paths based on detected ground truth (no `.evalyn/`, no API key, fresh agent code) - "Try the demo," "Instrument my agent," or "I already have traces."
2. **Demo path** is the default and ships seed data (one fake project, ~25 traces, one completed run, one calibration). The dashboard becomes immediately legible without the user typing anything.
3. **API key onboarding** is a non-blocking soft-gate: a banner, not a modal. The user can read everything; they only get gated when they ask the chat agent a question or click a write CLI.
4. **Progressive disclosure** ships 5 starter CLIs in the catalog by default with a "Show all 35" toggle. The chat panel becomes the primary discovery mechanism for everything beyond the starter set.

In 5 minutes the new user sees a populated dashboard, has run their first eval (against demo data), and has had at least one chat exchange. In 30 they have replaced demo data with their own.

## Current state assessment

Today's `Welcome.tsx` is 178 lines. It assumes the user already has (a) a most-recent run with id `82dddcc3` and (b) the mental model of "Quick eval", "Calibrate judge", "Annotate", "Build dataset", "One-click" as discrete things. None of this lands for someone who has not yet read the README. Specifically:

- The `Latest run` card is hard-coded to a fake id and will 404 on click.
- All five other cards open a CliForm for a CLI whose required positionals (`--dataset`, `--project`, `--metric-id`) cannot be filled because nothing exists.
- Sidebar panels render `// no files`, `// no runs yet`, `// catalog loading…` - terse and unwelcoming. The Files panel uses a literal code-comment slash `//` which reads as "broken" to a non-coder.
- ChatPanel placeholder copy `Ask anything · paste a CLI · @-mention a run` (`ChatPanel.tsx:493`) requires understanding three concepts and offers no examples.
- BottomPanel JobsList shows `// no jobs yet — run a CLI from the catalog to start one` (`JobsList.tsx:218`). The user has no way to know which CLI to start.
- SettingsModal exists (`SettingsModal.tsx`) but is never surfaced unless the user knows to open it; the spec §13.3 itself recommends "empty welcome with hint" which is the bare-minimum option.
- `KNOWN_ISSUES.md` does not list onboarding pain explicitly, but the deferred "compare side-by-side" entry suggests the team is in feature-build mode and onboarding is not yet on the critical path.

The spec §6 covers install impact but says nothing about what happens *after* the browser opens. §9.A documents the boot sequence ending at "Welcome view renders within ~200ms" with no further design. This research fills that gap.

The dashboard already ships an `EmptyForm` component in `Workspace.tsx:298-337` with the copy "Pick a command from the sidebar to start filling out a form." plus a button "Use <first-cli-id>" - which is closer to good but still places the burden of CLI selection on the user.

The good news: the underlying CLI surface area already has three usable onboarding ramps (`evalyn quickstart`, `evalyn one-click`, `evalyn init`) that the dashboard can wrap. The dashboard does not need to invent flows - it needs to surface and orchestrate the existing ones with better defaults and copy.

## Recommended first-run flow

The hero state branches on detected ground truth. The `/api/files/tree`, `/api/runs`, `/api/settings`, and `/api/jobs/recent` endpoints fetched in parallel at boot (spec §9.A.5) already give us everything we need to pick a branch.

**Branch decision tree** (computed once, on Welcome render):

```
has .evalyn/ AND has runs?       -> RETURNING USER hero (see "Re-onboarding")
has .evalyn/ AND no runs?        -> MID-PIPELINE hero ("you have traces, build a dataset")
no .evalyn/ AND no API key?      -> COLD START hero (default)
no .evalyn/ AND has API key?     -> WARM START hero (skip key prompt)
```

**COLD START hero** (the most common first-run case):

```
+---------------------------------------------------------------+
|                                                               |
|  WELCOME TO EVALYN                                            |
|                                                               |
|  Measure whether your LLM agent is doing                      |
|  what you want.                                               |
|                                                               |
|  Pick one to start:                                           |
|                                                               |
|  +---------------+  +---------------+  +---------------+      |
|  |  ▶  Try demo  |  |  +  Instrument |  |  ↑  I have    |     |
|  |               |  |     my agent  |  |     traces    |      |
|  |  See a real   |  |               |  |               |      |
|  |  eval in 30s  |  |  Add @eval to |  |  Skip ahead   |      |
|  |  - no setup.  |  |  my Python    |  |  to dataset.  |      |
|  |               |  |  agent.       |  |               |      |
|  |  [ Load demo ]|  |  [ Show me ]  |  |  [ Find them ]|      |
|  +---------------+  +---------------+  +---------------+      |
|                                                               |
|  ─────────────────────────────────────────────────            |
|  Already comfortable? Use the sidebar on the left or          |
|  ask the agent on the right.                                  |
|                                                               |
+---------------------------------------------------------------+
```

The three cards correspond to the three real onboarding shapes: (1) curious learner, (2) developer ready to instrument, (3) advanced user who has already collected traces elsewhere. None depend on API keys - we defer that prompt until it actually matters.

**Try demo** copies `examples/demo/` (a new fixture - see Demo Data Strategy below) into `.evalyn/demo-project/`, populates the file tree, opens a single CLI tab pre-filled with `evalyn analyze --run demo-001`, and surfaces a banner "You are looking at demo data. [Replace with my own]". The user clicks Run, sees the analyze output stream into the terminal, and has experienced the full loop.

**Instrument my agent** opens a Workspace tab containing the `evalyn quickstart` form with a sensible default - if the dashboard detected `.py` files in cwd, the `--agent-file` field is pre-filled. The form's "What this does" header explains in 2 lines: "Scans for your agent code, adds the @eval decorator, and writes evalyn.yaml. Safe to run - no traces are collected yet."

**I have traces** runs `evalyn list-calls --json` in the background and surfaces results inline ("Found 14 calls across 2 projects. Open one?") with a one-click "Build dataset from these" button.

**MID-PIPELINE hero** (user has `.evalyn/` but no runs - the second-most-common case): replace the three onboarding cards with three workflow-aware cards driven by `evalyn workflow`:

```
+---------------------------------------------------------------+
|  WELCOME BACK                                                 |
|                                                               |
|  You have 14 traces across 2 projects, no datasets yet.       |
|                                                               |
|  Suggested next step:                                         |
|                                                               |
|  +-----------------------------------------------------+      |
|  |  Build a dataset from your traces                   |     |
|  |                                                     |      |
|  |  evalyn build-dataset --project <pick one>          |     |
|  |                                                     |      |
|  |  [ Configure & run ]   [ Use one-click instead ]    |     |
|  +-----------------------------------------------------+      |
+---------------------------------------------------------------+
```

This embeds the existing `evalyn workflow` logic in the welcome state - the CLI already does context-aware suggestions, the dashboard just renders them.

## API-key onboarding

Never block. Always degrade gracefully. The user can browse the catalog, view traces, even run pure-objective CLIs (`list-calls`, `validate`, `status`) without a key.

**Banner placement**: thin amber strip across the top of the ChatPanel only (not the whole app). Visible only when the user has not configured at least one provider. Microcopy:

```
+-----------------------------------------------------+
| ◐  Add an LLM key to use the chat agent             |
|    or run any LLM-judge eval.    [ Add key ]   ✕    |
+-----------------------------------------------------+
```

Dismissible (`✕` writes a `dismissed_apikey_banner` flag to localStorage). Re-appears when the user attempts an action that requires a key, with adapted copy:

```
+-----------------------------------------------------+
| ◐  This needs an LLM key.   [ Add Anthropic key ]   |
+-----------------------------------------------------+
```

**The "Add key" flow** opens the existing `SettingsModal` (`SettingsModal.tsx`), but reorders the providers so that Anthropic is on top with a recommended badge. Microcopy at the top of the modal:

```
Connect a provider

The agent and any LLM-judge metrics need an API key.
Keys live in ~/.evalyn/credentials.json with mode 0600,
never sent to your browser, never logged.

(Use a fresh key with a low budget - we recommend $5.)
```

The "low budget" sentence is important and unique to evalyn's audience: people running evals can burn money fast. Notion does not need to say this. Stripe-style honesty about what's about to happen builds trust.

**Test button**: after key entry, the modal auto-runs the existing `test_provider()` 1-token call. On success, a green pill "✓ connected" appears next to the provider name and the modal is auto-dismissed after 1.5s. On failure, the error is rendered inline in plain English ("Anthropic returned 401 - check the key starts with `sk-ant-`").

**Ollama special-casing**: detect a running Ollama instance via `GET http://localhost:11434/api/tags` at boot. If found, surface a one-click "Use local Ollama (no key needed)" link in the banner. This is the killer feature for privacy-sensitive users and is currently invisible.

## Empty-state designs

Each empty state must answer: *(1) what would I see here if I had data?* *(2) how do I get data?* *(3) what is the lowest-effort thing I can do right now?*

### Files panel - no .evalyn/

Today: `// no files` (`FileTree.tsx:104`).

Proposed:

```
+----------------------------+
| WORKSPACE                  |
+----------------------------+
|                            |
|   No data yet              |
|                            |
|   Files appear here once   |
|   you run an eval or load  |
|   the demo dataset.        |
|                            |
|   [ Load demo ]            |
|   [ Run quickstart ]       |
|                            |
+----------------------------+
```

Icon: faded folder glyph above the headline. Two CTAs match the welcome cards. Both buttons prefill a Workspace form rather than running blindly.

### CLIs panel - catalog loaded but never used

Today: shows a flat alphabetical/grouped list of all 35 CLIs, no help.

Proposed (default, "starter" toggle on):

```
+----------------------------+
| COMMANDS                   |
| [filter…]  [group][a-z]    |
+----------------------------+
| START HERE · 5             |
| $ quickstart   ★            |
| $ one-click    ★            |
| $ list-calls               |
| $ status                   |
| $ workflow                 |
|                            |
| ─ Show all 35 commands ─   |
+----------------------------+
```

Star indicates "recommended for first-run." When the user clicks "Show all 35", the existing grouped catalog renders below the starter set. Once the user has run any CLI, the starter set collapses to a single row "★ Starter (5)" and the full catalog is the default. This avoids the wall.

### Runs panel - no runs yet

Today: `// no runs yet` (`RunsList.tsx:33`).

Proposed:

```
+----------------------------+
| RUNS                       |
+----------------------------+
|                            |
|   ─ No runs yet ─          |
|                            |
|   A run appears here for   |
|   each `evalyn run-eval`   |
|   or `one-click`.          |
|                            |
|   [ Run an eval ]          |
|                            |
+----------------------------+
```

The CTA opens the `run-eval` form, but pre-validates: if no datasets exist, the form's "What you need" header surfaces "You need a dataset first - try `build-dataset` or `quickstart`."

### Chat panel - first time opening

Today: `// Ask anything about your evals.` followed by `// Tools call CLIs; writes require your approval.` (`ChatPanel.tsx:553-557`).

Proposed:

```
+---------------------------------------+
| Ask agent · Anthropic                 |
+---------------------------------------+
|                                       |
|   Hi. I can read your traces, run     |
|   evals, and explain results.         |
|                                       |
|   Try one of these to get started:    |
|                                       |
|   ┌─────────────────────────────────┐ |
|   │  Walk me through evalyn        │  |
|   └─────────────────────────────────┘ |
|   ┌─────────────────────────────────┐ |
|   │  Show me what the demo data    │  |
|   │  contains                       │ |
|   └─────────────────────────────────┘ |
|   ┌─────────────────────────────────┐ |
|   │  Help me instrument my agent   │  |
|   └─────────────────────────────────┘ |
|                                       |
|   I will always ask before running    |
|   anything that writes data.          |
|                                       |
+---------------------------------------+
| Ask anything · ↵ to send         [↑]  |
+---------------------------------------+
```

The three suggestion chips are the Claude.ai pattern. They are clickable - clicking sends the literal text. After the user has used the agent once, the chips disappear and the placeholder shrinks to `Ask the agent…`.

The bottom-line "I will always ask before running anything that writes data" makes the read-only allowlist (spec §7 `agent.py`) visible upfront, building trust before the first confirmation modal fires.

### Jobs panel - no jobs run

Today: `// no jobs yet — run a CLI from the catalog to start one` (`JobsList.tsx:218`).

Proposed:

```
+--------------------------------------------------+
| ID  | COMMAND     | STATUS    | ELAPSED |        |
+--------------------------------------------------+
|                                                  |
|   ─ No jobs yet ─                                |
|                                                  |
|   Every CLI you run shows up here with live      |
|   output, status, and a cancel button.           |
|                                                  |
|   [ Run a CLI ]   [ See an example ]             |
|                                                  |
+--------------------------------------------------+
```

"See an example" opens a read-only modal with a static screenshot of a finished `list-calls` run, so the user knows what "live" looks like before committing to running anything.

## Progressive disclosure of 35 commands

The five-card starter set:

| ID | Why it's a starter | What it depends on |
|---|---|---|
| `quickstart` | The official onboarding ramp. Detects framework, instruments, configures. | Nothing |
| `one-click` | Full pipeline in one command. The "magic" demo. | Project + traces |
| `list-calls` | Read-only, safe, immediately legible. | Traces |
| `status` | Read-only, shows pipeline state. | Dataset (optional) |
| `workflow` | Self-documenting "what should I do next." | Nothing |

These are the only commands a beginner needs for their first session. The other 30 are still searchable via the filter input and discoverable via the agent.

**Tier 2** revealed by clicking "Show all": the existing grouped view (Data / Eval / Judge / Iterate / Infra). No change to today's behavior.

**Tier 3** (advanced, hidden behind a separate toggle): commands marked with `ADVANCED = {...}` constants per the spec §7 `introspect.py`. These remain hidden by default even in "Show all" view. Today the catalog has no commands marked advanced; this needs a one-time pass per module to add `ADVANCED = {"seed", "workers", "max_sim_seeds", ...}`.

**Chat as primary discovery for first session**: when the catalog first renders, surface a one-line nudge below the starter set:

```
Don't see what you need? Ask the agent →
```

Clicking focuses the chat composer with text "What command should I use to ___?" pre-populated. This converts catalog-search-failure into chat-engagement, which is the dashboard's strongest UX surface and the one spec §7's read-only allowlist makes safe.

## Demo data strategy

**Yes, ship seed data.** Stripe-style. The Files panel, Runs list, and Workspace should all have something to render the moment the dashboard opens, with a banner clarifying it is not real.

**What ships**: a `dashboard/evalyn_dashboard/demo_fixture/` directory containing:

- `demo-project/` (project name)
- `traces.sqlite` with 25 captured calls from a fake "research agent" - varied inputs, mixed pass/fail, two simulated providers (anthropic + openai)
- `datasets/research-v1-20260301-100000/` with a 25-item dataset.jsonl
- `datasets/.../metrics/metrics.json` with 5 metrics (3 objective, 2 LLM-judge with cached scores - no live LLM call needed to "see" the demo)
- `datasets/.../eval_runs/20260301-100100_demo001/` containing a complete `results.json` with realistic distributions
- `datasets/.../calibrations/helpfulness_accuracy/` with a finished calibration record

**Loading**: the "Load demo" button calls a new `POST /api/demo/load` endpoint that copies the fixture into `.evalyn/` (refusing if `.evalyn/` already exists with non-demo content). After load, the file tree, runs list, and welcome hero all repopulate via the existing fetch endpoints.

**Indication that it is demo**: a persistent thin top strip across the title bar:

```
+---------------------------------------------------------------+
| ◉ Demo data loaded · everything you see is fake.              |
|                              [ Replace with my own ]   [ ✕ ]  |
+---------------------------------------------------------------+
```

The `[Replace with my own]` button opens an inline guide: "1. Delete the demo (one click). 2. Run `evalyn quickstart` to instrument your agent." Both steps have one-click buttons.

**Why ship demo, not autogenerate**: generating fake traces with the user's own API key would burn money and feel deceptive. A baked-in fixture is honest, fast, and works offline.

**Relationship to `evalyn quickstart`**: the existing CLI does *not* ship demo data - it instruments the user's real code. The dashboard adds the demo path as a fourth option alongside quickstart's three (auto-detect / specify file / run agent). Document this addition in `dashboard.md`.

## Help and learnability layer

**Inline tooltips on every form field**: the introspector already extracts `help` text from argparse (`introspect.py` per spec §7). Render it inline below each input, not in a hover tooltip. Hover tooltips are unfriendly to keyboard users and unreviewable in screenshots.

**"What's this?" links**: every CLI form header links to the corresponding `docs/clis/<id>.md` rendered inline as a slide-over panel. Source: the markdown files already exist at `/mnt/c/Users/shiho/Desktop/projects/evalyn/docs/clis/*.md`. The dashboard ships them as static assets, served at `/api/docs/{cli_id}`. No live network access needed.

**Command transcript on every run**: today the Workspace's `RunCard` shows the command at the top. Add a "Copy command + paste into terminal" button below it with copy text:

```
$ evalyn run-eval --dataset data/myapp-v1/ --workers 4
```

This honors the project's principle that the dashboard is a wrapper, not a replacement, for the CLI. Power users can move workflows to scripts.

**"Next steps" cards after success**: when a job exits successfully, the RunCard's expanded view appends a yellow card:

```
+---------------------------------------------------+
| ✓ Done. Suggested next step:                      |
|                                                   |
| analyze the results to see where the eval failed  |
|                                                   |
| [ Open analyze ]   [ Skip ]                       |
+---------------------------------------------------+
```

The suggestion text comes from the same `evalyn workflow` logic, called server-side after each successful job. Match the README's pipeline ladder (collect → build → evaluate → calibrate → expand).

**Failure recovery cards**: when a job fails (non-zero exit), append a red card with the most actionable diagnostic:

```
+---------------------------------------------------+
| ✗ Failed: ANTHROPIC_API_KEY not set               |
|                                                   |
| [ Add Anthropic key ]   [ See full output ]       |
+---------------------------------------------------+
```

Pattern-match the last 50 lines of stderr against ~10 common failure signatures (missing key, missing dataset, no traces, network error, rate limit, OOM). Cheap, no LLM call.

## Re-onboarding for return users

The power user comes back two weeks later. Today they get the same fake-id Welcome cards. Proposed:

**RETURNING USER hero** (when `.evalyn/` exists with at least one run):

```
+---------------------------------------------------------------+
|  WELCOME BACK                                                 |
|                                                               |
|  Last session · 2 weeks ago                                   |
|                                                               |
|  +-----------------------------------------------------+      |
|  |  Your most recent run                              |       |
|  |  82dddcc3 · research-v1 · 84% pass · 2 weeks ago   |      |
|  |  [ Open ]  [ Compare to next ]  [ Re-run ]         |      |
|  +-----------------------------------------------------+      |
|                                                               |
|  Recent activity:                                             |
|  · 12 traces collected last week                              |
|  · 1 calibration in progress (helpfulness_accuracy)           |
|  · 0 jobs running                                             |
|                                                               |
|  [ Resume calibration ]   [ Workflow overview ]               |
+---------------------------------------------------------------+
```

The "Recent activity" lines come from cheap aggregations on the existing data: count traces in `.evalyn/traces.sqlite` from the last 7 days, scan calibrations directory for `status: "in_progress"`, etc.

**No tab/state restoration in v1**: the spec §10 explicitly defers this. The "Welcome back" hero replaces it with content-driven restoration - the user lands on what their data looks like now, not on whatever tabs they had open. This is "restoration without lock-in" because nothing the dashboard does is destructive: the user can ignore the suggestions and open whatever they want.

**Fresh-eyes mode**: a small link in the top-right of the welcome hero, "Reset welcome to first-run." Clicking shows the cold-start hero again, useful for screencasts and demos. Does not delete `.evalyn/`.

## Microcopy package

Headlines and labels in priority order. Voice: warm but not chatty, concrete, no marketing words ("powerful", "amazing", "robust"). No emojis (per project CLAUDE.md). No em dashes - use hyphens or colons.

| Surface | Text |
|---|---|
| Cold-start hero headline | `Welcome to evalyn` |
| Cold-start hero subhead | `Measure whether your LLM agent is doing what you want.` |
| Cold-start hero footer | `Already comfortable? Use the sidebar on the left or ask the agent on the right.` |
| Demo card title | `Try the demo` |
| Demo card body | `See a real eval in 30 seconds. No setup, no keys, no data sent anywhere.` |
| Demo card CTA | `Load demo` |
| Instrument card title | `Instrument my agent` |
| Instrument card body | `Add the @eval decorator to your Python agent. Takes about 2 minutes.` |
| Instrument card CTA | `Show me how` |
| Have-traces card title | `I have traces` |
| Have-traces card body | `Skip ahead. We will find your traces and build a dataset.` |
| Have-traces card CTA | `Find my traces` |
| API key banner | `Add an LLM key to use the chat agent or run any LLM-judge eval.` |
| API key banner CTA | `Add key` |
| API key gate (action-triggered) | `This needs an LLM key.` |
| Settings modal headline | `Connect a provider` |
| Settings modal body | `The agent and any LLM-judge metrics need an API key. Keys live in ~/.evalyn/credentials.json with mode 0600, never sent to your browser, never logged. Use a fresh key with a low budget - we recommend $5.` |
| Settings modal Ollama hint | `We detected Ollama running locally. Use it to skip the API key entirely.` |
| Files empty | `No data yet. Files appear here once you run an eval or load the demo dataset.` |
| Runs empty | `No runs yet. A run appears here for each evalyn run-eval or one-click.` |
| Jobs empty | `No jobs yet. Every CLI you run shows up here with live output, status, and a cancel button.` |
| CLI catalog starter section | `START HERE` |
| CLI catalog starter footer | `Show all 35 commands` |
| CLI catalog discovery nudge | `Don't see what you need? Ask the agent.` |
| Chat empty greeting | `Hi. I can read your traces, run evals, and explain results.` |
| Chat suggestion 1 | `Walk me through evalyn` |
| Chat suggestion 2 | `Show me what the demo data contains` |
| Chat suggestion 3 | `Help me instrument my agent` |
| Chat trust line | `I will always ask before running anything that writes data.` |
| Chat composer placeholder | `Ask anything` |
| Demo banner | `Demo data loaded. Everything you see is fake.` |
| Demo banner CTA | `Replace with my own` |
| Returning hero headline | `Welcome back` |
| Returning hero recent-run row | `Your most recent run: <id> · <dataset> · <pass>% pass · <time-ago>` |
| Job success card | `Done. Suggested next step:` |
| Job failure card (key missing) | `Failed: <PROVIDER>_API_KEY not set` |
| Job failure card (no dataset) | `Failed: no dataset found at that path. Did you mean <closest match>?` |

Avoid: "Awesome!", "Let's get started!", "You're all set!", any exclamation marks. The CLI tone (terse, period-terminated, factual) is the dashboard's tone.

## Risks and open questions

1. **Demo fixture maintenance burden**: the demo data must stay valid as the SDK schema evolves. Risk: schema migration breaks demo fixture, demo button errors. Mitigation: a CI test that loads the demo fixture against current schema on every PR. Open: where does the fixture live - in `dashboard/` or in `sdk/examples/`?

2. **Chat as primary discovery requires good agent quality**: if the agent gives bad CLI suggestions, the "Ask the agent" nudge becomes a trap. Need to validate that the read-only allowlist's 19 commands cover the questions a beginner asks. Open: should we ship 5-10 vetted "if user asks X, suggest Y" routing rules in the agent system prompt, separate from tool selection?

3. **"I have traces" branch assumes existing `.evalyn/` location**: if the user used `EVALYN_DB` to put traces elsewhere, the file-tree fetch returns empty and the branch silently misfires. Need to surface "Looking in .evalyn/. Different location? [Set EVALYN_DB]". Open: should the dashboard read user shell env at boot?

4. **Banner dismissal vs re-prompt logic**: how often is too often to re-show the API key banner after dismissal? Linear's pattern: dismiss = 7 days. Stripe's pattern: dismiss = until next session. Open: which fits a localhost dev tool best? Recommend Stripe-style (next session) because dev sessions are shorter.

5. **Demo data privacy**: are we sure the fake traces contain no PII or copyrighted content? The fixture must be hand-authored, not scraped. Open: who writes the 25-trace fixture, and what is the canonical "research agent" persona?

6. **Five starter CLIs is opinionated**: a user whose first need is `cluster-failures` will not see it. The "Ask the agent" escape hatch mitigates this, but only for users who realize they should ask. Open: should we A/B test starter sets via a hidden config flag once telemetry exists (which it does not, per spec §13.1)?

7. **Re-onboarding "last session" timestamps require state**: today the dashboard does not persist anything between server restarts (spec §2 non-goals: "Persistence of dashboard state"). The "2 weeks ago" line needs a `.evalyn/.dashboard_session` file with `last_seen_at`. Open: does this count as forbidden state persistence, or is it separate metadata?

8. **Dependency between hero branches and demo path**: the cold-start hero default-recommends the demo. If demo loading fails (disk full, permissions), the user has no fallback. Open: should the hero pre-flight the demo path and gracefully degrade to "Instrument my agent" if it cannot write?

9. **Microcopy needs user testing**: the proposed copy was written from first principles, not validated. Even the headline "Measure whether your LLM agent is doing what you want" is a hypothesis. Recommend running 5 first-time users through the cold-start hero and watching where they hesitate.
