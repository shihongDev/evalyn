# Evalyn Dashboard UX Improvement Plan

Date: 2026-05-01
Synthesis of: 5 parallel research agents covering Claude.ai design DNA, current dashboard audit, eval-tool competitive scan, onboarding & empty states, chat & CLI form surfaces.
Source files: `01-claude-design-dna.md`, `02-current-dashboard-audit.md`, `03-competitive-scan.md`, `04-onboarding-and-empty-states.md`, `05-chat-and-form-synthesis.md` in this directory.

## 1. Executive summary

The Evalyn dashboard is a competent IDE port of an existing CLI — accurate, complete, built carefully. What it is NOT yet is a product. Two things separate it from "feels like Claude":

1. **At least seven user-facing surfaces lie or disappoint.** Welcome card hardcoded to a fake run id. Chat-to-CliForm handoff that opens the wrong tab. Cmd+K shortcut hint with no palette behind it. Form-mode toggle the user can't reach. "Raw" mode textarea that pretends to be editable but discards every change. "Predicted" cost/duration math that fabricates numbers. File tree clicks that open empty tabs. Each one trains users that the UI is unreliable.

2. **The product reads as terminal-scratch, not human-warm.** Empty states are JS comments (`// no jobs yet`). Buttons say `... Running` instead of `Running...`. Modal titles are lowercase serif italics. Composer placeholder claims `paste a CLI · @-mention a run` — neither works. The voice is terse and abbreviated; Claude's voice is warm full sentences.

Closing those two gaps — before any new feature — is the path to "feels like Claude.ai."

The plan below sequences the work in three phases:

- **P0 (1 engineering week, ship before any new feature):** Stop the UI from lying. Replace fake data, fix broken seams, kill the fabrication panels. Apply baseline visual polish (typography, spacing, microcopy). Build the four-branch first-run hero.
- **P1 (3 engineering weeks):** Upgrade the two highest-touch surfaces — chat (collapsible tool cards, editable confirmations, slash commands, smart scroll, persistent threads) and form (file picker, recent-runs strip, post-run summary in place). Add the three borrowed patterns that buy the most leverage: real Cmd+K palette, "Get CLI" export, click-to-attach chat context.
- **P2 (deferred to v1.2+):** Run comparison with diff highlighting, trace-to-dataset promotion, inline chat at cursor, saved presets.

## 2. The central design tension (and how we resolve it)

Agent 1 (Claude.ai design DNA) flagged this directly: Claude.ai's defining philosophy is "the thread is the product" — single column, minimal chrome, tool calls as inline collapsible cards in the conversation, no command palette, no notification panel, no power-user dashboard. Restraint is the feature.

The Evalyn dashboard spec commits to the opposite shape: an IDE with TitleBar, Sidebar (Files / Commands / Eval runs / History), EditorTabs, BottomPanel, ChatPanel docked right. Five panes, multiple resource lists, keyboard shortcuts. Structurally the opposite of Claude.

This is not a contradiction we can resolve by becoming claude.ai. The Evalyn audience is engineers iterating on prompts and metrics — they need density, multiple resources visible at once, and keyboard-first navigation. The IDE shell is correct.

What we borrow from Claude is everything *inside* that shell:

- The visual system: warm cream/ink/coral palette, serif body for reading material, sans for chrome, generous spacing, almost-imperceptible shadows, one motion curve.
- The voice: warm, second-person-but-sparing, full sentences, periods not exclamations, no jargon.
- The interaction primitives: collapsible tool-call cards in the chat thread (not a side panel), streaming caret, smart-scroll, per-message actions on hover, edit-and-resubmit forks the conversation.
- The trust-and-forgiveness model: one-click destructive actions made safe by reversibility, not by modal gates.
- The deliberate omissions: no theme picker, no density toggle, no power-mode panel — the aesthetic is the product.

What we accept from the IDE archetype:

- Sidebar with multiple resource lists (consolidated to 3, not 4 — see P1).
- Cmd+K command palette (Claude omits it; we need it because we have 35 commands).
- Tabs.
- Side-by-side comparison views.
- Visible run history strips.

The synthesis: **Claude's polish, applied to an IDE that knows it's an IDE.**

## 3. P0 — Stop the UI from lying (1 engineering week)

Every item in this section is a credibility breaker. Each one independently identified by Agent 2 (current audit) and several confirmed by Agent 5 (chat/forms deep dive). Ship these before any new feature.

### 3.1 Welcome view: drive everything from real data

Where: `dashboard/frontend/src/views/Welcome.tsx:31-70` and `:165`.

Current: hard-coded `Latest run · 82dddcc3 · −4 pts` card and a canned "Why is gemini regressing?" agent suggestion. Both render regardless of what's in the user's `.evalyn/`.

Fix: replace with the four-branch hero state machine from Agent 4's research. Branch decision, computed once on Welcome render from existing parallel boot fetches:

```
has .evalyn/ AND has runs?       -> RETURNING USER hero
has .evalyn/ AND no runs?        -> MID-PIPELINE hero
no .evalyn/ AND no API key?      -> COLD START hero (default)
no .evalyn/ AND has API key?     -> WARM START hero
```

Cold-start hero replaces the six fake cards with three honest paths: **Try demo** (default — ships baked-in seed fixture), **Instrument my agent** (wraps `evalyn quickstart`), **I have traces** (auto-discovers via `list-calls`).

Returning-user hero shows real "Last session" + "Most recent run" + "Recent activity" derived from cheap aggregations on existing data. No restoration of tab state — content-driven restoration only.

See Agent 4 doc, sections "Recommended first-run flow" and "Re-onboarding for return users" for full ASCII wireframes and microcopy.

### 3.2 Chat → CliForm broken seam

Where: `dashboard/frontend/src/App.tsx:62-64`.

Current: `openCli()` (called by chat suggestion clicks) adds a `cli:`-kind tab, but `TabContent` short-circuits with `if (activeTab.kind === 'cli') return <Workspace />`. The tab title changes; the content does not. The agent's flagship moment ("here's a pre-filled command") opens the wrong view.

Fix (option A, lighter): re-render the Workspace using `tabs.find(activeTabId).cliId` instead of the global `activeCliId`, and have `openCli` set `activeCliId`. Half-day work.

Fix (option B, structural): mount `CliForm.tsx` (currently dead code) when `activeTab.kind === 'cli'`. Requires consolidating CliForm and the ActiveForm in Workspace. Larger but cleaner.

Recommend A for P0 to unblock the chat-suggestion flow; revisit B during P1 form-surface upgrade.

### 3.3 Predicted cost/duration is fabricated

Where: `dashboard/frontend/src/views/CliFormBody.tsx:178-208`, specifically L201 `0.4 + filledCount * 0.18` and L206 `3 + Math.round(filledCount * 1.4)`.

Current: the "preview" mode shows a confident "predicted cost: $X · predicted duration: Ym" panel computed from a heuristic that has nothing to do with the actual command. Users will trust these numbers.

Fix (cheaper): delete the panel. One-line change.

Fix (better): replace with real heuristic from `runHistory`. "Last similar run: 240 items · 3m12s · $0.42" computed by finding the most recent `RunRecord` for this `cliId` whose `--dataset` and `--metrics` match. Reuses existing store state. ~30 lines.

### 3.4 Raw mode textarea pretends to be editable

Where: `CliFormBody.tsx:55-83`, especially L73-75.

Current: a textarea labeled "Edit this string directly" whose `onChange` discards the user's input. Looks editable, isn't.

Fix (cheaper): replace the textarea with a single read-only command line + copy button.

Fix (richer): implement reverse-parse so paste-a-CLI works. Defer to P2; for P0 ship the read-only version.

### 3.5 Cmd+K shortcut is a lie

Where: `App.tsx:117-129` listens for Cmd/Ctrl-K and toggles `paletteOpen`; nothing renders that state. README and Welcome (`Welcome.tsx:168-171`) and ChatPanel (`ChatPanel.tsx:498-500`) all advertise the palette.

Fix (P0, cheaper): remove the listener and the kbd hints. 30 minutes.

Fix (P1, richer): build the real palette. Sketched in section 6.3.

For P0 ship the removal. The palette goes in P1.

### 3.6 Tool-call confirmation race

Where: documented as `dashboard/KNOWN_ISSUES.md` #4. `ChatPanel.tsx:204-219` Approve/Reject buttons call `confirmAgent(true|false)` without `call.id`. With multiple stale awaiting cards, clicking any approves whatever's currently pending.

Fix: plumb `tool_call_id` through `confirmAgent` to the backend, validate server-side. ~2 hours.

### 3.7 File tree dead-end

Where: `store.ts:439-446` `openFile()` adds a tab; `App.tsx:71-78` returns "coming soon" for any tab kind that isn't cli / job / workspace.

Fix (P0, cheaper): hide the Files sidebar tab if there's no consumer. 10 minutes.

Fix (P0, richer): wire up a `FileViewer` tab that calls existing `GET /api/files/read`. ~1 day.

Recommend the richer option even for P0. The Files tab is one of three sidebar surfaces; hiding it removes a third of the navigation. Better to make it work.

### 3.8 P0 visual + microcopy baseline

Adopt the design tokens from Agent 1's report (section "Tokens you can copy directly into Evalyn dashboard"). Three concrete changes:

1. **Bump body font to 15px** (currently 13px, `index.css:83`); bump section padding to 20-24px (currently 18px / 8-12px gaps). Make the UI breathe.
2. **Replace Unicode glyphs with a real icon set** (Phosphor or Lucide, ~400KB tree-shaken). Today the app uses `▤ ▶ ⌬ ≡ ✦ ⊘ ↗ ↑ ＋ ×` which render inconsistently across OSes — `▤` is empty on some Linux fonts. This is one of the largest "looks broken" sources for a meaningful share of users.
3. **Microcopy pass.** Agent 2 produced a 30-row replacement table. Highest-impact rows:
   - `// catalog loading…` → `Loading commands...`
   - `// no runs yet` → `No eval runs yet.`
   - `// no jobs yet — run a CLI from the catalog to start one` → `No jobs yet.`
   - `... Running` button → `Running...`
   - `Ask <em>agent</em>` chat header → `Ask Evalyn`
   - `Ask anything · paste a CLI · @-mention a run` placeholder → `Message Evalyn...`
   - `workspace settings` (lowercase serif italic) → `Settings`
   - Floating button `✦ Ask agent` → `Chat with Evalyn`

Apply the full tokens block on top of `index.css`. Override the existing `--bg-0`/`--bg-1`/`--text-0` vars with the new ones; old code keeps working but renders against the new palette.

Fix the WCAG contrast failure on `--text-3` (light theme: `#8e8479` on `#f4ebe1` = 3.4:1, fails AA for body text).

### P0 acceptance criteria

A new user installs `evalyn-dashboard`, runs `evalyn dashboard`, opens the browser, and:

1. Does not see any data they don't have.
2. Does not encounter a button or shortcut that does nothing.
3. Sees a Welcome screen that reflects their workspace state and offers three honest paths forward.
4. Reads body text at 15px against generous gutters with consistent iconography and full-sentence empty states.
5. Can complete the chat → CliForm handoff without confusion.
6. Sees no fabricated cost or duration numbers.

## 4. P0 — Onboarding hero + demo data (parallel work, same week)

This is technically part of P0 but big enough to call out. From Agent 4.

### 4.1 The four-branch hero

Already specified in 3.1. Implementation: rewrite `Welcome.tsx` from scratch (it's only 178 lines). Pull branch decision from `useStore(s => ({hasData: s.runs.length > 0, hasKey: s.settings.active != null, hasFiles: s.fileTree.length > 0}))`.

### 4.2 Demo data fixture

Ship `dashboard/evalyn_dashboard/demo_fixture/` with one project, ~25 traces from a fake "research agent," one completed run, one calibration. Hand-authored, not scraped. Loaded via new `POST /api/demo/load` that copies into `.evalyn/`. Persistent thin top strip indicates demo state with `[Replace with my own]` CTA.

### 4.3 Five-CLI starter set

Default catalog view shows only `quickstart`, `one-click`, `list-calls`, `status`, `workflow` under a "START HERE" header, with "Show all 35 commands" toggle below. Once the user has run any CLI, default flips to full grouped view; starter set becomes a single collapsible row.

### 4.4 Non-blocking API key banner

Thin amber strip across the ChatPanel only (not the whole app). Visible only when no provider is configured. Dismissible. Re-appears with adapted copy when the user attempts a key-requiring action.

Special-case: detect Ollama at boot via `GET http://localhost:11434/api/tags`. If present, surface "Use local Ollama (no key needed)" link in the banner.

Settings modal foregrounds Anthropic with a recommended badge. Microcopy includes the security claim ("keys live in `~/.evalyn/credentials.json` mode 0600, never sent to your browser, never logged") and the budget recommendation ("we recommend $5"). Honesty about what's about to happen builds trust.

### 4.5 Empty-state designs

Each empty panel answers three questions: what would I see here, how do I get data, what's the lowest-effort thing I can do right now. Files / CLIs / Runs / Chat / Jobs panels each get the treatment from Agent 4 doc, section "Empty-state designs."

### 4.6 Job success/failure cards with next-step recommendations

After a job exits, append a card with the single most actionable next step:

- Success: "✓ Done. Suggested next step: analyze the results to see where the eval failed. [Open analyze] [Skip]" — sourced from the existing `evalyn workflow` logic.
- Failure: "✗ Failed: `ANTHROPIC_API_KEY not set`. [Add Anthropic key] [See full output]" — pattern-matched against ~10 common stderr signatures (missing key, missing dataset, no traces, network error, rate limit, OOM). No LLM call needed.

## 5. P1 — Chat surface upgrade (~2 weeks)

From Agent 5. The chat panel is the dashboard's most-used and least-polished surface. P1 brings it to Claude.ai parity.

### 5.1 Collapsible tool cards (highest leverage)

Where: `ChatPanel.tsx:146-256`.

Today: every completed tool call dumps up to 320px of `<pre>` output. Four tool calls fills the whole panel before the agent's actual conclusion is visible.

Proposed: one-line summary by default — `▸ $ evalyn list-runs --limit 3   complete · 0.4s · 50 lines`. Click chevron to expand. Show first 8 lines as peek when collapsed, in `text-2` color, monospace 10.5px. Errored tool calls auto-expand and stay open.

This single change is the biggest visual upgrade in the chat surface. ~80 lines.

### 5.2 Smart auto-scroll

Where: `ChatPanel.tsx:525-528`.

Today: brute-force scroll-to-bottom on every event, even if the user has scrolled up to read.

Proposed: only auto-scroll when within 80px of bottom. If scrolled up, show a `↓ N new` pill bottom-right. ~30 lines.

### 5.3 Streaming caret

Today: text appears with no visible "thinking" indicator beyond a flat `thinking …` row below messages.

Proposed: blinking caret (`▌`) at end of streaming text. One CSS animation on existing `streaming: true` flag. <10 lines.

### 5.4 Per-message timestamp + copy + retry + branch

Today: ChatTurn (L311-326) has no per-message actions.

Proposed: hover-revealed action row at bottom of each assistant message: copy (`⎘`), retry-from-here (`↻`), branch (`⤴︎`). Per-message timestamp + (assistant only) model id in header.

### 5.5 Editable args before confirmation (most consequential)

Today: confirmation card shows the agent's argv and offers Approve/Reject. If the agent picked the wrong `--dataset`, you reject, switch context to the form, retype.

Proposed: confirmation card includes [edit args] button that inline-mounts a tiny form using existing `ParamField` components. Buttons become "Approve once" / "Approve · don't ask again this session for run-eval" / "Reject". Per-tool, per-session whitelist (in-memory, not persisted).

Card also surfaces "THIS WILL" bulleted side-effects in plain English:
```
THIS WILL
  • Spawn an evalyn run-eval subprocess
  • Write to .evalyn/runs/<new-id>/
  • Stream stdout to a job: tab
  • Estimated cost: ~$0.42 (based on 240 items)
  • Estimated time: ~3-5 min
```

Backend changes:
- `POST /api/agent/chat/{tid}/confirm` accepts optional `args_override: dict` payload.
- New `_Thread.session_auto_approve: set[str]` in `agent.py`.
- Static `SIDE_EFFECTS` dict in `agent.py` keyed by tool name; emitted with `confirmation_required` event.

### 5.6 Slash commands + @-mentions in composer

Today: composer placeholder claims `paste a CLI · @-mention a run` — neither implemented.

Proposed:
- Slash commands (client-side, expand to text on send): `/run <cli-id>`, `/compare <runA> <runB>`, `/explain <run-id>`, `/clear`, `/settings`.
- `@`-trigger popover: `@run-...` from `store.runs`, `@file:...` from `store.fileTree`. Selected mention serializes as `[run-id: abc123]`.
- Paste detection: if pasted string starts with `evalyn ` and ends with newline, show contextual `[Convert to slash command?]` button.
- Composer auto-grows from 2 to 8 rows, then internal scroll.
- Token count bottom-right, soft-warn at 80%, hard-block at 100%.

### 5.7 Persistent threads

Today: `＋` button wipes the thread. No history.

Proposed (v1): localStorage per origin, cap last 20 threads. Each thread: id, auto-derived title from first user message, last-modified timestamp, message count. Header gets a "thread switcher" caret.

Per-message branch (`⤴︎`) creates a new thread copying messages up to that point, re-runs last user message in the new thread.

Save-thread to `.evalyn/threads/<id>.md` for sharing. Defer disk persistence to v2.

### 5.8 Empty state with grounded suggestion chips

Cold start (no runs): three example prompts in the chat panel.
```
"Walk me through evalyn"
"Show me what the demo data contains"
"Help me instrument my agent"
```

Warm start (runs exist): suggestions computed from `store.runs`:
```
"Why did pass rate drop on run-...-ghi?"
  Latest run: 8pp regression vs. previous

"Compare my last 3 run-eval runs"

"Cluster failures in run-...-ghi"
  47 failed items
```

Trust line below the chips: "I will always ask before running anything that writes data." Makes the read-only allowlist visible upfront.

### 5.9 Typed error banners

Today: all errors collapse to one banner with "Open settings" CTA.

Proposed taxonomy:
- `auth` (provider 401): top banner persistent. "Open settings" + "Retry last message".
- `rate_limit`: top banner with countdown. "Retry in 0:38" auto-counts.
- `tool_failure` (subprocess exit != 0): inline in the tool card (red, expanded). "Show full output" / "Re-run with same args" / "Open in form to tweak".
- `timeout` (5min confirmation lapse): inline in the confirmation card. "Approve now" within 30s grace.
- `network` (WS disconnect): bottom toast, transient. Auto-reconnect with the same `RECONNECT_DELAYS_MS` pattern jobs use.
- `budget_exceeded` (8 tool calls): final-message slot. "Continue anyway (8 more)".

Backend: agent socket needs reconnect (today it has none). New `tool_call_error` event distinct from `tool_call_complete{ok: false}`.

## 6. P1 — Form surface upgrade (~2 weeks)

From Agent 5. The form is touched on every CLI invocation.

### 6.1 File picker for path fields (highest leverage)

Where: `ParamField.tsx:166-180`.

Today: bare `<input>` with `./...` placeholder. The dominant first-time error is a typo in `--dataset` or `--metrics`.

Proposed: combobox — typeable input + dropdown of matching `.evalyn/` paths from `store.fileTree`, plus a `[📁]` button opening a file picker modal scoped to `.evalyn/`. Combobox is non-blocking — typing still works for paths outside `.evalyn/`.

~60 lines + new component. Reuses existing `store.fileTree`.

### 6.2 Three-state bool toggle + visible defaults

Where: `ParamField.tsx:51-77`.

Today: two-button "true" / "false" — once you click, you can never restore the argparse default. Defaults render identically to user picks.

Proposed: three-state segmented control `[ default ] [ on ] [ off ]`. Default state omits the flag from assembled command (matching `buildCli.ts` L52-56). Show resolved value subtly: `(default = false)`. Use semantic flags where inferable from `argparse._StoreTrueAction`.

### 6.3 Number sliders + units + ranges

Where: `ParamField.tsx:132-148`.

Today: bare `<input type=number>`. `--workers 8`, `--confidence-samples 3`, `--num-traces 5` all look identical.

Proposed: number + unit + slider where range known.
```
--workers           [  4  ]  workers   ◯─────────●─────  (1-16)
```

Backend: add optional `range: {min, max, step}` and `unit: string` to `ParamSchema`. Curated per-CLI in command modules:
```python
RANGES = {"workers": (1, 16, 1), "confidence_samples": (1, 32, 1)}
```

This is the single most leverageable backend addition — unblocks sliders, validation, and clearer rendering across all 35 CLIs.

### 6.4 Recent-runs strip above the form

Where: net-new component above `CliForm`.

Today: Workspace has run history below the form, but the form itself doesn't surface previous runs of the active CLI.

Proposed: horizontal strip of last N runs of THIS cli:
```
Recent run-eval runs:
[●  3m ago  pass=0.84  spam.jsonl   ]
[○  1h ago  pass=0.82  spam.jsonl   ]
[+ all 12 runs of run-eval →]
```

Click → seed form via existing `editRunArgs`. Diff badge: `Form differs from last run by: workers (4 → 8)` using existing `diffArgs`.

### 6.5 Post-run summary in place (don't close the tab)

Today: on submit, form closes and tab swaps to a job tab. User loses form context.

Proposed: keep the same tab. Swap form-body region with run summary:
```
✓ Run completed · 3m 12s · pass=0.84 · $0.42
  run-2026-04-30-ghi
  Pass rate dropped 8pp vs. previous run. [Why? · ask agent]
  [Open Run viewer →]  [Re-run same args]  [Re-run · tweak…]  [Compare to previous]
```

No tab churn. The natural next step is one click away.

### 6.6 Field-scoped validation

Today: all errors land in top banner ("missing required: dataset, metrics").

Proposed: red left-border on the failing field row + inline message. Replace top banner with single anchor link: "3 fields need attention" that focus-scrolls to the first.

### 6.7 ESSENTIAL coverage + argument groups

Today: only 4 of 15 command modules declare an `ESSENTIAL` set. The other 31 CLIs render with only `required=True` params visible, hiding everything behind one "Show all options (38)" disclosure.

Proposed: two-track work, both backend.

1. Populate `ESSENTIAL` for the remaining 11 modules (data work in `sdk/`).
2. Preserve argparse argument groups (`add_argument_group("LLM judges")`) which are currently flattened by `introspect.py:118-146`. Render each group as a collapsible section with a one-line "what this is for" header.

### 6.8 Kill the 3-mode toggle, absorb its useful parts

Today: `cliFormMode` toggle is in `tweaks` (`store.ts:32-48`) but no UI surfaces it. The default `preview` mode wastes a 380px column on the fabricated cost panel (P0 fix removes the panel; this proposal removes the mode entirely).

Proposed: single canonical view. Form on left, sticky live-command on right (carrying existing `CliHighlighted` syntax-coloring). Two affordances below the command: `[⎘ copy]` and `[✎ paste raw]` (the latter opens a reverse-parse modal — defer to P2).

This removes a tweak no one can reach plus the broken raw-mode textarea, and consolidates to one mental model.

## 7. P1 — Borrowed patterns (~1 week)

From Agent 3. The three highest-leverage patterns from competitive scan.

### 7.1 Real Cmd+K command palette (Linear-style)

Open via Cmd+K. Three sections:
- **Run command**: all 35 CLIs, fuzzy-searchable, grouped by domain.
- **Open**: recent runs, datasets, projects.
- **Ask agent**: free-text input that focuses chat composer pre-filled.

Linear-style single-letter shortcuts after open: `R` for "run last command", `A` for "ask agent", `D` for "open dataset".

Note: this directly contradicts Claude.ai's deliberate omission of Cmd+K (Agent 1 finding). The contradiction is intentional — Evalyn isn't a chat product, it's a tool engineers use to operate on 35 commands. Cmd+K is a power-user accelerant that serves the audience.

### 7.2 "Get CLI" export button (Anthropic Console-style)

On every form, a button that copies the literal `evalyn ...` invocation (already partially present as the "Copy command" affordance — promote it to a primary action labeled "Get CLI"). Bridges dashboard users to terminal users; reinforces that the dashboard is a wrapper, not a replacement.

### 7.3 Click-to-attach context for chat (Cursor 3-style)

Every metric, every failed row, every run row gets an attach icon on hover. Click → chat input gets a chip representing that context (`@run-3a8f`, `@row-12`, `@metric-faithfulness`). Agent receives the chip as structured context.

This is the killer pattern for the chat-as-primary-discovery model. Eliminates copy-paste of IDs.

## 8. P2 — Deferred to v1.2+

From multiple agents. Not P0 / P1 because each is multi-week or requires backend additions that depend on P0/P1 stabilization.

### 8.1 Side-by-side run comparison (Braintrust-inspired)

Take two run IDs. Aligned table by input hash. Score deltas in red/green at column-header level. "Sort by regression" toggle. Click row → unified diff of input + both outputs in right panel.

`evalyn compare` already produces this data; the dashboard renders it visually instead of dumping CLI text. ~2 weeks engineering.

### 8.2 Trace → dataset promotion (Braintrust-inspired)

Multi-select failed rows on a run viewer → "Add to dataset" → pick existing or create new. Stores input + expected output as new dataset items. Closes the regression-test loop the dashboard currently has no UI flow for.

### 8.3 Inline chat at cursor (Cursor / Copilot Cmd+I)

Cmd+I opens an ephemeral chat at the focused element. Complements dock-right chat — long conversations dock right, "explain this row" / "rerun with X tweaked" go inline.

### 8.4 Saved presets (form persistence)

Star a run's args from the recent-runs strip → persists to localStorage keyed by `cliId`. v2 writes to `.evalyn/dashboard/presets.json` so teams can commit presets to the repo.

### 8.5 Insights surfaced inline on run summary

`evalyn insights` already generates KEY FINDINGS in CLI output. The dashboard run viewer should render these as a top section, not require the user to know to run `insights`.

### 8.6 Other deferred items

- Reverse-parse paste-a-CLI modal in form.
- Save-thread to `.evalyn/threads/<id>.md`.
- Final-suggestion card mini-preview / hover state.
- Long-text token count + edit/preview toggle.
- Multiselect popover for high-cardinality options.
- Big-number + sparkline summary cards (Stripe / Vercel pattern).
- Skeleton loading states for every panel (Stripe pattern).

## 9. Cross-cutting: the Claude design DNA we're applying

Everywhere in P0 + P1, the visual and verbal language follows these principles from Agent 1:

### 9.1 Design tokens

Drop-in CSS variable block from Agent 1 doc. Highlights:

- Page background: `#F5F0EB` (warm cream — close to the existing dashboard palette).
- Accent: `#AE5630` (coral, close to the existing `#c44918` burnt orange).
- Text: `#2C2C2A` primary, `#5F5E5A` secondary, `#888780` tertiary.
- Body font: 16px (currently 13). Section padding: 24px (currently 18).
- Border radii: 16px composer/bubbles, 12px cards, 8px buttons.
- Shadows: `0 4px 20px rgba(0,0,0,0.035)` — almost imperceptible.
- One curve, one duration: `cubic-bezier(0.165, 0.85, 0.45, 1)` at 300ms.

### 9.2 Voice rules

- Second person, sparingly. "How can I help you" — not "Hey there, friend!"
- Periods, not exclamations. No marketing-deck enthusiasm.
- Verbs over adjectives. "Refine", "import", "build" — never "powerful", "intuitive", "amazing".
- No emoji in product surfaces (project CLAUDE.md already enforces this).
- No em dashes — use hyphens or colons (project CLAUDE.md enforces).
- Acknowledge limits when relevant. "Even experienced designers have to ration exploration" — naming the user's real constraint builds trust.

### 9.3 Color is meaning, not category

Don't ramp 8 metric tiles through 8 colors. Pass/fail is green/red. Provider is gray. Use coral (the brand accent) only for CTAs and the brand mark.

### 9.4 Trust through reversibility, not modals

No "are you sure?" dialogs. Stop-generate is one click. Edits fork conversations. Deleted items go to a recoverable state, not a confirmation. Cancel button is always visible while running.

The single existing exception that deserves its modal: write-CLI confirmation in chat. Even that becomes more powerful in P1 (editable args, per-session whitelist) rather than more obstructive.

### 9.5 Serif for content, sans for chrome

If a surface is reading material (an analysis report, a chat message), use the serif. Buttons, labels, sidebars use sans. The existing `Instrument Serif` choice is good; use it deliberately, not decoratively (current "Ask <em>agent</em>" italic in chat header is decorative).

## 10. What we're NOT doing (and why)

Each item below is something competitors do or a sub-agent could have recommended. Each is deliberately omitted.

- **Not abandoning the IDE shell.** The work demands density and multiple resources visible at once. Claude.ai's mono-modal layout is wrong for this audience.
- **Not adding theme picker / density toggle / power-mode panel.** Claude omits these because the aesthetic IS the product. We follow the same discipline.
- **Not building a custom dashboard / chart-builder.** Identified as anti-pattern in Agent 3 scan (Langfuse v4, Helicone HQL). Burns design budget for a feature 5% of users will configure once.
- **Not gating users behind a sandbox / playground flow** (anti-pattern from Phoenix, Helicone). Evalyn users arrive with real data; show them their data immediately.
- **Not building HQL-style power-user analytics.** Anti-pattern from Helicone scan. Adds tax for casual users.
- **Not implementing custom trace views requiring per-project config.** Anti-pattern from Braintrust scan. Ship one well-designed default view per domain.
- **Not adding telemetry in v1.** Spec §13.1 already defers; respected. We commit to A/B testing the starter set after telemetry exists, not before.
- **Not persisting tab/jobs state across server restarts.** Spec §2 non-goal. The four-branch hero replaces tab restoration with content-driven restoration.

## 11. Open questions and validation needed

Each requires a decision or external validation before locking the corresponding implementation.

1. **Live Claude.ai measurement.** Agent 1's CSS tokens are from public sources and a maintained Claude clone, not direct measurement (WSL2 host blocked headless Chromium). Before shipping the token block, run `claude.ai` in DevTools and replace APPROXIMATE values with MEASURED ones for at least three surfaces (empty new chat, mid-conversation thread, artifact split-view).

2. **Demo fixture maintenance.** Where does the fixture live (`dashboard/` or `sdk/examples/`)? Who hand-authors the 25-trace "research agent" persona? CI test on every PR to verify the fixture loads against current SDK schema?

3. **Chat as primary discovery requires good agent quality.** If the agent gives bad CLI suggestions, the "Ask the agent" nudge becomes a trap. Validate that the read-only allowlist's 19 commands cover the questions a beginner asks. Should we ship 5-10 vetted "if user asks X, suggest Y" routing rules in the agent system prompt?

4. **Banner dismissal logic.** Stripe pattern (dismiss = until next session) vs Linear pattern (7 days)? Recommend Stripe for a localhost dev tool — sessions are shorter.

5. **`EVALYN_DB` env var for non-default trace location.** "I have traces" branch silently misfires if user moved traces. Surface "Looking in `.evalyn/`. Different location? [Set EVALYN_DB]"?

6. **Microcopy needs user testing.** Even "Measure whether your LLM agent is doing what you want" is a hypothesis. Run 5 first-time users through the cold-start hero before locking copy.

7. **Persistent threads localStorage capacity.** ~5MB cap per origin. Tool outputs can blow this fast. Recommend: store thread metadata in localStorage and last 50 messages per thread; drop tool outputs to a `output_truncated: true` flag and replay from `/api/jobs/{id}` if the job still exists.

8. **Confirmation editability protocol.** If user edits args on `run-eval` confirm card, does the agent get told the args were edited (so it can adjust narrative)? Recommend: yes — `{approve: true, args: {...}, edited: true}` and inject system note "user edited args before approving."

9. **"Last session" timestamps require state.** Spec §2 non-goal: persistence of dashboard state. The "2 weeks ago" line needs a `.evalyn/.dashboard_session` file with `last_seen_at`. Does this count as forbidden state, or is it metadata?

10. **Removing dead code.** `BottomPanel.tsx` (87 lines) is never imported. `CliForm.tsx` (158 lines) is dead in the live UI. P0 should decide: delete both, or restore `BottomPanel` as the persistent "what's running right now" surface.

## 12. Sequencing and resourcing

Critical-path estimate, single engineer:

| Phase | Scope | Wall-clock |
|---|---|---|
| P0 trust fixes (sec 3) | 7 surfaces fixed, microcopy + tokens applied | 1 week |
| P0 onboarding (sec 4) | Four-branch hero + demo fixture + 5-CLI starter + key banner + empty states + next-step cards | 1 week (parallelizable with above) |
| P1 chat (sec 5) | Collapsible cards, smart scroll, streaming caret, editable confirmations, slash commands, threads, typed errors | 2 weeks |
| P1 form (sec 6) | File picker, three-state bool, sliders, recent-runs strip, post-run-in-place, validation, ESSENTIAL coverage | 2 weeks |
| P1 borrowed patterns (sec 7) | Cmd+K palette, Get CLI button, click-to-attach context | 1 week |
| **Total to "feels like Claude"** | | **~5 weeks** with 2 engineers in parallel, ~7 weeks solo |

P0 must complete before any new feature. P1 can parallelize across two engineers (chat and form are independent). P2 is post-v1, deferred indefinitely.

## 13. Closing observation

Agent 2 nailed this in its closing line: "Claude.ai's polish isn't about more pixels; it's about fewer surfaces that lie or disappoint."

The dashboard ships at least seven surfaces that say one thing and do another. Closing those gaps — before adding any new feature — is the single highest-leverage thing this team can do. Everything in P0 fits in one engineering week. Everything in P1 takes another three. The work to "feels like Claude" is closer than it looks, but only if we resist the temptation to ship new features over fixing what's broken.

The five-agent research suggests one strong opinion: **stop building, start finishing.**

---

## Appendix: report file index

| # | File | Length | Author focus |
|---|---|---|---|
| 1 | `01-claude-design-dna.md` | ~3,000 words | Decompose what makes claude.ai feel "Claude" |
| 2 | `02-current-dashboard-audit.md` | ~4,700 words | Inventory and friction of current implementation |
| 3 | `03-competitive-scan.md` | ~3,900 words | Eval-tool + general-polish competitive references |
| 4 | `04-onboarding-and-empty-states.md` | ~4,400 words | First-run flow, demo data, progressive disclosure |
| 5 | `05-chat-and-form-synthesis.md` | ~6,600 words | Chat + form deep redesign with wireframes |
| 0 | `00-synthesis.md` | this file | Prioritized improvement plan derived from the five |
