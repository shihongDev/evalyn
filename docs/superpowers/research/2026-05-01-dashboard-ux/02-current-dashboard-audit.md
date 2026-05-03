# Current Evalyn Dashboard UX Audit

Date: 2026-05-01
Scope: Implementation in worktree at `/tmp/evalyn-dashboard-trunk/dashboard/`.
Methodology: Read every frontend component, view, store action, the
introspector and selected backend routers. Cross-referenced with the spec at
`/mnt/c/Users/shiho/Desktop/projects/evalyn/docs/superpowers/specs/2026-05-01-evalyn-dashboard-design.md`
and `dashboard/KNOWN_ISSUES.md`.

Convention: file paths are absolute. Claims marked "(read)" are observed in
the source. Claims marked "(infer)" describe runtime behavior I derived from
the code path but did not see executed.

---

## Executive summary

### Top pain points

1. **Welcome screen is partly fake.** The "Latest run" card hard-codes
   `82dddcc3 · −4 pts` and tries to open `82dddcc3.run` regardless of what is
   actually in the user's `.evalyn/` directory
   (`dashboard/frontend/src/views/Welcome.tsx:32-39`). A first-time user with
   an empty workspace clicks it and gets a "coming soon" placeholder
   (`App.tsx:71-78`). The hint about asking the agent suggests a fixed
   gemini-regression query that has nothing to do with their data
   (`Welcome.tsx:165`). This is the single biggest credibility hit on first
   impression.
2. **CLI catalog dumps every flag into "Show all options" for 31 of 35
   commands.** Only 4 of 15 command modules declare an `ESSENTIAL` set
   (`clustering.py`, `dataset.py`, `traces.py`, `evaluation.py` per
   `grep ESSENTIAL`). Result: forms like `calibrate` (38 params),
   `one-click` (28), `suggest-metrics` (17), `run-eval` (15) show ~2 fields
   in the default view and shove everything else behind a single disclosure.
   The spec promises a curated experience; the implementation delivers the
   raw argparse contents.
3. **File tree clicks open dead tabs.** `openFile` adds a tab
   (`store.ts:439-446`) but `TabContent` only knows three kinds: cli, job,
   workspace. Anything else falls into "This view is coming soon."
   (`App.tsx:71-78`). The backend `GET /api/files/read` exists
   (`api/files.py:91-113`) but no view consumes it. So the entire Files
   sidebar is a tease.
4. **Mode switching is hidden in a "Tweaks" panel that doesn't render.**
   `cliFormMode` (`form` / `preview` / `raw`) is read by `CliFormBody`
   (`CliFormBody.tsx:55-83, 151-211`) but the only place that sets it is
   `setTweak`, which is exposed by the store and never wired to a UI control
   I can find. README says "Three modes: Form, Preview, Raw" — the user
   can't switch among them.
5. **Microcopy reads like a developer's terminal scratch pad.** Examples
   from the code: `// catalog loading…` (`CliCatalog.tsx:99`), `// no runs
   yet` (`RunsList.tsx:34`), `// no jobs yet — run a CLI from the catalog
   to start one` (`JobsList.tsx:218`), `// no output yet — run a CLI to see
   streamed stdout/stderr` (`Terminal.tsx:128`), `workspace settings`
   (lowercase, `SettingsModal.tsx:336`), and `... Running` for the busy
   button (`Workspace.tsx:238`). Claude.ai talks to humans in sentences;
   evalyn talks to itself in code comments.

### Things already done well

1. **Editorial light theme is genuinely beautiful.** Cream paper / deep ink /
   warm orange accent (`styles/index.css:42-68`) is a real design choice, not
   a default. The Instrument Serif headlines (`Welcome.tsx:88-97`) feel
   intentional.
2. **Run history in the Workspace is a clever pattern.** Action panel on top,
   collapsible run cards underneath (`views/Workspace.tsx:339-441`). Each
   card surfaces a one-line "what changed since the last run" diff
   (`RunCard.tsx:110-130`). Pin / Edit / Remove are all reachable. This is
   strictly better than VS Code's "tabs everywhere" approach for this domain.
3. **Per-job WebSocket with auto-reconnect + `last_event_id` resume.**
   `subscribeJob` in `store.ts:505-601` handles reconnect backoff
   (`RECONNECT_DELAYS_MS`) and replays missed events. Most local IDEs don't
   bother. (Subscriber races are real but tracked, see `KNOWN_ISSUES.md`.)
4. **Read-only allowlist for the agent.** `READ_ONLY_ALLOWLIST` (`agent.py:60-82`)
   plus a per-tool-call confirmation gate is the right safety model — a
   non-trivial UX decision delivered well.
5. **`changed: …` chip on the active form.** When current values diverge from
   the most recent run for the same CLI, the header shows a compact
   diff (`Workspace.tsx:169-189`). Easy to miss but hugely useful for
   "what did I change since last time."

---

## Information architecture

### Layout (App.tsx:133-193)

Three columns: `Sidebar | (EditorTabs over content) | ChatPanel?`. There is
no bottom panel in the current build (it's commented as removed in
`App.tsx:1-6`). When the chat is hidden, a floating "✦ Ask agent" pill
appears bottom-right (`App.tsx:171-189`).

### What works

- The chat-on-the-right layout is conventional and correct (Cursor, Codeium,
  Claude.ai use the same shape). 420px (`ChatPanel.tsx:534`) is generous.
- The Workspace as the default view (instead of an empty editor) is a real
  improvement over the IDE archetype.

### Where users get lost

1. **Sidebar tabs duplicate functionality.** The sidebar has four tabs:
   Files, Commands, Eval runs, History (`Sidebar.tsx:22-27`). The Workspace
   already shows run history below the active form (`Workspace.tsx:406-437`).
   "Eval runs" in the sidebar pulls from `/api/runs` (a different concept:
   `RunMeta` from disk) while "History" shows live `Job` rows. The user has
   to learn that there are two different "runs" lists and one of them is
   filesystem-backed.
2. **TitleBar tells you you're "connected" but not where you are.** The
   only contextual hint is the active tab title. There is no project name,
   no `.evalyn/` directory path, no environment indicator
   (`TitleBar.tsx:60-71`). After three minutes you don't know if you're
   running against the right workspace.
3. **The chat panel is always 420px or zero.** No resize, no half-state. On
   a 1366px laptop screen that's 30% of the viewport always devoted to a
   panel that may be empty (`ChatPanel.tsx:531-543`).
4. **There is no breadcrumb between sidebar selection and workspace
   content.** Clicking a CLI in the sidebar swaps the form silently
   (`CliCatalog.tsx:68-73`). The active CLI's id only appears inside the
   "Active command" header bar (`Workspace.tsx:140-192`). A user who clicks
   `analyze` then `compare` won't realize anything changed unless they look
   at the bar.
5. **`cli:`-kind tabs exist but the tab doesn't actually show a CliForm.**
   `App.tsx:62-64` says "if activeTab.kind === 'cli'" we just render the
   Workspace anyway. So the legacy `CliForm.tsx` (158 lines) is reachable
   only via `openCli()` from chat suggestions (`ChatPanel.tsx:271`) — and
   even then the user sees the Workspace, not the CliForm. The `CliForm`
   file is dead code in the live UI.

---

## Interaction inventory

| Surface | What it does | Friction (1-5) | Notes |
|---|---|---|---|
| TitleBar > "Settings" button | Opens SettingsModal | 2 | Label is fine; placement is OK; lacks a gear icon |
| TitleBar > "connected" chip | Static; never updates | 4 | Always green. No actual liveness probe |
| Sidebar tab strip | Switch among Files / Commands / Eval runs / History | 3 | Four panes with overlapping concepts |
| Sidebar collapse rail | Icon-only mode at 48px | 2 | No keyboard shortcut to toggle |
| FileTree row click | Adds a tab; tab body says "coming soon" | 5 | Dead end |
| FileTree icons | `▤ ▶ ⌬ ≡ ·` | 4 | Glyphs are abstract; no tooltip explains them |
| CliCatalog filter input | Filters by id + blurb | 2 | No keyboard focus shortcut, placeholder is `filter…` (lowercase, ellipsis char) |
| CliCatalog alpha toggle | Switches to alphabetical | 3 | Toggle is `▤` / `abc` glyphs with title attr only |
| CliCatalog row click | Selects active CLI | 2 | Works; no preview |
| RunsList row click | Calls `openFile('<id>.run')` | 5 | Same dead end as file tree |
| JobsList row click (open) | Adds a job tab | 2 | Works |
| JobsList Cancel button | POST cancel | 2 | Single icon `⊘`, no confirm |
| Workspace ActiveForm header | Shows CLI name, group, command | 2 | Good |
| Workspace ActiveForm CliSwitcher | Native `<select>` of all CLI ids | 4 | 35 ids in a flat dropdown |
| ParamField (any kind) | Render argparse arg as form control | varies | See deep dive |
| "Show all options" disclosure | Reveals non-essential params | 3 | Disclosure exists but most CLIs put 90% of params here |
| Run button | POSTs `/api/cli/run` | 2 | Disabled while submitting; spinner says `... Running` |
| Copy command button | Writes built command to clipboard | 2 | Says "Copy" instead of an icon |
| Preview command line | Right-side card in `preview` mode | 2 | Good visual; predicted cost is fake math (`CliFormBody.tsx:200-207`) |
| RunCard header | Collapsed summary with status | 2 | Clear |
| RunCard expand | Shows Terminal output + Edit/Pin/Remove | 2 | Solid |
| RunCard Edit | Re-loads args into active form | 2 | Good |
| RunCard Remove | Cancels job + drops from history | 3 | No undo, no confirm |
| ChatHeader new conversation | Resets thread | 3 | Glyph `＋` (full-width plus) is unusual |
| ChatHeader settings gear | Opens SettingsModal | 2 | Good |
| ChatHeader close | Hides chat | 2 | Glyph `×` |
| Chat composer | Textarea + send | 3 | "↑" arrow as send button is unconventional |
| Chat tool-call card | Renders tool status, output, error | 3 | Approve/Reject share a single global pendingConfirmation |
| Chat suggestion card | Opens a CLI tab | 3 | Tab opens but tab content is the Workspace, not the cli view; pre-fill goes through `sessionStorage` (`ChatPanel.tsx:266-272`) — fragile |
| Chat error banner | Shows provider error + Settings link | 2 | Good |
| SettingsModal provider row | Expand, paste key, pick model, test | 3 | Two lowercase labels; no help text on ollama base url |
| SettingsModal Save key | Sends key, clears input | 3 | No "Saved" confirmation toast |
| SettingsModal Test | 1-token call | 2 | Pill changes color |
| Cmd/Ctrl-K palette | Opens / closes a palette | 5 | The palette is wired up (`App.tsx:117-129`) but no UI renders for it (`paletteOpen` is only read by setters). Keyboard shortcut is a lie. |
| Esc | Closes palette | 5 | Same |
| Ask-agent floating pill | Bottom-right when chat closed | 2 | Good |

Median friction across 30+ surfaces: ~3 (workable but unrefined). Five
surfaces are 5/5 (broken or misleading).

---

## Friction deep-dives

### A. Welcome view is fake (Welcome.tsx)

**Where:** `dashboard/frontend/src/views/Welcome.tsx:31-70`.

**What the user sees:** Six cards. The first ("Latest run · 82dddcc3 · −4
pts") is hardcoded mock data. The hint at the bottom suggests asking the
agent "Why is gemini regressing?" — also hardcoded. The five other cards
are real (they call `selectActiveCli`).

**Why it hurts:** A new user opens `evalyn dashboard`, sees a confident
"Latest run · 82dddcc3 · −4 pts," clicks it, and gets a generic
"coming soon" placeholder. Trust is gone in five seconds. The fictional
gemini suggestion implies the dashboard knows things about the workspace it
doesn't.

**Directional fix:** Drive the first card from the actual most recent run
(`store.runs[0]`). If `runs` is empty, replace the entire grid with an
onboarding flow: "Run `evalyn quickstart` to capture your first traces" or
"Drop a `dataset.jsonl` into `.evalyn/` to begin." Replace the hardcoded
agent prompt with a rotating set of suggestions derived from what the user
actually has (no runs -> "How do I get started?"; has traces but no dataset
-> "Build a dataset from my recent traces"; has runs -> "Compare my last two
runs and tell me what regressed.").

### B. CLI catalog overwhelms (CliCatalog.tsx + ESSENTIAL coverage)

**Where:** `dashboard/frontend/src/components/CliCatalog.tsx:60-163`,
introspection in `dashboard/evalyn_dashboard/introspect.py:203-259`. ESSENTIAL
declarations live in `sdk/evalyn_sdk/cli/commands/*.py`.

**Counts (from running `build_catalog()`):** 35 CLIs across 11 groups.
Param distribution: `calibrate` 38, `one-click` 28, `suggest-metrics` 17,
`run-eval` 15, `build-dataset` 14. Out of 35 CLIs, only 4 have any
`ESSENTIAL` set declared (clustering, dataset, traces, evaluation). The
other 31 will render with only `required=True` params in the default form
view, hiding everything else behind a single "Show all options (38)"
disclosure.

**What the user sees:** Click `calibrate` -> form shows 2 fields
(the required ones) and a button labelled "Show all options (36)". Click
that, get a wall of 36 ungrouped fields with sparse help text (the help
strings come from argparse `help=`, which are mostly terse like `"default: 4"`).

**Why it hurts:** The dashboard's value-add over `evalyn calibrate --help` is
supposed to be progressive disclosure and good defaults. Right now it's
largely the same wall of flags wrapped in a UI.

**Directional fix:** Two parts.
1. Either populate `ESSENTIAL` for every command module (data work in the
   SDK, not the dashboard) or auto-derive an essential set from the
   command-module docs / spec.
2. Group params semantically inside the form. argparse already supports
   argument groups (`add_argument_group("LLM judges")`); the introspector
   currently flattens these (`introspect.py:118-146`). Preserve the group,
   render each as a collapsible section with a one-line "what this is for"
   header.

### C. CliForm tabs don't work; chat suggestions silently fail (App.tsx:62-64)

**Where:** `dashboard/frontend/src/App.tsx:54-79`,
`ChatPanel.tsx:258-309`.

**Code path (read):** `SuggestionCard.handleClick` calls `openCli(cliId)`.
`openCli` (`store.ts:434-437`) adds a tab with `kind: 'cli'`. `TabContent`
checks `if (!activeTab || activeTab.kind === 'cli') return <Workspace />`.
So the tab opens, the title strip changes, but the visible content stays on
the Workspace — and the Workspace's active CLI is whatever the user last
clicked, not the suggested one.

**What the user sees (infer):** Agent finishes a turn with a "Open run-eval"
suggestion card. User clicks. A new tab labelled `run-eval` appears in the
strip. Nothing else changes. They'd swear the button did nothing.

**Why it hurts:** The flagship moment of the agentic loop — "agent thinks
about your evals, hands you a pre-filled command to launch" — is broken at
the seam where it transitions from chat to action.

**Directional fix:** Either (a) re-render the Workspace using
`tabs.find(activeTabId).cliId` instead of the global `activeCliId`, and
have `openCli` set `activeCliId` itself; or (b) actually mount `CliForm`
when `activeTab.kind === 'cli'`. Path (a) is closer to the current spirit;
path (b) requires consolidating the legacy CliForm and the ActiveForm.

### D. The palette doesn't exist (App.tsx:116-129)

**Where:** `App.tsx:117-129` listens for Cmd/Ctrl-K and toggles
`paletteOpen`. Nothing renders that state. Welcome view advertises `⌘K
palette` (`Welcome.tsx:168-171`); chat composer shows `↵ send · ⌘K
palette` (`ChatPanel.tsx:498-500`).

**Why it hurts:** Users hit ⌘K (a near-universal pattern) and nothing
appears. Worse, the UI tells them the palette exists.

**Directional fix:** Either build a real command palette (search across
CLIs, runs, files, recent jobs — like Linear/Cursor) or remove the
shortcut and the kbd hints. Building it is the richer path.

### E. Mode switcher is invisible (CliFormBody.tsx + store.ts)

**Where:** `cliFormMode` is in `tweaks` (`store.ts:32-48`). `setTweak` is
exposed but `tweaksOpen` is also a flag with no consumer. There is no
visible toggle in TitleBar, Sidebar, Workspace, or ChatPanel. The README
brags about three modes; the UI ships with whatever default the tweak sits
on (`preview`).

**Why it hurts:** The "raw command" textarea (`CliFormBody.tsx:55-83`) is a
power-user escape hatch. It exists. It's unreachable.

**Directional fix:** Three-segment control next to the Run button:
"Form / Preview / Raw". One CSS class change.

### F. The bottom panel and BottomPanel.tsx (BottomPanel.tsx)

**Where:** `dashboard/frontend/src/components/BottomPanel.tsx` exists (87
lines) but is not imported in `App.tsx` (`grep BottomPanel App.tsx`
returns nothing). The Terminal/Jobs/Problems split is dead code. JobsList is
mounted only via the Sidebar's "History" tab.

**Why it hurts:** Reading the README + spec, a user expects a persistent
bottom panel ala VS Code. There is none. Their job either has its own tab
(if they opened it via JobsList `↗`) or appears inline inside its
RunCard. Discovering this takes a while.

**Directional fix:** Decide. Either delete BottomPanel.tsx and update the
docs, or restore it as the persistent "what's running right now" surface.

### G. SettingsModal is functional but cold (SettingsModal.tsx)

**Where:** `SettingsModal.tsx:257-364`.

**What the user sees:**
- Modal title is `workspace settings` (lowercase) in serif italic.
- Body copy: `Choose a model provider for the agent. API keys are stored
  on your machine in ~/.evalyn/credentials.json (mode 600) and never sent
  to the browser.` That's actually pretty good — concrete, security-aware.
- Each provider row has a status pill that defaults to literal text
  "untested" (`TEST_PILL_TONE` map, `SettingsModal.tsx:28-33`). The label
  itself is shown in a chip — feels like a debug overlay.
- API key save shows `Saving…` then nothing. No checkmark. No "Saved at
  HH:MM" timestamp.
- Test button responds with a colored pill. If it fails, the error message
  shows the raw provider error (good!) but truncates nothing for long
  errors.
- Ollama row: there is no UI for the base URL. The README says you can set
  it, but the saveProvider payload only accepts `api_key` and `model`
  (`api.ts:117-124`). Users on a non-default Ollama port have to edit the
  credentials file by hand.

**Why it hurts:** Setup is the most stressful 90 seconds of using a new
tool. The modal shows status but doesn't reassure. Ollama users in
particular are silently locked out of the in-app config path.

**Directional fix:** Add a saved-key timestamp ("Last updated ago 3m"). On
successful Test, replace the pill with a one-line summary
("Connected. Sample completion: 'Hello!' in 312ms.") so the user has
proof, not just a green dot. Add an `Advanced` disclosure with the Ollama
base URL.

### H. Chat: confirmation buttons share state (ChatPanel.tsx:204-219, KNOWN_ISSUES.md #4)

**Where:** Already documented as `KNOWN_ISSUES #4`. ToolCallCard's
Approve/Reject buttons call `confirmAgent(true|false)` without passing the
specific `call.id`. With multiple stale awaiting cards, clicking any of
them confirms the currently-pending one.

**Why it hurts:** Imagine a flow where the agent proposes Tool A (you
hesitate), then proposes Tool B (now pending). The Tool A card still shows
Approve. You click "Approve" on A — you've actually approved B.

**Directional fix:** Plumb `tool_call_id` through `confirmAgent` and
validate it server-side. (Sketch exists in KNOWN_ISSUES; not yet shipped.)

---

## Microcopy audit

| Where | Current text | Suggested rewrite |
|---|---|---|
| `TitleBar.tsx:62` | `connected` (chip) | Remove the chip, or say `Local · 7401` so users know what they're connected to |
| `Welcome.tsx:97` | `Run and analyze evaluations.` | Keep — this is good |
| `Welcome.tsx:107-109` | `Pick a command from the left sidebar to fill out and run, or ask the agent on the right a question - it will pick the right command for you.` | `Pick a command from the sidebar, or just describe what you want and let the agent pick.` |
| `Welcome.tsx:165-166` | `Try the agent: "Why is gemini regressing?"` | Rotate suggestions based on workspace state. Empty workspace: `Try: "Help me get started with evalyn."` |
| `CliCatalog.tsx:99` | `// catalog loading…` | `Loading commands...` (drop the JS comment styling) |
| `CliCatalog.tsx:109` | `filter…` (placeholder) | `Search commands` |
| `CliCatalog.tsx:120` | title `By group` / `Alphabetical` | Visible label, not just title attr |
| `Workspace.tsx:158` | `Active command` (eyebrow) | Keep |
| `Workspace.tsx:238` | `... Running` (button label) | `Running...` (don't lead with ellipses) |
| `Workspace.tsx:242-243` | `preview:` (label before code) | Drop the label; the styled code block reads as preview |
| `Workspace.tsx:323` | `Pick a command from the sidebar to start filling out a form.` | `Choose a command from the left to begin.` |
| `Workspace.tsx:429` | `No runs yet. Fill the form above and click Run.` | `Your runs will appear here.` (less imperative) |
| `RunsList.tsx:34` | `// no runs yet` | `No eval runs yet.` |
| `JobsList.tsx:218` | `// no jobs yet — run a CLI from the catalog to start one` | `No jobs yet.` |
| `Terminal.tsx:128` | `// no output yet — run a CLI to see streamed stdout/stderr` | `Output will appear here once the job starts.` |
| `Terminal.tsx:135-144` | Faux blinking `$` cursor | Remove. Cargo-cult terminal aesthetics |
| `ChatPanel.tsx:362` | `Ask <em>agent</em>` (header) | `Ask Evalyn` or `Chat` |
| `ChatPanel.tsx:493` | `Ask anything · paste a CLI · @-mention a run` | `Message Evalyn...` (Claude.ai's pattern — short, friendly) |
| `ChatPanel.tsx:498-500` | `↵ send · ⌘K palette` | Drop the palette line until palette ships |
| `ChatPanel.tsx:553-557` | `// Ask anything about your evals.` `// Tools call CLIs; writes require your approval.` | `Ask anything about your evaluations. The agent can run any read-only command on its own; it will check with you before any write.` |
| `ChatPanel.tsx:201` | `This command writes to disk. Approve to run?` | `Evalyn wants to run this command. It will write to your project. Continue?` |
| `ChatPanel.tsx:574-576` | `thinking …` | `Thinking…` (capitalized, single line) |
| `SettingsModal.tsx:336` | `workspace settings` (lowercase) | `Settings` — match the title bar |
| `SettingsModal.tsx:165` | `•••••••• (set)` placeholder | `Key already saved. Paste a new one to replace.` |
| `App.tsx:187` | `✦ Ask agent` (floating button) | `Chat with Evalyn` (consistent identity) |
| `BottomPanel.tsx:31` | `Problems · 0` tab label | If there's no Problems pane, remove the tab |

The dominant voice problems: (a) lowercase serifs read as art-school, not
warm; (b) `// JS comments` as empty states; (c) overuse of CLI metaphors
("paste a CLI", "command from the catalog") to describe a UI that exists
specifically so the user doesn't have to think in CLI terms.

Claude.ai's voice is **direct, full-sentence, lowercase only when
intentional, no jargon**. Evalyn's voice is **terse, abbreviated, technical,
inconsistent capitalization**. They're not the same product yet.

---

## Visual / aesthetic audit (styles/index.css)

### Tokens

- Three families loaded via Google Fonts: Geist (sans), Geist Mono, Instrument
  Serif (`index.css:8`). Three families is a lot; bundle weight aside, the
  serif appears in title-only contexts (TitleBar brand, ChatHeader, modal
  titles, Welcome H1) where it works, but the `i` italics inside chat header
  ("Ask <i>agent</i>") and the lowercase serif "workspace settings" both
  feel decorative rather than purposeful.
- Color palette is well-tuned. Dark theme uses near-black `#0a0c10` and a
  warm `#ff7a3d` accent; light theme uses cream `#f4ebe1` and burnt orange
  `#c44918`. Text scale (`--text-0` through `--text-3`) gives four reliable
  greys.
- Pass/fail/warn/info colors are muted (sage `#7fc6a0`, salmon `#e88072`).
  This reads as editorial — pleasant — but `--fail #e88072` against
  `--bg-2 #fbf6ec` (light) has contrast around 3.5:1, below WCAG AA for
  normal text.
- Body font-size is **13px** (`index.css:83`), with most chrome at 11-12px
  and `kbd` / `chip` at 10px (`index.css:142, 274`). This is *dense*. Claude.ai
  uses 16px body. The typography is fine for an IDE but unfriendly for a
  product that wants to feel approachable.

### Density and spacing

- 18px section padding (`Workspace.tsx:193`), 14px row padding
  (`SettingsModal.tsx:103`), 8-12px gaps almost everywhere. Compare to
  Claude.ai's generous 24-32px gutters. Evalyn feels *compressed*.
- Cards use 10px border-radius for primary surfaces (Workspace ActiveForm)
  and 6px for chrome. Reasonable but not soft enough to feel friendly.
  Claude.ai's chat surfaces are 14-20px.

### Motion

- Button transitions: `background 0.12s, border-color 0.12s` (`index.css:194-197`).
  Bar fill: `width 0.3s ease`. That's all. No micro-interactions on tab
  switches, no fade-in on streamed messages, no stagger on initial render. The
  app feels static.

### Iconography

- Icons are Unicode glyphs throughout (`▤`, `$`, `▶`, `◷`, `⌬`, `≡`, `▾`, `▸`,
  `✦`, `⊘`, `↗`, `↑`, `＋`, `×`, `⎘`, `⚙`). No icon library. Glyph rendering
  varies wildly across OS/browsers (e.g. `▤` is empty on some Linux fonts).
  This will look broken for a non-trivial fraction of users.

### Verdict

The aesthetic is *art-directed* but not *polished*. The light theme is
beautiful in screenshots; the actual app feels like a 2008 IDE with a
cream paint job. Two fixes would close most of the gap: (1) bump body to
14-15px and gutters to 20-28px, (2) swap Unicode glyphs for a real icon set
(Phosphor / Lucide, ~400KB tree-shaken).

---

## Accessibility quick scan

- **Keyboard.** Cmd/Ctrl-K and Escape (palette, broken). No global shortcut
  for Run, send chat, focus search. No `tabindex` ordering audit.
- **Focus states.** Inputs get `border-color: var(--accent)` on focus
  (`index.css:246-249`). Buttons rely on browser default outlines (no
  custom `:focus-visible` rule). Bare `<div role="button" tabIndex={0}>`
  used for Welcome cards (`Welcome.tsx:122-126`) — focus ring will be a
  default thin browser outline that disappears against the dark theme.
- **ARIA.** Mixed. `role="alert"` on error banners (`Workspace.tsx:217`),
  `role="dialog" aria-modal="true"` on the settings modal (`SettingsModal.tsx:293-296`),
  `role="treeitem"` on file tree rows (`FileTree.tsx:45`). But the
  CliCatalog rows are clickable `<div>`s without role/aria-selected
  (`CliCatalog.tsx:22-43`). The RunCard expand toggle is a `<button
  aria-expanded>` — good (`RunCard.tsx:80-84`).
- **Color contrast.**
  - Light theme `--text-3 #8e8479` on `--bg-0 #f4ebe1` = ~3.4:1, fails AA
    for body text. Used heavily for "hint" / metadata.
  - Dark theme `--text-3 #5b6678` on `--bg-0 #0a0c10` = ~4.2:1, marginal.
  - `--accent #ff7a3d` on dark `--bg-0 #0a0c10` = ~5.1:1, passes AA.
- **Screen reader.** The Welcome H1 uses serif at 56px but no semantic
  landmarks (`<main>`, `<nav>`, `<aside>`) anywhere. The chat panel is a
  proper `<aside>` (`ChatPanel.tsx:531`); the sidebar is a `<div>`.
- **Reduced motion.** No `prefers-reduced-motion` media query in index.css.
  (Currently low-impact because there's barely any motion — but this will
  matter once microinteractions are added.)

Net: the app passes a basic functional bar but would fail an accessibility
audit. None of these are hard fixes.

---

## Strengths to preserve

1. **Action + Run History pattern.** `Workspace.tsx` is a real product
   insight: keep the form persistent at the top, accumulate runs underneath.
   Don't let an IDE-redesign throw this away.
2. **Editorial color palette.** The cream/ink/burnt-orange light theme has
   a point of view. Don't homogenize it into Tailwind grey.
3. **`changed:` chip on the active form.** Tiny, useful. Most tools don't
   do this.
4. **Per-run inline Terminal output.** Better than a global terminal that
   conflates jobs.
5. **Read-only allowlist + confirmation gate for the agent.** The right
   model. Keep it.
6. **The architecture (subprocess + WS + introspection).** The shape lets
   the dashboard track the CLI without duplication. Don't replace it.

---

## Recommended priority list

### P0 (ship within next iteration; user-trust-blocking)

1. Fix the Welcome view: drive cards from real `runs`; replace the fake
   `82dddcc3` card; replace the canned agent suggestion with workspace-aware
   prompts; add a true onboarding state for empty `.evalyn/`.
   (`Welcome.tsx`, ~half-day)
2. Fix the chat-suggestion -> CliForm flow. Either render `CliForm` for
   `cli:` tabs or have `openCli` set `activeCliId` and stop opening a tab.
   (`App.tsx:62-64`, `store.ts:434-437`, ChatPanel suggestion path,
   ~half-day)
3. Make file tree clicks do something real, or hide the Files tab.
   (`store.ts:openFile`, App.tsx TabContent, ~1 day to add a file viewer
   tab body that calls `/api/files/read`)
4. Plumb `tool_call_id` through chat confirmation so stale cards don't
   approve the wrong action. (KNOWN_ISSUES #4, ~2 hours)
5. Remove or implement the Cmd/Ctrl-K palette. The shortcut is currently a
   lie. (~30 min to remove; ~2 days to implement properly)

### P1 (high-value polish)

6. Microcopy pass across every empty state, button label, and modal.
   Replace `// JS comments` with sentences. Drop "command", "CLI" noise
   from user-facing copy where the user just wants to *do a thing*.
7. Add a visible CLI-form mode toggle (Form / Preview / Raw segmented
   control). The infrastructure exists.
8. Populate `ESSENTIAL` for the remaining 11 command modules — or move to
   argparse argument groups and have the introspector preserve them so the
   form can render sections instead of one disclosure.
9. Add an Ollama base-URL field to SettingsModal and the saveProvider API.
10. Replace Unicode glyphs with a real icon set.
11. Bump body font to 14-15px; bump section padding to 20-24px. Make the
    UI breathe.
12. Add focus-visible styles, fix `--text-3` contrast on light theme,
    audit `tabindex` ordering, add semantic landmarks (`<main>`, `<nav>`).

### P2 (richer experience)

13. Real command palette: search across CLIs, runs, files, recent commands.
14. Persistent project context in the TitleBar (workspace path, model in
    use, current `.evalyn/` size).
15. Resizeable chat panel with a Splitter; let users half-and-half it.
16. Streaming chat motion (fade-in tokens, subtle stagger) — the kind of
    detail that makes Claude.ai feel alive.
17. Group params semantically inside the form (use argparse argument groups
    or per-CLI manifest); collapsible sections with one-line headers.
18. SettingsModal: save toast + "last test passed at" timestamp.
19. Sidebar: collapse "Eval runs" and "History" into one tab with filters,
    and reconcile with Workspace run history (one mental model, not three).

---

## Closing observation

The dashboard is a competent IDE port of an existing CLI — accurate, complete,
and built carefully. What it isn't yet is a *product*. Claude.ai's polish
isn't about more pixels; it's about *fewer surfaces that lie or disappoint*.
Right now Evalyn ships at least five surfaces that say one thing and do
another (Welcome card, Files tab, palette shortcut, mode switcher, chat
suggestion -> form). Closing those gaps — before adding any new feature —
is the path to "feels like Claude.ai."
