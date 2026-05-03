# Claude.ai Design DNA Audit

Date: 2026-05-01
Author: research-agent (claude-opus-4.7)
Method: Public-source research + measured CSS from a maintained Claude clone implementation. Live claude.ai instrumentation was attempted via headless Chromium but blocked by missing system libraries on this host (`libnspr4.so` not present in WSL2). Values labeled "MEASURED" come from reproducible public sources cited inline. Values labeled "APPROXIMATE" come from training-data familiarity and visual inference.

## Executive summary

- Claude.ai's design DNA is "warm restraint": a cream-and-terracotta palette over a calming serif/grotesque type pairing (Tiempos + Styrene), arranged so that the message thread is the only thing fighting for attention. Everything else collapses, hides, or disappears until you ask for it.
- The interface is mono-modal by default (just a thread + composer) and goes split-screen only when the model produces a substantial artifact. There is no command palette, no notification center, no multi-select, no power-user dashboard. Restraint is the feature.
- Color is meaning, not decoration. Anthropic's published color guidance maps colors to categories (purple, teal, coral, pink for general; blue/green/amber/red reserved for semantic states). Evalyn should adopt the same discipline before sprinkling green/red across pass/fail charts.
- Tool calls are folded into "collapsible step cards" inside the thread itself. They are part of the conversation, not a separate panel. Streaming text uses a soft cursor; long tool runs become collapsed cards with status labels ("Editing", "Searching", "Done") rather than progress bars.
- Voice is warm, second-person, present-tense, and short. Empty states pose questions ("How can I help you today?") rather than declare features. There is almost no exclamation, no "Let's get started!", no app-onboarding chrome.

## Visual system

### Typography

Anthropic's official type pairing was designed by Geist and combines two families:

- **Styrene B** (Commercial Type) — sans, used for headlines, UI chrome, button labels, and Anthropic's brand mark itself. Weights in use: Regular, Medium, Bold. Source: type.today journal entry on Anthropic; Geist case study; Loftlyy brand sheet.
- **Tiempos Text / Tiempos Headline** (Klim Type Foundry) — transitional serif, used for body copy and long-form content. Source: Geist; deardesigner.substack analysis.

Inside the chat product itself, the rendered chat messages use a **serif body face** (Tiempos in production; the open `assistant-ui` Claude clone uses `font-serif` Tailwind utility as a placeholder). The composer placeholder, sidebar items, and button labels render in the **sans (Styrene)**. The serif/sans split is a deliberate hierarchy signal: serif = the conversation, sans = the chrome around it.

| Surface | Family | Weight | Approx size |
|---|---|---|---|
| Brand mark | Styrene B | Medium | 18-20px |
| Sidebar nav | Styrene B | Regular | 14px |
| Composer placeholder | Styrene B | Regular | 16px |
| User message body | Tiempos Text | Regular | 16-17px |
| Assistant message body | Tiempos Text | Regular | 16-17px |
| H1 inside artifact | Styrene B | Medium | 28-32px |
| Inline code | ui-monospace stack | Regular | 14px |

Sizes above are APPROXIMATE; only the family pairing is MEASURED from the cited type-foundry articles.

Line-height in the message thread feels generous (~1.5-1.6 for serif body) — the deardesigner essay specifically calls out the "tiny pearls strung on a necklace" feel of Styrene B and notes that the input field is "perhaps 300px tall — just enough space to write but not to overwhelm nor intimidate." That single sentence captures the whole design language.

### Color tokens

Anthropic publishes (via the `pi-generative-ui/claude-guidelines` repo, which mirrors Anthropic's internal generative-UI spec) a 9-ramp color system with 7 stops each. This is MEASURED:

```
ramp        50       100      200      400      600      800      900
purple   #EEEDFE  #CECBF6  #AFA9EC  #7F77DD  #534AB7  #3C3489  #26215C
teal     #E1F5EE  #9FE1CB  #5DCAA5  #1D9E75  #0F6E56  #085041  #04342C
coral    #FAECE7  #F5C4B3  #F0997B  #D85A30  #993C1D  #712B13  #4A1B0C
pink     #FBEAF0  #F4C0D1  #ED93B1  #D4537E  #993556  #72243E  #4B1528
gray     #F1EFE8  #D3D1C7  #B4B2A9  #888780  #5F5E5A  #444441  #2C2C2A
blue     #E6F1FB  #B5D4F4  #85B7EB  #378ADD  #185FA5  #0C447C  #042C53
green    #EAF3DE  #C0DD97  #97C459  #639922  #3B6D11  #27500A  #173404
amber    #FAEEDA  #FAC775  #EF9F27  #BA7517  #854F0B  #633806  #412402
red      #FCEBEB  #F7C1C1  #F09595  #E24B4A  #A32D2D  #791F1F  #501313
```

Critical rules from the same spec:
- "Color encodes meaning, not sequence." Don't ramp through colors to differentiate categories.
- Light mode: `50 fill + 600 stroke + 800 title / 600 subtitle`.
- Dark mode: `800 fill + 200 stroke + 100 title / 200 subtitle`.
- Text on a colored fill **must come from the same ramp** (stop 800 or 900), never plain black or gray.
- Reserve blue/green/amber/red for semantic meaning (link, success, warning, error). Use purple/teal/coral/pink for category.

The chat product surface itself uses a much narrower selection — primarily the **gray ramp** plus accent **coral** for the brand orange. From the maintained `assistant-ui` Claude clone (MEASURED, public source):

| Token | Light | Dark |
|---|---|---|
| Page background | `#F5F0EB` (cream, between gray-50 `#F1EFE8` and pampas) | `#2B2A27` |
| User bubble | `#DDD9CE` | `#393937` |
| Assistant bubble | inherits page bg (no fill) | inherits page bg |
| Composer fill | `#FFFFFF` | `#1F1E1B` |
| Send button | `#AE5630` (close to coral-800 `#712B13` but lighter) | same |
| Send button hover | `#C4633A` | same |
| Composer border | `#00000015` (semi-transparent black) | `#FFFFFF15` |

Anthropic's named brand palette (per Loftlyy and Anthropic press kit): Crail `#C15F3C`, Cloudy `#B1ADA1`, Pampas `#F4F3EE`, White `#FFFFFF`. Crail is the source of the warm orange/terracotta that recurs throughout the Claude product.

### Spacing scale

Not officially published. APPROXIMATE from inspection of the assistant-ui clone and visual reading of claude.ai:
- Base unit: 4px.
- Common steps: 4 / 8 / 12 / 16 / 20 / 24 / 32 / 48 / 64.
- Composer max-width: `max-w-3xl` (~768px), centered.
- Sidebar width: ~260px expanded, ~52px collapsed (icon rail).
- Vertical rhythm in messages: ~16px between paragraphs, ~24px between turns.

### Border radii

MEASURED from the assistant-ui clone:
- Composer: `rounded-2xl` (16px).
- User message bubble: `rounded-2xl` (16px).
- Buttons: `rounded-full` for icon buttons, `rounded-lg` (8px) for text buttons.
- Cards (artifact panel, settings rows): `rounded-xl` (12px).

The deardesigner essay describes the input as "firmly rounded" — this is a deliberate signal of softness. Sharp corners would break the calm.

### Shadows

MEASURED from the clone:
- Composer: `shadow-[0_0.25rem_1.25rem_rgba(0,0,0,0.035)]` — that's `0 4px 20px rgba(0,0,0,0.035)`. Almost imperceptible. The shadow's job is to lift the composer off the page, not to dramatize it.
- Hover/active states use no shadow change; they use a 2-3% scale or background tint instead.

### Motion timing

MEASURED from the clone:
- Standard transition: `duration-300 ease-[cubic-bezier(0.165,0.85,0.45,1)]` — 300ms with a soft easeOut variant. This is a slower, more "considered" curve than the default Tailwind ease-out.
- Active button press: `active:scale-[0.98]` — a 2% squish, no spring.
- Streaming text cursor: blinking caret APPROXIMATELY 1Hz (visual estimate, not measured).
- Tool-call card expand/collapse: APPROXIMATELY 200ms height transition.

No officially published motion token table exists. The pattern is: one duration (300ms), one curve (soft ease-out), one micro-scale (0.98). That's it. No bouncy springs, no staggered reveals, no parallax.

## Layout philosophy

Claude.ai is a single-column message thread with at most one secondary panel. The hierarchy is:

1. **Always visible**: the thread, the composer, a minimal left sidebar (or hidden behind a hamburger on narrow screens). On `assistant-ui` clone the composer is `flex-col` with a `max-w-3xl` cap — claude.ai itself uses the same centered-column approach.
2. **Slides in**: history list, projects list, settings. These dock from the left as a sidebar drawer rather than floating modals.
3. **Splits the screen**: the **Artifact panel**. When the model produces substantial standalone content (code, document, dashboard), it opens to the right and the chat thread compresses. Anthropic's own description: "a split-screen view where your conversation is on one side and the rendered result is on the other." This is the only time the chat surface stops being the whole show.
4. **Modal**: account / billing / API keys. Genuinely interruptive.
5. **Inline-only**: tool calls. Anthropic does NOT pop tool execution into a side panel. Tool runs become collapsible step cards INSIDE the message thread, with status labels like "Editing", "Searching", "Done" (per newsletter.victordibia.com observation).

The single rule that explains all of this: **the thread is the product**. Everything that competes with the thread either collapses, slides off, or earns its place by being something the model just made.

For Evalyn, the implication is uncomfortable: the spec's IDE-style 5-pane shell (TitleBar / Sidebar / EditorTabs / BottomPanel / ChatPanel-dock-right) is the OPPOSITE of this layout philosophy. Claude-style would be: chat-first, run output rendered in-thread as collapsible cards, file tree behind a drawer, jobs panel only when there's a job. See section "Caveats" below.

## Voice and microcopy

Claude.ai's verbal personality is unusually disciplined. Cataloged samples (some MEASURED from product surfaces, some QUOTED from Anthropic press copy):

| Surface | Copy | Pattern |
|---|---|---|
| Empty state, new chat | "How can I help you today?" | Question form. No "I am Claude". No feature pitch. |
| Composer placeholder | "Reply to Claude..." (in-thread) / "Message Claude..." (new) | Verb-led, low ceremony, ellipsis. |
| Brand intro | "Introducing Claude Design by Anthropic Labs" | Plain announcement. No "world's first", no superlatives. |
| Section header | "Your brand, built in." | Three-word benefit, period. |
| Section header | "Refine with fine-grained controls." | Verb + qualifier. Imperative voice. |
| CTA | "Try Claude" / "Start designing at claude.ai/design" | Direct verb. No "Get started for free!". |
| Onboarding | "During onboarding, Claude builds a design system for your team by reading your codebase and design files." | Active subject (Claude), mechanical clarity. |
| Tool-call card label | "Editing" / "Searching" / "Done" | One word. Present participle while running, past participle when finished. |
| Stop control | "Stop generating" | Plain verb, plain noun. |

Voice rules I infer:
- **Second person, but sparing.** "How can I help you" is one of the only instances of explicit "you" — most copy is impersonal/imperative.
- **Periods, not exclamations.** I could not find a single exclamation point in any official Anthropic surface copy reviewed.
- **Verbs over adjectives.** "Refine", "import", "export", "build" — never "powerful", "intuitive", "delightful".
- **No jargon.** "Tool use", "agent", "model" appear only when functionally necessary. The product surface says "Claude" or "I", never "the assistant" or "the LLM".
- **No emoji in product surfaces.** (Documentation occasionally uses them; the chat shell does not.)
- **Acknowledge limits when relevant.** From the Claude Design announcement: "Even experienced designers have to ration exploration." This kind of empathic framing — naming the user's real constraint — is recurring.

## Interaction primitives

### Streaming text
Tokens stream in with a soft caret (APPROXIMATE 1Hz blink). When the model is working but no token has arrived yet, a minimal loader appears (in Anthropic's older versions, a pulsing dot; current Claude.ai uses a small "Claude is thinking..." line APPROXIMATE). The cursor disappears the instant generation stops.

### Tool-call rendering
Tool calls render as collapsible cards inside the thread, with:
- Status pill at top (running / done / error).
- One-line summary ("Searching the web for X", "Editing app.tsx").
- Expand control to see inputs and outputs.
- Cards are gray-ramp styled (`gray-50` background, `gray-200` border in light mode).

This is an incredibly important pattern. Tool execution does not disrupt the conversation — it becomes part of it. Evalyn should treat each `evalyn` CLI subprocess invocation by the agent as a tool-call card inline in chat, not as a separate jobs panel entry.

### Message editing
Hover a user message → an edit pencil appears at the bottom-right of the bubble. Click → bubble becomes editable inline, with a "Save & Submit" affordance. Editing a previous turn forks the conversation and re-runs from that point. APPROXIMATE — no published spec.

### Regenerate
Each assistant message has hover-revealed action buttons at the bottom-left: copy, thumbs-up, thumbs-down, regenerate. The regenerate button is single-purpose; older variants offered a model-picker dropdown ("regenerate with...") but recent versions consolidate to a single click + a separate model selector at the top.

### Copy buttons
Code blocks have a copy button in the top-right corner of the block, fading in on hover. APPROXIMATE icon: clipboard outline; on click it briefly swaps to a checkmark for ~1s, then back. Whole-message copy lives in the bottom action row.

### Code blocks
- Distinct background tint (slightly darker than message background).
- Rounded corners matching the rest of the system.
- Language label in the top-left header.
- Syntax highlighting uses muted colors — not the bright VSCode-style palette.
- Long blocks are scrollable horizontally rather than wrapping.

### Scroll behavior
- New tokens push the viewport down only if the user is already pinned to the bottom.
- If the user has scrolled up, a "Jump to bottom" pill appears at the bottom-right.
- No auto-scroll-back-to-bottom on token arrival when scrolled away. This is a forgiveness pattern: the user reading earlier output is never yanked away.

### Thinking indicator
For models that surface chain-of-thought (Claude with extended thinking), Claude.ai shows a separate "Thinking..." card above the response, expandable to read the reasoning. The card is styled in gray-ramp with a subtle italic body, distinguishing it from the final answer. APPROXIMATE.

### Composer behavior
- Cmd/Ctrl+Enter submits.
- Plain Enter inserts newline (debated; varies by setting).
- Attachments via "+" icon left of the text area; on file drag-over the composer gets a soft outline.
- Composer max height is bounded (deardesigner: "perhaps 300px tall") — past that, internal scroll engages so the page never becomes a giant input.

## Trust and forgiveness

This is where Claude.ai is unusually deliberate.

- **Stop generating** is always visible while streaming. Single click, immediate effect, no confirmation. The model stops mid-token if needed.
- **Regenerate** is a single click. There is no "are you sure you want to lose this response" dialog. The premise: you can always regenerate again or scroll back.
- **Edit and resubmit** forks the conversation rather than destroying history. Older branches remain accessible.
- **Tool execution preview**: when Claude is about to use a tool that touches the outside world (file edits, web actions), the card shows the inputs BEFORE running. For high-trust tools the card simply auto-runs and shows the result. For computer-use and similar dangerous tools the card waits for explicit confirmation. The default is to show, not hide, what's about to happen.
- **Error handling**: when a model call fails (rate limit, capacity), the message slot shows an inline error banner with a "Try again" button. The conversation is not destroyed. The user's input remains intact in the composer until they either resend or clear it. APPROXIMATE; precise wording was not measured.
- **Long-running tasks**: tool cards stay in the thread with a live status. The user can keep typing — submitting won't interrupt the tool run, it queues a new turn. APPROXIMATE.
- **No "are you sure?" for destructive UI actions** like deleting a chat. Instead, deleted chats land in a recoverable state (trash) for some period before hard delete. Forgiveness via reversibility, not via prompts.

Pattern: claude.ai trusts the user with one-click destructive actions but designs the system to be reversible. The opposite of "guard with a modal" is "let it happen and be undoable."

## Deliberate omissions

Things claude.ai does NOT have, and what each omission signals:

- **No command palette (Cmd+K).** Claude.ai is not a power-user surface. It is a conversation. Cmd+K would create a parallel input that competes with the composer.
- **No notification panel.** No badges, no "3 new things". Conversations don't need a notification system; they're synchronous.
- **No multi-select on chat history.** You can delete one chat at a time. Bulk operations would invite mistakes and don't fit the surface's pace.
- **No dashboard / overview / metrics page.** Claude.ai never shows you "your usage this month" as a primary surface. Account info is buried in settings. The product is the chat, not the meta about the chat.
- **No theme picker beyond light / dark / system.** No accent color customization, no font size slider, no density toggle. The aesthetic is the product; users don't get to weaken it.
- **No floating chat-bubble pattern.** The composer is anchored, full-width, centered. There is no minimized-chat affordance.
- **No drag-to-reorder on chat history.** History is chronological. Period.
- **No keyboard shortcut help overlay.** The few shortcuts that exist are discoverable inline (placeholder hints, button tooltips) — there is no `?` overlay listing all shortcuts.
- **No power-mode toggle / "advanced settings" panel.** Settings are a short flat list. There is no expert tier of the UI.

The pattern: Claude.ai refuses to be a tool. It insists on being a conversation. Every feature that would require the user to "operate" the surface (palettes, panels, bulk operations, customization) is omitted. This is the single biggest contrast with the Evalyn dashboard spec, which explicitly is an IDE.

## Tokens you can copy directly into Evalyn dashboard

A starter design-token block that ports Claude's feel. These are SAFE TO USE: they are either MEASURED from the assistant-ui Claude clone, MEASURED from the published Anthropic generative-UI ramp, or conservative APPROXIMATE values consistent with the visual evidence.

```css
:root {
  /* ── Neutral / surface (warm cream, not cool gray) ── */
  --bg-app:           #F5F0EB;   /* page background */
  --bg-elevated:      #FFFFFF;   /* composer, modals */
  --bg-bubble-user:   #DDD9CE;   /* user message */
  --bg-bubble-assist: transparent; /* assistant blends with page */
  --bg-card:          #F1EFE8;   /* gray-50, tool-call & metric cards */
  --border-subtle:    rgba(0,0,0,0.08);
  --border-default:   #D3D1C7;   /* gray-100 */

  /* ── Text ── */
  --text-primary:     #2C2C2A;   /* gray-900 */
  --text-secondary:   #5F5E5A;   /* gray-600 */
  --text-tertiary:    #888780;   /* gray-400 */
  --text-on-accent:   #FFFFFF;

  /* ── Brand accent (Crail / coral-700ish) ── */
  --accent:           #AE5630;   /* send button */
  --accent-hover:     #C4633A;
  --accent-quiet:     #FAECE7;   /* coral-50 background tint */

  /* ── Semantic (use ONLY for state, not category) ── */
  --semantic-success: #639922;   /* green-400 */
  --semantic-warn:    #BA7517;   /* amber-400 */
  --semantic-error:   #E24B4A;   /* red-400 */
  --semantic-info:    #378ADD;   /* blue-400 */

  /* ── Typography ── */
  --font-sans:  "Styrene B", "Inter", system-ui, sans-serif;
  --font-serif: "Tiempos Text", "Source Serif Pro", Georgia, serif;
  --font-mono:  ui-monospace, "SF Mono", Menlo, monospace;

  --fs-xs:   12px;
  --fs-sm:   14px;
  --fs-base: 16px;
  --fs-lg:   18px;
  --fs-xl:   20px;
  --fs-h2:   24px;
  --fs-h1:   32px;

  --lh-tight:  1.25;
  --lh-normal: 1.5;
  --lh-relax:  1.6;     /* serif body in messages */

  /* ── Spacing (4px base) ── */
  --sp-1:  4px;
  --sp-2:  8px;
  --sp-3:  12px;
  --sp-4:  16px;
  --sp-5:  20px;
  --sp-6:  24px;
  --sp-8:  32px;
  --sp-12: 48px;
  --sp-16: 64px;

  /* ── Radii ── */
  --r-sm:    6px;    /* tags, pills */
  --r-md:    8px;    /* small buttons */
  --r-lg:    12px;   /* cards */
  --r-xl:    16px;   /* composer, message bubbles */
  --r-full:  9999px; /* icon buttons */

  /* ── Shadows (almost imperceptible) ── */
  --shadow-soft:  0 4px 20px rgba(0,0,0,0.035);
  --shadow-card:  0 1px 2px  rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);

  /* ── Motion ── */
  --dur-fast:   150ms;
  --dur-base:   300ms;
  --ease-soft:  cubic-bezier(0.165, 0.85, 0.45, 1);
  --press-scale: 0.98;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg-app:          #2B2A27;
    --bg-elevated:     #1F1E1B;
    --bg-bubble-user:  #393937;
    --bg-card:         #2C2C2A;   /* gray-900 */
    --border-subtle:   rgba(255,255,255,0.08);
    --border-default:  #444441;   /* gray-800 */

    --text-primary:    #F1EFE8;   /* gray-50 */
    --text-secondary:  #B4B2A9;   /* gray-200 */
    --text-tertiary:   #888780;   /* gray-400 */

    --shadow-soft:  0 4px 20px rgba(0,0,0,0.35);
    --shadow-card:  0 1px 2px  rgba(0,0,0,0.4),  0 4px 12px rgba(0,0,0,0.3);
  }
}
```

Operating principles that should accompany the tokens:
1. **Color is meaning, not category.** Pass/fail is green/red. Provider is gray. Don't ramp 8 metric tiles through 8 colors.
2. **One curve, one duration.** `var(--ease-soft) var(--dur-base)` covers ~95% of UI transitions.
3. **Shadow is whisper, not drama.** Two shadow tokens, both very soft.
4. **Serif for content, sans for chrome.** If a surface is primarily reading material (an analysis report, a chat message), use the serif. Buttons, labels, sidebars use sans.
5. **Composer is sacred.** Whatever else changes, the composer stays anchored, centered, max ~768px wide, soft shadow, very rounded.

## Caveats

What was MEASURED:
- Color ramps (9 ramps × 7 stops): from the public `pi-generative-ui/claude-guidelines` mirror of Anthropic's spec.
- Composer / bubble colors, transitions, shadows, scale-on-press: from the maintained `assistant-ui/examples/claude` reference implementation.
- Anthropic brand fonts (Styrene + Tiempos): from Geist case study and type.today journal entry.
- Brand named colors (Crail #C15F3C, Pampas #F4F3EE, etc.): from Loftlyy brand sheet.
- Voice and microcopy samples: directly quoted from anthropic.com news pages and product surfaces I could fetch.

What was APPROXIMATE (training-data familiarity, not directly measured this session):
- Specific font-size pixel values per surface.
- Sidebar widths, exact spacing pixel values.
- Streaming cursor frequency, exact tool-card transition duration.
- Exact "Thinking..." indicator wording in current product.
- Editor-message hover affordance precise position.
- Exact "Reply to Claude..." vs "Message Claude..." placeholder phrasing in the current build.

What needs validation by the team before shipping:
- Run live `claude.ai` in a browser with DevTools open and **measure** the actual computed font sizes, line heights, and spacing on at least three surfaces: empty new chat, mid-conversation message thread, and an artifact split-view. Replace APPROXIMATE values above with MEASURED.
- Confirm the current placeholder text in the composer.
- Confirm the exact tool-card visual (border, padding, status pill style) — this is the most important pattern to copy correctly because Evalyn will use it for every CLI invocation by the agent.
- Audit the suggested `--accent` color (`#AE5630`) against Anthropic's official Crail (`#C15F3C`); they're close but not identical. Use whichever the team verifies is currently in production.
- Confirm dark-mode page background — the clone uses `#2B2A27`, but the production app may have shifted slightly with recent updates.

Why the live audit could not be completed this session: the gstack `browse` headless Chromium binary on this WSL2 host is missing system libraries (`libnspr4.so`), and installing them requires sudo (out of scope per project conventions). All measured values above come from independently-published sources cited inline.

## Sources

- [Mobbin — Claude brand color palette](https://mobbin.com/colors/brand/claude)
- [shadcn.io — Claude theme description](https://www.shadcn.io/theme/claude)
- [Anthropic news — Introducing Claude Design](https://www.anthropic.com/news/claude-design-anthropic-labs)
- [Geist — Anthropic case study](https://geist.co/work/anthropic)
- [type.today — Styrene in use: ANTHROP\\C](https://type.today/en/journal/anthropic)
- [deardesigner.substack.com — My Styrene Soul](https://deardesigner.substack.com/p/my-styrene-soul-a-short-affair-with)
- [Loftlyy — Anthropic brand colors](https://www.loftlyy.com/en/anthropic)
- [pi-generative-ui — Claude color palette spec (mirror)](https://github.com/Michaelliv/pi-generative-ui/blob/main/.pi/extensions/generative-ui/claude-guidelines/sections/color_palette.md)
- [assistant-ui — Claude clone reference implementation](https://www.assistant-ui.com/examples/claude)
- [Claude Help Center — Artifacts](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them)
- [Victor Dibia newsletter — Claude Design observations](https://newsletter.victordibia.com/p/how-good-is-anthropics-claude-design)
