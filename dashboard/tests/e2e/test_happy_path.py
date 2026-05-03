"""One happy-path E2E test for the dashboard (Lane D1.2).

Flow:
1. Server is already up (``dashboard_server`` fixture in ``conftest.py``).
2. Navigate to ``/``. The bundled SPA loads with the CSRF token in
   ``<meta name="workbench-token">``; the browser's ``fetch`` automatically
   carries the meta-derived header (the frontend injects it in ``api.ts``).
3. Click the ``list-runs`` row in the CLI catalog sidebar. A ``cli:list-runs``
   tab opens with the form.
4. Fill ``--limit`` with ``5`` and click Run.
5. The form tab closes and a ``job:<id>`` tab opens. The Terminal panel
   begins streaming the subprocess stdout. We wait for an ``exit 0``
   line - that's our pass signal both for the run AND for the tab status
   indicator (the store flips ``Job.status`` to ``complete`` on
   ``exit_code === 0``).

Budget: ONE test. Adding more belongs in v1.1+.

Why no agent flow? The agent runtime needs an LLM provider key (OpenAI,
Anthropic, or local Ollama). CI can't reliably provision that, so the
agent surface is unit-tested in ``tests/test_agent.py`` and exercised
manually before release.
"""

from __future__ import annotations

import pytest

# pytest-playwright provides ``page``; if it's missing the suite shouldn't
# even be collected. importorskip both keeps the import error friendly and
# lets non-E2E developers skip the whole directory cleanly.
pytest.importorskip("playwright.sync_api")
pytest.importorskip("pytest_playwright")

from playwright.sync_api import Page, expect


# Reasonable per-step timeouts. ``list-runs`` against an empty .evalyn/
# returns within ~1s; we give the network/render path more headroom.
ACTION_TIMEOUT_MS = 5_000
EXIT_LINE_TIMEOUT_MS = 15_000


def test_list_runs_happy_path(page: Page, dashboard_server: str) -> None:
    """Click list-runs, run with limit=5, assert exit 0 streams in."""
    page.set_default_timeout(ACTION_TIMEOUT_MS)

    # 1. Load the SPA. ``base_url`` is wired by the conftest fixture.
    page.goto(f"{dashboard_server}/")

    # Sanity: the CSRF meta tag must be present - the frontend reads it on
    # every mutating request. If it's missing nothing else can succeed.
    meta = page.locator('meta[name="workbench-token"]')
    expect(meta).to_have_count(1)

    # 2. The sidebar defaults to the Commands tab now (P0 fix; the Files
    #    tab was hidden because clicking files opened "coming soon"
    #    placeholders). The CliCatalog renders by default and ``GET /api/cli``
    #    resolves on mount. Click is a no-op-but-harmless guard — if a
    #    future change ever flips the default again, this still recovers.
    page.get_by_role("button", name="Commands", exact=True).click()

    # Wait for the catalog to populate. ``CliCatalog.tsx`` shows
    # ``Loading commands...`` until the fetch flips the store; the
    # default view is the 5-CLI STARTER set (quickstart, one-click,
    # list-calls, status, workflow), so ``list-runs`` is hidden until
    # the user either expands "Show all" or types into the filter.
    # Typing in the filter is more robust because the filter view
    # always renders the matched CLI as a row regardless of starter
    # collapse state.
    search = page.get_by_label("Search commands")
    expect(search).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    search.fill("list-runs")
    list_runs_row = page.get_by_text("list-runs", exact=True).first
    expect(list_runs_row).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    list_runs_row.click()

    # 3. The form tab opens. ``ParamField`` for ``--limit`` (kind=number)
    #    renders an ``<input type=number>`` with id ``p-limit``. Fill 5.
    limit_input = page.locator("#p-limit")
    expect(limit_input).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    limit_input.fill("5")

    # 4. Click Run. The Workspace button label is `Run` while idle and
    #    `Running...` while the request is in flight (microcopy pass
    #    dropped the legacy `▶ Run` arrow glyph from the inline form).
    #    Exact match disambiguates from the sidebar's `Eval runs` tab.
    run_button = page.get_by_role("button", name="Run", exact=True)
    run_button.click()

    # 5. Streaming output assertion. The store appends an ``exit N`` line
    #    when the subprocess terminates (see ``store.ts`` exit handler).
    #    Asserting on the exit line covers both:
    #      a) WebSocket streaming actually reached the browser, and
    #      b) the process succeeded (exit 0 -> Job.status = 'complete',
    #         which the prompt calls "tab title indicates pass").
    terminal = page.locator('[data-testid="terminal-scroll"]')
    expect(terminal).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(terminal).to_contain_text("exit 0", timeout=EXIT_LINE_TIMEOUT_MS)
