/**
 * Commands - browseable index of every evalyn CLI subcommand.
 *
 * Reads `GET /api/cli` (the introspected catalog) and groups the result by
 * `group`. Two actions per row:
 *   - "Ask co-pilot ->" navigates to /copilot?prefill=<question> so the
 *     thread composer opens pre-seeded.
 *   - "Run" opens the in-page CLI runner sheet.
 *
 * Deep-link form: /commands?prefill=<cmd-id>&<param>=<value>... auto-opens
 * the runner with the supplied initial values, then strips the params so a
 * refresh doesn't re-fire.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Card, Eyebrow, Btn, Skeleton, UpdatingChip } from '../ui';
import { listCli, commandGroup, commandSummary } from '../api/cli';
import type { CliParam, CliSchema } from '../api/cli';
import { useV2Resource } from '../hooks/useV2Resource';
import { openCliRunner } from '../cliRunnerBridge';
import { E } from '../tokens';

/**
 * Coerce a string query-param value into the shape expected by a given
 * `CliParam`. Best-effort: bool from "true|1|yes|on", number via Number,
 * multiselect from comma-split. Strings/paths/long-text/select pass through
 * unchanged - the existing CliRunner kind-dispatch handles those.
 */
function coerceParam(p: CliParam, raw: string): unknown {
  if (p.kind === 'bool') {
    const v = raw.trim().toLowerCase();
    return v === 'true' || v === '1' || v === 'yes' || v === 'on';
  }
  if (p.kind === 'number') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : raw;
  }
  if (p.kind === 'multiselect') {
    return raw
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return raw;
}

function groupCommands(cmds: CliSchema[]): { group: string; items: CliSchema[] }[] {
  const buckets = new Map<string, CliSchema[]>();
  for (const cmd of cmds) {
    const g = commandGroup(cmd);
    const arr = buckets.get(g) ?? [];
    arr.push(cmd);
    buckets.set(g, arr);
  }
  // Sort each bucket by id for stable output.
  for (const arr of buckets.values()) {
    arr.sort((a, b) => a.id.localeCompare(b.id));
  }
  // Preserve a sensible group ordering: most-used first, "Other" last.
  const ORDER = [
    'Tracing',
    'Eval',
    'Dataset',
    'Analysis',
    'Annotation',
    'Insights',
    'Infrastructure',
    'Simulation',
    'Export',
    'Quickstart',
    'Other',
    'Misc',
  ];
  return Array.from(buckets.entries())
    .sort((a, b) => {
      const ai = ORDER.indexOf(a[0]);
      const bi = ORDER.indexOf(b[0]);
      const aRank = ai === -1 ? ORDER.length : ai;
      const bRank = bi === -1 ? ORDER.length : bi;
      if (aRank !== bRank) return aRank - bRank;
      return a[0].localeCompare(b[0]);
    })
    .map(([group, items]) => ({ group, items }));
}

function matchesQuery(cmd: CliSchema, q: string): boolean {
  if (!q) return true;
  const needle = q.toLowerCase();
  if (cmd.id.toLowerCase().includes(needle)) return true;
  if (commandGroup(cmd).toLowerCase().includes(needle)) return true;
  if (commandSummary(cmd).toLowerCase().includes(needle)) return true;
  return false;
}

export default function Commands() {
  const { data: cmds, err, reloading, isInitialLoad } = useV2Resource<CliSchema[]>(
    'commands',
    listCli,
  );
  const [query, setQuery] = useState('');
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  // Fire the deep-link auto-open at most once per page load. We can't rely
  // on the URL being clean after we strip the params (replaceState races
  // with the catalog fetch), so a ref is the cleanest one-shot guard.
  const deepLinkFiredRef = useRef(false);

  useEffect(() => {
    if (deepLinkFiredRef.current) return;
    if (!cmds) return;
    const prefill = searchParams.get('prefill');
    if (!prefill) return;
    const cmd = cmds.find((c) => c.id === prefill);
    if (!cmd) {
      // Bad/unknown command id: drop the param so a reload doesn't loop.
      deepLinkFiredRef.current = true;
      const next = new URLSearchParams(searchParams);
      next.delete('prefill');
      setSearchParams(next, { replace: true });
      return;
    }
    const initialValues: Record<string, unknown> = {};
    for (const p of cmd.params) {
      const v = searchParams.get(p.name);
      if (v != null) initialValues[p.name] = coerceParam(p, v);
    }
    deepLinkFiredRef.current = true;
    openCliRunner(cmd, { initialValues });
    // Strip ?prefill= and any consumed param names so a reload (or a back
    // nav into this page) doesn't re-fire the runner.
    const next = new URLSearchParams(searchParams);
    next.delete('prefill');
    for (const p of cmd.params) next.delete(p.name);
    setSearchParams(next, { replace: true });
  }, [cmds, searchParams, setSearchParams]);

  const grouped = useMemo(() => {
    if (!cmds) return null;
    const filtered = cmds.filter((c) => matchesQuery(c, query));
    return groupCommands(filtered);
  }, [cmds, query]);

  function askCoPilot(id: string) {
    const prefill = `Run the \`${id}\` command and explain the output.`;
    navigate(`/copilot?prefill=${encodeURIComponent(prefill)}`);
  }

  const totalCount = cmds?.length ?? 0;
  const filteredCount = grouped ? grouped.reduce((s, g) => s + g.items.length, 0) : 0;

  return (
    <AppShell contextChip={{ name: 'Commands', version: '' }}>
      <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <Eyebrow>All evalyn commands</Eyebrow>
          <UpdatingChip visible={reloading && !isInitialLoad} />
        </div>
        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 32,
            fontWeight: 400,
            margin: 0,
            color: E.text0,
            letterSpacing: '-0.015em',
            lineHeight: 1.1,
          }}
        >
          Commands
        </h1>
        <p style={{ fontSize: 14, color: E.text2, marginTop: 8, lineHeight: 1.55, maxWidth: 680 }}>
          Every CLI evalyn ships. The co-pilot can run the read-only ones automatically; writes
          always ask first.
        </p>

        {err && (
          <Card style={{ marginTop: 16, padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading commands</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
          </Card>
        )}

        {!cmds && !err && (
          <div
            style={{
              marginTop: 18,
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 14,
            }}
          >
            {[0, 1, 2, 3].map((i) => (
              <Card key={i} style={{ padding: 0, overflow: 'hidden' }}>
                <div
                  style={{
                    padding: '14px 18px',
                    borderBottom: `1px solid ${E.hair}`,
                  }}
                >
                  <Skeleton w={120} h={11} />
                </div>
                {[0, 1, 2].map((j) => (
                  <div
                    key={j}
                    style={{
                      padding: '12px 18px',
                      borderTop: j ? `1px solid ${E.hair}` : 'none',
                    }}
                  >
                    <Skeleton w="40%" h={13} />
                    <div style={{ marginTop: 6 }}>
                      <Skeleton w="80%" h={11} />
                    </div>
                  </div>
                ))}
              </Card>
            ))}
          </div>
        )}

        {cmds && (
          <>
            {/* SEARCH BAR - matches ExperimentsList filter bar style */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginTop: 18,
                padding: 8,
                background: E.panel,
                border: `1px solid ${E.hair}`,
                borderRadius: 10,
              }}
            >
              <input
                placeholder="Filter by id, group, or description..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{
                  flex: 1,
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: E.text1,
                  fontSize: 13,
                  padding: '4px 10px',
                }}
              />
              <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3 }}>
                {query ? `${filteredCount} / ${totalCount}` : `${totalCount} commands`}
              </span>
            </div>

            {grouped && grouped.length === 0 && (
              <Card
                style={{
                  marginTop: 18,
                  padding: 18,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                }}
              >
                <div style={{ flex: 1, fontSize: 13, color: E.text2 }}>
                  No commands match{' '}
                  <b style={{ color: E.ember }}>"{query}"</b>.
                  <div style={{ fontSize: 11, color: E.text3, marginTop: 2, fontFamily: E.fMono }}>
                    {totalCount} command{totalCount === 1 ? '' : 's'} available · search by name, group, or blurb
                  </div>
                </div>
                <Btn kind="secondary" size="sm" onClick={() => setQuery('')}>
                  Clear search
                </Btn>
              </Card>
            )}

            {grouped && grouped.length > 0 && (
              <div
                style={{
                  marginTop: 18,
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: 14,
                }}
              >
                {grouped.map(({ group, items }) => (
                  <Card key={group} style={{ padding: 0, overflow: 'hidden' }}>
                    <div
                      style={{
                        padding: '14px 18px',
                        borderBottom: `1px solid ${E.hair}`,
                        display: 'flex',
                        alignItems: 'center',
                      }}
                    >
                      <Eyebrow>{group}</Eyebrow>
                      <span style={{ flex: 1 }} />
                      <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}>
                        {items.length}
                      </span>
                    </div>
                    {items.map((cmd, i) => {
                      const summary = commandSummary(cmd);
                      return (
                        <div
                          key={cmd.id}
                          style={{
                            padding: '12px 18px',
                            borderTop: i ? `1px solid ${E.hair}` : 'none',
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: 10,
                          }}
                        >
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div
                              style={{
                                fontFamily: E.fMono,
                                fontSize: 12.5,
                                color: E.text0,
                                fontWeight: 500,
                              }}
                            >
                              {cmd.id}
                            </div>
                            {summary && (
                              <div
                                style={{
                                  fontSize: 11.5,
                                  color: E.text2,
                                  marginTop: 3,
                                  lineHeight: 1.45,
                                }}
                              >
                                {summary}
                              </div>
                            )}
                          </div>
                          <div
                            style={{
                              display: 'flex',
                              gap: 6,
                              flexShrink: 0,
                              alignItems: 'center',
                            }}
                          >
                            <Btn
                              kind="ghost"
                              size="sm"
                              onClick={() => askCoPilot(cmd.id)}
                              style={{ fontSize: 10.5 }}
                            >
                              Ask co-pilot →
                            </Btn>
                            <Btn
                              kind="primary"
                              size="sm"
                              onClick={() => openCliRunner(cmd)}
                              style={{ fontSize: 10.5 }}
                            >
                              Run
                            </Btn>
                          </div>
                        </div>
                      );
                    })}
                  </Card>
                ))}
              </div>
            )}
          </>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
