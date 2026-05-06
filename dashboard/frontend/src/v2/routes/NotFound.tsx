/**
 * NotFound - rendered for any route that doesn't match a known path.
 *
 * Was previously silent: the wildcard route in V2App rendered Home
 * for unknown URLs, so a typo or stale link landed the user on the
 * dashboard root with no indication of what happened. This page
 * surfaces the failed path explicitly and offers escape hatches.
 */

import { useLocation, useNavigate } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow } from '../ui';
import { E } from '../tokens';

export default function NotFound() {
  const location = useLocation();
  const navigate = useNavigate();
  return (
    <AppShell>
      <div style={{ padding: '64px 36px', maxWidth: 720, margin: '0 auto' }}>
        <Card style={{ padding: 32, textAlign: 'center' }}>
          <span
            style={{
              display: 'block',
              fontSize: 36,
              color: E.text4,
              fontFamily: E.fSerif,
              lineHeight: 1,
              marginBottom: 14,
            }}
            aria-hidden
          >
            ◌
          </span>
          <Eyebrow>Page not found</Eyebrow>
          <div
            style={{
              marginTop: 10,
              fontSize: 14,
              color: E.text2,
              lineHeight: 1.55,
            }}
          >
            We couldn't find a page at this URL.
          </div>
          <div
            style={{
              marginTop: 8,
              fontFamily: E.fMono,
              fontSize: 12,
              color: E.text3,
              wordBreak: 'break-all',
            }}
          >
            {location.pathname}
            {location.search}
          </div>
          <div
            style={{
              marginTop: 20,
              display: 'flex',
              gap: 8,
              justifyContent: 'center',
            }}
          >
            <Btn kind="primary" size="md" onClick={() => navigate('/')}>
              Back to Home
            </Btn>
            <Btn kind="secondary" size="md" onClick={() => navigate(-1)}>
              ← Previous page
            </Btn>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
