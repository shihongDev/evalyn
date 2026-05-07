/**
 * Browser notifications for backgrounded jobs.
 *
 * Customer-cared scenario: user kicks off a long eval, switches to
 * Slack / docs / another tab. Without this, they have to remember
 * to come back and check. With this, the OS notification surfaces
 * "run-eval finished (45s, exit 0)" the moment the job terminates.
 *
 * Design choices:
 *
 * - The Notification API is gated by user permission. We never
 *   request silently; the user has to click "Enable notifications"
 *   in the drawer header (which calls ``requestPermission`` here).
 *
 * - We only fire when ``document.visibilityState === 'hidden'``.
 *   Notifications when the tab is foreground are noise: the user is
 *   already looking at the result. Mirrors how chat apps (Slack,
 *   Discord) handle in-app notifications.
 *
 * - Each Notification is tagged by ``job_id`` so a status sequence
 *   (running -> failed) replaces rather than stacks. The OS dedupes
 *   by tag.
 *
 * - All API calls are wrapped in feature-detection guards so a
 *   browser without the API (older Safari, some embedded webviews)
 *   reports permission='denied' instead of throwing.
 */

export type NotificationPermissionState = 'granted' | 'denied' | 'default';

function isSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof Notification !== 'undefined' &&
    'permission' in Notification
  );
}

/** Read the current permission state without prompting. */
export function notificationPermission(): NotificationPermissionState {
  if (!isSupported()) return 'denied';
  return Notification.permission as NotificationPermissionState;
}

/** Prompt the user for permission. Returns the resulting state.
 *
 * Browsers gate this on a user gesture - it should be called from a
 * click handler, not on mount. Calling this when permission is
 * already 'granted' or 'denied' is a no-op (the browser short-circuits). */
export async function requestNotificationPermission(): Promise<NotificationPermissionState> {
  if (!isSupported()) return 'denied';
  try {
    const result = await Notification.requestPermission();
    return result as NotificationPermissionState;
  } catch {
    // Older browsers expose requestPermission as a callback-only API.
    // Returning 'denied' is safe - the user can re-trigger via the
    // drawer button.
    return 'denied';
  }
}

interface NotifyOpts {
  jobId: string;
  cliId: string;
  status: 'complete' | 'failed' | 'cancelled';
  exitCode?: number | null;
  durationS?: number | null;
}

/** Fire an OS notification for a terminal job, but only if:
 *   1. The browser supports the Notification API.
 *   2. The user has granted permission.
 *   3. The tab is currently hidden (foreground = no notification;
 *      the user is already looking at the runner).
 *
 * Tag is the job_id so a queued -> running -> failed sequence
 * replaces (rather than stacks) prior notifications for the same job.
 *
 * Best-effort: any error is logged at console.warn and swallowed.
 * A failed notification must not break job-completion handling. */
export function notifyJobTerminal(opts: NotifyOpts): void {
  if (!isSupported()) return;
  if (Notification.permission !== 'granted') return;
  if (typeof document !== 'undefined' && document.visibilityState !== 'hidden') {
    return;
  }
  try {
    const verb =
      opts.status === 'complete'
        ? 'finished'
        : opts.status === 'failed'
          ? 'failed'
          : 'was cancelled';
    const parts: string[] = [];
    if (opts.exitCode != null) parts.push(`exit ${opts.exitCode}`);
    if (opts.durationS != null) {
      parts.push(`${opts.durationS.toFixed(1)}s`);
    }
    const body = parts.length > 0 ? `${verb} (${parts.join(', ')})` : verb;
    const n = new Notification(`${opts.cliId} ${verb}`, {
      body,
      tag: `evalyn-job-${opts.jobId}`,
      // silent on success to be polite; failures should beep so the
      // user notices them.
      silent: opts.status === 'complete',
    });
    // Click on the notification focuses our tab so the user lands
    // back where they left off.
    n.onclick = () => {
      if (typeof window !== 'undefined') {
        window.focus();
        n.close();
      }
    };
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('notifyJobTerminal failed', err);
  }
}
