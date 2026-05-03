/**
 * AttachChip — pending click-to-attach context pill for the chat composer.
 *
 * Rendered above the textarea. Chips are wiped after the next message is
 * sent (handled in ChatComposer); here we only render and fire `onRemove`.
 */

import type { ChatAttachment } from '../store';

const KIND_GLYPH: Record<ChatAttachment['kind'], string> = {
  run: '◆',
  metric: '⚙',
  file: '◫',
};

interface AttachChipProps {
  attachment: ChatAttachment;
  onRemove: () => void;
}

function AttachChip({ attachment, onRemove }: AttachChipProps) {
  return (
  <span
    data-testid={`attach-chip-${attachment.id}`}
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: '2px 6px 2px 8px',
      background: 'var(--bg-2)',
      border: '1px solid var(--line)',
      borderRadius: 999,
      fontFamily: 'var(--mono)',
      fontSize: 11,
      color: 'var(--text-1)',
      maxWidth: 220,
    }}
  >
    <span aria-hidden="true" style={{ color: 'var(--accent)', fontSize: 10 }}>
      {KIND_GLYPH[attachment.kind]}
    </span>
    <span
      className="truncate"
      style={{
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        minWidth: 0,
      }}
    >
      {attachment.label}
    </span>
    <button
      type="button"
      aria-label={`Remove ${attachment.label}`}
      data-testid={`attach-chip-remove-${attachment.id}`}
      onClick={onRemove}
      style={{
        all: 'unset',
        cursor: 'pointer',
        color: 'var(--text-3)',
        fontSize: 11,
        padding: '0 2px',
        lineHeight: 1,
      }}
    >
      ×
    </button>
  </span>
  );
}

export default AttachChip;
