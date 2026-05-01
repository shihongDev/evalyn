/**
 * FileTree - workspace file/folder tree.
 *
 * Ported from /tmp/evalyn-dashboard-mock/wb-app.jsx (FileTreeView + FileTreeRow).
 * Data lives in `store.fileTree` and is populated by `loadFileTree()` once
 * Lane B1's `/api/files/tree` endpoint ships. Rows call `openFile(name)` on
 * click; folders toggle local expand state.
 */

import { useState } from 'react';
import { useStore } from '../store';
import type { FileNode } from '../types/jobs';

const iconFor = (node: FileNode): string => {
  if (node.kind === 'folder') return '▤';
  if (node.name.endsWith('.run')) return '▶';
  if (node.name.endsWith('.yaml')) return '⌬';
  if (node.name.endsWith('.jsonl')) return '≡';
  return '·';
};

const toneFor = (node: FileNode): string => {
  if (node.fail) return 'var(--fail)';
  if (node.warn) return 'var(--warn)';
  if (node.live) return 'var(--accent)';
  return 'var(--text-1)';
};

interface RowProps {
  node: FileNode;
  depth: number;
  activeFile: string | null;
}

const FileTreeRow = ({ node, depth, activeFile }: RowProps) => {
  const [expanded, setExpanded] = useState<boolean>(!!node.expand);
  const openFile = useStore((s) => s.openFile);
  const isFolder = node.kind === 'folder';
  const isActive = !isFolder && activeFile === node.name;
  const tone = toneFor(node);

  return (
    <>
      <div
        role="treeitem"
        aria-expanded={isFolder ? expanded : undefined}
        onClick={() => (isFolder ? setExpanded(!expanded) : openFile(node.name))}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          padding: `2px 12px 2px ${depth * 12 + 12}px`,
          fontSize: 12,
          fontFamily: 'var(--mono)',
          color: tone,
          cursor: 'pointer',
          background: isActive
            ? 'var(--accent-soft)'
            : node.active
              ? 'var(--bg-3)'
              : 'transparent',
          borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
        }}
      >
        <span style={{ width: 10, color: 'var(--text-3)', fontSize: 9 }}>
          {isFolder ? (expanded ? '▾' : '▸') : ''}
        </span>
        <span style={{ width: 14, color: isFolder ? 'var(--text-3)' : tone }}>
          {iconFor(node)}
        </span>
        <span className="grow truncate">{node.name}</span>
        {node.meta && (
          <span className="text-3" style={{ fontSize: 10 }}>
            {node.meta}
          </span>
        )}
        {node.live && (
          <span
            className="chip signal dot"
            style={{ height: 14, fontSize: 9, padding: '0 5px' }}
          >
            live
          </span>
        )}
      </div>
      {isFolder &&
        expanded &&
        (node.children ?? []).map((c, i) => (
          <FileTreeRow key={c.name + i} node={c} depth={depth + 1} activeFile={activeFile} />
        ))}
    </>
  );
};

const FileTree = () => {
  const tree = useStore((s) => s.fileTree);
  const activeTabId = useStore((s) => s.activeTabId);

  if (tree.length === 0) {
    return (
      <div
        className="mono text-3"
        style={{ padding: '24px 14px', fontSize: 11, lineHeight: 1.6 }}
      >
        // no files
      </div>
    );
  }

  return (
    <div role="tree" aria-label="workspace files">
      <div
        className="mono"
        style={{
          padding: '4px 14px',
          fontSize: 10,
          color: 'var(--text-3)',
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
        }}
      >
        workspace
      </div>
      {tree.map((n, i) => (
        <FileTreeRow key={n.name + i} node={n} depth={0} activeFile={activeTabId} />
      ))}
    </div>
  );
};

export default FileTree;
