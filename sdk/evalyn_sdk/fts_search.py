"""
SQLite FTS5 full-text search index for trace content.

Provides a SearchIndex backed by SQLite's FTS5 virtual table for
fast BM25-ranked search over trace inputs, outputs, and other fields.
Pure Python using only the stdlib sqlite3 module.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


def _sanitize_fts_query(query: str) -> str:
    """Escape user input for safe use in FTS5 MATCH.

    Wraps each whitespace-delimited token in double quotes to prevent
    FTS5 operator injection (AND, OR, NOT, NEAR, column:, *, etc.).
    """
    tokens = query.strip().split()
    if not tokens:
        return ""
    safe: list[str] = []
    for token in tokens:
        escaped = token.replace('"', '""')
        safe.append(f'"{escaped}"')
    return " ".join(safe)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search result from the FTS index."""

    item_id: str
    content: str
    score: float
    snippet: str = ""
    source_field: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content": self.content,
            "score": self.score,
            "snippet": self.snippet,
            "source_field": self.source_field,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        return cls(
            item_id=data["item_id"],
            content=data["content"],
            score=data.get("score", 0.0),
            snippet=data.get("snippet", ""),
            source_field=data.get("source_field", ""),
        )


# ---------------------------------------------------------------------------
# Search Index
# ---------------------------------------------------------------------------


class SearchIndex:
    """FTS5-backed full-text search index stored in SQLite."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5("
            "item_id, content, source_field UNINDEXED, "
            "tokenize='unicode61')"
        )
        self._conn.commit()

    def add_document(
        self, item_id: str, content: str, source_field: str = "input"
    ) -> None:
        """Add a single document to the index."""
        self._conn.execute(
            "INSERT INTO documents(item_id, content, source_field) VALUES (?, ?, ?)",
            (item_id, content, source_field),
        )
        self._conn.commit()

    def add_documents(self, docs: list[tuple[str, str, str]]) -> None:
        """Batch-add documents as (item_id, content, source_field) tuples."""
        self._conn.executemany(
            "INSERT INTO documents(item_id, content, source_field) VALUES (?, ?, ?)",
            docs,
        )
        self._conn.commit()

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """FTS5 MATCH search with BM25 ranking. Returns results best-first."""
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            cursor = self._conn.execute(
                "SELECT item_id, content, source_field, bm25(documents) "
                "FROM documents WHERE documents MATCH ? "
                "ORDER BY bm25(documents) LIMIT ?",
                (safe_query, limit),
            )
        except sqlite3.OperationalError:
            return []
        results: list[SearchResult] = []
        for row in cursor.fetchall():
            results.append(
                SearchResult(
                    item_id=row[0],
                    content=row[1],
                    score=-row[3],  # bm25() returns negative; negate for positive
                    snippet="",
                    source_field=row[2],
                )
            )
        return results

    def search_with_highlight(
        self, query: str, limit: int = 20
    ) -> list[SearchResult]:
        """Search with highlighted snippets using FTS5 snippet().

        Snippet markers use [[B]]/[[/B]] internally to avoid XSS.
        Call snippet_to_html() to convert to safe HTML.
        """
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            cursor = self._conn.execute(
                "SELECT item_id, content, source_field, bm25(documents), "
                "snippet(documents, 1, '[[B]]', '[[/B]]', '...', 32) "
                "FROM documents WHERE documents MATCH ? "
                "ORDER BY bm25(documents) LIMIT ?",
                (safe_query, limit),
            )
        except sqlite3.OperationalError:
            return []
        results: list[SearchResult] = []
        for row in cursor.fetchall():
            results.append(
                SearchResult(
                    item_id=row[0],
                    content=row[1],
                    score=-row[3],
                    snippet=row[4],
                    source_field=row[2],
                )
            )
        return results

    def count(self) -> int:
        """Total number of documents in the index."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM documents"
        )
        return cursor.fetchone()[0]

    def clear(self) -> None:
        """Delete all documents from the index."""
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def snippet_to_html(snippet: str) -> str:
    """Convert snippet markers to safe HTML.

    FTS5 snippets use [[B]]/[[/B]] markers to avoid XSS.
    This function escapes all content first, then converts markers to <b> tags.
    """
    import html as _html

    escaped = _html.escape(snippet)
    return escaped.replace("[[B]]", "<b>").replace("[[/B]]", "</b>")


def build_index_from_traces(
    traces: list[dict],
    index: SearchIndex,
    input_field: str = "input",
    output_field: str = "output",
) -> int:
    """Index traces into a SearchIndex. Returns the number of documents indexed."""
    docs: list[tuple[str, str, str]] = []
    for trace in traces:
        item_id = str(trace.get("id", trace.get("item_id", "")))
        inp = trace.get(input_field, "")
        out = trace.get(output_field, "")
        if inp:
            docs.append((item_id, str(inp), input_field))
        if out:
            docs.append((item_id, str(out), output_field))
    if docs:
        index.add_documents(docs)
    return len(docs)


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results for human-readable display."""
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        header = f"{i}. [{r.item_id}] (score: {r.score:.4f}, field: {r.source_field})"
        lines.append(header)
        display = r.snippet if r.snippet else r.content
        if len(display) > 200:
            display = display[:200] + "..."
        lines.append(f"   {display}")
        lines.append("")
    return "\n".join(lines)


def search_and_filter(
    index: SearchIndex,
    query: str,
    source_field: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    """Search the index with an optional source_field filter."""
    results = index.search(query, limit=limit if source_field is None else limit * 3)
    if source_field is not None:
        results = [r for r in results if r.source_field == source_field]
    return results[:limit]
