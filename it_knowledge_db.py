"""
SQLite store for curated IT knowledge chunks (seeded on first run).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from course_meta_chunks import COURSE_META_DOCUMENTS
from it_knowledge_seed import IT_DOCUMENTS

# Course FAQ + IT reference chunks (single table)
ALL_KB_DOCUMENTS: list[tuple[str, str, str, str]] = COURSE_META_DOCUMENTS + IT_DOCUMENTS


def knowledge_db_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "it_knowledge.sqlite"


def ensure_database() -> Path:
    """Create DB file, schema, and seed rows if empty."""
    path = knowledge_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS it_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                body TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'curated'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_it_docs_category ON it_docs (category)"
        )
        n = conn.execute("SELECT COUNT(*) FROM it_docs").fetchone()[0]
        expected = len(ALL_KB_DOCUMENTS)
        if n != expected:
            conn.execute("DELETE FROM it_docs")
            conn.executemany(
                "INSERT INTO it_docs (title, category, body, source) VALUES (?,?,?,?)",
                ALL_KB_DOCUMENTS,
            )
            conn.commit()
    finally:
        conn.close()
    return path
