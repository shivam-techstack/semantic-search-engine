"""
database.py — SQLite helper for document metadata.
Vectors are NOT stored here; they live in data/vectors.npy.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "search.db")


def get_connection():
    """Return a new SQLite connection with row_factory so rows behave like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    """Create the documents table if it does not already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id        INTEGER PRIMARY KEY,
            title     TEXT NOT NULL,
            category  TEXT NOT NULL,
            content   TEXT NOT NULL,
            preview   TEXT,
            file_name TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_document(doc_id, title, category, content, file_name=""):
    """Insert one document. preview is automatically the first 220 characters."""
    preview = content[:220].rstrip() + "..." if len(content) > 220 else content
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO documents (id, title, category, content, preview, file_name)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (doc_id, title, category, content, preview, file_name),
    )
    conn.commit()
    conn.close()


def get_all_documents():
    """Return all documents as a list of dicts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_document_by_id(doc_id):
    """Return a single document dict or None."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def document_count():
    """Return the total number of documents in the database."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    return count


def clear_documents():
    """Remove all documents (used when re-loading the dataset)."""
    conn = get_connection()
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
