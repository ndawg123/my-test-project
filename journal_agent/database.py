"""Database module for storing and searching journal entries using SQLite FTS5."""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("journal_entries.db")


class JournalDatabase:
    """SQLite database with FTS5 full-text search for journal entries."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open database connection and initialize schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_tables(self) -> None:
        """Create the journal entries table and FTS5 virtual table."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                extracted_text TEXT NOT NULL DEFAULT '',
                title TEXT,
                tags TEXT DEFAULT '',
                entry_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS journal_entries_fts USING fts5(
                title,
                extracted_text,
                tags,
                content='journal_entries',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS journal_entries_ai AFTER INSERT ON journal_entries BEGIN
                INSERT INTO journal_entries_fts(rowid, title, extracted_text, tags)
                VALUES (new.id, new.title, new.extracted_text, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS journal_entries_ad AFTER DELETE ON journal_entries BEGIN
                INSERT INTO journal_entries_fts(journal_entries_fts, rowid, title, extracted_text, tags)
                VALUES ('delete', old.id, old.title, old.extracted_text, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS journal_entries_au AFTER UPDATE ON journal_entries BEGIN
                INSERT INTO journal_entries_fts(journal_entries_fts, rowid, title, extracted_text, tags)
                VALUES ('delete', old.id, old.title, old.extracted_text, old.tags);
                INSERT INTO journal_entries_fts(rowid, title, extracted_text, tags)
                VALUES (new.id, new.title, new.extracted_text, new.tags);
            END;
        """)

    def insert_entry(
        self,
        image_path: str,
        extracted_text: str,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        entry_date: Optional[str] = None,
    ) -> int:
        """Insert a new journal entry.

        Args:
            image_path: Path to the stored image file.
            extracted_text: OCR-extracted text from the image.
            title: Optional title for the entry.
            tags: Optional list of tags for categorization.
            entry_date: Optional date of the journal entry (ISO format).

        Returns:
            The ID of the newly inserted entry.
        """
        now = datetime.now(timezone.utc).isoformat()
        tags_str = ",".join(tags) if tags else ""

        cursor = self.conn.execute(
            """
            INSERT INTO journal_entries
                (image_path, extracted_text, title, tags, entry_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (image_path, extracted_text, title, tags_str, entry_date, now, now),
        )
        self.conn.commit()
        entry_id = cursor.lastrowid
        logger.info("Inserted journal entry id=%d", entry_id)
        return entry_id

    def search(self, query: str, limit: int = 20, offset: int = 0) -> list[dict]:
        """Full-text search across all journal entries.

        Args:
            query: Search query string (supports FTS5 query syntax).
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching journal entries with relevance scores.
        """
        rows = self.conn.execute(
            """
            SELECT
                je.id,
                je.image_path,
                je.extracted_text,
                je.title,
                je.tags,
                je.entry_date,
                je.created_at,
                rank
            FROM journal_entries_fts
            JOIN journal_entries je ON je.id = journal_entries_fts.rowid
            WHERE journal_entries_fts MATCH ?
            ORDER BY rank
            LIMIT ? OFFSET ?
            """,
            (query, limit, offset),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_entry(self, entry_id: int) -> Optional[dict]:
        """Retrieve a single journal entry by ID."""
        row = self.conn.execute(
            "SELECT * FROM journal_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_entries(
        self, limit: int = 50, offset: int = 0, tag: Optional[str] = None
    ) -> list[dict]:
        """List journal entries, optionally filtered by tag.

        Args:
            limit: Maximum number of results.
            offset: Number of results to skip.
            tag: Optional tag to filter by.

        Returns:
            List of journal entries.
        """
        if tag:
            rows = self.conn.execute(
                """
                SELECT * FROM journal_entries
                WHERE ',' || tags || ',' LIKE ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (f"%,{tag},%", limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM journal_entries
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a journal entry by ID.

        Returns:
            True if the entry was deleted, False if not found.
        """
        cursor = self.conn.execute(
            "DELETE FROM journal_entries WHERE id = ?", (entry_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_entry(
        self,
        entry_id: int,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        extracted_text: Optional[str] = None,
    ) -> bool:
        """Update fields on an existing journal entry.

        Returns:
            True if the entry was updated, False if not found.
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        now = datetime.now(timezone.utc).isoformat()
        new_title = title if title is not None else entry["title"]
        new_tags = ",".join(tags) if tags is not None else entry["tags"]
        new_text = extracted_text if extracted_text is not None else entry["extracted_text"]

        self.conn.execute(
            """
            UPDATE journal_entries
            SET title = ?, tags = ?, extracted_text = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_title, new_tags, new_text, now, entry_id),
        )
        self.conn.commit()
        return True

    def count_entries(self) -> int:
        """Return the total number of journal entries."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM journal_entries").fetchone()
        return row["cnt"]
