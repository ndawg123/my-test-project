"""Database module with pluggable backends: SQLite FTS5 and Notion."""

import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class DatabaseProvider(ABC):
    """Abstract base for journal entry storage backends."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abstractmethod
    def insert_entry(
        self,
        image_path: str,
        extracted_text: str,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        entry_date: Optional[str] = None,
    ) -> int | str: ...

    @abstractmethod
    def search(self, query: str, limit: int = 20, offset: int = 0) -> list[dict]: ...

    @abstractmethod
    def get_entry(self, entry_id: int | str) -> Optional[dict]: ...

    @abstractmethod
    def list_entries(
        self, limit: int = 50, offset: int = 0, tag: Optional[str] = None
    ) -> list[dict]: ...

    @abstractmethod
    def delete_entry(self, entry_id: int | str) -> bool: ...

    @abstractmethod
    def update_entry(
        self,
        entry_id: int | str,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        extracted_text: Optional[str] = None,
    ) -> bool: ...

    @abstractmethod
    def count_entries(self) -> int: ...


# ---------------------------------------------------------------------------
# SQLite + FTS5 implementation
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path("journal_entries.db")


class JournalDatabase(DatabaseProvider):
    """SQLite database with FTS5 full-text search for journal entries."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_tables(self) -> None:
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
        rows = self.conn.execute(
            """
            SELECT
                je.id, je.image_path, je.extracted_text, je.title,
                je.tags, je.entry_date, je.created_at, rank
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
        row = self.conn.execute(
            "SELECT * FROM journal_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_entries(
        self, limit: int = 50, offset: int = 0, tag: Optional[str] = None
    ) -> list[dict]:
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
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM journal_entries").fetchone()
        return row["cnt"]


# ---------------------------------------------------------------------------
# Notion implementation
# ---------------------------------------------------------------------------


class NotionDatabase(DatabaseProvider):
    """Store journal entries in a Notion database.

    Requires:
        - NOTION_API_KEY env var (or pass api_key)
        - NOTION_DATABASE_ID env var (or pass database_id)

    Expected Notion database properties:
        - Title        (title)           — entry title
        - Image Path   (rich_text)       — local path to stored image
        - Tags         (multi_select)    — categorization tags
        - Entry Date   (date)            — date of the journal entry
        - Created At   (created_time)    — auto-managed by Notion
        - Updated At   (last_edited_time) — auto-managed by Notion

    The extracted OCR text is stored in the page body (as paragraph blocks)
    so it isn't limited by the 2000-char rich_text property cap.
    """

    # Notion rich_text blocks are capped at 2000 chars each.
    _BLOCK_TEXT_LIMIT = 2000

    def __init__(
        self,
        api_key: Optional[str] = None,
        database_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("NOTION_API_KEY", "")
        self.database_id = database_id or os.environ.get("NOTION_DATABASE_ID", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from notion_client import Client
            self._client = Client(auth=self.api_key)
        return self._client

    def connect(self) -> None:
        if not self.api_key:
            raise ValueError("NOTION_API_KEY is required for Notion database.")
        if not self.database_id:
            raise ValueError("NOTION_DATABASE_ID is required for Notion database.")
        # Validate connection by retrieving the database
        self.client.databases.retrieve(database_id=self.database_id)
        logger.info("Connected to Notion database %s", self.database_id)

    def close(self) -> None:
        self._client = None

    # -- helpers --

    def _text_to_blocks(self, text: str) -> list[dict]:
        """Split text into Notion paragraph blocks (max 2000 chars each)."""
        blocks = []
        for i in range(0, len(text), self._BLOCK_TEXT_LIMIT):
            chunk = text[i : i + self._BLOCK_TEXT_LIMIT]
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": chunk}}]
                },
            })
        return blocks or [{
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": ""}}]},
        }]

    def _blocks_to_text(self, page_id: str) -> str:
        """Read all paragraph blocks from a page and concatenate their text."""
        blocks = self.client.blocks.children.list(block_id=page_id)
        parts = []
        for block in blocks.get("results", []):
            if block["type"] == "paragraph":
                for rt in block["paragraph"].get("rich_text", []):
                    parts.append(rt.get("plain_text", ""))
        return "".join(parts)

    def _page_to_dict(self, page: dict, include_text: bool = True) -> dict:
        """Convert a Notion page object to a flat dict matching our schema."""
        props = page["properties"]

        title = ""
        if "Title" in props and props["Title"]["title"]:
            title = props["Title"]["title"][0]["plain_text"]

        image_path = ""
        if "Image Path" in props and props["Image Path"]["rich_text"]:
            image_path = props["Image Path"]["rich_text"][0]["plain_text"]

        tags_list = []
        if "Tags" in props:
            tags_list = [opt["name"] for opt in props["Tags"].get("multi_select", [])]
        tags_str = ",".join(tags_list)

        entry_date = None
        if "Entry Date" in props and props["Entry Date"]["date"]:
            entry_date = props["Entry Date"]["date"]["start"]

        created_at = page.get("created_time", "")
        updated_at = page.get("last_edited_time", "")

        extracted_text = ""
        if include_text:
            extracted_text = self._blocks_to_text(page["id"])

        return {
            "id": page["id"],
            "image_path": image_path,
            "extracted_text": extracted_text,
            "title": title or None,
            "tags": tags_str,
            "entry_date": entry_date,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    # -- interface methods --

    def insert_entry(
        self,
        image_path: str,
        extracted_text: str,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        entry_date: Optional[str] = None,
    ) -> str:
        properties: dict = {
            "Image Path": {"rich_text": [{"text": {"content": image_path}}]},
        }

        if title:
            properties["Title"] = {"title": [{"text": {"content": title}}]}
        else:
            properties["Title"] = {"title": [{"text": {"content": "Untitled"}}]}

        if tags:
            properties["Tags"] = {
                "multi_select": [{"name": t} for t in tags]
            }

        if entry_date:
            properties["Entry Date"] = {"date": {"start": entry_date}}

        children = self._text_to_blocks(extracted_text)

        page = self.client.pages.create(
            parent={"database_id": self.database_id},
            properties=properties,
            children=children,
        )
        page_id = page["id"]
        logger.info("Inserted Notion page %s", page_id)
        return page_id

    def search(self, query: str, limit: int = 20, offset: int = 0) -> list[dict]:
        # Notion API search across the database
        response = self.client.databases.query(
            database_id=self.database_id,
            page_size=min(limit + offset, 100),
        )

        results = []
        for page in response.get("results", []):
            if page.get("archived"):
                continue
            entry = self._page_to_dict(page)
            # Client-side text matching since Notion doesn't support
            # full-text body search via the database query API
            query_lower = query.lower()
            searchable = f"{entry.get('title', '')} {entry['extracted_text']} {entry['tags']}".lower()
            if query_lower in searchable:
                entry["rank"] = 0.0  # Notion doesn't provide relevance ranking
                results.append(entry)

        # Apply offset/limit
        return results[offset : offset + limit]

    def get_entry(self, entry_id: str) -> Optional[dict]:
        try:
            page = self.client.pages.retrieve(page_id=entry_id)
            if page.get("archived"):
                return None
            return self._page_to_dict(page)
        except Exception:
            return None

    def list_entries(
        self, limit: int = 50, offset: int = 0, tag: Optional[str] = None
    ) -> list[dict]:
        filter_obj = None
        if tag:
            filter_obj = {
                "property": "Tags",
                "multi_select": {"contains": tag},
            }

        kwargs = {
            "database_id": self.database_id,
            "page_size": min(limit + offset, 100),
            "sorts": [{"timestamp": "created_time", "direction": "descending"}],
        }
        if filter_obj:
            kwargs["filter"] = filter_obj

        response = self.client.databases.query(**kwargs)
        pages = [p for p in response.get("results", []) if not p.get("archived")]

        entries = []
        for page in pages[offset : offset + limit]:
            entries.append(self._page_to_dict(page, include_text=False))

        return entries

    def delete_entry(self, entry_id: str) -> bool:
        try:
            self.client.pages.update(page_id=entry_id, archived=True)
            return True
        except Exception:
            return False

    def update_entry(
        self,
        entry_id: str,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        extracted_text: Optional[str] = None,
    ) -> bool:
        try:
            page = self.client.pages.retrieve(page_id=entry_id)
            if page.get("archived"):
                return False
        except Exception:
            return False

        properties: dict = {}
        if title is not None:
            properties["Title"] = {"title": [{"text": {"content": title}}]}
        if tags is not None:
            properties["Tags"] = {
                "multi_select": [{"name": t} for t in tags]
            }

        if properties:
            self.client.pages.update(page_id=entry_id, properties=properties)

        if extracted_text is not None:
            # Clear existing blocks and write new ones
            existing = self.client.blocks.children.list(block_id=entry_id)
            for block in existing.get("results", []):
                try:
                    self.client.blocks.delete(block_id=block["id"])
                except Exception:
                    pass
            new_blocks = self._text_to_blocks(extracted_text)
            self.client.blocks.children.append(block_id=entry_id, children=new_blocks)

        return True

    def count_entries(self) -> int:
        response = self.client.databases.query(
            database_id=self.database_id,
            page_size=100,
        )
        count = sum(1 for p in response.get("results", []) if not p.get("archived"))
        # Handle pagination for large databases
        while response.get("has_more"):
            response = self.client.databases.query(
                database_id=self.database_id,
                page_size=100,
                start_cursor=response["next_cursor"],
            )
            count += sum(1 for p in response.get("results", []) if not p.get("archived"))
        return count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_database_provider(
    provider: str | None = None, **kwargs
) -> DatabaseProvider:
    """Return a database provider instance.

    Args:
        provider: "sqlite" or "notion". Defaults to env var DATABASE_PROVIDER
                  or "sqlite".
        **kwargs: Passed through to the provider constructor.
    """
    provider = (provider or os.environ.get("DATABASE_PROVIDER", "sqlite")).lower()

    if provider == "sqlite":
        return JournalDatabase(**kwargs)
    elif provider == "notion":
        return NotionDatabase(**kwargs)
    else:
        raise ValueError(f"Unknown database provider: {provider!r}. Use 'sqlite' or 'notion'.")
