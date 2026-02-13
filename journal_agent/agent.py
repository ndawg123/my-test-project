"""Journal Agent - Orchestrates image upload, OCR processing, and search."""

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from journal_agent.database import DatabaseProvider, JournalDatabase, get_database_provider
from journal_agent.ocr import OCRProvider, extract_text, extract_text_with_confidence, get_ocr_provider, set_default_provider

logger = logging.getLogger(__name__)


class JournalAgent:
    """Agent that manages the full lifecycle of journal entry images.

    Handles uploading images, running OCR to extract text, storing results
    in a searchable database, and querying across all entries.
    """

    def __init__(
        self,
        db_path: str | Path = "journal_entries.db",
        upload_dir: str | Path = "uploads",
        ocr_provider: str | OCRProvider | None = None,
        db_provider: str | DatabaseProvider | None = None,
        notion_api_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,
    ):
        # --- OCR provider ---
        if isinstance(ocr_provider, OCRProvider):
            self._ocr = ocr_provider
        else:
            self._ocr = get_ocr_provider(ocr_provider)
        set_default_provider(self._ocr)

        # --- Database provider ---
        if isinstance(db_provider, DatabaseProvider):
            self.db = db_provider
        elif db_provider == "notion" or (db_provider is None and notion_api_key):
            from journal_agent.database import NotionDatabase
            self.db = NotionDatabase(
                api_key=notion_api_key,
                database_id=notion_database_id,
            )
        else:
            self.db = get_database_provider(
                db_provider, db_path=db_path
            )

        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Initialize the agent (open DB connection)."""
        self.db.connect()
        logger.info("JournalAgent started (OCR: %s, DB: %s)", type(self._ocr).__name__, type(self.db).__name__)

    def stop(self) -> None:
        """Shut down the agent (close DB connection)."""
        self.db.close()
        logger.info("JournalAgent stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def ingest(
        self,
        image_path: str | Path,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        entry_date: Optional[str] = None,
        preprocess: bool = True,
    ) -> dict:
        """Ingest a journal entry image: copy to uploads, run OCR, store in DB.

        Args:
            image_path: Path to the source image file.
            title: Optional title for the entry.
            tags: Optional tags for categorization.
            entry_date: Optional date string (ISO format) for the journal entry.
            preprocess: Whether to preprocess the image before OCR.

        Returns:
            Dict with entry id, stored image path, and extracted text.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Copy image to uploads directory with a unique filename
        ext = image_path.suffix.lower()
        unique_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        stored_path = self.upload_dir / unique_name
        shutil.copy2(image_path, stored_path)

        # Run OCR
        extracted_text = extract_text(stored_path, preprocess=preprocess)

        # Store in database
        entry_id = self.db.insert_entry(
            image_path=str(stored_path),
            extracted_text=extracted_text,
            title=title,
            tags=tags,
            entry_date=entry_date,
        )

        logger.info("Ingested entry id=%s from %s (%d chars extracted)", entry_id, image_path.name, len(extracted_text))

        return {
            "id": entry_id,
            "image_path": str(stored_path),
            "extracted_text": extracted_text,
            "title": title,
            "tags": tags or [],
            "entry_date": entry_date,
        }

    def ingest_with_confidence(
        self,
        image_path: str | Path,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        entry_date: Optional[str] = None,
    ) -> dict:
        """Ingest with detailed OCR confidence data.

        Same as ingest() but also returns per-word confidence scores.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = image_path.suffix.lower()
        unique_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        stored_path = self.upload_dir / unique_name
        shutil.copy2(image_path, stored_path)

        ocr_result = extract_text_with_confidence(stored_path)

        entry_id = self.db.insert_entry(
            image_path=str(stored_path),
            extracted_text=ocr_result["text"],
            title=title,
            tags=tags,
            entry_date=entry_date,
        )

        return {
            "id": entry_id,
            "image_path": str(stored_path),
            "extracted_text": ocr_result["text"],
            "ocr_details": ocr_result["details"],
            "title": title,
            "tags": tags or [],
            "entry_date": entry_date,
        }

    def search(self, query: str, limit: int = 20, offset: int = 0) -> list[dict]:
        """Search across all journal entries by text content.

        Args:
            query: Full-text search query (supports FTS5 syntax like AND, OR, NEAR
                   when using SQLite; plain substring match with Notion).
            limit: Max results to return.
            offset: Number of results to skip.

        Returns:
            List of matching entries sorted by relevance.
        """
        return self.db.search(query, limit=limit, offset=offset)

    def get_entry(self, entry_id: int | str) -> Optional[dict]:
        """Get a single journal entry by its ID."""
        return self.db.get_entry(entry_id)

    def list_entries(
        self, limit: int = 50, offset: int = 0, tag: Optional[str] = None
    ) -> list[dict]:
        """List journal entries, optionally filtered by tag."""
        return self.db.list_entries(limit=limit, offset=offset, tag=tag)

    def delete_entry(self, entry_id: int | str) -> bool:
        """Delete a journal entry and its stored image.

        Returns:
            True if the entry was deleted.
        """
        entry = self.db.get_entry(entry_id)
        if not entry:
            return False

        # Remove stored image file
        image_path = Path(entry["image_path"])
        if image_path.exists():
            image_path.unlink()
            logger.info("Deleted image file: %s", image_path)

        return self.db.delete_entry(entry_id)

    def update_entry(
        self,
        entry_id: int | str,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Update metadata on an existing entry."""
        return self.db.update_entry(entry_id, title=title, tags=tags)

    def re_ocr_entry(self, entry_id: int | str, preprocess: bool = True) -> Optional[dict]:
        """Re-run OCR on an existing entry's image.

        Useful if OCR settings or preprocessing have improved.

        Returns:
            Updated entry dict, or None if entry not found.
        """
        entry = self.db.get_entry(entry_id)
        if not entry:
            return None

        new_text = extract_text(entry["image_path"], preprocess=preprocess)
        self.db.update_entry(entry_id, extracted_text=new_text)

        entry["extracted_text"] = new_text
        return entry

    def stats(self) -> dict:
        """Return summary statistics about the journal database."""
        return {
            "total_entries": self.db.count_entries(),
            "upload_dir": str(self.upload_dir),
            "db_backend": type(self.db).__name__,
            "ocr_backend": type(self._ocr).__name__,
        }
