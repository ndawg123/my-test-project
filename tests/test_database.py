"""Tests for the journal database module."""

import tempfile
from pathlib import Path

import pytest

from journal_agent.database import JournalDatabase


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    database = JournalDatabase(db_path)
    database.connect()
    yield database
    database.close()
    Path(db_path).unlink(missing_ok=True)


class TestJournalDatabase:
    def test_insert_and_get_entry(self, db):
        entry_id = db.insert_entry(
            image_path="/tmp/test.png",
            extracted_text="Today I went for a walk in the park.",
            title="Park Walk",
            tags=["outdoors", "walk"],
            entry_date="2025-01-15",
        )
        assert entry_id == 1

        entry = db.get_entry(entry_id)
        assert entry is not None
        assert entry["title"] == "Park Walk"
        assert entry["extracted_text"] == "Today I went for a walk in the park."
        assert entry["tags"] == "outdoors,walk"
        assert entry["entry_date"] == "2025-01-15"

    def test_search_full_text(self, db):
        db.insert_entry(
            image_path="/tmp/a.png",
            extracted_text="The weather was sunny and warm today.",
            title="Sunny Day",
        )
        db.insert_entry(
            image_path="/tmp/b.png",
            extracted_text="It rained all day, very gloomy.",
            title="Rainy Day",
        )
        db.insert_entry(
            image_path="/tmp/c.png",
            extracted_text="Went to the beach and enjoyed the sun.",
            title="Beach Trip",
        )

        results = db.search("sunny OR sun")
        assert len(results) == 2

        results = db.search("rained")
        assert len(results) == 1
        assert results[0]["title"] == "Rainy Day"

    def test_search_no_results(self, db):
        db.insert_entry(
            image_path="/tmp/a.png", extracted_text="Hello world."
        )
        results = db.search("nonexistent")
        assert results == []

    def test_delete_entry(self, db):
        entry_id = db.insert_entry(
            image_path="/tmp/a.png", extracted_text="To be deleted."
        )
        assert db.delete_entry(entry_id) is True
        assert db.get_entry(entry_id) is None
        assert db.delete_entry(entry_id) is False

    def test_update_entry(self, db):
        entry_id = db.insert_entry(
            image_path="/tmp/a.png",
            extracted_text="Original text.",
            title="Original",
        )
        assert db.update_entry(entry_id, title="Updated Title") is True
        entry = db.get_entry(entry_id)
        assert entry["title"] == "Updated Title"

    def test_update_nonexistent(self, db):
        assert db.update_entry(999, title="Nope") is False

    def test_list_entries(self, db):
        for i in range(5):
            db.insert_entry(
                image_path=f"/tmp/{i}.png",
                extracted_text=f"Entry number {i}",
            )
        entries = db.list_entries(limit=3)
        assert len(entries) == 3

    def test_list_entries_by_tag(self, db):
        db.insert_entry(
            image_path="/tmp/a.png",
            extracted_text="tagged entry",
            tags=["travel", "food"],
        )
        db.insert_entry(
            image_path="/tmp/b.png",
            extracted_text="untagged entry",
        )
        results = db.list_entries(tag="travel")
        assert len(results) == 1
        assert "travel" in results[0]["tags"]

    def test_count_entries(self, db):
        assert db.count_entries() == 0
        db.insert_entry(image_path="/tmp/a.png", extracted_text="one")
        db.insert_entry(image_path="/tmp/b.png", extracted_text="two")
        assert db.count_entries() == 2

    def test_search_after_update_reflects_new_text(self, db):
        entry_id = db.insert_entry(
            image_path="/tmp/a.png",
            extracted_text="Original content about cats.",
        )
        results = db.search("cats")
        assert len(results) == 1

        db.update_entry(entry_id, extracted_text="Updated content about dogs.")

        results = db.search("cats")
        assert len(results) == 0

        results = db.search("dogs")
        assert len(results) == 1

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        with JournalDatabase(db_path) as database:
            database.insert_entry(
                image_path="/tmp/a.png", extracted_text="Context manager test."
            )
            assert database.count_entries() == 1

        Path(db_path).unlink(missing_ok=True)
