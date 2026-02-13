"""Tests for the Notion database provider using mocked Notion API."""

from unittest.mock import MagicMock, patch

import pytest

from journal_agent.database import NotionDatabase


@pytest.fixture
def notion_db():
    """Create a NotionDatabase with mocked client."""
    db = NotionDatabase(api_key="test-key", database_id="test-db-id")
    db._client = MagicMock()
    return db


def _make_page(page_id, title="Test", image_path="/tmp/test.png", tags=None, archived=False):
    """Helper to build a mock Notion page object."""
    return {
        "id": page_id,
        "archived": archived,
        "created_time": "2025-01-15T10:00:00.000Z",
        "last_edited_time": "2025-01-15T12:00:00.000Z",
        "properties": {
            "Title": {"title": [{"plain_text": title}]},
            "Image Path": {"rich_text": [{"plain_text": image_path}]},
            "Tags": {"multi_select": [{"name": t} for t in (tags or [])]},
            "Entry Date": {"date": {"start": "2025-01-15"} if True else None},
        },
    }


class TestNotionDatabase:
    def test_connect_validates_credentials(self):
        db = NotionDatabase(api_key="", database_id="db-id")
        with pytest.raises(ValueError, match="NOTION_API_KEY"):
            db.connect()

    def test_connect_validates_database_id(self):
        db = NotionDatabase(api_key="key", database_id="")
        with pytest.raises(ValueError, match="NOTION_DATABASE_ID"):
            db.connect()

    def test_insert_entry(self, notion_db):
        notion_db.client.pages.create.return_value = {"id": "page-123"}

        entry_id = notion_db.insert_entry(
            image_path="/tmp/test.png",
            extracted_text="Hello world from my journal.",
            title="My Entry",
            tags=["travel", "food"],
            entry_date="2025-03-01",
        )

        assert entry_id == "page-123"
        call_kwargs = notion_db.client.pages.create.call_args
        props = call_kwargs.kwargs["properties"]
        assert props["Title"]["title"][0]["text"]["content"] == "My Entry"
        assert props["Image Path"]["rich_text"][0]["text"]["content"] == "/tmp/test.png"
        assert len(props["Tags"]["multi_select"]) == 2

    def test_insert_entry_untitled(self, notion_db):
        notion_db.client.pages.create.return_value = {"id": "page-456"}
        notion_db.insert_entry(image_path="/tmp/x.png", extracted_text="text")
        call_kwargs = notion_db.client.pages.create.call_args
        props = call_kwargs.kwargs["properties"]
        assert props["Title"]["title"][0]["text"]["content"] == "Untitled"

    def test_get_entry(self, notion_db):
        page = _make_page("page-1", title="Found")
        notion_db.client.pages.retrieve.return_value = page
        notion_db.client.blocks.children.list.return_value = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "Body text"}]},
                }
            ]
        }

        entry = notion_db.get_entry("page-1")
        assert entry is not None
        assert entry["id"] == "page-1"
        assert entry["title"] == "Found"
        assert entry["extracted_text"] == "Body text"

    def test_get_entry_archived_returns_none(self, notion_db):
        page = _make_page("page-2", archived=True)
        notion_db.client.pages.retrieve.return_value = page
        assert notion_db.get_entry("page-2") is None

    def test_get_entry_not_found(self, notion_db):
        notion_db.client.pages.retrieve.side_effect = Exception("Not found")
        assert notion_db.get_entry("nonexistent") is None

    def test_delete_entry(self, notion_db):
        assert notion_db.delete_entry("page-1") is True
        notion_db.client.pages.update.assert_called_once_with(
            page_id="page-1", archived=True
        )

    def test_delete_entry_failure(self, notion_db):
        notion_db.client.pages.update.side_effect = Exception("fail")
        assert notion_db.delete_entry("page-1") is False

    def test_list_entries(self, notion_db):
        notion_db.client.databases.query.return_value = {
            "results": [
                _make_page("p1", title="First"),
                _make_page("p2", title="Second"),
            ],
        }

        entries = notion_db.list_entries(limit=10)
        assert len(entries) == 2
        assert entries[0]["title"] == "First"

    def test_list_entries_with_tag_filter(self, notion_db):
        notion_db.client.databases.query.return_value = {
            "results": [_make_page("p1", tags=["travel"])],
        }
        entries = notion_db.list_entries(tag="travel")
        call_kwargs = notion_db.client.databases.query.call_args.kwargs
        assert call_kwargs["filter"]["property"] == "Tags"

    def test_search_matches_text(self, notion_db):
        notion_db.client.databases.query.return_value = {
            "results": [_make_page("p1", title="Hiking")],
        }
        notion_db.client.blocks.children.list.return_value = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": "We went hiking in the mountains"}]
                    },
                }
            ]
        }

        results = notion_db.search("hiking")
        assert len(results) == 1
        assert results[0]["title"] == "Hiking"

    def test_search_no_match(self, notion_db):
        notion_db.client.databases.query.return_value = {
            "results": [_make_page("p1", title="Cooking")],
        }
        notion_db.client.blocks.children.list.return_value = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "Made pasta today"}]},
                }
            ]
        }

        results = notion_db.search("hiking")
        assert len(results) == 0

    def test_update_entry(self, notion_db):
        notion_db.client.pages.retrieve.return_value = _make_page("p1")
        assert notion_db.update_entry("p1", title="Updated") is True
        notion_db.client.pages.update.assert_called_once()

    def test_update_entry_not_found(self, notion_db):
        notion_db.client.pages.retrieve.side_effect = Exception("not found")
        assert notion_db.update_entry("nonexistent", title="X") is False

    def test_count_entries(self, notion_db):
        notion_db.client.databases.query.return_value = {
            "results": [
                _make_page("p1"),
                _make_page("p2"),
                _make_page("p3", archived=True),
            ],
            "has_more": False,
        }
        assert notion_db.count_entries() == 2

    def test_text_to_blocks_splits_long_text(self, notion_db):
        long_text = "x" * 5000
        blocks = notion_db._text_to_blocks(long_text)
        assert len(blocks) == 3  # 2000 + 2000 + 1000
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "x" * 2000

    def test_text_to_blocks_empty(self, notion_db):
        blocks = notion_db._text_to_blocks("")
        assert len(blocks) == 1

    def test_close_clears_client(self, notion_db):
        assert notion_db._client is not None
        notion_db.close()
        assert notion_db._client is None
