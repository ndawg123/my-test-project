"""Tests for the FastAPI REST API."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from journal_agent import api
from journal_agent.agent import JournalAgent


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def client(tmp_dir):
    db_path = tmp_dir / "test.db"
    upload_dir = tmp_dir / "uploads"

    test_agent = JournalAgent(db_path=db_path, upload_dir=upload_dir)
    test_agent.start()
    api.agent = test_agent

    with TestClient(api.app) as c:
        yield c

    # The lifespan shutdown will clean up the agent


@pytest.fixture
def sample_image_bytes():
    """Create a sample image and return its bytes."""
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Test journal content", fill="black")

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestAPI:
    @patch("journal_agent.agent.extract_text", return_value="Extracted journal text")
    def test_upload_entry(self, mock_ocr, client, sample_image_bytes):
        response = client.post(
            "/entries/upload",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
            params={"title": "My Entry", "tags": "travel,food"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["extracted_text"] == "Extracted journal text"
        assert data["title"] == "My Entry"

    @patch("journal_agent.agent.extract_text", return_value="Searchable content about dogs")
    def test_search(self, mock_ocr, client, sample_image_bytes):
        client.post(
            "/entries/upload",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        response = client.get("/entries/search", params={"q": "dogs"})
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1

    @patch("journal_agent.agent.extract_text", return_value="text")
    def test_list_entries(self, mock_ocr, client, sample_image_bytes):
        client.post(
            "/entries/upload",
            files={"image": ("a.png", sample_image_bytes, "image/png")},
        )
        client.post(
            "/entries/upload",
            files={"image": ("b.png", sample_image_bytes, "image/png")},
        )
        response = client.get("/entries")
        assert response.status_code == 200
        assert len(response.json()) == 2

    @patch("journal_agent.agent.extract_text", return_value="text")
    def test_get_entry(self, mock_ocr, client, sample_image_bytes):
        upload = client.post(
            "/entries/upload",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        entry_id = upload.json()["id"]
        response = client.get(f"/entries/{entry_id}")
        assert response.status_code == 200
        assert response.json()["id"] == entry_id

    def test_get_entry_not_found(self, client):
        response = client.get("/entries/999")
        assert response.status_code == 404

    @patch("journal_agent.agent.extract_text", return_value="text")
    def test_delete_entry(self, mock_ocr, client, sample_image_bytes):
        upload = client.post(
            "/entries/upload",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        entry_id = upload.json()["id"]
        response = client.delete(f"/entries/{entry_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        response = client.get(f"/entries/{entry_id}")
        assert response.status_code == 404

    @patch("journal_agent.agent.extract_text", return_value="text")
    def test_update_entry(self, mock_ocr, client, sample_image_bytes):
        upload = client.post(
            "/entries/upload",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
            params={"title": "Old Title"},
        )
        entry_id = upload.json()["id"]
        response = client.patch(
            f"/entries/{entry_id}",
            json={"title": "New Title", "tags": ["updated"]},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "New Title"

    @patch("journal_agent.agent.extract_text", return_value="text")
    def test_get_image(self, mock_ocr, client, sample_image_bytes):
        upload = client.post(
            "/entries/upload",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        entry_id = upload.json()["id"]
        response = client.get(f"/entries/{entry_id}/image")
        assert response.status_code == 200
        assert "image" in response.headers.get("content-type", "")

    def test_stats(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_entries"] == 0
