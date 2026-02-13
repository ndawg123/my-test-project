"""Tests for the Journal Agent orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from journal_agent.agent import JournalAgent


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_image(tmp_dir):
    """Create a simple test image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Hello journal entry test", fill="black")
    path = tmp_dir / "sample.png"
    img.save(path)
    return path


@pytest.fixture
def agent(tmp_dir):
    db_path = tmp_dir / "test.db"
    upload_dir = tmp_dir / "uploads"
    a = JournalAgent(db_path=db_path, upload_dir=upload_dir)
    a.start()
    yield a
    a.stop()


class TestJournalAgent:
    @patch("journal_agent.agent.extract_text", return_value="Mocked OCR text from journal")
    def test_ingest(self, mock_ocr, agent, sample_image):
        result = agent.ingest(
            image_path=sample_image,
            title="Test Entry",
            tags=["test"],
            entry_date="2025-03-01",
        )
        assert result["id"] == 1
        assert result["extracted_text"] == "Mocked OCR text from journal"
        assert result["title"] == "Test Entry"
        assert Path(result["image_path"]).exists()

    @patch("journal_agent.agent.extract_text", return_value="Searchable text about hiking mountains")
    def test_ingest_and_search(self, mock_ocr, agent, sample_image):
        agent.ingest(image_path=sample_image, title="Hiking Trip")
        results = agent.search("hiking")
        assert len(results) == 1
        assert results[0]["title"] == "Hiking Trip"

    @patch("journal_agent.agent.extract_text", return_value="Some text")
    def test_delete_removes_image(self, mock_ocr, agent, sample_image):
        result = agent.ingest(image_path=sample_image)
        stored_path = Path(result["image_path"])
        assert stored_path.exists()

        agent.delete_entry(result["id"])
        assert not stored_path.exists()
        assert agent.get_entry(result["id"]) is None

    @patch("journal_agent.agent.extract_text", return_value="Some text")
    def test_stats(self, mock_ocr, agent, sample_image):
        assert agent.stats()["total_entries"] == 0
        agent.ingest(image_path=sample_image)
        assert agent.stats()["total_entries"] == 1

    def test_ingest_file_not_found(self, agent):
        with pytest.raises(FileNotFoundError):
            agent.ingest(image_path="/nonexistent/file.png")

    @patch("journal_agent.agent.extract_text", side_effect=["Original", "Re-OCR result"])
    def test_re_ocr(self, mock_ocr, agent, sample_image):
        result = agent.ingest(image_path=sample_image)
        updated = agent.re_ocr_entry(result["id"])
        assert updated is not None
        assert updated["extracted_text"] == "Re-OCR result"

    def test_re_ocr_nonexistent(self, agent):
        assert agent.re_ocr_entry(999) is None

    @patch("journal_agent.agent.extract_text", return_value="text")
    def test_update_entry(self, mock_ocr, agent, sample_image):
        result = agent.ingest(image_path=sample_image, title="Old")
        assert agent.update_entry(result["id"], title="New")
        entry = agent.get_entry(result["id"])
        assert entry["title"] == "New"

    @patch("journal_agent.agent.extract_text", return_value="text")
    def test_list_entries(self, mock_ocr, agent, sample_image):
        agent.ingest(image_path=sample_image, title="A")
        agent.ingest(image_path=sample_image, title="B")
        entries = agent.list_entries()
        assert len(entries) == 2

    def test_context_manager(self, tmp_dir, sample_image):
        db_path = tmp_dir / "ctx.db"
        upload_dir = tmp_dir / "ctx_uploads"
        with patch("journal_agent.agent.extract_text", return_value="ctx text"):
            with JournalAgent(db_path=db_path, upload_dir=upload_dir) as a:
                result = a.ingest(image_path=sample_image)
                assert result["id"] == 1
