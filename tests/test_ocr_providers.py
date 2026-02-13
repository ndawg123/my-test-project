"""Tests for OCR provider factory and backward-compatible functions."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image, ImageDraw

from journal_agent.ocr import (
    TesseractOCR,
    OCRProvider,
    get_ocr_provider,
    set_default_provider,
    extract_text,
    _validate_image_path,
)


@pytest.fixture
def sample_image():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (200, 50), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 15), "Test text", fill="black")
        img.save(f.name)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


class TestValidation:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            _validate_image_path(Path("/nonexistent/file.png"))

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "file.txt"
        bad_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            _validate_image_path(bad_file)


class TestFactory:
    def test_default_is_tesseract(self):
        provider = get_ocr_provider()
        assert isinstance(provider, TesseractOCR)

    def test_tesseract_explicit(self):
        assert isinstance(get_ocr_provider("tesseract"), TesseractOCR)

    @patch.dict("os.environ", {"OCR_PROVIDER": "tesseract"})
    def test_env_var(self):
        assert isinstance(get_ocr_provider(), TesseractOCR)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown OCR provider"):
            get_ocr_provider("invalid")

    @patch("journal_agent.ocr.GoogleVisionOCR.__init__", return_value=None)
    def test_google_vision_factory(self, mock_init):
        provider = get_ocr_provider("google_vision")
        assert type(provider).__name__ == "GoogleVisionOCR"


class TestSetDefaultProvider:
    def test_set_and_use(self, sample_image):
        mock_provider = MagicMock(spec=OCRProvider)
        mock_provider.extract_text.return_value = "Custom provider result"

        set_default_provider(mock_provider)
        result = extract_text(sample_image)
        assert result == "Custom provider result"
        mock_provider.extract_text.assert_called_once()

        # Reset to default
        set_default_provider(TesseractOCR())
