"""OCR module with pluggable providers for extracting text from journal images."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def _validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )


def preprocess_image(image: Image.Image) -> Image.Image:
    """Apply preprocessing to improve OCR accuracy on handwritten/printed journal entries."""
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class OCRProvider(ABC):
    """Abstract base for OCR providers."""

    @abstractmethod
    def extract_text(self, image_path: str | Path, preprocess: bool = True) -> str:
        ...

    @abstractmethod
    def extract_text_with_confidence(self, image_path: str | Path) -> dict:
        ...


# ---------------------------------------------------------------------------
# Tesseract (local) provider
# ---------------------------------------------------------------------------


class TesseractOCR(OCRProvider):
    """OCR via local Tesseract installation."""

    def extract_text(self, image_path: str | Path, preprocess: bool = True) -> str:
        import pytesseract

        image_path = Path(image_path)
        _validate_image_path(image_path)
        image = Image.open(image_path)
        if preprocess:
            image = preprocess_image(image)
        text = pytesseract.image_to_string(image).strip()
        logger.info("Tesseract extracted %d chars from %s", len(text), image_path.name)
        return text

    def extract_text_with_confidence(self, image_path: str | Path) -> dict:
        import pytesseract

        image_path = Path(image_path)
        _validate_image_path(image_path)
        image = Image.open(image_path)
        image = preprocess_image(image)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        for i, word in enumerate(data["text"]):
            if word.strip():
                words.append({
                    "word": word,
                    "confidence": data["conf"][i],
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                })
        full_text = " ".join(w["word"] for w in words)
        return {"text": full_text, "details": words}


# ---------------------------------------------------------------------------
# Google Cloud Vision provider
# ---------------------------------------------------------------------------


class GoogleVisionOCR(OCRProvider):
    """OCR via Google Cloud Vision API — optimized for handwriting."""

    def __init__(self):
        from google.cloud import vision  # noqa: F401 — validate import at init
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google.cloud import vision
            self._client = vision.ImageAnnotatorClient()
        return self._client

    def extract_text(self, image_path: str | Path, preprocess: bool = True) -> str:
        from google.cloud import vision

        image_path = Path(image_path)
        _validate_image_path(image_path)

        content = image_path.read_bytes()
        image = vision.Image(content=content)

        # Use DOCUMENT_TEXT_DETECTION for best handwriting results
        response = self.client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        text = response.full_text_annotation.text.strip() if response.full_text_annotation else ""
        logger.info("Google Vision extracted %d chars from %s", len(text), image_path.name)
        return text

    def extract_text_with_confidence(self, image_path: str | Path) -> dict:
        from google.cloud import vision

        image_path = Path(image_path)
        _validate_image_path(image_path)

        content = image_path.read_bytes()
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        words = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join(s.text for s in word.symbols)
                            confidence = word.confidence
                            # Bounding box
                            vertices = word.bounding_box.vertices
                            left = vertices[0].x if vertices else 0
                            top = vertices[0].y if vertices else 0
                            right = vertices[2].x if len(vertices) > 2 else left
                            bottom = vertices[2].y if len(vertices) > 2 else top
                            words.append({
                                "word": word_text,
                                "confidence": round(confidence * 100, 1),
                                "left": left,
                                "top": top,
                                "width": right - left,
                                "height": bottom - top,
                            })

        full_text = " ".join(w["word"] for w in words)
        return {"text": full_text, "details": words}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_ocr_provider(provider: str | None = None) -> OCRProvider:
    """Return an OCR provider instance.

    Args:
        provider: "tesseract" or "google_vision". Defaults to env var OCR_PROVIDER
                  or "tesseract".
    """
    provider = (provider or os.environ.get("OCR_PROVIDER", "tesseract")).lower()

    if provider == "tesseract":
        return TesseractOCR()
    elif provider in ("google_vision", "google_cloud_vision", "google"):
        return GoogleVisionOCR()
    else:
        raise ValueError(f"Unknown OCR provider: {provider!r}. Use 'tesseract' or 'google_vision'.")


# ---------------------------------------------------------------------------
# Convenience functions (backward-compatible)
# ---------------------------------------------------------------------------

_default_provider: OCRProvider | None = None


def _get_default() -> OCRProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = get_ocr_provider()
    return _default_provider


def set_default_provider(provider: OCRProvider) -> None:
    """Override the default OCR provider."""
    global _default_provider
    _default_provider = provider


def extract_text(image_path: str | Path, preprocess: bool = True) -> str:
    """Extract text using the default OCR provider."""
    return _get_default().extract_text(image_path, preprocess=preprocess)


def extract_text_with_confidence(image_path: str | Path) -> dict:
    """Extract text with confidence using the default OCR provider."""
    return _get_default().extract_text_with_confidence(image_path)
