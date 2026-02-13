"""OCR module for extracting text from journal entry images."""

import logging
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Apply preprocessing to improve OCR accuracy on handwritten/printed journal entries."""
    # Convert to grayscale
    image = image.convert("L")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    # Apply slight denoise
    image = image.filter(ImageFilter.MedianFilter(size=3))

    return image


def extract_text(image_path: str | Path, preprocess: bool = True) -> str:
    """Extract text from a journal entry image using Tesseract OCR.

    Args:
        image_path: Path to the image file.
        preprocess: Whether to apply image preprocessing for better OCR results.

    Returns:
        Extracted text string.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is not a supported image format.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    supported_formats = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    if image_path.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported: {', '.join(sorted(supported_formats))}"
        )

    image = Image.open(image_path)

    if preprocess:
        image = preprocess_image(image)

    text = pytesseract.image_to_string(image)
    text = text.strip()

    logger.info("Extracted %d characters from %s", len(text), image_path.name)
    return text


def extract_text_with_confidence(image_path: str | Path) -> dict:
    """Extract text with per-word confidence scores.

    Returns:
        Dict with 'text' (full extracted text) and 'details' (list of word-level data).
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

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
