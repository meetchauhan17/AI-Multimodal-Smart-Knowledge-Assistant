"""
modules/image_captioning.py — Image captioning using the LLMProvider's vision capability.

Exposes caption_image() to analyze and describe image files or bytes in various styles.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Dict, Any, Union

from PIL import Image

from core.logger import logger
from core.llm_provider import LLMProvider

# Style prompt mapping for the vision LLM
STYLE_PROMPTS = {
    "descriptive": "Describe this image in a single clear, descriptive sentence.",
    "short": "Provide a very short and concise caption for this image in 5 to 8 words suitable for alt-text.",
    "detailed": (
        "Provide a detailed paragraph describing this image, including key objects, "
        "their layout/setting, colors, and the general mood."
    )
}


def _process_and_validate_image(image_input: str | Path | bytes) -> tuple[bytes, str]:
    """Load image from path or bytes, validate it, resize if large, and return normalized bytes + format.
    
    Raises ValueError or IOError if the file is invalid or corrupt.
    """
    try:
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        else:
            img = Image.open(Path(image_input))
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image format: {exc}")

    # Read properties
    img_format = img.format or "JPEG"
    width, height = img.size
    max_edge = max(width, height)

    # Resize if long edge is larger than 2048px
    if max_edge > 2048:
        scale = 2048.0 / max_edge
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Resize using LANCZOS filter
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")

    # Output back to bytes
    out_buf = io.BytesIO()
    save_format = img_format if img_format in ("JPEG", "PNG", "WEBP") else "JPEG"
    img.save(out_buf, format=save_format)
    return out_buf.getvalue(), save_format


def caption_image(image_path_or_bytes: str | Path | bytes, style: str = "descriptive") -> Dict[str, Any]:
    """Generate a natural-language description of the image.

    Args:
        image_path_or_bytes: Path to the image file or raw image bytes.
        style:               Description style: 'descriptive' (default), 'short', or 'detailed'.

    Returns:
        A dictionary containing:
            - caption: Generated description text.
            - provider_used: LLM provider that processed the query.
            - latency_ms: Call latency.
            - error: Optional error description string if any validation failed.
    """
    start_time = time.perf_counter()
    style_lower = style.strip().lower()
    
    if style_lower not in STYLE_PROMPTS:
        logger.warning(f"Unknown style {style!r} requested. Defaulting to 'descriptive'")
        style_lower = "descriptive"

    # Pre-flight checks and normalization
    try:
        image_bytes, img_format = _process_and_validate_image(image_path_or_bytes)
    except Exception as exc:
        err_msg = str(exc)
        logger.error(f"Image validation failed: {err_msg}")
        return {
            "caption": "",
            "provider_used": "",
            "latency_ms": 0.0,
            "error": err_msg,
        }

    # Resolve prompt
    prompt = STYLE_PROMPTS[style_lower]
    logger.info(f"Generating image caption using style={style_lower!r}")

    # Execute via LLMProvider
    try:
        provider = LLMProvider()
        response = provider.generate_vision(image_bytes, prompt)
        latency = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "caption": response.text,
            "provider_used": response.provider_used,
            "latency_ms": latency,
            "error": None,
        }
    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to generate caption from LLM: {exc}"
        logger.exception(err_msg)
        return {
            "caption": "",
            "provider_used": "",
            "latency_ms": latency,
            "error": err_msg,
        }


# ── Scaffold Class Wrapper for Backwards Compatibility ──────────────────────

class ImageCaptioner:
    """Generates a natural-language caption for an image (wrapped class)."""

    def __init__(self, provider: str = "gemini") -> None:
        self.provider = provider
        logger.debug(f"ImageCaptioner initialized (legacy wrapper)")

    def caption(self, image: Union[str, Path, Image.Image]) -> Any:
        """Caption helper method calling caption_image."""
        from core.schemas import ImageCaptionResult
        
        # If input is a PIL image, save it to bytes first
        if isinstance(image, Image.Image):
            out_buf = io.BytesIO()
            img_format = image.format or "PNG"
            image.save(out_buf, format=img_format)
            image_input: str | Path | bytes = out_buf.getvalue()
        else:
            image_input = image

        res = caption_image(image_input, style="descriptive")
        if res["error"]:
            raise RuntimeError(res["error"])

        return ImageCaptionResult(caption=res["caption"])


if __name__ == "__main__":
    # CLI main block verification
    print("ImageCaptioning module CLI runner.")
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Analyzing: {img_path}")
        for style in ["descriptive", "short", "detailed"]:
            print(f"\n--- Style: {style} ---")
            res = caption_image(img_path, style=style)
            print("Result:", res)
    else:
        print("No image provided. Usage: python image_captioning.py <image_file_path>")
