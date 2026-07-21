"""
modules/image_generation.py — Image generation using Pollinations.ai (free, keyless).

Exposes generate_image() to synthesize images from text descriptions.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Tuple

import requests

from config.settings import settings
from core.logger import logger

# Directory to save generated images.
# Absolute path so images are always stored in the project folder regardless
# of the caller's working directory (scripts, tests, Gradio app, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_IMAGES_DIR = _PROJECT_ROOT / "generated_images"
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Basic safety blacklist keywords to fail-fast client-side
SAFETY_BLACKLIST = [
    "nsfw", "naked", "nude", "porn", "explicit", "gore", "blood", "kill", "suicide", "bomb", "terrorist", "abuse"
]


def _check_prompt_safety(prompt: str) -> Tuple[bool, str | None]:
    """Check if a prompt is empty or contains restricted words."""
    cleaned = prompt.strip()
    if not cleaned:
        return False, "Prompt cannot be empty."

    prompt_lower = cleaned.lower()
    for keyword in SAFETY_BLACKLIST:
        if keyword in prompt_lower:
            return False, f"Prompt violates safety guidelines: contains restricted keyword '{keyword}'."

    return True, None


def _get_unique_filename(prompt: str) -> str:
    """Generate a filename from timestamp and short hash of prompt."""
    timestamp = int(time.time())
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    return f"gen_{timestamp}_{prompt_hash}.png"


def generate_image(prompt: str, style_hint: str | None = None) -> Dict[str, Any]:
    """Generate an image from a text prompt.

    Attempts Google Gemini Imagen first. On failure (such as region limits, quota,
    or lack of api key), falls back to Pollinations.ai free keyless GET API.

    Args:
        prompt:      The text description of the image to generate.
        style_hint:  Optional style string (e.g. 'cinematic', 'anime', 'oil painting')
                     appended to prompt.

    Returns:
        A dictionary containing:
            - image_path: Path to the saved image file or empty string if failed.
            - provider_used: 'gemini' or 'pollinations' or empty string.
            - latency_ms: Call latency.
            - error: Optional error description string if failed.
    """
    start_time = time.perf_counter()

    # 1. Safety Check
    is_safe, violation_reason = _check_prompt_safety(prompt)
    if not is_safe:
        logger.warning(f"Image generation prompt rejected: {violation_reason}")
        return {
            "image_path": "",
            "provider_used": "",
            "latency_ms": 0.0,
            "error": violation_reason,
        }

    # Append style hint if provided
    final_prompt = prompt.strip()
    if style_hint and style_hint.strip():
        final_prompt = f"{final_prompt}, {style_hint.strip()}"

    filename = _get_unique_filename(final_prompt)
    output_path = GENERATED_IMAGES_DIR / filename

    # Generate with Pollinations.ai (free, no API key needed)
    try:
        logger.info(f"Generating image with Pollinations.ai for prompt: '{final_prompt[:60]}...'")
        encoded_prompt = urllib.parse.quote(final_prompt)
        url = settings.image_gen_fallback_url.format(prompt=encoded_prompt)

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        latency = (time.perf_counter() - start_time) * 1000.0

        logger.info(f"Pollinations image generation successful. Saved to {output_path}")
        return {
            "image_path": str(output_path),
            "provider_used": "pollinations",
            "latency_ms": latency,
            "error": None,
        }

    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Image generation failed: {exc}"
        logger.error(err_msg)
        return {
            "image_path": "",
            "provider_used": "",
            "latency_ms": latency,
            "error": err_msg,
        }


# ── Scaffold Class Wrapper for Backwards Compatibility ──────────────────────

class ImageGenerator:
    """Generates images from text prompts (wrapped class)."""

    def __init__(self, url_pattern: str | None = None) -> None:
        self.url_pattern = url_pattern or settings.image_gen_fallback_url

    def generate(self, prompt: str, output_path: str | None = None) -> str:
        """Legacy generate method."""
        if output_path:
            res = generate_image(prompt)
            if res["error"]:
                raise RuntimeError(res["error"])
            # Copy to target path
            Path(output_path).write_bytes(Path(res["image_path"]).read_bytes())
            return output_path
        
        # If no output path is requested, build and return the fallback URL
        return self.build_url(prompt)

    def build_url(self, prompt: str) -> str:
        """Return the generation URL for the given prompt without fetching it."""
        encoded = urllib.parse.quote(prompt)
        return self.url_pattern.format(prompt=encoded)


if __name__ == "__main__":
    # CLI main block verification
    print("ImageGeneration module CLI runner.")
    test_prompt = "A futuristic city in the clouds, digital art"
    print(f"Generating image for prompt: '{test_prompt}'")
    res = generate_image(test_prompt)
    print("Result:", res)
    if res["image_path"] and os.path.exists(res["image_path"]):
        print(f"Image saved successfully, size: {os.path.getsize(res['image_path'])} bytes")
