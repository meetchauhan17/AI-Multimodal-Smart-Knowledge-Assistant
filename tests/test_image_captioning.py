import io
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from core.schemas import LLMResponse
from modules.image_captioning import caption_image


def create_dummy_image(color=(255, 0, 0), size=(100, 100)) -> bytes:
    """Helper to generate a solid-color dummy image in memory."""
    img = Image.new("RGB", size, color)
    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def red_image_bytes():
    return create_dummy_image(color=(255, 0, 0), size=(100, 100))


@pytest.fixture
def large_blue_image_file():
    """Generate a large image file (greater than 2048px) to verify the resizing branch."""
    data = create_dummy_image(color=(0, 0, 255), size=(3000, 1000))
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        temp_path.write_bytes(data)
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_caption_image_success_descriptive(red_image_bytes):
    """Test that a descriptive style request successfully routes to LLM and returns the caption."""
    mock_resp = LLMResponse(
        text="A simple solid red background square.",
        provider_used="groq",
        latency_ms=120.0,
        fallback_triggered=False
    )

    with mock.patch("core.llm_provider.LLMProvider.generate_vision", return_value=mock_resp) as mock_vision:
        result = caption_image(red_image_bytes, style="descriptive")
        
        assert result["error"] is None
        assert "red" in result["caption"].lower()
        assert result["provider_used"] == "groq"
        assert result["latency_ms"] > 0
        
        # Verify prompt style matches descriptive
        mock_vision.assert_called_once()
        called_args = mock_vision.call_args[0]
        # called_args[0] is image_bytes, called_args[1] is prompt
        assert "single clear, descriptive sentence" in called_args[1]


def test_caption_image_resizing_flow(large_blue_image_file):
    """Test that a large image (> 2048px) is resized before calling the vision LLM."""
    mock_resp = LLMResponse(
        text="A blue landscape.",
        provider_used="gemini",
        latency_ms=80.0,
        fallback_triggered=False
    )

    with mock.patch("core.llm_provider.LLMProvider.generate_vision", return_value=mock_resp) as mock_vision:
        result = caption_image(large_blue_image_file, style="short")
        
        assert result["error"] is None
        assert "blue" in result["caption"].lower()
        
        # Verify mock_vision was called with resized bytes
        mock_vision.assert_called_once()
        sent_bytes = mock_vision.call_args[0][0]
        
        # Load the sent bytes back using PIL to assert the dimensions
        resized_img = Image.open(io.BytesIO(sent_bytes))
        assert max(resized_img.size) == 2048
        assert resized_img.size[0] == 2048
        # aspect ratio 3000:1000 is 3:1, so 2048:682
        assert resized_img.size[1] == 682


def test_caption_image_validation_rejects_invalid_file():
    """Test that non-image inputs are rejected during pre-flight checks."""
    invalid_data = b"This is not an image file, just a plain text block."
    result = caption_image(invalid_data)
    
    assert result["caption"] == ""
    assert result["error"] is not None
    assert "Invalid or corrupted image" in result["error"]
